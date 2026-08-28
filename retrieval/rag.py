import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


from llm_client import OllamaClient, OpenAICompatibleClient
from retriever import BM25, HybridRetriever
from router import RoutePath, TreeRouter
from taxonomy import TaxonomyIndex


class TreeGuidedRAG:
    """V4.1 Simple Tree-RAG.

    Online flow:
      1. one-shot Tree planner (one LLM call)
      2. local multi-channel BM25 retrieval
         - Tree-local
         - global atomic rescue
         - FAQ-level provenance retrieval
      3. Answer/Judge LLM
      4. optional ONE evidence-guided refinement retrieval + final Answer LLM
      5. if DB is still insufficient, clearly-labelled model knowledge fallback

    V4 intentionally has no progressive batch controller, repeated sufficiency
    checker, sibling scheduler, or EvidenceSelector loop.
    """

    def __init__(
        self,
        config,
        router_llm=None,
        answer_llm=None,
        embedding_llm=None,
    ):
        self.config = config
        self.taxonomy = TaxonomyIndex(config.csv_path)

        ollama_client = OllamaClient(config.ollama_url, timeout=config.timeout)
        ncl_client = OpenAICompatibleClient(
            config.ncl_api_url,
            timeout=config.timeout,
            verify_ssl=config.ncl_verify_ssl,
        )

        def generation_client(provider):
            provider = str(provider).strip().lower()
            if provider == "ncl":
                return ncl_client
            if provider == "ollama":
                return ollama_client
            raise ValueError(f"未知 LLM provider: {provider!r}；目前只支援 'ncl' 或 'ollama'")

        self.router_llm = router_llm or generation_client(config.router_provider)
        self.answer_llm = answer_llm or generation_client(config.answer_provider)
        self.embedding_llm = embedding_llm or ollama_client

        self.router = TreeRouter(
            self.taxonomy,
            self.router_llm,
            config.router_model,
            think=config.router_think,
            candidate_pool=config.tree_candidate_pool,
            top_paths=config.router_top_paths,
        )

        self.retriever = HybridRetriever(
            self.taxonomy,
            self.embedding_llm,
            config.embedding_model,
            use_embedding=config.use_embedding,
            batch_size=config.embedding_batch_size,
        )

        self._all_indices = list(range(len(self.taxonomy.df)))
        (
            self._faq_ids,
            self._faq_indices,
            self._faq_bm25,
            self._faq_source_units,
            self._faq_position,
        ) = self._build_faq_index()

    # ------------------------------------------------------------------
    # Local retrieval channels
    # ------------------------------------------------------------------

    def _build_faq_index(self):
        if "faq_id" not in self.taxonomy.df.columns:
            return [], {}, None, {}, {}

        groups = {}
        for idx, raw_faq_id in self.taxonomy.df["faq_id"].items():
            faq_id = str(raw_faq_id).strip()
            if not faq_id:
                continue
            groups.setdefault(faq_id, []).append(int(idx))

        faq_ids = list(groups.keys())
        texts = []
        source_units = {}

        for faq_id in faq_ids:
            chunks = []
            atomics = []

            for idx in groups[faq_id]:
                record = self.taxonomy.document_record(idx)
                question = str(record.get("question", "")).strip()
                answer = str(record.get("answer", "")).strip()

                chunks.append(f"問題：{question}\n答案：{answer}")
                atomics.append(
                    {
                        "idx": int(idx),
                        "atomic_id": str(record.get("atomic_id", "")).strip(),
                        "question_date": str(record.get("question_date", "")).strip(),
                        "question": question,
                        "answer": answer,
                        "taxonomy_paths": record.get("taxonomy_paths", ""),
                    }
                )

            texts.append("\n".join(chunks))
            source_units[faq_id] = {
                "source_id": f"FAQ:{faq_id}",
                "faq_id": faq_id,
                "atomics": atomics,
            }

        faq_position = {faq_id: i for i, faq_id in enumerate(faq_ids)}

        return (
            faq_ids,
            groups,
            BM25(texts) if texts else None,
            source_units,
            faq_position,
        )

    def _tree_k(self, rank):
        if rank == 0:
            return self.config.tree_primary_top_k
        if rank == 1:
            return self.config.tree_secondary_top_k
        return self.config.tree_tertiary_top_k

    def _tree_hits(self, query, paths, channel="tree", top_k_override=None):
        hits = []
        for route_rank, path in enumerate(paths):
            indices = self.taxonomy.docs_for_path(path)
            if not indices:
                continue
            top_k = (
                int(top_k_override)
                if top_k_override is not None
                else self._tree_k(route_rank)
            )
            local = self.retriever.search(
                query,
                indices,
                top_k=top_k,
            )
            for local_rank, hit in enumerate(local, start=1):
                hits.append(
                    {
                        **hit,
                        "channel": channel,
                        "channel_rank": local_rank,
                        "route_rank": route_rank + 1,
                        "route_score": float(path.score),
                        "path": path.display(),
                    }
                )
        return hits

    def _global_hits(self, query, top_k=None, channel="global"):
        top_k = int(top_k or self.config.global_top_k)
        hits = self.retriever.search(query, self._all_indices, top_k=top_k)
        return [
            {
                **hit,
                "channel": channel,
                "channel_rank": rank,
                "route_rank": None,
                "route_score": 0.0,
                "path": "GLOBAL",
            }
            for rank, hit in enumerate(hits, start=1)
        ]

    def _faq_hits(self, query):
        if self._faq_bm25 is None or not self._faq_ids:
            return []

        scores = self._faq_bm25.scores(query)
        order = sorted(
            range(len(self._faq_ids)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[: min(self.config.faq_top_k, len(self._faq_ids))]

        result = []
        channel_rank = 0
        for faq_pos in order:
            faq_id = self._faq_ids[faq_pos]
            indices = self._faq_indices[faq_id]

            # Within the selected FAQ, order atomic units by local relevance, but
            # keep multiple siblings so a decomposed complete answer can recover.
            ordered = self.retriever.search(
                query,
                indices,
                top_k=min(self.config.faq_max_units_per_faq, len(indices)),
            )
            for hit in ordered:
                channel_rank += 1
                result.append(
                    {
                        **hit,
                        "channel": "faq",
                        "channel_rank": channel_rank,
                        "route_rank": None,
                        "route_score": 0.0,
                        "path": f"FAQ:{faq_id}",
                        "faq_id": faq_id,
                    }
                )
        return result

    def _provenance_hits(self, query, *anchor_groups):
        """Complete original FAQ provenance for high-ranked atomic hits.

        Atomic decomposition can fragment one coherent source answer. Anchor
        slots are distributed across retrieval channels so Tree-local evidence
        cannot crowd out a strong global rescue hit (and vice versa). Selected
        FAQ siblings are then restored locally; no extra LLM call is involved.
        """
        groups = [list(g or []) for g in anchor_groups if g]
        if not groups or "faq_id" not in self.taxonomy.df.columns:
            return []

        max_faqs = max(1, int(self.config.provenance_max_faqs))
        per_group = max(1, max_faqs // len(groups))
        scan_limit = max(1, int(self.config.provenance_anchor_top_k))
        faq_ids = []
        faq_anchor_rank = {}
        seen_faqs = set()

        for group in groups:
            added = 0
            for hit in group[:scan_limit]:
                idx = int(hit["idx"])
                faq_id = str(self.taxonomy.df.at[idx, "faq_id"]).strip()
                if not faq_id or faq_id in seen_faqs:
                    continue
                seen_faqs.add(faq_id)
                faq_ids.append(faq_id)
                added += 1
                faq_anchor_rank[faq_id] = added
                if added >= per_group or len(faq_ids) >= max_faqs:
                    break
            if len(faq_ids) >= max_faqs:
                break

        result = []
        stride = max(1, int(self.config.provenance_max_units_per_faq))
        for faq_pos, faq_id in enumerate(faq_ids, start=1):
            indices = self._faq_indices.get(faq_id, [])
            if not indices:
                continue
            ordered = self.retriever.search(
                query,
                indices,
                top_k=min(stride, len(indices)),
            )

            # Provenance is a completion/support channel, not an independent
            # relevance vote. Preserve sibling order and give every sibling its
            # own rank so one FAQ cannot flood the final context with many
            # artificial rank-1 candidates.
            for sibling_rank, hit in enumerate(ordered, start=1):
                result.append(
                    {
                        **hit,
                        "channel": "provenance",
                        "channel_rank": (faq_pos - 1) * stride + sibling_rank,
                        "route_rank": None,
                        "route_score": 0.0,
                        "path": f"PROVENANCE_FAQ:{faq_id}",
                        "faq_id": faq_id,
                    }
                )
        return result

    def _channel_contribution(self, hit):
        rank = max(1, int(hit.get("channel_rank", 1)))
        channel = hit.get("channel")
        if channel in {"tree", "refine_tree", "refine_sibling"}:
            # Tree is a relevance prior, not a hard gate. A path discovered by
            # evidence-guided refinement may be appended after the initial routes;
            # do not suppress its best local hit merely because of list position.
            route_score = max(0.0, min(1.0, float(hit.get("route_score", 0.0))))
            route_prior = 0.75 + 0.25 * route_score
            weight = float(self.config.tree_channel_weight) * route_prior
        elif channel in {"global", "global_rewrite", "refine_global"}:
            weight = float(self.config.global_channel_weight)
        elif channel == "first_round":
            weight = float(self.config.carryover_channel_weight)
        elif channel == "provenance":
            weight = float(self.config.provenance_channel_weight)
        else:
            weight = float(self.config.faq_channel_weight)
        return weight / (60.0 + rank)

    def _merge_hits(self, *groups, limit=None):
        merged = {}
        for group in groups:
            for hit in group or []:
                idx = int(hit["idx"])
                entry = merged.setdefault(
                    idx,
                    {
                        "idx": idx,
                        "rrf_score": 0.0,
                        "max_bm25": float("-inf"),
                        "channels": [],
                        "paths": [],
                    },
                )
                entry["rrf_score"] += self._channel_contribution(hit)
                entry["max_bm25"] = max(
                    entry["max_bm25"], float(hit.get("bm25", 0.0))
                )
                channel = str(hit.get("channel", ""))
                if channel and channel not in entry["channels"]:
                    entry["channels"].append(channel)
                path = str(hit.get("path", ""))
                if path and path not in entry["paths"]:
                    entry["paths"].append(path)

        ranked = sorted(
            merged.values(),
            key=lambda x: (x["rrf_score"], x["max_bm25"]),
            reverse=True,
        )
        if limit and limit > 0:
            ranked = ranked[: int(limit)]

        evidence = []
        for rank, item in enumerate(ranked, start=1):
            record = self.taxonomy.document_record(item["idx"])
            evidence.append(
                {
                    **record,
                    "rank": rank,
                    "retrieval_score": float(item["rrf_score"]),
                    "bm25": float(item["max_bm25"]),
                    "channels": item["channels"],
                    "retrieved_paths": item["paths"],
                }
            )
        return evidence

    def _initial_retrieve(self, query, plan):
        paths = plan["paths"]
        rewrite = str(plan.get("search_query", "")).strip() or query
        combined_query = query if rewrite == query else f"{query}\n{rewrite}"

        # Global and FAQ retrieval do not depend on the remote planner. In run(),
        # their original-query versions are launched concurrently with planning.
        tree_hits = self._tree_hits(combined_query, paths)
        rewrite_global = []
        if rewrite and rewrite.strip() != query.strip():
            rewrite_global = self._global_hits(
                rewrite,
                top_k=self.config.global_rewrite_top_k,
                channel="global_rewrite",
            )
        return tree_hits, rewrite_global

    @staticmethod
    def _structural_constraints(text):
        text = str(text or "")
        lower = text.lower()
        found = set()

        # Leader/LDR positions, including forms such as LDR/06-07.
        for m in re.finditer(
            r"(?i)\b(?:leader|ldr)\s*/?\s*(\d{2})(?:\s*-\s*(\d{2}))?",
            text,
        ):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            if 0 <= start <= end <= 99 and end - start <= 10:
                for pos in range(start, end + 1):
                    found.add(f"ldr/{pos:02d}")

        # Explicit MARC tag references.
        tag_patterns = [
            r"(?i)\btag\s*0*(\d{3})\b",
            r"(?i)\bmarc(?:\s*21)?\s*(?:欄位|field)?\s*0*(\d{3})\b",
            r"欄位\s*0*(\d{3})\b",
            r"(\d{3})\s*段\b",
        ]
        for pattern in tag_patterns:
            for m in re.finditer(pattern, text):
                found.add(f"tag:{int(m.group(1)):03d}")

        # Fixed-field positions such as 008/23 or 008/18-34.
        for m in re.finditer(
            r"\b(\d{3})\s*/\s*(\d{2})(?:\s*-\s*(\d{2}))?\b",
            text,
        ):
            tag = int(m.group(1))
            start = int(m.group(2))
            end = int(m.group(3)) if m.group(3) else start
            found.add(f"tag:{tag:03d}")
            if 0 <= start <= end <= 99 and end - start <= 20:
                for pos in range(start, end + 1):
                    found.add(f"{tag:03d}/{pos:02d}")

        # Subfields.
        for m in re.finditer(r"\$([a-z0-9])", lower):
            found.add(f"subfield:${m.group(1)}")

        # Indicators, including "#4", "指標1", "indicator 2".
        for m in re.finditer(r"#([0-9#])", text):
            found.add(f"indicator-value:{m.group(1)}")
        for m in re.finditer(r"(?i)(?:指標|indicator)\s*([12])", text):
            found.add(f"indicator:{m.group(1)}")

        return found

    @classmethod
    def _constraint_overlap(cls, target_text, evidence_text):
        target = cls._structural_constraints(target_text)
        if not target:
            return 0
        evidence = cls._structural_constraints(evidence_text)
        return len(target & evidence)

    @staticmethod
    def _compact_feedback_text(item, answer_chars=220):
        question = str(item.get("question", "")).strip()
        answer = re.sub(r"\s+", " ", str(item.get("answer", "")).strip())
        if answer_chars and answer_chars > 0:
            answer = answer[: int(answer_chars)]
        return "\n".join(x for x in (question, answer) if x)

    def _build_refinement_search_query(
        self,
        original_query,
        refinement_query,
        missing,
        first_round_evidence,
    ):
        """Build one evidence-guided second-pass query.

        This is local pseudo-relevance feedback, not another LLM call.
        The missing aspect remains the main target, while terminology from the
        most relevant first-round evidence helps bridge vocabulary mismatch
        (e.g. user wording vs. older cataloguing terminology).
        """
        base_parts = [
            str(original_query or "").strip(),
            str(refinement_query or "").strip(),
            str(missing or "").strip(),
        ]
        base_query = "\n".join(x for x in base_parts if x)

        evidence = list(first_round_evidence or [])
        if not evidence:
            return base_query, []

        feedback_texts = [
            self._compact_feedback_text(
                item,
                answer_chars=self.config.refinement_feedback_answer_chars,
            )
            for item in evidence
        ]
        feedback_texts = [x for x in feedback_texts if x]
        if not feedback_texts:
            return base_query, []

        target_text = (
            f"{original_query}\n{refinement_query}\n{missing}".strip()
        )
        scores = BM25(feedback_texts).scores(
            f"{refinement_query}\n{missing}".strip() or original_query
        )
        overlaps = [
            self._constraint_overlap(target_text, text)
            for text in feedback_texts
        ]
        has_constraints = bool(self._structural_constraints(target_text))

        candidate_indices = list(range(len(feedback_texts)))
        if has_constraints and any(overlaps):
            # When the query specifies an exact field/position/subfield,
            # use only evidence aligned to at least one of those constraints.
            candidate_indices = [i for i in candidate_indices if overlaps[i] > 0]

        order = sorted(
            candidate_indices,
            key=lambda i: (overlaps[i], float(scores[i])),
            reverse=True,
        )

        selected = []
        selected_ids = []
        limit = max(0, int(self.config.refinement_feedback_top_k))
        for i in order[:limit]:
            selected.append(feedback_texts[i])
            evidence_id = str(evidence[i].get("atomic_id", "")).strip()
            if evidence_id:
                selected_ids.append(evidence_id)

        if not selected:
            return base_query, []

        expanded = (
            f"{base_query}\n"
            "第一輪相關 evidence 用語：\n"
            + "\n".join(selected)
        ).strip()
        return expanded, selected_ids

    def _same_parent_sibling_scope(self, base_paths):
        """Build local sibling-L3 search scopes under routed L2 parents.

        Each parent scope is searched as one union of sibling L3 documents.
        This avoids the ranking distortion caused by resetting local rank inside
        every sibling node independently.
        """
        scopes = []
        trace_paths = []
        seen_parents = set()
        parent_limit = max(1, int(self.config.refinement_parent_limit))

        # Collect all already-selected L3 values under each routed parent.
        selected_by_parent = {}
        parent_score = {}
        for base in base_paths:
            if not base.l1 or not base.l2:
                continue
            parent = (base.l1, base.l2)
            parent_score[parent] = max(
                float(base.score),
                float(parent_score.get(parent, 0.0)),
            )
            if base.l3:
                selected_by_parent.setdefault(parent, set()).add(base.l3)

        for base in base_paths:
            if not base.l1 or not base.l2:
                continue
            parent = (base.l1, base.l2)
            if parent in seen_parents:
                continue
            seen_parents.add(parent)
            if len(scopes) >= parent_limit:
                break

            l3s = self.taxonomy.l3_nodes(base.l1, base.l2)
            if not l3s:
                continue

            selected_l3s = selected_by_parent.get(parent, set())
            sibling_l3s = [l3 for l3 in l3s if l3 not in selected_l3s]
            if not sibling_l3s:
                continue

            indices = []
            seen_indices = set()
            inherited_score = (
                max(0.0, min(1.0, float(parent_score.get(parent, base.score))))
                * 0.90
            )

            for l3 in sibling_l3s:
                for idx in self.taxonomy.docs_for_l3(base.l1, base.l2, l3):
                    idx = int(idx)
                    if idx in seen_indices:
                        continue
                    seen_indices.add(idx)
                    indices.append(idx)

                path_text = " > ".join((base.l1, base.l2, l3))
                trace_paths.append(
                    RoutePath(
                        l1=base.l1,
                        l2=base.l2,
                        l3=l3,
                        score=inherited_score,
                        trace=[
                            {
                                "level": "PATH_SIBLING",
                                "node": path_text,
                                "score": inherited_score,
                                "reason": (
                                    "Evidence-guided refinement: sibling L3 "
                                    "under the already-routed L2 parent"
                                ),
                            }
                        ],
                    )
                )

            if indices:
                scopes.append(
                    {
                        "l1": base.l1,
                        "l2": base.l2,
                        "sibling_l3s": sibling_l3s,
                        "indices": indices,
                        "score": inherited_score,
                    }
                )

        return scopes, trace_paths

    def _sibling_scope_hits(self, query, scopes):
        hits = []
        per_parent_top_k = max(1, int(self.config.refinement_sibling_top_k))

        for parent_rank, scope in enumerate(scopes, start=1):
            local = self.retriever.search(
                query,
                scope["indices"],
                top_k=min(per_parent_top_k, len(scope["indices"])),
            )
            offset = (parent_rank - 1) * per_parent_top_k
            parent_path = f"{scope['l1']} > {scope['l2']} > [sibling L3 union]"

            for local_rank, hit in enumerate(local, start=1):
                hits.append(
                    {
                        **hit,
                        "channel": "refine_sibling",
                        "channel_rank": offset + local_rank,
                        "route_rank": parent_rank,
                        "route_score": float(scope["score"]),
                        "path": parent_path,
                    }
                )

        return hits

    def _refinement_retrieve(
        self,
        original_query,
        refinement_query,
        missing,
        base_paths,
        first_round_evidence,
    ):
        search_query, feedback_ids = self._build_refinement_search_query(
            original_query,
            refinement_query,
            missing,
            first_round_evidence,
        )

        sibling_scopes, sibling_paths = self._same_parent_sibling_scope(
            base_paths
        )

        # Keep original routed paths for continuity and expose sibling L3 paths
        # in trace, but retrieve sibling documents as one union per L2 parent.
        path_map = {p.key(): p for p in base_paths}
        for path in sibling_paths:
            path_map.setdefault(path.key(), path)
        paths = list(path_map.values())

        # First-round evidence already preserves the originally routed nodes.
        # The refinement round therefore spends its local Tree budget only on
        # the missing local neighborhood: sibling L3 documents under the same
        # routed L2 parent. Global/FAQ rescue remains available in parallel.
        tree_hits = self._sibling_scope_hits(
            search_query,
            sibling_scopes,
        )
        global_hits = self._global_hits(
            search_query,
            top_k=self.config.global_top_k,
            channel="refine_global",
        )
        faq_hits = self._faq_hits(search_query)

        meta = {
            "requested_query": str(refinement_query or "").strip(),
            "expanded_query": search_query,
            "feedback_evidence_ids": feedback_ids,
            "sibling_paths": [p.display() for p in sibling_paths],
            "sibling_parent_scopes": [
                {
                    "parent": f"{scope['l1']} > {scope['l2']}",
                    "sibling_l3s": list(scope["sibling_l3s"]),
                    "document_count": len(scope["indices"]),
                }
                for scope in sibling_scopes
            ],
        }
        return paths, tree_hits, global_hits, faq_hits, meta

    # ------------------------------------------------------------------
    # Answer / judge
    # ------------------------------------------------------------------

    @staticmethod
    def _answer_scope(query):
        """Local completeness intent only; never participates in Tree routing."""
        q = str(query or "").strip().lower()

        set_markers = (
            "有哪些",
            "還有哪些",
            "還有其他",
            "還有什麼",
            "其他工具",
            "其他方法",
            "其他選項",
            "除了",
            "列出",
            "所有工具",
            "所有方法",
            "哪些工具",
            "哪些方法",
            "哪些選項",
            "替代方案",
            "alternatives",
            "other tools",
            "other methods",
            "what other",
            "which tools",
            "which methods",
            "besides ",
            "list ",
        )

        return "set" if any(x in q for x in set_markers) else "focused"

    def _source_evidence_for_set_query(self, query, evidence, answer_scope):
        """Add source completeness only after normal atomic retrieval succeeds.

        Atomic retrieval remains primary. A FAQ/source family is eligible only
        if multiple atomic units from that same FAQ already survived Top-K.
        """
        if answer_scope != "set":
            return []
        if self._faq_bm25 is None or not evidence:
            return []

        stats = {}
        for item in evidence:
            faq_id = str(item.get("faq_id", "")).strip()
            if not faq_id or faq_id not in self._faq_source_units:
                continue

            s = stats.setdefault(
                faq_id,
                {"count": 0, "best_rank": 10**9, "channels": set()},
            )
            s["count"] += 1
            s["best_rank"] = min(
                s["best_rank"],
                int(item.get("rank", 10**9)),
            )
            s["channels"].update(item.get("channels", []))

        min_hits = max(1, int(self.config.source_min_atomic_hits))
        candidates = [
            faq_id
            for faq_id, s in stats.items()
            if s["count"] >= min_hits
        ]
        if not candidates:
            return []

        faq_scores = self._faq_bm25.scores(query)
        ranked = []

        for faq_id in candidates:
            pos = self._faq_position.get(faq_id)
            faq_score = float(faq_scores[pos]) if pos is not None else 0.0
            s = stats[faq_id]

            # Prefer a coherent family already represented by more atomic hits.
            key = (
                int(s["count"]),
                -int(s["best_rank"]),
                faq_score,
            )
            ranked.append((key, faq_id, faq_score, s))

        ranked.sort(key=lambda x: x[0], reverse=True)

        top_k = max(0, int(self.config.source_context_top_k))
        max_units = max(1, int(self.config.source_max_units_per_faq))

        result = []
        for _, faq_id, faq_score, s in ranked[:top_k]:
            source = self._faq_source_units[faq_id]
            result.append(
                {
                    "source_id": source["source_id"],
                    "faq_id": faq_id,
                    "faq_score": faq_score,
                    "atomic_hit_count": int(s["count"]),
                    "best_atomic_rank": int(s["best_rank"]),
                    "channels": sorted(s["channels"]),
                    "atomics": list(source["atomics"])[:max_units],
                }
            )

        return result

    @staticmethod
    def _source_evidence_context(source_evidence):
        blocks = []

        for source in source_evidence or []:
            lines = [
                f"[SOURCE {source.get('source_id', '')}]",
                (
                    "用途：這是原始 FAQ/source 的完整性視圖；"
                    "Atomic evidence 仍是主要精準證據。"
                ),
            ]

            for atom in source.get("atomics", []):
                atomic_id = str(atom.get("atomic_id", "")).strip()
                lines.append(
                    f"- [{atomic_id}] 問題：{atom.get('question', '')}\n"
                    f"  答案：{atom.get('answer', '')}"
                )

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    @staticmethod
    def _evidence_context(evidence):
        blocks = []
        for item in evidence:
            source_id = str(item.get("atomic_id", "")).strip()
            channels = ",".join(item.get("channels", []))
            blocks.append(
                f"[{source_id}] retrieval_channels={channels}\n"
                f"日期：{item.get('question_date', '')}\n"
                f"問題：{item.get('question', '')}\n"
                f"答案：{item.get('answer', '')}\n"
                f"taxonomy：{item.get('taxonomy_paths', '')}"
            )
        return "\n\n".join(blocks)

    def _answer_schema(self, allowed_ids, allow_partial):
        # V4.0.2:
        # First pass is retrieval diagnosis only. It may either answer from DB
        # evidence (grounded) or request exactly one refinement (partial).
        # Model-knowledge fallback is intentionally unavailable until AFTER the
        # refinement round has been attempted.
        statuses = (
            ["grounded", "partial"]
            if allow_partial
            else ["grounded", "knowledge_fallback"]
        )
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": statuses},
                "answer": {"type": "string"},
                "missing": {"type": "string"},
                "refinement_query": {"type": "string"},
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": allowed_ids},
                },
                "knowledge_used": {"type": "boolean"},
            },
            "required": [
                "status",
                "answer",
                "missing",
                "refinement_query",
                "evidence_ids",
                "knowledge_used",
            ],
        }

    def _answer_once(self, query, evidence, source_evidence=None, allow_partial=True):
        ids = [
            str(x.get("atomic_id", "")).strip()
            for x in evidence
            if str(x.get("atomic_id", "")).strip()
        ]

        # Source-level context may contain relevant siblings that did not fit
        # into the atomic Top-K. Allow those IDs in the grounded citation set.
        for source in source_evidence or []:
            for atom in source.get("atomics", []):
                atomic_id = str(atom.get("atomic_id", "")).strip()
                if atomic_id and atomic_id not in ids:
                    ids.append(atomic_id)

        source_context = self._source_evidence_context(source_evidence)

        # If source-level completeness is active, the selected source family is
        # the primary synthesis view.  Remove duplicate atomics from the normal
        # Top-K context and place the complete source first.  Atomic retrieval
        # still selected/validated this source; this only changes what the
        # Answer/Judge sees first.
        source_atomic_ids = {
            str(atom.get("atomic_id", "")).strip()
            for source in (source_evidence or [])
            for atom in source.get("atomics", [])
            if str(atom.get("atomic_id", "")).strip()
        }
        supporting_evidence = [
            item
            for item in evidence
            if str(item.get("atomic_id", "")).strip() not in source_atomic_ids
        ]
        atomic_context = self._evidence_context(
            supporting_evidence if source_context else evidence
        )

        context_parts = []
        if source_context:
            context_parts.append(
                "=== PRIMARY Source-level FAQ evidence（完整集合/原始語境）===\n"
                "這是集合型問題的主要完整性來源。回答前必須逐項檢視此 SOURCE "
                "中的每一個 atomic unit；不要只回答最相似的前幾項。\n"
                + source_context
            )
        if atomic_context:
            label = (
                "=== Secondary atomic evidence（其他補充證據）===\n"
                if source_context
                else "=== Atomic evidence（精準檢索）===\n"
            )
            context_parts.append(label + atomic_context)

        context = "\n\n".join(context_parts)
        schema = self._answer_schema(ids, allow_partial=allow_partial)

        if allow_partial:
            status_instructions = """請選擇 status：
- grounded：核心答案可由目前 evidence 直接支持或安全組合支持。knowledge_used=false。
- partial：目前 evidence 尚不足以直接支持完整答案。即使你知道答案，也不得使用模型知識；請填 missing 與一個精準 refinement_query，讓系統先再查資料庫一次。

本輪禁止 knowledge_fallback。資料庫不足時必須回 partial，而不是直接用模型知識回答。"""
            final_instruction = (
                "這是第一輪檢索判斷。只做 evidence-grounded 判斷或提出 refinement；"
                "不得以模型既有知識補答。"
            )
        else:
            status_instructions = """請選擇 status：
- grounded：核心答案可由目前 evidence 直接支持或安全組合支持。knowledge_used=false。
- knowledge_fallback：已完成一次 refinement 後，資料庫仍不足以完整回答；此時才可使用模型知識回答，knowledge_used=true，而且必須清楚標示哪些內容未由本次資料庫直接驗證。"""
            final_instruction = (
                "這是 refinement 後的最終判斷。若資料庫仍不完整，不可再回 partial；"
                "才可使用 knowledge_fallback，並把資料庫可確認內容與模型知識補充分開。"
            )

        prompt = f"""你是國家圖書館 Tree-RAG 的 Answer/Judge。

使用者問題：
{query}

本次資料庫檢索 evidence：
{context if context else '(沒有檢索到 evidence)'}

{status_instructions}

規則：
1. evidence 中的正式規則、數字、碼數、位址、indicator、subfield、年份、順序與條件不得被改寫成不同內容。
2. 不得為了調和不同 evidence 自行創造 evidence 沒寫出的「比例、顯著程度、主從門檻、例外或適用條件」。
3. 使用者若指定 MARC/RDA 欄位、位址、indicator 或 subfield，不能用其他欄位的相近規則冒充直接證據。
4. 枚舉/有哪些/還有其他/除了某項之外類問題，要整合目前 evidence 中所有明確支持的選項；不要只列第一個。
   若有 PRIMARY Source-level FAQ evidence：
   - 先逐一檢視該 SOURCE 中「每一個」atomic unit，再作答。
   - 使用者明確排除的項目不要列入答案。
   - 其餘凡直接符合問題條件的項目都要納入；不得因為前兩三項語意較相似就停止。
   - grounded 的 evidence_ids 應包含實際支持答案各項目的 source atomic_id。
5. evidence_ids 只能填上方實際存在的 atomic_id。
6. grounded 時不要加入模型既有知識。
7. knowledge_fallback 時，答案中必須出現「【模型知識補充｜未由本次資料庫直接驗證】」。若 evidence 有可確認部分，可先列「資料庫可確認」再補充。
8. partial 的 refinement_query 應使用原問題的精確 constraint + missing aspect；不得把你猜測的答案值塞進搜尋詞。
9. 回答使用繁體中文，直接回答館員問題，不輸出內部 reasoning。

{final_instruction}
"""

        result = self.answer_llm.chat_json(
            self.config.answer_model,
            [{"role": "user", "content": prompt}],
            schema,
            temperature=0.0,
            think=self.config.answer_think,
        )

        status = str(result.get("status", "")).strip()
        knowledge_used = bool(result.get("knowledge_used", False))

        # Deterministic lifecycle guard:
        #   pass 1  -> grounded | partial
        #   pass 2  -> grounded | knowledge_fallback
        # Do not rely only on the LLM following the enum because some
        # OpenAI-compatible backends return JSON without schema enforcement.
        if allow_partial:
            if status != "grounded" or knowledge_used:
                status = "partial"
                knowledge_used = False
                # A model-generated fallback answer from pass 1 is deliberately
                # discarded; it must not become user-facing before DB refinement.
                result["answer"] = ""
        else:
            if status != "grounded":
                status = "knowledge_fallback"
                knowledge_used = True
            elif knowledge_used:
                status = "knowledge_fallback"
                knowledge_used = True

        allowed = set(ids)
        result["evidence_ids"] = [
            str(x).strip()
            for x in result.get("evidence_ids", [])
            if str(x).strip() in allowed
        ]
        result["status"] = status
        result["knowledge_used"] = status == "knowledge_fallback" or knowledge_used

        if allow_partial and status == "partial":
            missing = str(result.get("missing", "")).strip()
            refinement_query = str(result.get("refinement_query", "")).strip()
            if not missing:
                missing = "目前資料庫 evidence 尚不足以直接支持完整答案"
            if not refinement_query:
                refinement_query = f"{query} {missing}".strip()
            result["missing"] = missing
            result["refinement_query"] = refinement_query

        answer = str(result.get("answer", "")).strip()
        if result["status"] == "knowledge_fallback" and self.config.allow_model_knowledge_fallback:
            marker = "【模型知識補充｜未由本次資料庫直接驗證】"
            if marker not in answer:
                answer = f"{marker}\n{answer}".strip()
        result["answer"] = answer
        return result

    # ------------------------------------------------------------------
    # Gap logging for later DB curation
    # ------------------------------------------------------------------

    def _log_gap(self, query, initial, final, paths, evidence, refinement_used):
        queue_path = Path(self.config.fallback_queue_path)
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "initial_status": initial.get("status"),
            "final_status": final.get("status"),
            "initial_missing": initial.get("missing", ""),
            "refinement_query": initial.get("refinement_query", ""),
            "refinement_used": bool(refinement_used),
            "knowledge_used": bool(final.get("knowledge_used", False)),
            "routed_paths": [p.display() for p in paths],
            "evidence_ids": [x.get("atomic_id", "") for x in evidence],
            "answer": final.get("answer", ""),
            "review_status": "pending",
        }
        with queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Public run
    # ------------------------------------------------------------------

    def run(self, query, **_compat_kwargs):
        query = str(query or "").strip()
        if not query:
            raise ValueError("query 不可為空")

        t0 = time.perf_counter()

        # Planner is remote; global and FAQ retrieval are local and independent.
        with ThreadPoolExecutor(max_workers=3) as pool:
            plan_future = pool.submit(self.router.plan, query)
            global_future = pool.submit(self._global_hits, query)
            faq_future = pool.submit(self._faq_hits, query)
            plan = plan_future.result()
            global_hits = global_future.result()
            faq_hits = faq_future.result()

        # Completeness intent is local and orthogonal to Tree routing.
        answer_scope = self._answer_scope(query)

        t_plan_parallel = time.perf_counter()

        tree_hits, rewrite_global = self._initial_retrieve(query, plan)
        # Provenance completion follows the lexical rescue channels. Global
        # hits are direct query matches across the whole DB; using them as FAQ
        # anchors avoids a broad Tree node flooding provenance with unrelated
        # siblings while still repairing atomic fragmentation.
        provenance_hits = self._provenance_hits(
            query,
            global_hits,
            rewrite_global,
            tree_hits,
        )
        evidence = self._merge_hits(
            tree_hits,
            global_hits,
            rewrite_global,
            faq_hits,
            provenance_hits,
            limit=self.config.context_limit,
        )
        source_evidence = self._source_evidence_for_set_query(
            query,
            evidence,
            answer_scope,
        )
        t_retrieval = time.perf_counter()

        initial = self._answer_once(
            query,
            evidence,
            source_evidence=source_evidence,
            allow_partial=self.config.enable_refinement,
        )
        t_first_answer = time.perf_counter()

        final = initial
        refinement_used = False
        refinement_trace = None
        final_paths = list(plan["paths"])

        if (
            self.config.enable_refinement
            and initial.get("status") == "partial"
        ):
            refinement_used = True
            refinement_query = str(initial.get("refinement_query", "")).strip()
            if not refinement_query:
                missing = str(initial.get("missing", "")).strip()
                refinement_query = f"{query} {missing}".strip()

            (
                final_paths,
                refine_tree,
                refine_global,
                refine_faq,
                refinement_trace,
            ) = self._refinement_retrieve(
                query,
                refinement_query,
                str(initial.get("missing", "")).strip(),
                plan["paths"],
                evidence,
            )

            refine_provenance = self._provenance_hits(
                refinement_query,
                refine_global,
                refine_faq,
                refine_tree,
            )
            refined_evidence = self._merge_hits(
                # Keep first-round evidence in the candidate pool by recreating a
                # lightweight provenance channel from their indices.
                [
                    {
                        "idx": x["idx"],
                        "bm25": x.get("bm25", 0.0),
                        "channel": "first_round",
                        "channel_rank": i,
                        "route_rank": None,
                        "route_score": 0.0,
                        "path": "FIRST_ROUND",
                    }
                    for i, x in enumerate(evidence, start=1)
                ],
                refine_tree,
                refine_global,
                refine_faq,
                refine_provenance,
                limit=self.config.refinement_context_limit,
            )
            evidence = refined_evidence
            source_evidence = self._source_evidence_for_set_query(
                query,
                evidence,
                answer_scope,
            )
            final = self._answer_once(
                query,
                evidence,
                source_evidence=source_evidence,
                allow_partial=False,
            )

        t_final = time.perf_counter()

        # Initial non-grounded status is itself a useful DB-gap signal, even if
        # refinement later succeeds from a different slice of the same database.
        if initial.get("status") != "grounded" or final.get("knowledge_used"):
            self._log_gap(
                query,
                initial,
                final,
                final_paths,
                evidence,
                refinement_used,
            )

        timing = {
            "planner_plus_parallel_local_retrieval": round(t_plan_parallel - t0, 4),
            "tree_merge_retrieval": round(t_retrieval - t_plan_parallel, 4),
            "first_answer_judge": round(t_first_answer - t_retrieval, 4),
            "refinement_and_final_answer": round(t_final - t_first_answer, 4),
            "total": round(t_final - t0, 4),
        }

        return {
            "answer": final.get("answer", ""),
            "mode": final.get("status", "knowledge_fallback"),
            "knowledge_used": bool(final.get("knowledge_used", False)),
            "initial_decision": initial,
            "final_decision": final,
            "answer_scope": answer_scope,
            "refinement_used": refinement_used,
            "refinement": refinement_trace,
            "planner": {
                "query_focus": plan.get("query_focus", ""),
                "search_query": plan.get("search_query", query),
                "fallback": bool(plan.get("fallback", False)),
            },
            "paths": [
                {
                    "path": p.display(),
                    "score": float(p.score),
                    "trace": p.trace,
                }
                for p in final_paths
            ],
            "source_retrieval": {
                "source_count": len(source_evidence),
                "sources": [
                    {
                        "source_id": x.get("source_id", ""),
                        "faq_id": x.get("faq_id", ""),
                        "faq_score": x.get("faq_score", 0.0),
                        "atomic_hit_count": x.get("atomic_hit_count", 0),
                        "best_atomic_rank": x.get("best_atomic_rank", 0),
                        "channels": x.get("channels", []),
                        "atomic_ids": [
                            a.get("atomic_id", "")
                            for a in x.get("atomics", [])
                        ],
                        "synthesis_priority": "primary",
                    }
                    for x in source_evidence
                ],
            },
            "retrieval": {
                "evidence_count": len(evidence),
                "evidence": [
                    {
                        "rank": x["rank"],
                        "atomic_id": x.get("atomic_id", ""),
                        "faq_id": x.get("faq_id", ""),
                        "question_date": x.get("question_date", ""),
                        "question": x.get("question", ""),
                        "answer": x.get("answer", ""),
                        "channels": x.get("channels", []),
                        "retrieved_paths": x.get("retrieved_paths", []),
                        "retrieval_score": x.get("retrieval_score", 0.0),
                    }
                    for x in evidence
                ],
            },
            "timing": timing,
        }
