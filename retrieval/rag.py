import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


from answer_embedding_retriever import AnswerEmbeddingRetriever
from llm_client import OllamaClient, OpenAICompatibleClient
from retriever import BM25, HybridRetriever
from router import RoutePath, TreeRouter
from taxonomy import TaxonomyIndex


OUT_OF_SCOPE_MESSAGE = "您的問題跟編目問題無關。"


class TreeGuidedRAG:
    """V4.3 Tree-Guided Hierarchical RAG.

    Online flow:
      1. one-shot Planner/Scope Gate (one LLM call)
         - scope decision
         - 1~3 semantic Atomic Retrieval Units
         - focused BM25 query + lexical keywords
         - Tree path prior
      2. unit-aware multi-view local BM25 retrieval
         - focused Tree-local
         - focused Global
         - keyword Global
         - original-query Tree/Global/FAQ rescue
         - FAQ/provenance completion
      3. per-unit RRF + cross-unit evidence coverage merge
      4. Answer/Judge LLM
      5. optional ONE unit-aware evidence-guided refinement + final Answer LLM
      6. if DB is still insufficient, clearly-labelled model knowledge fallback

    Tree routing is a relevance prior, never a hard evidence gate.
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

        # V2-A: independent global answer-only E5 rescue.
        #
        # Keep config.use_embedding=False so the legacy HybridRetriever remains
        # BM25-only.  This separate retriever loads the precomputed
        # answer_embeddings.npy built with multilingual-e5-large.
        self.answer_embedding_retriever = None
        if getattr(self.config, "use_answer_embedding", False):
            self.answer_embedding_retriever = AnswerEmbeddingRetriever(
                embedding_path=self.config.answer_embedding_path,
                expected_rows=len(self.taxonomy.df),
                model_name=self.config.answer_embedding_model,
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

    def _answer_embedding_hits(self, query):
        """Global answer-only semantic rescue; no Tree restriction and no rerank."""
        if self.answer_embedding_retriever is None:
            return []
        return self.answer_embedding_retriever.search(
            query,
            top_k=self.config.answer_embedding_top_k,
        )

    def _append_answer_embedding_evidence(self, evidence, embedding_hits):
        """Append only unique answer-embedding hits after the original BM25 context.

        Design for V2-A:
        - preserve the original BM25/RRF-selected evidence and its order exactly;
        - do not feed embedding hits into _merge_hits / RRF;
        - deduplicate exact atomic units;
        - append semantic rescue hits in embedding rank order;
        - do not impose a second context cutoff here.

        Therefore the maximum first-round context is:
            original context_limit + answer_embedding_top_k
        before duplicate removal.
        """
        selected = [dict(item) for item in (evidence or [])]

        seen_atomic_ids = {
            str(item.get("atomic_id", "")).strip()
            for item in selected
            if str(item.get("atomic_id", "")).strip()
        }
        seen_indices = {
            int(item["idx"])
            for item in selected
            if item.get("idx") is not None
        }

        candidate_ids = []
        added_ids = []
        duplicate_ids = []

        for hit in embedding_hits or []:
            idx = int(hit["idx"])
            record = self.taxonomy.document_record(idx)
            atomic_id = str(record.get("atomic_id", "")).strip()
            candidate_ids.append(atomic_id or f"ROW:{idx}")

            is_duplicate = (
                (atomic_id and atomic_id in seen_atomic_ids)
                or idx in seen_indices
            )
            if is_duplicate:
                duplicate_ids.append(atomic_id or f"ROW:{idx}")
                continue

            selected.append(
                {
                    **record,
                    "idx": idx,
                    "rank": len(selected) + 1,
                    "retrieval_score": float(hit.get("dense", 0.0)),
                    "bm25": 0.0,
                    "dense": float(hit.get("dense", 0.0)),
                    "channels": ["answer_embedding"],
                    "retrieved_paths": ["GLOBAL_ANSWER_EMBEDDING"],
                    "retrieval_units": [],
                }
            )

            seen_indices.add(idx)
            if atomic_id:
                seen_atomic_ids.add(atomic_id)
            added_ids.append(atomic_id or f"ROW:{idx}")

        # Keep the pre-existing order.  Only refresh display ranks.
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank

        trace = {
            "enabled": self.answer_embedding_retriever is not None,
            "top_k": int(getattr(self.config, "answer_embedding_top_k", 0)),
            "candidate_ids": candidate_ids,
            "added_ids": added_ids,
            "duplicate_ids": duplicate_ids,
            "original_evidence_count": len(evidence or []),
            "final_evidence_count": len(selected),
        }
        return selected, trace

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
                        "retrieval_units": [],
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

                # Preserve unit provenance through RRF.  Initial retrieval already
                # keeps units separate; refinement must not erase that identity
                # before the Answer/Judge sees the evidence.
                unit_values = []
                unit_id = str(hit.get("unit_id", "")).strip()
                if unit_id:
                    unit_values.append(unit_id)
                unit_values.extend(
                    str(x).strip()
                    for x in (hit.get("retrieval_units", []) or [])
                    if str(x).strip()
                )
                for value in unit_values:
                    if value not in entry["retrieval_units"]:
                        entry["retrieval_units"].append(value)

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
                    "retrieval_units": item.get("retrieval_units", []),
                }
            )
        return evidence

    def _rerank_refinement_by_missing(self, evidence, missing):
        """Rerank one refinement unit using only Judge missing relevance.

        RRF remains responsible for candidate generation/recall. Once refinement
        starts, the ranking objective changes: evidence should be ordered only by
        how directly it matches the unresolved `missing` description. No RRF,
        channel count, FAQ-family preference, or other unit contributes to this
        second-stage score. Ties keep the original unit-local order only as a
        deterministic fallback.
        """
        items = [dict(x) for x in (evidence or [])]
        if not items:
            return items

        target = str(missing or "").strip()
        if not target:
            return items

        texts = [
            f"問題：{str(item.get('question', '')).strip()}\n"
            f"答案：{str(item.get('answer', '')).strip()}"
            for item in items
        ]
        scores = BM25(texts).scores(target)

        decorated = []
        for original_pos, (item, score) in enumerate(zip(items, scores)):
            item["pre_missing_rerank_rank"] = int(item.get("rank", original_pos + 1))
            item["missing_relevance_score"] = float(score)
            decorated.append((float(score), original_pos, item))

        # Missing relevance is the ONLY refinement ranking score. Original order
        # is used only to make exact-score ties deterministic; it is not blended
        # into the relevance score.
        decorated.sort(key=lambda row: (-row[0], row[1]))
        reranked = [row[2] for row in decorated]
        for rank, item in enumerate(reranked, start=1):
            item["rank"] = rank
        return reranked

    @staticmethod
    def _tag_hits(hits, unit_id=None, query_view=None):
        """Attach trace-only metadata without changing retrieval scores."""
        tagged = []
        for hit in hits or []:
            item = dict(hit)
            if unit_id:
                item["unit_id"] = unit_id
            if query_view:
                item["query_view"] = query_view
            tagged.append(item)
        return tagged

    def _retrieval_units(self, query, plan):
        """Normalize Planner output; default conservatively to one unit."""
        raw_units = list(plan.get("retrieval_units", []) or [])[:3]
        units = []
        seen_queries = set()

        for i, raw in enumerate(raw_units, start=1):
            unit_query = str(raw.get("query", "")).strip()
            if not unit_query:
                continue
            key = unit_query.casefold()
            if key in seen_queries:
                continue
            seen_queries.add(key)

            keywords = []
            seen_keywords = set()
            for value in list(raw.get("keywords", []) or [])[:8]:
                kw = str(value or "").strip()
                if not kw:
                    continue
                kw_key = kw.casefold()
                if kw_key in seen_keywords:
                    continue
                seen_keywords.add(kw_key)
                keywords.append(kw)

            units.append(
                {
                    "unit_id": f"U{len(units) + 1}",
                    "query": unit_query,
                    "keywords": keywords,
                }
            )

        # Fail-safe: decomposition is optional. A normal single question must
        # continue to work even if the Planner omits retrieval_units.
        if not units:
            fallback_query = str(plan.get("search_query", "")).strip() or query
            units = [
                {
                    "unit_id": "U1",
                    "query": fallback_query,
                    "keywords": [],
                }
            ]

        return units

    def _retrieve_one_unit(self, unit, paths):
        """Multi-view BM25 retrieval for one semantic retrieval unit."""
        unit_id = unit["unit_id"]
        focused_query = str(unit.get("query", "")).strip()
        keyword_query = " ".join(unit.get("keywords", [])).strip()

        # View A: semantic focused query inside routed Tree paths.
        tree_hits = self._tag_hits(
            self._tree_hits(
                focused_query,
                paths,
                channel="tree",
            ),
            unit_id=unit_id,
            query_view="focused_tree",
        )

        # View B: the same focused query globally, protecting recall when the
        # correct atomic unit sits outside the routed Tree node.
        global_hits = self._tag_hits(
            self._global_hits(
                focused_query,
                top_k=self.config.unit_global_top_k,
                channel="global_rewrite",
            ),
            unit_id=unit_id,
            query_view="focused_global",
        )

        # View C: compact lexical anchors. This is deliberately Global because
        # exact cataloging tokens may reveal evidence missed by taxonomy routing.
        keyword_hits = []
        if keyword_query and keyword_query.casefold() != focused_query.casefold():
            keyword_hits = self._tag_hits(
                self._global_hits(
                    keyword_query,
                    top_k=self.config.unit_keyword_top_k,
                    channel="global_rewrite",
                ),
                unit_id=unit_id,
                query_view="keywords",
            )

        # FAQ/source-family view still helps when one original FAQ was split into
        # multiple atomic units.
        faq_hits = self._tag_hits(
            self._faq_hits(focused_query),
            unit_id=unit_id,
            query_view="focused_faq",
        )

        provenance_hits = self._tag_hits(
            self._provenance_hits(
                focused_query,
                tree_hits,
                global_hits,
                keyword_hits,
                faq_hits,
            ),
            unit_id=unit_id,
            query_view="unit_provenance",
        )

        # Merge within the unit FIRST. This prevents an easy sub-question from
        # flooding the final context and starving another unit of evidence.
        evidence = self._merge_hits(
            tree_hits,
            global_hits,
            keyword_hits,
            faq_hits,
            provenance_hits,
            limit=self.config.unit_context_top_k,
        )

        return {
            "unit_id": unit_id,
            "query": focused_query,
            "keywords": list(unit.get("keywords", [])),
            "evidence": evidence,
            "raw_hits": {
                "tree": tree_hits,
                "global": global_hits,
                "keywords": keyword_hits,
                "faq": faq_hits,
                "provenance": provenance_hits,
            },
        }

    def _combine_unit_evidence(self, unit_results, rescue_evidence, limit):
        """Round-robin unit coverage, then fill spare slots with rescue hits."""
        limit = max(1, int(limit))
        selected = []
        by_idx = {}

        def add_item(raw_item, unit_id=None):
            idx = int(raw_item["idx"])
            if idx in by_idx:
                existing = by_idx[idx]
                if unit_id:
                    units = existing.setdefault("retrieval_units", [])
                    if unit_id not in units:
                        units.append(unit_id)
                for channel in raw_item.get("channels", []):
                    channels = existing.setdefault("channels", [])
                    if channel not in channels:
                        channels.append(channel)
                for path in raw_item.get("retrieved_paths", []):
                    paths = existing.setdefault("retrieved_paths", [])
                    if path not in paths:
                        paths.append(path)
                existing["retrieval_score"] = max(
                    float(existing.get("retrieval_score", 0.0)),
                    float(raw_item.get("retrieval_score", 0.0)),
                )
                existing["bm25"] = max(
                    float(existing.get("bm25", 0.0)),
                    float(raw_item.get("bm25", 0.0)),
                )
                return False

            item = dict(raw_item)
            item["retrieval_units"] = [unit_id] if unit_id else []
            by_idx[idx] = item
            selected.append(item)
            return True

        max_depth = max(
            (len(result.get("evidence", [])) for result in unit_results),
            default=0,
        )

        # U1-rank1, U2-rank1, ... then rank2, etc.
        for depth in range(max_depth):
            for result in unit_results:
                evidence = result.get("evidence", [])
                if depth >= len(evidence):
                    continue
                add_item(evidence[depth], unit_id=result["unit_id"])
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break

        # Original-query rescue preserves robustness against a bad Planner rewrite
        # or an unnecessary decomposition.
        if len(selected) < limit:
            for item in rescue_evidence or []:
                add_item(item, unit_id=None)
                if len(selected) >= limit:
                    break

        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        return selected

    def _initial_retrieve(self, query, plan):
        paths = plan["paths"]
        units = self._retrieval_units(query, plan)

        # Preserve the literal user wording as a Tree view. This is useful for
        # exact tokens such as 245, $n, LDR/19, #4, or a rare subject term.
        original_tree_hits = self._tag_hits(
            self._tree_hits(query, paths, channel="tree"),
            unit_id=None,
            query_view="original_tree",
        )

        # Backward-compatible overall Planner rewrite remains a global rescue view.
        rewrite = str(plan.get("search_query", "")).strip()
        rewrite_global = []
        if rewrite and rewrite.casefold() != query.casefold():
            rewrite_global = self._tag_hits(
                self._global_hits(
                    rewrite,
                    top_k=self.config.global_rewrite_top_k,
                    channel="global_rewrite",
                ),
                unit_id=None,
                query_view="planner_rewrite",
            )

        # All unit retrieval below is local BM25; parallelism adds no LLM calls.
        unit_results = []
        workers = max(1, min(3, len(units)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._retrieve_one_unit, unit, paths)
                for unit in units
            ]
            for future in futures:
                unit_results.append(future.result())

        return units, unit_results, original_tree_hits, rewrite_global

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
            r"(?i)\btag\s*0*(\d{3})(?!\d)",
            r"(?i)\bmarc(?:\s*21)?\s*(?:欄位|field)?\s*0*(\d{3})(?!\d)",
            r"欄位\s*0*(\d{3})(?!\d)",
            r"0*(\d{3})\s*(?:欄位|field)",
            r"(\d{3})\s*段",
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
        unit_query,
        refinement_query,
        missing,
        unit_first_round_evidence,
    ):
        """Build one unit-local evidence-guided second-pass query.

        PRF is deliberately scoped to the retrieval unit being refined. Evidence
        from other units never participates in feedback selection.  The Judge's
        ``missing`` description is the main relevance target; the unit query is
        retained for structural constraints (MARC tag, subfield, indicator, etc.).
        Low-quality feedback is optional rather than forced to fill Top-K.
        """
        base_parts = [
            str(unit_query or "").strip(),
            str(refinement_query or "").strip(),
            str(missing or "").strip(),
        ]
        base_query = "\n".join(x for x in base_parts if x)

        evidence = list(unit_first_round_evidence or [])
        feedback_items = []
        for item in evidence:
            feedback_text = self._compact_feedback_text(
                item,
                answer_chars=self.config.refinement_feedback_answer_chars,
            )
            if feedback_text:
                feedback_items.append((item, feedback_text))

        if not feedback_items:
            return base_query, [], []

        # Missing-aspect relevance decides whether a first-round item is useful
        # feedback.  The full unit context is used only for exact structural
        # constraints so an unrelated MARC field cannot drift the query.
        target_text = (
            f"{refinement_query}\n{missing}".strip()
            or str(unit_query or "").strip()
        )
        constraint_target = (
            f"{unit_query}\n{refinement_query}\n{missing}".strip()
        )
        feedback_texts = [text for _, text in feedback_items]
        scores = BM25(feedback_texts).scores(target_text)
        overlaps = [
            self._constraint_overlap(constraint_target, text)
            for text in feedback_texts
        ]
        has_constraints = bool(self._structural_constraints(constraint_target))

        candidate_indices = list(range(len(feedback_items)))
        if has_constraints:
            # Exact MARC/Leader/subfield/indicator constraints are hard PRF
            # guards. If none of the accepted feedback evidence shares the
            # structural constraint, use no PRF rather than letting generic
            # lexical overlap drift the refinement into another field.
            candidate_indices = [i for i in candidate_indices if overlaps[i] > 0]

        order = sorted(
            candidate_indices,
            key=lambda i: (overlaps[i], float(scores[i])),
            reverse=True,
        )

        # Do not force refinement_feedback_top_k items into PRF.  Without a
        # structural match, require a positive BM25 score and at least 35% of
        # the best candidate's score.  This is a conservative drift guard.
        best_score = max((float(scores[i]) for i in candidate_indices), default=0.0)
        relative_ratio = float(
            getattr(self.config, "refinement_feedback_relative_floor", 0.35)
        )
        relative_ratio = max(0.0, min(1.0, relative_ratio))
        relative_floor = best_score * relative_ratio if best_score > 0 else 0.0

        selected = []
        selected_ids = []
        selected_index_set = set()
        limit = max(0, int(self.config.refinement_feedback_top_k))
        for i in order:
            if len(selected) >= limit:
                break
            score = float(scores[i])
            overlap = int(overlaps[i])
            if overlap <= 0:
                if score <= 0 or score < relative_floor:
                    continue
            selected.append(feedback_texts[i])
            selected_index_set.add(i)
            evidence_id = str(feedback_items[i][0].get("atomic_id", "")).strip()
            if evidence_id:
                selected_ids.append(evidence_id)

        feedback_trace = []
        for i, (item, _) in enumerate(feedback_items):
            feedback_trace.append(
                {
                    "atomic_id": str(item.get("atomic_id", "")).strip(),
                    "bm25_score": float(scores[i]),
                    "constraint_overlap": int(overlaps[i]),
                    "selected": i in selected_index_set,
                }
            )

        if not selected:
            return base_query, [], feedback_trace

        expanded = (
            f"{base_query}\n"
            "此 unit 第一輪高可信 evidence 用語：\n"
            + "\n".join(selected)
        ).strip()
        return expanded, selected_ids, feedback_trace

    @staticmethod
    def _unique_strings(values):
        result = []
        seen = set()
        for value in values or []:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _select_refinement_unit_results(
        self,
        unit_results,
        requested_unit_ids,
        refinement_query,
        missing,
    ):
        """Choose only the units that need the second retrieval round.

        The first-pass Judge is the primary selector.  A deterministic BM25
        fallback is used only when the Judge omits/returns invalid unit IDs.
        """
        unit_results = list(unit_results or [])
        by_id = {str(x.get("unit_id", "")): x for x in unit_results}
        requested = [
            unit_id
            for unit_id in self._unique_strings(requested_unit_ids)
            if unit_id in by_id
        ]
        if requested:
            selected = [by_id[unit_id] for unit_id in requested[:3]]
            return selected, {
                "selection_source": "judge",
                "requested_unit_ids": requested,
                "selected_unit_ids": [x["unit_id"] for x in selected],
                "fallback_scores": [],
            }

        if not unit_results:
            return [], {
                "selection_source": "none",
                "requested_unit_ids": [],
                "selected_unit_ids": [],
                "fallback_scores": [],
            }
        if len(unit_results) == 1:
            return [unit_results[0]], {
                "selection_source": "single_unit_fallback",
                "requested_unit_ids": [],
                "selected_unit_ids": [unit_results[0]["unit_id"]],
                "fallback_scores": [],
            }

        target = f"{refinement_query}\n{missing}".strip()
        unit_texts = [
            f"{x.get('query', '')}\n{' '.join(x.get('keywords', []))}".strip()
            for x in unit_results
        ]
        scores = BM25(unit_texts).scores(target) if target else [0.0] * len(unit_texts)
        ranked = sorted(
            range(len(unit_results)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )
        best = ranked[0]
        selected = [unit_results[best]]
        score_trace = [
            {
                "unit_id": unit_results[i]["unit_id"],
                "score": float(scores[i]),
            }
            for i in ranked
        ]
        return selected, {
            "selection_source": "bm25_fallback",
            "requested_unit_ids": [],
            "selected_unit_ids": [unit_results[best]["unit_id"]],
            "fallback_scores": score_trace,
        }

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
        unit_id,
        unit_query,
        refinement_query,
        missing,
        base_paths,
        unit_first_round_evidence,
    ):
        search_query, feedback_ids, feedback_trace = (
            self._build_refinement_search_query(
                unit_query,
                refinement_query,
                missing,
                unit_first_round_evidence,
            )
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

        # Every refinement channel is scoped to the selected unit.  The same
        # retrieval algorithm is used for every unit; only its own query/evidence
        # are visible, preventing cross-unit PRF/provenance contamination.
        tree_hits = self._tag_hits(
            self._sibling_scope_hits(search_query, sibling_scopes),
            unit_id=unit_id,
            query_view="unit_refinement_tree",
        )
        global_hits = self._tag_hits(
            self._global_hits(
                search_query,
                top_k=self.config.global_top_k,
                channel="refine_global",
            ),
            unit_id=unit_id,
            query_view="unit_refinement_global",
        )
        faq_hits = self._tag_hits(
            self._faq_hits(search_query),
            unit_id=unit_id,
            query_view="unit_refinement_faq",
        )

        # Critical continuity rule: the selected unit's own first-round evidence
        # is the first provenance anchor group.  If a unit already reached the
        # correct source family, the second round can inspect sibling atomics
        # without depending on a noisy new search to rediscover that family.
        provenance_query = (
            f"{unit_query}\n{refinement_query}\n{missing}".strip()
        )
        provenance_hits = self._tag_hits(
            self._provenance_hits(
                provenance_query,
                unit_first_round_evidence,
                global_hits,
                faq_hits,
                tree_hits,
            ),
            unit_id=unit_id,
            query_view="unit_refinement_provenance",
        )

        # Unit-local second-pass ranking. Every first-round evidence item from
        # THIS unit remains a candidate, and all new Tree/Global/FAQ/Provenance
        # hits are merged and ranked only against one another inside this unit.
        # No other retrieval unit participates in this ranking, and no single
        # feedback hit or FAQ family is privileged merely for being rank #1.
        unit_first_round_hits = []
        for rank, item in enumerate(unit_first_round_evidence or [], start=1):
            unit_first_round_hits.append(
                {
                    "idx": int(item["idx"]),
                    "bm25": item.get("bm25", 0.0),
                    "channel": "first_round",
                    "channel_rank": rank,
                    "route_rank": None,
                    "route_score": 0.0,
                    "path": "UNIT_FIRST_ROUND",
                    "unit_id": unit_id,
                    "retrieval_units": [unit_id],
                }
            )

        unit_refined_evidence = self._merge_hits(
            unit_first_round_hits,
            tree_hits,
            global_hits,
            faq_hits,
            provenance_hits,
            limit=None,
        )

        # Keep the multi-channel RRF order for refinement. A pure lexical
        # rerank against `missing` over-rewards wording overlap and can promote
        # documents that repeat decision words while answering a different
        # cataloging rule. The missing aspect still drives the second-pass query,
        # but it is not used as a single-score replacement for RRF.

        meta = {
            "unit_id": unit_id,
            "unit_query": str(unit_query or "").strip(),
            "requested_query": str(refinement_query or "").strip(),
            "expanded_query": search_query,
            "feedback_evidence_ids": feedback_ids,
            "feedback_candidates": feedback_trace,
            "provenance_anchor_evidence_ids": [
                str(x.get("atomic_id", "")).strip()
                for x in (unit_first_round_evidence or [])
                if str(x.get("atomic_id", "")).strip()
            ],
            "provenance_evidence_ids": [
                str(self.taxonomy.document_record(int(x["idx"])).get("atomic_id", "")).strip()
                for x in provenance_hits
            ],
            "refinement_ranking_mode": "rrf_multichannel",
            "unit_refined_evidence_ids": [
                str(x.get("atomic_id", "")).strip()
                for x in unit_refined_evidence
                if str(x.get("atomic_id", "")).strip()
            ],
            "missing_rerank_scores": [],
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
        return (
            paths,
            tree_hits,
            global_hits,
            faq_hits,
            provenance_hits,
            unit_refined_evidence,
            meta,
        )

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
            unit_tags = ",".join(item.get("retrieval_units", []) or []) or "unscoped"
            blocks.append(
                f"[{source_id}] retrieval_units={unit_tags}; retrieval_channels={channels}\n"
                f"日期：{item.get('question_date', '')}\n"
                f"問題：{item.get('question', '')}\n"
                f"答案：{item.get('answer', '')}\n"
                f"taxonomy：{item.get('taxonomy_paths', '')}"
            )
        return "\n\n".join(blocks)

    def _answer_schema(self, allowed_ids, allow_partial, unit_ids=None):
        # V4.0.2:
        # First pass is retrieval diagnosis only. It may either answer from DB
        # evidence (grounded) or request exactly one refinement (partial).
        # Model-knowledge fallback is intentionally unavailable until AFTER the
        # refinement round has been attempted.
        statuses = (
            ["grounded", "partial"]
            if allow_partial
            else ["grounded", "partial", "knowledge_fallback"]
        )
        unit_ids = self._unique_strings(unit_ids)
        unit_item_schema = (
            {"type": "string", "enum": unit_ids}
            if unit_ids
            else {"type": "string"}
        )
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": statuses},
                "answer": {"type": "string"},
                "missing": {"type": "string"},
                "refinement_query": {"type": "string"},
                "refinement_unit_ids": {
                    "type": "array",
                    "items": unit_item_schema,
                },
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": allowed_ids},
                },
                "knowledge_used": {"type": "boolean"},
                "question_requirement": {
                    "type": "string",
                    "enum": [
                        "general_rule",
                        "case_specific",
                        "exact_fact",
                        "current_latest",
                        "mixed",
                    ],
                },
                "evidence_directly_answers": {"type": "boolean"},
                "required_precision_supported": {"type": "boolean"},
                "unsupported_answer_aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "status",
                "answer",
                "missing",
                "refinement_query",
                "refinement_unit_ids",
                "evidence_ids",
                "knowledge_used",
                "question_requirement",
                "evidence_directly_answers",
                "required_precision_supported",
                "unsupported_answer_aspects",
            ],
        }

    def _answer_once(
        self,
        query,
        evidence,
        source_evidence=None,
        allow_partial=True,
        retrieval_units=None,
        prior_decision=None,
    ):
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
        retrieval_units = list(retrieval_units or [])
        unit_ids = [str(x.get("unit_id", "")).strip() for x in retrieval_units]
        unit_lines = [
            f"[{x.get('unit_id', '')}] {x.get('query', '')}"
            for x in retrieval_units
            if str(x.get("unit_id", "")).strip()
        ]
        unit_context = "\n".join(unit_lines) or "(單一未標記 retrieval unit)"

        prior_sufficiency_context = ""
        if prior_decision:
            prior_requirement = str(
                prior_decision.get("question_requirement", "")
            ).strip()
            prior_direct = bool(
                prior_decision.get("evidence_directly_answers", False)
            )
            prior_precision = bool(
                prior_decision.get("required_precision_supported", False)
            )
            prior_unsupported = self._unique_strings(
                prior_decision.get("unsupported_answer_aspects", []) or []
            )

            prior_lines = [
                "=== 第一輪尚未解決的充分性條件（refinement 後必須重新驗證）===",
                f"第一輪 question_requirement: {prior_requirement or '(未標記)'}",
                f"第一輪 evidence_directly_answers: {prior_direct}",
                f"第一輪 required_precision_supported: {prior_precision}",
                "第一輪 unsupported_answer_aspects:",
            ]
            if prior_unsupported:
                prior_lines.extend(f"- {x}" for x in prior_unsupported)
            else:
                prior_lines.append("- (無)")

            prior_lines.extend(
                [
                    "",
                    "最終判斷規則：",
                    "1. 若第一輪曾有 unsupported_answer_aspects，refinement 後必須逐項確認是否被本輪 evidence『直接解決』。",
                    "2. 只有當 evidence 明確提供能解決該缺口的新資訊時，才可把 required_precision_supported 改成 true。",
                    "3. 只是再次看到第一輪相同的 evidence、相近案例、舊版資料或一般背景，不算解決原缺口。",
                    "4. 若缺口涉及 current/latest/version，必須有 evidence 明確支持目前/最新版狀態，或明確證明舊版規則截至所問時點仍有效；僅有舊版的確切答案不足。",
                    "5. 若任何第一輪 unsupported aspect 仍未被直接解決，最終不得把整題判成完整 grounded；答案應保留已由資料庫確認的部分，並清楚說明仍無法確定的部分。",
                ]
            )
            prior_sufficiency_context = "\n".join(prior_lines)

        schema = self._answer_schema(
            ids,
            allow_partial=allow_partial,
            unit_ids=unit_ids,
        )

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
- grounded：核心答案的每一個必要部分，都已由目前 evidence 直接支持或安全組合支持。knowledge_used=false。
- partial：refinement 後仍有某些使用者要求的精確面向未被 evidence 直接支持，但資料庫已能確認部分內容。knowledge_used=false。此時不要硬猜；答案必須先說「資料庫可確認的內容」，再清楚說「現有材料仍無法確認的部分」。
- knowledge_fallback：refinement 後資料庫仍不足，而你確實需要使用模型既有知識補充一般知識時才使用。knowledge_used=true，且必須清楚標示未由本次資料庫直接驗證的部分。"""
            final_instruction = (
                "這是 refinement 後的最終判斷。"
                "若 evidence 只能支持部分答案，請使用 partial，"
                "先保留已確認的資料庫事實，再明確說明仍無法定案的面向；"
                "不要為了給出完整答案而把舊版、相近案例或未驗證的精確值當成已確認。"
            )

        prompt = f"""你是國家圖書館 Tree-RAG 的 Answer/Judge。

使用者問題：
{query}

本題 retrieval units：
{unit_context}

{prior_sufficiency_context if prior_sufficiency_context else ""}

本次資料庫檢索 evidence：
{context if context else '(沒有檢索到 evidence)'}

{status_instructions}

在決定 status 與作答前，先完成以下「需求－證據充分性檢查」。這不是要輸出長篇推理，只需把結論填入 JSON 欄位：

A. question_requirement
先判斷使用者真正要求的答案層級：
- general_rule：一般規則、定義、概念、清單、欄位意義等。
- case_specific：要求對某一特定資料、實物、紀錄或館務情境作判斷。
- exact_fact：要求確切類號、代碼、indicator、subfield、年份、數值、欄位內容或其他精確值。
- current_latest：要求「現行、目前、最新版、最新修訂、截至現在」等具有版本或時效性的結論。
- mixed：同時包含上述兩種以上要求。

B. evidence_directly_answers
判斷目前 evidence 是否直接回答了使用者真正問的事項。
- true：evidence 明確支持核心結論。
- false：evidence 只有相近案例、一般背景、部分欄位、部分子問題，或只能間接推論。
注意：『語意相關』不等於『直接支持答案』。

C. required_precision_supported
判斷 evidence 是否支持到使用者要求的精確程度。
- 若使用者問 exact code / exact field / exact class number，evidence 必須直接支持該確切值。
- 若使用者問 current/latest，evidence 必須能證明其版本或時點足以代表目前/最新版；舊版資料不能自動推成現行版本仍相同。
- 若問題包含多個子項，每個會影響核心答案的子項都必須有支持，不能因多數子項已找到就把整題視為充分。
- 若特定個案的唯一結論仍依賴使用者未提供的實物、語言、版本、內容比重、來源位置、館藏政策、系統設定等事實，應視為不足。

D. unsupported_answer_aspects
如果 evidence_directly_answers=false 或 required_precision_supported=false，列出「目前 evidence 還不能支持」的具體答案面向。
例如：
- 「255 $d 的確切定義」
- 「008/33 雙月刊的確切代碼」
- 「現行最新版是否仍沿用 624.13」
不要只寫「資料不足」；要指出缺的是哪個答案面向。

完成 A～D 後，再決定 status：
- grounded：只有 evidence_directly_answers=true 且 required_precision_supported=true，且核心答案各部分均可由 evidence 直接或安全組合支持時才可選。
- partial：第一輪若仍有 unsupported_answer_aspects，應優先回 partial 並針對缺口 refinement。
- knowledge_fallback：只在 refinement 後仍缺「一般資料庫知識」時使用；不得把未被 evidence 支持的 exact code、exact number、current/latest 狀態或使用者未提供的個案事實，僅憑模型記憶說成已確定。


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
   - 若 status=partial，evidence_ids 只能列「目前已能直接支持的部分」所使用的證據。
   - 不要把僅僅相關、但仍不足以支持該部分結論的 evidence 放進 evidence_ids。
   - 系統會在 refinement 時保留這些已接受 evidence，因此這個欄位代表「已支持、可安全 carry-forward」的證據，而不是一般相關文件清單。
6. grounded 時不要加入模型既有知識。
7. knowledge_fallback 時，答案中必須出現「【模型知識補充｜未由本次資料庫直接驗證】」。若 evidence 有可確認部分，可先列「資料庫可確認」再補充。
8. 逐一判斷每個 retrieval unit 是否已被 evidence 覆蓋。evidence 行首的 retrieval_units 表示該證據由哪些 unit 的檢索取得；這是 coverage 判斷的重要線索，但不是硬性限制，同一正式證據可安全支援多個 unit。
9. 允許「多筆 evidence 的安全組合」：例如一筆說明特定 indicator/條件的意義，另一筆說明該欄位的一般使用規則；只要兩者沒有衝突，且組合後沒有新增 evidence 未寫出的條件，就可以共同支持答案。不要要求所有結論都必須逐字出現在同一個 atomic unit。
10. 不得把「相關 evidence」誤當成「足以支持使用者要求精度的 evidence」。
   - 若 question_requirement=exact_fact，沒有直接支持確切值時，不得只靠相近欄位、相近案例或常識補成確切答案。
   - 若 question_requirement=current_latest，舊版、舊答覆或未標示時點的 evidence 不能自動證明現行最新版仍相同。
   - 若問題有多個子項，只要核心子項仍在 unsupported_answer_aspects 中，就不能把整題視為完整 grounded。
   - 若是 case_specific，其他案例的定案不能替代目前個案缺少的必要事實。
11. refinement 是「補第一輪缺口」，不是重新忘記第一輪限制後從頭猜答案。
   - 若第一輪 required_precision_supported=false，第二輪必須先檢查原 unsupported_answer_aspects 是否真的被新 evidence 解決。
   - 未被解決時，不得僅因原答案再次出現在 evidence 中，就把 required_precision_supported 改成 true。
   - 對 current/latest/version 類問題尤其如此：2007 年版的正確答案只能證明 2007 年版，不能單獨證明現行最新版。

12. 若最終 status=partial，答案採「已確認事實 + 未確認限制」的形式：
   - 先直接回答目前 evidence 已經能確定的部分，不要只說「資料不足」。
   - 再指出使用者原問題中哪一個面向仍未被現有材料支持。
   - 例如，若 evidence 只證明 2007 年版《武則天傳》為 624.13，而題目問「現行最新版」，應回答：
     「依目前資料可確認，《中文圖書分類法》2007 年版中《武則天傳》為 624.13；但現有材料未提供足以確認現行最新版是否仍沿用此號的證據，因此不能把 624.13 直接當成現行最新版的定案。」
   - 這種回答不是 knowledge_fallback，因為沒有使用模型知識補造缺口。

13. knowledge_fallback 只能補一般知識，不能把下列內容說成已由本次資料庫確定：
   - evidence 未支持的 exact code / exact number / exact field value；
   - 未被 evidence 證明的 current/latest/version 狀態；
   - 使用者未提供、且只能由實物或館內情境確認的個案事實。
   對這些內容應明確標示仍無法由本次資料確定，而不是自行補成唯一答案。
14. partial 的 refinement_query 只應描述仍未回答完整的 unit 及其 missing aspect；不要把已經有充分 evidence 的 unit 再混入 query，也不得把你猜測的答案值塞進搜尋詞。
15. refinement_unit_ids：
   - status=partial 時，只列仍缺 evidence、需要第二輪檢索的 unit_id。
   - 已有充分 evidence 的 unit 不得列入，即使整題仍為 partial。
   - status=grounded 或 knowledge_fallback 時填空陣列。
16. 回答使用繁體中文，直接回答館員問題，不輸出內部 reasoning。

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

        # --------------------------------------------------------------
        # General sufficiency diagnostics (observation only)
        # --------------------------------------------------------------
        # These fields are normalized for trace analysis only.
        # Python does NOT override status or answer based on them.
        allowed_requirements = {
            "general_rule",
            "case_specific",
            "exact_fact",
            "current_latest",
            "mixed",
        }
        question_requirement = str(
            result.get("question_requirement", "")
        ).strip()
        if question_requirement not in allowed_requirements:
            question_requirement = "general_rule"

        evidence_directly_answers = bool(
            result.get("evidence_directly_answers", False)
        )
        required_precision_supported = bool(
            result.get("required_precision_supported", False)
        )
        unsupported_answer_aspects = self._unique_strings(
            result.get("unsupported_answer_aspects", []) or []
        )

        # Keep the diagnostic fields internally consistent for analysis,
        # without changing the model's status/answer.
        if evidence_directly_answers and required_precision_supported:
            unsupported_answer_aspects = []

        result["question_requirement"] = question_requirement
        result["evidence_directly_answers"] = evidence_directly_answers
        result["required_precision_supported"] = required_precision_supported
        result["unsupported_answer_aspects"] = unsupported_answer_aspects

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
            if status == "partial":
                # Final partial is a legitimate "supported fact + unresolved limit"
                # outcome.  It does not imply model-knowledge use.
                knowledge_used = False
            elif status != "grounded":
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
        allowed_unit_ids = set(unit_ids)
        result["refinement_unit_ids"] = [
            x
            for x in self._unique_strings(result.get("refinement_unit_ids", []))
            if x in allowed_unit_ids
        ]
        if status != "partial" or not allow_partial:
            result["refinement_unit_ids"] = []
        result["status"] = status
        result["knowledge_used"] = status == "knowledge_fallback" or knowledge_used

        if status == "partial":
            missing = str(result.get("missing", "")).strip()
            if not missing:
                unsupported = self._unique_strings(
                    result.get("unsupported_answer_aspects", []) or []
                )
                if unsupported:
                    missing = "目前仍無法由 evidence 確認：" + "、".join(unsupported)
                else:
                    missing = "目前資料庫 evidence 尚不足以直接支持完整答案"
            result["missing"] = missing

            if allow_partial:
                refinement_query = str(result.get("refinement_query", "")).strip()
                if not refinement_query:
                    refinement_query = f"{query} {missing}".strip()
                result["refinement_query"] = refinement_query
            else:
                # Final partial: there is no third retrieval round.
                result["refinement_query"] = ""

        answer = str(result.get("answer", "")).strip()
        if result["status"] == "knowledge_fallback" and self.config.allow_model_knowledge_fallback:
            marker = "【模型知識補充｜未由本次資料庫直接驗證】"
            if marker not in answer:
                answer = f"{marker}\n{answer}".strip()
        result["answer"] = answer
        return result

    def _carry_forward_accepted_evidence(
        self,
        initial_decision,
        first_round_evidence,
        refined_evidence,
        limit,
    ):
        """Preserve evidence explicitly accepted by the first-pass Judge.

        Refinement is additive: it may discover new evidence for the missing
        aspect, but it must not evict evidence that the first-pass Judge already
        cited as directly supporting a resolved part of the question.

        This is deliberately narrower than blanket first-round protection:
        only ``initial_decision['evidence_ids']`` are protected.  All other
        first-round hits continue to compete normally in RRF.
        """
        limit = max(1, int(limit))

        accepted_ids = []
        seen_ids = set()
        for value in initial_decision.get("evidence_ids", []) or []:
            atomic_id = str(value or "").strip()
            if not atomic_id or atomic_id in seen_ids:
                continue
            seen_ids.add(atomic_id)
            accepted_ids.append(atomic_id)

        first_by_id = {
            str(item.get("atomic_id", "")).strip(): item
            for item in (first_round_evidence or [])
            if str(item.get("atomic_id", "")).strip()
        }
        refined_by_id = {
            str(item.get("atomic_id", "")).strip(): item
            for item in (refined_evidence or [])
            if str(item.get("atomic_id", "")).strip()
        }

        selected = []
        selected_idx = set()
        protected_ids = []

        def append_item(raw, protected=False):
            if not raw:
                return False
            idx = int(raw["idx"])
            if idx in selected_idx:
                return False
            item = dict(raw)
            if protected:
                channels = list(item.get("channels", []))
                if "protected_first_round" not in channels:
                    channels.append("protected_first_round")
                item["channels"] = channels

                paths = list(item.get("retrieved_paths", []))
                if "PROTECTED_FIRST_ROUND" not in paths:
                    paths.append("PROTECTED_FIRST_ROUND")
                item["retrieved_paths"] = paths

            selected_idx.add(idx)
            selected.append(item)
            return True

        # Accepted evidence is placed first so a later RRF merge cannot push it
        # outside the final context budget. Prefer the richer refined copy when
        # it is still present; otherwise recover the original first-round copy.
        for atomic_id in accepted_ids:
            raw = refined_by_id.get(atomic_id) or first_by_id.get(atomic_id)
            if append_item(raw, protected=True):
                protected_ids.append(atomic_id)
            if len(selected) >= limit:
                break

        # Fill the remaining context slots with the normal refinement ranking.
        if len(selected) < limit:
            for item in refined_evidence or []:
                append_item(item, protected=False)
                if len(selected) >= limit:
                    break

        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank

        return selected, protected_ids

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
            "initial_question_requirement": initial.get(
                "question_requirement", "general_rule"
            ),
            "initial_evidence_directly_answers": bool(
                initial.get("evidence_directly_answers", False)
            ),
            "initial_required_precision_supported": bool(
                initial.get("required_precision_supported", False)
            ),
            "initial_unsupported_answer_aspects": list(
                initial.get("unsupported_answer_aspects", []) or []
            ),
            "final_question_requirement": final.get(
                "question_requirement", "general_rule"
            ),
            "final_evidence_directly_answers": bool(
                final.get("evidence_directly_answers", False)
            ),
            "final_required_precision_supported": bool(
                final.get("required_precision_supported", False)
            ),
            "final_unsupported_answer_aspects": list(
                final.get("unsupported_answer_aspects", []) or []
            ),
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

        # Planner is remote; original-query Global/FAQ and answer embedding
        # retrieval are local and independent.
        with ThreadPoolExecutor(max_workers=4) as pool:
            plan_future = pool.submit(self.router.plan, query)
            global_future = pool.submit(self._global_hits, query)
            faq_future = pool.submit(self._faq_hits, query)
            answer_embedding_future = pool.submit(
                self._answer_embedding_hits,
                query,
            )

            plan = plan_future.result()
            global_hits = global_future.result()
            faq_hits = faq_future.result()
            answer_embedding_hits = answer_embedding_future.result()

        t_plan_parallel = time.perf_counter()

        # --------------------------------------------------------------
        # Scope Gate: in-domain-but-unanswerable is NOT out_of_scope.
        # This check must happen after `plan` exists and before answer/retrieval
        # logic that assumes a normal cataloging question.
        # --------------------------------------------------------------
        if not plan.get("in_scope", True):
            scope_decision = {
                "status": "out_of_scope",
                "answer": OUT_OF_SCOPE_MESSAGE,
                "missing": "",
                "refinement_query": "",
                "evidence_ids": [],
                "knowledge_used": False,
                "question_requirement": "general_rule",
                "evidence_directly_answers": False,
                "required_precision_supported": False,
                "unsupported_answer_aspects": [],
            }
            elapsed = round(time.perf_counter() - t0, 4)
            return {
                "answer": OUT_OF_SCOPE_MESSAGE,
                "mode": "out_of_scope",
                "knowledge_used": False,
                "initial_decision": scope_decision,
                "final_decision": scope_decision,
                "answer_scope": "out_of_scope",
                "refinement_used": False,
                "refinement": None,
                "planner": {
                    "in_scope": False,
                    "query_focus": "",
                    "search_query": "",
                    "retrieval_units": [],
                    "fallback": bool(plan.get("fallback", False)),
                },
                "unit_retrieval": [],
                "paths": [],
                "source_retrieval": {
                    "source_count": 0,
                    "sources": [],
                },
                "retrieval": {
                    "evidence_count": 0,
                    "evidence": [],
                },
                "timing": {
                    "planner_plus_parallel_local_retrieval": round(
                        t_plan_parallel - t0, 4
                    ),
                    "tree_merge_retrieval": 0.0,
                    "first_answer_judge": 0.0,
                    "refinement_and_final_answer": 0.0,
                    "total": elapsed,
                },
            }

        # Completeness intent is local and orthogonal to Tree routing.
        answer_scope = self._answer_scope(query)

        (
            retrieval_units,
            unit_results,
            original_tree_hits,
            rewrite_global,
        ) = self._initial_retrieve(query, plan)

        # Original-query rescue is deliberately independent of decomposition.
        # If the Planner over-splits or rewrites poorly, literal user wording still
        # has a full Tree + Global + FAQ + provenance path into final context.
        original_provenance = self._provenance_hits(
            query,
            global_hits,
            rewrite_global,
            original_tree_hits,
            faq_hits,
        )
        rescue_evidence = self._merge_hits(
            original_tree_hits,
            global_hits,
            rewrite_global,
            faq_hits,
            original_provenance,
            limit=self.config.context_limit,
        )

        # Ensure each semantic unit gets evidence quota before original-query
        # rescue fills remaining slots.
        evidence = self._combine_unit_evidence(
            unit_results,
            rescue_evidence,
            limit=self.config.context_limit,
        )

        # V2-A causal experiment:
        # preserve the complete original BM25-selected context, then append
        # unique global answer-embedding Top-K hits.  No RRF/reranker is applied
        # to the embedding lane.
        evidence, answer_embedding_trace = (
            self._append_answer_embedding_evidence(
                evidence,
                answer_embedding_hits,
            )
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
            retrieval_units=retrieval_units,
        )
        t_first_answer = time.perf_counter()

        final = initial
        refinement_used = False
        refinement_trace = None
        final_paths = list(plan["paths"])

        # V4.3.2-U: one second-pass round, but at retrieval-unit granularity.
        # Resolved units are preserved; only units identified as missing are
        # re-searched, each with its own first-round evidence pool.
        if self.config.enable_refinement and initial.get("status") == "partial":
            refinement_used = True
            missing = str(initial.get("missing", "")).strip()
            refinement_query = str(initial.get("refinement_query", "")).strip()
            if not refinement_query:
                refinement_query = missing or query

            selected_unit_results, unit_selection = (
                self._select_refinement_unit_results(
                    unit_results,
                    initial.get("refinement_unit_ids", []),
                    refinement_query,
                    missing,
                )
            )

            path_map = {p.key(): p for p in plan["paths"]}
            unit_refinement_traces = []
            refined_by_unit_id = {}

            accepted_first_pass_ids = {
                str(x).strip()
                for x in (initial.get("evidence_ids", []) or [])
                if str(x).strip()
            }

            for unit_result in selected_unit_results:
                full_unit_first_round = list(unit_result.get("evidence", []) or [])
                accepted_unit_first_round = [
                    item
                    for item in full_unit_first_round
                    if str(item.get("atomic_id", "")).strip()
                    in accepted_first_pass_ids
                ]

                (
                    unit_paths,
                    refine_tree,
                    refine_global,
                    refine_faq,
                    refine_provenance,
                    unit_refined_evidence,
                    unit_meta,
                ) = self._refinement_retrieve(
                    unit_result["unit_id"],
                    unit_result["query"],
                    refinement_query,
                    missing,
                    plan["paths"],
                    accepted_unit_first_round,
                )
                unit_meta["first_round_evidence_ids_before_acceptance_gate"] = [
                    str(x.get("atomic_id", "")).strip()
                    for x in full_unit_first_round
                    if str(x.get("atomic_id", "")).strip()
                ]
                unit_meta["accepted_first_round_seed_ids"] = [
                    str(x.get("atomic_id", "")).strip()
                    for x in accepted_unit_first_round
                    if str(x.get("atomic_id", "")).strip()
                ]
                unit_meta["refinement_seed_policy"] = "judge_accepted_only"
                for path in unit_paths:
                    path_map.setdefault(path.key(), path)
                refined_by_unit_id[unit_result["unit_id"]] = unit_refined_evidence
                unit_refinement_traces.append(unit_meta)

            final_paths = list(path_map.values())
            first_round_evidence = list(evidence)

            # Keep units independent through refinement. Resolved units retain
            # their first-round evidence; only the selected unit(s) are replaced
            # by their own deeper local ranking. Cross-unit competition starts
            # only after those independent rankings are complete.
            final_unit_results = []
            for unit_result in unit_results:
                unit_id = unit_result["unit_id"]
                local_evidence = refined_by_unit_id.get(
                    unit_id,
                    unit_result.get("evidence", []),
                )
                final_unit_results.append(
                    {
                        "unit_id": unit_id,
                        "query": unit_result.get("query", ""),
                        "keywords": list(unit_result.get("keywords", []) or []),
                        "evidence": local_evidence,
                    }
                )

            refined_evidence = self._combine_unit_evidence(
                final_unit_results,
                rescue_evidence,
                limit=self.config.refinement_context_limit,
            )

            # V4.3.1 carry-forward remains unchanged: evidence explicitly accepted
            # by the first Judge is protected while new evidence fills the missing
            # unit.
            pre_protection_ids = [
                str(x.get("atomic_id", "")).strip()
                for x in refined_evidence
                if str(x.get("atomic_id", "")).strip()
            ]
            evidence, protected_ids = self._carry_forward_accepted_evidence(
                initial,
                first_round_evidence,
                refined_evidence,
                limit=self.config.refinement_context_limit,
            )

            refinement_trace = {
                "requested_query": refinement_query,
                "missing": missing,
                **unit_selection,
                "units": unit_refinement_traces,
                "feedback_evidence_ids": self._unique_strings(
                    evidence_id
                    for meta in unit_refinement_traces
                    for evidence_id in meta.get("feedback_evidence_ids", [])
                ),
                "accepted_first_pass_evidence_ids": list(
                    initial.get("evidence_ids", []) or []
                ),
                "pre_protection_evidence_ids": pre_protection_ids,
                "protected_evidence_ids": protected_ids,
                "final_unit_evidence_ids": {
                    result["unit_id"]: [
                        str(x.get("atomic_id", "")).strip()
                        for x in result.get("evidence", [])
                        if str(x.get("atomic_id", "")).strip()
                    ]
                    for result in final_unit_results
                },
                "final_evidence_ids": [
                    str(x.get("atomic_id", "")).strip()
                    for x in evidence
                    if str(x.get("atomic_id", "")).strip()
                ],
            }

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
                retrieval_units=retrieval_units,
                prior_decision=initial,
            )

        t_final = time.perf_counter()

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
            "planner_plus_parallel_local_retrieval": round(
                t_plan_parallel - t0, 4
            ),
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
                "in_scope": bool(plan.get("in_scope", True)),
                "query_focus": plan.get("query_focus", ""),
                "search_query": plan.get("search_query", query),
                "retrieval_units": plan.get("retrieval_units", []),
                "fallback": bool(plan.get("fallback", False)),
            },
            "answer_embedding": answer_embedding_trace,
            "unit_retrieval": [
                {
                    "unit_id": result["unit_id"],
                    "query": result["query"],
                    "keywords": result["keywords"],
                    "evidence_ids": [
                        x.get("atomic_id", "")
                        for x in result.get("evidence", [])
                    ],
                    "evidence": [
                        {
                            "rank": x.get("rank", 0),
                            "atomic_id": x.get("atomic_id", ""),
                            "faq_id": x.get("faq_id", ""),
                            "question": x.get("question", ""),
                            "answer": x.get("answer", ""),
                            "channels": x.get("channels", []),
                            "retrieval_score": x.get("retrieval_score", 0.0),
                        }
                        for x in result.get("evidence", [])
                    ],
                }
                for result in unit_results
            ],
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
                        "retrieval_units": x.get("retrieval_units", []),
                        "retrieval_score": x.get("retrieval_score", 0.0),
                    }
                    for x in evidence
                ],
            },
            "timing": timing,
        }

