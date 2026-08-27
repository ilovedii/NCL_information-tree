import time

from candidate_orderer import CandidateOrderer
from context_builder import ContextBuilder
from evidence_selector import EvidenceSelector
from knowledge_loader import KnowledgeLoader
from llm_client import OllamaClient, OpenAICompatibleClient
from retriever import HybridRetriever
from router import TreeRouter
from static_knowledge import StaticKnowledgeStore
from taxonomy import TaxonomyIndex


class TreeGuidedRAG:
    """V3.1 Progressive Tree-Guided RAG.

    Key idea:
    - Taxonomy still decides the search space.
    - Local ordering only decides what is inspected first inside that space.
    - Evidence is assessed one batch at a time.
    - Sufficiency is the stopping criterion.
    - No taxonomy reclassification is performed at query time.
    - No candidate is permanently removed by local ordering.
    """

    def __init__(
        self,
        config,
        router_llm=None,
        evidence_llm=None,
        answer_llm=None,
        embedding_llm=None,
    ):
        self.config = config
        self.taxonomy = TaxonomyIndex(config.csv_path)

        ollama_client = OllamaClient(
            config.ollama_url,
            timeout=config.timeout,
        )
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
            raise ValueError(
                f"未知 LLM provider: {provider!r}；目前只支援 'ncl' 或 'ollama'"
            )

        self.router_llm = router_llm or generation_client(config.router_provider)
        self.evidence_llm = evidence_llm or generation_client(config.evidence_provider)
        self.answer_llm = answer_llm or generation_client(config.answer_provider)
        self.embedding_llm = embedding_llm or ollama_client

        self.router = TreeRouter(
            self.taxonomy,
            self.router_llm,
            config.router_model,
            think=config.router_think,
        )
        self.knowledge_loader = KnowledgeLoader(self.taxonomy)
        self.knowledge_store = StaticKnowledgeStore(
            self.taxonomy,
            cache_dir=config.static_knowledge_dir,
            include_topic=config.static_include_topic,
            exact_dedup=config.static_exact_dedup,
        )
        self.evidence_selector = EvidenceSelector(
            self.evidence_llm,
            config.evidence_model,
            think=config.evidence_think,
            batch_size=config.evidence_batch_size,
            min_supporting_without_direct=(
                config.sufficiency_min_supporting_without_direct
            ),
        )
        self.candidate_orderer = CandidateOrderer()
        self.context_builder = ContextBuilder(self.taxonomy)
        self.retriever = HybridRetriever(
            self.taxonomy,
            self.embedding_llm,
            config.embedding_model,
            use_embedding=config.use_embedding,
            batch_size=config.embedding_batch_size,
        )

    @staticmethod
    def _node_key(l1, l2=None, l3=None):
        return (str(l1 or ""), str(l2 or ""), str(l3 or ""))

    @staticmethod
    def _path_key(path):
        return TreeGuidedRAG._node_key(path.l1, path.l2, path.l3)

    @staticmethod
    def _unit_key(unit):
        source_ids = tuple(sorted(str(x).strip() for x in unit.get("source_ids", []) if str(x).strip()))
        if source_ids:
            return ("source_ids", source_ids)
        return (
            "content",
            str(unit.get("content", "")).strip(),
            str(unit.get("time_scope", "")).strip(),
        )

    def _load_path_pack(self, path, role, force_rebuild=False):
        bundle = self.knowledge_loader.load_path(path, role=role)
        return self.knowledge_store.load(
            bundle,
            force_rebuild=force_rebuild,
        )

    def _load_node_pack(self, l1, l2=None, l3=None, role="sibling", force_rebuild=False):
        bundle = self.knowledge_loader.load_node(
            l1,
            l2=l2,
            l3=l3,
            role=role,
        )
        return self.knowledge_store.load(
            bundle,
            force_rebuild=force_rebuild,
        )

    @staticmethod
    def _l2_route_score(path):
        for step in path.trace:
            if step.get("level") == "L2" and step.get("node") == path.l2:
                try:
                    return float(step.get("score", 0.0))
                except (TypeError, ValueError):
                    return 0.0
        try:
            return float(path.score)
        except (TypeError, ValueError):
            return 0.0

    def _make_state(self, pack, route_score, origin, source_path=None):
        return {
            "key": self._node_key(
                pack.get("l1"),
                pack.get("l2"),
                pack.get("l3"),
            ),
            "pack": pack,
            "remaining_units": list(pack.get("knowledge_units", [])),
            "route_score": float(route_score or 0.0),
            "origin": str(origin),
            "source_path": source_path,
            "batches": 0,
            "assessed_unique": 0,
            "opened": False,
            "family_expanded": False,
        }

    def _slice_pack(self, pack, units):
        sliced = dict(pack)
        sliced["knowledge_units"] = list(units)
        sliced["document_count"] = len(units)
        sliced["source_coverage_ratio"] = 1.0 if units else 0.0
        return sliced

    def _fallback_pack(self, query, paths, force=False):
        if not self.config.use_fallback_retrieval:
            return None

        if force:
            should_fallback = True
        elif not paths:
            should_fallback = True
        else:
            should_fallback = (
                paths[0].score < self.config.routing_confidence_threshold
            )

        if not should_fallback:
            return None

        indices = list(range(len(self.taxonomy.df)))
        query_vector = (
            self.retriever.query_embedding(query)
            if self.retriever.use_embedding
            else None
        )
        hits = self.retriever.search(
            query,
            indices,
            top_k=self.config.fallback_top_k,
            query_vector=query_vector,
        )

        units = []
        source_ids = []
        for i, hit in enumerate(hits, start=1):
            record = self.taxonomy.document_record(hit["idx"])
            atomic_id = str(record.get("atomic_id", "")).strip()
            if atomic_id:
                source_ids.append(atomic_id)
            unit = self.knowledge_store.unit_from_document(
                record,
                knowledge_id=f"F{i:03d}",
                unit_type="fallback_knowledge",
            )
            units.append(unit)

        return {
            "fingerprint": "fallback",
            "cache_hit": False,
            "static_source": "global_hybrid_fallback",
            "role": "fallback",
            "level": "GLOBAL",
            "l1": "",
            "l2": "",
            "l3": "",
            "path": "Global Hybrid Fallback",
            "document_count": len(units),
            "date_start": "",
            "date_end": "",
            "all_source_ids": source_ids,
            "knowledge_units": units,
            "coverage_note": (
                "Progressive Tree frontier 已耗盡或 Router 信心過低，因此加入"
                "全域 Hybrid Retrieval 作為最後安全網。"
            ),
            "source_coverage_ratio": 1.0 if units else 0.0,
        }

    def _faq_family_pack(self, prioritized_evidence):
        """Recover atomic siblings from the same original FAQ provenance."""
        anchor_ids = []
        seen_anchor_ids = set()
        direct_count = 0

        for item in prioritized_evidence:
            if item.get("utility") != "direct":
                continue

            direct_count += 1
            for source_id in item.get("source_ids", []):
                atomic_id = str(source_id).strip()
                if atomic_id and atomic_id not in seen_anchor_ids:
                    seen_anchor_ids.add(atomic_id)
                    anchor_ids.append(atomic_id)

            if direct_count >= self.config.max_anchor_evidence:
                break

        if not anchor_ids:
            return None

        sibling_indices = self.taxonomy.sibling_indices_for_atomic_ids(
            anchor_ids,
            max_faqs=self.config.max_anchor_faqs,
            max_siblings_per_faq=self.config.max_siblings_per_faq,
        )
        if not sibling_indices:
            return None

        existing_source_ids = {
            str(source_id).strip()
            for item in prioritized_evidence
            for source_id in item.get("source_ids", [])
            if str(source_id).strip()
        }

        units = []
        expanded_source_ids = []
        expanded_faq_ids = []
        seen_faq_ids = set()

        for idx in sibling_indices:
            record = self.taxonomy.document_record(idx)
            atomic_id = str(record.get("atomic_id", "")).strip()
            faq_id = str(record.get("faq_id", "")).strip()

            if not atomic_id or atomic_id in existing_source_ids:
                continue

            unit = self.knowledge_store.unit_from_document(
                record,
                knowledge_id=f"FAQ{len(units) + 1:03d}",
                unit_type="related_source",
            )
            units.append(unit)
            expanded_source_ids.append(atomic_id)

            if faq_id and faq_id not in seen_faq_ids:
                seen_faq_ids.add(faq_id)
                expanded_faq_ids.append(faq_id)

        if not units:
            return None

        return {
            "fingerprint": "faq_family_expansion",
            "cache_hit": False,
            "static_source": "faq_provenance",
            "role": "faq_family",
            "level": "FAQ",
            "l1": "",
            "l2": "",
            "l3": "",
            "path": "Related Original FAQ",
            "document_count": len(units),
            "date_start": "",
            "date_end": "",
            "all_source_ids": expanded_source_ids,
            "knowledge_units": units,
            "coverage_note": (
                "根據 direct evidence 的原始 faq_id，補回同一原始問答中"
                "被 atomic decomposition 拆開的 knowledge units。"
            ),
            "source_coverage_ratio": 1.0,
            "anchor_source_ids": anchor_ids,
            "expanded_faq_ids": expanded_faq_ids,
            "expanded_source_ids": expanded_source_ids,
        }

    def _expand_faq_once(self, query, prioritized_evidence, knowledge_packs):
        if not self.config.use_faq_expansion:
            return prioritized_evidence, None, 0

        faq_pack = self._faq_family_pack(prioritized_evidence)
        if faq_pack is None:
            return prioritized_evidence, None, 0

        before = len(prioritized_evidence)
        prioritized_evidence = self.evidence_selector.extend_prioritized(
            query,
            prioritized_evidence,
            [faq_pack],
        )
        added = len(prioritized_evidence) - before
        if added <= 0:
            return prioritized_evidence, None, 0

        knowledge_packs.append(faq_pack)
        return prioritized_evidence, faq_pack, added

    @staticmethod
    def _faq_trace(faq_packs, added_total):
        if not faq_packs:
            return {"triggered": False, "added_unique_evidence": 0}

        anchor_source_ids = []
        expanded_faq_ids = []
        expanded_source_ids = []
        for pack in faq_packs:
            for value in pack.get("anchor_source_ids", []):
                if value not in anchor_source_ids:
                    anchor_source_ids.append(value)
            for value in pack.get("expanded_faq_ids", []):
                if value not in expanded_faq_ids:
                    expanded_faq_ids.append(value)
            for value in pack.get("expanded_source_ids", []):
                if value not in expanded_source_ids:
                    expanded_source_ids.append(value)

        return {
            "triggered": True,
            "rounds": len(faq_packs),
            "anchor_source_ids": anchor_source_ids,
            "expanded_faq_ids": expanded_faq_ids,
            "expanded_source_ids": expanded_source_ids,
            "added_unique_evidence": added_total,
        }

    def _check_sufficiency(self, query, prioritized_evidence):
        if not self.config.use_sufficiency_check:
            return {
                "sufficient": self.evidence_selector.has_direct(prioritized_evidence),
                "query_aspects": [],
                "covered_aspects": [],
                "missing_aspects": [],
                "reason": "sufficiency check disabled; fallback to direct evidence",
                "relevant_evidence_count": len(
                    self.evidence_selector.filter_for_context(prioritized_evidence)
                ),
                "direct_evidence_count": sum(
                    1
                    for item in prioritized_evidence
                    if item.get("utility") == "direct"
                ),
                "checked_by_llm": False,
            }
        return self.evidence_selector.check_sufficiency(
            query,
            prioritized_evidence,
        )

    @staticmethod
    def _relevant_evidence_ids(evidence):
        return {
            item.get("evidence_id")
            for item in evidence or []
            if item.get("utility") in {"direct", "supporting"}
        }

    def _focus_text(self, query, sufficiency, prioritized_evidence):
        """Build local ordering focus without another API call."""
        parts = []

        for value in sufficiency.get("missing_aspects", []) if sufficiency else []:
            value = str(value).strip()
            if value:
                parts.append(value)

        # Reuse the few evidence units already judged relevant. They often expose
        # domain vocabulary absent from the user's wording (e.g. 書目性質).
        relevant = self.evidence_selector.filter_for_context(
            prioritized_evidence,
            include_background=False,
        )
        for item in relevant[:3]:
            content = str(item.get("content", "")).strip()
            if content:
                parts.append(content[:800])

        return "\n".join(parts)

    def _register_pack(self, pack, knowledge_packs, opened_pack_keys):
        key = self._node_key(
            pack.get("l1"),
            pack.get("l2"),
            pack.get("l3"),
        )
        # FAQ/fallback use synthetic empty taxonomy keys; distinguish by path.
        if not any(key):
            key = ("runtime", pack.get("role", ""), pack.get("path", ""))

        if key not in opened_pack_keys:
            opened_pack_keys.add(key)
            knowledge_packs.append(pack)

    def _add_sibling_states(
        self,
        source_state,
        states,
        state_by_key,
        routed_path_by_key,
        force_rebuild=False,
    ):
        if not self.config.use_sibling_l3_expansion:
            return []

        pack = source_state["pack"]
        l1 = pack.get("l1")
        l2 = pack.get("l2")
        l3 = pack.get("l3")
        if not l1 or not l2 or not l3:
            return []

        added = []
        parent_score = self._l2_route_score(source_state["source_path"]) if source_state.get("source_path") is not None else source_state.get("route_score", 0.0)
        inherited_score = float(parent_score) * float(self.config.sibling_route_discount)

        for sibling_l3 in self.taxonomy.l3_nodes(l1, l2):
            if sibling_l3 == l3:
                continue
            key = self._node_key(l1, l2, sibling_l3)
            if key in state_by_key:
                continue

            routed_path = routed_path_by_key.get(key)
            if routed_path is not None:
                role = "alternative"
                route_score = float(routed_path.score)
                source_path = routed_path
                origin = "routed_alternative"
            else:
                role = "sibling"
                route_score = inherited_score
                source_path = None
                origin = "sibling_l3"

            sibling_pack = self._load_node_pack(
                l1,
                l2=l2,
                l3=sibling_l3,
                role=role,
                force_rebuild=force_rebuild,
            )
            state = self._make_state(
                sibling_pack,
                route_score=route_score,
                origin=origin,
                source_path=source_path,
            )
            states.append(state)
            state_by_key[key] = state
            added.append(state)

        source_state["family_expanded"] = True
        return added

    def _frontier_priority(self, query, states, focus_text):
        active = [state for state in states if state.get("remaining_units")]
        if not active:
            return []

        pseudo_packs = []
        for state in active:
            pseudo_pack = dict(state["pack"])
            pseudo_pack["knowledge_units"] = state["remaining_units"]
            pseudo_packs.append(pseudo_pack)
        local_scores = self.candidate_orderer.score_packs_global(
            query,
            pseudo_packs,
            extra_text=focus_text,
        )

        lo = min(local_scores)
        hi = max(local_scores)
        span = hi - lo

        ranked = []
        for state, local in zip(active, local_scores):
            local_norm = (local - lo) / span if span > 1e-9 else 0.0
            route_score = max(0.0, min(1.0, float(state.get("route_score", 0.0))))
            priority = (
                float(self.config.frontier_local_weight) * local_norm
                + float(self.config.frontier_route_weight) * route_score
            )
            ranked.append(
                {
                    "state": state,
                    "priority": float(priority),
                    "local_score": float(local),
                    "local_norm": float(local_norm),
                    "route_score": route_score,
                }
            )

        ranked.sort(
            key=lambda item: (
                -item["priority"],
                -item["route_score"],
                item["state"]["pack"].get("path", ""),
            )
        )
        return ranked

    def _take_ordered_batch(self, query, state, focus_text):
        remaining = list(state.get("remaining_units", []))
        if not remaining:
            return []

        if self.config.use_local_candidate_ordering:
            ordered = self.candidate_orderer.order_units(
                query,
                remaining,
                extra_text=focus_text,
            )
        else:
            ordered = remaining

        if self.config.use_progressive_search:
            batch_size = max(1, int(self.config.progressive_batch_size))
        else:
            batch_size = max(1, len(ordered))
        batch = ordered[:batch_size]
        selected_keys = {self._unit_key(unit) for unit in batch}
        state["remaining_units"] = [
            unit
            for unit in remaining
            if self._unit_key(unit) not in selected_keys
        ]
        return batch

    def _assess_one_progressive_batch(
        self,
        query,
        state,
        prioritized_evidence,
        sufficiency,
        knowledge_packs,
        opened_pack_keys,
        faq_packs,
        progressive_history,
    ):
        focus_text = self._focus_text(query, sufficiency, prioritized_evidence)
        batch = self._take_ordered_batch(query, state, focus_text)
        if not batch:
            return prioritized_evidence, sufficiency, 0

        self._register_pack(state["pack"], knowledge_packs, opened_pack_keys)
        state["opened"] = True
        state["batches"] += 1

        before_ids = {
            item.get("evidence_id") for item in prioritized_evidence or []
        }
        stats_before = self.evidence_selector.stats()
        batch_t0 = time.perf_counter()

        batch_pack = self._slice_pack(state["pack"], batch)
        prioritized_evidence = self.evidence_selector.extend_prioritized(
            query,
            prioritized_evidence,
            [batch_pack],
        )

        # Precise FAQ provenance recovery is incremental and can stop an
        # enumeration-type query before broader node expansion.
        faq_added = 0
        faq_pack = None
        prioritized_evidence, faq_pack, faq_added = self._expand_faq_once(
            query,
            prioritized_evidence,
            knowledge_packs,
        )
        if faq_pack is not None:
            faq_packs.append(faq_pack)

        new_evidence = [
            item
            for item in prioritized_evidence
            if item.get("evidence_id") not in before_ids
        ]
        state_added = sum(
            1
            for item in new_evidence
            if item.get("role") == state["pack"].get("role")
            and item.get("path") == state["pack"].get("path")
        )
        state["assessed_unique"] += state_added

        # Re-run sufficiency only when relevant evidence changed. If a batch
        # contributes only background/low_relevance, the previous judgement is
        # still valid and no extra API call is needed.
        new_relevant = [
            item
            for item in new_evidence
            if item.get("utility") in {"direct", "supporting"}
        ]
        if sufficiency is None or new_relevant:
            sufficiency = self._check_sufficiency(
                query,
                prioritized_evidence,
            )
            sufficiency_rechecked = True
        else:
            sufficiency_rechecked = False

        stats_after = self.evidence_selector.stats()
        elapsed = time.perf_counter() - batch_t0

        progressive_history.append(
            {
                "step": len(progressive_history) + 1,
                "origin": state.get("origin"),
                "role": state["pack"].get("role"),
                "path": state["pack"].get("path"),
                "batch_index_for_node": state.get("batches", 0),
                "batch_candidate_count": len(batch),
                "node_remaining_after_batch": len(state.get("remaining_units", [])),
                "new_unique_evidence": len(new_evidence),
                "new_relevant_evidence": len(new_relevant),
                "faq_added_unique_evidence": faq_added,
                "sufficiency_rechecked": sufficiency_rechecked,
                "sufficient_after_batch": bool(
                    sufficiency.get("sufficient", False)
                ) if sufficiency else False,
                "missing_aspects_after_batch": list(
                    sufficiency.get("missing_aspects", [])
                ) if sufficiency else [],
                "evidence_api_calls_added": (
                    stats_after["evidence_batch_api_calls"]
                    - stats_before["evidence_batch_api_calls"]
                ),
                "sufficiency_api_calls_added": (
                    stats_after["sufficiency_api_calls"]
                    - stats_before["sufficiency_api_calls"]
                ),
                "elapsed_seconds": round(elapsed, 4),
            }
        )

        return prioritized_evidence, sufficiency, faq_added

    def _answer(self, query, context_pack, sufficiency):
        system = """你是國家圖書館領域的最終問答模型。請只依提供的正式 taxonomy、分類路徑與最終 Evidence Context 回答。最終 Context 已經過 relevance filtering，預設只保留 direct 與 supporting knowledge；原始 corpus 中的 background / low_relevance 並未被刪除，只是不送入本次回答。若 evidence sufficiency 顯示仍有 missing_aspects，必須明確指出缺少哪些知識，不得自行補齊或虛構。若存在版本差異、例外或衝突，應依 supporting evidence 說明。若問題明確指定 MARC 21，不能把 CMARC 專屬的欄位操作規則直接轉寫成 MARC 21 規則，除非 evidence 明確建立兩者對照；可使用跨 evidence 組合出的基本定義差異，但不可擴張成未被支持的操作差異。回答時保留支持結論的 evidence_id / atomic source。"""

        missing = sufficiency.get("missing_aspects", []) or []
        sufficiency_text = (
            f"sufficient={bool(sufficiency.get('sufficient', False))}\n"
            f"query_aspects={sufficiency.get('query_aspects', [])}\n"
            f"missing_aspects={missing}\n"
            f"reason={sufficiency.get('reason', '')}"
        )

        user = f"""請回答以下問題：
{query}

Evidence Sufficiency：
{sufficiency_text}

以下是 Tree-Guided Progressive RAG 建立的最終 Knowledge Pack：
{context_pack}

請使用以下格式：
分類路徑：
回答：
判斷依據：
證據 ID：
知識充分性：直接支持 / 可合理推論 / 證據不足
"""
        return self.answer_llm.chat_text(
            self.config.answer_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            think=self.config.answer_think,
        )

    def build_static_knowledge_packs(
        self,
        force_rebuild=False,
        include_l2_parents=True,
    ):
        return self.knowledge_store.build_all(
            self.knowledge_loader,
            include_l2_parents=include_l2_parents,
            force_rebuild=force_rebuild,
        )

    def build_node_summaries(self, force_rebuild=False, include_l2_parents=True):
        return self.build_static_knowledge_packs(
            force_rebuild=force_rebuild,
            include_l2_parents=include_l2_parents,
        )

    def run(
        self,
        query,
        l1_beam=None,
        l2_beam=None,
        final_beam=None,
        force_rebuild_summaries=False,
        force_rebuild_knowledge=None,
        final_context_unit_limit=None,
    ):
        total_t0 = time.perf_counter()
        self.evidence_selector.reset_stats()

        query = str(query).strip()
        if not query:
            raise ValueError("query 不可為空")

        l1_beam = self.config.l1_beam if l1_beam is None else l1_beam
        l2_beam = self.config.l2_beam if l2_beam is None else l2_beam
        final_beam = self.config.final_beam if final_beam is None else final_beam
        if final_context_unit_limit is None:
            final_context_unit_limit = self.config.final_context_unit_limit
        if force_rebuild_knowledge is None:
            force_rebuild_knowledge = bool(force_rebuild_summaries)

        timings = {}

        # 1) Tree-first routing: unchanged. No reclassification occurs here.
        t0 = time.perf_counter()
        paths = self.router.route_tree(
            query,
            l1_beam=l1_beam,
            l2_global_beam=l2_beam,
            final_beam=final_beam,
        )
        timings["routing"] = time.perf_counter() - t0

        if not paths:
            raise RuntimeError("Tree Router 未回傳任何候選路徑")

        # 2) Build routed node states locally. Only the primary gets the first
        # LLM batch; alternatives remain in the frontier until needed.
        t0 = time.perf_counter()
        states = []
        state_by_key = {}
        routed_path_by_key = {}

        for rank, path in enumerate(paths):
            key = self._path_key(path)
            routed_path_by_key[key] = path
            if key in state_by_key:
                continue
            role = "primary" if rank == 0 else "alternative"
            origin = "primary" if rank == 0 else "routed_alternative"
            pack = self._load_path_pack(
                path,
                role=role,
                force_rebuild=force_rebuild_knowledge,
            )
            state = self._make_state(
                pack,
                route_score=path.score,
                origin=origin,
                source_path=path,
            )
            states.append(state)
            state_by_key[key] = state

        timings["frontier_prepare"] = time.perf_counter() - t0

        primary_state = state_by_key[self._path_key(paths[0])]
        knowledge_packs = []
        opened_pack_keys = set()
        prioritized_evidence = []
        sufficiency = None
        sufficiency_initial = None
        faq_packs = []
        faq_added_total = 0
        progressive_history = []
        sibling_expansion_paths = []

        # 3) Force exactly one Tree-primary batch first.
        t0 = time.perf_counter()
        prioritized_evidence, sufficiency, faq_added = self._assess_one_progressive_batch(
            query,
            primary_state,
            prioritized_evidence,
            sufficiency,
            knowledge_packs,
            opened_pack_keys,
            faq_packs,
            progressive_history,
        )
        faq_added_total += faq_added
        if sufficiency is None:
            sufficiency = self._check_sufficiency(query, prioritized_evidence)
        sufficiency_initial = sufficiency

        # Open the primary L3 family only after the first primary batch proves
        # the current evidence is still insufficient. This reuses existing L3
        # labels; it does not reclassify any record.
        if (
            not sufficiency.get("sufficient", False)
            and primary_state["pack"].get("l3")
            and not primary_state.get("family_expanded")
        ):
            added_states = self._add_sibling_states(
                primary_state,
                states,
                state_by_key,
                routed_path_by_key,
                force_rebuild=force_rebuild_knowledge,
            )
            sibling_expansion_paths.extend(
                state["pack"].get("path", "") for state in added_states
            )
        timings["primary_probe"] = time.perf_counter() - t0

        # 4) Best-first progressive frontier. Every remaining unit stays reachable.
        t0 = time.perf_counter()
        while not sufficiency.get("sufficient", False):
            focus = self._focus_text(query, sufficiency, prioritized_evidence)
            ranked_frontier = self._frontier_priority(
                query,
                states,
                focus,
            )
            if not ranked_frontier:
                break

            chosen = ranked_frontier[0]
            state = chosen["state"]

            prioritized_evidence, sufficiency, faq_added = self._assess_one_progressive_batch(
                query,
                state,
                prioritized_evidence,
                sufficiency,
                knowledge_packs,
                opened_pack_keys,
                faq_packs,
                progressive_history,
            )
            faq_added_total += faq_added

            if sufficiency.get("sufficient", False):
                break

            # Once a routed/sibling L3 actually receives a batch, its sibling
            # family can be added lazily. Duplicate node keys are suppressed.
            if (
                state["pack"].get("l3")
                and not state.get("family_expanded")
            ):
                added_states = self._add_sibling_states(
                    state,
                    states,
                    state_by_key,
                    routed_path_by_key,
                    force_rebuild=force_rebuild_knowledge,
                )
                sibling_expansion_paths.extend(
                    child["pack"].get("path", "") for child in added_states
                )

        timings["progressive_tree_search"] = time.perf_counter() - t0

        # 5) If the Tree frontier is exhausted and evidence is still insufficient,
        # invoke the existing global fallback once as a final safety net.
        fallback_pack = None
        fallback_added = 0
        t0 = time.perf_counter()
        if not sufficiency.get("sufficient", False):
            force_fallback = bool(self.config.fallback_on_tree_exhausted)
            fallback_pack = self._fallback_pack(
                query,
                paths,
                force=force_fallback,
            )
            if fallback_pack is not None:
                self._register_pack(
                    fallback_pack,
                    knowledge_packs,
                    opened_pack_keys,
                )
                before_ids = {
                    item.get("evidence_id") for item in prioritized_evidence
                }
                prioritized_evidence = self.evidence_selector.extend_prioritized(
                    query,
                    prioritized_evidence,
                    [fallback_pack],
                )
                fallback_added = sum(
                    1
                    for item in prioritized_evidence
                    if item.get("evidence_id") not in before_ids
                )
                if fallback_added:
                    sufficiency = self._check_sufficiency(
                        query,
                        prioritized_evidence,
                    )
        timings["global_fallback"] = time.perf_counter() - t0

        all_prioritized_count = len(prioritized_evidence)

        # 6) Final answer receives only direct/supporting evidence.
        final_context_evidence = self.evidence_selector.filter_for_context(
            prioritized_evidence,
            include_background=self.config.final_include_background,
        )
        final_context_before_limit = len(final_context_evidence)
        if final_context_unit_limit and final_context_unit_limit > 0:
            final_context_evidence = final_context_evidence[:final_context_unit_limit]

        t0 = time.perf_counter()
        context_pack = self.context_builder.build(
            query,
            paths,
            knowledge_packs,
            final_context_evidence,
        )
        answer = self._answer(query, context_pack, sufficiency)
        timings["final_answer"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - total_t0

        # Node-level assessment coverage: storage coverage remains 1.0, while
        # assessed coverage shows how much of that node the LLM actually needed.
        assessed_by_node = {}
        state_meta = {}
        for state in states:
            path = state["pack"].get("path", "")
            assessed_by_node[path] = state.get("assessed_unique", 0)
            state_meta[path] = state

        node_knowledge = []
        for pack in knowledge_packs:
            path = pack.get("path", "")
            candidate_count = len(pack.get("knowledge_units", []))
            assessed_count = assessed_by_node.get(path, candidate_count)
            if pack.get("role") in {"faq_family", "fallback"}:
                assessed_count = candidate_count
            node_knowledge.append(
                {
                    "role": pack.get("role"),
                    "path": path,
                    "static_source": pack.get("static_source", ""),
                    "document_count": pack.get("document_count", 0),
                    "date_start": pack.get("date_start", ""),
                    "date_end": pack.get("date_end", ""),
                    "knowledge_unit_count": candidate_count,
                    "source_coverage_ratio": pack.get(
                        "source_coverage_ratio", 0.0
                    ),
                    "coverage_note": pack.get("coverage_note", ""),
                    "cache_hit": pack.get("cache_hit", False),
                    "assessed_unit_count": assessed_count,
                    "assessment_coverage_ratio": (
                        assessed_count / candidate_count
                        if candidate_count
                        else 0.0
                    ),
                }
            )

        frontier_remaining = sum(
            len(state.get("remaining_units", [])) for state in states
        )
        frontier_total_candidates = sum(
            len(state["pack"].get("knowledge_units", [])) for state in states
        )

        return {
            "query": query,
            "paths": [
                {
                    "path": path.display(),
                    "score": path.score,
                    "trace": path.trace,
                }
                for path in paths
            ],
            "node_knowledge": node_knowledge,
            "node_summaries": node_knowledge,
            "evidence": prioritized_evidence,
            "final_context_evidence": final_context_evidence,
            "sufficiency": {
                "after_primary_probe": sufficiency_initial,
                "final": sufficiency,
            },
            "parent_fallback": {
                "triggered": bool(sibling_expansion_paths),
                "reason": (
                    "replaced by progressive sibling-L3 expansion; no whole-L2 parent pack"
                    if sibling_expansion_paths
                    else "not needed"
                ),
                "added_unique_evidence": sum(
                    state.get("assessed_unique", 0)
                    for state in states
                    if state.get("origin") == "sibling_l3"
                ),
                "path": "",
                "route_uncertain": False,
            },
            "sibling_expansion": {
                "triggered": bool(sibling_expansion_paths),
                "candidate_paths_added": list(dict.fromkeys(sibling_expansion_paths)),
                "opened_paths": [
                    state["pack"].get("path", "")
                    for state in states
                    if state.get("origin") == "sibling_l3"
                    and state.get("opened")
                ],
            },
            "faq_expansion": self._faq_trace(faq_packs, faq_added_total),
            "progressive_search": {
                "batch_size": int(self.config.progressive_batch_size),
                "history": progressive_history,
                "frontier_node_count": len(states),
                "frontier_total_candidate_units": frontier_total_candidates,
                "frontier_remaining_unassessed_units": frontier_remaining,
                "stopped_early": bool(
                    sufficiency.get("sufficient", False)
                    and frontier_remaining > 0
                ),
                "local_ordering_is_hard_filter": False,
            },
            "evidence_evaluation": {
                "final_unique_evidence": all_prioritized_count,
                "utility_counts": self.evidence_selector.utility_counts(
                    prioritized_evidence
                ),
                "final_context_before_limit": final_context_before_limit,
                "final_context_after_limit": len(final_context_evidence),
                "fallback_added_unique_evidence": fallback_added,
                **self.evidence_selector.stats(),
            },
            "evidence_total_before_context_limit": all_prioritized_count,
            "timing_seconds": {k: round(v, 4) for k, v in timings.items()},
            "context_pack": context_pack,
            "answer": answer,
        }
