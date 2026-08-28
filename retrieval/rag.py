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
    """V3.3 Evidence-Safe Progressive Tree-Guided RAG with Service Mode.

    Core principles:
    - Taxonomy decides the search space.
    - BM25-like local ordering only decides what is inspected first.
    - Evidence Selector verifies relevance.
    - Sufficiency decides whether search can stop.
    - Service Mode bounds online LLM work and disables online embeddings.
    - No taxonomy reclassification is performed at query time.
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

        # This service-oriented rag.py defaults to Service Mode even when an
        # older config.py does not yet contain a service_mode field.
        self.service_mode = bool(getattr(config, "service_mode", True))

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
        self.evidence_llm = (
            evidence_llm or generation_client(config.evidence_provider)
        )
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

        # Service Mode disables Evidence-LLM thinking to reduce latency.
        evidence_think = (
            False if self.service_mode else bool(config.evidence_think)
        )
        self.evidence_selector = EvidenceSelector(
            self.evidence_llm,
            config.evidence_model,
            think=evidence_think,
            batch_size=config.evidence_batch_size,
            min_supporting_without_direct=(
                config.sufficiency_min_supporting_without_direct
            ),
            enable_conflict_analysis=bool(
                getattr(config, "enable_conflict_analysis", True)
            ),
        )

        self.candidate_orderer = CandidateOrderer()
        self.context_builder = ContextBuilder(self.taxonomy)

        # Critical for Service Mode:
        # do NOT build/query the 5914-item embedding index online.
        effective_use_embedding = (
            bool(config.use_embedding) and not self.service_mode
        )
        self.retriever = HybridRetriever(
            self.taxonomy,
            self.embedding_llm,
            config.embedding_model,
            use_embedding=effective_use_embedding,
            batch_size=config.embedding_batch_size,
        )

    # ------------------------------------------------------------------
    # Service helpers
    # ------------------------------------------------------------------

    def _service_int(self, name, default, minimum=0):
        value = getattr(self.config, name, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = int(default)
        return max(minimum, value)

    def _service_float(self, name, default, minimum=0.0):
        value = getattr(self.config, name, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(default)
        return max(minimum, value)

    def _progressive_batch_size(self):
        configured = max(
            1,
            int(getattr(self.config, "progressive_batch_size", 40)),
        )
        if not self.service_mode:
            return configured

        # Works even with the older config.py where service_batch_size
        # does not exist.
        service_batch = self._service_int(
            "service_batch_size",
            8,
            minimum=1,
        )
        return min(configured, service_batch)

    # ------------------------------------------------------------------
    # Taxonomy / pack helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_key(l1, l2=None, l3=None):
        return (str(l1 or ""), str(l2 or ""), str(l3 or ""))

    @staticmethod
    def _path_key(path):
        return TreeGuidedRAG._node_key(path.l1, path.l2, path.l3)

    @staticmethod
    def _unit_key(unit):
        source_ids = tuple(
            sorted(
                str(x).strip()
                for x in unit.get("source_ids", [])
                if str(x).strip()
            )
        )
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

    def _load_node_pack(
        self,
        l1,
        l2=None,
        l3=None,
        role="sibling",
        force_rebuild=False,
    ):
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

    # ------------------------------------------------------------------
    # Global fallback
    # ------------------------------------------------------------------

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

        # In Service Mode self.retriever.use_embedding is False, so this
        # becomes a pure BM25 fallback and never builds the embedding index.
        query_vector = (
            self.retriever.query_embedding(query)
            if self.retriever.use_embedding
            else None
        )

        top_k = int(self.config.fallback_top_k)
        if self.service_mode:
            top_k = min(
                top_k,
                self._service_int(
                    "service_fallback_top_k",
                    8,
                    minimum=1,
                ),
            )

        hits = self.retriever.search(
            query,
            indices,
            top_k=top_k,
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

        fallback_name = (
            "Global Hybrid Fallback"
            if self.retriever.use_embedding
            else "Global BM25 Fallback"
        )
        fallback_source = (
            "global_hybrid_fallback"
            if self.retriever.use_embedding
            else "global_bm25_fallback"
        )

        return {
            "fingerprint": "fallback",
            "cache_hit": False,
            "static_source": fallback_source,
            "role": "fallback",
            "level": "GLOBAL",
            "l1": "",
            "l2": "",
            "l3": "",
            "path": fallback_name,
            "document_count": len(units),
            "date_start": "",
            "date_end": "",
            "all_source_ids": source_ids,
            "knowledge_units": units,
            "coverage_note": (
                "Progressive Tree evidence 仍不足，因此加入全域 BM25 Retrieval"
                " 作為服務模式最後安全網。"
                if not self.retriever.use_embedding
                else
                "Progressive Tree evidence 仍不足，因此加入全域 Hybrid Retrieval"
                " 作為最後安全網。"
            ),
            "source_coverage_ratio": 1.0 if units else 0.0,
        }

    # ------------------------------------------------------------------
    # FAQ provenance expansion
    # ------------------------------------------------------------------

    def _faq_family_pack(self, prioritized_evidence):
        """Recover atomic siblings from the same original FAQ provenance."""
        anchor_ids = []
        seen_anchor_ids = set()
        direct_count = 0

        max_anchor_evidence = int(self.config.max_anchor_evidence)
        max_anchor_faqs = int(self.config.max_anchor_faqs)
        max_siblings_per_faq = int(self.config.max_siblings_per_faq)

        if self.service_mode:
            max_anchor_evidence = min(max_anchor_evidence, 3)
            max_anchor_faqs = min(max_anchor_faqs, 2)
            max_siblings_per_faq = min(max_siblings_per_faq, 8)

        for item in prioritized_evidence:
            if item.get("utility") != "direct":
                continue

            direct_count += 1
            for source_id in item.get("source_ids", []):
                atomic_id = str(source_id).strip()
                if atomic_id and atomic_id not in seen_anchor_ids:
                    seen_anchor_ids.add(atomic_id)
                    anchor_ids.append(atomic_id)

            if direct_count >= max_anchor_evidence:
                break

        if not anchor_ids:
            return None

        sibling_indices = self.taxonomy.sibling_indices_for_atomic_ids(
            anchor_ids,
            max_faqs=max_anchor_faqs,
            max_siblings_per_faq=max_siblings_per_faq,
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

    def _expand_faq_once(
        self,
        query,
        prioritized_evidence,
        knowledge_packs,
    ):
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
            return {
                "triggered": False,
                "added_unique_evidence": 0,
            }

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

    # ------------------------------------------------------------------
    # Sufficiency / ordering
    # ------------------------------------------------------------------

    def _check_sufficiency(self, query, prioritized_evidence):
        if not self.config.use_sufficiency_check:
            return {
                "sufficient": self.evidence_selector.has_direct(
                    prioritized_evidence
                ),
                "query_aspects": [],
                "covered_aspects": [],
                "missing_aspects": [],
                "reason": (
                    "sufficiency check disabled; fallback to direct evidence"
                ),
                "relevant_evidence_count": len(
                    self.evidence_selector.filter_for_context(
                        prioritized_evidence
                    )
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

        for value in (
            sufficiency.get("missing_aspects", [])
            if sufficiency
            else []
        ):
            value = str(value).strip()
            if value:
                parts.append(value)

        relevant = self.evidence_selector.filter_for_context(
            prioritized_evidence,
            include_background=False,
        )

        for item in relevant[:3]:
            content = str(item.get("content", "")).strip()
            if content:
                parts.append(content[:800])

        return "\n".join(parts)

    def _register_pack(
        self,
        pack,
        knowledge_packs,
        opened_pack_keys,
    ):
        key = self._node_key(
            pack.get("l1"),
            pack.get("l2"),
            pack.get("l3"),
        )

        # FAQ/fallback use synthetic empty taxonomy keys; distinguish by path.
        if not any(key):
            key = (
                "runtime",
                pack.get("role", ""),
                pack.get("path", ""),
            )

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

        parent_score = (
            self._l2_route_score(source_state["source_path"])
            if source_state.get("source_path") is not None
            else source_state.get("route_score", 0.0)
        )

        inherited_score = (
            float(parent_score)
            * float(self.config.sibling_route_discount)
        )

        for sibling_l3 in self.taxonomy.l3_nodes(l1, l2):
            if sibling_l3 == l3:
                continue

            key = self._node_key(l1, l2, sibling_l3)

            # V3.4.1: merge duplicate routing/sibling identity instead of
            # silently skipping it.
            #
            # A node can already exist because the Router returned it as a
            # lower-ranked alternative, while it is ALSO a same-parent L3
            # sibling of the current node. In that case, keeping only the
            # low Router score loses the hierarchical recovery signal and can
            # starve the sibling under the Service Mode step budget.
            if key in state_by_key:
                existing_state = state_by_key[key]
                existing_route_score = float(
                    existing_state.get("route_score", 0.0)
                )

                # Promote only when the inherited sibling score is stronger.
                # This preserves a genuinely strong Router score when one
                # already exists, while restoring same-parent recovery for
                # cases such as T01 Leader/07.
                if (
                    existing_state.get("origin") == "routed_alternative"
                    and inherited_score > existing_route_score
                ):
                    existing_state["route_score"] = float(inherited_score)
                    existing_state["origin"] = "sibling_l3"
                    existing_state["sibling_promoted_from_routed"] = True
                    added.append(existing_state)

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
        """Rank the next search node without turning BM25 into a hard retriever.

        V3.3 policy:
        - routed alternatives: Router score is authoritative for cross-branch order;
        - same-parent sibling L3: preserve local hierarchical recovery by combining
          inherited parent-route score with a bounded local BM25 signal;
        - primary/retry states: retain the generic route/local blend.

        This preserves T01 Leader/07 sibling recovery while fixing U02/U03, where
        a lower-scored branch could displace Router #2 because node-local BM25 was
        weighted too heavily.
        """
        active = [
            state
            for state in states
            if state.get("remaining_units")
        ]
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
            local_norm = (
                (local - lo) / span
                if span > 1e-9
                else 0.0
            )
            route_score = max(
                0.0,
                min(1.0, float(state.get("route_score", 0.0))),
            )
            origin = str(state.get("origin", ""))

            if origin == "routed_alternative":
                # Across routed branches, preserve the LLM Router ranking.
                priority = route_score
                policy = "router_score"
            elif origin == "sibling_l3":
                # Same-parent L3 recovery is the one locality exception. The
                # sibling inherits a discounted parent score and needs local
                # evidence signal to outrank a strong routed alternative.
                priority = (
                    float(getattr(self.config, "sibling_route_weight", 0.70))
                    * route_score
                    + float(getattr(self.config, "sibling_local_weight", 0.30))
                    * local_norm
                )
                policy = "same_parent_sibling_hybrid"
            else:
                priority = (
                    float(self.config.frontier_route_weight) * route_score
                    + float(self.config.frontier_local_weight) * local_norm
                )
                policy = "generic_hybrid"

            ranked.append(
                {
                    "state": state,
                    "priority": float(priority),
                    "local_score": float(local),
                    "local_norm": float(local_norm),
                    "route_score": route_score,
                    "ranking_policy": policy,
                }
            )

        ranked.sort(
            key=lambda item: (
                -item["priority"],
                -item["route_score"],
                item["state"]["pack"].get("path", ""),
            )
        )

        # Guard against weak locality bias. A same-parent sibling may override
        # Router #2 only when its hybrid score wins by a meaningful margin.
        # This protects T01 while preventing a near-tie sibling from displacing
        # a clearly ranked routed alternative in U02/U03-like queries.
        if ranked and ranked[0]["state"].get("origin") == "sibling_l3":
            routed_items = [
                item
                for item in ranked
                if item["state"].get("origin") == "routed_alternative"
            ]
            if routed_items:
                best_routed = max(routed_items, key=lambda item: item["priority"])
                margin = float(
                    getattr(self.config, "sibling_override_margin", 0.05)
                )
                if ranked[0]["priority"] < best_routed["priority"] + margin:
                    ranked.remove(best_routed)
                    best_routed = dict(best_routed)
                    best_routed["ranking_policy"] = "router_score_margin_guard"
                    ranked.insert(0, best_routed)

        return ranked

    def _take_ordered_batch(
        self,
        query,
        state,
        focus_text,
    ):
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
            batch_size = self._progressive_batch_size()
        else:
            batch_size = max(1, len(ordered))

        batch = ordered[:batch_size]

        selected_keys = {
            self._unit_key(unit)
            for unit in batch
        }

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
        """Assess one node batch, check sufficiency, then expand FAQ only if needed.

        V3.2 expanded FAQ before asking whether the newly found direct evidence was
        already sufficient. V3.3 reverses that order. This removes avoidable FAQ
        Evidence-LLM calls on simple/direct hits, while preserving FAQ fragmentation
        recovery when a direct anchor exists but the answer is still incomplete.
        """
        focus_text = self._focus_text(
            query,
            sufficiency,
            prioritized_evidence,
        )
        batch = self._take_ordered_batch(query, state, focus_text)
        if not batch:
            return prioritized_evidence, sufficiency, 0

        self._register_pack(state["pack"], knowledge_packs, opened_pack_keys)
        state["opened"] = True
        state["batches"] += 1

        before_ids = {
            item.get("evidence_id")
            for item in prioritized_evidence or []
        }
        before_missing = set(
            str(x).strip()
            for x in (sufficiency or {}).get("missing_aspects", [])
            if str(x).strip()
        )
        stats_before = self.evidence_selector.stats()
        batch_t0 = time.perf_counter()

        batch_pack = self._slice_pack(state["pack"], batch)
        prioritized_evidence = self.evidence_selector.extend_prioritized(
            query,
            prioritized_evidence,
            [batch_pack],
        )

        batch_new = [
            item
            for item in prioritized_evidence
            if item.get("evidence_id") not in before_ids
        ]
        batch_relevant = [
            item
            for item in batch_new
            if item.get("utility") in {"direct", "supporting"}
        ]

        # Sufficiency comes BEFORE FAQ expansion.
        if sufficiency is None or batch_relevant:
            sufficiency = self._check_sufficiency(query, prioritized_evidence)
            sufficiency_rechecked = True
        else:
            sufficiency_rechecked = False

        # Only when still insufficient and we have a direct anchor do we pay for
        # provenance recovery. This preserves T03 MarcEdit fragmentation recovery.
        faq_added = 0
        faq_pack = None
        allow_faq_round = (not self.service_mode or not faq_packs)
        if (
            allow_faq_round
            and not bool((sufficiency or {}).get("sufficient", False))
            and self.evidence_selector.has_direct(prioritized_evidence)
        ):
            (
                prioritized_evidence,
                faq_pack,
                faq_added,
            ) = self._expand_faq_once(
                query,
                prioritized_evidence,
                knowledge_packs,
            )
            if faq_pack is not None:
                faq_packs.append(faq_pack)
            if faq_added > 0:
                sufficiency = self._check_sufficiency(query, prioritized_evidence)
                sufficiency_rechecked = True

        new_evidence = [
            item
            for item in prioritized_evidence
            if item.get("evidence_id") not in before_ids
        ]
        state_added = sum(
            1
            for item in new_evidence
            if (
                item.get("role") == state["pack"].get("role")
                and item.get("path") == state["pack"].get("path")
            )
        )
        state["assessed_unique"] += state_added

        new_relevant = [
            item
            for item in new_evidence
            if item.get("utility") in {"direct", "supporting"}
        ]
        new_direct = [
            item
            for item in new_evidence
            if item.get("utility") == "direct"
        ]
        after_missing = set(
            str(x).strip()
            for x in (sufficiency or {}).get("missing_aspects", [])
            if str(x).strip()
        )
        missing_aspects_reduced = bool(before_missing - after_missing)

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
                "new_direct_evidence": len(new_direct),
                "missing_aspects_reduced": missing_aspects_reduced,
                "made_progress": bool(new_direct or missing_aspects_reduced),
                "faq_added_unique_evidence": faq_added,
                "sufficiency_rechecked": sufficiency_rechecked,
                "sufficient_after_batch": bool(
                    (sufficiency or {}).get("sufficient", False)
                ),
                "missing_aspects_after_batch": list(
                    (sufficiency or {}).get("missing_aspects", [])
                ),
                "frontier_priority": state.get("last_frontier_priority"),
                "frontier_route_score": state.get("last_frontier_route_score"),
                "frontier_local_norm": state.get("last_frontier_local_norm"),
                "frontier_ranking_policy": state.get("last_frontier_policy", "primary"),
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

    # ------------------------------------------------------------------
    # Final answer
    # ------------------------------------------------------------------

    def _answer_policy(self, sufficiency):
        rule_sensitive = bool(sufficiency.get("rule_sensitive", False))
        conflict = sufficiency.get("conflict_resolution", {}) or {}
        conflict_status = str(conflict.get("status", "none"))
        retrieval_sufficient = bool(sufficiency.get("sufficient", False))

        allow_knowledge = bool(
            getattr(self.config, "allow_knowledge_assisted_answer", True)
        )
        if (
            rule_sensitive
            and bool(
                getattr(
                    self.config,
                    "block_knowledge_assist_for_rule_sensitive",
                    True,
                )
            )
        ):
            allow_knowledge = False
        if conflict_status == "unresolved":
            allow_knowledge = False

        if retrieval_sufficient:
            mode = (
                "evidence_compositional"
                if sufficiency.get("coverage_mode") in {"evidence_compositional", "compositional"}
                else "evidence_grounded"
            )
        elif conflict_status == "unresolved":
            mode = "librarian_review_conflict"
        elif rule_sensitive:
            mode = "abstain_rule_sensitive"
        elif allow_knowledge:
            mode = "knowledge_assisted_allowed"
        else:
            mode = "abstain_insufficient"

        return {
            "mode": mode,
            "retrieval_sufficient": retrieval_sufficient,
            "rule_sensitive": rule_sensitive,
            "knowledge_assist_allowed": allow_knowledge,
            "coverage_mode": sufficiency.get("coverage_mode", "insufficient"),
            "evidence_relationship": sufficiency.get(
                "evidence_relationship", "unknown"
            ),
            "conflict_status": conflict_status,
            "preferred_evidence_ids": conflict.get(
                "preferred_evidence_ids", []
            ),
            "superseded_evidence_ids": conflict.get(
                "superseded_evidence_ids", []
            ),
        }

    def _answer(self, query, context_pack, sufficiency, answer_policy):
        system = """你是國家圖書館領域的最終問答模型。你必須區分「檢索證據可支持的內容」與「模型自行推測」。

核心規則：
1. 先依 Evidence Context 回答；不可把 background/相似案例類比成新的正式編目規則。
2. 若 retrieval sufficient=false 且 rule_sensitive=true（例如 MARC/RDA 欄位、指標、分欄、代碼、分類/作者號等正式規則），不得用模型常識補成確定答案；只回答 evidence 已支持的部分，並明確指出尚缺哪個規則。
3. 若 conflict_status=resolved_by_date，優先採用 Python 已標出的 preferred_evidence_ids；較舊的 conflicting evidence 只作歷史/差異說明，不可與最新規則並列成同等答案。
4. 若 conflict_status=unresolved，不得自行選邊；明確告知現有資料在相同適用條件下有不一致，交由館員依最新版正式規範確認。
5. conditional 不是 conflict。若 evidence 因適用條件不同而答案不同，應把條件說清楚。
6. 非 rule-sensitive 且允許 knowledge assist 時，若使用模型知識，必須明確寫「以下補充未由本次檢索證據直接驗證：」；不得偽造 evidence ID。
7. 使用者端不要輸出 direct/supporting 等內部 relevance 標籤，也不要輸出「知識充分性：直接支持」。館員只需要答案、依據，以及必要的推論/衝突/不足提醒。
8. 證據 ID 只能引用 Context 中真正存在者。
9. 若 evidence 涉及數字、碼數、位址、指標值、分欄順序、固定格式或逐項操作規則，必須忠實保留 evidence 的明確內容；不得在改寫或摘要時自行改變數值、碼數、位置或條件。若文字敘述與 evidence 中的具體範例看似不一致，不得自行猜測，應只陳述 evidence 可確認的部分。
"""

        conflict = sufficiency.get("conflict_resolution", {}) or {}
        missing = sufficiency.get("missing_aspects", []) or []
        policy_text = (
            f"mode={answer_policy.get('mode')}\n"
            f"retrieval_sufficient={answer_policy.get('retrieval_sufficient')}\n"
            f"rule_sensitive={answer_policy.get('rule_sensitive')}\n"
            f"knowledge_assist_allowed={answer_policy.get('knowledge_assist_allowed')}\n"
            f"coverage_mode={answer_policy.get('coverage_mode')}\n"
            f"evidence_relationship={answer_policy.get('evidence_relationship')}\n"
            f"conflict_status={answer_policy.get('conflict_status')}\n"
            f"preferred_evidence_ids={answer_policy.get('preferred_evidence_ids')}\n"
            f"superseded_evidence_ids={answer_policy.get('superseded_evidence_ids')}"
        )
        sufficiency_text = (
            f"query_aspects={sufficiency.get('query_aspects', [])}\n"
            f"covered_aspects={sufficiency.get('covered_aspects', [])}\n"
            f"missing_aspects={missing}\n"
            f"reason={sufficiency.get('reason', '')}\n"
            f"conflict_groups={conflict.get('groups', [])}"
        )

        user = f"""請回答以下問題：
{query}

Answer Policy（系統內部判斷，只用來約束回答，不要逐欄照抄）：
{policy_text}

Evidence Sufficiency / Conflict Trace：
{sufficiency_text}

以下是 Tree-Guided Progressive RAG 建立的最終 Knowledge Pack：
{context_pack}

請使用以下格式：
分類路徑：
回答：
判斷依據：
證據來源：
注意事項：

注意事項的規則：
- 正常且無特殊風險時寫「無」。
- 若是 evidence 組合推論，簡短說明「此結論由多筆既有證據組合得出」。
- 若 evidence 不足，列出真正缺少的規則/資訊。
- 若有 unresolved conflict，列出衝突與需館員確認之處。
- 不要輸出 direct/supporting/知識充分性 等內部標籤。
"""

        answer_think = False if self.service_mode else bool(self.config.answer_think)
        return self.answer_llm.chat_text(
            self.config.answer_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            think=answer_think,
        )

    # ------------------------------------------------------------------
    # Offline static knowledge
    # ------------------------------------------------------------------

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

    def build_node_summaries(
        self,
        force_rebuild=False,
        include_l2_parents=True,
    ):
        return self.build_static_knowledge_packs(
            force_rebuild=force_rebuild,
            include_l2_parents=include_l2_parents,
        )

    # ------------------------------------------------------------------
    # Main query pipeline
    # ------------------------------------------------------------------

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

        l1_beam = (
            self.config.l1_beam
            if l1_beam is None
            else l1_beam
        )
        l2_beam = (
            self.config.l2_beam
            if l2_beam is None
            else l2_beam
        )
        final_beam = (
            self.config.final_beam
            if final_beam is None
            else final_beam
        )

        if final_context_unit_limit is None:
            final_context_unit_limit = (
                self.config.final_context_unit_limit
            )

        # Older config.py uses 0 = unlimited. Service Mode caps final context
        # by default so Final Answer latency remains bounded.
        if (
            self.service_mode
            and (
                final_context_unit_limit is None
                or int(final_context_unit_limit) <= 0
            )
        ):
            final_context_unit_limit = self._service_int(
                "service_final_context_unit_limit",
                12,
                minimum=1,
            )

        if force_rebuild_knowledge is None:
            force_rebuild_knowledge = bool(
                force_rebuild_summaries
            )

        timings = {}

        # --------------------------------------------------------------
        # 1) Tree-first routing
        # --------------------------------------------------------------
        t0 = time.perf_counter()

        paths = self.router.route_tree(
            query,
            l1_beam=l1_beam,
            l2_global_beam=l2_beam,
            final_beam=final_beam,
        )

        timings["routing"] = time.perf_counter() - t0

        if not paths:
            raise RuntimeError(
                "Tree Router 未回傳任何候選路徑"
            )

        # --------------------------------------------------------------
        # 2) Build routed node states locally
        # --------------------------------------------------------------
        t0 = time.perf_counter()

        states = []
        state_by_key = {}
        routed_path_by_key = {}

        for rank, path in enumerate(paths):
            key = self._path_key(path)
            routed_path_by_key[key] = path

            if key in state_by_key:
                continue

            role = (
                "primary"
                if rank == 0
                else "alternative"
            )
            origin = (
                "primary"
                if rank == 0
                else "routed_alternative"
            )

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

        timings["frontier_prepare"] = (
            time.perf_counter() - t0
        )

        primary_state = state_by_key[
            self._path_key(paths[0])
        ]

        knowledge_packs = []
        opened_pack_keys = set()
        prioritized_evidence = []
        sufficiency = None
        sufficiency_initial = None
        faq_packs = []
        faq_added_total = 0
        progressive_history = []
        sibling_expansion_paths = []
        service_stop_reason = ""

        # --------------------------------------------------------------
        # 3) Mandatory primary-node first batch
        # --------------------------------------------------------------
        t0 = time.perf_counter()

        (
            prioritized_evidence,
            sufficiency,
            faq_added,
        ) = self._assess_one_progressive_batch(
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
            sufficiency = self._check_sufficiency(
                query,
                prioritized_evidence,
            )

        sufficiency_initial = sufficiency

        # Only expand sibling L3 candidates if the first primary batch
        # did not provide sufficient evidence.
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
                state["pack"].get("path", "")
                for state in added_states
            )

        timings["primary_probe"] = (
            time.perf_counter() - t0
        )

        # --------------------------------------------------------------
        # 4) Progressive frontier
        # --------------------------------------------------------------
        t0 = time.perf_counter()

        while not sufficiency.get("sufficient", False):
            if self.service_mode:
                elapsed_total = (
                    time.perf_counter() - total_t0
                )

                max_steps = self._service_int(
                    "service_max_progressive_steps",
                    3,
                    minimum=1,
                )

                search_budget = self._service_float(
                    "service_search_budget_seconds",
                    38.0,
                    minimum=1.0,
                )

                # progressive_history already includes the first primary batch.
                if len(progressive_history) >= max_steps:
                    service_stop_reason = (
                        "max_progressive_steps"
                    )
                    break

                if elapsed_total >= search_budget:
                    service_stop_reason = (
                        "search_time_budget"
                    )
                    break

            focus = self._focus_text(
                query,
                sufficiency,
                prioritized_evidence,
            )

            ranked_frontier = self._frontier_priority(
                query,
                states,
                focus,
            )

            # Service Mode prevents repeatedly scanning a large node and
            # opening many alternative/sibling nodes online.
            if self.service_mode:
                max_batches_per_node = self._service_int(
                    "service_max_batches_per_node",
                    2,
                    minimum=1,
                )

                max_nonprimary_nodes = self._service_int(
                    "service_max_nonprimary_nodes",
                    2,
                    minimum=0,
                )

                opened_nonprimary_keys = {
                    state.get("key")
                    for state in states
                    if (
                        state.get("opened")
                        and state.get("origin") != "primary"
                    )
                }
                filtered_frontier = []
                for item in ranked_frontier:
                    candidate_state = item["state"]
                    if int(candidate_state.get("batches", 0)) >= max_batches_per_node:
                        continue
                    if candidate_state.get("origin") != "primary":
                        candidate_key = candidate_state.get("key")
                        is_new_nonprimary = candidate_key not in opened_nonprimary_keys
                        if is_new_nonprimary and len(opened_nonprimary_keys) >= max_nonprimary_nodes:
                            continue
                    filtered_frontier.append(item)
                ranked_frontier = filtered_frontier

            if not ranked_frontier:
                service_stop_reason = (
                    "service_frontier_limit"
                    if self.service_mode
                    else "frontier_exhausted"
                )
                break

            chosen = ranked_frontier[0]
            state = chosen["state"]
            state["last_frontier_priority"] = chosen.get("priority")
            state["last_frontier_route_score"] = chosen.get("route_score")
            state["last_frontier_local_norm"] = chosen.get("local_norm")
            state["last_frontier_policy"] = chosen.get("ranking_policy", "")

            (
                prioritized_evidence,
                sufficiency,
                faq_added,
            ) = self._assess_one_progressive_batch(
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
                service_stop_reason = "sufficient"
                break

            # Once a routed/sibling L3 actually receives a batch,
            # add its sibling family lazily.
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
                    child["pack"].get("path", "")
                    for child in added_states
                )

        timings["progressive_tree_search"] = (
            time.perf_counter() - t0
        )

        # --------------------------------------------------------------
        # 5) Final safety net
        # --------------------------------------------------------------
        fallback_pack = None
        fallback_added = 0
        t0 = time.perf_counter()

        allow_fallback = True

        if self.service_mode:
            fallback_cutoff = self._service_float(
                "service_fallback_cutoff_seconds",
                42.0,
                minimum=1.0,
            )

            elapsed_before_fallback = (
                time.perf_counter() - total_t0
            )

            if elapsed_before_fallback >= fallback_cutoff:
                allow_fallback = False
                if not service_stop_reason:
                    service_stop_reason = (
                        "fallback_skipped_time_budget"
                    )

        if (
            not sufficiency.get("sufficient", False)
            and allow_fallback
        ):
            force_fallback = bool(
                self.config.fallback_on_tree_exhausted
            )

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
                    item.get("evidence_id")
                    for item in prioritized_evidence
                }

                prioritized_evidence = (
                    self.evidence_selector.extend_prioritized(
                        query,
                        prioritized_evidence,
                        [fallback_pack],
                    )
                )

                fallback_added = sum(
                    1
                    for item in prioritized_evidence
                    if (
                        item.get("evidence_id")
                        not in before_ids
                    )
                )

                if fallback_added:
                    sufficiency = self._check_sufficiency(
                        query,
                        prioritized_evidence,
                    )

        timings["global_fallback"] = (
            time.perf_counter() - t0
        )

        all_prioritized_count = len(
            prioritized_evidence
        )

        # --------------------------------------------------------------
        # 6) Final context + answer
        # --------------------------------------------------------------
        final_context_evidence = (
            self.evidence_selector.filter_for_context(
                prioritized_evidence,
                include_background=(
                    self.config.final_include_background
                ),
            )
        )

        final_context_before_limit = len(
            final_context_evidence
        )

        if (
            final_context_unit_limit
            and final_context_unit_limit > 0
        ):
            final_context_evidence = (
                final_context_evidence[
                    :final_context_unit_limit
                ]
            )

        t0 = time.perf_counter()

        context_pack = self.context_builder.build(
            query,
            paths,
            knowledge_packs,
            final_context_evidence,
        )

        answer_policy = self._answer_policy(sufficiency)
        answer = self._answer(
            query,
            context_pack,
            sufficiency,
            answer_policy,
        )

        timings["final_answer"] = (
            time.perf_counter() - t0
        )
        timings["total"] = (
            time.perf_counter() - total_t0
        )

        # --------------------------------------------------------------
        # Trace / metrics
        # --------------------------------------------------------------
        assessed_by_node = {}
        state_meta = {}

        for state in states:
            path = state["pack"].get("path", "")
            assessed_by_node[path] = state.get(
                "assessed_unique",
                0,
            )
            state_meta[path] = state

        node_knowledge = []

        for pack in knowledge_packs:
            path = pack.get("path", "")
            candidate_count = len(
                pack.get("knowledge_units", [])
            )

            assessed_count = assessed_by_node.get(
                path,
                candidate_count,
            )

            if pack.get("role") in {
                "faq_family",
                "fallback",
            }:
                assessed_count = candidate_count

            node_knowledge.append(
                {
                    "role": pack.get("role"),
                    "path": path,
                    "static_source": pack.get(
                        "static_source",
                        "",
                    ),
                    "document_count": pack.get(
                        "document_count",
                        0,
                    ),
                    "date_start": pack.get(
                        "date_start",
                        "",
                    ),
                    "date_end": pack.get(
                        "date_end",
                        "",
                    ),
                    "knowledge_unit_count": (
                        candidate_count
                    ),
                    "source_coverage_ratio": pack.get(
                        "source_coverage_ratio",
                        0.0,
                    ),
                    "coverage_note": pack.get(
                        "coverage_note",
                        "",
                    ),
                    "cache_hit": pack.get(
                        "cache_hit",
                        False,
                    ),
                    "assessed_unit_count": (
                        assessed_count
                    ),
                    "assessment_coverage_ratio": (
                        assessed_count / candidate_count
                        if candidate_count
                        else 0.0
                    ),
                }
            )

        frontier_remaining = sum(
            len(state.get("remaining_units", []))
            for state in states
        )

        frontier_total_candidates = sum(
            len(
                state["pack"].get(
                    "knowledge_units",
                    [],
                )
            )
            for state in states
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
            "final_context_evidence": (
                final_context_evidence
            ),
            "sufficiency": {
                "after_primary_probe": (
                    sufficiency_initial
                ),
                "final": sufficiency,
            },
            "parent_fallback": {
                "triggered": bool(
                    sibling_expansion_paths
                ),
                "reason": (
                    "replaced by progressive sibling-L3 expansion; "
                    "no whole-L2 parent pack"
                    if sibling_expansion_paths
                    else "not needed"
                ),
                "added_unique_evidence": sum(
                    state.get(
                        "assessed_unique",
                        0,
                    )
                    for state in states
                    if (
                        state.get("origin")
                        == "sibling_l3"
                    )
                ),
                "path": "",
                "route_uncertain": False,
            },
            "sibling_expansion": {
                "triggered": bool(
                    sibling_expansion_paths
                ),
                "candidate_paths_added": list(
                    dict.fromkeys(
                        sibling_expansion_paths
                    )
                ),
                "opened_paths": [
                    state["pack"].get(
                        "path",
                        "",
                    )
                    for state in states
                    if (
                        state.get("origin")
                        == "sibling_l3"
                        and state.get("opened")
                    )
                ],
            },
            "faq_expansion": self._faq_trace(
                faq_packs,
                faq_added_total,
            ),
            "progressive_search": {
                "batch_size": (
                    self._progressive_batch_size()
                ),
                "service_mode": self.service_mode,
                "service_stop_reason": (
                    service_stop_reason
                ),
                "service_search_budget_seconds": (
                    self._service_float(
                        "service_search_budget_seconds",
                        38.0,
                        minimum=1.0,
                    )
                    if self.service_mode
                    else None
                ),
                "service_max_progressive_steps": (
                    self._service_int(
                        "service_max_progressive_steps",
                        3,
                        minimum=1,
                    )
                    if self.service_mode
                    else None
                ),
                "service_max_batches_per_node": (
                    self._service_int(
                        "service_max_batches_per_node",
                        2,
                        minimum=1,
                    )
                    if self.service_mode
                    else None
                ),
                "service_max_nonprimary_nodes": (
                    self._service_int(
                        "service_max_nonprimary_nodes",
                        2,
                        minimum=0,
                    )
                    if self.service_mode
                    else None
                ),
                "embedding_enabled_online": (
                    self.retriever.use_embedding
                ),
                "history": progressive_history,
                "frontier_node_count": len(states),
                "frontier_total_candidate_units": (
                    frontier_total_candidates
                ),
                "frontier_remaining_unassessed_units": (
                    frontier_remaining
                ),
                "stopped_early": bool(
                    (
                        sufficiency.get(
                            "sufficient",
                            False,
                        )
                        or self.service_mode
                    )
                    and frontier_remaining > 0
                ),
                "local_ordering_is_hard_filter": False,
                "frontier_policy": (
                    "routed alternatives by Router score; same-parent L3 siblings "
                    "use discounted parent score + bounded local BM25 signal"
                ),
                "visited_path_sequence": [
                    item.get("path", "")
                    for item in progressive_history
                ],
            },
            "evidence_evaluation": {
                "final_unique_evidence": (
                    all_prioritized_count
                ),
                "utility_counts": (
                    self.evidence_selector.utility_counts(
                        prioritized_evidence
                    )
                ),
                "final_context_before_limit": (
                    final_context_before_limit
                ),
                "final_context_after_limit": len(
                    final_context_evidence
                ),
                "fallback_added_unique_evidence": (
                    fallback_added
                ),
                **self.evidence_selector.stats(),
            },
            "evidence_total_before_context_limit": (
                all_prioritized_count
            ),
            "decision_trace": {
                **answer_policy,
                "query_aspects": sufficiency.get("query_aspects", []),
                "covered_aspects": sufficiency.get("covered_aspects", []),
                "missing_aspects": sufficiency.get("missing_aspects", []),
                "conflict_groups": (
                    sufficiency.get("conflict_resolution", {}) or {}
                ).get("groups", []),
            },
            "timing_seconds": {
                k: round(v, 4)
                for k, v in timings.items()
            },
            "context_pack": context_pack,
            "answer": answer,
        }
