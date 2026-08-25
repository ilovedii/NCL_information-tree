from context_builder import ContextBuilder
from evidence_selector import EvidenceSelector
from knowledge_loader import KnowledgeLoader
from knowledge_summarizer import KnowledgeSummarizer
from llm_client import OllamaClient
from retriever import HybridRetriever
from router import TreeRouter
from taxonomy import TaxonomyIndex


class TreeGuidedRAG:
    def __init__(
        self,
        config,
        router_llm=None,
        summarizer_llm=None,
        evidence_llm=None,
        answer_llm=None,
        embedding_llm=None,
    ):
        self.config = config
        self.taxonomy = TaxonomyIndex(config.csv_path)
        default_llm = OllamaClient(config.ollama_url, timeout=config.timeout)
        self.router_llm = router_llm or default_llm
        self.summarizer_llm = summarizer_llm or default_llm
        self.evidence_llm = evidence_llm or default_llm
        self.answer_llm = answer_llm or default_llm
        self.embedding_llm = embedding_llm or default_llm
        self.router = TreeRouter(
            self.taxonomy,
            self.router_llm,
            config.router_model,
            think=config.router_think,
        )
        self.knowledge_loader = KnowledgeLoader(self.taxonomy)
        self.knowledge_summarizer = KnowledgeSummarizer(
            self.taxonomy,
            self.summarizer_llm,
            config.summarizer_model,
            cache_dir=config.summary_cache_dir,
            batch_size=config.summary_batch_size,
            max_units_per_batch=config.summary_max_units_per_batch,
            merge_group_size=config.summary_merge_group_size,
            merge_max_units=config.summary_merge_max_units,
            think=config.summarizer_think,
            use_cache=config.use_summary_cache,
        )
        self.evidence_selector = EvidenceSelector(
            self.evidence_llm,
            config.evidence_model,
            think=config.evidence_think,
            batch_size=config.evidence_batch_size,
        )
        self.context_builder = ContextBuilder(self.taxonomy)
        self.retriever = HybridRetriever(
            self.taxonomy,
            self.embedding_llm,
            config.embedding_model,
            use_embedding=config.use_embedding,
            batch_size=config.embedding_batch_size,
        )

    def _knowledge_bundles(self, paths):
        bundles = []
        seen = set()
        for rank, path in enumerate(paths):
            key = path.key()
            if key in seen:
                continue
            seen.add(key)
            role = "primary" if rank == 0 else "alternative"
            bundles.append(self.knowledge_loader.load_path(path, role=role))
        if self.config.include_parent_summary and paths and paths[0].l3:
            parent = self.knowledge_loader.load_parent(paths[0])
            if parent is not None:
                key = (parent["l1"], parent["l2"], parent["l3"])
                if key not in seen:
                    bundles.append(parent)
        return bundles

    def _summarize_bundles(self, bundles, force_rebuild=False):
        summaries = []
        for bundle in bundles:
            summaries.append(
                self.knowledge_summarizer.summarize(
                    bundle,
                    force_rebuild=force_rebuild,
                )
            )
        return summaries

    def _fallback_summary(self, query, paths):
        if not self.config.use_fallback_retrieval:
            return None
        if not paths:
            should_fallback = True
        else:
            should_fallback = paths[0].score < self.config.routing_confidence_threshold
        if not should_fallback:
            return None
        indices = list(range(len(self.taxonomy.df)))
        query_vector = self.retriever.query_embedding(query) if self.retriever.use_embedding else None
        hits = self.retriever.search(
            query,
            indices,
            top_k=self.config.fallback_top_k,
            query_vector=query_vector,
        )
        units = []
        source_ids = []
        for i, hit in enumerate(hits, start=1):
            idx = hit["idx"]
            atomic_id = str(self.taxonomy.df.at[idx, "atomic_id"])
            source_ids.append(atomic_id)
            units.append(
                {
                    "knowledge_id": f"F{i:03d}",
                    "type": "background",
                    "content": f"Q：{self.taxonomy.df.at[idx, 'question']}\nA：{self.taxonomy.df.at[idx, 'answer']}",
                    "time_scope": str(self.taxonomy.df.at[idx, "question_date"]),
                    "source_ids": [atomic_id],
                }
            )
        return {
            "fingerprint": "fallback",
            "cache_hit": False,
            "role": "fallback",
            "level": "GLOBAL",
            "path": "Global Hybrid Fallback",
            "document_count": len(units),
            "date_start": "",
            "date_end": "",
            "all_source_ids": source_ids,
            "knowledge_units": units,
            "coverage_note": "Tree routing 信心低，因此加入全域 Hybrid Retrieval 作為安全網。",
            "source_coverage_ratio": 1.0 if units else 0.0,
        }

    def _faq_family_summary(self, prioritized_evidence):
        """Expand direct evidence to atomic siblings from the same original FAQ.

        This is a deterministic provenance lookup, not another taxonomy-routing step.
        The expanded units are returned as a summary-like bundle so the existing
        EvidenceSelector can re-rank them against the current user query.
        """
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
            atomic_id = str(record["atomic_id"]).strip()
            faq_id = str(record.get("faq_id", "")).strip()

            if not atomic_id or atomic_id in existing_source_ids:
                continue

            units.append(
                {
                    "knowledge_id": f"FAQ{len(units) + 1:03d}",
                    "type": "related_source",
                    "content": f"Q：{record['question']}\nA：{record['answer']}",
                    "time_scope": record["question_date"],
                    "source_ids": [atomic_id],
                }
            )
            expanded_source_ids.append(atomic_id)
            if faq_id and faq_id not in seen_faq_ids:
                seen_faq_ids.add(faq_id)
                expanded_faq_ids.append(faq_id)

        if not units:
            return None

        return {
            "fingerprint": "faq_family_expansion",
            "cache_hit": False,
            "role": "faq_family",
            "level": "FAQ",
            "path": "Related Original FAQ",
            "document_count": len(units),
            "date_start": "",
            "date_end": "",
            "all_source_ids": expanded_source_ids,
            "knowledge_units": units,
            "coverage_note": (
                "根據 direct evidence 的原始 faq_id，補回同一原始問答中"
                "被 atomic decomposition 拆開、但目前 Tree 節點未取回的知識。"
            ),
            "source_coverage_ratio": 1.0,
            "anchor_source_ids": anchor_ids,
            "expanded_faq_ids": expanded_faq_ids,
            "expanded_source_ids": expanded_source_ids,
        }

    def _answer(self, query, context_pack):
        system = """你是國家圖書館領域的最終問答模型。請只依提供的正式 taxonomy、分類路徑與節點知識摘要回答。Knowledge Units 已由 Evidence Prioritizer 標示 direct、supporting、background、low_relevance；請優先依 direct 與 supporting 作答，但在需要判斷例外、時間演變、版本差異或衝突時，必須檢查其他單元。若知識沒有直接答案，可以在已提供規則之間做保守歸納並清楚標示推論。若仍不足，明確指出缺少什麼知識，不得虛構。回答時保留支持結論的 atomic_id。"""
        user = f"""請回答以下問題：
{query}

以下是 Tree-Guided Hierarchical RAG 建立的 Knowledge Pack：
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
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            think=self.config.answer_think,
        )

    def build_node_summaries(self, force_rebuild=False, include_l2_parents=True):
        results = []
        nodes = self.taxonomy.summary_nodes(include_l2_parents=include_l2_parents)
        for i, node in enumerate(nodes, start=1):
            bundle = self.knowledge_loader.load_summary_node(node)
            summary = self.knowledge_summarizer.summarize(
                bundle,
                force_rebuild=force_rebuild,
            )
            results.append(
                {
                    "index": i,
                    "total": len(nodes),
                    "path": summary["path"],
                    "document_count": summary["document_count"],
                    "knowledge_units": len(summary["knowledge_units"]),
                    "cache_hit": summary.get("cache_hit", False),
                    "source_coverage_ratio": summary.get("source_coverage_ratio", 0.0),
                }
            )
        return results

    def run(
        self,
        query,
        l1_beam=None,
        l2_beam=None,
        final_beam=None,
        force_rebuild_summaries=False,
        final_context_unit_limit=None,
    ):
        query = str(query).strip()
        if not query:
            raise ValueError("query 不可為空")
        l1_beam = self.config.l1_beam if l1_beam is None else l1_beam
        l2_beam = self.config.l2_beam if l2_beam is None else l2_beam
        final_beam = self.config.final_beam if final_beam is None else final_beam
        if final_context_unit_limit is None:
            final_context_unit_limit = self.config.final_context_unit_limit
        paths = self.router.route_tree(
            query,
            l1_beam=l1_beam,
            l2_global_beam=l2_beam,
            final_beam=final_beam,
        )
        bundles = self._knowledge_bundles(paths)
        summaries = self._summarize_bundles(
            bundles,
            force_rebuild=force_rebuild_summaries,
        )
        fallback = self._fallback_summary(query, paths)
        if fallback is not None:
            summaries.append(fallback)
        prioritized_evidence = self.evidence_selector.prioritize(query, summaries)

        faq_family = None
        if self.config.use_faq_expansion:
            faq_family = self._faq_family_summary(prioritized_evidence)
            if faq_family is not None:
                summaries.append(faq_family)
                # Re-run relevance reasoning because same-FAQ siblings are candidates,
                # not automatically valid answers to the current query.
                prioritized_evidence = self.evidence_selector.prioritize(query, summaries)

        all_prioritized_count = len(prioritized_evidence)
        if final_context_unit_limit and final_context_unit_limit > 0:
            prioritized_evidence = prioritized_evidence[:final_context_unit_limit]
        context_pack = self.context_builder.build(
            query,
            paths,
            summaries,
            prioritized_evidence,
        )
        answer = self._answer(query, context_pack)
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
            "node_summaries": [
                {
                    "role": summary.get("role"),
                    "path": summary.get("path"),
                    "document_count": summary.get("document_count", 0),
                    "date_start": summary.get("date_start", ""),
                    "date_end": summary.get("date_end", ""),
                    "knowledge_unit_count": len(summary.get("knowledge_units", [])),
                    "source_coverage_ratio": summary.get("source_coverage_ratio", 0.0),
                    "coverage_note": summary.get("coverage_note", ""),
                    "cache_hit": summary.get("cache_hit", False),
                }
                for summary in summaries
            ],
            "evidence": prioritized_evidence,
            "faq_expansion": (
                {
                    "triggered": True,
                    "anchor_source_ids": faq_family.get("anchor_source_ids", []),
                    "expanded_faq_ids": faq_family.get("expanded_faq_ids", []),
                    "expanded_source_ids": faq_family.get("expanded_source_ids", []),
                }
                if faq_family is not None
                else {"triggered": False}
            ),
            "evidence_total_before_context_limit": all_prioritized_count,
            "context_pack": context_pack,
            "answer": answer,
        }