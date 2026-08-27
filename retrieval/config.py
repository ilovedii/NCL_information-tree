from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class AppConfig:
    csv_path: str = str(PROJECT_DIR / "data" / "llm_retrieval.csv")

    # Local Ollama is retained for fallback embeddings only.
    ollama_url: str = "http://localhost:11434"

    # NCL OpenAI-compatible API.
    ncl_api_url: str = "https://catsmartap.ncl.edu.tw/v1"
    ncl_verify_ssl: bool = False

    # Generation provider: "ncl" or "ollama".
    router_provider: str = "ncl"
    evidence_provider: str = "ncl"
    answer_provider: str = "ncl"

    router_model: str = "google/gemma-4-31B-it"
    evidence_model: str = "google/gemma-4-31B-it"
    answer_model: str = "google/gemma-4-31B-it"

    embedding_model: str = "qwen3-embedding:0.6b"

    router_think: bool = True
    evidence_think: bool = True
    answer_think: bool = True

    # Keep taxonomy/router behavior unchanged for a clean ablation.
    l1_beam: int = 2
    l2_beam: int = 3
    final_beam: int = 4

    # Static knowledge: no query-time LLM summarization.
    static_knowledge_dir: str = str(PROJECT_DIR / "cache" / "static_node_knowledge")
    static_include_topic: bool = True
    static_exact_dedup: bool = True

    # ------------------------------------------------------------------
    # Progressive evidence search
    # ------------------------------------------------------------------
    # Number of locally ordered units sent to the Evidence LLM at one time.
    # Candidates after this batch are NOT deleted; they remain in the frontier.
    progressive_batch_size: int = 40
    use_progressive_search: bool = True

    # Locally rank remaining nodes/units with dependency-free lexical BM25-like
    # scoring. This is ordering only, never a hard Top-K gate.
    use_local_candidate_ordering: bool = True

    # When an unselected sibling L3 is added to the frontier, inherit the
    # confidence of its routed L2 parent with a modest discount.
    sibling_route_discount: float = 0.85

    # Relative influence when choosing which frontier node gets the next batch.
    frontier_local_weight: float = 0.80
    frontier_route_weight: float = 0.20

    # Expand sibling L3 nodes around a routed L3 path when evidence is still
    # insufficient. Existing taxonomy assignments are reused; no reclassification.
    use_sibling_l3_expansion: bool = True

    # Evidence selection and sufficiency.
    evidence_batch_size: int = 40
    use_sufficiency_check: bool = True
    sufficiency_min_supporting_without_direct: int = 2
    final_include_background: bool = False
    final_context_unit_limit: int = 0

    # Provenance-based FAQ family expansion.
    use_faq_expansion: bool = True
    max_anchor_evidence: int = 5
    max_anchor_faqs: int = 3
    max_siblings_per_faq: int = 15

    # Final safety net. Route-confidence fallback is preserved; additionally,
    # if the progressive Tree frontier is completely exhausted and evidence is
    # still insufficient, the same global fallback can be forced once.
    use_fallback_retrieval: bool = True
    fallback_on_tree_exhausted: bool = True
    routing_confidence_threshold: float = 0.20
    fallback_top_k: int = 12
    use_embedding: bool = True
    embedding_batch_size: int = 64

    # ------------------------------------------------------------------
    # Backward-compatible flags from previous versions.
    # They are kept so old CLI commands do not break, but Parent is no longer
    # loaded as one giant L2 pack in the progressive pipeline.
    # ------------------------------------------------------------------
    include_parent_summary: bool = True
    parent_expand_on_router_uncertain: bool = True
    parent_fallback_direct_score: float = 0.80
    parent_fallback_path_threshold: float = 0.75

    timeout: int = 180
