from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class AppConfig:
    csv_path: str = str(
        PROJECT_DIR / "data" / "llm_retrieval.csv"
    )

    # Ollama stays available as an optional provider.
    # Service Mode below does NOT use online embedding.
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

    # Kept for backward compatibility / research mode.
    embedding_model: str = "qwen3-embedding:0.6b"

    # Service latency: Router may keep thinking;
    # Evidence/Final Answer disable think to reduce latency.
    router_think: bool = True
    evidence_think: bool = False
    answer_think: bool = False

    # ------------------------------------------------------------------
    # Ranked hierarchical routing
    # ------------------------------------------------------------------
    # Router keeps ranked alternatives so RAG can backtrack without
    # making another Router API call.
    router_l1_max_choices: int = 2
    router_l2_max_choices: int = 3
    router_l3_max_choices: int = 8

    # Beam retained after each hierarchy stage.
    l1_beam: int = 2
    l2_beam: int = 3
    final_beam: int = 12

    # Static knowledge: no query-time LLM summarization.
    static_knowledge_dir: str = str(
        PROJECT_DIR / "cache" / "static_node_knowledge"
    )
    static_include_topic: bool = True
    static_exact_dedup: bool = True

    # ------------------------------------------------------------------
    # Progressive evidence search
    # ------------------------------------------------------------------
    # BM25-like CandidateOrderer is ONLY used inside the selected node.
    # It does not choose L1/L2/L3 in Service Mode.
    progressive_batch_size: int = 8
    evidence_batch_size: int = 8
    use_progressive_search: bool = True
    use_local_candidate_ordering: bool = True

    # Research-mode compatibility. Service Mode uses LLM-ranked backtracking.
    use_sibling_l3_expansion: bool = True
    sibling_route_discount: float = 0.60
    # Routed alternatives are now Router-score dominant.
    # Local BM25 is still used inside each node and as a light signal for
    # same-parent sibling recovery; it is not allowed to override a clearly
    # better routed branch by itself.
    frontier_local_weight: float = 0.25
    frontier_route_weight: float = 0.75

    # Same-parent L3 recovery gets a bounded local-evidence bonus.
    # This preserves T01 Leader/07 sibling recovery while avoiding the
    # U02/U03 failure where a lower-scored branch displaced Router #2.
    sibling_local_weight: float = 0.30
    sibling_route_weight: float = 0.70
    sibling_override_margin: float = 0.05

    # Evidence / sufficiency.
    use_sufficiency_check: bool = True
    sufficiency_min_supporting_without_direct: int = 2
    final_include_background: bool = False
    final_context_unit_limit: int = 12

    # Provenance-based FAQ family expansion.
    use_faq_expansion: bool = True
    max_anchor_evidence: int = 3
    max_anchor_faqs: int = 2
    max_siblings_per_faq: int = 8

    # ------------------------------------------------------------------
    # Service Mode: ranked hierarchical backtracking
    # ------------------------------------------------------------------
    service_mode: bool = True

    # Primary L3 + at most two further routed nodes.
    # Typical policy:
    # L3 #1 -> L3 #2 -> L3 #3,
    # then stop / fallback if time remains.
    service_max_progressive_steps: int = 7

    # One batch per routed node by default.
    # This prioritizes breadth over repeatedly scanning a wrong L3.
    service_max_batches_per_node: int = 5

    # Allow up to two distinct non-primary nodes in one query.
    # V3.2 effectively locked onto the first non-primary node, which caused
    # U02/U03 to skip a higher-scored Router alternative.
    service_max_nonprimary_nodes: int = 2

    # If True, a node may get one extra batch ONLY if the previous batch
    # added direct evidence or reduced missing_aspects.
    # Keep False initially for the <60s service target.
    service_retry_node_on_progress: bool = False

    # Stop launching new Evidence-LLM work after this elapsed query time.
    service_search_budget_seconds: float = 200.0

    # Global BM25 fallback is only attempted while enough budget remains.
    service_fallback_cutoff_seconds: float = 43.0

    # Final answer context cap.
    service_final_context_unit_limit: int = 12

    # ------------------------------------------------------------------
    # Knowledge-assisted final fallback
    # ------------------------------------------------------------------
    # When retrieval remains insufficient, Final Answer may use model
    # domain knowledge only for non-rule-sensitive questions. Exact MARC/RDA/
    # code/field/indicator cataloguing rules must abstain when evidence is
    # insufficient instead of analogically inventing a rule (U01 safeguard).
    allow_knowledge_assisted_answer: bool = True
    block_knowledge_assist_for_rule_sensitive: bool = True

    # Conflict analysis is folded into the existing Sufficiency LLM call.
    # LLM classifies evidence relationships; Python resolves true same-scope
    # conflicts by question_date first. Ties/missing dates remain unresolved.
    enable_conflict_analysis: bool = True

    # ------------------------------------------------------------------
    # Final retrieval safety net
    # ------------------------------------------------------------------
    use_fallback_retrieval: bool = True
    fallback_on_tree_exhausted: bool = True
    routing_confidence_threshold: float = 0.20

    # Service Mode uses BM25-only global fallback.
    fallback_top_k: int = 8
    use_embedding: bool = False
    embedding_batch_size: int = 64

    # ------------------------------------------------------------------
    # Backward-compatible flags
    # ------------------------------------------------------------------
    include_parent_summary: bool = True
    parent_expand_on_router_uncertain: bool = True
    parent_fallback_direct_score: float = 0.80
    parent_fallback_path_threshold: float = 0.75

    # Per API call timeout, not overall SLA.
    timeout: int = 45
