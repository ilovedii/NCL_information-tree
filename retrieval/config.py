from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class AppConfig:
    """V4 Simple Tree-RAG configuration.

    Design goal: two main LLM stages only.
    1) one-shot Tree retrieval planning
    2) grounded answer / fallback decision

    All document retrieval between them is local BM25.
    """

    csv_path: str = str(PROJECT_DIR / "data" / "llm_retrieval.csv")

    # Providers
    ollama_url: str = "http://localhost:11434"
    ncl_api_url: str = "https://catsmartap.ncl.edu.tw/v1"
    ncl_verify_ssl: bool = False
    router_provider: str = "ncl"
    answer_provider: str = "ncl"

    # Models
    router_model: str = "google/gemma-4-31B-it"
    answer_model: str = "google/gemma-4-31B-it"
    embedding_model: str = "qwen3-embedding:0.6b"  # compatibility; V4 default is BM25-only
    router_think: bool = False
    answer_think: bool = False

    # ------------------------------------------------------------------
    # Stage 1: one-shot Tree planning
    # ------------------------------------------------------------------
    # First shortlist taxonomy leaves locally with BM25, then ask ONE LLM call
    # to choose the most plausible paths and create a safe retrieval rewrite.
    tree_candidate_pool: int = 16
    router_top_paths: int = 3

    # ------------------------------------------------------------------
    # Multi-channel local retrieval
    # ------------------------------------------------------------------
    # Tree is a retrieval prior, not a hard filter.
    tree_primary_top_k: int = 6
    tree_secondary_top_k: int = 5
    tree_tertiary_top_k: int = 4

    # Global atomic rescue prevents taxonomy/node assignment misses.
    global_top_k: int = 8
    global_rewrite_top_k: int = 4

    # FAQ-level retrieval prevents an original complete answer from being lost
    # after atomic decomposition.
    faq_top_k: int = 2
    faq_max_units_per_faq: int = 8

    # Provenance completion: if a high-ranked atomic hit belongs to an original
    # FAQ, retrieve relevant sibling atoms from that FAQ. This is query-agnostic
    # and repairs information lost by atomic decomposition.
    provenance_anchor_top_k: int = 4  # per retrieval channel
    provenance_max_faqs: int = 6
    provenance_max_units_per_faq: int = 10

    # RRF-style channel weights. These are ranking priors, not hard gates.
    tree_channel_weight: float = 1.20
    global_channel_weight: float = 1.00
    faq_channel_weight: float = 1.05
    provenance_channel_weight: float = 0.35
    carryover_channel_weight: float = 0.55

    # Final atomic-evidence context for the Answer LLM.
    context_limit: int = 16
    refinement_context_limit: int = 20

    # ------------------------------------------------------------------
    # Source-level FAQ context (V4.2)
    # ------------------------------------------------------------------
    # Atomic units remain the primary retrieval granularity. For questions
    # whose answer is a set/list, V4.2 may additionally attach one or two
    # original FAQ/source families as a completeness view.
    source_context_top_k: int = 1
    source_min_atomic_hits: int = 2
    source_max_units_per_faq: int = 12

    # ------------------------------------------------------------------
    # Evidence-guided refinement
    # ------------------------------------------------------------------
    # If the first Answer LLM says the database is partially relevant but a
    # specific fact is missing, perform exactly ONE extra local retrieval round.
    enable_refinement: bool = True

    # V4.1: refinement stays inside the already-routed local subtree first.
    # It explores sibling L3 nodes under the same L2 parent instead of
    # re-running lexical routing over the whole taxonomy.
    refinement_parent_limit: int = 2
    refinement_sibling_top_k: int = 12

    # Pseudo-relevance feedback: reuse terminology from the most relevant
    # first-round evidence when constructing the second retrieval query.
    refinement_feedback_top_k: int = 2
    refinement_feedback_answer_chars: int = 220

    # Deprecated compatibility knob from V4.0.x. Kept so older CLI/config
    # overrides do not break, but V4.1 no longer uses whole-tree lexical
    # refinement paths.
    refinement_extra_tree_paths: int = 0

    # ------------------------------------------------------------------
    # Knowledge fallback
    # ------------------------------------------------------------------
    # When database evidence is still insufficient, the model answers from
    # its own knowledge, but the user-facing answer is always explicitly labelled.

    # Allow transparent model-knowledge fallback only after DB retrieval/refinement
    # is still insufficient. The answer must remain explicitly labelled.
    allow_model_knowledge_fallback: bool = True

    # Every non-grounded first pass is logged for later human review / DB growth.
    fallback_queue_path: str = str(
        PROJECT_DIR / "logs" / "knowledge_fallback_queue.jsonl"
    )

    # Pure BM25 by default. Existing retriever remains compatible with embeddings.
    use_embedding: bool = False
    embedding_batch_size: int = 64

    # Per remote API call timeout.
    timeout: int = 60
