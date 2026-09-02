from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class AppConfig:


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

    tree_candidate_pool: int = 16
    router_top_paths: int = 3

    tree_primary_top_k: int = 6
    tree_secondary_top_k: int = 5
    tree_tertiary_top_k: int = 4

    # Global atomic rescue prevents taxonomy/node assignment misses.
    global_top_k: int = 8
    global_rewrite_top_k: int = 4

    unit_global_top_k: int = 6
    unit_keyword_top_k: int = 6

    unit_context_top_k: int = 4

    faq_top_k: int = 2
    faq_max_units_per_faq: int = 8

    provenance_anchor_top_k: int = 4  # per retrieval channel
    provenance_max_faqs: int = 6
    provenance_max_units_per_faq: int = 10

    tree_channel_weight: float = 1.20
    global_channel_weight: float = 1.00
    faq_channel_weight: float = 1.05
    provenance_channel_weight: float = 0.35
    carryover_channel_weight: float = 0.55

    context_limit: int = 16
    refinement_context_limit: int = 20


    source_context_top_k: int = 1
    source_min_atomic_hits: int = 2
    source_max_units_per_faq: int = 12

    enable_refinement: bool = True

 
    refinement_parent_limit: int = 2
    refinement_sibling_top_k: int = 12

    refinement_feedback_top_k: int = 2
    refinement_feedback_answer_chars: int = 220

    refinement_extra_tree_paths: int = 0

    allow_model_knowledge_fallback: bool = True

    fallback_queue_path: str = str(
        PROJECT_DIR / "logs" / "knowledge_fallback_queue.jsonl"
    )

    use_embedding: bool = False
    embedding_batch_size: int = 64

    timeout: int = 60

    use_answer_embedding: bool = True

    answer_embedding_model: str = "intfloat/multilingual-e5-large"

    answer_embedding_path: str = str(
        PROJECT_DIR / "embeddings" / "answer_embeddings.npy"
    )

    answer_embedding_top_k: int = 5
