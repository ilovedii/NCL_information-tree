from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class AppConfig:
    csv_path: str = str(PROJECT_DIR / "data" / "llm_retrieval_master.csv")
    ollama_url: str = "http://localhost:11434"
    router_model: str = "qwen3:8b"
    summarizer_model: str = "qwen3:8b"
    evidence_model: str = "qwen3:8b"
    answer_model: str = "qwen3:8b"
    embedding_model: str = "qwen3-embedding:0.6b"
    router_think: bool = True
    summarizer_think: bool = True
    evidence_think: bool = True
    answer_think: bool = True
    l1_beam: int = 2
    l2_beam: int = 3
    final_beam: int = 4
    summary_batch_size: int = 20
    summary_max_units_per_batch: int = 24
    summary_merge_group_size: int = 6
    summary_merge_max_units: int = 60
    summary_cache_dir: str = str(PROJECT_DIR / "cache" / "node_summaries")
    use_summary_cache: bool = True
    evidence_batch_size: int = 40
    final_context_unit_limit: int = 0
    include_parent_summary: bool = True
    # V3.1: provenance-based FAQ family expansion
    use_faq_expansion: bool = True
    max_anchor_evidence: int = 5
    max_anchor_faqs: int = 3
    max_siblings_per_faq: int = 15
    use_fallback_retrieval: bool = True
    routing_confidence_threshold: float = 0.20
    fallback_top_k: int = 12
    use_embedding: bool = True
    embedding_batch_size: int = 64
    timeout: int = 180