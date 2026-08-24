import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from config import AppConfig
from rag import TreeGuidedRAG


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("請輸入 true 或 false")


def build_arg_parser():
    defaults = AppConfig()
    parser = argparse.ArgumentParser(description="Tree-Guided Hierarchical RAG with Ollama")
    parser.add_argument("--csv", default=defaults.csv_path)
    parser.add_argument("--query", default=None)
    parser.add_argument("--ollama-url", default=defaults.ollama_url)
    parser.add_argument("--router-model", default=defaults.router_model)
    parser.add_argument("--summarizer-model", default=defaults.summarizer_model)
    parser.add_argument("--evidence-model", default=defaults.evidence_model)
    parser.add_argument("--answer-model", default=defaults.answer_model)
    parser.add_argument("--embedding-model", default=defaults.embedding_model)
    parser.add_argument("--router-think", type=parse_bool, default=defaults.router_think)
    parser.add_argument("--summarizer-think", type=parse_bool, default=defaults.summarizer_think)
    parser.add_argument("--evidence-think", type=parse_bool, default=defaults.evidence_think)
    parser.add_argument("--answer-think", type=parse_bool, default=defaults.answer_think)
    parser.add_argument("--l1-beam", type=int, default=defaults.l1_beam)
    parser.add_argument("--l2-beam", type=int, default=defaults.l2_beam)
    parser.add_argument("--final-beam", type=int, default=defaults.final_beam)
    parser.add_argument("--summary-batch-size", type=int, default=defaults.summary_batch_size)
    parser.add_argument("--summary-max-units-per-batch", type=int, default=defaults.summary_max_units_per_batch)
    parser.add_argument("--summary-merge-group-size", type=int, default=defaults.summary_merge_group_size)
    parser.add_argument("--summary-merge-max-units", type=int, default=defaults.summary_merge_max_units)
    parser.add_argument("--summary-cache-dir", default=defaults.summary_cache_dir)
    parser.add_argument("--use-summary-cache", type=parse_bool, default=defaults.use_summary_cache)
    parser.add_argument("--evidence-batch-size", type=int, default=defaults.evidence_batch_size)
    parser.add_argument("--final-context-unit-limit", type=int, default=defaults.final_context_unit_limit)
    parser.add_argument("--include-parent-summary", type=parse_bool, default=defaults.include_parent_summary)
    parser.add_argument("--use-fallback-retrieval", type=parse_bool, default=defaults.use_fallback_retrieval)
    parser.add_argument("--routing-confidence-threshold", type=float, default=defaults.routing_confidence_threshold)
    parser.add_argument("--fallback-top-k", type=int, default=defaults.fallback_top_k)
    parser.add_argument("--use-embedding", type=parse_bool, default=defaults.use_embedding)
    parser.add_argument("--embedding-batch-size", type=int, default=defaults.embedding_batch_size)
    parser.add_argument("--timeout", type=int, default=defaults.timeout)
    parser.add_argument("--build-node-summaries-only", action="store_true")
    parser.add_argument("--force-rebuild-summaries", action="store_true")
    parser.add_argument("--build-embeddings-only", action="store_true")
    parser.add_argument("--show-stats", action="store_true")
    parser.add_argument("--trace-out", default=None)
    return parser


def config_from_args(args):
    return replace(
        AppConfig(),
        csv_path=args.csv,
        ollama_url=args.ollama_url,
        router_model=args.router_model,
        summarizer_model=args.summarizer_model,
        evidence_model=args.evidence_model,
        answer_model=args.answer_model,
        embedding_model=args.embedding_model,
        router_think=args.router_think,
        summarizer_think=args.summarizer_think,
        evidence_think=args.evidence_think,
        answer_think=args.answer_think,
        l1_beam=args.l1_beam,
        l2_beam=args.l2_beam,
        final_beam=args.final_beam,
        summary_batch_size=args.summary_batch_size,
        summary_max_units_per_batch=args.summary_max_units_per_batch,
        summary_merge_group_size=args.summary_merge_group_size,
        summary_merge_max_units=args.summary_merge_max_units,
        summary_cache_dir=args.summary_cache_dir,
        use_summary_cache=args.use_summary_cache,
        evidence_batch_size=args.evidence_batch_size,
        final_context_unit_limit=args.final_context_unit_limit,
        include_parent_summary=args.include_parent_summary,
        use_fallback_retrieval=args.use_fallback_retrieval,
        routing_confidence_threshold=args.routing_confidence_threshold,
        fallback_top_k=args.fallback_top_k,
        use_embedding=args.use_embedding,
        embedding_batch_size=args.embedding_batch_size,
        timeout=args.timeout,
    )


def print_result(result):
    print("\n=== Tree Paths ===")
    for i, path in enumerate(result["paths"], start=1):
        print(f"{i}. {path['path']}  score={path['score']:.4f}")
        for step in path["trace"]:
            print(
                f"   {step['level']}: {step['node']}  score={step['score']:.3f}  reason={step['reason']}"
            )
    print("\n=== Node Summaries ===")
    for i, item in enumerate(result["node_summaries"], start=1):
        print(
            f"{i}. role={item['role']} path={item['path']} documents={item['document_count']} units={item['knowledge_unit_count']} coverage={item['source_coverage_ratio']:.3f} cache_hit={item['cache_hit']}"
        )
        print(f"   date={item['date_start']} ~ {item['date_end']}")
        print(f"   note={item['coverage_note']}")
    print("\n=== Evidence Priority ===")
    for i, item in enumerate(result["evidence"], start=1):
        print(
            f"{i}. {item['evidence_id']} utility={item['utility']} score={item['score']:.3f} role={item['role']} path={item['path']}"
        )
        print(f"   {item['content']}")
        print(f"   source_ids={','.join(item['source_ids'])}")
    print("\n=== Answer ===")
    print(result["answer"])


def run_query(rag, query, args):
    result = rag.run(
        query,
        l1_beam=args.l1_beam,
        l2_beam=args.l2_beam,
        final_beam=args.final_beam,
        force_rebuild_summaries=args.force_rebuild_summaries,
        final_context_unit_limit=args.final_context_unit_limit,
    )
    print_result(result)
    if args.trace_out:
        Path(args.trace_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def main():
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    rag = TreeGuidedRAG(config)
    if args.show_stats:
        print(json.dumps(rag.taxonomy.stats(), ensure_ascii=False, indent=2))
        if not args.query and not args.build_node_summaries_only and not args.build_embeddings_only:
            return
    if args.build_node_summaries_only:
        results = rag.build_node_summaries(force_rebuild=args.force_rebuild_summaries)
        for item in results:
            print(
                f"[{item['index']}/{item['total']}] {item['path']} documents={item['document_count']} units={item['knowledge_units']} coverage={item['source_coverage_ratio']:.3f} cache_hit={item['cache_hit']}"
            )
        return
    if args.build_embeddings_only:
        if not config.use_embedding:
            raise ValueError("--build-embeddings-only 需要 --use-embedding true")
        rag.retriever.ensure_embeddings()
        print(f"Fallback embedding index 已建立：{rag.retriever.cache_path}")
        return
    if args.query:
        run_query(rag, args.query, args)
        return
    print("Tree-Guided Hierarchical RAG 互動模式。輸入 exit 結束。")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue
        try:
            run_query(rag, query, args)
        except Exception as exc:
            print(f"執行失敗：{exc}", file=sys.stderr)


if __name__ == "__main__":
    main()