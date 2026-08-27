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
    parser = argparse.ArgumentParser(
        description="V3.1 Progressive Tree-Guided Evidence Search"
    )

    parser.add_argument("--csv", default=defaults.csv_path)
    parser.add_argument("--query", default=None)

    parser.add_argument("--ollama-url", default=defaults.ollama_url)
    parser.add_argument("--ncl-api-url", default=defaults.ncl_api_url)
    parser.add_argument(
        "--ncl-verify-ssl",
        type=parse_bool,
        default=defaults.ncl_verify_ssl,
    )

    parser.add_argument("--router-provider", default=defaults.router_provider)
    parser.add_argument("--evidence-provider", default=defaults.evidence_provider)
    parser.add_argument("--answer-provider", default=defaults.answer_provider)

    parser.add_argument("--router-model", default=defaults.router_model)
    parser.add_argument("--evidence-model", default=defaults.evidence_model)
    parser.add_argument("--answer-model", default=defaults.answer_model)
    parser.add_argument("--embedding-model", default=defaults.embedding_model)

    parser.add_argument(
        "--router-think", type=parse_bool, default=defaults.router_think
    )
    parser.add_argument(
        "--evidence-think", type=parse_bool, default=defaults.evidence_think
    )
    parser.add_argument(
        "--answer-think", type=parse_bool, default=defaults.answer_think
    )

    parser.add_argument("--l1-beam", type=int, default=defaults.l1_beam)
    parser.add_argument("--l2-beam", type=int, default=defaults.l2_beam)
    parser.add_argument("--final-beam", type=int, default=defaults.final_beam)

    parser.add_argument(
        "--static-knowledge-dir", default=defaults.static_knowledge_dir
    )
    parser.add_argument(
        "--static-include-topic",
        type=parse_bool,
        default=defaults.static_include_topic,
    )
    parser.add_argument(
        "--static-exact-dedup",
        type=parse_bool,
        default=defaults.static_exact_dedup,
    )

    # Progressive search knobs.
    parser.add_argument(
        "--use-progressive-search",
        type=parse_bool,
        default=defaults.use_progressive_search,
    )
    parser.add_argument(
        "--progressive-batch-size",
        type=int,
        default=defaults.progressive_batch_size,
    )
    parser.add_argument(
        "--use-local-candidate-ordering",
        type=parse_bool,
        default=defaults.use_local_candidate_ordering,
    )
    parser.add_argument(
        "--use-sibling-l3-expansion",
        type=parse_bool,
        default=defaults.use_sibling_l3_expansion,
    )
    parser.add_argument(
        "--sibling-route-discount",
        type=float,
        default=defaults.sibling_route_discount,
    )
    parser.add_argument(
        "--frontier-local-weight",
        type=float,
        default=defaults.frontier_local_weight,
    )
    parser.add_argument(
        "--frontier-route-weight",
        type=float,
        default=defaults.frontier_route_weight,
    )

    parser.add_argument(
        "--evidence-batch-size",
        type=int,
        default=defaults.evidence_batch_size,
    )
    parser.add_argument(
        "--use-sufficiency-check",
        type=parse_bool,
        default=defaults.use_sufficiency_check,
    )
    parser.add_argument(
        "--sufficiency-min-supporting-without-direct",
        type=int,
        default=defaults.sufficiency_min_supporting_without_direct,
    )
    parser.add_argument(
        "--final-include-background",
        type=parse_bool,
        default=defaults.final_include_background,
    )
    parser.add_argument(
        "--final-context-unit-limit",
        type=int,
        default=defaults.final_context_unit_limit,
    )

    parser.add_argument(
        "--use-faq-expansion",
        type=parse_bool,
        default=defaults.use_faq_expansion,
    )
    parser.add_argument(
        "--max-anchor-evidence",
        type=int,
        default=defaults.max_anchor_evidence,
    )
    parser.add_argument(
        "--max-anchor-faqs",
        type=int,
        default=defaults.max_anchor_faqs,
    )
    parser.add_argument(
        "--max-siblings-per-faq",
        type=int,
        default=defaults.max_siblings_per_faq,
    )

    parser.add_argument(
        "--use-fallback-retrieval",
        type=parse_bool,
        default=defaults.use_fallback_retrieval,
    )
    parser.add_argument(
        "--fallback-on-tree-exhausted",
        type=parse_bool,
        default=defaults.fallback_on_tree_exhausted,
    )
    parser.add_argument(
        "--routing-confidence-threshold",
        type=float,
        default=defaults.routing_confidence_threshold,
    )
    parser.add_argument(
        "--fallback-top-k", type=int, default=defaults.fallback_top_k
    )
    parser.add_argument(
        "--use-embedding", type=parse_bool, default=defaults.use_embedding
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=defaults.embedding_batch_size,
    )

    # Backward-compatible options from previous Adaptive releases.
    parser.add_argument(
        "--include-parent-summary",
        type=parse_bool,
        default=defaults.include_parent_summary,
        help="相容舊指令；Progressive 版不再一次載入整個 L2 Parent。",
    )
    parser.add_argument(
        "--parent-expand-on-router-uncertain",
        type=parse_bool,
        default=defaults.parent_expand_on_router_uncertain,
    )
    parser.add_argument(
        "--parent-fallback-direct-score",
        type=float,
        default=defaults.parent_fallback_direct_score,
    )
    parser.add_argument(
        "--parent-fallback-path-threshold",
        type=float,
        default=defaults.parent_fallback_path_threshold,
    )

    parser.add_argument("--timeout", type=int, default=defaults.timeout)

    parser.add_argument("--build-static-knowledge-only", action="store_true")
    parser.add_argument("--force-rebuild-static-knowledge", action="store_true")
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
        ncl_api_url=args.ncl_api_url,
        ncl_verify_ssl=args.ncl_verify_ssl,
        router_provider=args.router_provider,
        evidence_provider=args.evidence_provider,
        answer_provider=args.answer_provider,
        router_model=args.router_model,
        evidence_model=args.evidence_model,
        answer_model=args.answer_model,
        embedding_model=args.embedding_model,
        router_think=args.router_think,
        evidence_think=args.evidence_think,
        answer_think=args.answer_think,
        l1_beam=args.l1_beam,
        l2_beam=args.l2_beam,
        final_beam=args.final_beam,
        static_knowledge_dir=args.static_knowledge_dir,
        static_include_topic=args.static_include_topic,
        static_exact_dedup=args.static_exact_dedup,
        use_progressive_search=args.use_progressive_search,
        progressive_batch_size=args.progressive_batch_size,
        use_local_candidate_ordering=args.use_local_candidate_ordering,
        use_sibling_l3_expansion=args.use_sibling_l3_expansion,
        sibling_route_discount=args.sibling_route_discount,
        frontier_local_weight=args.frontier_local_weight,
        frontier_route_weight=args.frontier_route_weight,
        evidence_batch_size=args.evidence_batch_size,
        use_sufficiency_check=args.use_sufficiency_check,
        sufficiency_min_supporting_without_direct=(
            args.sufficiency_min_supporting_without_direct
        ),
        final_include_background=args.final_include_background,
        final_context_unit_limit=args.final_context_unit_limit,
        use_faq_expansion=args.use_faq_expansion,
        max_anchor_evidence=args.max_anchor_evidence,
        max_anchor_faqs=args.max_anchor_faqs,
        max_siblings_per_faq=args.max_siblings_per_faq,
        use_fallback_retrieval=args.use_fallback_retrieval,
        fallback_on_tree_exhausted=args.fallback_on_tree_exhausted,
        routing_confidence_threshold=args.routing_confidence_threshold,
        fallback_top_k=args.fallback_top_k,
        use_embedding=args.use_embedding,
        embedding_batch_size=args.embedding_batch_size,
        include_parent_summary=args.include_parent_summary,
        parent_expand_on_router_uncertain=args.parent_expand_on_router_uncertain,
        parent_fallback_direct_score=args.parent_fallback_direct_score,
        parent_fallback_path_threshold=args.parent_fallback_path_threshold,
        timeout=args.timeout,
    )


def print_result(result):
    print("\n=== Tree Paths ===")
    for i, path in enumerate(result["paths"], start=1):
        print(f"{i}. {path['path']}  score={path['score']:.4f}")
        for step in path["trace"]:
            print(
                f"   {step['level']}: {step['node']}  "
                f"score={step['score']:.3f}  reason={step['reason']}"
            )

    print("\n=== Static / Opened Node Knowledge ===")
    for i, item in enumerate(result.get("node_knowledge", []), start=1):
        print(
            f"{i}. role={item['role']} path={item['path']} "
            f"documents={item['document_count']} "
            f"units={item['knowledge_unit_count']} "
            f"storage_coverage={item['source_coverage_ratio']:.3f} "
            f"assessed={item.get('assessed_unit_count', 0)} "
            f"assessment_coverage={item.get('assessment_coverage_ratio', 0.0):.3f} "
            f"cache_hit={item['cache_hit']} source={item['static_source']}"
        )

    print("\n=== Progressive Search ===")
    progressive = result.get("progressive_search", {})
    summary = {
        key: value
        for key, value in progressive.items()
        if key != "history"
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in progressive.get("history", []):
        print(
            f"step={item['step']} origin={item['origin']} "
            f"path={item['path']} batch={item['batch_index_for_node']} "
            f"candidates={item['batch_candidate_count']} "
            f"new={item['new_unique_evidence']} "
            f"new_relevant={item['new_relevant_evidence']} "
            f"faq_added={item['faq_added_unique_evidence']} "
            f"sufficient={item['sufficient_after_batch']} "
            f"remaining={item['node_remaining_after_batch']} "
            f"time={item['elapsed_seconds']:.3f}s"
        )
        if item.get("missing_aspects_after_batch"):
            print(
                "   missing="
                + json.dumps(
                    item["missing_aspects_after_batch"],
                    ensure_ascii=False,
                )
            )

    print("\n=== Sibling L3 Expansion ===")
    print(
        json.dumps(
            result.get("sibling_expansion", {}),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== FAQ Expansion ===")
    print(
        json.dumps(
            result.get("faq_expansion", {}),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== Sufficiency ===")
    print(
        json.dumps(
            result.get("sufficiency", {}),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== Evidence Evaluation ===")
    print(
        json.dumps(
            result.get("evidence_evaluation", {}),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== Final Context Evidence ===")
    for i, item in enumerate(result.get("final_context_evidence", []), start=1):
        print(
            f"{i}. {item['evidence_id']} utility={item['utility']} "
            f"score={item['score']:.3f} role={item['role']} path={item['path']}"
        )
        print(f"   {item['content']}")
        print(f"   source_ids={','.join(item['source_ids'])}")

    print("\n=== Timing ===")
    print(
        json.dumps(
            result.get("timing_seconds", {}),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== Answer ===")
    print(result["answer"])


def run_query(rag, query, args):
    force_rebuild = (
        args.force_rebuild_static_knowledge
        or args.force_rebuild_summaries
    )
    result = rag.run(
        query,
        l1_beam=args.l1_beam,
        l2_beam=args.l2_beam,
        final_beam=args.final_beam,
        force_rebuild_knowledge=force_rebuild,
        final_context_unit_limit=args.final_context_unit_limit,
    )
    print_result(result)

    if args.trace_out:
        path = Path(args.trace_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def main():
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    rag = TreeGuidedRAG(config)

    build_static_only = (
        args.build_static_knowledge_only
        or args.build_node_summaries_only
    )
    force_rebuild_static = (
        args.force_rebuild_static_knowledge
        or args.force_rebuild_summaries
    )

    if args.show_stats:
        print(json.dumps(rag.taxonomy.stats(), ensure_ascii=False, indent=2))
        if not args.query and not build_static_only and not args.build_embeddings_only:
            return

    if build_static_only:
        results = rag.build_static_knowledge_packs(
            force_rebuild=force_rebuild_static,
        )
        for item in results:
            print(
                f"[{item['index']}/{item['total']}] {item['path']} "
                f"documents={item['document_count']} "
                f"units={item['knowledge_units']} "
                f"coverage={item['source_coverage_ratio']:.3f} "
                f"cache_hit={item['cache_hit']}"
            )
        print("完成：Static Knowledge 建置不呼叫 LLM/API。")
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

    print("V3.1 Progressive Tree-Guided RAG 互動模式。輸入 exit 結束。")
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