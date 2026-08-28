import argparse
import json
from dataclasses import replace
from pathlib import Path

from config import AppConfig
from rag import TreeGuidedRAG


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"無法解析布林值：{value}")


def build_arg_parser():
    defaults = AppConfig()
    parser = argparse.ArgumentParser(description="V4 Simple Tree-RAG")
    parser.add_argument("--query", default=None)
    parser.add_argument("--csv", default=defaults.csv_path)

    parser.add_argument("--ollama-url", default=defaults.ollama_url)
    parser.add_argument("--ncl-api-url", default=defaults.ncl_api_url)
    parser.add_argument("--ncl-verify-ssl", type=parse_bool, default=defaults.ncl_verify_ssl)
    parser.add_argument("--router-provider", default=defaults.router_provider)
    parser.add_argument("--answer-provider", default=defaults.answer_provider)
    parser.add_argument("--router-model", default=defaults.router_model)
    parser.add_argument("--answer-model", default=defaults.answer_model)
    parser.add_argument("--router-think", type=parse_bool, default=defaults.router_think)
    parser.add_argument("--answer-think", type=parse_bool, default=defaults.answer_think)

    parser.add_argument("--tree-candidate-pool", type=int, default=defaults.tree_candidate_pool)
    parser.add_argument("--router-top-paths", type=int, default=defaults.router_top_paths)
    parser.add_argument("--global-top-k", type=int, default=defaults.global_top_k)
    parser.add_argument("--faq-top-k", type=int, default=defaults.faq_top_k)
    parser.add_argument("--context-limit", type=int, default=defaults.context_limit)
    parser.add_argument("--enable-refinement", type=parse_bool, default=defaults.enable_refinement)
    parser.add_argument("--fallback-queue-path", default=defaults.fallback_queue_path)
    parser.add_argument("--timeout", type=int, default=defaults.timeout)

    # Optional debug artifact. Normal stdout still contains answer only.
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
        answer_provider=args.answer_provider,
        router_model=args.router_model,
        answer_model=args.answer_model,
        router_think=args.router_think,
        answer_think=args.answer_think,
        tree_candidate_pool=args.tree_candidate_pool,
        router_top_paths=args.router_top_paths,
        global_top_k=args.global_top_k,
        faq_top_k=args.faq_top_k,
        context_limit=args.context_limit,
        enable_refinement=args.enable_refinement,
        fallback_queue_path=args.fallback_queue_path,
        timeout=args.timeout,
    )


def run_query(rag, query, args):
    result = rag.run(query)
    print(result["answer"])

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

    if args.query:
        run_query(rag, args.query, args)
        return

    print("V4 Simple Tree-RAG 互動模式。輸入 exit 結束。")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break
        run_query(rag, query, args)


if __name__ == "__main__":
    main()
