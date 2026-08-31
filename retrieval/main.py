import argparse
import csv
import json
import time
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

    raise argparse.ArgumentTypeError(
        f"無法解析布林值：{value}"
    )


def build_arg_parser():
    defaults = AppConfig()

    parser = argparse.ArgumentParser(
        description="NCL Tree-Guided RAG"
    )

    # =========================================================
    # Single-query mode
    # =========================================================
    parser.add_argument(
        "--query",
        default=None,
    )

    parser.add_argument(
        "--trace-out",
        default=None,
    )

    # =========================================================
    # Batch evaluation mode
    # =========================================================
    parser.add_argument(
        "--test-csv",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--out",
        default="outputs/first50.csv",
    )

    parser.add_argument(
        "--trace-dir",
        default="traces/first50",
    )

    # =========================================================
    # RAG config
    # =========================================================
    parser.add_argument(
        "--csv",
        default=defaults.csv_path,
    )

    parser.add_argument(
        "--ollama-url",
        default=defaults.ollama_url,
    )

    parser.add_argument(
        "--ncl-api-url",
        default=defaults.ncl_api_url,
    )

    parser.add_argument(
        "--ncl-verify-ssl",
        type=parse_bool,
        default=defaults.ncl_verify_ssl,
    )

    parser.add_argument(
        "--router-provider",
        default=defaults.router_provider,
    )

    parser.add_argument(
        "--answer-provider",
        default=defaults.answer_provider,
    )

    parser.add_argument(
        "--router-model",
        default=defaults.router_model,
    )

    parser.add_argument(
        "--answer-model",
        default=defaults.answer_model,
    )

    parser.add_argument(
        "--router-think",
        type=parse_bool,
        default=defaults.router_think,
    )

    parser.add_argument(
        "--answer-think",
        type=parse_bool,
        default=defaults.answer_think,
    )

    parser.add_argument(
        "--tree-candidate-pool",
        type=int,
        default=defaults.tree_candidate_pool,
    )

    parser.add_argument(
        "--router-top-paths",
        type=int,
        default=defaults.router_top_paths,
    )

    parser.add_argument(
        "--global-top-k",
        type=int,
        default=defaults.global_top_k,
    )

    parser.add_argument(
        "--faq-top-k",
        type=int,
        default=defaults.faq_top_k,
    )

    parser.add_argument(
        "--context-limit",
        type=int,
        default=defaults.context_limit,
    )

    parser.add_argument(
        "--enable-refinement",
        type=parse_bool,
        default=defaults.enable_refinement,
    )

    parser.add_argument(
        "--fallback-queue-path",
        default=defaults.fallback_queue_path,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=defaults.timeout,
    )

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


# =============================================================
# Single query
# =============================================================
def run_query(rag, query, args):
    result = rag.run(query)

    print(result["answer"])

    if args.trace_out:
        path = Path(args.trace_out)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return result


# =============================================================
# Batch evaluation
# =============================================================
def run_batch(rag, args):
    test_csv = Path(args.test_csv)
    output_path = Path(args.out)
    trace_dir = Path(args.trace_dir)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Read test questions
    # ---------------------------------------------------------
    with test_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    selected = rows[
        args.start:
        args.start + args.limit
    ]

    if not selected:
        print(
            f"No questions selected. "
            f"start={args.start}, "
            f"limit={args.limit}, "
            f"total_rows={len(rows)}"
        )
        return

    print(
        f"Testing {len(selected)} questions "
        f"from row {args.start + 1}"
    )

    fieldnames = list(
        selected[0].keys()
    ) + [
        "V4_answer",
        "V4_mode",
        "V4_knowledge_used",
        "V4_answer_scope",
        "V4_refinement_used",
        "V4_evidence_ids",
        "V4_total_seconds",
        "V4_trace",
        "V4_error",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for local_i, row in enumerate(
            selected,
            start=1,
        ):
            absolute_i = (
                args.start
                + local_i
            )

            eval_id = (
                row.get("eval_id")
                or f"q{absolute_i}"
            )

            question = (
                row.get("題目", "")
                or ""
            ).strip()

            print()
            print("=" * 80)

            print(
                f"[{local_i}/{len(selected)}] "
                f"{eval_id}"
            )

            print(question)

            start_time = (
                time.perf_counter()
            )

            result = None
            error = ""

            try:
                result = rag.run(
                    question
                )

            except Exception as exc:
                error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            out_row = dict(row)

            # =================================================
            # Success
            # =================================================
            if result is not None:

                trace_path = (
                    trace_dir
                    / f"{eval_id}.json"
                )

                trace_data = dict(
                    result
                )

                trace_data["_eval"] = {
                    "eval_id": eval_id,
                    "question": question,
                    "gold": row.get(
                        "標準答案gold",
                        "",
                    ),
                    "question_type": row.get(
                        "question_type",
                        "",
                    ),
                    "answerability": row.get(
                        "answerability",
                        "",
                    ),
                }

                trace_path.write_text(
                    json.dumps(
                        trace_data,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                final_decision = (
                    result.get(
                        "final_decision"
                    )
                    or {}
                )

                evidence_ids = (
                    final_decision.get(
                        "evidence_ids"
                    )
                    or []
                )

                if not evidence_ids:

                    initial_decision = (
                        result.get(
                            "initial_decision"
                        )
                        or {}
                    )

                    evidence_ids = (
                        initial_decision.get(
                            "evidence_ids"
                        )
                        or []
                    )

                timing = (
                    result.get("timing")
                    or {}
                )

                total_seconds = (
                    timing.get(
                        "total",
                        elapsed,
                    )
                )

                out_row[
                    "V4_answer"
                ] = result.get(
                    "answer",
                    "",
                )

                out_row[
                    "V4_mode"
                ] = result.get(
                    "mode",
                    "",
                )

                out_row[
                    "V4_knowledge_used"
                ] = result.get(
                    "knowledge_used",
                    False,
                )

                out_row[
                    "V4_answer_scope"
                ] = result.get(
                    "answer_scope",
                    "",
                )

                out_row[
                    "V4_refinement_used"
                ] = result.get(
                    "refinement_used",
                    False,
                )

                out_row[
                    "V4_evidence_ids"
                ] = "|".join(
                    evidence_ids
                )

                out_row[
                    "V4_total_seconds"
                ] = round(
                    float(total_seconds),
                    4,
                )

                out_row[
                    "V4_trace"
                ] = str(
                    trace_path
                )

                out_row[
                    "V4_error"
                ] = ""

                print(
                    "mode:",
                    out_row["V4_mode"],
                )

                print(
                    "knowledge:",
                    out_row[
                        "V4_knowledge_used"
                    ],
                )

                print(
                    "time:",
                    out_row[
                        "V4_total_seconds"
                    ],
                )

                print(
                    "answer:",
                    out_row[
                        "V4_answer"
                    ][:400],
                )

            # =================================================
            # Error
            # =================================================
            else:

                out_row[
                    "V4_answer"
                ] = ""

                out_row[
                    "V4_mode"
                ] = "ERROR"

                out_row[
                    "V4_knowledge_used"
                ] = ""

                out_row[
                    "V4_answer_scope"
                ] = ""

                out_row[
                    "V4_refinement_used"
                ] = ""

                out_row[
                    "V4_evidence_ids"
                ] = ""

                out_row[
                    "V4_total_seconds"
                ] = round(
                    elapsed,
                    4,
                )

                out_row[
                    "V4_trace"
                ] = ""

                out_row[
                    "V4_error"
                ] = error

                print(
                    "ERROR:",
                    error,
                )

            writer.writerow(
                out_row
            )

            # 每題立即存檔
            f.flush()

    print()
    print("=" * 80)

    print(
        f"Finished: {output_path}"
    )

    print(
        f"Traces: {trace_dir}"
    )


# =============================================================
# Main
# =============================================================
def main():
    args = (
        build_arg_parser()
        .parse_args()
    )

    config = config_from_args(
        args
    )

    rag = TreeGuidedRAG(
        config
    )

    # ---------------------------------------------------------
    # Batch mode
    # ---------------------------------------------------------
    if args.test_csv:
        run_batch(
            rag,
            args,
        )
        return

    # ---------------------------------------------------------
    # Single query mode
    # ---------------------------------------------------------
    if args.query:
        run_query(
            rag,
            args.query,
            args,
        )
        return

    # ---------------------------------------------------------
    # Interactive mode
    # ---------------------------------------------------------
    print(
        "NCL Tree-RAG 互動模式。"
        "輸入 exit 結束。"
    )

    while True:

        try:
            query = input(
                "\nQuery> "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print()
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        run_query(
            rag,
            query,
            args,
        )


if __name__ == "__main__":
    main()