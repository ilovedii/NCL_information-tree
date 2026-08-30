import argparse
import csv
import json
import time
from pathlib import Path

from config import AppConfig
from rag import TreeGuidedRAG


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-csv",
        required=True,
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
        default="outputs/v423_first50.csv",
    )

    parser.add_argument(
        "--trace-dir",
        default="traces/v423_first50",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Build RAG once
    # ---------------------------------------------------------
    config = AppConfig()
    rag = TreeGuidedRAG(config)

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
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

    print(
        f"Testing {len(selected)} questions "
        f"from row {args.start + 1}"
    )

    # ---------------------------------------------------------
    # Output columns
    # ---------------------------------------------------------
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

        # -----------------------------------------------------
        # Run questions
        # -----------------------------------------------------
        for i, row in enumerate(
            selected,
            start=1,
        ):
            eval_id = row.get(
                "eval_id",
                f"q{i}",
            )

            question = (
                row.get("題目", "")
                or ""
            ).strip()

            print()
            print("=" * 80)
            print(
                f"[{i}/{len(selected)}] "
                f"{eval_id}"
            )
            print(question)

            start_time = time.perf_counter()

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

            # -------------------------------------------------
            # Success
            # -------------------------------------------------
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
                    out_row[
                        "V4_mode"
                    ],
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

            # -------------------------------------------------
            # Error
            # -------------------------------------------------
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

            # 每題立即保存
            f.flush()

    print()
    print("=" * 80)
    print(
        f"Finished: {output_path}"
    )
    print(
        f"Traces: {trace_dir}"
    )


if __name__ == "__main__":
    main()