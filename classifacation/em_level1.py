import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


from ncl_taxonomy_l1 import CATEGORY_NAMES, category_profile_text


OUTPUT_FIELDS = [
    "faq_id",
    "question_date",
    "atomic_id",
    "question",
    "answer",
    "category",
]


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "faq_id",
        "question_date",
        "atomic_id",
        "question",
        "answer",
    }

    if not rows:
        raise ValueError("輸入 CSV 為空")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"缺少欄位：{missing}")

    return rows


def sample_rows(rows, sample_size, seed):
    """
    與 LLM 版本相同的抽樣方式。
    只要 input、sample_size、seed 相同，就會抽到同一批資料。
    """
    if sample_size <= 0 or sample_size >= len(rows):
        return rows

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[i] for i in idx]


def build_text(row, text_mode="qa", max_answer_chars=2000):
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()

    if len(answer) > max_answer_chars:
        answer = answer[:max_answer_chars] + "……"

    if text_mode == "question":
        return f"query: 問題：{question}"

    return f"query: 問題：{question}\n答案：{answer}"


def choose_device(device_arg):
    if device_arg != "auto":
        return device_arg

    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_completed_ids(path):
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return set()

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != OUTPUT_FIELDS:
            raise ValueError(
                "既有輸出 CSV 欄位與目前程式不一致。"
                "請改用新的 output 檔名，或加 --overwrite 重新建立。"
            )

        return {row["atomic_id"] for row in reader if row.get("atomic_id")}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="data/atomic_qa_final.csv",
    )

    ap.add_argument(
        "--output",
        default="data/em_level1.csv",
    )

    ap.add_argument(
        "--sample-size",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--model",
        default="intfloat/multilingual-e5-large",
        help="SentenceTransformer",
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
    )

    ap.add_argument(
        "--text-mode",
        choices=["question", "qa"],
        default="qa",
        help="question=只嵌入問題；qa=嵌入問題+答案",
    )

    ap.add_argument(
        "--max-answer-chars",
        type=int,
        default=2000,
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="忽略既有輸出，重新從頭建立 CSV。",
    )

    args = ap.parse_args()

    rows = load_csv(args.input)
    rows = sample_rows(rows, args.sample_size, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed_ids = load_completed_ids(output_path)

    pending_rows = [row for row in rows if row["atomic_id"] not in completed_ids]

    print(f"輸入樣本：{len(rows)}")
    print(f"已完成：{len(rows) - len(pending_rows)}")
    print(f"待處理：{len(pending_rows)}")

    if not pending_rows:
        print("沒有需要處理的新資料。")
        return

    device = choose_device(args.device)
    print(f"Embedding model：{args.model}")
    print(f"Device：{device}")
    print(f"Text mode：{args.text_mode}")

    model = SentenceTransformer(
        args.model,
        device=device,
    )

    # Level 1 類別向量：
    # 不再只嵌入短類別名稱，而是使用國圖完整分類標準作為 category prototype。
    # 對 multilingual-e5-large，候選類別文字使用 passage: 前綴。
    category_texts = [
        "passage: " + category_profile_text(name) for name in CATEGORY_NAMES
    ]

    category_embeddings = model.encode(
        category_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    is_new_file = not output_path.exists() or output_path.stat().st_size == 0
    mode = "w" if is_new_file else "a"

    with open(
        output_path,
        mode,
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_FIELDS,
            extrasaction="ignore",
        )

        if is_new_file:
            writer.writeheader()
            f.flush()

        total_pending = len(pending_rows)

        for batch_start in range(0, total_pending, args.batch_size):
            batch_rows = pending_rows[batch_start : batch_start + args.batch_size]

            texts = [
                build_text(
                    row,
                    text_mode=args.text_mode,
                    max_answer_chars=args.max_answer_chars,
                )
                for row in batch_rows
            ]

            qa_embeddings = model.encode(
                texts,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # normalized vector 的 dot product 等價於 cosine similarity
            scores = qa_embeddings @ category_embeddings.T

            for j, row in enumerate(batch_rows):
                best_idx = int(np.argmax(scores[j]))

                out = dict(row)
                out["category"] = CATEGORY_NAMES[best_idx]

                writer.writerow(out)
                f.flush()

                current = batch_start + j + 1

                print(
                    f"[{current}/{total_pending}] "
                    f"{row['atomic_id']} "
                    f"-> {CATEGORY_NAMES[best_idx]}"
                )

    print()
    print(f"完成：{args.output}")


if __name__ == "__main__":
    main()
