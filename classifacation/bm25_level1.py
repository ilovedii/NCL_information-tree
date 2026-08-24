import argparse
import csv
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
from ckip_transformers.nlp import CkipWordSegmenter
from rank_bm25 import BM25Okapi

from ncl_taxonomy_l1 import CATEGORY_NAMES, category_profile_text


OUTPUT_FIELDS = [
    "faq_id",
    "question_date",
    "atomic_id",
    "question",
    "answer",
    "category",
]

TECH_TOKEN_PATTERN = re.compile(
    r"""
    [A-Za-z0-9]+\$[A-Za-z0-9]+
    |
    \$[A-Za-z0-9]+
    |
    [A-Za-z0-9]+(?:[./:_-][A-Za-z0-9]+)+
    |
    [A-Za-z]+\d+
    |
    \d+[A-Za-z]+
    |
    [A-Za-z]+
    |
    \d+(?:\.\d+)?
    """,
    re.VERBOSE,
)


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
    與 Embedding 版本相同的抽樣方式。
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
        return question

    return f"{question}\n{answer}"


def normalize_text(text):
    """
    只做必要 normalization：
    1. Unicode NFKC
    2. 英文字母轉小寫
    不做停用詞移除、不做詞性篩選、不做 stemming。
    """
    text = unicodedata.normalize("NFKC", text)
    return text.lower()


def expand_technical_token(token):

    token = token.lower()
    expanded = [token]

    if "$" in token and not token.startswith("$"):
        left, right = token.split("$", 1)
        if left:
            expanded.append(left)
        if right:
            expanded.append("$" + right)

    parts = re.split(r"[./:_-]+", token)
    if len(parts) > 1:
        expanded.extend(part for part in parts if part)

    # 去重但保留順序
    return list(dict.fromkeys(expanded))


def prepare_for_ckip(text):

    text = normalize_text(text)

    technical_tokens = []
    for match in TECH_TOKEN_PATTERN.finditer(text):
        technical_tokens.extend(expand_technical_token(match.group(0)))

    chinese_part = TECH_TOKEN_PATTERN.sub(" ", text)
    return chinese_part, technical_tokens


def is_content_token(token):
    """
    只排除純標點或純空白
    """
    token = token.strip()
    if not token:
        return False

    return any(ch.isalnum() for ch in token) or "$" in token


def tokenize_texts(texts, ws_driver, batch_size=32):
    """
    中英文混合 tokenizer：
    - 中文：CKIP Word Segmentation
    - 英文/數字/專業代碼：regex 保留
    """
    chinese_parts = []
    technical_token_lists = []

    for text in texts:
        chinese_part, technical_tokens = prepare_for_ckip(text)
        chinese_parts.append(chinese_part)
        technical_token_lists.append(technical_tokens)

    all_tokens = []

    for start in range(0, len(chinese_parts), batch_size):
        batch_chinese = chinese_parts[start:start + batch_size]

        ws_results = ws_driver(
            batch_chinese,
            use_delim=True,
            batch_size=batch_size,
            show_progress=False,
        )

        for offset, ws_tokens in enumerate(ws_results):
            idx = start + offset

            chinese_tokens = [
                token.strip().lower()
                for token in ws_tokens
                if is_content_token(token)
            ]

            tokens = chinese_tokens + technical_token_lists[idx]
            all_tokens.append(tokens)

    return all_tokens


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

        return {
            row["atomic_id"]
            for row in reader
            if row.get("atomic_id")
        }


def choose_ckip_device(device_arg):

    if device_arg == "cpu":
        return -1

    if device_arg == "cuda":
        return 0

    # auto
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="data/atomic_qa_final.csv",
    )

    ap.add_argument(
        "--output",
        default="data/bm25_l1.csv",
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
        "--text-mode",
        choices=["question", "qa"],
        default="qa",
        help="question=只使用問題；qa=使用問題+答案",
    )

    ap.add_argument(
        "--max-answer-chars",
        type=int,
        default=2000,
    )

    ap.add_argument(
        "--ckip-model",
        default="bert-base",
        choices=["albert-tiny", "albert-base", "bert-tiny", "bert-base"],
        help="CKIP Word Segmentation 模型",
    )

    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="CKIP 執行裝置",
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="CKIP 斷詞 batch size",
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

    pending_rows = [
        row for row in rows
        if row["atomic_id"] not in completed_ids
    ]

    print(f"輸入樣本：{len(rows)}")
    print(f"已完成：{len(rows) - len(pending_rows)}")
    print(f"待處理：{len(pending_rows)}")
    print(f"Text mode：{args.text_mode}")

    if not pending_rows:
        print("沒有需要處理的新資料。")
        return

    ckip_device = choose_ckip_device(args.device)
    print(f"CKIP model：{args.ckip_model}")
    print(f"CKIP device：{'cuda:0' if ckip_device == 0 else 'cpu'}")

    ws_driver = CkipWordSegmenter(
        model=args.ckip_model,
        device=ckip_device,
    )

    # ------------------------------------------------------------------
    # 建立 Level 1 taxonomy BM25 corpus
    # ------------------------------------------------------------------
    category_texts = [
        category_profile_text(name)
        for name in CATEGORY_NAMES
    ]

    tokenized_categories = tokenize_texts(
        category_texts,
        ws_driver=ws_driver,
        batch_size=args.batch_size,
    )

    bm25 = BM25Okapi(tokenized_categories)

    # ------------------------------------------------------------------
    # 一次完成待分類 QA 的斷詞，避免每題重複呼叫 CKIP model
    # ------------------------------------------------------------------
    pending_texts = [
        build_text(
            row,
            text_mode=args.text_mode,
            max_answer_chars=args.max_answer_chars,
        )
        for row in pending_rows
    ]

    tokenized_queries = tokenize_texts(
        pending_texts,
        ws_driver=ws_driver,
        batch_size=args.batch_size,
    )

    is_new_file = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )
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

        for i, (row, query_tokens) in enumerate(
            zip(pending_rows, tokenized_queries),
            start=1,
        ):
            scores = bm25.get_scores(query_tokens)

            best_idx = int(np.argmax(scores))
            category = CATEGORY_NAMES[best_idx]

            out = dict(row)
            out["category"] = category

            writer.writerow(out)
            f.flush()

            print(
                f"[{i}/{total_pending}] "
                f"{row['atomic_id']} "
                f"-> {category}"
            )

    print()
    print(f"完成：{args.output}")


if __name__ == "__main__":
    main()