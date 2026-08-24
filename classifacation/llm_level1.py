import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

import requests



from ncl_taxonomy_l1 import CATEGORY_NAMES, category_standard_text


SYSTEM_PROMPT = f"""
請依下列「國家圖書館分類標準」，將每一則 FAQ 歸入唯一一個 Level 1 類別。

判斷原則：
1. 以 question 與 answer 的實際內容判斷主要問題所屬類別。
2. 必須使用下列國家圖書館分類標準，不自行新增、刪除或改寫類別。
3. 每一題只能選擇一個最主要的 Level 1 類別。
4. 若同時涉及多個類別，仍需依問題真正想取得的答案或要完成的工作，選擇最核心的一類。
5. 「其他」只在前七類均不適用，或明確符合「其他」分類標準時使用。
6. category 只能填類別名稱本身，不含編號。
7. 除指定 JSON 外，不輸出其他文字。

{category_standard_text()}
""".strip()


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"faq_id", "question_date", "atomic_id", "question", "answer"}

    if not rows:
        raise ValueError("輸入 CSV 為空")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"缺少欄位：{missing}")

    return rows


def sample_rows(rows, sample_size, seed):
    if sample_size <= 0 or sample_size >= len(rows):
        return rows

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), sample_size))

    return [rows[i] for i in idx]


def make_prompt(row):
    answer = (row.get("answer") or "").strip()

    if len(answer) > 2000:
        answer = answer[:2000] + "……"

    return f"""
【FAQ】
atomic_id: {row['atomic_id']}
question: {row['question']}
answer: {answer}

請輸出：
{{
  "category": "8個類別之一"
}}
""".strip()



CATEGORY_ALIASES = {
    "作者號及輔助分號": "作者號及輔助區分號",
}


def normalize_category_name(value):

    if value is None:
        return None

    value = str(value).strip()

    value = re.sub(
        r"^\s*[（(]?\s*\d+\s*[）)]?\s*[\.．、:：-]?\s*",
        "",
        value,
    ).strip()

    return CATEGORY_ALIASES.get(value, value)


def parse_json(text):
    text = (text or "").strip()

    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"找不到 JSON：{text[:300]}")
        obj = json.loads(match.group(0))

    category = normalize_category_name(
        obj.get("category") or ""
    )

    if category not in CATEGORY_NAMES:
        raise ValueError(
            f"非法 category：{category}"
        )

    return {
        "llm_category": category,
    }



def call_ollama(prompt, model, base_url):
    url = base_url.rstrip("/") + "/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0
        },
    }

    r = requests.post(
        url,
        json=payload,
        timeout=180,
    )

    r.raise_for_status()

    return r.json()["message"]["content"]


def call_openai_compatible(
    prompt,
    model,
    base_url,
    api_key,
):
    url = base_url.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    r = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=180,
    )

    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]


OUTPUT_FIELDS = [
    "faq_id",
    "question_date",
    "atomic_id",
    "question",
    "answer",
    "llm_category",
]



def load_completed_ids(path):
    """讀取既有輸出中的 atomic_id，供續跑時跳過。"""
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


def count_output_rows(path):
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return 0

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return sum(1 for _ in csv.DictReader(f))



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="data/atomic_qa_final.csv",
    )

    ap.add_argument(
        "--output",
        default="data/llm_test.csv",
    )

    # 0 = 全部
    ap.add_argument(
        "--sample-size",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--provider",
        choices=[
            "ollama",
            "openai_compatible",
        ],
        default="ollama",
    )

    ap.add_argument(
        "--model",
        default="qwen3:8b",
    )

    ap.add_argument(
        "--base-url",
        default=None,
    )

    ap.add_argument(
        "--api-key",
        default=os.getenv(
            "OPENAI_API_KEY",
            "",
        ),
    )

    ap.add_argument(
        "--retries",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="忽略既有輸出，重新從頭建立 CSV。",
    )

    args = ap.parse_args()

    rows = load_csv(args.input)

    rows = sample_rows(
        rows,
        args.sample_size,
        args.seed,
    )

    if args.provider == "ollama":
        base_url = (
            args.base_url
            or "http://127.0.0.1:11434"
        )
    else:
        base_url = (
            args.base_url
            or "http://127.0.0.1:8000/v1"
        )

    output_path = Path(args.output)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed_ids = load_completed_ids(output_path)

    if completed_ids:
        print(
            f"偵測到既有輸出：{args.output}，"
            f"已完成 {len(completed_ids)} 筆，將自動續跑。"
        )

    is_new_file = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )
    mode = "w" if is_new_file else "a"

    processed_this_run = 0
    skipped = 0

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

        for i, row in enumerate(rows, start=1):
            if row["atomic_id"] in completed_ids:
                skipped += 1
                print(
                    f"[{i}/{len(rows)}] "
                    f"{row['atomic_id']} -> SKIP (already completed)"
                )
                continue

            prompt = make_prompt(row)
            result = None
            last_error = None

            for attempt in range(1, args.retries + 1):
                try:
                    if args.provider == "ollama":
                        text_result = call_ollama(
                            prompt,
                            args.model,
                            base_url,
                        )
                    else:
                        text_result = call_openai_compatible(
                            prompt,
                            args.model,
                            base_url,
                            args.api_key,
                        )

                    result = parse_json(text_result)
                    break

                except Exception as e:
                    last_error = e
                    print(
                        f"[{i}/{len(rows)}] "
                        f"{row['atomic_id']} "
                        f"attempt {attempt} failed: {e}"
                    )
                    time.sleep(min(2 ** attempt, 10))

            if result is None:
                result = {
                    "llm_category": "",
                }

            out = dict(row)
            out.update(result)

            # 每完成一題就立即寫入 CSV
            writer.writerow(out)
            # 強制 flush，程式中斷時已完成的資料仍會保留
            f.flush()

            processed_this_run += 1
            completed_ids.add(row["atomic_id"])

            print(
                f"[{i}/{len(rows)}] "
                f"{row['atomic_id']} "
                f"-> {result['llm_category']}"
            )

    total_done = count_output_rows(output_path)

    print()
    print(f"完成：{args.output}")
    print(f"本次新增：{processed_this_run}")
    print(f"本次跳過：{skipped}")
    print(f"輸出目前總筆數：{total_done}")


if __name__ == "__main__":
    main()