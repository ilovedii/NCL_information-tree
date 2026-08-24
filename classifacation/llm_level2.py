import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import requests

from ncl_taxonomy_l2 import (
    L2_MAX_LABELS,
    l2_category_names,
    l2_standard_text,
)


PARENT_L1 = "權威控制"
L2_NAMES = l2_category_names(PARENT_L1)
L2_ID_TO_NAME = {
    i: name for i, name in enumerate(L2_NAMES, start=1)
}


SYSTEM_PROMPT = f"""
你正在進行國家圖書館 FAQ 的階層式分類。

本次資料已經確定：
L1 = {PARENT_L1}

你的工作只是在此 L1 底下選擇 Level 2 類別。

{l2_standard_text(PARENT_L1)}

【重要判斷規則】

1. 每題至少選 1 個、最多選 {L2_MAX_LABELS} 個 L2。
2. labels 依重要性排序，第 1 個必須是最主要的 L2。
3. 不得為了湊數而多選；單一知識面向就只選 1 個。
4. 優先選最具體的類別，不要同時標記「泛類」與「專類」。

特別注意以下邊界：

A. 「名稱權威」處理：
   個人名稱、團體名稱、會議名稱等名稱實體的權威控制，
   包括首選名稱、標準形式、同名辨識與名稱選擇。
   若問題核心是為索書號產生作者號、著者號或 Cutter number，
   應屬 L1「作者號及輔助區分號」；
   若只是 MARC Authority 欄位、指標、分欄或代碼結構，
   應屬 L1「機讀編目格式」。

B. 「題名與作品權威」處理：
   劃一題名、作品題名、叢書題名，以及其他題名型權威標目的
   建立、選擇與區別。
   若只是書目紀錄中的一般題名著錄，應屬 L1「編目法」；
   本類只處理「題名作為權威實體」時的控制問題。

C. 「權威標目選擇與形式」處理：
   權威標目或首選檢索點應採何種形式，
   包括名稱、日期、附加識別資訊及其他標目形式的規範化。
   若核心是個人／團體／會議「哪個名稱應作首選名稱」，
   優先歸「名稱權威」；
   若重點是標目的結構、日期或附加識別資訊如何規範化，歸本類。

D. 「異名與參照關係」處理：
   異名、別名、筆名、譯名、不同語文形式，以及
   見／見自等參照關係的建立與維護。
   若問題是主題詞彙中的 USE／UF／BT／NT／RT 關係，
   應屬 L1「主題法」；
   本類處理名稱或權威實體之間的異名與導引關係。

E. 「權威紀錄、權威檔與鏈結資料」處理：
   權威紀錄的建立、維護、合併，
   權威檔或權威資料庫，以及 URI、識別碼與鏈結式權威資料。
   若題目只問 Authority Format 的欄位、分欄、指標或代碼，
   應屬 L1「機讀編目格式」；
   本類重點是權威資料本身的管理、控制、服務與鏈結。

請只輸出 JSON，不要解釋。

可用 L2 代碼：
{chr(10).join(f"{i} = {name}" for i, name in L2_ID_TO_NAME.items())}

輸出格式：
{{
  "labels": [2]
}}

或真的需要多標時：
{{
  "labels": [3, 5]
}}
""".strip()


OUTPUT_FIELDS = [
    "faq_id",
    "question_date",
    "atomic_id",
    "question",
    "answer",
    "l1_category",
    "l2_category_1",
    "l2_category_2",
    "l2_category_3",
    "l2_count",
]


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("輸入 CSV 為空")

    required = {
        "faq_id",
        "question_date",
        "atomic_id",
        "question",
        "answer",
        "llm_category",
    }

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"缺少欄位：{missing}")

    return rows


def make_prompt(row):
    answer = (row.get("answer") or "").strip()

    if len(answer) > 2500:
        answer = answer[:2500] + "……"

    return f"""
【FAQ】
atomic_id: {row['atomic_id']}
question: {row['question']}
answer: {answer}

請依 system 中的「{PARENT_L1} Level 2 分類標準」判斷，
只輸出 labels JSON。
""".strip()


def parse_json(text):
    text = (text or "").strip()

    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"找不到 JSON：{text[:300]}")
        obj = json.loads(match.group(0))

    labels = obj.get("labels")

    if isinstance(labels, int):
        labels = [labels]

    if not isinstance(labels, list):
        raise ValueError(f"labels 不是 list：{labels}")

    normalized = []

    for value in labels:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"非法 L2 代碼：{value}")

        if value not in L2_ID_TO_NAME:
            raise ValueError(f"L2 代碼超出範圍：{value}")

        if value not in normalized:
            normalized.append(value)

    if not normalized:
        raise ValueError("labels 不可為空")

    if len(normalized) > L2_MAX_LABELS:
        raise ValueError(
            f"labels 超過 {L2_MAX_LABELS} 個：{normalized}"
        )

    return [L2_ID_TO_NAME[x] for x in normalized]


def call_ollama(prompt, model, base_url, timeout):
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
            "temperature": 0,
        },
    }

    r = requests.post(
        url,
        json=payload,
        timeout=timeout,
    )

    r.raise_for_status()

    return r.json()["message"]["content"]


def call_openai_compatible(
    prompt,
    model,
    base_url,
    api_key,
    timeout,
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
        timeout=timeout,
    )

    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]


def load_completed_ids(path):
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return set()

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != OUTPUT_FIELDS:
            raise ValueError(
                "既有輸出 CSV 欄位與目前程式不一致。"
                "請改用新的 output 檔名，或加 --overwrite。"
            )

        return {
            row["atomic_id"]
            for row in reader
            if row.get("atomic_id")
            and row.get("l2_category_1")
        }


def count_output_rows(path):
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return 0

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="data/llm_level1.csv",
        help="已完成的 LLM Level 1 結果。",
    )

    ap.add_argument(
        "--output",
        default="data/llm_level2-8.csv",
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
        "--timeout",
        type=int,
        default=240,
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="刪除既有輸出並重新跑本 L1 的全部 L2。",
    )

    args = ap.parse_args()

    rows = load_csv(args.input)

    # 本程式只處理 L1 = 權威控制
    rows = [
        row
        for row in rows
        if (row.get("llm_category") or "").strip() == PARENT_L1
    ]

    print(f"L1 = {PARENT_L1}")
    print(f"待分類資料：{len(rows)} 筆")
    print(f"L2 類別數：{len(L2_NAMES)}")
    print(f"每題最多 L2：{L2_MAX_LABELS}")

    if not rows:
        raise ValueError(
            f"輸入檔找不到 L1 = {PARENT_L1} 的資料"
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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed_ids = load_completed_ids(output_path)

    if completed_ids:
        print(
            f"偵測到既有輸出：已完成 "
            f"{len(completed_ids)} 筆，將自動續跑。"
        )

    is_new_file = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    mode = "w" if is_new_file else "a"

    processed = 0
    skipped = 0
    failed = 0

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

            atomic_id = row["atomic_id"]

            if atomic_id in completed_ids:
                skipped += 1
                print(
                    f"[{i}/{len(rows)}] "
                    f"{atomic_id} -> SKIP"
                )
                continue

            prompt = make_prompt(row)
            labels = None
            last_error = None

            for attempt in range(
                1,
                args.retries + 1,
            ):
                try:
                    if args.provider == "ollama":
                        text_result = call_ollama(
                            prompt,
                            args.model,
                            base_url,
                            args.timeout,
                        )
                    else:
                        text_result = call_openai_compatible(
                            prompt,
                            args.model,
                            base_url,
                            args.api_key,
                            args.timeout,
                        )

                    labels = parse_json(text_result)
                    break

                except Exception as e:
                    last_error = e

                    print(
                        f"[{i}/{len(rows)}] "
                        f"{atomic_id} "
                        f"attempt {attempt} failed: {e}"
                    )

                    time.sleep(
                        min(2 ** attempt, 10)
                    )

            if labels is None:
                labels = []
                failed += 1

            out = {
                "faq_id": row["faq_id"],
                "question_date": row["question_date"],
                "atomic_id": atomic_id,
                "question": row["question"],
                "answer": row["answer"],
                "l1_category": PARENT_L1,
                "l2_category_1": labels[0] if len(labels) >= 1 else "",
                "l2_category_2": labels[1] if len(labels) >= 2 else "",
                "l2_category_3": labels[2] if len(labels) >= 3 else "",
                "l2_count": len(labels),
            }

            writer.writerow(out)
            f.flush()

            processed += 1

            if labels:
                completed_ids.add(atomic_id)

            shown = " | ".join(labels) if labels else "FAILED"

            print(
                f"[{i}/{len(rows)}] "
                f"{atomic_id} -> {shown}"
            )

    total = count_output_rows(output_path)

    print()
    print(f"完成：{args.output}")
    print(f"本次新增：{processed}")
    print(f"本次跳過：{skipped}")
    print(f"本次失敗：{failed}")
    print(f"輸出目前總筆數：{total}")


if __name__ == "__main__":
    main()