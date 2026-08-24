import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import requests

from ncl_taxonomy_l3 import (
    L3_MAX_LABELS,
    l3_category_names,
    l3_standard_text,
)


# ============================================================
# 本支程式只處理：
# L1 = 編目法
# L2 = 款目、檢索點與 RDA 實體關係
# ============================================================
PARENT_L1 = "編目法"
PARENT_L2 = "款目、檢索點與 RDA 實體關係"

L3_NAMES = l3_category_names(PARENT_L1, PARENT_L2)
L3_ID_TO_NAME = {
    i: name for i, name in enumerate(L3_NAMES, start=1)
}


SYSTEM_PROMPT = f"""
你正在進行國家圖書館 FAQ 的階層式分類。

本次資料已經確定：
L1 = {PARENT_L1}
L2 = {PARENT_L2}

你的工作只是在這個固定 L2 底下選擇 Level 3 類別。
不得更改 L1，也不得改判到其他 L2。

{l3_standard_text(PARENT_L1, PARENT_L2)}

【重要判斷規則】

1. 每題至少選 1 個、最多選 {L3_MAX_LABELS} 個 L3。
2. labels 依重要性排序，第 1 個必須是最主要的 L3。
3. 不得為了湊數而多選；只有一個主要知識面向就只選 1 個。
4. 本次只判斷「款目、檢索點與 RDA 實體關係」這個面向。
   即使原題同時具有其他 L2 標籤，也不要跨到其他 L2 的 L3。
5. question 與 answer 要一起判斷；若 question 過短或有代名詞，
   可利用 answer 還原它真正詢問的分類問題。
6. 優先判斷「使用者究竟在做哪一種分類決策」，不要只看關鍵字。

【本 L2 特別邊界】

A. 主要款目與附加款目
   - 核心是主要款目、附加款目的功能、選擇條件、差異，
     或某名稱／題名應作主要款目還是附加款目。
   - 例如：何者應作主要款目；title main entry 與 title added entry 有何不同；
     團體名稱應作主要款目或附加款目。
   - 若核心只是 MARC 1XX、7XX 欄位的 tag、indicator 或 subfield，
     應屬 L1「機讀編目格式」。

B. 檢索點選擇與建立
   - 核心是個人、團體、題名、作品或其他名稱是否需要建立檢索點，
     以及應建立哪些、幾個檢索點。
   - 例如：評論者是否需要建立檢索點；
     合集中各作品是否另立檢索點；
     某責任者是否應建立附加檢索點。
   - 若核心是名稱的權威標準形式、同名辨識、異名參照，
     應屬 L1「權威控制」。

C. 創作者、貢獻者與關係標示
   - 核心是 RDA 中 creator、contributor、publisher 等角色判定，
     或創作者／貢獻者與作品、表現形等之間的關係標示。
   - 例如：編者屬創作者還是貢獻者；
     關係標示語應如何選擇；
     某責任角色應使用何種 relationship designator。
   - 若只是責任者名稱在題名及責任者敘述中的照錄，
     應屬 L2「題名與責任者敘述」。

D. RDA／FRBR／LRM 實體與屬性
   - 核心是作品（Work）、表現形（Expression）、
     載體表現（Manifestation）、單件（Item）等實體概念，
     以及其屬性、RDA／FRBR／LRM 概念模型的辨識與解釋。
   - 例如：作品與表現形如何區分；
     ISBN 屬於哪一實體的屬性；
     某資訊在 RDA 中屬於何種 entity/attribute。
   - 若問題核心是兩個實體「彼此之間」的關係，
     優先歸「作品與相關實體關係」。

E. 作品與相關實體關係
   - 核心是作品、表現形、載體表現、單件及其他實體之間的關係，
     包括翻譯、改編、衍生、重製、版本、相關作品等。
   - 例如：翻譯作品與原作是何種關係；
     改編作品是否為新作品；
     不同表現形之間如何建立關聯。
   - 若只是問某一實體本身的定義或屬性，
     優先歸「RDA／FRBR／LRM 實體與屬性」。
   - 若核心只是權威題名／劃一題名的標準形式，
     應屬 L1「權威控制」。

判斷優先順序：
1. 問主要款目、附加款目、main entry / added entry
   → 「主要款目與附加款目」。
2. 問是否需要建立某個檢索點、要建幾個檢索點
   → 「檢索點選擇與建立」。
3. 問 creator / contributor / relationship designator / 責任角色
   → 「創作者、貢獻者與關係標示」。
4. 問 Work / Expression / Manifestation / Item、
   FRBR／LRM／RDA 實體或屬性本身
   → 「RDA／FRBR／LRM 實體與屬性」。
5. 問翻譯、改編、衍生、相關作品、不同實體間的關聯
   → 「作品與相關實體關係」。

特別注意：
- 「作者應如何寫在245 $c」屬題名與責任者敘述，不屬本 L3。
- 「100/700 的 indicator 或 subfield 怎麼填」屬機讀編目格式，不屬本 L3。
- 「某名稱的標準形式、異名、同名區分」屬權威控制，不屬本 L3。
- 「翻譯作品」若問題只是問譯者是不是 contributor，選
  「創作者、貢獻者與關係標示」；若問原作與譯作之間的關係，
  選「作品與相關實體關係」。
- 原則上每題只選 1 個 L3；只有題目同時包含兩個彼此獨立、
  且都是真正核心的款目／檢索點／RDA 關係問題時，才選第 2 個。

若同一題同時真的包含兩個獨立面向，最多可選 2 個；
否則只選最主要的一個。

請只輸出 JSON，不要解釋。

可用 L3 代碼：
{chr(10).join(f"{i} = {name}" for i, name in L3_ID_TO_NAME.items())}

輸出格式：
{{
  "labels": [1]
}}

或真的需要多標時：
{{
  "labels": [2, 4]
}}
""".strip()


OUTPUT_FIELDS = [
    "faq_id",
    "question_date",
    "atomic_id",
    "question",
    "answer",
    "l1_category",
    "l2_category",
    "source_l2_category_1",
    "source_l2_category_2",
    "source_l2_category_3",
    "l3_category_1",
    "l3_category_2",
    "l3_count",
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
        "l1_category",
        "l2_category_1",
        "l2_category_2",
        "l2_category_3",
        "l2_count",
    }

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"缺少欄位：{missing}")

    return rows


def row_belongs_to_parent(row):
    """只要 PARENT_L2 出現在該題任一 L2 label，就應進入本 L3 runner。"""
    if (row.get("l1_category") or "").strip() != PARENT_L1:
        return False

    l2_labels = {
        (row.get("l2_category_1") or "").strip(),
        (row.get("l2_category_2") or "").strip(),
        (row.get("l2_category_3") or "").strip(),
    }

    return PARENT_L2 in l2_labels


def make_prompt(row):
    answer = (row.get("answer") or "").strip()

    if len(answer) > 2500:
        answer = answer[:2500] + "……"

    source_l2 = [
        x for x in [
            (row.get("l2_category_1") or "").strip(),
            (row.get("l2_category_2") or "").strip(),
            (row.get("l2_category_3") or "").strip(),
        ]
        if x
    ]

    return f"""
【FAQ】
atomic_id: {row['atomic_id']}
question: {row['question']}
answer: {answer}

既有 L1：{row['l1_category']}
既有 L2 labels：{" | ".join(source_l2)}

現在只針對：
L1 = {PARENT_L1}
L2 = {PARENT_L2}

判斷最適合的 Level 3。
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
            raise ValueError(f"非法 L3 代碼：{value}")

        if value not in L3_ID_TO_NAME:
            raise ValueError(f"L3 代碼超出範圍：{value}")

        if value not in normalized:
            normalized.append(value)

    if not normalized:
        raise ValueError("labels 不可為空")

    if len(normalized) > L3_MAX_LABELS:
        raise ValueError(
            f"labels 超過 {L3_MAX_LABELS} 個：{normalized}"
        )

    return [L3_ID_TO_NAME[x] for x in normalized]


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
            and row.get("l3_category_1")
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
        default="data/llm_level2_corrected.csv",
        help="已完成並校正的 Level 2 結果。",
    )

    ap.add_argument(
        "--output",
        default="data/llm_level3-10.csv",
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
        help="刪除既有輸出並依新版 taxonomy 重新跑本 L1/L2 的全部 L3。",
    )

    args = ap.parse_args()

    rows = load_csv(args.input)

    # 本程式只處理固定的 L1/L2。
    # 注意：PARENT_L2 出現在 l2_category_1/2/3 任一欄都會納入。
    rows = [
        row
        for row in rows
        if row_belongs_to_parent(row)
    ]

    print(f"L1 = {PARENT_L1}")
    print(f"L2 = {PARENT_L2}")
    print(f"待分類資料：{len(rows)} 筆")
    print(f"L3 類別數：{len(L3_NAMES)}")
    print(f"每題最多 L3：{L3_MAX_LABELS}")

    if not rows:
        raise ValueError(
            f"輸入檔找不到 L1={PARENT_L1} / L2={PARENT_L2} 的資料"
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

            # 和舊 Level 2 runner 不同：
            # 最終失敗時不寫空白列。
            # 下次重新執行時會自動再嘗試這筆，避免 duplicate blank rows。
            if labels is None:
                failed += 1
                print(
                    f"[{i}/{len(rows)}] "
                    f"{atomic_id} -> FAILED "
                    f"({last_error})"
                )
                continue

            out = {
                "faq_id": row["faq_id"],
                "question_date": row["question_date"],
                "atomic_id": atomic_id,
                "question": row["question"],
                "answer": row["answer"],
                "l1_category": PARENT_L1,
                "l2_category": PARENT_L2,
                "source_l2_category_1": row.get("l2_category_1", ""),
                "source_l2_category_2": row.get("l2_category_2", ""),
                "source_l2_category_3": row.get("l2_category_3", ""),
                "l3_category_1": labels[0] if len(labels) >= 1 else "",
                "l3_category_2": labels[1] if len(labels) >= 2 else "",
                "l3_count": len(labels),
            }

            writer.writerow(out)
            f.flush()

            processed += 1
            completed_ids.add(atomic_id)

            shown = " | ".join(labels)

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