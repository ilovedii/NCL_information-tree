import json
import os
import pandas as pd
import requests
import time


# ============================================================
# 1. 基本設定
# ============================================================

INPUT_FILE = "onlySubject.csv"
OUTPUT_FILE = "onlySubject_qa.csv"
BATCH_SIZE = 5
BATCH_SLEEP_SECONDS = 30

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"



# ============================================================
# 2. 中文 Prompt
# ============================================================

SYSTEM_PROMPT = """
你是一個「FAQ 知識單元抽取與檢索問題生成器」。

輸入包含：
1. 主旨（Subject）
2. 原始答覆（Answer）

你的任務是：
使用 Subject 確認主題，主要分析 Answer，
將其中可被未來使用者「獨立詢問、獨立回答」的資訊
拆成 Knowledge QA，並為每個知識單元產生一個
適合語意檢索的 canonical question。


【核心原則】

1. Question 與 Answer 必須完全由 Subject 與原始 Answer 支持。
2. 不得加入原文沒有的知識、條件、推論或判斷。
3. Question 必須可以單獨理解，並保留會影響答案的重要條件。
4. Question 應是未來使用者可能詢問的標準化檢索問題，
   不需要還原原始使用者的實際問句。
5. 不要加入「請問、您好、想詢問」等客套文字。


【拆分原則】

若 Answer 中存在多個可以被獨立詢問、獨立回答的資訊，
則拆成多個 QA。

適合獨立拆分的資訊包括：
- 不同規則
- 不同條件下的處理方式
- 例外
- 定義
- 不同對象各自對應的結果
- 可獨立查詢的數值、日期、版本、分類號、URL 等

以下通常不要另外拆分：
- 同一規則的原因或補充說明
- 同一程序的連續步驟
- 相同結論的重複描述
- 單純用來說明一般規則的案例


【資訊完整覆蓋】

原始 Answer 中所有具有知識價值的資訊都必須被保留。

包括但不限於：
規則、條件、例外、處理方式、數值、日期、版本、
分類號、書名與結果、人物與結果、機構、URL、
限制、注意事項及具體列舉項目。

每一項實質資訊至少要出現在一個 QA 的 Answer 中。

如果某項資訊無法合理放入既有 QA，
就建立新的 QA，不得省略。

可以刪除：
問候語、敬語、署名及無知識價值的客套文字。


【Example 與獨立資訊的判斷】

必須區分：

A. Supporting Example
如果案例只是用來說明一個一般規則，
則保留在該規則的 Answer 中，
通常不要另外建立 Question。

例如：
「文學作家的國籍通常以最新國籍認定。
石黑一雄後入英國籍，因此歸為英國。」

→ 石黑一雄可保留在一般規則的 Answer 中，
不必另外拆成 QA。


B. Independent Factual Item
如果不同對象各自具有不同、可獨立查詢的結果，
則每個「對象 → 結果」都是獨立 knowledge unit。

例如：
《王時敏》 → 909.8
《書畫家語林》 → 830.99
《中國藝術象徵辭典》 → 961.204

應分別產生：
「《王時敏》的中文圖書分類號為何？」
「《書畫家語林》的中文圖書分類號為何？」
「《中國藝術象徵辭典》的中文圖書分類號為何？」


【重要】

判斷是否拆分的核心是：

「這項資訊是否可以被未來使用者單獨詢問，
而原始 Answer 是否提供了明確答案？」

如果 YES，建立獨立 QA。

在輸出前檢查：
原始 Answer 中所有實質資訊是否都已被至少一個 QA 覆蓋。


【輸出格式】

只輸出合法 JSON：

{
  "qa": [
    {
      "question": "標準化檢索問題",
      "answer": "能完整回答此問題的原始答案內容"
    }
  ]
}

"""


# ============================================================
# 3. 呼叫 Ollama
# ============================================================

def split_knowledge_qa(question, answer, max_retries=3):

    user_prompt = f"""


【原始問題】
{question}

【原始答案】
{answer}


只輸出 JSON。
"""

    for attempt in range(1, max_retries + 1):

        try:

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "stream": False,
                    "format": "json",

                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],

                    "options": {
                        "temperature": 0
                    }
                },

                # 最多等 10 分鐘
                timeout=600
            )

            response.raise_for_status()

            result = response.json()

            content = result["message"]["content"]

            return json.loads(content)

        except requests.exceptions.ReadTimeout:

            print(
                f"⚠ Ollama timeout "
                f"({attempt}/{max_retries})"
            )

        except requests.exceptions.RequestException as e:

            print(
                f"⚠ Ollama request error "
                f"({attempt}/{max_retries}): {e}"
            )

        except json.JSONDecodeError as e:

            print(
                f"⚠ JSON 格式錯誤 "
                f"({attempt}/{max_retries}): {e}"
            )

        if attempt < max_retries:
            print("等待 5 秒後重試...")
            time.sleep(5)

    return None


def main():

    # --------------------------------------------------------
    # 讀取完整 QA CLEAN
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        encoding="utf-8-sig"
    )

    df["序號"] = (
        df["序號"]
        .astype(str)
        .str.strip()
    )

    print("=" * 70)
    print("FAQ Atomic QA Decomposition")
    print("=" * 70)
    print(f"原始 FAQ 總數：{len(df)}")

    # --------------------------------------------------------
    # 檢查已經完成的 FAQ
    # --------------------------------------------------------

    processed_faq_ids = set()

    if os.path.exists(OUTPUT_FILE):

        old_df = pd.read_csv(
            OUTPUT_FILE,
            dtype=str,
            encoding="utf-8-sig"
        )

        if "faq_id" in old_df.columns:

            processed_faq_ids = set(
                old_df["faq_id"]
                .dropna()
                .astype(str)
                .str.strip()
            )

        print(f"已完成 FAQ 數量：{len(processed_faq_ids)}")

    else:

        print("尚無輸出檔案，將建立新檔案。")

    # --------------------------------------------------------
    # 只留下尚未處理的 FAQ
    # --------------------------------------------------------

    remaining_df = df[
        ~df["序號"]
        .astype(str)
        .str.strip()
        .isin(processed_faq_ids)
    ].copy()

    total_remaining = len(remaining_df)

    print(f"本次尚需處理：{total_remaining} 筆")
    print("=" * 70)

    # --------------------------------------------------------
    # 一題一題處理
    # --------------------------------------------------------

    for count, (_, row) in enumerate(
        remaining_df.iterrows(),
        start=1
    ):

        # serial = row["serial"]
        faq_id = str(row["序號"]).strip()
        question_date = row["提問日期"]

        original_question = row["主旨"]
        original_answer = row["答覆內容"]

        print()
        print("=" * 70)
        print(
            f"[{count}/{total_remaining}] "
            f"FAQ {faq_id}"
        )
        print("=" * 70)

        print("\n【QA CLEAN 原始 Question】")
        print(original_question)
        print()
        print("【QA CLEAN 原始 Answer】")
        print(original_answer)
        print()

        # ----------------------------------------------------
        # 呼叫 Ollama
        # ----------------------------------------------------


        result = split_knowledge_qa(
            original_question,
            original_answer
        )


        if result is None:

            print(f"❌ FAQ {faq_id} 處理失敗")

            # 把失敗的 FAQ 記下來
            with open(
                "failed_faq.txt",
                "a",
                encoding="utf-8"
            ) as f:
                f.write(f"{faq_id}\n")

        else:

            print(">>> Ollama 回傳完成")

            qa_list = result.get("qa", [])

            output_rows = []

            # ------------------------------------------------
            # 解析 QA
            # ------------------------------------------------

            for i, qa in enumerate(qa_list, start=1):

                if not isinstance(qa, dict):
                    print(
                        f"⚠ FAQ {faq_id} "
                        f"第 {i} 個 QA 格式錯誤"
                    )
                    continue

                question = qa.get("question")
                answer = qa.get("answer")

                if not question or not answer:
                    print(
                        f"⚠ FAQ {faq_id} "
                        f"第 {i} 個 QA 缺少 question/answer"
                    )
                    print(
                        json.dumps(
                            qa,
                            ensure_ascii=False,
                            indent=2
                        )
                    )
                    continue

                print(f"Q{i}: {question}")
                print(f"A{i}: {answer}")
                print()

                output_rows.append({
                    "faq_id": faq_id,
                    "question_date": question_date,
                    "atomic_id": f"{faq_id}_A{i:02d}",
                    "question": question,
                    "answer": answer
                })

            # ------------------------------------------------
            # 每完成一個 FAQ 就立即寫入
            # ------------------------------------------------

            if output_rows:

                output_df = pd.DataFrame(output_rows)

                file_exists = os.path.exists(
                    OUTPUT_FILE
                )

                output_df.to_csv(
                    OUTPUT_FILE,
                    mode="a",
                    header=not file_exists,
                    index=False,
                    encoding="utf-8-sig"
                )

                processed_faq_ids.add(faq_id)

                print(
                    f"✓ FAQ {faq_id} 完成，"
                    f"產生 {len(output_rows)} 組 QA"
                )

            else:

                print(
                    f"❌ FAQ {faq_id} "
                    f"沒有可寫入的 QA"
                )

                with open(
                    "failed_faq01.txt",
                    "a",
                    encoding="utf-8"
                ) as f:
                    f.write(f"{faq_id}\n")

        # ----------------------------------------------------
        # 每處理 5 筆休息
        # ----------------------------------------------------

        if (
            count % BATCH_SIZE == 0
            and count < total_remaining
        ):

            print()
            print("=" * 70)
            print(
                f"已處理 {count} / "
                f"{total_remaining} 筆"
            )
           

            time.sleep(BATCH_SLEEP_SECONDS)

    # --------------------------------------------------------
    # 全部完成
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("全部處理完成")
    print("=" * 70)

    
    print(f"輸出檔案：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()