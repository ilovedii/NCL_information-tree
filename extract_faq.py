import json
import os
import pandas as pd
import requests
import time


# ============================================================
# 1. 基本設定
# ============================================================

INPUT_FILE = "reference_qa_clean.csv"
OUTPUT_FILE = "atomic_qa.csv"
BATCH_SIZE = 5
BATCH_SLEEP_SECONDS = 30

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"



# ============================================================
# 2. 中文 Prompt
# ============================================================

SYSTEM_PROMPT = """

你是一個「FAQ 問答拆分器」。

你的任務是將一筆 FAQ 整理成一組或多組 Question-Answer。

最重要的原則是：

「要產生幾個 Question，只由原始問題中的資訊需求決定。」

原始答案只能用來找出每個 Question 對應的 Answer。
不能因為原始答案很長、包含多個段落、條列、規則、案例、
原因、補充說明或結論，就額外創造新的 Question。


【處理方式】

第一步：
先只看「原始問題」，判斷其中有幾個可以獨立回答的資訊需求。

第二步：
根據這些資訊需求建立 Question。

第三步：
再從「原始答案」中找出每個 Question 對應的 Answer。

第四步：
檢查原始答案中的所有實質資訊是否都已包含在某個 Answer 中。
如果某段資訊仍然是在回答原始問題中的同一個資訊需求，
應將它合併進對應的 Answer，而不是另外創造新的 Question。

不得遺漏原始答案中與原始問題相關的重要規則、條件、
處理方式、例外、數值、限制或注意事項。


【例子一：原始問題包含多個獨立問題】

原始問題：
館員好，我想問你是誰，還有你今年幾歲？

原始答案：
您好，我叫 Amy，我今年 23 歲。

應拆成：

Q1：你是誰？
A1：我叫 Amy。

Q2：你今年幾歲？
A2：我今年 23 歲。

原因：
「姓名」與「年齡」是兩個可以分開詢問、
分開回答的資訊需求。


【例子二：多個條件共同構成同一個問題，不拆】

原始問題：
如果我是學生，而且目前人在國外，可以線上申請嗎？

原始答案：
具有學生身分且目前人在國外者，可以使用線上申請。

正確做法：

Q1：
具有學生身分且目前人在國外時，是否可以線上申請？

A1：
具有學生身分且目前人在國外者，可以使用線上申請。

原因：
「學生」與「人在國外」是同一問題中的共同條件，
不是兩個獨立資訊需求。


【例子三：原始問題只有一題，但答案包含很多內容】

原始問題：
圖書館在進行編目外包時，應如何控管編目品質？

原始答案：
可訂定錯誤率認定標準、抽查方法、錯誤率計算方式、
廠商罰則、作業人員改善方式及人員更換機制。

即使答案包含很多不同措施，
它們都共同回答同一個問題，
因此不能拆成多組 QA。

正確做法：

Q1：
圖書館在進行編目外包時，應如何控管編目品質？

A1：
可訂定錯誤率認定標準、抽查方法、錯誤率計算方式、
廠商罰則、作業人員改善方式及人員更換機制。


【拆分原則】

1. Question 的數量只由「原始問題」中的資訊需求決定。

2. 判斷是否拆分時，先不要根據原始答案的段落、
   條列或知識數量決定。

3. 只有原始問題中確實存在兩個以上可以分開詢問、
   分開回答的資訊需求時，才拆成多組 QA。

4. 拆出的 Question 必須能在原始問題中找到對應的語意。
   可以刪除客套話、整理語句、補足必要上下文，
   但不能創造原始問題沒有詢問的新問題。

5. 如果多個條件共同描述同一個使用情境，
   或必須共同成立才能回答，就不能拆開。

6. 即使原始答案包含多個段落、條列、規則、案例、
   原因、補充說明或結論，
   只要它們共同回答同一個資訊需求，
   就必須保留在同一組 QA 中。

7. 原始答案只能用來產生對應的 Answer，
   不得因答案中的額外資訊新增 Question。

8. 每個拆出的 Question 必須能單獨理解，
   並保留會影響答案的重要條件，
   例如時間、對象、版本、使用情境或文獻類型。

9. Answer 只能根據原始答案整理，
   不得加入外部知識或自行推論。

10. 如果原始問題只有一個資訊需求，
    無論原始答案多長，都只輸出一組 QA。

11. 可以刪除客套話、重複敘述或沒有實質資訊的文字，
    但不得刪除會影響答案意義的重要內容。

12. 如果原始問題只有一個資訊需求，
    無論原始答案多長，都只輸出一組 QA。


【輸出格式】

只輸出合法 JSON，
不要輸出其他說明文字。

格式：

{
  "qa": [
    {
      "question": "問題1",
      "answer": "答案1"
    },
    {
      "question": "問題2",
      "answer": "答案2"
    }
  ]
}

"""


# ============================================================
# 3. 呼叫 Ollama
# ============================================================

def split_knowledge_qa(question, answer, max_retries=3):

    user_prompt = f"""
請將以下 FAQ 按照原始問題中的資訊需求進行拆分。

【原始問題】
{question}

【原始答案】
{answer}

請先根據原始問題判斷有幾個獨立問題，
再從原始答案中找出各問題對應的答案。

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
                    "failed_faq.txt",
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