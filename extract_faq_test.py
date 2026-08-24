import json
import os
import pandas as pd
import requests


# ============================================================
# 基本設定
# ============================================================

INPUT_FILE = "onlySubject.csv"
OUTPUT_FILE = "atomic_qa_test.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"



TEST_SERIALS = ["1897"]


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
# 呼叫 Ollama
# ============================================================

def split_qa(question, answer):

    user_prompt = f"""
    
【主旨】
{question}
    
【原始答案】
{answer}
    
    
只輸出 JSON。
"""

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
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    content = result["message"]["content"]

    return json.loads(content)


# ============================================================
# Main
# ============================================================

def main():

    # 讀取 QA CLEAN
    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        encoding="utf-8-sig"
    )

    # 只挑指定題目
    test_df = df[
        df["serial"].isin(TEST_SERIALS)
    ].copy()

    output_rows = []

    for _, row in test_df.iterrows():

        serial = row["serial"]
        faq_id = row["序號"]
        question_date = row["提問日期"]
        question = row["主旨"]
        answer = row["答覆內容"]

        print("=" * 60)
        print(f"FAQ {faq_id}")
        print()
        print("原問題：")
        print(question)
        print()
        print("原答案：")
        print(answer)
        print()

        # 呼叫 Ollama
        result = split_qa(
            question,
            answer
        )

        qa_list = result.get("qa", [])

        print("拆成：", len(qa_list), "組 QA")
        print()

        for i, qa in enumerate(
            qa_list,
            start=1
        ):

            atomic_id = f"{faq_id}_A{i:02d}"

            print(f"Q{i}: {qa['question']}")
            print(f"A{i}: {qa['answer']}")
            print()

            # 一個 QA 加一列
            output_rows.append({
                "faq_id": faq_id,
                "question_date": question_date,
                "atomic_id": atomic_id,
                "question": qa["question"],
                "answer": qa["answer"]
            })

    if output_rows:

        output_df = pd.DataFrame(output_rows)

        output_df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"已輸出 {len(output_df)} 筆 atomic QA")
        print(f"檔案：{OUTPUT_FILE}")
                


if __name__ == "__main__":
    main()