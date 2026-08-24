class EvidenceSelector:
    def __init__(self, llm, model, think=True, batch_size=40):
        self.llm = llm
        self.model = model
        self.think = think
        self.batch_size = max(1, int(batch_size))

    def _flatten(self, summaries):
        items = []
        counter = 1
        for summary_index, summary in enumerate(summaries, start=1):
            for unit in summary.get("knowledge_units", []):
                item = {
                    "evidence_id": f"E{counter:04d}",
                    "summary_index": summary_index,
                    "role": summary.get("role", "node"),
                    "path": summary.get("path", ""),
                    "knowledge_id": unit.get("knowledge_id", ""),
                    "type": unit.get("type", "background"),
                    "content": unit.get("content", ""),
                    "time_scope": unit.get("time_scope", ""),
                    "source_ids": list(unit.get("source_ids", [])),
                }
                items.append(item)
                counter += 1
        return items

    def _assess_batch(self, query, items):
        ids = [item["evidence_id"] for item in items]
        sections = []
        for item in items:
            sections.append(
                f"[{item['evidence_id']}] role={item['role']} path={item['path']} type={item['type']} time={item['time_scope']}\n{item['content']}\nsources={','.join(item['source_ids'])}"
            )
        schema = {
            "type": "object",
            "properties": {
                "assessments": {
                    "type": "array",
                    "minItems": len(items),
                    "maxItems": len(items),
                    "items": {
                        "type": "object",
                        "properties": {
                            "evidence_id": {"type": "string", "enum": ids},
                            "utility": {
                                "type": "string",
                                "enum": ["direct", "supporting", "background", "low_relevance"],
                            },
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "reason": {"type": "string"},
                        },
                        "required": ["evidence_id", "utility", "score", "reason"],
                    },
                }
            },
            "required": ["assessments"],
        }
        prompt = f"""你是 Evidence Prioritizer。你不是來刪除知識，也不是直接回答問題。請評估每一個 knowledge unit 對目前問題的用途，所有輸入 unit 都必須各自得到一次評估。

使用者問題：
{query}

Knowledge Units：
{chr(10).join(sections)}

分級規則：
1. direct：內容可直接支持問題答案。
2. supporting：不能單獨回答，但提供必要規則、條件、例外、推論基礎或鄰近知識。
3. background：與問題所屬領域相關，可增加完整性，但不是回答核心。
4. low_relevance：對本題幫助低，但仍保留在完整知識上下文中。
5. historical_change、conflict、exception 即使不是直接答案，只要可能影響現行判斷，至少應評估為 supporting。
6. score 表示對本題的實用程度，不代表知識真偽。
7. reason 只寫短理由，不要輸出詳細思考過程。
8. 必須評估所有 evidence_id，不可遺漏、不可新增。
"""
        result = self.llm.chat_json(
            self.model,
            [{"role": "user", "content": prompt}],
            schema,
            temperature=0.0,
            think=self.think,
        )
        return result.get("assessments", [])

    def prioritize(self, query, summaries):
        items = self._flatten(summaries)
        if not items:
            return []
        assessments = {}
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            for result in self._assess_batch(query, batch):
                evidence_id = str(result.get("evidence_id", ""))
                if evidence_id in assessments:
                    continue
                assessments[evidence_id] = {
                    "utility": str(result.get("utility", "background")),
                    "score": float(result.get("score", 0.0)),
                    "reason": str(result.get("reason", "")).strip(),
                }
        utility_rank = {
            "direct": 0,
            "supporting": 1,
            "background": 2,
            "low_relevance": 3,
        }
        output = []
        for item in items:
            assessment = assessments.get(
                item["evidence_id"],
                {
                    "utility": "background",
                    "score": 0.0,
                    "reason": "模型未回傳此 unit 的評估，保留為背景知識。",
                },
            )
            merged = dict(item)
            merged.update(assessment)
            output.append(merged)
        output.sort(
            key=lambda item: (
                utility_rank.get(item.get("utility"), 9),
                -float(item.get("score", 0.0)),
                item.get("evidence_id", ""),
            )
        )
        return output