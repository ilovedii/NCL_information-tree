from collections import Counter


class EvidenceSelector:
    ROLE_PRIORITY = {
        "primary": 0,
        "alternative": 1,
        "sibling": 2,
        "faq_family": 3,
        "fallback": 4,
        "parent": 5,
        "node": 6,
    }

    UTILITY_PRIORITY = {
        "direct": 0,
        "supporting": 1,
        "background": 2,
        "low_relevance": 3,
    }

    FINAL_CONTEXT_UTILITIES = {"direct", "supporting"}

    def __init__(
        self,
        llm,
        model,
        think=True,
        batch_size=40,
        min_supporting_without_direct=2,
    ):
        self.llm = llm
        self.model = model
        self.think = think
        self.batch_size = max(1, int(batch_size))
        self.min_supporting_without_direct = max(
            1,
            int(min_supporting_without_direct),
        )
        self.reset_stats()

    def reset_stats(self):
        self._evidence_batch_calls = 0
        self._evidence_items_assessed = 0
        self._sufficiency_calls = 0
        self._sufficiency_deterministic_skips = 0

    def stats(self):
        return {
            "evidence_batch_api_calls": self._evidence_batch_calls,
            "evidence_items_assessed": self._evidence_items_assessed,
            "sufficiency_api_calls": self._sufficiency_calls,
            "sufficiency_deterministic_skips": self._sufficiency_deterministic_skips,
        }

    @staticmethod
    def _clean_source_ids(source_ids):
        output = []
        seen = set()
        for source_id in source_ids or []:
            value = str(source_id).strip()
            if value and value not in seen:
                seen.add(value)
                output.append(value)
        return output

    def _identity(self, item):
        """Deterministic duplicate identity.

        The same atomic source may be reachable from several taxonomy paths,
        FAQ recovery, or fallback retrieval. It should be assessed only once.
        This is exact deduplication, not semantic pruning.
        """
        source_ids = tuple(sorted(self._clean_source_ids(item.get("source_ids", []))))
        if source_ids:
            return ("source_ids", source_ids)
        return (
            "content",
            str(item.get("content", "")).strip(),
            str(item.get("time_scope", "")).strip(),
        )

    def identity(self, item):
        """Public wrapper used by progressive search bookkeeping."""
        return self._identity(item)

    def _next_counter(self, evidence):
        largest = 0
        for item in evidence or []:
            evidence_id = str(item.get("evidence_id", ""))
            if evidence_id.startswith("E") and evidence_id[1:].isdigit():
                largest = max(largest, int(evidence_id[1:]))
        return largest + 1

    def _flatten(self, packs, start_counter=1, existing_evidence=None):
        """Flatten packs and suppress only deterministic duplicates.

        No Top-K is applied. Every unique atomic source contained in supplied
        packs remains eligible for assessment when its progressive batch opens.
        """
        items = []
        item_by_key = {}
        blocked_keys = set()

        for existing in existing_evidence or []:
            blocked_keys.add(self._identity(existing))

        counter = max(1, int(start_counter))

        for pack_index, pack in enumerate(packs, start=1):
            for unit in pack.get("knowledge_units", []):
                candidate = {
                    "evidence_id": f"E{counter:04d}",
                    "summary_index": pack_index,
                    "role": pack.get("role", "node"),
                    "path": pack.get("path", ""),
                    "knowledge_id": unit.get("knowledge_id", ""),
                    "type": unit.get("type", "background"),
                    "content": unit.get("content", ""),
                    "time_scope": unit.get("time_scope", ""),
                    "source_ids": self._clean_source_ids(unit.get("source_ids", [])),
                    "all_paths": [pack.get("path", "")],
                    "all_roles": [pack.get("role", "node")],
                }
                key = self._identity(candidate)

                if key in blocked_keys:
                    continue

                if key in item_by_key:
                    existing = item_by_key[key]
                    path = candidate["path"]
                    role = candidate["role"]
                    if path and path not in existing["all_paths"]:
                        existing["all_paths"].append(path)
                    if role and role not in existing["all_roles"]:
                        existing["all_roles"].append(role)

                    if self.ROLE_PRIORITY.get(role, 99) < self.ROLE_PRIORITY.get(
                        existing["role"], 99
                    ):
                        existing["role"] = role
                        existing["path"] = path
                    continue

                items.append(candidate)
                item_by_key[key] = candidate
                counter += 1

        return items

    def _assess_batch(self, query, items):
        ids = [item["evidence_id"] for item in items]
        sections = []
        for item in items:
            sections.append(
                f"[{item['evidence_id']}] role={item['role']} path={item['path']} "
                f"type={item['type']} time={item['time_scope']}\n"
                f"{item['content']}\n"
                f"sources={','.join(item['source_ids'])}"
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
                                "enum": [
                                    "direct",
                                    "supporting",
                                    "background",
                                    "low_relevance",
                                ],
                            },
                            "score": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "evidence_id",
                            "utility",
                            "score",
                            "reason",
                        ],
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
1. direct：內容能直接支持使用者實際詢問的核心答案，而且必須符合問題中的明確限制條件。
2. supporting：不能單獨回答，但提供必要規則、條件、例外、推論基礎、格式對照或鄰近知識。
3. background：與問題領域相關，可增加理解，但不是回答核心。
4. low_relevance：對本題幫助低。
5. 「constraint alignment」必須嚴格檢查：欄位、位址、格式/版本、實體、代碼角色與關係若不一致，不得只因關鍵字重疊就標成 direct。
6. 例如問題問 Leader/07 的 a 與 m：僅談 Leader/06=a、Leader/06=m 或 Leader/07=s 的資料不能算 direct；最多是 supporting/background，除非它同時直接回答 Leader/07 的 a 或 m。
7. 若問題明確指定 MARC 21，而 evidence 明確只描述 CMARC 的操作規則，該操作規則不得直接視為 MARC 21 的 direct evidence；若它只是提供可被其他證據佐證的一般概念，可標 supporting。
8. historical_change、conflict、exception、版本差異，只要可能改變最終判斷，至少應評估為 supporting，不可因為不是主答案就降成 background。
9. score 表示對本題的實用程度，不代表知識真偽。
10. reason 只寫短理由，不要輸出詳細思考過程。
11. 必須評估所有 evidence_id，不可遺漏、不可新增。
"""

        self._evidence_batch_calls += 1
        self._evidence_items_assessed += len(items)
        result = self.llm.chat_json(
            self.model,
            [{"role": "user", "content": prompt}],
            schema,
            temperature=0.0,
            think=self.think,
        )
        return result.get("assessments", [])

    def _assess_items(self, query, items):
        if not items:
            return []

        assessments = {}
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            for result in self._assess_batch(query, batch):
                evidence_id = str(result.get("evidence_id", ""))
                if evidence_id in assessments:
                    continue
                assessments[evidence_id] = {
                    "utility": str(result.get("utility", "background")),
                    "score": float(result.get("score", 0.0)),
                    "reason": str(result.get("reason", "")).strip(),
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
        return output

    def _sort(self, evidence):
        return sorted(
            evidence,
            key=lambda item: (
                self.UTILITY_PRIORITY.get(item.get("utility"), 9),
                -float(item.get("score", 0.0)),
                item.get("evidence_id", ""),
            ),
        )

    def prioritize(self, query, packs):
        """Assess all unique evidence in the supplied packs.

        Retained for compatibility. The progressive RAG pipeline normally uses
        extend_prioritized() with one locally ordered batch at a time.
        """
        items = self._flatten(packs)
        return self._sort(self._assess_items(query, items))

    def extend_prioritized(self, query, existing_evidence, new_packs):
        """Assess only newly introduced evidence and merge with prior results."""
        existing_evidence = list(existing_evidence or [])
        start_counter = self._next_counter(existing_evidence)
        new_items = self._flatten(
            new_packs,
            start_counter=start_counter,
            existing_evidence=existing_evidence,
        )
        assessed_new = self._assess_items(query, new_items)
        return self._sort(existing_evidence + assessed_new)

    @staticmethod
    def has_direct(evidence, min_score=0.0):
        threshold = float(min_score)
        return any(
            item.get("utility") == "direct"
            and float(item.get("score", 0.0)) >= threshold
            for item in evidence or []
        )

    def filter_for_context(self, evidence, include_background=False):
        """Return only evidence appropriate for the final answer prompt.

        Knowledge is not deleted from storage or the trace. By default only
        direct/supporting evidence enters the final answer context.
        """
        allowed = set(self.FINAL_CONTEXT_UTILITIES)
        if include_background:
            allowed.add("background")
        return self._sort(
            [item for item in evidence or [] if item.get("utility") in allowed]
        )

    @staticmethod
    def utility_counts(evidence):
        return dict(
            Counter(item.get("utility", "unknown") for item in evidence or [])
        )

    def check_sufficiency(self, query, evidence):
        """Judge whether currently relevant evidence covers the whole query.

        The checker accepts compositional coverage: if separate evidence items
        directly establish A=X and B=Y, that can be sufficient for a basic
        "A 與 B 有何不同" question. It must not, however, invent operational
        differences that the evidence does not establish.
        """
        relevant = self.filter_for_context(evidence, include_background=False)
        direct = [item for item in relevant if item.get("utility") == "direct"]
        supporting = [
            item for item in relevant if item.get("utility") == "supporting"
        ]

        if not relevant:
            self._sufficiency_deterministic_skips += 1
            return {
                "sufficient": False,
                "query_aspects": [],
                "covered_aspects": [],
                "missing_aspects": ["目前尚未找到可支持核心答案的 evidence"],
                "reason": "目前沒有 direct/supporting evidence，因此繼續擴張搜尋。",
                "relevant_evidence_count": 0,
                "direct_evidence_count": 0,
                "checked_by_llm": False,
            }

        if not direct and len(supporting) < self.min_supporting_without_direct:
            self._sufficiency_deterministic_skips += 1
            return {
                "sufficient": False,
                "query_aspects": [],
                "covered_aspects": [],
                "missing_aspects": ["目前證據仍不足以完整支持核心答案"],
                "reason": (
                    "尚無 direct evidence，且 supporting evidence 數量不足，"
                    "先繼續 progressive expansion。"
                ),
                "relevant_evidence_count": len(relevant),
                "direct_evidence_count": 0,
                "checked_by_llm": False,
            }

        ids = [item["evidence_id"] for item in relevant]
        blocks = []
        for item in relevant:
            blocks.append(
                f"[{item['evidence_id']}] utility={item.get('utility')} "
                f"score={float(item.get('score', 0.0)):.3f} "
                f"role={item.get('role')} path={item.get('path')}\n"
                f"{item.get('content', '')}\n"
                f"sources={','.join(item.get('source_ids', []))}"
            )

        schema = {
            "type": "object",
            "properties": {
                "query_aspects": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 12,
                    "items": {"type": "string"},
                },
                "covered_aspects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "aspect": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ids,
                                },
                            },
                        },
                        "required": ["aspect", "evidence_ids"],
                    },
                },
                "missing_aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "sufficient": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": [
                "query_aspects",
                "covered_aspects",
                "missing_aspects",
                "sufficient",
                "reason",
            ],
        }

        prompt = f"""你是 Evidence Sufficiency Checker。你的工作不是重新排序 evidence，而是判斷目前已篩出的 direct/supporting evidence 是否足以完整回答使用者問題。

使用者問題：
{query}

目前可用 evidence：
{chr(10).join(blocks)}

判斷規則：
1. 先把使用者真正要求回答的資訊拆成 query_aspects；只拆實質需求，不要把語氣詞拆成 aspect。
2. sufficient=true 只有在每一個實質 aspect 都可由目前 evidence 支持時才成立。
3. 允許「組合式覆蓋（compositional coverage）」：若 evidence A 已建立 A=X，evidence B 已建立 B=Y，使用者只問 A 與 B 的基本定義差異，則 X 與 Y 本身即可構成該比較 aspect 的支持，不要求資料庫一定另有一句逐字寫成「A 與 B 的差異是...」。
4. 但若使用者要求的是完整操作規則、適用條件、例外、版本差異或欄位連動，不能只靠兩個名詞定義推導；缺少操作面證據時仍須判 insufficient。
5. constraint alignment 仍然有效：欄位、位址、格式/版本、實體、代碼角色若不一致，不得把它們視為同一 aspect 的直接支持。
6. 若使用者明確指定 MARC 21，CMARC 專屬的欄位操作規則不能直接當成 MARC 21 規則；除非 evidence 明確提供兩格式的對照或一致性依據。
7. 多意圖、比較、條件、例外、兩段式問題必須逐項覆蓋。
8. 若使用者問「有哪些、還有其他、列出、全部、完整」等枚舉型問題，不能只因找到一個例子就判 sufficient；需有足以支撐該回答範圍的多項證據，或同源 FAQ/provenance 已補足。
9. 不得假設未提供的知識存在，也不得用常識補齊缺口。
10. covered_aspects 的 evidence_ids 只能使用目前提供的 ID。
11. missing_aspects 明確指出仍缺什麼；若沒有缺口，回傳空陣列。
12. reason 只寫簡短、可稽核的判斷理由，不要輸出詳細思考過程。
"""

        self._sufficiency_calls += 1
        result = self.llm.chat_json(
            self.model,
            [{"role": "user", "content": prompt}],
            schema,
            temperature=0.0,
            think=self.think,
        )

        query_aspects = [
            str(x).strip()
            for x in result.get("query_aspects", [])
            if str(x).strip()
        ]
        covered_aspects = []
        allowed_ids = set(ids)
        for item in result.get("covered_aspects", []):
            aspect = str(item.get("aspect", "")).strip()
            evidence_ids = [
                str(eid).strip()
                for eid in item.get("evidence_ids", [])
                if str(eid).strip() in allowed_ids
            ]
            if aspect:
                covered_aspects.append(
                    {"aspect": aspect, "evidence_ids": evidence_ids}
                )

        missing_aspects = [
            str(x).strip()
            for x in result.get("missing_aspects", [])
            if str(x).strip()
        ]
        sufficient = bool(result.get("sufficient", False)) and not missing_aspects

        return {
            "sufficient": sufficient,
            "query_aspects": query_aspects,
            "covered_aspects": covered_aspects,
            "missing_aspects": missing_aspects,
            "reason": str(result.get("reason", "")).strip(),
            "relevant_evidence_count": len(relevant),
            "direct_evidence_count": len(direct),
            "checked_by_llm": True,
        }
