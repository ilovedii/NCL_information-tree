from collections import Counter
from datetime import datetime
import re


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
        enable_conflict_analysis=True,
    ):
        self.llm = llm
        self.model = model
        self.think = think
        self.batch_size = max(1, int(batch_size))
        self.min_supporting_without_direct = max(
            1,
            int(min_supporting_without_direct),
        )
        self.enable_conflict_analysis = bool(enable_conflict_analysis)
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

    @staticmethod
    def _is_rule_sensitive_query(query):
        """Detect formal cataloguing/classification-rule questions conservatively.

        The detector is deterministic and intentionally separate from answer
        generation. A bare three-digit number is not enough; MARC-like field
        numbers must appear with cataloguing context, unless the query already
        contains an explicit formal-rule marker such as MARC, RDA, $n, Leader,
        classification number, etc.
        """
        text = str(query or "").strip()
        if not text:
            return False

        explicit_rule_patterns = [
            r"\bMARC\s*21\b",
            r"\bMARC\b",
            r"\bCMARC\b",
            r"\bRDA\b",
            r"\bAACR2?\b",
            r"\bLeader\b",
            r"\bLDR\b",
            r"\b00[15678]\b",
            r"\$[0-9A-Za-z]",
            r"\bsubfield\b",
            r"\bindicator\b",
            r"分類號",
            r"作者號",
            r"主題標目",
            r"複分",
            r"著錄規則",
            r"編目規則",
        ]
        if any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in explicit_rule_patterns
        ):
            return True

        # MARC-like 3-digit field number must be accompanied by cataloguing
        # context. This catches "264 ... 年份著錄" but not "我有264本書".
        has_field_number = bool(re.search(r"(?<!\d)\d{3}(?!\d)", text))
        has_catalog_context = bool(
            re.search(
                r"欄位|字段|field|分欄|子欄位|指標|位址|"
                r"著錄|編目|版權年|出版年|代碼|題名|責任者",
                text,
                flags=re.IGNORECASE,
            )
        )
        if has_field_number and has_catalog_context:
            return True

        # Explicit MARC field + indicator notation, e.g. 264 4, 264 #4, 245 10.
        if re.search(
            r"(?<!\d)\d{3}\s*#?\s*[0-9#]{1,2}(?!\d)",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        return False

    @staticmethod
    def _parse_evidence_date(value):
        text = str(value or "").strip()
        if not text:
            return None
        candidates = [text, text[:10]]
        for candidate in candidates:
            try:
                return datetime.fromisoformat(candidate).date()
            except ValueError:
                pass
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d", "%Y/%m", "%Y-%m"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        return None

    def _resolve_conflict_groups(self, groups, relevant):
        """Resolve only true same-scope conflicts, using date first.

        LLM decides semantic relationship. Python does the temporal resolution.
        If any member lacks a usable date or newest date is tied, the conflict is
        left unresolved for librarian review. Nothing is deleted from evidence.
        """
        evidence_by_id = {
            str(item.get("evidence_id", "")): item
            for item in relevant or []
        }
        normalized = []
        unresolved = []
        resolved = []

        for group in groups or []:
            relationship = str(group.get("relationship", "unknown")).strip()
            same_scope = bool(group.get("same_scope", False))
            ids = []
            for eid in group.get("evidence_ids", []):
                eid = str(eid).strip()
                if eid in evidence_by_id and eid not in ids:
                    ids.append(eid)
            if len(ids) < 2:
                continue

            entry = {
                "evidence_ids": ids,
                "relationship": relationship,
                "same_scope": same_scope,
                "reason": str(group.get("reason", "")).strip(),
                "resolution_status": "not_applicable",
                "resolution_method": "none",
                "preferred_evidence_ids": [],
                "superseded_evidence_ids": [],
            }

            if relationship == "conflict" and same_scope:
                dated = []
                all_dates_present = True
                for eid in ids:
                    date_value = self._parse_evidence_date(
                        evidence_by_id[eid].get("time_scope", "")
                    )
                    if date_value is None:
                        all_dates_present = False
                    dated.append((eid, date_value))

                if all_dates_present:
                    newest = max(date_value for _, date_value in dated)
                    newest_ids = [eid for eid, date_value in dated if date_value == newest]
                    if len(newest_ids) == 1:
                        preferred = newest_ids[0]
                        entry.update(
                            {
                                "resolution_status": "resolved",
                                "resolution_method": "latest_question_date",
                                "preferred_evidence_ids": [preferred],
                                "superseded_evidence_ids": [
                                    eid for eid in ids if eid != preferred
                                ],
                            }
                        )
                        resolved.append(entry)
                    else:
                        entry.update(
                            {
                                "resolution_status": "unresolved",
                                "resolution_method": "date_tie",
                            }
                        )
                        unresolved.append(entry)
                else:
                    entry.update(
                        {
                            "resolution_status": "unresolved",
                            "resolution_method": "missing_date_metadata",
                        }
                    )
                    unresolved.append(entry)

            normalized.append(entry)

        if unresolved:
            status = "unresolved"
        elif resolved:
            status = "resolved_by_date"
        else:
            status = "none"

        return {
            "status": status,
            "groups": normalized,
            "resolved_groups": resolved,
            "unresolved_groups": unresolved,
            "preferred_evidence_ids": [
                eid
                for group in resolved
                for eid in group.get("preferred_evidence_ids", [])
            ],
            "superseded_evidence_ids": [
                eid
                for group in resolved
                for eid in group.get("superseded_evidence_ids", [])
            ],
        }

    def _insufficient_result(self, reason, missing, relevant, direct, rule_sensitive):
        return {
            "sufficient": False,
            "query_aspects": [],
            "covered_aspects": [],
            "missing_aspects": list(missing),
            "reason": reason,
            "relevant_evidence_count": len(relevant),
            "direct_evidence_count": len(direct),
            "checked_by_llm": False,
            "rule_sensitive": bool(rule_sensitive),
            "coverage_mode": "insufficient",
            "evidence_relationship": "unknown",
            "conflict_detected": False,
            "conflict_resolution": {
                "status": "none",
                "groups": [],
                "resolved_groups": [],
                "unresolved_groups": [],
                "preferred_evidence_ids": [],
                "superseded_evidence_ids": [],
            },
        }

    def check_sufficiency(self, query, evidence):
        """Judge full-query coverage and classify evidence relationships.

        Conflict analysis is folded into this same LLM call. The LLM only
        classifies semantic relationships; Python resolves true same-scope
        conflicts by question_date, keeping unresolved conflicts for librarians.
        """
        relevant = self.filter_for_context(evidence, include_background=False)
        direct = [item for item in relevant if item.get("utility") == "direct"]
        supporting = [
            item for item in relevant if item.get("utility") == "supporting"
        ]
        rule_sensitive = self._is_rule_sensitive_query(query)

        if not relevant:
            self._sufficiency_deterministic_skips += 1
            return self._insufficient_result(
                "目前沒有 direct/supporting evidence，因此繼續擴張搜尋。",
                ["目前尚未找到可支持核心答案的 evidence"],
                relevant,
                direct,
                rule_sensitive,
            )

        if not direct and len(supporting) < self.min_supporting_without_direct:
            self._sufficiency_deterministic_skips += 1
            return self._insufficient_result(
                "尚無 direct evidence，且 supporting evidence 數量不足，先繼續 progressive expansion。",
                ["目前證據仍不足以完整支持核心答案"],
                relevant,
                direct,
                rule_sensitive,
            )

        ids = [item["evidence_id"] for item in relevant]
        blocks = []
        for item in relevant:
            blocks.append(
                f"[{item['evidence_id']}] utility={item.get('utility')} "
                f"score={float(item.get('score', 0.0)):.3f} "
                f"role={item.get('role')} path={item.get('path')} "
                f"date={item.get('time_scope', '')}\n"
                f"{item.get('content', '')}\n"
                f"sources={','.join(item.get('source_ids', []))}"
            )

        relationship_values = [
            "compatible",
            "complementary",
            "conditional",
            "conflict",
            "unknown",
        ]
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
                                "items": {"type": "string", "enum": ids},
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
                "coverage_mode": {
                    "type": "string",
                    "enum": ["evidence_grounded", "evidence_compositional", "insufficient"],
                },
                "evidence_relationship": {
                    "type": "string",
                    "enum": relationship_values,
                },
                "conflict_groups": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "evidence_ids": {
                                "type": "array",
                                "minItems": 2,
                                "items": {"type": "string", "enum": ids},
                            },
                            "relationship": {
                                "type": "string",
                                "enum": relationship_values,
                            },
                            "same_scope": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "evidence_ids",
                            "relationship",
                            "same_scope",
                            "reason",
                        ],
                    },
                },
                "reason": {"type": "string"},
            },
            "required": [
                "query_aspects",
                "covered_aspects",
                "missing_aspects",
                "sufficient",
                "coverage_mode",
                "evidence_relationship",
                "conflict_groups",
                "reason",
            ],
        }

        conflict_instruction = (
            "請同時做 evidence relationship/conflict classification。"
            if self.enable_conflict_analysis
            else "conflict_groups 一律回空陣列，evidence_relationship 回 unknown。"
        )

        prompt = f"""你是 Evidence Sufficiency Checker。你的工作不是重新排序 evidence，而是判斷目前已篩出的 direct/supporting evidence 是否足以完整回答使用者問題。{conflict_instruction}

使用者問題：
{query}

系統 deterministic 判定：
rule_sensitive={str(rule_sensitive).lower()}
若 rule_sensitive=true，必須套用下方所有 rule-sensitive 限制；不得自行把它降級成一般問答。

目前可用 evidence：
{chr(10).join(blocks)}

判斷規則：

1. 先把使用者真正要求回答的資訊拆成 query_aspects；只拆實質需求，不要把同一需求過度細分。

2. sufficient=true 的條件是：
   每一個「回答核心問題所必要的 aspect」都能由目前 evidence 直接支持，
   或由多筆 evidence 的明確內容安全組合支持。

3. 不要求某一筆 evidence 必須逐字回答完整問題。
   不同 evidence 可以共同建立定義、關係、比較、結構或結論。

4. 對正式操作規則、MARC/RDA 欄位/指標/分欄、代碼選擇、
   分類/作者號規則、版本差異或例外，仍允許 evidence_compositional。

   關鍵不是「是否存在完全相同情境的單一 evidence」，
   而是：
   - 所需結論是否可以由 evidence 中已明確存在的定義、關係、
     結構、規則或多個相互補充的例子建立；
   - 組合過程是否不需要新增 evidence 未提供的正式規則、
     例外、條件、門檻或適用範圍。

   如果必須自行補上一條新的正式規則才能完成答案，
   才必須判 insufficient。

5. constraint alignment 必須嚴格：
   欄位、位址、格式/版本、實體、代碼角色、關係及適用條件若不一致，
   不得視為同一規則的直接支持。

6. 多意圖、比較、條件、例外、兩段式問題必須逐項覆蓋。
   但如果使用者問的是「兩種情況是否有差異」，
   evidence 已足以證明兩者具有不同結構角色或適用結構時，
   不要求 evidence 必須逐字出現「兩者有差異」這句話。

7. 枚舉型問題（有哪些、全部、還有其他）不能只找到一例就 sufficient。

8. evidence_relationship 只描述目前 relevant evidence 的整體語意關係：
   - compatible：可以同時成立，沒有實質衝突。
   - complementary：回答不同部分，可安全組合。
   - conditional：結論不同是因適用條件不同，並非真正矛盾。
   - conflict：相同規則範圍、相同適用條件下得到互斥結論。
   - unknown：證據不足以判斷關係。

9. conflict_groups 只列值得稽核的 evidence 組。
   same_scope=true 只有在格式/欄位/規範/資源類型/適用條件
   足以視為同一規則範圍時成立。

10. 不要用日期替 evidence 判定語意上誰對誰錯。
    日期解析與「最新優先」由 Python 後處理；
    你只負責判斷是否是真正 same-scope conflict。

11. coverage_mode 只能使用以下三種：

    - evidence_grounded：
      evidence 已直接或明確建立回答核心問題所需的規則或結論。

    - evidence_compositional：
      沒有單一 evidence 完整回答問題，
      但多筆 evidence 的明確內容可以安全組合成核心答案，
      且組合沒有創造新的正式規則。

    - insufficient：
      回答核心問題仍需要補上一個 evidence 未支持的關鍵事實、
      正式規則、適用條件、例外或推測。

12. evidence_compositional 的判定原則：

    可以：
    - 組合不同 evidence 的明確定義。
    - 組合不同 evidence 的結構關係。
    - 組合一般規則與明確例子。
    - 從多筆 evidence 已明示的關係得出直接結構性結論。

    不可以：
    - 只因語意相似就推導規則。
    - 把單一案例提升成普遍規則。
    - 新增 evidence 沒有提到的例外、門檻或適用條件。
    - 使用模型既有知識補完正式規範。
    - 把「可能」自行改成「一定」。

13. 判斷 sufficient 時，要區分「核心結論」與「額外細節」。

    如果 evidence 足以回答使用者的核心問題，
    但不足以回答某些更細的原因、例外或完整規則，
    可以：
    - sufficient=true
    - coverage_mode=evidence_compositional
    - 在 reason 中指出哪些細節仍未被 evidence 明確支持。

    不要因為缺少非必要的額外細節，
    就把整個問題判成 insufficient。

14. 對 rule-sensitive 問題尤其保守：

    如果 sufficient=true 且 coverage_mode=evidence_compositional，
    每一個核心 claim 都必須可以指出對應 evidence_ids。

    如果某一個核心 claim 需要依賴：
    - 常識
    - 模型既有知識
    - 類比
    - 猜測
    - 未提供的正式規則

    則該 aspect 必須視為 uncovered。

15. 對 rule-sensitive 問題，必須區分「一般正式規則」與「具體案例」。

    若使用者詢問的是一般規則，例如：
    - 某類資料應如何著錄
    - 某欄位應使用哪個值
    - 某類資源一般如何處理

    而目前支持核心結論的 evidence 只有單一書目、單一作品、
    單一案例、範例或「例如」型 evidence：

    - 該 evidence 只能直接支持該具體案例。
    - 不得僅依該案例把一般規則判為 evidence_grounded。
    - 不得把該案例自動提升成所有同類資料都適用的通則。
    - 必須另有 evidence 明確提供一般規則、定義、適用條件，
      或由多筆 evidence 建立不需新增規則的完整規則鏈，
      才能判 sufficient=true。

    若只有具體案例而缺少一般規則：
    - sufficient=false
    - coverage_mode=insufficient
    - 該一般規則 aspect 必須列入 missing_aspects

    例外：若使用者詢問的本身就是與該 evidence 相同或等價的
    具體案例，該案例 evidence 可以直接支持該案例答案。

16. covered_aspects / conflict_groups 的 evidence_ids
    只能使用目前提供的 ID。

17. reason 只寫簡短、可稽核理由，
    說明哪些 evidence 支持哪些核心 aspect，
    不要輸出詳細思考過程。
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

        raw_groups = []
        if self.enable_conflict_analysis:
            for group in result.get("conflict_groups", []):
                evidence_ids = [
                    str(eid).strip()
                    for eid in group.get("evidence_ids", [])
                    if str(eid).strip() in allowed_ids
                ]
                if len(set(evidence_ids)) < 2:
                    continue
                relationship = str(group.get("relationship", "unknown")).strip()
                if relationship not in relationship_values:
                    relationship = "unknown"
                raw_groups.append(
                    {
                        "evidence_ids": list(dict.fromkeys(evidence_ids)),
                        "relationship": relationship,
                        "same_scope": bool(group.get("same_scope", False)),
                        "reason": str(group.get("reason", "")).strip(),
                    }
                )

        conflict_resolution = self._resolve_conflict_groups(raw_groups, relevant)
        conflict_detected = conflict_resolution["status"] in {
            "resolved_by_date",
            "unresolved",
        }

        sufficient = bool(result.get("sufficient", False)) and not missing_aspects
        if conflict_resolution["status"] == "unresolved":
            sufficient = False
            marker = "現有 evidence 在相同適用條件下存在未解決衝突"
            if marker not in missing_aspects:
                missing_aspects.append(marker)

        coverage_mode = str(result.get("coverage_mode", "insufficient")).strip()
        if coverage_mode == "compositional":
            # Backward compatibility for older traces/models.
            coverage_mode = "evidence_compositional"
        if coverage_mode not in {
            "evidence_grounded",
            "evidence_compositional",
            "insufficient",
        }:
            coverage_mode = "insufficient"
        if not sufficient:
            coverage_mode = "insufficient"

        evidence_relationship = str(
            result.get("evidence_relationship", "unknown")
        ).strip()
        if evidence_relationship not in relationship_values:
            evidence_relationship = "unknown"

        return {
            "sufficient": sufficient,
            "query_aspects": query_aspects,
            "covered_aspects": covered_aspects,
            "missing_aspects": missing_aspects,
            "reason": str(result.get("reason", "")).strip(),
            "relevant_evidence_count": len(relevant),
            "direct_evidence_count": len(direct),
            "checked_by_llm": True,
            "rule_sensitive": bool(rule_sensitive),
            "coverage_mode": coverage_mode,
            "evidence_relationship": evidence_relationship,
            "conflict_detected": conflict_detected,
            "conflict_resolution": conflict_resolution,
        }
