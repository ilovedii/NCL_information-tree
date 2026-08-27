from dataclasses import dataclass

from llm_client import OllamaError


@dataclass
class RoutePath:
    l1: str
    l2: str | None
    l3: str | None
    score: float
    trace: list

    def key(self):
        return self.l1, self.l2, self.l3

    def display(self):
        return " > ".join(
            x for x in (self.l1, self.l2, self.l3) if x
        )


class TreeRouter:
    """LLM taxonomy router that preserves ranked alternatives at each level.

    The router proposes hypotheses. It does not decide truth.
    Evidence/Sufficiency later verify whether a routed path is useful.
    """

    def __init__(
        self,
        taxonomy,
        llm,
        model,
        think=True,
        max_choices=3,
        l1_max_choices=None,
        l2_max_choices=None,
        l3_max_choices=None,
    ):
        self.taxonomy = taxonomy
        self.llm = llm
        self.model = model
        self.think = think

        # Backward-compatible default.
        self.max_choices = int(max_choices)

        self.l1_max_choices = int(
            l1_max_choices
            if l1_max_choices is not None
            else max_choices
        )
        self.l2_max_choices = int(
            l2_max_choices
            if l2_max_choices is not None
            else max_choices
        )
        self.l3_max_choices = int(
            l3_max_choices
            if l3_max_choices is not None
            else max_choices
        )

    def _route(
        self,
        query,
        current_path,
        level,
        candidates,
        max_choices=None,
    ):
        candidates = list(candidates)
        if not candidates:
            return []

        ids = [f"C{i + 1}" for i in range(len(candidates))]
        mapping = dict(zip(ids, candidates))

        cards = []
        for cid, node in zip(ids, candidates):
            if level == "L1":
                card = self.taxonomy.node_card(
                    "L1",
                    node=node,
                )
            elif level == "L2":
                card = self.taxonomy.node_card(
                    "L2",
                    l1=current_path.l1,
                    node=node,
                )
            else:
                card = self.taxonomy.node_card(
                    "L3",
                    l1=current_path.l1,
                    l2=current_path.l2,
                    node=node,
                )

            cards.append(f"[{cid}]\n{card}")

        if max_choices is None:
            if level == "L1":
                max_choices = self.l1_max_choices
            elif level == "L2":
                max_choices = self.l2_max_choices
            else:
                max_choices = self.l3_max_choices

        max_items = min(
            max(1, int(max_choices)),
            len(candidates),
        )

        schema = {
            "type": "object",
            "properties": {
                "query_focus": {"type": "string"},
                "uncertain": {"type": "boolean"},
                "selected": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max_items,
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {
                                "type": "string",
                                "enum": ids,
                            },
                            "score": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "candidate_id",
                            "score",
                            "reason",
                        ],
                    },
                },
            },
            "required": [
                "query_focus",
                "uncertain",
                "selected",
            ],
        }

        if level == "L3":
            ranking_instruction = f"""
9. 這是 L3 路由。請把「最可能包含可回答證據」的節點排前面。
10. 為支援 evidence-driven backtracking，可保留最多 {max_items} 個有合理可能性的 L3 候選；
    不要因第一名很明顯就只保留一個。若第二、第三候選仍有可能包含跨節點或歷史分類下的證據，
    應保留並給較低分數。
"""
        else:
            ranking_instruction = f"""
9. 最多保留 {max_items} 個真正合理的候選，並依 score 由高到低排序。
"""

        prompt = f"""你是國家圖書館知識分類樹的路由器。
你的工作是根據正式 taxonomy 提出「可能包含答案證據的 ranked node hypotheses」，
不是直接回答使用者問題，也不是宣告第一名一定正確。

使用者問題：
{query}

目前路徑：
{current_path.display() if current_path else "ROOT"}

候選節點：
{chr(10).join(cards)}

判斷規則：
1. 正式定義、正式例子與分類邊界是主要判斷依據；代表資料只能作輔助。
2. 判斷使用者真正想解決的規則、操作、概念或知識類型，不要只依關鍵字。
3. 必須遵守目前父節點，不得跨越 taxonomy 父子關係。
4. 第一名應是最具體、最可能包含答案證據的節點。
5. Router 是 proposal/ranking，不是 truth decision；Evidence Selector 之後會驗證。
6. score 為 0 到 1 的節點相關程度。
7. reason 只寫可稽核的短理由，指出符合哪個正式定義或邊界。
8. 必須只使用候選 ID，不得建立新的分類。
{ranking_instruction}
"""

        result = self.llm.chat_json(
            self.model,
            [{"role": "user", "content": prompt}],
            schema,
            temperature=0.0,
            think=self.think,
        )

        output = []
        seen = set()

        for item in result.get("selected", []):
            cid = item.get("candidate_id")

            if cid not in mapping or cid in seen:
                continue

            seen.add(cid)

            output.append(
                {
                    "node": mapping[cid],
                    "score": float(
                        item.get("score", 0.0)
                    ),
                    "reason": str(
                        item.get("reason", "")
                    ).strip(),
                    "query_focus": str(
                        result.get("query_focus", "")
                    ).strip(),
                    "uncertain": bool(
                        result.get("uncertain", False)
                    ),
                }
            )

        if not output:
            raise OllamaError(
                "Router 沒有選出有效候選節點"
            )

        output.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return output

    def route_tree(
        self,
        query,
        l1_beam=2,
        l2_global_beam=3,
        final_beam=12,
    ):
        # L1 ranking.
        l1_decisions = self._route(
            query,
            None,
            "L1",
            self.taxonomy.l1_nodes(),
            max_choices=self.l1_max_choices,
        )

        l1_paths = [
            RoutePath(
                l1=decision["node"],
                l2=None,
                l3=None,
                score=max(
                    decision["score"],
                    1e-6,
                ),
                trace=[
                    {
                        "level": "L1",
                        **decision,
                    }
                ],
            )
            for decision in l1_decisions[:l1_beam]
        ]

        # L2 ranking under each retained L1.
        l2_paths = []

        for path in l1_paths:
            decisions = self._route(
                query,
                path,
                "L2",
                self.taxonomy.l2_nodes(
                    path.l1
                ),
                max_choices=self.l2_max_choices,
            )

            for decision in decisions:
                l2_paths.append(
                    RoutePath(
                        l1=path.l1,
                        l2=decision["node"],
                        l3=None,
                        score=(
                            path.score
                            * max(
                                decision["score"],
                                1e-6,
                            )
                        ),
                        trace=(
                            path.trace
                            + [
                                {
                                    "level": "L2",
                                    **decision,
                                }
                            ]
                        ),
                    )
                )

        l2_paths.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        l2_paths = l2_paths[
            :l2_global_beam
        ]

        # L3 ranking under each retained L2.
        final_paths = []

        for path in l2_paths:
            children = self.taxonomy.l3_nodes(
                path.l1,
                path.l2,
            )

            if not children:
                final_paths.append(path)
                continue

            decisions = self._route(
                query,
                path,
                "L3",
                children,
                max_choices=self.l3_max_choices,
            )

            for decision in decisions:
                final_paths.append(
                    RoutePath(
                        l1=path.l1,
                        l2=path.l2,
                        l3=decision["node"],
                        score=(
                            path.score
                            * max(
                                decision["score"],
                                1e-6,
                            )
                        ),
                        trace=(
                            path.trace
                            + [
                                {
                                    "level": "L3",
                                    **decision,
                                }
                            ]
                        ),
                    )
                )

        final_paths.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return final_paths[:final_beam]
