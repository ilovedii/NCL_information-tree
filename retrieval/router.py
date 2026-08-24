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
        return " > ".join(x for x in (self.l1, self.l2, self.l3) if x)


class TreeRouter:
    def __init__(self, taxonomy, llm, model, think=True, max_choices=3):
        self.taxonomy = taxonomy
        self.llm = llm
        self.model = model
        self.think = think
        self.max_choices = max_choices

    def _route(self, query, current_path, level, candidates):
        candidates = list(candidates)
        if not candidates:
            return []
        ids = [f"C{i + 1}" for i in range(len(candidates))]
        mapping = dict(zip(ids, candidates))
        cards = []
        for cid, node in zip(ids, candidates):
            if level == "L1":
                card = self.taxonomy.node_card("L1", node=node)
            elif level == "L2":
                card = self.taxonomy.node_card("L2", l1=current_path.l1, node=node)
            else:
                card = self.taxonomy.node_card("L3", l1=current_path.l1, l2=current_path.l2, node=node)
            cards.append(f"[{cid}]\n{card}")
        max_items = min(self.max_choices, len(candidates))
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
                            "candidate_id": {"type": "string", "enum": ids},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "reason": {"type": "string"},
                        },
                        "required": ["candidate_id", "score", "reason"],
                    },
                },
            },
            "required": ["query_focus", "uncertain", "selected"],
        }
        prompt = f"""你是國家圖書館知識分類樹的路由器。你的唯一工作是根據正式 taxonomy 判斷下一個節點，不要回答使用者問題。

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
4. 優先選最具體且符合分類邊界的節點。
5. 一般保留 1 到 2 個；只有跨類或邊界確實不清時才增加候選，最多 {max_items} 個。
6. score 為 0 到 1 的節點相關程度。
7. reason 只寫可稽核的短理由，指出符合哪個正式定義或邊界，不要輸出詳細思考過程。
8. 必須只使用候選 ID，不得建立新的分類。
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
                    "score": float(item.get("score", 0.0)),
                    "reason": str(item.get("reason", "")).strip(),
                    "query_focus": str(result.get("query_focus", "")).strip(),
                    "uncertain": bool(result.get("uncertain", False)),
                }
            )
        if not output:
            raise OllamaError("Router 沒有選出有效候選節點")
        output.sort(key=lambda x: x["score"], reverse=True)
        return output

    def route_tree(self, query, l1_beam=2, l2_global_beam=3, final_beam=4):
        l1_decisions = self._route(query, None, "L1", self.taxonomy.l1_nodes())
        l1_paths = [
            RoutePath(
                l1=decision["node"],
                l2=None,
                l3=None,
                score=max(decision["score"], 1e-6),
                trace=[{"level": "L1", **decision}],
            )
            for decision in l1_decisions[:l1_beam]
        ]
        l2_paths = []
        for path in l1_paths:
            decisions = self._route(query, path, "L2", self.taxonomy.l2_nodes(path.l1))
            for decision in decisions:
                l2_paths.append(
                    RoutePath(
                        l1=path.l1,
                        l2=decision["node"],
                        l3=None,
                        score=path.score * max(decision["score"], 1e-6),
                        trace=path.trace + [{"level": "L2", **decision}],
                    )
                )
        l2_paths.sort(key=lambda x: x.score, reverse=True)
        l2_paths = l2_paths[:l2_global_beam]
        final_paths = []
        for path in l2_paths:
            children = self.taxonomy.l3_nodes(path.l1, path.l2)
            if not children:
                final_paths.append(path)
                continue
            decisions = self._route(query, path, "L3", children)
            for decision in decisions:
                final_paths.append(
                    RoutePath(
                        l1=path.l1,
                        l2=path.l2,
                        l3=decision["node"],
                        score=path.score * max(decision["score"], 1e-6),
                        trace=path.trace + [{"level": "L3", **decision}],
                    )
                )
        final_paths.sort(key=lambda x: x.score, reverse=True)
        return final_paths[:final_beam]