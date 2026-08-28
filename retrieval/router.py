from dataclasses import dataclass

from llm_client import LLMClientError
from retriever import BM25


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
    """V4 one-shot Tree retrieval planner.

    Unlike V3, this router does not make separate L1/L2/L3 LLM calls.
    It first uses local BM25 over taxonomy leaf cards to create a small candidate
    pool, then uses exactly one LLM call to rank complete Tree paths.

    Tree routing is therefore a retrieval prior, not a hard evidence filter.
    """

    def __init__(
        self,
        taxonomy,
        llm,
        model,
        think=False,
        candidate_pool=16,
        top_paths=3,
    ):
        self.taxonomy = taxonomy
        self.llm = llm
        self.model = model
        self.think = bool(think)
        self.candidate_pool = max(3, int(candidate_pool))
        self.top_paths = max(1, int(top_paths))

        self._path_specs = self._build_terminal_path_specs()
        self._path_cards = [self._compact_path_card(spec) for spec in self._path_specs]
        self._path_bm25 = BM25(self._path_cards)

    def _build_terminal_path_specs(self):
        specs = []
        for l1 in self.taxonomy.l1_nodes():
            for l2 in self.taxonomy.l2_nodes(l1):
                l3s = self.taxonomy.l3_nodes(l1, l2)
                if l3s:
                    for l3 in l3s:
                        specs.append({"l1": l1, "l2": l2, "l3": l3})
                else:
                    specs.append({"l1": l1, "l2": l2, "l3": None})
        return specs

    @staticmethod
    def _profile_text(profile):
        parts = [
            str(profile.get("description", "")).strip(),
            str(profile.get("example", "")).strip(),
            str(profile.get("boundary", "")).strip(),
        ]
        return "\n".join(x for x in parts if x)

    def _compact_path_card(self, spec):
        l1 = spec["l1"]
        l2 = spec["l2"]
        l3 = spec.get("l3")
        path = " > ".join(x for x in (l1, l2, l3) if x)

        sections = [f"完整路徑：{path}"]
        sections.append(
            "L1：" + self._profile_text(self.taxonomy.node_profile("L1", node=l1))
        )
        sections.append(
            "L2："
            + self._profile_text(
                self.taxonomy.node_profile("L2", l1=l1, node=l2)
            )
        )
        if l3:
            sections.append(
                "L3："
                + self._profile_text(
                    self.taxonomy.node_profile("L3", l1=l1, l2=l2, node=l3)
                )
            )
        return "\n".join(sections)

    def _lexical_shortlist(self, query, pool=None):
        if not self._path_specs:
            return []
        pool = min(max(1, int(pool or self.candidate_pool)), len(self._path_specs))
        scores = self._path_bm25.scores(query)
        order = sorted(
            range(len(self._path_specs)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:pool]
        return [
            {
                "spec": self._path_specs[i],
                "card": self._path_cards[i],
                "lexical_score": float(scores[i]),
                "lexical_rank": rank,
            }
            for rank, i in enumerate(order, start=1)
        ]

    def lexical_paths(self, query, top_k=3):
        """Local-only Tree paths for refinement; zero LLM/API calls."""
        short = self._lexical_shortlist(query, pool=max(top_k, self.candidate_pool))
        if not short:
            return []
        best = max(float(x["lexical_score"]) for x in short) or 1.0
        paths = []
        for item in short[:top_k]:
            spec = item["spec"]
            score = max(0.0, min(1.0, float(item["lexical_score"]) / best))
            paths.append(
                RoutePath(
                    l1=spec["l1"],
                    l2=spec["l2"],
                    l3=spec.get("l3"),
                    score=score,
                    trace=[
                        {
                            "level": "PATH_LOCAL",
                            "node": " > ".join(
                                x for x in (spec["l1"], spec["l2"], spec.get("l3")) if x
                            ),
                            "score": score,
                            "reason": "Evidence-guided local taxonomy BM25 refinement",
                        }
                    ],
                )
            )
        return paths

    def plan(self, query):
        shortlist = self._lexical_shortlist(query)
        if not shortlist:
            return {"paths": [], "search_query": query, "query_focus": query, "fallback": True}

        ids = [f"P{i}" for i in range(1, len(shortlist) + 1)]
        mapping = dict(zip(ids, shortlist))
        candidate_text = "\n\n".join(
            f"[{pid}] lexical_rank={item['lexical_rank']}\n{item['card']}"
            for pid, item in zip(ids, shortlist)
        )

        max_items = min(self.top_paths, len(shortlist))
        schema = {
            "type": "object",
            "properties": {
                "query_focus": {"type": "string"},
                "search_query": {"type": "string"},
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
            "required": ["query_focus", "search_query", "selected"],
        }

        prompt = f"""你是國家圖書館 Tree-RAG 的一次性 Retrieval Planner。
你的工作只有兩件事：
1. 從候選 taxonomy 完整路徑中選出最可能包含答案證據的前 {max_items} 條路徑。
2. 產生一個適合 BM25 檢索的 search_query。

使用者問題：
{query}

候選完整路徑：
{candidate_text}

規則：
- Tree path 是搜尋優先序，不代表答案一定在該節點。
- 保留跨節點可能性；不要只因第一名合理就忽略其他合理路徑。
- search_query 必須保留使用者明確寫出的 MARC/RDA 欄位、位址、indicator、subfield、代碼或名稱。
- search_query 可加入候選 taxonomy 卡片中已出現的同義術語/英文術語。
- 不得把你自己猜測的答案內容、未出現在使用者問題或 taxonomy 卡片中的規則值塞進 search_query。
- 不要回答使用者問題。
"""

        try:
            result = self.llm.chat_json(
                self.model,
                [{"role": "user", "content": prompt}],
                schema,
                temperature=0.0,
                think=self.think,
            )
        except LLMClientError:
            # Availability fallback: keep the system useful even if planner API fails.
            local_paths = self.lexical_paths(query, top_k=max_items)
            return {
                "paths": local_paths,
                "search_query": query,
                "query_focus": query,
                "fallback": True,
            }

        paths = []
        seen = set()
        for item in result.get("selected", []):
            pid = str(item.get("candidate_id", "")).strip()
            if pid not in mapping:
                continue
            spec = mapping[pid]["spec"]
            key = (spec["l1"], spec["l2"], spec.get("l3"))
            if key in seen:
                continue
            seen.add(key)
            score = max(0.0, min(1.0, float(item.get("score", 0.0))))
            path_text = " > ".join(x for x in key if x)
            paths.append(
                RoutePath(
                    l1=spec["l1"],
                    l2=spec["l2"],
                    l3=spec.get("l3"),
                    score=score,
                    trace=[
                        {
                            "level": "PATH",
                            "node": path_text,
                            "score": score,
                            "reason": str(item.get("reason", "")).strip(),
                            "query_focus": str(result.get("query_focus", "")).strip(),
                        }
                    ],
                )
            )

        if not paths:
            paths = self.lexical_paths(query, top_k=max_items)

        paths.sort(key=lambda x: x.score, reverse=True)
        return {
            "paths": paths[:max_items],
            "search_query": str(result.get("search_query", "")).strip() or query,
            "query_focus": str(result.get("query_focus", "")).strip() or query,
            "fallback": False,
        }

    # Backward-compatible method name used by earlier callers.
    def route_tree(self, query, l1_beam=2, l2_global_beam=3, final_beam=3):
        return self.plan(query)["paths"][:final_beam]
