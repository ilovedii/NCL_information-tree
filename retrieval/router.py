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
    """V4.3 one-shot Tree retrieval planner.

    One remote LLM call performs:
      1. domain/scope gating,
      2. semantic Atomic Retrieval Unit planning,
      3. BM25-focused query/keyword generation,
      4. complete taxonomy-path ranking.

    The planner defaults to ONE retrieval unit. It only decomposes a user query
    when distinct parts require different rules/evidence. Tree routing remains a
    retrieval prior rather than a hard evidence filter.
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
                                x
                                for x in (spec["l1"], spec["l2"], spec.get("l3"))
                                if x
                            ),
                            "score": score,
                            "reason": "Evidence-guided local taxonomy BM25 refinement",
                        }
                    ],
                )
            )
        return paths

    @staticmethod
    def _fallback_unit(query):
        return [{"query": query, "keywords": []}]

    def plan(self, query):
        query = str(query or "").strip()
        shortlist = self._lexical_shortlist(query)

        # Fail-open: taxonomy/planner availability problems must not reject a
        # legitimate cataloging question.
        if not shortlist:
            return {
                "in_scope": True,
                "paths": [],
                "search_query": query,
                "query_focus": query,
                "retrieval_units": self._fallback_unit(query),
                "fallback": True,
            }

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
                "in_scope": {"type": "boolean"},
                "query_focus": {"type": "string"},
                # Compatibility field retained for older downstream code.
                "search_query": {"type": "string"},
                "retrieval_units": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "keywords": {
                                "type": "array",
                                "minItems": 0,
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query", "keywords"],
                    },
                },
                "selected": {
                    "type": "array",
                    # out_of_scope is allowed to return no path.
                    "minItems": 0,
                    "maxItems": max_items,
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string", "enum": ids},
                            "score": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["candidate_id", "score", "reason"],
                    },
                },
            },
            "required": [
                "in_scope",
                "query_focus",
                "search_query",
                "retrieval_units",
                "selected",
            ],
        }

        prompt = f"""你是國家圖書館 Tree-RAG 的一次性 Retrieval Planner。
你只負責檢索規劃，不回答使用者問題。

你的工作：
1. 判斷問題是否屬於圖書館編目、分類或書目組織領域。
2. 將問題整理成 1 到 3 個 Atomic Retrieval Units。
3. 為每個 unit 產生適合 BM25 的 focused query 與 keywords。
4. 從候選 taxonomy 完整路徑中選出最可能包含證據的前 {max_items} 條路徑。

使用者問題：
{query}

候選完整路徑：
{candidate_text}

【Scope Gate】
- MARC、RDA、分類法、作者號、主題標目、權威控制、書目紀錄、
  ISBN/ISSN、題名、著者、出版、索書號、圖書館編目實務等相關問題：
  in_scope=true。
- 明顯與上述領域無關：in_scope=false。
- 不確定時一律 in_scope=true（fail-open）。
- 資料庫沒有足夠答案，不等於 out-of-scope。
- 若 in_scope=false：retrieval_units=[]、selected=[]、query_focus=""、
  search_query=""。

【Atomic Retrieval Unit：最重要的規則】
- 預設只產生 1 個 retrieval unit，不要為了拆題而拆題。
- 每個 unit 必須是一個「可由一組一致 evidence 獨立回答的完整知識需求」。
- 不得因為句子有兩個問號、「如果」、「還是」、「以及」就機械式拆分。
- 多個條件若屬於同一規則、同一比較、同一版本對照、同一決策，
  或可由同一組完整 evidence 一起回答，維持 1 個 unit。
- 只有當不同部分確實需要不同知識規則或不同 evidence 才拆成 2 或 3 個 unit。
- 最多 3 個 unit。

【拆分後必須改寫成完整、可獨立理解的句子】
- 如果拆成 2～3 個 units，每一個 unit 都必須是「單獨拿出來給另一個人看，也知道在問什麼」的完整問題。
- 不可以留下依賴前文才能理解的殘句，例如「那第二種呢？」、「如果同一冊又有兩個很接近的號碼呢？」、「這時應先看什麼？」。
- 遇到「它、此、這種、前者、後者、第一種、第二種、同一冊、又、上述、前述」等承接前文的說法時，必須從原始問題複製必要上下文到該 unit，將指涉補完整。
- 補回的內容只能來自使用者原始問題中已經明確出現的資訊；不得自行加入答案、規則結論或使用者未提到的專業判斷。
- 特別要保留會影響檢索的專業限定，例如 MARC/RDA 欄位、indicator、subfield、Leader/LDR 位址、分類法或規範名稱、版本、類號、索書號體系、資料類型與比較對象。
- 完整性優先於簡短。可以稍微重複原問題中的必要詞語，不能為了精簡而讓 unit 失去上下文。

例 1：
「兔子飼養在2007年版與增訂七版各用什麼類號？」
這是同一版本對照知識，應為 1 unit。

例 2：
原問題：「050欄位第二指標為4時，可以直接當正式索書號嗎？如果同一冊又有兩個很接近的號碼，應先看什麼？」
可拆成 2 units，但第二個 unit 不可只寫「同一冊有兩個很接近的號碼時應先看什麼？」。
較好的完整寫法是：
「在 MARC 050／LC 索書號的情境下，如果同一冊出現兩個很接近的索書號，應依據什麼資訊判斷應採用哪一個？」
注意：這只是補回原題上下文，不得自行加入 LC Authorities、Shelflisting、Main Entry 等答案端術語。

【focused query】
- unit.query 本身就是主要檢索問題，因此必須是完整自然句，不是關鍵字片段。
- 保留真正決定答案的核心概念與條件。
- 必須保留使用者明確寫出的 MARC/RDA 欄位、Leader/LDR 位址、indicator、subfield、代碼、類號、版本或專有名稱。
- 若 unit 原本依賴另一句的上下文，先補完整再產生 query；不要只靠 keywords 補救殘缺 query。
- 可去除語氣詞與確實不影響檢索的背景敘述。
- 不得加入你自己猜測的答案值或答案端術語。

【keywords】
- 先形成完整 unit.query，再從該完整問題抽取 keywords。
- keywords 是 BM25 lexical anchors，不是答案，也不能取代完整 query。
- 優先保留高鑑別力的術語、欄位、indicator、subfield、代碼、類號、版本、專名與核心概念。
- 若某個 unit 補回了原問題中的必要上下文，keywords 也應保留其中最重要的專業詞。
- 可加入候選 taxonomy 卡片中已明確出現、且不改變原意的同義詞或英文術語。
- 不得加入使用者問題與 taxonomy 候選中都沒有、只是你猜測的答案。
例如問「寬恕沒有分類專號時應歸哪裡」，可用
["寬恕", "分類", "專號"]；不可自行猜成 ["寬恕", "199.8", "愛"]。

【Tree path】
- Tree path 是搜尋優先序，不是答案位置的硬限制。
- 保留跨節點可能性；不要只因第一名合理就忽略其他合理路徑。
- reason 只寫簡短可稽核理由。
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
            local_paths = self.lexical_paths(query, top_k=max_items)
            return {
                "in_scope": True,
                "paths": local_paths,
                "search_query": query,
                "query_focus": query,
                "retrieval_units": self._fallback_unit(query),
                "fallback": True,
            }

        in_scope = bool(result.get("in_scope", True))
        if not in_scope:
            return {
                "in_scope": False,
                "paths": [],
                "search_query": "",
                "query_focus": "",
                "retrieval_units": [],
                "fallback": False,
            }

        query_focus = str(result.get("query_focus", "")).strip() or query
        search_query = str(result.get("search_query", "")).strip() or query

        # Normalize retrieval units conservatively. Missing/invalid decomposition
        # falls back to exactly one unit rather than breaking retrieval.
        retrieval_units = []
        seen_unit_queries = set()
        for raw_unit in result.get("retrieval_units", [])[:3]:
            unit_query = str(raw_unit.get("query", "")).strip()
            if not unit_query:
                continue
            unit_key = unit_query.casefold()
            if unit_key in seen_unit_queries:
                continue
            seen_unit_queries.add(unit_key)

            keywords = []
            seen_keywords = set()
            for raw_kw in raw_unit.get("keywords", [])[:8]:
                kw = str(raw_kw or "").strip()
                if not kw:
                    continue
                key = kw.casefold()
                if key in seen_keywords:
                    continue
                seen_keywords.add(key)
                keywords.append(kw)

            retrieval_units.append(
                {
                    "query": unit_query,
                    "keywords": keywords,
                }
            )

        if not retrieval_units:
            retrieval_units = self._fallback_unit(search_query)

        paths = []
        seen_paths = set()
        for item in result.get("selected", []):
            pid = str(item.get("candidate_id", "")).strip()
            if pid not in mapping:
                continue

            spec = mapping[pid]["spec"]
            key = (spec["l1"], spec["l2"], spec.get("l3"))
            if key in seen_paths:
                continue
            seen_paths.add(key)

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
                            "query_focus": query_focus,
                        }
                    ],
                )
            )

        # Lexical path fallback only for an in-domain query.
        if not paths:
            paths = self.lexical_paths(query, top_k=max_items)

        paths.sort(key=lambda x: x.score, reverse=True)
        return {
            "in_scope": True,
            "paths": paths[:max_items],
            "search_query": search_query,
            "query_focus": query_focus,
            "retrieval_units": retrieval_units,
            "fallback": False,
        }

    # Backward-compatible method name used by earlier callers.
    def route_tree(self, query, l1_beam=2, l2_global_beam=3, final_beam=3):
        return self.plan(query)["paths"][:final_beam]
