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

【拆分後必須保留 shared context】
- 如果拆成 2～3 個 units，每個 unit 都必須單獨可理解，而且保留原問題中會改變檢索範圍的必要條件。
- 遇到「它、此、這種、前者、後者、第一種、第二種、同一冊、又、上述、前述、這時」等承接前文的表達，
  必須把原問題中必要的 shared context 補回該 unit。
- 特別不得遺失：MARC/CMARC/RDA、欄位號碼、indicator、subfield、Leader/LDR 位址、
  分類法或規範名稱與版本、類號、索書號體系、資料類型、系統名稱與比較對象。
- 補回的內容只能來自使用者原始問題；不得加入答案、規則結論或使用者未提供的專業判斷。
- 完整性優先於簡短；可以重複必要專業詞，不能為了縮短 query 而丟失 context。

合成例 1（示範不應過度拆分）：
「某個本館自訂欄位在舊版與新版作業規範中的定義是否改變，以及前後如何對照？」
這是同一個版本對照知識，若可由同一組 evidence 回答，維持 1 unit。

合成例 2（示範拆分後要保留 shared context）：
原問題：
「在本館自訂 9XX 欄位的作業規範中，第一指標的值要怎麼判讀？
如果同一筆紀錄又有另一個 9XX 欄位，兩者衝突時應怎麼查核？」
若確實需要不同 evidence，可拆成 2 units；但第二個 unit 仍須保留
「本館自訂 9XX 欄位的作業規範」這個 shared context，
不能只寫「兩者衝突時應怎麼查核？」。
注意：例子只示範 context preservation，不提供任何答案值。

【focused query】
- unit.query 本身就是主要檢索問題，必須是完整自然句，不是關鍵字片段。
- 先補完整 shared context，再形成 focused query；不能只靠 keywords 補救殘缺 query。
- 必須保留會改變答案範圍的專業限制與明確排除條件。
- 可去除語氣詞與確實不影響檢索的背景敘述。
- 不得加入自行猜測的答案值或答案端術語。

【keywords】
- 先形成完整 unit.query，再抽取 keywords。
- keywords 是 BM25 的「正向 lexical anchors」，不是完整語義，也不是答案。
- 優先保留高鑑別力的術語、欄位、indicator、subfield、代碼、類號、版本、專名與正向核心概念。
- 若原問題明確寫「不是 X」、「不屬於 X」、「排除 X」、「不要 X」、「非 X」，
  X 必須留在 unit.query 作為語義限制，但不要把 X 單獨列成正向 keyword，避免 BM25 反而大量召回被排除的主題。
- 若問題是在 A 與 B 之間真正做比較，而不是排除其中一方，A、B 都可以保留為 keywords。
- 不得加入使用者問題與 taxonomy 候選中都沒有、只是你猜測的答案。

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
