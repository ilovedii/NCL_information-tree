class ContextBuilder:
    def __init__(self, taxonomy):
        self.taxonomy = taxonomy

    def _route_text(self, paths):
        lines = []
        for i, path in enumerate(paths, start=1):
            lines.append(f"路徑 {i}：{path.display()} | path_score={path.score:.4f}")
            for step in path.trace:
                lines.append(
                    f"  {step['level']}：{self.taxonomy.node_label(step['node'])} | "
                    f"score={step['score']:.3f} | {step['reason']}"
                )
        return "\n".join(lines)

    def _cards_text(self, paths):
        cards = []
        seen = set()
        for path in paths:
            for card in self.taxonomy.path_node_cards(path):
                if card in seen:
                    continue
                seen.add(card)
                cards.append(card)
        return "\n\n".join(cards)

    def _knowledge_pack_text(self, knowledge_packs):
        blocks = []
        for pack in knowledge_packs:
            blocks.append(
                "\n".join(
                    [
                        f"role={pack.get('role', 'node')}",
                        f"path={pack.get('path', '')}",
                        f"source={pack.get('static_source', 'runtime')}",
                        f"documents={pack.get('document_count', 0)}",
                        f"date_range={pack.get('date_start', '')} ~ {pack.get('date_end', '')}",
                        f"knowledge_units={len(pack.get('knowledge_units', []))}",
                        f"source_coverage_ratio={pack.get('source_coverage_ratio', 0.0):.3f}",
                        f"coverage_note={pack.get('coverage_note', '')}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _evidence_text(self, prioritized_evidence):
        blocks = []
        for item in prioritized_evidence:
            blocks.append(
                "\n".join(
                    [
                        f"[{item['evidence_id']}] utility={item.get('utility', 'background')} "
                        f"score={item.get('score', 0.0):.3f}",
                        f"role={item.get('role', 'node')}",
                        f"path={item.get('path', '')}",
                        f"knowledge_id={item.get('knowledge_id', '')}",
                        f"type={item.get('type', 'background')}",
                        f"time_scope={item.get('time_scope', '')}",
                        f"knowledge={item.get('content', '')}",
                        f"source_ids={','.join(item.get('source_ids', []))}",
                        f"priority_reason={item.get('reason', '')}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def build(self, query, paths, knowledge_packs, prioritized_evidence):
        return f"""使用者問題
{query}

分類決策路徑
{self._route_text(paths)}

正式 Taxonomy 節點資訊
{self._cards_text(paths)}

節點靜態知識概況
{self._knowledge_pack_text(knowledge_packs)}

最終回答 Context（已過濾；預設僅保留 direct / supporting）
{self._evidence_text(prioritized_evidence)}
"""
