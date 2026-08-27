import hashlib
import json
import re
from pathlib import Path


class StaticKnowledgeStore:
    """Deterministic, coverage-preserving node knowledge store.

    This class does NOT call an LLM and does NOT summarize at query time.
    Each atomic record is converted into a compact knowledge unit:

        主題：<atomic question>
        知識：<atomic answer>

    Exact duplicate units may be merged by unioning source_ids. No semantic
    Top-K pruning is performed here, so every atomic source in the selected
    node remains represented.
    """

    def __init__(
        self,
        taxonomy,
        cache_dir,
        include_topic=True,
        exact_dedup=True,
    ):
        self.taxonomy = taxonomy
        self.cache_dir = Path(cache_dir)
        self.include_topic = bool(include_topic)
        self.exact_dedup = bool(exact_dedup)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _text(value):
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() == "nan":
            return ""
        return text

    @staticmethod
    def _normalize_for_dedup(text):
        return re.sub(r"\s+", " ", str(text)).strip()

    @staticmethod
    def _safe_name(path):
        base = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", path).strip("_")
        suffix = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
        return f"{base[:100]}__{suffix}.json"

    def _cache_path(self, bundle):
        return self.cache_dir / self._safe_name(bundle["path"])

    def _fingerprint(self, bundle):
        payload = {
            "path": bundle["path"],
            "include_topic": self.include_topic,
            "exact_dedup": self.exact_dedup,
            "documents": [
                [
                    self._text(item.get("atomic_id")),
                    self._text(item.get("question_date")),
                    self._text(item.get("question")),
                    self._text(item.get("answer")),
                ]
                for item in bundle.get("documents", [])
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def unit_from_document(self, document, knowledge_id, unit_type="atomic_knowledge"):
        """Convert one atomic document into a non-QA knowledge representation."""
        atomic_id = self._text(document.get("atomic_id"))
        topic = self._text(document.get("question"))
        knowledge = self._text(document.get("answer"))

        parts = []
        if self.include_topic and topic:
            parts.append(f"主題：{topic}")
        if knowledge:
            parts.append(f"知識：{knowledge}")
        elif topic:
            # Preserve visibility even if a row unexpectedly lacks an answer.
            parts.append(f"知識：{topic}")

        return {
            "knowledge_id": knowledge_id,
            "type": unit_type,
            "content": "\n".join(parts),
            "time_scope": self._text(document.get("question_date")),
            "source_ids": [atomic_id] if atomic_id else [],
        }

    def _build(self, bundle):
        documents = list(bundle.get("documents", []))
        fingerprint = self._fingerprint(bundle)

        all_source_ids = []
        seen_source_ids = set()
        units = []
        unit_by_key = {}

        for document in documents:
            atomic_id = self._text(document.get("atomic_id"))
            if atomic_id and atomic_id not in seen_source_ids:
                seen_source_ids.add(atomic_id)
                all_source_ids.append(atomic_id)

            candidate = self.unit_from_document(
                document,
                knowledge_id=f"K{len(units) + 1:04d}",
            )
            key = self._normalize_for_dedup(candidate["content"])

            if self.exact_dedup and key in unit_by_key:
                existing = unit_by_key[key]
                for source_id in candidate["source_ids"]:
                    if source_id not in existing["source_ids"]:
                        existing["source_ids"].append(source_id)
                continue

            units.append(candidate)
            unit_by_key[key] = candidate

        represented_source_ids = {
            source_id
            for unit in units
            for source_id in unit.get("source_ids", [])
        }
        coverage_ratio = (
            len(represented_source_ids) / len(all_source_ids)
            if all_source_ids
            else 0.0
        )

        dates = [
            self._text(item.get("question_date"))
            for item in documents
            if self._text(item.get("question_date"))
        ]

        return {
            "fingerprint": fingerprint,
            "cache_hit": False,
            "static_source": "atomic_knowledge",
            "role": bundle.get("role", "node"),
            "level": bundle.get("level", ""),
            "path": bundle.get("path", ""),
            "document_count": len(documents),
            "date_start": dates[0] if dates else "",
            "date_end": dates[-1] if dates else "",
            "all_source_ids": all_source_ids,
            "knowledge_units": units,
            "coverage_note": (
                "未使用 LLM 摘要。選定節點中的每個 atomic record 都被轉成"
                "「主題＋知識」靜態 knowledge unit；只合併內容完全相同的重複單元，"
                "並聯集 source_ids，不做語意 Top-K 刪除。"
            ),
            "source_coverage_ratio": coverage_ratio,
        }

    def load(self, bundle, force_rebuild=False):
        """Load a deterministic node pack, rebuilding locally when stale/missing."""
        cache_path = self._cache_path(bundle)
        fingerprint = self._fingerprint(bundle)

        if not force_rebuild and cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = None

            if isinstance(data, dict) and data.get("fingerprint") == fingerprint:
                data["cache_hit"] = True
                data["role"] = bundle.get("role", data.get("role", "node"))
                return data

        data = self._build(bundle)
        cache_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return data

    def build_all(self, knowledge_loader, include_l2_parents=True, force_rebuild=False):
        """Prebuild every node pack locally. This performs zero LLM/API calls."""
        results = []
        nodes = self.taxonomy.summary_nodes(include_l2_parents=include_l2_parents)
        for i, node in enumerate(nodes, start=1):
            bundle = knowledge_loader.load_summary_node(node)
            pack = self.load(bundle, force_rebuild=force_rebuild)
            results.append(
                {
                    "index": i,
                    "total": len(nodes),
                    "path": pack["path"],
                    "document_count": pack["document_count"],
                    "knowledge_units": len(pack["knowledge_units"]),
                    "cache_hit": pack.get("cache_hit", False),
                    "source_coverage_ratio": pack.get("source_coverage_ratio", 0.0),
                    "static_source": pack.get("static_source", "atomic_knowledge"),
                }
            )
        return results
