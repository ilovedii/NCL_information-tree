import hashlib
import json
import re
from pathlib import Path


class KnowledgeSummarizer:
    def __init__(
        self,
        taxonomy,
        llm,
        model,
        cache_dir,
        batch_size=20,
        max_units_per_batch=24,
        merge_group_size=6,
        merge_max_units=60,
        think=True,
        use_cache=True,
    ):
        self.taxonomy = taxonomy
        self.llm = llm
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.batch_size = max(1, int(batch_size))
        self.max_units_per_batch = max(1, int(max_units_per_batch))
        self.merge_group_size = max(2, int(merge_group_size))
        self.merge_max_units = max(1, int(merge_max_units))
        self.think = think
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _node_card(self, bundle):
        if bundle["level"] == "L3":
            return self.taxonomy.node_card(
                "L3",
                l1=bundle["l1"],
                l2=bundle["l2"],
                node=bundle["l3"],
            )
        if bundle["level"] == "L2":
            return self.taxonomy.node_card("L2", l1=bundle["l1"], node=bundle["l2"])
        return self.taxonomy.node_card("L1", node=bundle["l1"])

    def _fingerprint(self, bundle):
        payload = {
            "path": bundle["path"],
            "model": self.model,
            "batch_size": self.batch_size,
            "max_units_per_batch": self.max_units_per_batch,
            "merge_group_size": self.merge_group_size,
            "merge_max_units": self.merge_max_units,
            "documents": [
                [
                    item["atomic_id"],
                    item["question_date"],
                    item["question"],
                    item["answer"],
                ]
                for item in bundle["documents"]
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _safe_name(self, path):
        base = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", path).strip("_")
        suffix = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
        return f"{base[:100]}__{suffix}.json"

    def _cache_path(self, bundle):
        return self.cache_dir / self._safe_name(bundle["path"])

    def _load_cache(self, bundle, fingerprint):
        if not self.use_cache:
            return None
        path = self._cache_path(bundle)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("fingerprint") != fingerprint:
            return None
        data["cache_hit"] = True
        return data

    def _save_cache(self, bundle, data):
        if not self.use_cache:
            return
        path = self._cache_path(bundle)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _unit_schema(self, max_items, source_ids=None):
        source_item = {"type": "string"}
        if source_ids and len(source_ids) <= 80:
            source_item["enum"] = list(source_ids)
        return {
            "type": "object",
            "properties": {
                "knowledge_units": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max_items,
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "rule",
                                    "procedure",
                                    "definition",
                                    "exception",
                                    "boundary",
                                    "historical_change",
                                    "conflict",
                                    "example",
                                    "background",
                                ],
                            },
                            "content": {"type": "string"},
                            "time_scope": {"type": "string"},
                            "source_ids": {
                                "type": "array",
                                "minItems": 1,
                                "items": source_item,
                            },
                        },
                        "required": ["type", "content", "time_scope", "source_ids"],
                    },
                },
                "coverage_note": {"type": "string"},
            },
            "required": ["knowledge_units", "coverage_note"],
        }

    def _format_documents(self, documents):
        blocks = []
        for item in documents:
            blocks.append(
                f"[{item['atomic_id']}] date={item['question_date']}\nQ={item['question']}\nA={item['answer']}"
            )
        return "\n\n".join(blocks)

    def _summarize_batch(self, bundle, documents, batch_number, total_batches):
        source_ids = [item["atomic_id"] for item in documents]
        schema = self._unit_schema(self.max_units_per_batch, source_ids=source_ids)
        prompt = f"""你是國家圖書館知識整理模型。你的工作不是回答某個使用者問題，而是忠實整理指定 taxonomy 節點中的全部知識。

節點：
{self._node_card(bundle)}

這是依時間由舊到新排列的第 {batch_number}/{total_batches} 批資料：
{self._format_documents(documents)}

整理規則：
1. 這批資料全部都必須納入判讀，不得先依相似度或主觀相關性篩除。
2. 將重複說法合併成可重用的 knowledge unit，但不可刪除具有不同條件、例外、版本、時間差異或邊界的知識。
3. 若後期資料修正、取代或補充早期資料，建立 historical_change，清楚寫出時間演變。
4. 若資料彼此不一致而無法從時間判定取代關係，建立 conflict，不可自行消解衝突。
5. 每個 unit 必須保留能支持該內容的 atomic_id。
6. time_scope 可填明確日期區間、某日期後、歷史資料、目前資料或空字串。
7. 不得加入這批 QA 未支持的外部知識。
8. coverage_note 簡述這批資料涵蓋哪些面向以及是否存在資訊缺口。
"""
        return self.llm.chat_json(
            self.model,
            [{"role": "user", "content": prompt}],
            schema,
            temperature=0.0,
            think=self.think,
        )

    def _normalize_units(self, units, allowed_source_ids):
        allowed = set(allowed_source_ids)
        normalized = []
        for unit in units:
            content = str(unit.get("content", "")).strip()
            if not content:
                continue
            source_ids = []
            for source_id in unit.get("source_ids", []):
                source_id = str(source_id)
                if source_id in allowed and source_id not in source_ids:
                    source_ids.append(source_id)
            if not source_ids:
                continue
            normalized.append(
                {
                    "type": str(unit.get("type", "background")),
                    "content": content,
                    "time_scope": str(unit.get("time_scope", "")).strip(),
                    "source_ids": source_ids,
                }
            )
        return normalized

    def _merge_group(self, bundle, summaries, stage):
        all_units = []
        allowed_sources = []
        for summary in summaries:
            all_units.extend(summary.get("knowledge_units", []))
            for source_id in summary.get("all_source_ids", []):
                if source_id not in allowed_sources:
                    allowed_sources.append(source_id)
        schema = self._unit_schema(self.merge_max_units, source_ids=None)
        payload = json.dumps(
            [
                {
                    "type": unit["type"],
                    "content": unit["content"],
                    "time_scope": unit.get("time_scope", ""),
                    "source_ids": unit.get("source_ids", []),
                }
                for unit in all_units
            ],
            ensure_ascii=False,
        )
        prompt = f"""你是國家圖書館節點知識合併模型。請把多批已整理的 knowledge units 合併成較完整且去重的節點知識，不是針對某個查詢挑選答案。

節點：
{self._node_card(bundle)}

合併階段：{stage}

輸入 knowledge units：
{payload}

合併規則：
1. 所有輸入 units 都必須被考慮，不得因與某個未知未來問題看似不相關而刪除。
2. 可合併語意相同的重複規則，合併後要聯集 source_ids。
3. 不可合併具有不同條件、適用範圍、例外、時間、版本或衝突的規則。
4. historical_change、conflict、exception、boundary 優先保留。
5. 不得新增輸入 units 未支持的知識。
6. source_ids 只能使用輸入中已存在的 atomic_id。
7. coverage_note 說明合併後仍保留了哪些主要知識面向與已知衝突。
"""
        result = self.llm.chat_json(
            self.model,
            [{"role": "user", "content": prompt}],
            schema,
            temperature=0.0,
            think=self.think,
        )
        units = self._normalize_units(result.get("knowledge_units", []), allowed_sources)
        return {
            "knowledge_units": units,
            "coverage_note": str(result.get("coverage_note", "")).strip(),
            "all_source_ids": allowed_sources,
        }

    def summarize(self, bundle, force_rebuild=False):
        documents = list(bundle.get("documents", []))
        fingerprint = self._fingerprint(bundle)
        if not force_rebuild:
            cached = self._load_cache(bundle, fingerprint)
            if cached is not None:
                cached["role"] = bundle.get("role", cached.get("role", "node"))
                return cached
        all_source_ids = [item["atomic_id"] for item in documents]
        if not documents:
            result = {
                "fingerprint": fingerprint,
                "cache_hit": False,
                "role": bundle.get("role", "node"),
                "level": bundle["level"],
                "path": bundle["path"],
                "document_count": 0,
                "date_start": "",
                "date_end": "",
                "all_source_ids": [],
                "knowledge_units": [],
                "coverage_note": "此節點沒有可整理的 QA。",
                "source_coverage_ratio": 0.0,
            }
            self._save_cache(bundle, result)
            return result
        batches = [
            documents[start:start + self.batch_size]
            for start in range(0, len(documents), self.batch_size)
        ]
        summaries = []
        for i, batch in enumerate(batches, start=1):
            batch_result = self._summarize_batch(bundle, batch, i, len(batches))
            batch_ids = [item["atomic_id"] for item in batch]
            summaries.append(
                {
                    "knowledge_units": self._normalize_units(
                        batch_result.get("knowledge_units", []),
                        batch_ids,
                    ),
                    "coverage_note": str(batch_result.get("coverage_note", "")).strip(),
                    "all_source_ids": batch_ids,
                }
            )
        stage = 1
        while len(summaries) > 1:
            next_level = []
            for start in range(0, len(summaries), self.merge_group_size):
                group = summaries[start:start + self.merge_group_size]
                if len(group) == 1:
                    next_level.append(group[0])
                else:
                    next_level.append(self._merge_group(bundle, group, stage))
            summaries = next_level
            stage += 1
        final = summaries[0]
        units = final.get("knowledge_units", [])
        for i, unit in enumerate(units, start=1):
            unit["knowledge_id"] = f"K{i:03d}"
        referenced = set()
        for unit in units:
            referenced.update(unit.get("source_ids", []))
        result = {
            "fingerprint": fingerprint,
            "cache_hit": False,
            "role": bundle.get("role", "node"),
            "level": bundle["level"],
            "path": bundle["path"],
            "document_count": len(documents),
            "date_start": documents[0]["question_date"],
            "date_end": documents[-1]["question_date"],
            "all_source_ids": all_source_ids,
            "knowledge_units": units,
            "coverage_note": final.get("coverage_note", ""),
            "source_coverage_ratio": len(referenced) / len(set(all_source_ids)) if all_source_ids else 0.0,
        }
        self._save_cache(bundle, result)
        return result