from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ncl_taxonomy_l1 import CATEGORY_NAMES, CATEGORY_PROFILES, L1_TAXONOMY_VALIDATION
from ncl_taxonomy_l2 import L2_CATEGORY_PROFILES, L2_TAXONOMY_VALIDATION
from ncl_taxonomy_l3 import (
    SELECTIVE_L3_CATEGORY_PROFILES,
    L3_TAXONOMY_VALIDATION,
    has_l3,
)


class TaxonomyIndex:
    def __init__(self, csv_path, representative_count=4, strict=True):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path, encoding="utf-8-sig").fillna("")
        self.representative_count = representative_count
        self.strict = strict
        self.node_docs = defaultdict(set)
        self.row_paths = defaultdict(list)
        self.validation_issues = []
        self._validate_columns()
        self._build_memberships()
        self._validate_memberships()

    def _validate_columns(self):
        required = {"atomic_id", "question_date", "question", "answer", "l1_category"}
        for slot in range(1, 4):
            required.add(f"l2_category_{slot}")
            required.add(f"l3_for_l2_{slot}")
        missing = sorted(required - set(self.df.columns))
        if missing:
            raise ValueError(f"CSV 缺少必要欄位：{', '.join(missing)}")

    def _split_l3(self, value):
        return [x.strip() for x in str(value).split("|") if x.strip()]

    def _issue(self, message):
        self.validation_issues.append(message)

    def _build_memberships(self):
        for idx, row in self.df.iterrows():
            l1 = str(row["l1_category"]).strip()
            if not l1:
                self._issue(f"row {idx} 缺少 L1")
                continue
            self.node_docs[("L1", l1)].add(idx)
            has_l2_membership = False
            for slot in range(1, 4):
                l2 = str(row.get(f"l2_category_{slot}", "")).strip()
                if not l2:
                    continue
                has_l2_membership = True
                self.node_docs[("L2", l1, l2)].add(idx)
                l3s = self._split_l3(row.get(f"l3_for_l2_{slot}", ""))
                if l3s:
                    for l3 in l3s:
                        self.node_docs[("L3", l1, l2, l3)].add(idx)
                        self.row_paths[idx].append((l1, l2, l3))
                else:
                    self.row_paths[idx].append((l1, l2, None))
            if not has_l2_membership:
                self._issue(f"row {idx} / {row['atomic_id']} 缺少 L2")
                self.row_paths[idx].append((l1, None, None))
            self.row_paths[idx] = list(dict.fromkeys(self.row_paths[idx]))

    def _validate_memberships(self):
        for idx, paths in self.row_paths.items():
            for l1, l2, l3 in paths:
                if l1 not in CATEGORY_PROFILES:
                    self._issue(f"row {idx} 使用未知 L1：{l1}")
                    continue
                if l2 is None:
                    continue
                if l2 not in L2_CATEGORY_PROFILES.get(l1, {}):
                    self._issue(f"row {idx} 使用未知 L2：{l1} > {l2}")
                    continue
                if l3 is not None:
                    profiles = SELECTIVE_L3_CATEGORY_PROFILES.get((l1, l2), {})
                    if l3 not in profiles:
                        self._issue(f"row {idx} 使用未知 L3：{l1} > {l2} > {l3}")
        if self.strict and self.validation_issues:
            sample = "\n".join(self.validation_issues[:30])
            extra = "" if len(self.validation_issues) <= 30 else f"\n另有 {len(self.validation_issues) - 30} 項"
            raise ValueError(f"Taxonomy membership validation failed:\n{sample}{extra}")

    def l1_nodes(self):
        return list(CATEGORY_NAMES)

    def l2_nodes(self, l1):
        return list(L2_CATEGORY_PROFILES.get(l1, {}).keys())

    def l3_nodes(self, l1, l2):
        return list(SELECTIVE_L3_CATEGORY_PROFILES.get((l1, l2), {}).keys())

    def is_terminal_l2(self, l1, l2):
        return not has_l3(l1, l2)

    def node_label(self, node):
        return "" if node is None else str(node)

    def node_profile(self, level, l1=None, l2=None, node=None):
        if level == "L1":
            return CATEGORY_PROFILES[node]
        if level == "L2":
            return L2_CATEGORY_PROFILES[l1][node]
        if level == "L3":
            return SELECTIVE_L3_CATEGORY_PROFILES[(l1, l2)][node]
        raise ValueError(f"未知 level：{level}")

    def docs_for_path(self, path):
        if path.l3:
            return sorted(self.node_docs.get(("L3", path.l1, path.l2, path.l3), set()))
        if path.l2:
            return sorted(self.node_docs.get(("L2", path.l1, path.l2), set()))
        return sorted(self.node_docs.get(("L1", path.l1), set()))

    def docs_for_l1(self, l1):
        return sorted(self.node_docs.get(("L1", l1), set()))

    def docs_for_l2(self, l1, l2):
        return sorted(self.node_docs.get(("L2", l1, l2), set()))

    def docs_for_l3(self, l1, l2, l3):
        return sorted(self.node_docs.get(("L3", l1, l2, l3), set()))

    def docs_for_siblings(self, l1, l2, selected_l3s):
        result = set()
        for l3 in self.l3_nodes(l1, l2):
            if l3 in selected_l3s:
                continue
            result.update(self.node_docs.get(("L3", l1, l2, l3), set()))
        return sorted(result)

    def _date_value(self, idx, date_column="question_date"):
        value = str(self.df.at[idx, date_column]).strip() if date_column in self.df.columns else ""
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return pd.Timestamp.max
        return parsed

    def sort_indices_by_date(self, indices, date_column="question_date"):
        unique = list(dict.fromkeys(int(x) for x in indices))
        return sorted(unique, key=lambda idx: (self._date_value(idx, date_column), idx))

    def document_record(self, idx):
        idx = int(idx)
        return {
            "idx": idx,
            "atomic_id": str(self.df.at[idx, "atomic_id"]),
            "faq_id": str(self.df.at[idx, "faq_id"]) if "faq_id" in self.df.columns else "",
            "question_date": str(self.df.at[idx, "question_date"]),
            "question": str(self.df.at[idx, "question"]),
            "answer": str(self.df.at[idx, "answer"]),
            "taxonomy_paths": str(self.df.at[idx, "taxonomy_paths"]) if "taxonomy_paths" in self.df.columns else "",
        }

    def indices_for_atomic_ids(self, atomic_ids):
        """Return dataframe indices that match the supplied atomic IDs."""
        atomic_ids = {
            str(value).strip()
            for value in atomic_ids
            if str(value).strip()
        }
        if not atomic_ids:
            return []

        atomic_series = self.df["atomic_id"].astype(str).str.strip()
        return self.df.index[atomic_series.isin(atomic_ids)].tolist()

    def sibling_indices_for_atomic_ids(
        self,
        atomic_ids,
        max_faqs=3,
        max_siblings_per_faq=15,
    ):
        """
        Expand retrieved atomic units through their original FAQ provenance.

        The taxonomy path is not changed. This method only follows the
        source relation: atomic_id -> faq_id -> other atomic units from the
        same original FAQ.
        """
        if "faq_id" not in self.df.columns:
            return []

        anchor_indices = self.indices_for_atomic_ids(atomic_ids)
        if not anchor_indices:
            return []

        faq_ids = []
        seen_faqs = set()
        for idx in anchor_indices:
            faq_id = str(self.df.at[idx, "faq_id"]).strip()
            if not faq_id or faq_id in seen_faqs:
                continue
            seen_faqs.add(faq_id)
            faq_ids.append(faq_id)
            if len(faq_ids) >= max_faqs:
                break

        if not faq_ids:
            return []

        faq_series = self.df["faq_id"].astype(str).str.strip()
        result = []
        seen_indices = set()

        for faq_id in faq_ids:
            sibling_indices = self.df.index[faq_series == faq_id].tolist()
            if max_siblings_per_faq and max_siblings_per_faq > 0:
                sibling_indices = sibling_indices[:max_siblings_per_faq]

            for idx in sibling_indices:
                idx = int(idx)
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)
                result.append(idx)

        return result

    def documents_for_indices(self, indices, date_column="question_date"):
        ordered = self.sort_indices_by_date(indices, date_column=date_column)
        return [self.document_record(idx) for idx in ordered]

    def documents_for_path(self, path, date_column="question_date"):
        return self.documents_for_indices(self.docs_for_path(path), date_column=date_column)

    def documents_for_node(self, l1, l2=None, l3=None, date_column="question_date"):
        if l3:
            indices = self.docs_for_l3(l1, l2, l3)
        elif l2:
            indices = self.docs_for_l2(l1, l2)
        else:
            indices = self.docs_for_l1(l1)
        return self.documents_for_indices(indices, date_column=date_column)

    def summary_nodes(self, include_l2_parents=True):
        nodes = []
        for l1 in self.l1_nodes():
            for l2 in self.l2_nodes(l1):
                if include_l2_parents or self.is_terminal_l2(l1, l2):
                    nodes.append({
                        "level": "L2",
                        "l1": l1,
                        "l2": l2,
                        "l3": None,
                        "path": f"{l1} > {l2}",
                    })
                for l3 in self.l3_nodes(l1, l2):
                    nodes.append({
                        "level": "L3",
                        "l1": l1,
                        "l2": l2,
                        "l3": l3,
                        "path": f"{l1} > {l2} > {l3}",
                    })
        return nodes

    def gold_paths_for_index(self, idx):
        return list(self.row_paths.get(int(idx), []))

    def gold_paths_for_atomic_id(self, atomic_id):
        matches = self.df.index[self.df["atomic_id"].astype(str) == str(atomic_id)].tolist()
        paths = []
        for idx in matches:
            paths.extend(self.gold_paths_for_index(idx))
        return list(dict.fromkeys(paths))

    def _representative_indices(self, indices, count=None):
        count = count or self.representative_count
        unique = []
        seen = set()
        for idx in sorted(indices):
            question = str(self.df.at[idx, "question"]).strip()
            if not question or question in seen:
                continue
            seen.add(question)
            unique.append(idx)
        if len(unique) <= count:
            return unique
        positions = np.linspace(0, len(unique) - 1, num=count, dtype=int)
        return [unique[int(i)] for i in positions]

    def node_card(self, level, l1=None, l2=None, node=None, include_answers=False):
        profile = self.node_profile(level, l1=l1, l2=l2, node=node)
        if level == "L1":
            key = ("L1", node)
            title = node
            path_text = node
            children = self.l2_nodes(node)
        elif level == "L2":
            key = ("L2", l1, node)
            title = node
            path_text = f"{l1} > {node}"
            children = self.l3_nodes(l1, node)
        elif level == "L3":
            key = ("L3", l1, l2, node)
            title = node
            path_text = f"{l1} > {l2} > {node}"
            children = []
        else:
            raise ValueError(f"未知 level：{level}")
        indices = sorted(self.node_docs.get(key, set()))
        reps = self._representative_indices(indices)
        lines = [
            f"節點名稱：{title}",
            f"完整路徑：{path_text}",
            f"正式定義：{profile['description']}",
            f"正式例子：{profile['example']}",
        ]
        boundary = str(profile.get("boundary", "")).strip()
        if boundary:
            lines.append(f"分類邊界：{boundary}")
        if level == "L2" and not children:
            lines.append("節點型態：Terminal L2，沒有建立 Level 3，選中後直接進行此 L2 節點內 retrieval")
        if children:
            lines.append("下層節點：" + "、".join(children))
        lines.append(f"實際資料筆數：{len(indices)}")
        if reps:
            lines.append("代表資料：")
            for idx in reps:
                q = str(self.df.at[idx, "question"]).strip()
                if include_answers:
                    a = str(self.df.at[idx, "answer"]).strip()
                    lines.append(f"- Q：{q}\n  A：{a}")
                else:
                    lines.append(f"- {q}")
        return "\n".join(lines)

    def path_node_cards(self, path):
        cards = [self.node_card("L1", node=path.l1)]
        if path.l2:
            cards.append(self.node_card("L2", l1=path.l1, node=path.l2))
        if path.l3:
            cards.append(self.node_card("L3", l1=path.l1, l2=path.l2, node=path.l3))
        return cards

    def stats(self):
        l2_count = sum(len(v) for v in L2_CATEGORY_PROFILES.values())
        l3_count = sum(len(v) for v in SELECTIVE_L3_CATEGORY_PROFILES.values())
        selective_parents = len(SELECTIVE_L3_CATEGORY_PROFILES)
        terminal_l2 = l2_count - selective_parents
        selective_unassigned = 0
        for paths in self.row_paths.values():
            for l1, l2, l3 in paths:
                if l2 and has_l3(l1, l2) and l3 is None:
                    selective_unassigned += 1
        return {
            "documents": len(self.df),
            "l1_count": len(CATEGORY_NAMES),
            "l2_count": l2_count,
            "selective_l3_parent_count": selective_parents,
            "l3_count": l3_count,
            "terminal_l2_count": terminal_l2,
            "path_memberships": sum(len(x) for x in self.row_paths.values()),
            "selective_l3_memberships_without_l3": selective_unassigned,
            "membership_validation_issues": len(self.validation_issues),
            "l1_definition_validation": L1_TAXONOMY_VALIDATION,
            "l2_definition_validation": L2_TAXONOMY_VALIDATION,
            "l3_definition_validation": L3_TAXONOMY_VALIDATION,
        }