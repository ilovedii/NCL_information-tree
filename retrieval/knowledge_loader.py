class KnowledgeLoader:
    def __init__(self, taxonomy, date_column="question_date"):
        self.taxonomy = taxonomy
        self.date_column = date_column

    def load_path(self, path, role="primary"):
        documents = self.taxonomy.documents_for_path(path, date_column=self.date_column)
        return {
            "role": role,
            "level": "L3" if path.l3 else "L2" if path.l2 else "L1",
            "l1": path.l1,
            "l2": path.l2,
            "l3": path.l3,
            "path": path.display(),
            "documents": documents,
        }

    def load_parent(self, path):
        if not path.l2:
            return None
        documents = self.taxonomy.documents_for_node(
            path.l1,
            l2=path.l2,
            date_column=self.date_column,
        )
        return {
            "role": "parent",
            "level": "L2",
            "l1": path.l1,
            "l2": path.l2,
            "l3": None,
            "path": f"{path.l1} > {path.l2}",
            "documents": documents,
        }

    def load_node(self, l1, l2=None, l3=None, role="node"):
        documents = self.taxonomy.documents_for_node(
            l1,
            l2=l2,
            l3=l3,
            date_column=self.date_column,
        )
        parts = [x for x in (l1, l2, l3) if x]
        return {
            "role": role,
            "level": "L3" if l3 else "L2" if l2 else "L1",
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "path": " > ".join(parts),
            "documents": documents,
        }

    def load_summary_node(self, node):
        return self.load_node(
            node["l1"],
            l2=node.get("l2"),
            l3=node.get("l3"),
            role="precompute",
        )