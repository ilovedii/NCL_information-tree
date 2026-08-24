from dataclasses import dataclass


@dataclass
class RoutingMetrics:
    total: int = 0
    l1_hit: int = 0
    l2_hit: int = 0
    l3_total: int = 0
    l3_hit: int = 0
    full_path_hit: int = 0

    def as_dict(self):
        return {
            "total": self.total,
            "l1_recall": self.l1_hit / self.total if self.total else 0.0,
            "l2_recall": self.l2_hit / self.total if self.total else 0.0,
            "l3_total": self.l3_total,
            "l3_recall": self.l3_hit / self.l3_total if self.l3_total else 0.0,
            "full_path_recall": self.full_path_hit / self.total if self.total else 0.0,
        }


def parse_predicted_paths(result):
    paths = []
    for item in result.get("paths", []):
        trace = item.get("trace", [])
        values = {step.get("level"): step.get("node") for step in trace}
        paths.append((values.get("L1"), values.get("L2"), values.get("L3")))
    return paths


def update_routing_metrics(metrics, predicted_paths, gold_paths):
    metrics.total += 1
    gold_l1 = {x[0] for x in gold_paths if x[0]}
    gold_l2 = {(x[0], x[1]) for x in gold_paths if x[0] and x[1]}
    gold_l3 = {(x[0], x[1], x[2]) for x in gold_paths if x[0] and x[1] and x[2]}
    pred_l1 = {x[0] for x in predicted_paths if x[0]}
    pred_l2 = {(x[0], x[1]) for x in predicted_paths if x[0] and x[1]}
    pred_l3 = {(x[0], x[1], x[2]) for x in predicted_paths if x[0] and x[1] and x[2]}
    if pred_l1 & gold_l1:
        metrics.l1_hit += 1
    if pred_l2 & gold_l2:
        metrics.l2_hit += 1
    if gold_l3:
        metrics.l3_total += 1
        if pred_l3 & gold_l3:
            metrics.l3_hit += 1
    if set(predicted_paths) & set(gold_paths):
        metrics.full_path_hit += 1
    return metrics


def summary_source_coverage(result):
    summaries = result.get("node_summaries", [])
    if not summaries:
        return 0.0
    return sum(float(item.get("source_coverage_ratio", 0.0)) for item in summaries) / len(summaries)


def evidence_source_recall(result, gold_atomic_ids, utility_levels=None):
    gold = {str(x) for x in gold_atomic_ids}
    allowed = set(utility_levels or ["direct", "supporting", "background", "low_relevance"])
    sources = set()
    for item in result.get("evidence", []):
        if item.get("utility") not in allowed:
            continue
        sources.update(str(x) for x in item.get("source_ids", []))
    return 1.0 if sources & gold else 0.0


def direct_supporting_source_recall(result, gold_atomic_ids):
    return evidence_source_recall(
        result,
        gold_atomic_ids,
        utility_levels=["direct", "supporting"],
    )