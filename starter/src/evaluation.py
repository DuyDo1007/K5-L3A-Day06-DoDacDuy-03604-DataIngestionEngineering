"""Evidence retrieval scoring and transparent split-level aggregation."""
import re
from typing import Any


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check page/source plus quote; exact match is only an evidence-retrieval proxy."""
    expected_source = question["gold_source_id"]
    expected_pages = {str(page) for page in question.get("gold_pages", [question["gold_page"]])}
    evidence_quotes = question.get("evidence_quotes", [question["evidence_quote"]])
    normalized_quotes = [_normalize(quote) for quote in evidence_quotes]

    for hit in hits:
        metadata = hit.get("metadata", {})
        source_match = metadata.get("document_id") == expected_source
        page_match = str(metadata.get("page")) in expected_pages
        hit_text = _normalize(hit.get("text", ""))
        evidence_match = any(quote and quote in hit_text for quote in normalized_quotes)
        hit["evidence_match"] = bool(source_match and page_match and evidence_match)

    hit_at_1 = bool(hits and hits[0]["evidence_match"])
    hit_at_3 = any(hit["evidence_match"] for hit in hits[:3])
    return {
        "question_id": question["id"],
        "split": question["split"],
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": hit_at_3,
        "hits": hits,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate integer numerators and denominators, not only percentages."""
    summary: dict[str, Any] = {}
    for trace in traces:
        key = f"{trace['strategy']}_{trace['split']}"
        entry = summary.setdefault(key, {"n": 0, "hit_at_1": 0, "evidence_hit_at_3": 0})
        entry["n"] += 1
        entry["hit_at_1"] += int(trace["hit_at_1"])
        entry["evidence_hit_at_3"] += int(trace["evidence_hit_at_3"])
    for entry in summary.values():
        denominator = entry["n"]
        entry["hit_at_1_rate"] = entry["hit_at_1"] / denominator if denominator else 0.0
        entry["evidence_hit_at_3_rate"] = (
            entry["evidence_hit_at_3"] / denominator if denominator else 0.0
        )
    return summary
