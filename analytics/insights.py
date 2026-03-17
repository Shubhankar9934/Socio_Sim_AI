"""
Automated textual insights from aggregated survey results.
"""

from typing import Dict, List


def generate_insights(
    aggregated: Dict[str, Dict[str, float]],
    segment_name: str = "segment",
    top_n: int = 3,
) -> List[str]:
    """
    aggregated: segment -> {answer_value: proportion}.
    Returns list of short insight strings.
    """
    insights = []
    if not aggregated:
        return insights
    segments = list(aggregated.keys())
    for seg in segments:
        dist = aggregated[seg]
        if not dist:
            continue
        top = sorted(dist.items(), key=lambda x: -x[1])[:top_n]
        parts = [f"{v}: {p:.0%}" for v, p in top]
        insights.append(f"{segment_name} {seg}: " + ", ".join(parts))
    return insights


def compare_segments(
    aggregated: Dict[str, Dict[str, float]],
    metric_answer: str,
    segment_name: str = "Area",
) -> List[str]:
    """Compare one metric (answer value) across segments."""
    insights = []
    for seg, dist in aggregated.items():
        p = dist.get(metric_answer, 0.0)
        insights.append(f"{segment_name} {seg}: {metric_answer} = {p:.1%}")
    return insights


def high_frequency_insight(
    aggregated: Dict[str, Dict[str, float]],
    high_keys: List[str] | None = None,
) -> str:
    """Single summary sentence for high-frequency answers by segment."""
    if not aggregated:
        return "No data to summarize."
    if high_keys is None:
        high_keys = ["3-4 per week", "daily", "multiple per day",
                     "often", "very often"]
    best = None
    best_val = -1.0
    worst = None
    worst_val = 2.0
    for seg, dist in aggregated.items():
        high_freq = sum(dist.get(k, 0) for k in high_keys)
        if high_freq > best_val:
            best_val = high_freq
            best = seg
        if high_freq < worst_val:
            worst_val = high_freq
            worst = seg
    if best and worst:
        return f"High frequency in {best} ({best_val:.0%}); lower in {worst} ({worst_val:.0%})."
    return "Insufficient segments for comparison."


delivery_frequency_insight = high_frequency_insight
