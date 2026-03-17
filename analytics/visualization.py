"""
Charts: bar charts by segment, distribution plots (matplotlib/plotly).
"""

from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bar_chart_by_segment(
    aggregated: Dict[str, Dict[str, float]],
    title: str = "Survey responses by segment",
    xlabel: str = "Segment",
    output_path: Optional[str] = None,
) -> None:
    """
    aggregated: segment -> {answer_value: proportion}.
    Creates a grouped bar chart; saves to output_path if given.
    """
    if not aggregated:
        return
    segments = list(aggregated.keys())
    all_answers = set()
    for dist in aggregated.values():
        all_answers.update(dist.keys())
    all_answers = sorted(all_answers)
    x = np.arange(len(segments))
    width = 0.8 / max(len(all_answers), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, ans in enumerate(all_answers):
        vals = [aggregated.get(seg, {}).get(ans, 0) for seg in segments]
        offset = (i - len(all_answers) / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=ans)
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=100)
    plt.close()


def distribution_plot(
    values: List[Any],
    title: str = "Distribution",
    output_path: Optional[str] = None,
) -> None:
    """Simple histogram or value_counts bar plot."""
    from collections import Counter
    counts = Counter(values)
    labels = list(counts.keys())
    heights = list(counts.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, heights)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=100)
    plt.close()
