"""Runtime metrics collected during simulation, not just post-hoc.

``MetricsCollector`` accumulates per-response and per-round data from
``CognitiveTrace`` objects, producing a ``SessionMetrics`` snapshot
on demand.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionMetrics:
    consistency_score: float = 1.0
    diversity_entropy: float = 0.0
    fatigue_decay_r2: float = 0.0
    contradiction_count: int = 0
    invariant_violations: int = 0
    duplicate_rate: float = 0.0
    mean_confidence: float = 0.0
    mean_response_word_count: float = 0.0
    response_length_stddev: float = 0.0
    intent_distribution: Dict[str, int] = field(default_factory=dict)
    post_processing_counts: Dict[str, int] = field(default_factory=dict)
    total_responses: int = 0
    rounds_completed: int = 0


class MetricsCollector:
    """Accumulates metrics across responses and rounds."""

    def __init__(self) -> None:
        self._confidence_sum: float = 0.0
        self._response_count: int = 0
        self._contradiction_count: int = 0
        self._invariant_violations: int = 0
        self._intent_counts: Dict[str, int] = Counter()
        self._pp_counts: Dict[str, int] = Counter()
        self._response_lengths: List[int] = []
        self._turn_numbers: List[int] = []
        self._answer_counts: Dict[str, int] = Counter()
        self._narrative_hashes: Dict[str, int] = Counter()
        self._rounds: int = 0

    def record_response(self, trace_dict: Dict[str, Any]) -> None:
        """Record a single response's cognitive trace."""
        self._response_count += 1

        conf_band = trace_dict.get("confidence_band", "medium")
        conf_val = {"high": 0.8, "medium": 0.5, "low": 0.2}.get(conf_band, 0.5)
        self._confidence_sum += conf_val

        intent = trace_dict.get("intent_class", "unknown")
        self._intent_counts[intent] += 1

        for pp in trace_dict.get("post_processing_applied", []):
            self._pp_counts[pp] += 1

        violations = trace_dict.get("invariant_violations", [])
        self._invariant_violations += len(violations)

        contradiction = trace_dict.get("contradiction_detected")
        if contradiction:
            self._contradiction_count += 1

        answer = trace_dict.get("sampled_option", "")
        if answer:
            self._answer_counts[answer] += 1

        final = trace_dict.get("final_response", "")
        self._response_lengths.append(len(final.split()) if final else 0)
        self._turn_numbers.append(trace_dict.get("turn_count", 0))

        narrative_key = final[:80] if final else ""
        self._narrative_hashes[narrative_key] += 1

    def record_round(self, round_responses: List[Dict[str, Any]]) -> None:
        """Record a completed survey round."""
        self._rounds += 1
        for resp in round_responses:
            trace = resp.get("cognitive_trace")
            if trace:
                self.record_response(trace)

    def snapshot(self) -> SessionMetrics:
        """Produce current metrics summary."""
        total = max(1, self._response_count)

        # Diversity entropy
        answer_total = sum(self._answer_counts.values())
        entropy = 0.0
        if answer_total > 0:
            probs = [c / answer_total for c in self._answer_counts.values() if c > 0]
            entropy = -sum(p * math.log2(p) for p in probs) if len(probs) > 1 else 0.0

        # Duplicate rate
        dup_count = sum(1 for c in self._narrative_hashes.values() if c > 1)
        dup_rate = dup_count / max(1, len(self._narrative_hashes))

        # Fatigue decay R^2 (simple linear fit)
        r2 = self._compute_length_decay_r2()

        consistency = 1.0 - (self._contradiction_count / total)

        mean_len, std_len = self._response_length_stats()

        return SessionMetrics(
            consistency_score=max(0.0, consistency),
            diversity_entropy=entropy,
            fatigue_decay_r2=r2,
            contradiction_count=self._contradiction_count,
            invariant_violations=self._invariant_violations,
            duplicate_rate=dup_rate,
            mean_confidence=self._confidence_sum / total,
            mean_response_word_count=mean_len,
            response_length_stddev=std_len,
            intent_distribution=dict(self._intent_counts),
            post_processing_counts=dict(self._pp_counts),
            total_responses=self._response_count,
            rounds_completed=self._rounds,
        )

    def _response_length_stats(self) -> tuple[float, float]:
        """Mean and sample stddev of response word counts (diversity-of-length signal)."""
        xs = self._response_lengths
        n = len(xs)
        if n == 0:
            return 0.0, 0.0
        mean = sum(xs) / n
        if n < 2:
            return mean, 0.0
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        return mean, math.sqrt(var)

    def _compute_length_decay_r2(self) -> float:
        """R-squared of response_length vs turn_count (negative = decay)."""
        n = len(self._response_lengths)
        if n < 3:
            return 0.0
        xs = self._turn_numbers[:n]
        ys = self._response_lengths[:n]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        if ss_tot == 0:
            return 0.0
        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        ss_xx = sum((x - x_mean) ** 2 for x in xs)
        if ss_xx == 0:
            return 0.0
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        return 1.0 - ss_res / ss_tot
