"""Global coordination layer for population-level constraint enforcement.

``SimulationCoordinator`` runs between survey rounds to monitor distribution
health, prevent answer collapse, and verify demographic correlations hold.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluation.invariants import check_population_invariants


@dataclass
class PopulationHealth:
    """Summary of population-level health indicators."""
    entropy_status: str = "pass"
    collapse_status: str = "pass"
    correlation_status: str = "pass"
    invariant_violations: List[str] = field(default_factory=list)


class SimulationCoordinator:
    """Population-level enforcement that runs between survey rounds."""

    def __init__(
        self,
        min_entropy: float = 0.5,
        max_dominant_fraction: float = 0.60,
        max_js_divergence: float = 0.25,
    ):
        self.min_entropy = min_entropy
        self.max_dominant_fraction = max_dominant_fraction
        self.max_js_divergence = max_js_divergence
        self._round_health_history: List[PopulationHealth] = []

    def enforce_distribution_health(
        self,
        round_responses: List[Dict[str, Any]],
        reference_dist: Optional[Dict[str, float]] = None,
    ) -> PopulationHealth:
        """Check distribution health and return status."""
        health = PopulationHealth()

        answer_counts: Dict[str, int] = Counter()
        for resp in round_responses:
            opt = resp.get("sampled_option", "")
            if opt:
                answer_counts[opt] += 1

        total = sum(answer_counts.values())
        if total == 0:
            return health

        # Entropy check
        probs = [c / total for c in answer_counts.values() if c > 0]
        entropy = -sum(p * math.log2(p) for p in probs) if len(probs) > 1 else 0.0
        max_e = math.log2(len(probs)) if len(probs) > 1 else 1.0
        norm_entropy = entropy / max_e if max_e > 0 else 0.0
        if norm_entropy < self.min_entropy:
            health.entropy_status = "fail"
            health.invariant_violations.append(
                f"low entropy: {norm_entropy:.3f} < {self.min_entropy}"
            )

        # Collapse check
        max_frac = max(answer_counts.values()) / total if total else 0
        if max_frac > self.max_dominant_fraction:
            health.collapse_status = "warn"
            health.invariant_violations.append(
                f"dominant option at {max_frac:.1%}"
            )

        # JS divergence from reference
        if reference_dist:
            js = self._js_divergence(answer_counts, reference_dist, total)
            if js > self.max_js_divergence:
                health.invariant_violations.append(
                    f"JS divergence from reference: {js:.3f} > {self.max_js_divergence}"
                )

        pop_violations = check_population_invariants(round_responses)
        health.invariant_violations.extend(pop_violations)

        self._round_health_history.append(health)
        return health

    def prevent_collapse(self, distributions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Return temperature adjustments per agent if entropy is too low.

        Agents whose distributions have very low entropy get a temperature
        boost to encourage diversity in the next round.
        """
        adjustments: Dict[str, float] = {}
        for agent_id, dist in distributions.items():
            total = sum(dist.values())
            if total == 0:
                continue
            probs = [v / total for v in dist.values() if v > 0]
            entropy = -sum(p * math.log2(p) for p in probs) if len(probs) > 1 else 0.0
            max_e = math.log2(len(probs)) if len(probs) > 1 else 1.0
            norm = entropy / max_e if max_e > 0 else 0.0
            if norm < 0.3:
                adjustments[agent_id] = 0.2
        return adjustments

    def enforce_correlation_invariants(
        self,
        agents: List[Dict[str, Any]],
    ) -> List[str]:
        """Verify income<->brand and age<->habit correlations hold.

        Returns list of violation messages.
        """
        violations: List[str] = []
        if len(agents) < 20:
            return violations

        income_map = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
        incomes = []
        brand_scores = []
        for a in agents:
            inc = income_map.get(str(a.get("income", "")).lower())
            bs = a.get("brand_loyalty", a.get("brand_score"))
            if inc is not None and bs is not None:
                incomes.append(inc)
                brand_scores.append(float(bs))

        if len(incomes) >= 10:
            corr = self._pearson(incomes, brand_scores)
            if corr is not None and corr < -0.3:
                violations.append(
                    f"income<->brand correlation unexpectedly negative: {corr:.3f}"
                )

        return violations

    def compute_population_health(
        self,
        round_responses: List[Dict[str, Any]],
        reference_dist: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """Return pass/warn/fail for each health indicator."""
        health = self.enforce_distribution_health(round_responses, reference_dist)
        return {
            "entropy": health.entropy_status,
            "collapse": health.collapse_status,
            "correlation": health.correlation_status,
            "violations_count": str(len(health.invariant_violations)),
        }

    @staticmethod
    def _js_divergence(
        counts: Dict[str, int],
        reference: Dict[str, float],
        total: int,
    ) -> float:
        """Jensen-Shannon divergence between observed and reference distributions."""
        all_keys = set(counts.keys()) | set(reference.keys())
        ref_total = sum(reference.values()) or 1.0
        js = 0.0
        for key in all_keys:
            p = (counts.get(key, 0) / total) if total else 0
            q = (reference.get(key, 0) / ref_total)
            m = (p + q) / 2
            if m > 0:
                if p > 0:
                    js += 0.5 * p * math.log2(p / m)
                if q > 0:
                    js += 0.5 * q * math.log2(q / m)
        return js

    @staticmethod
    def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
        n = len(xs)
        if n < 3:
            return None
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
        sy = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
        if sx == 0 or sy == 0:
            return None
        return cov / (sx * sy)
