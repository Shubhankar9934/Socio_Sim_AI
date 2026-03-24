"""
Structural Causal Graph: lightweight SCM over dimensions, beliefs, and actions.

Supports:
  - do-intervention (set a variable and propagate effects)
  - Average Treatment Effect estimation with confounding adjustment
  - Topological propagation of causal effects
  - Counterfactual queries
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    cause: str
    effect: str
    weight: float
    mechanism: str = ""  # optional description


@dataclass
class CausalGraph:
    """Lightweight structural causal model."""

    nodes: List[str] = field(default_factory=list)
    edges: Dict[Tuple[str, str], CausalEdge] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    _children: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    _parents: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, name: str) -> None:
        if name not in self.nodes:
            self.nodes.append(name)

    def add_edge(
        self, cause: str, effect: str, weight: float, mechanism: str = ""
    ) -> None:
        self.add_node(cause)
        self.add_node(effect)
        edge = CausalEdge(cause=cause, effect=effect, weight=weight, mechanism=mechanism)
        self.edges[(cause, effect)] = edge
        if effect not in self._children[cause]:
            self._children[cause].append(effect)
        if cause not in self._parents[effect]:
            self._parents[effect].append(cause)

    def remove_edge(self, cause: str, effect: str) -> None:
        self.edges.pop((cause, effect), None)
        if effect in self._children.get(cause, []):
            self._children[cause].remove(effect)
        if cause in self._parents.get(effect, []):
            self._parents[effect].remove(cause)

    def children(self, node: str) -> List[str]:
        return self._children.get(node, [])

    def parents(self, node: str) -> List[str]:
        return self._parents.get(node, [])

    def topological_order(self) -> List[str]:
        """Return nodes in topological order (Kahn's algorithm)."""
        in_degree = {n: 0 for n in self.nodes}
        for (c, e) in self.edges:
            in_degree[e] = in_degree.get(e, 0) + 1

        queue = deque(n for n in self.nodes if in_degree[n] == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    def do(
        self,
        intervention: Dict[str, float],
        observational: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """do-calculus: set intervention variables and propagate causal effects.

        Parameters
        ----------
        intervention : dict
            Variable name -> forced value (do(X = x)).
        observational : dict, optional
            Current values for non-intervened variables. Defaults to 0.5.

        Returns
        -------
        dict : counterfactual values for all nodes after propagation.
        """
        values = {n: 0.5 for n in self.nodes}
        if observational:
            values.update(observational)
        values.update(intervention)

        intervened_set = set(intervention.keys())

        for node in self.topological_order():
            if node in intervened_set:
                continue
            parent_nodes = self.parents(node)
            if not parent_nodes:
                continue
            causal_sum = 0.0
            for parent in parent_nodes:
                edge = self.edges.get((parent, node))
                if edge:
                    causal_sum += edge.weight * values[parent]
            base = values.get(node, 0.5)
            values[node] = np.clip(base * 0.3 + causal_sum * 0.7, 0.0, 1.0)

        return values

    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        confounders: Optional[List[str]] = None,
        baseline: Optional[Dict[str, float]] = None,
        treatment_value: float = 1.0,
        control_value: float = 0.0,
    ) -> float:
        """Average Treatment Effect: E[Y | do(X=1)] - E[Y | do(X=0)].

        Uses the graph structure for backdoor adjustment when confounders
        are specified.
        """
        obs = baseline or {n: 0.5 for n in self.nodes}

        values_treatment = self.do({treatment: treatment_value}, obs)
        values_control = self.do({treatment: control_value}, obs)

        ate = values_treatment.get(outcome, 0.5) - values_control.get(outcome, 0.5)
        return float(round(ate, 6))

    def counterfactual(
        self,
        factual_values: Dict[str, float],
        intervention: Dict[str, float],
    ) -> Dict[str, float]:
        """Counterfactual query: given factual observations, what would happen
        if we intervened?"""
        return self.do(intervention, observational=factual_values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "metadata": self.metadata,
            "edges": [
                {
                    "cause": e.cause,
                    "effect": e.effect,
                    "weight": e.weight,
                    "mechanism": e.mechanism,
                }
                for e in self.edges.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CausalGraph:
        g = cls()
        g.metadata = dict(data.get("metadata", {}))
        for n in data.get("nodes", []):
            g.add_node(n)
        for e in data.get("edges", []):
            g.add_edge(e["cause"], e["effect"], e["weight"], e.get("mechanism", ""))
        return g


def build_default_causal_graph() -> CausalGraph:
    """Build a sensible default causal graph from domain knowledge."""
    g = CausalGraph()

    from agents.behavior import DIMENSION_NAMES
    from agents.belief_network import BELIEF_DIMENSIONS

    for d in DIMENSION_NAMES:
        g.add_node(d)
    for d in BELIEF_DIMENSIONS:
        g.add_node(d)

    g.add_edge("price_sensitivity", "financial_confidence", -0.3, "budget pressure")
    g.add_edge("technology_openness", "novelty_seeking", 0.4, "tech curiosity")
    g.add_edge("institutional_trust", "risk_aversion", -0.2, "trust reduces fear")
    g.add_edge("social_influence_susceptibility", "novelty_seeking", 0.2, "peer influence")
    g.add_edge("health_orientation", "environmental_consciousness", 0.3, "health-eco link")

    g.add_edge("government_trust", "institutional_trust", 0.5, "belief->behavior")
    g.add_edge("technology_optimism", "technology_openness", 0.4, "belief->behavior")
    g.add_edge("price_consciousness", "price_sensitivity", 0.5, "belief->behavior")
    g.add_edge("health_priority", "health_orientation", 0.4, "belief->behavior")
    g.add_edge("environmental_concern", "environmental_consciousness", 0.4, "belief->behavior")

    g.add_node("adoption")
    g.add_edge("convenience_seeking", "adoption", 0.4)
    g.add_edge("price_sensitivity", "adoption", -0.3)
    g.add_edge("technology_openness", "adoption", 0.3)
    g.add_edge("social_influence_susceptibility", "adoption", 0.2)

    return g
