"""
Homophily-weighted opinion diffusion and adoption cascade.
Similar personas exert stronger influence on each other.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import networkx as nx

from population.personas import Persona
from social.network import agent_id_to_node, neighbors, node_to_agent_id


# ---------------------------------------------------------------------------
# Persona similarity (homophily kernel)
# ---------------------------------------------------------------------------

_CATEGORICAL_WEIGHTS = {
    "nationality": 0.25,
    "age": 0.15,
    "location": 0.15,
    "income": 0.10,
    "occupation": 0.05,
}
_LIFESTYLE_WEIGHT = 0.30  # cosine similarity of the 6 lifestyle coefficients


def _lifestyle_vector(p: Persona) -> np.ndarray:
    ls = p.lifestyle
    return np.array([
        ls.luxury_preference,
        ls.tech_adoption,
        ls.dining_out,
        ls.convenience_preference,
        ls.price_sensitivity,
        ls.primary_service_preference,
    ], dtype=np.float64)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0:
        return 0.0
    return dot / norm


def persona_similarity(a: Persona, b: Persona) -> float:
    """Compute 0-1 homophily score between two personas.

    Combines categorical attribute matches (nationality, age, location,
    income, occupation) with cosine similarity over lifestyle coefficients.
    """
    score = (
        _CATEGORICAL_WEIGHTS["nationality"] * (a.nationality == b.nationality)
        + _CATEGORICAL_WEIGHTS["age"] * (a.age == b.age)
        + _CATEGORICAL_WEIGHTS["location"] * (a.location == b.location)
        + _CATEGORICAL_WEIGHTS["income"] * (a.income == b.income)
        + _CATEGORICAL_WEIGHTS["occupation"] * (a.occupation == b.occupation)
        + _LIFESTYLE_WEIGHT * _cosine_similarity(
            _lifestyle_vector(a), _lifestyle_vector(b),
        )
    )
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# Edge-weight helpers (read pre-computed similarity from graph)
# ---------------------------------------------------------------------------

def _edge_similarity(G: nx.Graph, u_node: int, v_node: int) -> float:
    """Read similarity weight from edge; default 1.0 for backward compat."""
    return G[u_node][v_node].get("similarity", 1.0)


# ---------------------------------------------------------------------------
# Adoption probability (unchanged signature, now receives weighted fraction)
# ---------------------------------------------------------------------------

def adoption_probability(
    base_prob: float,
    weighted_using: float,
    weighted_total: float,
    influence_factor: float = 0.05,
) -> float:
    """P(adopt) = base_prob + influence_factor * (weighted_using / weighted_total).

    Parameters are now similarity-weighted sums rather than raw counts.
    """
    if weighted_total <= 0:
        return base_prob
    social = influence_factor * (weighted_using / weighted_total)
    return min(1.0, max(0.0, base_prob + social))


# ---------------------------------------------------------------------------
# Trait counting (similarity-weighted)
# ---------------------------------------------------------------------------

def count_friends_with_trait(
    G: nx.Graph,
    agent_id: str,
    trait_by_agent: Dict[str, bool],
) -> int:
    """Number of neighbors for whom trait_by_agent[id] is True (unweighted)."""
    neighbor_ids = neighbors(G, agent_id)
    return sum(1 for nid in neighbor_ids if trait_by_agent.get(nid, False))


def fraction_friends_with_trait(
    G: nx.Graph,
    agent_id: str,
    trait_by_agent: Dict[str, bool],
) -> float:
    """Similarity-weighted fraction of neighbors with trait.

    Each neighbor's contribution is scaled by the pre-computed edge
    similarity weight.  Falls back to uniform weight (1.0) if the
    similarity attribute is missing.
    """
    node = agent_id_to_node(G, agent_id)
    if node is None:
        return 0.0

    weighted_trait = 0.0
    weighted_total = 0.0
    for n_node in G.neighbors(node):
        nid = node_to_agent_id(G, n_node)
        sim = _edge_similarity(G, node, n_node)
        weighted_total += sim
        if trait_by_agent.get(nid, False):
            weighted_trait += sim

    if weighted_total <= 0:
        return 0.0
    return weighted_trait / weighted_total


# ---------------------------------------------------------------------------
# Adoption cascade (similarity-weighted)
# ---------------------------------------------------------------------------

def cascade_step(
    G: nx.Graph,
    current_adopters: Dict[str, bool],
    base_prob_fn: Callable[[str], float],
    influence_factor: float = 0.05,
    rng=None,
) -> Dict[str, bool]:
    """One step of adoption cascade with homophily-weighted influence.

    Each friend's influence is scaled by the edge similarity weight.
    """
    import random
    rng = rng or random
    next_adopters = dict(current_adopters)
    agent_ids = list(G.graph.get("agent_ids", []))

    for i, agent_id in enumerate(agent_ids):
        if next_adopters.get(agent_id):
            continue

        base = base_prob_fn(agent_id)
        node = agent_id_to_node(G, agent_id)
        if node is None:
            continue

        weighted_using = 0.0
        weighted_total = 0.0
        for n_node in G.neighbors(node):
            nid = node_to_agent_id(G, n_node)
            sim = _edge_similarity(G, node, n_node)
            weighted_total += sim
            if current_adopters.get(nid, False):
                weighted_using += sim

        p = adoption_probability(base, weighted_using, weighted_total, influence_factor)
        if rng.random() < p:
            next_adopters[agent_id] = True

    return next_adopters


def run_cascade(
    G: nx.Graph,
    initial_adopters: Optional[Dict[str, bool]] = None,
    base_prob_fn: Optional[Callable[[str], float]] = None,
    influence_factor: float = 0.05,
    max_steps: int = 20,
    rng=None,
) -> Dict[str, bool]:
    """Run adoption cascade until convergence or max_steps."""
    import random
    rng = rng or random
    base_prob_fn = base_prob_fn or (lambda _: 0.2)
    current = dict(initial_adopters or {})
    for _ in range(max_steps):
        next_ = cascade_step(G, current, base_prob_fn, influence_factor, rng)
        if next_ == current:
            break
        current = next_
    return current
