"""
Vectorized Decision Engine: numpy-based batch operations for scaling
to 100k+ agents.  Replaces per-agent Python loops with matrix operations
for behavior updates, social influence, and macro aggregation.

LLM narrative generation remains per-agent (survey mode only).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.behavior import DIMENSION_NAMES, BehavioralLatentState, _N_DIMS
from agents.belief_network import BELIEF_DIMENSIONS, BeliefNetwork, _N_BELIEFS
from core.rng import ensure_np_rng


def build_trait_matrix(agents: List[Dict[str, Any]]) -> np.ndarray:
    """Build an (N x 12) matrix of latent behavioral dimensions.

    Agents without a latent_state get a row of 0.5s (neutral).
    """
    rows = []
    for a in agents:
        state = a.get("state")
        if state and hasattr(state, "latent_state"):
            rows.append(state.latent_state.to_vector())
        else:
            rows.append(np.full(_N_DIMS, 0.5))
    if not rows:
        return np.empty((0, _N_DIMS), dtype=np.float64)
    return np.vstack(rows)


def write_trait_matrix(agents: List[Dict[str, Any]], mat: np.ndarray) -> None:
    """Write an (N x 12) matrix back into agent latent states (clamped to [0,1])."""
    mat_clamped = np.clip(mat, 0.0, 1.0)
    for i, a in enumerate(agents):
        state = a.get("state")
        if state and hasattr(state, "latent_state"):
            state.latent_state = BehavioralLatentState.from_vector(mat_clamped[i])


def vectorized_decide(
    trait_matrix: np.ndarray,
    weight_vector: np.ndarray,
    n_options: int,
    temperature: float = 1.0,
    belief_matrix: np.ndarray | None = None,
    belief_weight_vector: np.ndarray | None = None,
    habit_matrix: np.ndarray | None = None,
    cultural_prior: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Batch softmax decision for N agents with conviction shaping, habit bias,
    cultural prior, and anti-collapse.

    Parameters
    ----------
    trait_matrix : (N, 12) array of behavioral dimensions
    weight_vector : (12,) weights for dimensions relevant to the question
    n_options : number of answer options
    temperature : softmax temperature
    belief_matrix : optional (N, 7) belief dimensions
    belief_weight_vector : optional (7,) belief weights for the question
    habit_matrix : optional (N, 5) habit profile values
    cultural_prior : optional (n_options,) prior from domain config
    rng : numpy random generator

    Returns
    -------
    distributions : (N, n_options) probability matrix
    """
    rng = ensure_np_rng(rng, key="vectorized_decide")
    N = trait_matrix.shape[0]
    scores = trait_matrix @ weight_vector
    scores = np.clip(scores, 0.0, 1.0)

    # Fold in belief dimensions when available
    if belief_matrix is not None and belief_weight_vector is not None:
        belief_scores = belief_matrix @ belief_weight_vector
        belief_scores = np.clip(belief_scores, 0.0, 1.0)
        scores = 0.7 * scores + 0.3 * belief_scores

    option_indices = np.arange(1, n_options + 1, dtype=np.float64)
    raw = np.outer(scores, option_indices / n_options) + np.outer(1.0 - scores, option_indices[::-1] / n_options)

    # Habit bias: agents with strong primary-service tendency get sharpened distributions
    if habit_matrix is not None and habit_matrix.shape[0] == N:
        habit_strength = habit_matrix[:, 0]  # primary_service_tendency
        sharpening = 1.0 + 0.5 * habit_strength
        raw = raw * sharpening.reshape(-1, 1)

    # Cultural prior blend
    if cultural_prior is not None and cultural_prior.shape[0] == n_options:
        cultural = cultural_prior / (cultural_prior.sum() + 1e-12)
        raw = raw + 0.15 * np.log(cultural + 1e-12)

    # Conviction shaping: push strong scores further from center
    deviation = scores - 0.5
    conviction = np.abs(deviation)
    conviction_multiplier = 1.0 + 0.4 * conviction
    raw = raw * conviction_multiplier.reshape(-1, 1)

    raw = raw / temperature
    raw -= raw.max(axis=1, keepdims=True)
    exp = np.exp(raw)
    distributions = exp / exp.sum(axis=1, keepdims=True)

    # Anti-collapse: if any option has <2% probability, inject minimum floor
    floor = 0.02
    below_floor = distributions < floor
    if below_floor.any():
        distributions = np.maximum(distributions, floor)
        distributions = distributions / distributions.sum(axis=1, keepdims=True)

    # Small per-agent noise for stochastic realism
    noise = rng.normal(0.0, 0.01, distributions.shape)
    distributions = np.clip(distributions + noise, 0.0, 1.0)
    distributions = distributions / distributions.sum(axis=1, keepdims=True)

    return distributions


def vectorized_sample(distributions: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample one option index per agent from (N, K) probability distributions.

    Returns (N,) array of sampled option indices (0-based).
    """
    rng = ensure_np_rng(rng, key="vectorized_sample")
    n, k = distributions.shape
    cumulative = distributions.cumsum(axis=1)
    u = rng.random(n).reshape(-1, 1)
    return (u < cumulative).argmax(axis=1)


def vectorized_social_influence(
    trait_matrix: np.ndarray,
    adjacency_norm,
    influence_rate: float = 0.02,
    susceptibility_idx: int = DIMENSION_NAMES.index("social_influence_susceptibility"),
) -> np.ndarray:
    """Full 12-dimension social diffusion via row-normalized adjacency.

    ``adjacency_norm`` should be pre-normalized (rows sum to 1) so that
    ``adjacency_norm @ X`` directly yields neighbor mean vectors.
    Accepts dense (N, N) ndarray or scipy sparse matrix.
    """
    neighbor_means = np.asarray(adjacency_norm @ trait_matrix)

    susceptibility = trait_matrix[:, susceptibility_idx].reshape(-1, 1)
    delta = influence_rate * susceptibility * (neighbor_means - trait_matrix)

    updated = trait_matrix + delta
    return np.clip(updated, 0.0, 1.0)


def vectorized_behavior_ema(
    trait_matrix: np.ndarray,
    answer_scores: np.ndarray,
    dim_weight_vector: np.ndarray,
    learning_rate: float = 0.05,
) -> np.ndarray:
    """Vectorized EMA update of behavioral dimensions.

    Parameters
    ----------
    trait_matrix : (N, 12)
    answer_scores : (N,) answer scores in [0, 1]
    dim_weight_vector : (12,) per-dimension relevance weights (from QUESTION_DIMENSION_MAP)
    learning_rate : EMA rate
    """
    targets = np.where(
        dim_weight_vector > 0,
        answer_scores.reshape(-1, 1),
        1.0 - answer_scores.reshape(-1, 1),
    )
    abs_weights = np.abs(dim_weight_vector)
    delta = learning_rate * abs_weights * (targets - trait_matrix)
    return np.clip(trait_matrix + delta, 0.0, 1.0)


def vectorized_macro_aggregation(trait_matrix: np.ndarray) -> Dict[str, float]:
    """Compute population-level means for each dimension (for macro feedback)."""
    if trait_matrix.shape[0] == 0:
        return {d: 0.5 for d in DIMENSION_NAMES}
    means = trait_matrix.mean(axis=0)
    return {d: float(means[i]) for i, d in enumerate(DIMENSION_NAMES)}


# ---------------------------------------------------------------------------
# Belief matrix operations (mirrors trait matrix pattern for BeliefNetwork)
# ---------------------------------------------------------------------------


def build_belief_matrix(agents: List[Dict[str, Any]]) -> np.ndarray:
    """Build an (N x 7) matrix of belief dimensions.

    Agents without beliefs get a row of 0.5s (neutral).
    """
    rows = []
    for a in agents:
        state = a.get("state")
        if state and hasattr(state, "beliefs"):
            rows.append(state.beliefs.to_vector())
        else:
            rows.append(np.full(_N_BELIEFS, 0.5))
    if not rows:
        return np.empty((0, _N_BELIEFS), dtype=np.float64)
    return np.vstack(rows)


def write_belief_matrix(agents: List[Dict[str, Any]], mat: np.ndarray) -> None:
    """Write an (N x 7) matrix back into agent belief networks (clamped to [0,1])."""
    mat_clamped = np.clip(mat, 0.0, 1.0)
    for i, a in enumerate(agents):
        state = a.get("state")
        if state and hasattr(state, "beliefs"):
            state.beliefs = BeliefNetwork.from_vector(mat_clamped[i])


def vectorized_belief_diffusion(
    belief_matrix: np.ndarray,
    adjacency_norm,
    diffusion_rate: float = 0.03,
    susceptibility: np.ndarray | None = None,
) -> np.ndarray:
    """Social diffusion of beliefs via row-normalized adjacency.

    Same pattern as ``vectorized_social_influence`` but for the 7-dim
    belief space.  When *susceptibility* (N,) is provided, each agent's
    belief shift is gated by their social-influence susceptibility.
    """
    neighbor_means = np.asarray(adjacency_norm @ belief_matrix)
    delta = diffusion_rate * (neighbor_means - belief_matrix)
    if susceptibility is not None:
        delta = delta * susceptibility.reshape(-1, 1)
    return np.clip(belief_matrix + delta, 0.0, 1.0)


# ---------------------------------------------------------------------------
# StateMatrix: columnar store for 50k+ agents
# ---------------------------------------------------------------------------

_N_HABITS = 5  # primary_service_tendency, alternative_tendency, budget_consciousness, health_strictness, tech_comfort


class StateMatrix:
    """Columnar numpy store for N agents' numerical state.

    Primary data lives in dense arrays.  Bulk operations (social diffusion,
    macro feedback, cultural influence) operate directly on the matrices,
    avoiding per-object Python overhead.

    Use ``sync_from_agents`` to pull current ``AgentState`` values into the
    matrix and ``sync_to_agents`` to push matrix values back.
    """

    def __init__(self, n: int) -> None:
        self.latent = np.full((n, _N_DIMS), 0.5)
        self.beliefs = np.full((n, _N_BELIEFS), 0.5)
        self.habits = np.zeros((n, _N_HABITS))
        self._agent_ids: List[str] = [""] * n
        self._id_to_row: Dict[str, int] = {}

    @classmethod
    def from_agents(cls, agents: List[Dict[str, Any]]) -> "StateMatrix":
        sm = cls(len(agents))
        sm.sync_from_agents(agents)
        return sm

    def sync_from_agents(self, agents: List[Dict[str, Any]]) -> None:
        """Read ``AgentState`` objects into matrix rows."""
        for i, a in enumerate(agents):
            persona = a.get("persona")
            aid = persona.agent_id if persona else f"__anon_{i}"
            self._agent_ids[i] = aid
            self._id_to_row[aid] = i

            state = a.get("state")
            if state is None:
                continue
            if hasattr(state, "latent_state"):
                self.latent[i] = state.latent_state.to_vector()
            if hasattr(state, "beliefs"):
                self.beliefs[i] = state.beliefs.to_vector()
            hp = getattr(state, "habit_profile", None)
            if hp is not None:
                self.habits[i] = np.array([
                    hp.primary_service_tendency,
                    hp.alternative_tendency,
                    hp.budget_consciousness,
                    hp.health_strictness,
                    hp.tech_comfort,
                ])

    def sync_to_agents(self, agents: List[Dict[str, Any]]) -> None:
        """Write matrix rows back into ``AgentState`` objects."""
        latent_clamped = np.clip(self.latent, 0.0, 1.0)
        belief_clamped = np.clip(self.beliefs, 0.0, 1.0)
        for i, a in enumerate(agents):
            state = a.get("state")
            if state is None:
                continue
            if hasattr(state, "latent_state"):
                state.latent_state = BehavioralLatentState.from_vector(latent_clamped[i])
            if hasattr(state, "beliefs"):
                state.beliefs = BeliefNetwork.from_vector(belief_clamped[i])

    def row(self, agent_id: str) -> int:
        """Return the row index for *agent_id*."""
        return self._id_to_row[agent_id]

    def __len__(self) -> int:
        return len(self._agent_ids)
