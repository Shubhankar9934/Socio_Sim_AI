"""
Generic probabilistic decision engine: P(response | persona, context, factors).

Uses a Factor Graph to combine personality, income, social, location, belief,
and memory influences into a single behavioural score, then converts that
score into a probability distribution over answer options via softmax.

Stochastic enhancements:
  - Per-agent softmax temperature derived from personality traits
  - Dirichlet noise injection for distribution fingerprinting
  - Per-agent factor-weight perturbation
  - Conviction profiles (certain/leaning/diffuse/bimodal/anchored)
  - Cultural behavior priors (nationality × family × income)
  - Demographic plausibility resampling

A cognitive dissonance adjustment is applied post-softmax to enforce
consistency with the agent's stored beliefs and behavioral state.

Works for **any** scale length and **any** survey domain — the behaviour is
entirely driven by the QuestionModel config in config/question_models.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from agents.factor_graph import DecisionContext, FactorGraph, get_or_build_graph
from agents.factors import build_factor_graph
from agents.perception import Perception, detect_question_model
from agents.personality import PersonalityTraits
from agents.realism import (
    apply_conviction_shaping,
    apply_habit_bias,
    assign_conviction_profile,
    get_cultural_prior,
    suggest_plausible_resampling,
)
from config.question_models import QuestionModel
from config.reference_distributions import get_reference_distribution
from population.personas import Persona

if TYPE_CHECKING:
    from agents.state import AgentState


# ── Softmax utility ─────────────────────────────────────────────────────

def _softmax(x: List[float], temperature: float = 1.0) -> List[float]:
    """Softmax with temperature control. Higher temperature = more uniform."""
    arr = np.array(x, dtype=np.float64) / temperature
    arr -= arr.max()
    e = np.exp(arr)
    probs = e / e.sum()
    return probs.tolist()


# ── Per-agent softmax temperature ────────────────────────────────────────

def _agent_softmax_temperature(
    base_temp: float,
    persona: Persona,
    traits: PersonalityTraits,
) -> float:
    """Derive a per-agent softmax temperature from personality.

    Decisive agents (high convenience pref, low price sensitivity) get a lower
    temperature (peakier distributions).  Indecisive agents get higher temperature
    (flatter, more uniform distributions).
    """
    decisiveness = traits.convenience_preference * (1.0 - traits.price_sensitivity)
    agent_temp = base_temp * (0.7 + 0.6 * decisiveness)
    return max(0.4, min(2.5, agent_temp))


# ── Entropy-adaptive Dirichlet noise injection ──────────────────────────

_DIRICHLET_ALPHA = 0.4
_NOISE_MIN = 0.04
_NOISE_MAX = 0.08


def _inject_dirichlet_noise(
    probs: List[float],
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """Mix Dirichlet noise into a probability vector, scaled by entropy.

    Confident (peaky) distributions get minimal noise (_NOISE_MIN).
    Uncertain (flat) distributions get more noise (_NOISE_MAX).
    This preserves decisive agents' sharpness while adding realistic
    spread to indecisive agents.
    """
    n = len(probs)
    if n < 2:
        return probs
    gen = rng if rng is not None else np.random.default_rng()

    probs_arr = np.array(probs, dtype=np.float64)
    probs_arr = np.clip(probs_arr, 1e-12, None)
    entropy = -np.sum(probs_arr * np.log(probs_arr))
    max_entropy = np.log(n)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
    noise_strength = _NOISE_MIN + (_NOISE_MAX - _NOISE_MIN) * normalized_entropy

    noise = gen.dirichlet([_DIRICHLET_ALPHA] * n)
    blended = (1 - noise_strength) * probs_arr + noise_strength * noise
    blended = blended / blended.sum()
    return blended.tolist()


# ── Factor weight perturbation ───────────────────────────────────────────

_FACTOR_WEIGHT_NOISE_STD = 0.10


def _perturbed_graph_score(
    graph: FactorGraph,
    context: DecisionContext,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute graph score with per-agent Gaussian noise on factor weights."""
    if not graph._factors:
        return 0.5

    gen = rng if rng is not None else np.random.default_rng()
    score = 0.0
    total_weight = 0.0
    for fn, w in graph._factors:
        raw = fn(context)
        clamped = max(0.0, min(1.0, raw))
        perturbed_w = w * (1.0 + gen.normal(0, _FACTOR_WEIGHT_NOISE_STD))
        perturbed_w = max(0.01, perturbed_w)
        score += perturbed_w * clamped
        total_weight += abs(perturbed_w)

    if total_weight == 0:
        return 0.5
    return max(0.0, min(1.0, score / total_weight))


# ── Per-option noise, reference prior, conviction spikes ─────────────────

_RAW_SCORE_NOISE_STD = 0.12
_PRIOR_WEIGHT = 0.20
_CULTURAL_PRIOR_WEIGHT = 0.18
_SPIKE_PROB = 0.15
_SPIKE_BOOST = 0.4


# ── Generic distribution generator ──────────────────────────────────────

def compute_distribution(
    question_model: QuestionModel,
    context: DecisionContext,
    agent_state: Optional["AgentState"] = None,
    persona: Optional[Persona] = None,
    traits: Optional[PersonalityTraits] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Compute a probability distribution over the model's scale.

    Pipeline (11-stage):
     1. Factor graph inference with per-agent weight perturbation -> score in [0,1]
     2. Logit-based raw scores (Gaussian noise on symmetric weight vector)
     3. Blend with reference-distribution prior (population-level shape)
     3.5. Habit profile bias injection (cross-survey consistency)
     4. Blend with cultural behavior prior (nationality x family x income)
     5. Per-option Gaussian noise
     6. Random conviction spike (~15% of agents)
     7. Per-agent softmax temperature -> valid probability distribution
     8. Dirichlet noise for per-agent uniqueness
     9. Conviction profile shaping (certain/bimodal/diffuse/anchored)
    10. Cognitive dissonance adjustment (if agent_state available)
    """
    gen = rng if rng is not None else np.random.default_rng()

    graph = get_or_build_graph(
        question_model.name,
        lambda: build_factor_graph(question_model),
    )
    score = _perturbed_graph_score(graph, context, rng=rng)

    # Stage 2: trait-vector logit scoring
    # Build option embeddings from question dimension weights x trait vector,
    # producing multi-dimensional logits instead of a scalar ramp.
    n = len(question_model.scale)
    position = np.linspace(-1.0, 1.0, n)

    if traits is not None and question_model.dimension_weights:
        trait_names = list(question_model.dimension_weights.keys())
        dim_weights = np.array([question_model.dimension_weights[t] for t in trait_names])
        trait_vals = np.array([getattr(traits, t, 0.5) for t in trait_names])
        # option_embeddings[t, o] = dim_weight[t] * position[o]
        option_embeddings = np.outer(dim_weights, position)
        raw_logits = trait_vals @ option_embeddings
    else:
        raw_logits = score * position

    logit_noise = gen.normal(0, 0.25, size=n)
    raw_scores = (raw_logits + logit_noise).tolist()

    # Stage 3: reference prior
    ref = get_reference_distribution(question_model.name, question_model.scale)
    if ref:
        ref_vals = [ref.get(opt, 1.0 / n) for opt in question_model.scale]
        raw_scores = [
            (1 - _PRIOR_WEIGHT) * r + _PRIOR_WEIGHT * rv
            for r, rv in zip(raw_scores, ref_vals)
        ]

    # Stage 3.5: habit profile bias (cross-survey consistency)
    if agent_state is not None and agent_state.habit_profile is not None:
        raw_scores = apply_habit_bias(
            raw_scores, question_model.scale, agent_state.habit_profile,
        )

    # Stage 4: cultural behavior prior
    if persona is not None:
        cultural = get_cultural_prior(persona, scale=question_model.scale)
        if cultural:
            cultural_vals = [cultural.get(opt, 1.0 / n) for opt in question_model.scale]
            raw_scores = [
                (1 - _CULTURAL_PRIOR_WEIGHT) * r + _CULTURAL_PRIOR_WEIGHT * cv
                for r, cv in zip(raw_scores, cultural_vals)
            ]

    # Stage 5: per-option noise
    option_noise = gen.normal(0, _RAW_SCORE_NOISE_STD, size=n)
    raw_scores = [r + float(ns) for r, ns in zip(raw_scores, option_noise)]

    # Stage 6: conviction spike
    if gen.random() < _SPIKE_PROB:
        spike_idx = int(gen.integers(0, n))
        raw_scores[spike_idx] += _SPIKE_BOOST

    # Stage 7: softmax with per-agent temperature
    temp = question_model.temperature
    if persona is not None and traits is not None:
        temp = _agent_softmax_temperature(temp, persona, traits)

    # Activation modulation: high emotional activation lowers temperature
    # (sharper, more extreme choices) -- range [0.6*temp, temp].
    _act = context.environment.get("activation", 0.0) if context else 0.0
    temp *= max(0.6, 1.0 - 0.4 * float(_act))

    probs = _softmax(raw_scores, temp)

    # Stage 8: Dirichlet noise
    probs = _inject_dirichlet_noise(probs, rng=rng)

    # Stage 9: conviction profile shaping
    if persona is not None:
        profile = assign_conviction_profile(persona, rng=gen)
        probs = apply_conviction_shaping(probs, profile, rng=gen)

    total = sum(probs)
    if abs(total - 1.0) > 1e-6:
        probs = [p / total for p in probs]
    assert all(p >= 0 for p in probs), f"Negative probability detected: {probs}"

    dist = dict(zip(question_model.scale, probs))

    # Stage 10: cognitive dissonance
    if agent_state is not None:
        from agents.dissonance import apply_cognitive_dissonance, compute_consistency_score

        consistency = compute_consistency_score(agent_state, question_model)
        dist = apply_cognitive_dissonance(
            dist, consistency, question_model.scale, agent_state=agent_state,
        )

    # Stage 11: cross-question memory bias
    if agent_state is not None and agent_state.structured_memory:
        from agents.memory_rules import apply_memory_rules
        dist = apply_memory_rules(dist, question_model.name, agent_state.structured_memory)

    # Stage 12: bounded-rational bias pipeline (confirmation, loss aversion,
    # anchoring, bandwagon, availability) with residual mixing + entropy floor
    if agent_state is not None:
        from agents.biases import apply_all_biases

        bias_context: Dict[str, Any] = {
            "topic": question_model.name,
            "topic_importance": context.environment.get("topic_importance", 0.5) if context else 0.5,
            "media_conflict": context.environment.get("media_conflict", 0.0) if context else 0.0,
            "behavioral_dimension_weights": context.environment.get("behavioral_dimension_weights", {}) if context else {},
            "recent_event_scores": context.environment.get("recent_event_scores"),
        }
        neighbor_dist = context.environment.get("neighbor_distribution") if context else None
        prior_dist = context.environment.get("prior_distribution") if context else None

        dist = apply_all_biases(
            dist, question_model.scale, agent_state, bias_context,
            neighbor_dist_dict=neighbor_dist,
            prior_dist_dict=prior_dist,
        )

    # Stage 13: Goal/utility blending (gentle nudge from active goals)
    if agent_state is not None:
        from agents.utility import blend_utility_into_distribution
        goal_profile = getattr(agent_state, "goal_profile", None)
        dist = blend_utility_into_distribution(dist, goal_profile, agent_state)

    return dist


# ── Sampling ────────────────────────────────────────────────────────────

_NUCLEUS_P = 0.92
_RELATIVE_FLOOR = 0.20
_RESAMPLE_FLOOR = 0.35


def sample_from_distribution(
    dist: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
    **_kwargs: Any,
) -> str:
    """Nucleus (top-p) sampling with a relative probability floor and
    post-sample resample guard.

    1. Sort options by descending probability.
    2. Accumulate until reaching ``_NUCLEUS_P`` (92%) of total mass.
    3. Exclude any option where P(option) < ``_RELATIVE_FLOOR`` * P(top).
    4. Keep at least the top-2 options as a safety floor.
    5. Renormalize and sample.
    6. If sampled option's original P < ``_RESAMPLE_FLOOR`` * P(top),
       resample from top-2 only.

    This prevents extreme mismatches (e.g. P(top)=0.74, sampled at 0.12)
    while still allowing minor stochastic variation.
    """
    gen = rng if rng is not None else np.random.default_rng()

    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    top_prob = items[0][1]

    nucleus: List[tuple] = []
    mass = 0.0

    for opt, p in items:
        if p < top_prob * _RELATIVE_FLOOR:
            continue
        nucleus.append((opt, p))
        mass += p
        if mass >= _NUCLEUS_P:
            break

    if len(nucleus) < 2:
        nucleus = list(items[:2])

    opts, probs_arr = zip(*nucleus)
    probs = np.array(probs_arr, dtype=np.float64)
    probs = probs / probs.sum()

    sampled = str(gen.choice(opts, p=probs))

    if dist.get(sampled, 0.0) < top_prob * _RESAMPLE_FLOOR:
        top2 = list(items[:2])
        t2_opts, t2_probs = zip(*top2)
        t2_p = np.array(t2_probs, dtype=np.float64)
        t2_p = t2_p / t2_p.sum()
        sampled = str(gen.choice(t2_opts, p=t2_p))

    return sampled


# ── Main entry point ────────────────────────────────────────────────────

def decide(
    perception: Perception,
    persona: Persona,
    traits: PersonalityTraits,
    *,
    friends_using: float = 0.0,
    location_quality: float = 0.5,
    memories: Optional[List[str]] = None,
    environment: Optional[Dict[str, Any]] = None,
    agent_state: Optional["AgentState"] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Dict[str, float], str]:
    """Compute probability distribution and sampled answer.

    Fully generic: the QuestionModel is resolved from the Perception,
    so no question-type-specific branching is needed.
    The *environment* dict is merged into the DecisionContext so factors
    can read event-driven world parameters, the agent's latent state,
    and belief network.

    If ``agent_state`` is provided, cognitive dissonance adjustment is
    applied after the base distribution is computed.

    If ``rng`` is provided, sampling is deterministic.
    """
    from config.belief_mappings import get_belief_dimensions
    from config.question_models import get_behavioral_dimensions

    question_model = detect_question_model(perception)
    if not question_model.scale:
        return {}, ""

    qm_key = question_model.name

    env: Dict[str, Any] = {"dimension_weights": dict(question_model.dimension_weights)}
    env["behavioral_dimension_weights"] = get_behavioral_dimensions(qm_key)
    env["belief_dimension_weights"] = get_belief_dimensions(qm_key)
    if agent_state is not None:
        env.setdefault("beliefs", agent_state.beliefs)
    if environment:
        env.update(environment)

    context = DecisionContext(
        persona=persona,
        traits=traits,
        perception=perception,
        friends_using=friends_using,
        location_quality=location_quality,
        memories=memories or [],
        environment=env,
    )

    dist = compute_distribution(
        question_model, context,
        agent_state=agent_state,
        persona=persona,
        traits=traits,
        rng=rng,
    )
    chosen = sample_from_distribution(dist, rng=rng)

    # Demographic plausibility gate: catch implausible persona-answer combos
    dist, chosen = suggest_plausible_resampling(persona, dist, chosen, rng=rng)

    return dist, chosen


def decide_as_action(
    perception: Perception,
    persona: Persona,
    traits: PersonalityTraits,
    *,
    friends_using: float = 0.0,
    location_quality: float = 0.5,
    memories: Optional[List[str]] = None,
    environment: Optional[Dict[str, Any]] = None,
    agent_state: Optional["AgentState"] = None,
    rng: Optional[np.random.Generator] = None,
    action_template: Optional[Any] = None,
) -> "Action":
    """Like decide() but returns a universal Action object."""
    from agents.actions import Action, ActionTemplate

    dist, chosen = decide(
        perception, persona, traits,
        friends_using=friends_using, location_quality=location_quality,
        memories=memories, environment=environment,
        agent_state=agent_state, rng=rng,
    )

    question_model = detect_question_model(perception)
    scale = question_model.scale or []
    idx = scale.index(chosen) if chosen in scale else len(scale) // 2
    answer_score = idx / max(1, len(scale) - 1)

    if action_template and isinstance(action_template, ActionTemplate):
        at, tgt = action_template.action_type, action_template.target
    else:
        at, tgt = "choose", "behavior"

    return Action.from_survey_answer(
        agent_id=persona.agent_id,
        question=perception.raw_question,
        answer=chosen,
        answer_score=answer_score,
        action_type=at,
        target=tgt,
    )
