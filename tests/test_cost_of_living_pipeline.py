from types import SimpleNamespace

import numpy as np

from agents.decision import compute_distribution
from agents.factor_graph import DecisionContext
from agents.personality import personality_from_persona
from agents.state import AgentState
from config.question_models import QUESTION_MODELS
from population.synthesis import generate_population


def _ctx(persona):
    return DecisionContext(
        persona=persona,
        traits=personality_from_persona(persona),
        perception=SimpleNamespace(raw_question="How satisfied are you with cost of living?"),
        friends_using=0.2,
        location_quality=0.5,
        memories=[],
        environment={"activation": 0.1},
    )


def test_cost_of_living_constraints_hard_zero():
    base = generate_population(1, method="bayesian", seed=777)[0]
    persona = base.model_copy(update={"income": "<10k", "household_size": "5+"})
    state = AgentState.from_persona(persona)
    traits = personality_from_persona(persona)
    qm = QUESTION_MODELS["cost_of_living_satisfaction"]

    dist = compute_distribution(qm, _ctx(persona), agent_state=state, persona=persona, traits=traits, rng=np.random.default_rng(41))
    assert dist["4"] == 0.0
    assert dist["5"] == 0.0


def test_cost_of_living_demographic_separation_expected_rating():
    qm = QUESTION_MODELS["cost_of_living_satisfaction"]
    index_map = {k: i for i, k in enumerate(qm.scale)}
    personas = generate_population(450, method="bayesian", seed=242)

    low_scores = []
    high_scores = []
    for p in personas:
        state = AgentState.from_persona(p)
        traits = personality_from_persona(p)
        dist = compute_distribution(qm, _ctx(p), agent_state=state, persona=p, traits=traits, rng=np.random.default_rng(51))
        expected_idx = sum(index_map[k] * v for k, v in dist.items())
        if p.income in ("<10k", "10-25k") and p.household_size == "5+":
            low_scores.append(expected_idx)
        if p.income == "50k+" and p.household_size in ("1", "2"):
            high_scores.append(expected_idx)

    assert low_scores and high_scores
    assert (sum(low_scores) / len(low_scores)) < (sum(high_scores) / len(high_scores))


def test_cost_of_living_middle_collapse_guard():
    qm = QUESTION_MODELS["cost_of_living_satisfaction"]
    personas = generate_population(220, method="bayesian", seed=303)
    top3 = 0
    for i, p in enumerate(personas):
        state = AgentState.from_persona(p)
        traits = personality_from_persona(p)
        dist = compute_distribution(qm, _ctx(p), agent_state=state, persona=p, traits=traits, rng=np.random.default_rng(1000 + i))
        top = max(dist, key=dist.get)
        if top == "3":
            top3 += 1
    share = top3 / max(1, len(personas))
    assert share < 0.90

