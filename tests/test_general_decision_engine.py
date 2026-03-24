from types import SimpleNamespace

import numpy as np

from agents.decision import compute_distribution
from agents.factor_graph import DecisionContext
from agents.personality import personality_from_persona
from agents.state import AgentState
from config.question_models import QUESTION_MODELS, load_generated_models_into_registry
from population.synthesis import generate_population


def _ctx(persona, question: str, *, location_quality: float = 0.5):
    return DecisionContext(
        persona=persona,
        traits=personality_from_persona(persona),
        perception=SimpleNamespace(raw_question=question),
        friends_using=0.2,
        location_quality=location_quality,
        memories=[],
        environment={"activation": 0.05},
    )


def test_policy_question_responds_to_belief_difference():
    qm = QUESTION_MODELS["policy_support"]
    persona = generate_population(1, method="bayesian", seed=141)[0]

    support_state = AgentState.from_persona(persona)
    support_state.beliefs.set("government_trust", 0.9)
    support_state.beliefs.set("environmental_concern", 0.8)

    oppose_state = AgentState.from_persona(persona)
    oppose_state.beliefs.set("government_trust", 0.2)
    oppose_state.beliefs.set("environmental_concern", 0.2)

    traits = personality_from_persona(persona)
    question = "Do you support the new city policy to reduce congestion and emissions?"
    support_dist = compute_distribution(qm, _ctx(persona, question), agent_state=support_state, persona=persona, traits=traits, rng=np.random.default_rng(77))
    oppose_dist = compute_distribution(qm, _ctx(persona, question), agent_state=oppose_state, persona=persona, traits=traits, rng=np.random.default_rng(77))

    support_score = support_dist["support"] + support_dist["strongly support"]
    oppose_score = oppose_dist["oppose"] + oppose_dist["strongly oppose"]
    assert support_score > 0.45
    assert oppose_score > 0.30
    assert support_dist["strongly support"] > oppose_dist["strongly support"]


def test_center_collapse_reduced_across_multiple_question_families():
    personas = generate_population(160, method="bayesian", seed=808)
    questions = [
        ("cost_of_living_satisfaction", "How satisfied are you with the cost of living in your city?"),
        ("policy_support", "Do you support the new city policy to reduce congestion and emissions?"),
        ("tech_adoption_likelihood", "How likely are you to try a new public service app next month?"),
    ]

    for model_key, question in questions:
        qm = QUESTION_MODELS[model_key]
        center = 0
        for i, persona in enumerate(personas):
            state = AgentState.from_persona(persona)
            traits = personality_from_persona(persona)
            dist = compute_distribution(qm, _ctx(persona, question), agent_state=state, persona=persona, traits=traits, rng=np.random.default_rng(500 + i))
            top = max(dist, key=dist.get)
            if top in {"3", "neutral", "Neutral"}:
                center += 1
        share = center / len(personas)
        assert share < 0.92


def test_safety_question_separates_high_and_low_safety_contexts():
    load_generated_models_into_registry(force=True)
    qm = QUESTION_MODELS["neighborhood_safety_perception"]
    base = generate_population(1, method="bayesian", seed=909)[0]

    high = base.model_copy(
        update={
            "income": "50k+",
            "location": "Jumeirah",
            "age": "25-34",
            "household_size": "1",
            "personal_anchors": base.personal_anchors.model_copy(update={"commute_method": "car"}),
        }
    )
    low = base.model_copy(
        update={
            "income": "<10k",
            "location": "Deira",
            "age": "55+",
            "household_size": "5+",
            "personal_anchors": base.personal_anchors.model_copy(update={"commute_method": "walk"}),
        }
    )

    def _expected(persona):
        state = AgentState.from_persona(persona)
        traits = personality_from_persona(persona)
        dist = compute_distribution(
            qm,
            _ctx(persona, "How safe do you feel walking alone in your neighborhood at night? Options: 1 (Very unsafe), 2, 3, 4, 5 (Very safe)"),
            agent_state=state,
            persona=persona,
            traits=traits,
            rng=np.random.default_rng(123),
        )
        index = {label: i for i, label in enumerate(qm.scale)}
        return sum(index[label] * prob for label, prob in dist.items()), max(dist, key=dist.get)

    high_score, high_top = _expected(high)
    low_score, low_top = _expected(low)
    assert high_score > low_score
    assert high_top in {"4", "5 (Very safe)"}
    assert low_top in {"1 (Very unsafe)", "2", "3"}

