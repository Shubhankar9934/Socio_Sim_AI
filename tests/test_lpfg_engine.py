from types import SimpleNamespace

import numpy as np

from agents.decision import compute_distribution, sample_from_distribution
from agents.factor_graph import DecisionContext
from agents.personality import personality_from_persona
from agents.state import AgentState
from config.question_models import QUESTION_MODELS, QuestionModel
from population.synthesis import generate_population


def _single_agent_context():
    persona = generate_population(1, method="bayesian", seed=123)[0]
    state = AgentState.from_persona(persona)
    traits = personality_from_persona(persona)
    qm = QUESTION_MODELS["food_delivery_frequency"]
    ctx = DecisionContext(
        persona=persona,
        traits=traits,
        perception=SimpleNamespace(raw_question="How often do you order delivery?"),
        friends_using=0.2,
        location_quality=0.5,
        memories=[],
        environment={"activation": 0.1},
    )
    return persona, state, traits, qm, ctx


def test_dominance_fusion_scale_reduces_effective_boost(monkeypatch):
    from agents import decision as dec

    logits = np.zeros(5, dtype=np.float64)
    scale_keys = ["a", "b", "c", "d", "e"]
    factor_signals = [
        {
            "factor": "behavioral",
            "raw_scalar": 0.95,
            "importance": 0.4,
            "centered": 0.45,
            "weight": 0.2,
        },
        {"factor": "x", "raw_scalar": 0.5, "importance": 0.01, "centered": 0.0, "weight": 0.1},
    ]
    question_spec = {"dominant_factors": ["behavioral"]}

    class _SFull:
        dominance_fusion_scale = 1.0
        dominance_fusion_boost_cap = 2.0

    monkeypatch.setattr(dec, "get_settings", lambda: _SFull())
    trace_full: dict = {}
    out_full = dec._apply_dominance_fusion(
        logits.copy(), scale_keys, factor_signals, question_spec, trace_full,
    )

    class _SSoft:
        dominance_fusion_scale = 0.5
        dominance_fusion_boost_cap = 2.0

    monkeypatch.setattr(dec, "get_settings", lambda: _SSoft())
    trace_soft: dict = {}
    out_soft = dec._apply_dominance_fusion(
        logits.copy(), scale_keys, factor_signals, question_spec, trace_soft,
    )

    assert trace_full["dominance"]["boost_strength"] > trace_soft["dominance"]["boost_strength"]
    assert np.linalg.norm(out_full - logits) > np.linalg.norm(out_soft - logits)


def test_dominance_fusion_boost_cap_limits_boost_strength(monkeypatch):
    from agents import decision as dec

    logits = np.zeros(5, dtype=np.float64)
    scale_keys = ["a", "b", "c", "d", "e"]
    factor_signals = [
        {
            "factor": "behavioral",
            "raw_scalar": 0.99,
            "importance": 0.5,
            "centered": 0.49,
            "weight": 0.3,
        },
        {"factor": "x", "raw_scalar": 0.5, "importance": 0.01, "centered": 0.0, "weight": 0.1},
    ]
    question_spec = {"dominant_factors": ["behavioral"]}

    class _S:
        dominance_fusion_scale = 1.0
        dominance_fusion_boost_cap = 0.45

    monkeypatch.setattr(dec, "get_settings", lambda: _S())
    trace: dict = {}
    dec._apply_dominance_fusion(
        logits.copy(), scale_keys, factor_signals, question_spec, trace,
    )
    assert trace["dominance"]["boost_strength"] == 0.45
    assert trace["dominance"]["boost_strength_pre_cap"] > 0.45


def test_anti_collapse_trace_records_skip_for_non_likert5():
    persona, state, traits, _, ctx = _single_agent_context()
    qm = QuestionModel(
        name="test_three_option",
        scale=["low", "mid", "high"],
        factor_weights={"personality": 0.5, "income": 0.25, "behavioral": 0.25},
        temperature=1.0,
    )
    compute_distribution(
        qm,
        ctx,
        agent_state=state,
        persona=persona,
        traits=traits,
        rng=np.random.default_rng(42),
    )
    trace = ctx.environment.get("__decision_trace", {})
    ac = trace.get("anti_collapse") or {}
    assert ac.get("skipped") is True
    assert ac.get("reason") == "not_likert5"


def test_lpfg_distribution_is_seed_deterministic():
    persona, state, traits, qm, ctx = _single_agent_context()
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    d1 = compute_distribution(qm, ctx, agent_state=state, persona=persona, traits=traits, rng=rng1)
    d2 = compute_distribution(qm, ctx, agent_state=state, persona=persona, traits=traits, rng=rng2)
    assert d1 == d2
    assert abs(sum(d1.values()) - 1.0) < 1e-9
    trace = ctx.environment.get("__decision_trace", {})
    assert trace.get("factor_contributions")
    assert trace.get("stages")
    assert "pre_constraints" in trace["stages"]
    assert "post_constraints" in trace["stages"]
    assert "post_biases" in trace["stages"]
    assert "post_calibration" in trace["stages"]
    sample_entry = trace["factor_contributions"][0]
    assert "type" in sample_entry
    assert "raw_factor_output" in sample_entry
    assert "applied_delta_norm" in sample_entry


def test_hard_constraint_enforced_both_stages(monkeypatch):
    persona, state, traits, qm, ctx = _single_agent_context()
    combo = {
        "filters": {"income": persona.income, "household_size": persona.household_size},
        "option": "multiple per day",
        "severity": "hard",
    }
    cfg = SimpleNamespace(implausible_combos=[combo], factor_couplings=[])
    monkeypatch.setattr("config.domain.get_domain_config", lambda *args, **kwargs: cfg)
    d = compute_distribution(
        qm,
        ctx,
        agent_state=state,
        persona=persona,
        traits=traits,
        rng=np.random.default_rng(7),
    )
    assert d["multiple per day"] == 0.0
    trace = ctx.environment.get("__decision_trace", {})
    assert any(c.get("stage") == "log_space" for c in trace.get("constraints_applied", []))
    assert any(c.get("stage") == "prob_space" for c in trace.get("constraints_applied", []))


def test_entropy_aware_sampling_prefers_top2_when_low_entropy(monkeypatch):
    class _S:
        decision_sampling_mode = "top_p_seeded"
        decision_confident_threshold = 0.6
        decision_entropy_threshold = 0.95

    monkeypatch.setattr("agents.decision.get_settings", lambda: _S())
    dist = {"rarely": 0.75, "1-2 per week": 0.15, "3-4 per week": 0.07, "daily": 0.02, "multiple per day": 0.01}
    sampled = sample_from_distribution(dist, rng=np.random.default_rng(5))
    assert sampled in {"rarely", "1-2 per week"}


def test_demographic_sanity_low_income_large_household_vs_high_income():
    personas = generate_population(1000, method="bayesian", seed=321)
    qm = QUESTION_MODELS["food_delivery_frequency"]
    index_map = {k: i for i, k in enumerate(qm.scale)}

    low_scores = []
    high_scores = []
    for p in personas:
        state = AgentState.from_persona(p)
        traits = personality_from_persona(p)
        ctx = DecisionContext(
            persona=p,
            traits=traits,
            perception=SimpleNamespace(raw_question="How often do you order delivery?"),
            friends_using=0.2,
            location_quality=0.5,
            memories=[],
            environment={"activation": 0.1},
        )
        dist = compute_distribution(qm, ctx, agent_state=state, persona=p, traits=traits, rng=np.random.default_rng(11))
        expected_idx = sum(index_map[k] * v for k, v in dist.items())
        if p.income in ("<10k", "10-25k") and p.household_size == "5+":
            low_scores.append(expected_idx)
        if p.income == "50k+" and p.household_size in ("1", "2"):
            high_scores.append(expected_idx)

    assert low_scores and high_scores
    assert (sum(low_scores) / len(low_scores)) <= (sum(high_scores) / len(high_scores))


def test_invariant_failure_returns_explicit_fallback(monkeypatch):
    persona, state, traits, qm, _ = _single_agent_context()
    from agents.decision import decide
    from agents.perception import perceive

    monkeypatch.setattr(
        "agents.decision._validate_distribution_invariants",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("forced_invariant_failure")),
    )
    p = perceive("How often do you order delivery?")
    dist, chosen = decide(
        p,
        persona,
        traits,
        memories=[],
        environment={},
        agent_state=state,
        rng=np.random.default_rng(123),
    )
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert chosen in dist

