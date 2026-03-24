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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
from config.settings import get_settings
from core.rng import ensure_np_rng
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
    gen = ensure_np_rng(rng, key="decision_dirichlet")

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

    gen = ensure_np_rng(rng, key="decision_factor_perturb")
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
_SPIKE_PROB = 0.15
_SPIKE_BOOST = 0.4
_EPS = 1e-9

_FACTOR_TYPES: Dict[str, str] = {
    "income": "likelihood",
    "location": "likelihood",
    "behavioral": "likelihood",
    "personality": "modifier",
    "social": "modifier",
    "memory": "modifier",
    "belief": "modifier",
}


def _segment_prior_distribution(persona: Optional[Persona], scale: List[str]) -> Dict[str, float]:
    """Get segment-conditioned prior distribution if available for this scale."""
    if persona is None:
        return {}


def _normalize_dist(dist: Dict[str, float], scale: List[str]) -> Dict[str, float]:
    out = {k: max(0.0, float(dist.get(k, 0.0))) for k in scale}
    total = sum(out.values())
    if total <= 0:
        return {k: 1.0 / max(1, len(scale)) for k in scale}
    return {k: v / total for k, v in out.items()}


def _question_spec(question_model_key: str, question_text: str = "") -> Dict[str, Any]:
    spec: Dict[str, Any] = {}
    try:
        from config.generated_registry import get_generated_model_payload

        payload = get_generated_model_payload(question_model_key)
        if isinstance(payload, dict):
            spec.update(payload)
    except Exception:
        pass

    text = (question_text or "").lower()
    if "semantic_profile" not in spec:
        if question_model_key == "cost_of_living_satisfaction" or any(tok in text for tok in ("cost of living", "afford", "expense", "price")):
            spec["semantic_profile"] = "economic_pressure"
        elif question_model_key == "tech_adoption_likelihood" or any(tok in text for tok in ("app", "wallet", "platform", "technology", "digital service")):
            spec["semantic_profile"] = "lifestyle_frequency"
        elif any(tok in text for tok in ("exercise", "work out", "workout", "fitness", "physical activity", "gym")):
            spec["semantic_profile"] = "health_behavior"
        elif question_model_key == "policy_support" or any(tok in text for tok in ("support", "oppose", "policy", "law", "government")):
            spec["semantic_profile"] = "policy_opinion"
        elif any(tok in text for tok in ("trust", "confidence in")):
            spec["semantic_profile"] = "social_trust"
        elif any(tok in text for tok in ("safe", "safety", "crime", "walking alone")):
            spec["semantic_profile"] = "safety_perception"
        else:
            spec["semantic_profile"] = "generic_attitude"
    if "dominant_factors" not in spec:
        defaults = {
            "economic_pressure": ["income", "behavioral", "belief"],
            "health_behavior": ["behavioral", "personality", "belief"],
            "policy_opinion": ["belief", "social", "personality"],
            "social_trust": ["belief", "social", "location"],
            "safety_perception": ["location", "income", "belief", "behavioral"],
            "lifestyle_frequency": ["behavioral", "personality", "income"],
            "generic_attitude": ["behavioral", "belief", "personality"],
        }
        spec["dominant_factors"] = defaults.get(spec.get("semantic_profile", "generic_attitude"), defaults["generic_attitude"])
    if "narrative_guidance" not in spec:
        spec["narrative_guidance"] = {}
    return spec


def _scalar_to_distribution(value: float, scale: List[str], sharpness: float = 4.0) -> Dict[str, float]:
    n = len(scale)
    if n == 0:
        return {}
    pos = float(np.clip(value, 0.0, 1.0)) * max(1, n - 1)
    xs = np.arange(n, dtype=np.float64)
    logits = -sharpness * ((xs - pos) ** 2) / max(1.0, n - 1)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return {opt: float(p) for opt, p in zip(scale, probs)}


def _semantic_profile_score(profile: str, persona: Optional[Persona], context: DecisionContext) -> Optional[float]:
    if persona is None:
        return None
    pa = getattr(persona, "personal_anchors", None)
    lifestyle = getattr(persona, "lifestyle", None)
    beliefs = context.environment.get("beliefs") if context and context.environment is not None else None
    if profile == "economic_pressure":
        return 1.0 - _economic_pressure(persona)
    if profile == "health_behavior":
        health_focus = {
            "fitness-focused": 0.95,
            "very health-conscious": 0.90,
            "active": 0.75,
            "trying to be healthier": 0.62,
            "moderate": 0.50,
            "relaxed": 0.35,
            "don't think about it": 0.18,
        }.get(getattr(pa, "health_focus", ""), 0.5)
        hobby_bonus = 0.0
        hobby = str(getattr(pa, "hobby", "")).lower()
        if any(tok in hobby for tok in ("gym", "running", "swimming", "football", "cricket", "cycling", "walking", "surfing")):
            hobby_bonus = 0.10
        if any(tok in hobby for tok in ("netflix", "watching tv", "sleeping", "scrolling social media", "nothing really")):
            hobby_bonus = -0.10
        return float(max(0.0, min(1.0, health_focus + hobby_bonus)))
    if profile == "policy_opinion":
        if beliefs is not None:
            try:
                score = 0.6 * beliefs.get("government_trust", 0.5) + 0.4 * beliefs.get("environmental_concern", 0.5)
                return float(max(0.0, min(1.0, score)))
            except Exception:
                pass
        return 0.5
    if profile == "social_trust":
        if beliefs is not None:
            try:
                return float(max(0.0, min(1.0, beliefs.belief_score({"government_trust": 0.45, "innovation_curiosity": 0.05}))))
            except Exception:
                pass
        return 0.5
    if profile == "safety_perception":
        location_bias = {
            "Jumeirah": 0.74,
            "Downtown": 0.68,
            "Dubai Marina": 0.66,
            "JLT": 0.61,
            "Business Bay": 0.57,
            "Al Barsha": 0.54,
            "JVC": 0.52,
            "Others": 0.47,
            "Deira": 0.40,
            "Al Karama": 0.42,
        }
        base = float(location_bias.get(getattr(persona, "location", ""), max(0.0, min(1.0, context.location_quality))))
        commute = str(getattr(pa, "commute_method", "")).lower()
        if "walk" in commute:
            base -= 0.08
        elif "metro" in commute or "bus" in commute:
            base -= 0.04
        elif "car" in commute:
            base += 0.05
        age = str(getattr(persona, "age", ""))
        if age == "55+":
            base -= 0.06
        elif age == "18-24":
            base += 0.03
        if str(getattr(persona, "income", "")) == "<10k":
            base -= 0.06
        elif str(getattr(persona, "income", "")) == "50k+":
            base += 0.04
        if str(getattr(persona, "household_size", "")) == "5+":
            base -= 0.03
        if beliefs is not None:
            try:
                base += 0.06 * (beliefs.get("government_trust", 0.5) - 0.5)
            except Exception:
                pass
        return float(max(0.0, min(1.0, base)))
    if profile == "lifestyle_frequency":
        if lifestyle is None:
            return 0.5
        return float(max(0.0, min(1.0, 0.28 * lifestyle.convenience_preference + 0.18 * lifestyle.primary_service_preference + 0.20 * (1.0 - lifestyle.price_sensitivity) + 0.16 * lifestyle.dining_out + 0.18 * lifestyle.tech_adoption)))
    return None


def _apply_dominance_fusion(
    logits: np.ndarray,
    scale: List[str],
    factor_signals: List[Dict[str, float]],
    question_spec: Dict[str, Any],
    trace: Dict[str, Any],
) -> np.ndarray:
    if not factor_signals or not scale:
        return logits
    dominant_names = [str(name) for name in question_spec.get("dominant_factors", [])]
    prioritized = [item for item in factor_signals if item["factor"] in dominant_names]
    candidates = prioritized or factor_signals
    candidates = sorted(candidates, key=lambda item: item["importance"], reverse=True)
    if not candidates:
        return logits
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    second_importance = float(second["importance"]) if second else 0.0
    if best["importance"] < 0.05:
        return logits
    dominance_margin = float(best["importance"] - second_importance)
    if dominance_margin < 0.02 and abs(best["centered"]) < 0.12:
        return logits
    target_dist = _scalar_to_distribution(best["raw_scalar"], scale, sharpness=7.0 if len(scale) <= 5 else 5.0)
    vec = np.array([target_dist[k] for k in scale], dtype=np.float64)
    settings = get_settings()
    dom_scale = float(getattr(settings, "dominance_fusion_scale", 1.0))
    boost_pre_cap = min(1.35, 0.55 + 2.2 * float(best["importance"])) * dom_scale
    boost_cap = float(getattr(settings, "dominance_fusion_boost_cap", 0.52))
    boost_strength = min(boost_pre_cap, boost_cap)
    updated = logits + boost_strength * np.log(vec + _EPS)
    trace["dominance"] = {
        "factor": best["factor"],
        "raw_scalar": float(best["raw_scalar"]),
        "importance": float(best["importance"]),
        "margin": dominance_margin,
        "boost_strength_pre_cap": float(boost_pre_cap),
        "boost_strength": float(boost_strength),
        "dominance_fusion_scale": dom_scale,
        "dominance_fusion_boost_cap": boost_cap,
        "dominant_factors": dominant_names,
    }
    return updated


def _build_base_prior(
    question_model: QuestionModel,
    persona: Optional[Persona],
) -> Tuple[Dict[str, float], str]:
    n = len(question_model.scale)
    uniform = {opt: 1.0 / max(1, n) for opt in question_model.scale}
    if n == 0:
        return {}, "empty_scale"

    cultural = get_cultural_prior(persona, scale=question_model.scale) if persona is not None else None
    if cultural:
        return _normalize_dist(cultural, question_model.scale), "cultural_prior"

    segment = _segment_prior_distribution(persona, question_model.scale)
    if segment:
        return _normalize_dist(segment, question_model.scale), "segment_prior"

    reference = get_reference_distribution(question_model.name, question_model.scale)
    if reference:
        return _normalize_dist(reference, question_model.scale), "reference_prior"
    return uniform, "uniform_prior"


def _match_filters(persona: Persona, filters: Dict[str, Any]) -> bool:
    for key, expected in filters.items():
        value = getattr(persona, key, None)
        if value is None and hasattr(persona, "family"):
            value = getattr(persona.family, key, None)
        if value is None and hasattr(persona, "personal_anchors"):
            value = getattr(persona.personal_anchors, key, None)
        if value != expected:
            return False
    return True


def _severity_multiplier(severity: str, settings: Any) -> float:
    sev = (severity or "semi-hard").lower()
    if sev == "hard":
        return 0.0
    if sev == "soft":
        return float(getattr(settings, "soft_constraint_multiplier", 0.8))
    return 0.01


def _rule_matches_question(rule: Dict[str, Any], question_model_key: Optional[str]) -> bool:
    if not question_model_key:
        return True
    scoped = rule.get("question_model_key")
    if scoped is None:
        return True
    if isinstance(scoped, str):
        return scoped == question_model_key
    if isinstance(scoped, list):
        return question_model_key in scoped
    return True


def _load_constraint_rules(question_model_key: Optional[str]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    try:
        from config.domain import get_domain_config

        rules.extend(get_domain_config().implausible_combos)
    except Exception:
        pass
    try:
        from config.generated_registry import load_generated_registry

        reg = load_generated_registry()
        generated = reg.get("constraints", {})
        if question_model_key and isinstance(generated, dict):
            scoped = generated.get(question_model_key, [])
            if isinstance(scoped, list):
                rules.extend(scoped)
    except Exception:
        pass
    return rules


def _apply_constraints_log_space(
    logits: np.ndarray,
    scale: List[str],
    persona: Optional[Persona],
    trace: Dict[str, Any],
    question_model_key: Optional[str] = None,
) -> np.ndarray:
    if persona is None:
        return logits
    rules = _load_constraint_rules(question_model_key)
    settings = get_settings()
    fired: List[Dict[str, Any]] = []
    updated = logits.copy()
    for rule in rules:
        if not _rule_matches_question(rule, question_model_key):
            continue
        filters = rule.get("filters", {})
        option = rule.get("option")
        if not option or option not in scale:
            continue
        if not _match_filters(persona, filters):
            continue
        idx = scale.index(option)
        severity = rule.get("severity", "semi-hard")
        mult = _severity_multiplier(severity, settings)
        if mult <= 0.0:
            updated[idx] = -1e9
        else:
            updated[idx] = updated[idx] + float(np.log(mult + _EPS))
        fired.append({
            "stage": "log_space",
            "option": option,
            "severity": severity,
            "multiplier": mult,
            "filters": filters,
        })
    if fired:
        trace.setdefault("constraints_applied", []).extend(fired)
    return updated


def _apply_constraints_prob_space(
    dist: Dict[str, float],
    persona: Optional[Persona],
    trace: Dict[str, Any],
    question_model_key: Optional[str] = None,
) -> Dict[str, float]:
    if persona is None or not dist:
        return dist
    rules = _load_constraint_rules(question_model_key)
    settings = get_settings()
    out = dict(dist)
    fired: List[Dict[str, Any]] = []
    for rule in rules:
        if not _rule_matches_question(rule, question_model_key):
            continue
        filters = rule.get("filters", {})
        option = rule.get("option")
        if option not in out or not _match_filters(persona, filters):
            continue
        severity = rule.get("severity", "semi-hard")
        mult = _severity_multiplier(severity, settings)
        out[option] = out[option] * mult
        fired.append({
            "stage": "prob_space",
            "option": option,
            "severity": severity,
            "multiplier": mult,
            "filters": filters,
        })
    out = _normalize_dist(out, list(out.keys()))
    if fired:
        trace.setdefault("constraints_applied", []).extend(fired)
    return out


def _apply_factor_couplings(
    logits: np.ndarray,
    scale: List[str],
    persona: Optional[Persona],
    trace: Dict[str, Any],
) -> np.ndarray:
    if persona is None:
        return logits
    try:
        from config.domain import get_domain_config

        couplings = get_domain_config().factor_couplings
    except Exception:
        couplings = []
    updated = logits.copy()
    fired: List[Dict[str, Any]] = []
    for coupling in couplings:
        filters = coupling.get("filters", {})
        if filters and not _match_filters(persona, filters):
            continue
        effects = coupling.get("option_effects", {})
        strength = float(coupling.get("strength", 1.0))
        label = coupling.get("name", "coupling")
        for option, effect in effects.items():
            if option not in scale:
                continue
            idx = scale.index(option)
            delta = strength * float(effect)
            updated[idx] += delta
        if effects:
            fired.append({
                "name": label,
                "strength": strength,
                "option_effects": effects,
                "filters": filters,
            })
    if fired:
        trace["couplings"] = fired
    return updated


def _safe_calibration_blend(
    dist: Dict[str, float],
    reference: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    if not dist or not reference or alpha <= 0:
        return dist
    alpha = max(0.0, min(0.3, alpha))
    keys = sorted(set(dist.keys()) | set(reference.keys()))
    blended = {
        k: (1.0 - alpha) * float(dist.get(k, 0.0)) + alpha * float(reference.get(k, 0.0))
        for k in keys
    }
    for k, v in dist.items():
        if v > 0.6:
            blended[k] = max(blended[k], 0.9 * v)
    return _normalize_dist(blended, keys)


def _entropy(probs: List[float]) -> float:
    arr = np.array(probs, dtype=np.float64)
    arr = np.clip(arr, _EPS, 1.0)
    return float(-(arr * np.log(arr)).sum())


def _entropy_shape(dist: Dict[str, float], temperature: float) -> Dict[str, float]:
    if not dist:
        return dist
    keys = list(dist.keys())
    probs = np.array([dist[k] for k in keys], dtype=np.float64)
    logits = np.log(np.clip(probs, _EPS, 1.0)) / max(0.3, float(temperature))
    logits -= logits.max()
    shaped = np.exp(logits)
    shaped /= shaped.sum()
    return {k: float(v) for k, v in zip(keys, shaped)}


def _is_numeric_likert5(scale: List[str]) -> bool:
    return len(scale) == 5 and all(str(v).isdigit() for v in scale)


def _center_index(scale: List[str]) -> Optional[int]:
    if len(scale) != 5:
        return None
    return len(scale) // 2


def _economic_pressure(persona: Optional[Persona]) -> float:
    if persona is None:
        return 0.5
    income_score = {
        "<10k": 1.0,
        "10-25k": 0.75,
        "25-50k": 0.45,
        "50k+": 0.15,
    }.get(getattr(persona, "income", ""), 0.5)
    household_score = {
        "1": 0.10,
        "2": 0.25,
        "3-4": 0.60,
        "5+": 1.0,
    }.get(getattr(persona, "household_size", ""), 0.5)
    return float(max(0.0, min(1.0, 0.55 * income_score + 0.45 * household_score)))


def _reduce_middle_collapse(
    dist: Dict[str, float],
    scale: List[str],
    persona: Optional[Persona],
    settings: Any,
    trace: Dict[str, Any],
) -> Dict[str, float]:
    """Bounded redistribution from middle option for extreme center collapse."""
    center_idx = _center_index(scale)
    if not dist or center_idx is None:
        trace["anti_collapse"] = {
            "triggered": False,
            "skipped": True,
            "reason": "not_likert5",
        }
        return dist
    mid_key = scale[center_idx]
    mid_prob = float(dist.get(mid_key, 0.0))
    threshold = float(getattr(settings, "anti_collapse_middle_guard_threshold", 0.45))
    profile = str(trace.get("question_spec", {}).get("semantic_profile", "")).strip()
    if profile in {"safety_perception", "policy_opinion"}:
        threshold = min(threshold, 0.40)
    q_lower = str(trace.get("question_text", "") or "").lower()
    relocation_q = any(
        tok in q_lower
        for tok in ("relocate", "relocation", "move away", "move out", "leave dubai", "another city")
    )
    mid_lower = str(mid_key).strip().lower()
    neutral_label = mid_lower in {"neutral", "unsure", "undecided", "mixed", "depends"}
    exploring_middle = "undecided" in mid_lower or "exploring" in mid_lower or "options" in mid_lower
    # Relocation + genuinely exploratory middle option: allow a high middle mass;
    # do not force redistribution as aggressively.
    if exploring_middle and relocation_q:
        threshold = max(threshold, 0.55)
    elif neutral_label and not relocation_q:
        threshold = min(threshold, 0.40)
    elif neutral_label and relocation_q:
        threshold = max(threshold, 0.48)
    ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top_prob = ordered[0]
    second_prob = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    margin = float(top_prob - second_prob)
    trigger = (mid_prob >= threshold) or (top_key == mid_key and margin <= 0.08)
    if not trigger:
        return dist

    dominance_meta = trace.get("dominance", {}) if isinstance(trace, dict) else {}
    semantic_meta = trace.get("semantic_profile_signal", {}) if isinstance(trace, dict) else {}
    directional_score = dominance_meta.get("raw_scalar")
    if directional_score is None:
        directional_score = semantic_meta.get("score")
    if directional_score is None and persona is not None:
        directional_score = 1.0 - _economic_pressure(persona)
    directional_score = float(directional_score if directional_score is not None else 0.5)

    redistribute_cap = float(getattr(settings, "anti_collapse_max_redistribution", 0.16))
    if profile == "safety_perception":
        redistribute_cap = max(redistribute_cap, 0.22)

    fc = trace.get("factor_contributions") if isinstance(trace, dict) else None
    if isinstance(fc, list) and len(fc) >= 2:
        dead = sum(
            1
            for f in fc
            if abs(float(f.get("raw_factor_output", 0.5)) - 0.5) < 0.02
        )
        if dead >= len(fc):
            return dist
        if dead >= 3:
            redistribute_cap *= 0.72

    if exploring_middle and relocation_q:
        redistribute_cap = min(redistribute_cap, 0.14)
    budget = max(0.0, mid_prob - threshold)
    if top_key == mid_key and margin <= 0.08:
        budget += 0.06 + (0.08 - margin)
    if neutral_label and top_key == mid_key:
        budget += 0.04
    budget = min(redistribute_cap, budget)
    if budget <= 0:
        return dist

    updated = dict(dist)
    updated[mid_key] = max(0.0, updated[mid_key] - budget)
    if directional_score < 0.44:
        updated[scale[0]] = updated.get(scale[0], 0.0) + budget * 0.65
        updated[scale[1]] = updated.get(scale[1], 0.0) + budget * 0.35
        direction = "downshift"
    elif directional_score > 0.56:
        updated[scale[3]] = updated.get(scale[3], 0.0) + budget * 0.45
        updated[scale[4]] = updated.get(scale[4], 0.0) + budget * 0.55
        direction = "upshift"
    else:
        updated[scale[1]] = updated.get(scale[1], 0.0) + budget * 0.5
        updated[scale[3]] = updated.get(scale[3], 0.0) + budget * 0.5
        direction = "balanced_shift"

    trace["anti_collapse"] = {
        "triggered": True,
        "middle_key": mid_key,
        "middle_before": mid_prob,
        "top_key": top_key,
        "margin": margin,
        "redistributed_mass": float(budget),
        "directional_score": directional_score,
        "direction": direction,
    }
    return _normalize_dist(updated, scale)


def _apply_cost_of_living_demographic_tilt(
    dist: Dict[str, float],
    question_model_key: str,
    scale: List[str],
    persona: Optional[Persona],
    trace: Dict[str, Any],
) -> Dict[str, float]:
    if question_model_key != "cost_of_living_satisfaction" or not _is_numeric_likert5(scale):
        return dist
    pressure = _economic_pressure(persona)
    if pressure >= 0.68:
        mult = np.array([1.55, 1.30, 0.82, 0.62, 0.48], dtype=np.float64)
        mode = "high_pressure"
    elif pressure <= 0.32:
        mult = np.array([0.65, 0.82, 0.95, 1.28, 1.45], dtype=np.float64)
        mode = "low_pressure"
    else:
        return dist
    arr = np.array([dist.get(k, 0.0) for k in scale], dtype=np.float64)
    arr = np.clip(arr * mult, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return dist
    arr = arr / total
    trace["cost_of_living_tilt"] = {
        "mode": mode,
        "pressure": float(pressure),
    }
    return {k: float(v) for k, v in zip(scale, arr)}


def _validate_distribution_invariants(dist: Dict[str, float], scale: List[str]) -> None:
    if not dist:
        raise ValueError("distribution_empty")
    total = float(sum(dist.values()))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"distribution_not_normalized:{total}")
    if any((v < 0.0 or v > 1.0) for v in dist.values()):
        raise ValueError("distribution_out_of_range")
    if set(dist.keys()) != set(scale):
        raise ValueError("distribution_key_mismatch")


# ── Generic distribution generator ──────────────────────────────────────

def compute_distribution(
    question_model: QuestionModel,
    context: DecisionContext,
    agent_state: Optional["AgentState"] = None,
    persona: Optional[Persona] = None,
    traits: Optional[PersonalityTraits] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Compute LPFG distribution with explicit priors, constraints, and trace."""
    settings = get_settings()
    gen = ensure_np_rng(rng, key=f"decision_compute:{question_model.name}")
    noise_budget = max(0.0, min(0.15, settings.decision_noise_budget))
    if not question_model.scale:
        return {}
    n = len(question_model.scale)
    trace: Dict[str, Any] = {
        "question_model": question_model.name,
        "factor_contributions": [],
        "stages": {},
    }
    question_text = getattr(getattr(context, "perception", None), "raw_question", "") or ""
    trace["question_text"] = question_text
    question_spec = _question_spec(question_model.name, question_text)
    trace["question_spec"] = {
        "semantic_profile": question_spec.get("semantic_profile"),
        "dominant_factors": list(question_spec.get("dominant_factors", [])),
        "narrative_guidance": dict(question_spec.get("narrative_guidance", {}) or {}),
    }

    base_prior, prior_source = _build_base_prior(question_model, persona)
    logits = np.log(np.array([base_prior[k] for k in question_model.scale], dtype=np.float64) + _EPS)
    trace["base_prior_source"] = prior_source
    trace["base_prior"] = base_prior

    graph = get_or_build_graph(
        question_model.name,
        lambda: build_factor_graph(question_model),
    )
    uniform = np.ones(n, dtype=np.float64) / n
    factor_signals: List[Dict[str, float]] = []
    for fn, weight in graph._factors:
        name = fn.__name__.replace("_factor", "")
        raw_scalar = float(max(0.0, min(1.0, fn(context))))
        factor_dist = _scalar_to_distribution(raw_scalar, question_model.scale)
        vec = np.array([factor_dist[k] for k in question_model.scale], dtype=np.float64)
        factor_type = _FACTOR_TYPES.get(name, "modifier")
        if factor_type == "likelihood":
            delta = float(weight) * np.log(vec + _EPS)
        else:
            delta = float(weight) * (vec - uniform)
        logits = logits + delta
        importance = float(abs(raw_scalar - 0.5) * abs(weight))
        trace["factor_contributions"].append({
            "factor": name,
            "type": factor_type,
            "weight": float(weight),
            "raw_factor_output": raw_scalar,
            "importance": importance,
            "applied_delta_norm": float(np.linalg.norm(delta)),
        })
        factor_signals.append(
            {
                "factor": name,
                "raw_scalar": raw_scalar,
                "weight": float(weight),
                "centered": float(raw_scalar - 0.5),
                "importance": importance,
            }
        )

    semantic_score = _semantic_profile_score(str(question_spec.get("semantic_profile", "")), persona, context)
    if semantic_score is not None:
        semantic_dist = _scalar_to_distribution(semantic_score, question_model.scale, sharpness=6.0 if len(question_model.scale) <= 5 else 4.5)
        semantic_vec = np.array([semantic_dist[k] for k in question_model.scale], dtype=np.float64)
        semantic_importance = min(0.45, 0.35 * abs(semantic_score - 0.5) * 2.0)
        logits = logits + semantic_importance * np.log(semantic_vec + _EPS)
        factor_signals.append(
            {
                "factor": "semantic_profile",
                "raw_scalar": float(semantic_score),
                "weight": float(semantic_importance),
                "centered": float(semantic_score - 0.5),
                "importance": float(abs(semantic_score - 0.5) * semantic_importance),
            }
        )
        trace["semantic_profile_signal"] = {
            "profile": question_spec.get("semantic_profile"),
            "score": float(semantic_score),
            "weight": float(semantic_importance),
        }

    logits = _apply_dominance_fusion(logits, question_model.scale, factor_signals, question_spec, trace)

    # Habit profile as modifier
    if agent_state is not None and agent_state.habit_profile is not None:
        habit_logits = apply_habit_bias(logits.tolist(), question_model.scale, agent_state.habit_profile)
        logits = np.array(habit_logits, dtype=np.float64)
        trace["habit_bias_applied"] = True

    # Conditional couplings
    logits = _apply_factor_couplings(logits, question_model.scale, persona, trace)

    # Per-option noise (bounded by budget)
    option_noise_std = _RAW_SCORE_NOISE_STD * noise_budget
    option_noise = gen.normal(0, option_noise_std, size=n)
    logits = logits + option_noise
    trace["option_noise_std"] = float(option_noise_std)

    # Conviction spike (bounded by budget)
    spike_prob = _SPIKE_PROB * noise_budget
    if gen.random() < spike_prob:
        spike_idx = int(gen.integers(0, n))
        logits[spike_idx] = logits[spike_idx] + _SPIKE_BOOST
        trace["conviction_spike"] = {"index": spike_idx, "boost": float(_SPIKE_BOOST)}

    # Constraints stage A (log-space)
    trace["stages"]["pre_constraints"] = {
        "distribution": _normalize_dist(
            {k: float(v) for k, v in zip(question_model.scale, _softmax(logits.tolist(), temperature=max(0.4, question_model.temperature)))},
            question_model.scale,
        )
    }
    logits = _apply_constraints_log_space(
        logits,
        question_model.scale,
        persona,
        trace,
        question_model_key=question_model.name,
    )

    # Softmax with per-agent temperature
    temp = question_model.temperature
    if persona is not None and traits is not None:
        temp = _agent_softmax_temperature(temp, persona, traits)

    # Activation modulation: high emotional activation lowers temperature
    # (sharper, more extreme choices) -- range [0.6*temp, temp].
    _act = context.environment.get("activation", 0.0) if context else 0.0
    temp *= max(0.6, 1.0 - 0.4 * float(_act))
    probs = _softmax(logits.tolist(), temp)

    # Dirichlet noise
    if noise_budget > 0:
        probs = _inject_dirichlet_noise(probs, rng=rng)

    # Conviction shaping
    if persona is not None:
        profile = assign_conviction_profile(persona, rng=gen)
        probs = apply_conviction_shaping(probs, profile, rng=gen)
        trace["conviction_profile"] = profile.value

    dist = dict(zip(question_model.scale, probs))
    dist = _normalize_dist(dist, question_model.scale)
    trace["stages"]["post_constraints"] = {"distribution": {k: float(v) for k, v in dist.items()}}

    # Constraints stage B (prob-space)
    dist = _apply_constraints_prob_space(dist, persona, trace, question_model_key=question_model.name)
    dist = _apply_cost_of_living_demographic_tilt(dist, question_model.name, question_model.scale, persona, trace)

    # Cognitive dissonance
    if agent_state is not None:
        from agents.dissonance import apply_cognitive_dissonance, compute_consistency_score

        consistency = compute_consistency_score(agent_state, question_model)
        trace["consistency_score"] = float(consistency)
        dist = apply_cognitive_dissonance(
            dist, consistency, question_model.scale, agent_state=agent_state,
        )

    # Cross-question memory bias
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
    trace["stages"]["post_biases"] = {"distribution": {k: float(v) for k, v in dist.items()}}

    # Stage 14: competition-based neutral penalty (plan: output_quality_refined)
    penalty = getattr(settings, "neutral_penalty_when_competing_above", 0.0)
    if penalty > 0 and dist:
        _NEUTRAL_LABELS = frozenset({
            "neutral", "no opinion", "undecided", "neither",
            "not sure", "mixed", "depends",
        })
        neutral_key = None
        for k in dist:
            if k.lower().strip() in _NEUTRAL_LABELS:
                neutral_key = k
                break
        if neutral_key is not None:
            items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            if len(items) >= 2:
                second_best = items[1][1]
                if second_best > 0.4:
                    dist = dict(dist)
                    dist[neutral_key] = dist[neutral_key] * (1.0 - penalty)
                    total = sum(dist.values())
                    if total > 0:
                        dist = {k: v / total for k, v in dist.items()}

    # Single mild shaping stage before sampling.
    e_before = _entropy(list(dist.values()))
    shaping_temperature = float(getattr(settings, "decision_entropy_temperature", 0.75))
    if prior_source in {"reference_prior", "cultural_prior"} or question_spec.get("source") == "adaptive":
        shaping_temperature = max(0.93, shaping_temperature)
    dist = _entropy_shape(dist, temperature=shaping_temperature)
    e_after = _entropy(list(dist.values()))
    trace["entropy"] = {"before": float(e_before), "after": float(e_after)}

    # Calibration is corrective, not dominant.
    try:
        ref = get_reference_distribution(question_model.name, question_model.scale)
    except Exception:
        ref = {}
    alpha = float(getattr(settings, "calibration_blend_alpha", 0.2))
    if prior_source in {"reference_prior", "cultural_prior"} or question_spec.get("source") == "adaptive":
        alpha = 0.0
        trace["calibration_skipped_reason"] = prior_source if prior_source in {"reference_prior", "cultural_prior"} else "adaptive_spec"
    dist = _safe_calibration_blend(
        dist,
        _normalize_dist(ref, question_model.scale) if ref else {},
        alpha=alpha,
    )
    trace["stages"]["post_calibration"] = {"distribution": {k: float(v) for k, v in dist.items()}}

    # Bounded middle-collapse guard for numeric Likert scales.
    dist = _reduce_middle_collapse(dist, question_model.scale, persona, settings, trace)

    # Final constraints recheck + invariants
    dist = _apply_constraints_prob_space(dist, persona, trace, question_model_key=question_model.name)
    dist = _normalize_dist(dist, question_model.scale)

    floor_alpha = float(getattr(settings, "categorical_entropy_floor_alpha", 0.0) or 0.0)
    if floor_alpha > 0 and len(question_model.scale) != 5:
        u = 1.0 / max(1, len(question_model.scale))
        dist = {
            k: (1.0 - floor_alpha) * float(dist.get(k, 0.0)) + floor_alpha * u
            for k in question_model.scale
        }
        dist = _normalize_dist(dist, question_model.scale)
        trace["categorical_entropy_floor"] = {"alpha": floor_alpha}

    try:
        _validate_distribution_invariants(dist, question_model.scale)
    except Exception as exc:
        trace["invariant_failure"] = str(exc)
        trace["fallback_used"] = "uniform_after_invariant_failure"
        dist = _normalize_dist(
            {k: 1.0 for k in question_model.scale},
            question_model.scale,
        )

    trace["final_distribution"] = {k: float(v) for k, v in dist.items()}
    if context.environment is not None:
        context.environment["__decision_trace"] = trace
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
    settings = get_settings()
    gen = ensure_np_rng(rng, key="decision_sample")

    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    if not items:
        return ""
    if settings.decision_sampling_mode == "deterministic_argmax":
        return str(items[0][0])
    if (
        settings.decision_sampling_mode == "argmax_if_confident"
        and items[0][1] >= settings.decision_confident_threshold
    ):
        return str(items[0][0])
    top_prob = items[0][1]
    probs_only = [p for _, p in items]
    n = len(probs_only)
    ent = _entropy(probs_only)
    norm_ent = ent / np.log(n) if n > 1 else 0.0
    if norm_ent <= settings.decision_entropy_threshold and len(items) >= 2:
        top2 = list(items[:2])
        t2_opts, t2_probs = zip(*top2)
        t2_p = np.array(t2_probs, dtype=np.float64)
        t2_p = t2_p / t2_p.sum()
        return str(gen.choice(t2_opts, p=t2_p))

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

    try:
        dist = compute_distribution(
            question_model, context,
            agent_state=agent_state,
            persona=persona,
            traits=traits,
            rng=rng,
        )
    except Exception as exc:
        dist = _normalize_dist({k: 1.0 for k in question_model.scale}, question_model.scale)
        env["__decision_trace"] = {
            "question_model": question_model.name,
            "invariant_failure": str(exc),
            "fallback_used": "uniform_after_compute_exception",
            "final_distribution": dist,
        }
    if environment is not None and "__decision_trace" in env:
        environment["__decision_trace"] = env["__decision_trace"]
    chosen = sample_from_distribution(dist, rng=rng)
    if chosen and dist.get(chosen, 0.0) <= 0.0:
        ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        chosen = ordered[0][0] if ordered else chosen
        env.setdefault("__decision_trace", {}).setdefault("post_sampling_guard", {})
        env["__decision_trace"]["post_sampling_guard"]["hard_constraint_violation_avoided"] = True
    if context.environment is not None and "__decision_trace" in context.environment:
        ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        context.environment["__decision_trace"]["sampling"] = {
            "mode": get_settings().decision_sampling_mode,
            "chosen": chosen,
            "top_option": ordered[0][0] if ordered else "",
            "top_probability": float(ordered[0][1]) if ordered else 0.0,
        }
        if environment is not None:
            environment["__decision_trace"] = context.environment["__decision_trace"]

    # Demographic plausibility gate: catch implausible persona-answer combos
    pre_resample_choice = chosen
    pre_resample_dist = dict(dist)
    dist, chosen = suggest_plausible_resampling(persona, dist, chosen, rng=rng)
    if context.environment is not None and "__decision_trace" in context.environment:
        trace = context.environment["__decision_trace"]
        if chosen != pre_resample_choice:
            trace["demographic_plausibility_resample"] = {
                "original_choice": pre_resample_choice,
                "resampled_choice": chosen,
                "original_probability": float(pre_resample_dist.get(pre_resample_choice, 0.0)),
                "resampled_probability": float(dist.get(chosen, 0.0)),
            }
        if environment is not None:
            environment["__decision_trace"] = trace

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
