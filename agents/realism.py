"""
Stratified Human Imperfection Engine
=====================================
Multi-layer realism system that injects realistic human imperfections at every
stage of the synthetic survey pipeline.  Pushes realism from ~9.0 → 9.6+.

Layer 1 — Conviction Profiles:   Shapes probability distributions to match real
          human certainty patterns (peaky, bimodal, diffuse, anchored).

Layer 2 — Cultural Behavior Priors:  Nationality × family_size → delivery/cooking
          behavior priors based on Dubai ethnographic patterns.

Layer 3 — Demographic Consistency:  Catches implausible persona–answer combos
          BEFORE they reach the LLM (e.g. family of 2, low income → "multiple/day").

Layer 4 — Response Texture:  Injects "messy human" patterns — vague answers,
          hedging, typos, incomplete thoughts — to break the clean-AI signal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from population.personas import Persona


# ═══════════════════════════════════════════════════════════════════════════
# Layer 0 — Habit Profile (Cross-Survey Behavioral Consistency)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HabitProfile:
    """Persistent behavioral tendencies derived deterministically from persona.

    These values are fixed per-agent and injected into every survey question
    so the same agent answers consistently across different surveys.
    """
    primary_service_tendency: float = 0.5
    alternative_tendency: float = 0.5
    budget_consciousness: float = 0.5
    health_strictness: float = 0.5
    tech_comfort: float = 0.5
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def delivery_tendency(self) -> float:
        """Backward-compat alias."""
        return self.primary_service_tendency

    @delivery_tendency.setter
    def delivery_tendency(self, v: float) -> None:
        self.primary_service_tendency = v

    @property
    def cooking_tendency(self) -> float:
        """Backward-compat alias."""
        return self.alternative_tendency

    @cooking_tendency.setter
    def cooking_tendency(self, v: float) -> None:
        self.alternative_tendency = v


def derive_habit_profile(persona: "Persona") -> HabitProfile:
    """Deterministically derive a HabitProfile from persona demographics + lifestyle.

    No randomness -- given the same persona, the same profile is produced,
    ensuring cross-survey consistency.
    """
    ls = persona.lifestyle

    primary = (
        0.3 * ls.primary_service_preference
        + 0.2 * ls.convenience_preference
        + 0.1 * ls.tech_adoption
        + 0.1 * (1.0 - ls.price_sensitivity)
        + 0.1 * ls.dining_out
        + 0.2 * (0.7 if persona.household_size == "1" else
                  0.5 if persona.household_size == "2" else
                  0.3 if persona.household_size == "3-4" else 0.15)
    )

    alternative = 1.0 - primary * 0.8

    budget_map = {"<10k": 0.85, "10-25k": 0.65, "25-50k": 0.40, "50k+": 0.20}
    budget_consciousness = (
        0.5 * budget_map.get(persona.income, 0.5)
        + 0.5 * ls.price_sensitivity
    )

    health_map = {
        "very health-conscious": 0.90, "fitness-focused": 0.85,
        "active": 0.65, "trying to be healthier": 0.55,
        "moderate": 0.40, "relaxed": 0.25, "don't think about it": 0.10,
    }
    health_strictness = health_map.get(
        getattr(persona.personal_anchors, "health_focus", "moderate"), 0.40
    )

    tech_comfort = (
        0.6 * ls.tech_adoption
        + 0.2 * ls.convenience_preference
        + 0.2 * (0.8 if persona.age in ("18-24", "25-34") else
                  0.5 if persona.age in ("35-44",) else 0.3)
    )

    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    return HabitProfile(
        primary_service_tendency=_clamp(primary),
        alternative_tendency=_clamp(alternative),
        budget_consciousness=_clamp(budget_consciousness),
        health_strictness=_clamp(health_strictness),
        tech_comfort=_clamp(tech_comfort),
    )


# Habit influence matrix: maps 5 habit dimensions to option-position bias.
# Each row = one habit dimension, each column position encodes how that
# dimension biases toward low-frequency (left) or high-frequency (right) options.
# Matrix is applied as: bias = habit_vector @ habit_matrix(n_options)
_HABIT_DIMENSION_DIRECTION: List[float] = [
    +1.0,   # primary_service_tendency: high -> more frequent
    -0.6,   # alternative_tendency:     high -> less frequent primary
    -0.4,   # budget_consciousness:     high -> less frequent (costly)
    -0.55,  # health_strictness:        high -> prefers alternative for control
    +0.2,   # tech_comfort:             high -> slight boost (app comfort)
]

_HABIT_INFLUENCE_STRENGTH = 0.22


def _build_habit_matrix(n_options: int) -> np.ndarray:
    """Build a (5, n_options) matrix projecting habit dimensions to option logits."""
    position = np.linspace(-1.0, 1.0, n_options)
    directions = np.array(_HABIT_DIMENSION_DIRECTION)
    return np.outer(directions, position)


def apply_habit_bias(
    raw_scores: List[float],
    scale: List[str],
    habit_profile: "HabitProfile",
) -> List[float]:
    """Apply habit influence via matrix multiplication: habit_vector @ habit_matrix.

    Produces a multi-dimensional bias that considers all 5 habit dimensions
    simultaneously rather than just delivery_tendency alone.
    """
    n = len(scale)
    habit_vec = np.array([
        habit_profile.primary_service_tendency,
        habit_profile.alternative_tendency,
        habit_profile.budget_consciousness,
        habit_profile.health_strictness,
        habit_profile.tech_comfort,
    ])
    habit_matrix = _build_habit_matrix(n)
    bias = habit_vec @ habit_matrix * _HABIT_INFLUENCE_STRENGTH
    return [r + float(b) for r, b in zip(raw_scores, bias)]


# EMA rate for habit updates after each answer (behavior inertia)
_HABIT_EMA_RATE = 0.1

def _get_answer_habit_scores() -> Dict[str, float]:
    """Load answer-to-habit scores from domain config, with fallback."""
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.answer_habit_scores:
            return cfg.answer_habit_scores
    except Exception:
        pass
    return {
        "rarely": 0.1,
        "1-2 per week": 0.35,
        "3-4 per week": 0.6,
        "daily": 0.8,
        "multiple per day": 0.95,
    }


def update_habits_after_answer(
    habit_profile: "HabitProfile",
    sampled_option: str,
    alpha: float = _HABIT_EMA_RATE,
) -> None:
    """EMA-update habit profile based on the agent's sampled answer.

    This creates behavioral inertia: an agent who answers frequently
    will gradually drift toward higher primary_service_tendency, making
    future high-frequency answers more likely.
    """
    scores = _get_answer_habit_scores()
    answer_score = scores.get(sampled_option)
    if answer_score is None:
        return

    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    habit_profile.primary_service_tendency = _clamp(
        (1.0 - alpha) * habit_profile.primary_service_tendency + alpha * answer_score
    )
    habit_profile.alternative_tendency = _clamp(
        (1.0 - alpha) * habit_profile.alternative_tendency + alpha * (1.0 - answer_score * 0.8)
    )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 — Conviction Profiles
# ═══════════════════════════════════════════════════════════════════════════

class ConvictionProfile(str, Enum):
    CERTAIN = "certain"          # Very sure — one option dominates (0.70+)
    LEANING = "leaning"          # Moderate certainty — top option 0.40–0.60
    DIFFUSE = "diffuse"          # Genuinely unsure — fairly flat
    BIMODAL = "bimodal"          # Torn between two options
    ANCHORED = "anchored"        # Strong prior from life circumstance


# Assignment weights by archetype — certain archetypes are more decisive
_CONVICTION_BY_ARCHETYPE: Dict[str, Dict[ConvictionProfile, float]] = {
    "busy_professional":  {ConvictionProfile.CERTAIN: 0.35, ConvictionProfile.LEANING: 0.30, ConvictionProfile.DIFFUSE: 0.10, ConvictionProfile.BIMODAL: 0.10, ConvictionProfile.ANCHORED: 0.15},
    "family_cook":        {ConvictionProfile.CERTAIN: 0.40, ConvictionProfile.LEANING: 0.25, ConvictionProfile.DIFFUSE: 0.05, ConvictionProfile.BIMODAL: 0.10, ConvictionProfile.ANCHORED: 0.20},
    "health_focused":     {ConvictionProfile.CERTAIN: 0.30, ConvictionProfile.LEANING: 0.30, ConvictionProfile.DIFFUSE: 0.10, ConvictionProfile.BIMODAL: 0.15, ConvictionProfile.ANCHORED: 0.15},
    "budget_conscious":   {ConvictionProfile.CERTAIN: 0.35, ConvictionProfile.LEANING: 0.25, ConvictionProfile.DIFFUSE: 0.10, ConvictionProfile.BIMODAL: 0.15, ConvictionProfile.ANCHORED: 0.15},
    "convenience_seeker": {ConvictionProfile.CERTAIN: 0.30, ConvictionProfile.LEANING: 0.35, ConvictionProfile.DIFFUSE: 0.10, ConvictionProfile.BIMODAL: 0.10, ConvictionProfile.ANCHORED: 0.15},
    "social_foodie":      {ConvictionProfile.CERTAIN: 0.20, ConvictionProfile.LEANING: 0.30, ConvictionProfile.DIFFUSE: 0.15, ConvictionProfile.BIMODAL: 0.20, ConvictionProfile.ANCHORED: 0.15},
    "young_explorer":     {ConvictionProfile.CERTAIN: 0.22, ConvictionProfile.LEANING: 0.30, ConvictionProfile.DIFFUSE: 0.18, ConvictionProfile.BIMODAL: 0.20, ConvictionProfile.ANCHORED: 0.10},
    "traditionalist":     {ConvictionProfile.CERTAIN: 0.45, ConvictionProfile.LEANING: 0.25, ConvictionProfile.DIFFUSE: 0.05, ConvictionProfile.BIMODAL: 0.05, ConvictionProfile.ANCHORED: 0.20},
    "default":            {ConvictionProfile.CERTAIN: 0.25, ConvictionProfile.LEANING: 0.33, ConvictionProfile.DIFFUSE: 0.12, ConvictionProfile.BIMODAL: 0.15, ConvictionProfile.ANCHORED: 0.15},
}


def assign_conviction_profile(
    persona: "Persona",
    rng: Optional[np.random.Generator] = None,
) -> ConvictionProfile:
    """Assign a conviction profile based on archetype and personality noise."""
    gen = rng or np.random.default_rng()
    archetype = getattr(persona.personal_anchors, "archetype", "default")
    weights = _CONVICTION_BY_ARCHETYPE.get(archetype, _CONVICTION_BY_ARCHETYPE["default"])
    profiles = list(weights.keys())
    probs = np.array([weights[p] for p in profiles])
    probs /= probs.sum()
    return profiles[int(gen.choice(len(profiles), p=probs))]


def apply_conviction_shaping(
    probs: List[float],
    profile: ConvictionProfile,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """Reshape a probability distribution to match the conviction profile.

    This is the key mechanism for producing extreme/bimodal/peaky distributions
    that real humans exhibit, instead of uniformly smooth softmax outputs.
    """
    gen = rng or np.random.default_rng()
    arr = np.array(probs, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return probs

    if profile == ConvictionProfile.CERTAIN:
        # Sharpen: raise to power 2.5–4.0, then renormalize
        power = gen.uniform(2.5, 4.0)
        arr = np.power(arr, power)

    elif profile == ConvictionProfile.LEANING:
        # Moderate sharpening: power 1.3–2.0
        power = gen.uniform(1.3, 2.0)
        arr = np.power(arr, power)

    elif profile == ConvictionProfile.DIFFUSE:
        # Flatten: raise to power 0.3–0.6 (makes distribution more uniform)
        power = gen.uniform(0.3, 0.6)
        arr = np.power(arr, power)

    elif profile == ConvictionProfile.BIMODAL:
        # Create two peaks: boost top-2 options, suppress the rest
        sorted_idx = np.argsort(arr)[::-1]
        top2 = sorted_idx[:2]
        boost = gen.uniform(1.5, 3.0)
        for idx in top2:
            arr[idx] *= boost
        suppress = gen.uniform(0.2, 0.5)
        for idx in sorted_idx[2:]:
            arr[idx] *= suppress

    elif profile == ConvictionProfile.ANCHORED:
        # One option gets extreme weight from life circumstance
        anchor_idx = int(np.argmax(arr))
        anchor_boost = gen.uniform(3.0, 6.0)
        arr[anchor_idx] *= anchor_boost

    # Renormalize
    total = arr.sum()
    if total > 0:
        arr = arr / total
    else:
        arr = np.ones(n) / n

    # Peakiness floor: no agent should produce a truly uniform distribution.
    # Power 2.0 at threshold 0.30 is aggressive enough to concentrate mass
    # even when the input is nearly flat (e.g. all options within 0.04 of each other).
    if arr.max() < 0.30:
        arr = np.power(arr, 2.0)
        arr = arr / arr.sum()

    return arr.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Cultural Behavior Priors
# ═══════════════════════════════════════════════════════════════════════════

def _load_cultural_priors() -> tuple:
    """Load cultural priors, family modifiers, and income modifiers from domain config."""
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        return cfg.cultural_priors, cfg.family_modifiers, cfg.income_modifiers
    except Exception:
        return {}, {}, {}


def get_cultural_prior(
    persona: "Persona",
    scale: Optional[List[str]] = None,
) -> Optional[Dict[str, float]]:
    """Compute a culturally-informed prior distribution.

    Combines: base_nationality_prior + family_modifier + income_modifier.
    Returns None if no cultural priors are configured or scale doesn't match.
    """
    cultural_priors, family_modifiers, income_modifiers = _load_cultural_priors()
    if not cultural_priors:
        return None

    ref_scale = set(next(iter(cultural_priors.values())).keys()) if cultural_priors else set()
    if scale is not None and ref_scale and set(scale) != ref_scale:
        return None

    nationality = persona.nationality
    fallback_key = "Other" if "Other" in cultural_priors else next(iter(cultural_priors), None)
    base = cultural_priors.get(nationality, cultural_priors.get(fallback_key, {}))
    if not base:
        return None
    prior = dict(base)

    family_mod = family_modifiers.get(persona.household_size, {})
    for k in prior:
        prior[k] += family_mod.get(k, 0.0)

    income_mod = income_modifiers.get(persona.income, {})
    for k in prior:
        prior[k] += income_mod.get(k, 0.0)

    for k in prior:
        prior[k] = max(0.01, prior[k])
    total = sum(prior.values())
    for k in prior:
        prior[k] /= total

    return prior


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — Demographic Consistency Validation
# ═══════════════════════════════════════════════════════════════════════════

def _load_implausible_combos() -> List[Tuple[dict, str, str]]:
    """Load implausible combos from domain config, with hardcoded fallback."""
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.implausible_combos:
            return [
                (item["filters"], item["option"], item["warning"])
                for item in cfg.implausible_combos
            ]
    except Exception:
        pass
    return []


def validate_demographic_plausibility(
    persona: "Persona",
    sampled_option: str,
) -> Optional[str]:
    """Return a warning string if the persona-answer combo is implausible.

    Checks both top-level persona attributes and personal_anchors fields.
    Rules are loaded from domain config.
    """
    combos = _load_implausible_combos()
    for filters, option, warning in combos:
        if sampled_option != option:
            continue
        match = True
        for key, val in filters.items():
            persona_val = getattr(persona, key, None)
            if persona_val is None:
                persona_val = getattr(persona.personal_anchors, key, None)
            if persona_val != val:
                match = False
                break
        if match:
            return warning
    return None


def suggest_plausible_resampling(
    persona: "Persona",
    dist: Dict[str, float],
    sampled_option: str,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, float], str]:
    """If the sampled option is implausible, resample from a corrected distribution.

    Reduces the probability of the implausible option and resamples.
    Returns the (possibly modified) distribution and new sampled option.
    """
    warning = validate_demographic_plausibility(persona, sampled_option)
    if warning is None:
        return dist, sampled_option

    gen = rng if rng is not None else np.random.default_rng()

    corrected = dict(dist)
    corrected[sampled_option] *= 0.15
    total = sum(corrected.values())
    for k in corrected:
        corrected[k] /= total

    options = list(corrected.keys())
    probs = [corrected[k] for k in options]
    new_sample = str(gen.choice(options, p=probs))
    return corrected, new_sample


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4 — Response Texture (Messy Human Patterns)
# ═══════════════════════════════════════════════════════════════════════════

def _load_vague_answers() -> Dict[str, List[str]]:
    """Load vague answer templates from domain config."""
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.vague_answers:
            return cfg.vague_answers
    except Exception:
        pass
    return {}


VAGUE_ANSWERS: Dict[str, List[str]] = {}


def _get_vague_answers() -> Dict[str, List[str]]:
    global VAGUE_ANSWERS
    if not VAGUE_ANSWERS:
        VAGUE_ANSWERS = _load_vague_answers()
    return VAGUE_ANSWERS

# Probability of using a vague/short answer instead of narrative
VAGUE_ANSWER_PROBABILITY = 0.30


def should_use_vague_answer(rng: Optional[random.Random] = None) -> bool:
    """Decide whether this agent should give a terse/vague answer."""
    r = rng or random.Random()
    return r.random() < VAGUE_ANSWER_PROBABILITY


def pick_vague_answer(
    sampled_option: str,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Return a short, messy human answer for the given option."""
    r = rng or random.Random()
    pool = _get_vague_answers().get(sampled_option)
    if not pool:
        return None
    return r.choice(pool)


# Hedging phrases that can be prepended to any answer
HEDGING_PHRASES: List[str] = [
    "Hmm, ", "I mean, ", "Uh, ", "Well, ", "I guess ",
    "Probably ", "Like, ", "So basically ", "I'd say ",
    "Honestly ", "Idk, ", "Umm ",
]

HEDGING_PROBABILITY = 0.12


def maybe_add_hedging(
    text: str,
    rng: Optional[random.Random] = None,
) -> str:
    """Occasionally prepend a hedging phrase for realism."""
    r = rng or random.Random()
    if r.random() < HEDGING_PROBABILITY:
        hedge = r.choice(HEDGING_PHRASES)
        # Don't double-hedge
        if not text.lower().startswith(tuple(h.lower().strip() for h in HEDGING_PHRASES)):
            return hedge + text[0].lower() + text[1:]
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Layer 5 — Fragment Transform (Post-LLM Truncation)
# ═══════════════════════════════════════════════════════════════════════════

import re as _re

_FRAGMENT_BASE_PROB = 0.25

_TRAILING_FILLERS = [
    " honestly", " tbh", " lol", " idk", " haha",
    " basically", " ya know", " I guess", " or something",
]

_SENTENCE_END = _re.compile(r'[.!?]')
_CLAUSE_BREAK = _re.compile(r'[,\-–—]')


def maybe_fragmentize(
    text: str,
    profile: "NarrativeStyleProfile",
    rng: Optional[random.Random] = None,
) -> str:
    """Occasionally truncate LLM output into a casual fragment.

    Profile-aware: agents with low grammar_quality / high slang_level
    fragment more often.  Skips text that is already short (< 30 chars).
    """
    if len(text) < 30:
        return text

    r = rng or random.Random()

    prob = _FRAGMENT_BASE_PROB
    if profile.grammar_quality < 0.40:
        prob += 0.15
    elif profile.grammar_quality > 0.75:
        prob -= 0.12
    if profile.slang_level > 0.50:
        prob += 0.10

    prob = max(0.0, min(prob, 0.60))

    if r.random() >= prob:
        return text

    strategy = r.choice(["first_sentence", "first_clause", "lowercase_strip", "filler"])

    if strategy == "first_sentence":
        m = _SENTENCE_END.search(text, 8)
        if m:
            fragment = text[: m.start()].strip()
            if len(fragment) >= 8:
                return fragment

    elif strategy == "first_clause":
        m = _CLAUSE_BREAK.search(text, 8)
        if m:
            fragment = text[: m.start()].strip()
            if len(fragment) >= 8:
                return fragment

    elif strategy == "lowercase_strip":
        m = _SENTENCE_END.search(text, 8)
        fragment = text[: m.start()].strip() if m else text.strip()
        fragment = fragment.rstrip(".!?,;:-–—")
        if fragment:
            return fragment[0].lower() + fragment[1:]

    elif strategy == "filler":
        m = _SENTENCE_END.search(text, 8)
        fragment = text[: m.start()].strip() if m else text.strip()
        fragment = fragment.rstrip(".!?,;:-–—")
        if fragment:
            return fragment + r.choice(_TRAILING_FILLERS)

    return text


# ═══════════════════════════════════════════════════════════════════════════
# Layer 6 — Polish Degradation (Strip LLM-ish Phrasing)
# ═══════════════════════════════════════════════════════════════════════════

_POLISH_PROB = 0.30

_POLISH_REPLACEMENTS: List[tuple] = [
    (_re.compile(r"\bI find myself\b", _re.I), ""),
    (_re.compile(r"\bI usually\b", _re.I), "I"),
    (_re.compile(r"^Sure[.,]\s*", _re.I), ""),
    (_re.compile(r"^Roughly\b", _re.I), "Like"),
    (_re.compile(r"^Honestly[,?]\s*", _re.I), ""),
    (_re.compile(r"\bI would say\b", _re.I), ""),
    (_re.compile(r"\bto be honest\b", _re.I), "tbh"),
    (_re.compile(r"\bI order food delivery\b", _re.I), "I order"),
    (_re.compile(r"\bfood delivery\b", _re.I), "delivery"),
    (_re.compile(r"\bquite frequently\b", _re.I), "a lot"),
    (_re.compile(r"\bfrequently\b", _re.I), "a lot"),
    (_re.compile(r"\bNot gonna lie\b", _re.I), "Ngl"),
    (_re.compile(r"\bI tend to\b", _re.I), "I"),
    (_re.compile(r"\bI'd have to say\b", _re.I), ""),
    (_re.compile(r"\bI have to say\b", _re.I), ""),
]


def degrade_polish(
    text: str,
    rng: Optional[random.Random] = None,
) -> str:
    """Occasionally strip LLM-polished phrasing to sound more human.

    Applies regex replacements that remove or casualize overly formal
    constructions that real survey respondents wouldn't use.
    """
    if len(text) < 15:
        return text

    r = rng or random.Random()
    if r.random() >= _POLISH_PROB:
        return text

    for pattern, replacement in _POLISH_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # Clean up double spaces and leading spaces from removals
    text = _re.sub(r"  +", " ", text).strip()

    # Fix broken capitalization after removal at start of string
    if text and text[0] == " ":
        text = text.lstrip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text if text else "Yeah."


if TYPE_CHECKING:
    from agents.narrative import NarrativeStyleProfile
