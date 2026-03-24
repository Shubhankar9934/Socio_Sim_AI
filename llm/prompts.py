"""
Prompt templates: agent reasoning with narrative diversity, persona compression,
LLM-as-judge, and per-agent temperature jitter.
"""

import random
import re
from typing import Any, Dict, List, Optional

from agents.narrative import (
    NarrativeStyleProfile,
    build_style_instruction,
    contains_duration_anti_pattern,
    format_avoid_phrases_line,
    format_voice_instruction_line,
    is_banned_pattern,
    pick_length_from_profile,
    pick_narrative_style,
    pick_opening,
    pick_opening_deduplicated,
    pick_persona_anchor,
    pick_response_length,
    pick_sentence_structure,
    pick_style_from_profile,
    pick_tone,
    pick_tone_from_profile,
    validate_narrative_consistency,
    vague_answer_probability_for_profile,
)
from agents.realism import (
    _fragment_looks_dangling,
    degrade_polish,
    inject_thinking_markers,
    maybe_add_hedging,
    maybe_add_micro_contradiction,
    maybe_add_redundancy,
    maybe_fragmentize,
    pick_vague_answer,
    should_use_vague_answer,
    trim_weak_terminal_suffix,
    validate_demographic_plausibility,
)
from population.personas import Persona
from core.rng import ensure_py_rng
from agents.response_contract import compute_confidence_band, tone_for_confidence_band
from agents.intent_router import build_turn_understanding_rules
from agents.context_relevance import (
    build_relevance_from_turn_understanding,
    filter_memories_for_topic,
    sample_response_shape,
)
from config.settings import get_settings as _get_settings


# ---------------------------------------------------------------------------
# Pool of system-prompt variants — randomly selected per agent to break the
# single-template signal that makes synthetic text detectable.
# ---------------------------------------------------------------------------

def _load_system_prompts() -> List[str]:
    """Load system prompts from domain config with {city_name} templating."""
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.system_prompts:
            return [p.format(city_name=cfg.city_name) for p in cfg.system_prompts]
    except Exception:
        pass
    return [
        "You simulate a survey respondent. Answer briefly and realistically as the persona. "
        "Do not break character. Give a short natural answer (1-3 sentences). "
        "NEVER start with 'With my', 'Since I', 'As someone who', 'As a', or 'Being a'.",
    ]


_SYSTEM_PROMPTS: List[str] = _load_system_prompts()

_SYSTEM_PROMPT_CASUAL_FILLER_MARKERS = (
    "say 'like'",
    "say \"like\"",
    "you know naturally",
    "think out loud. say",
    "'like', 'i mean', 'you know'",
)


def _system_prompt_mandates_casual_fillers(prompt: str) -> bool:
    t = (prompt or "").lower()
    return any(m in t for m in _SYSTEM_PROMPT_CASUAL_FILLER_MARKERS)


# Short register-specific tags appended to the base system prompt for lexical diversity.
_VOICE_REGISTER_FRAGMENTS: Dict[str, List[str]] = {
    "analytical": [
        "Prefer clear, economical wording; skip stock filler phrases.",
        "Answer like someone who thinks before speaking — no performative hedging.",
    ],
    "conversational": [
        "Sound like a normal chat — relaxed, but still answer the question.",
        "Keep it human and informal without rambling.",
    ],
    "blunt": [
        "Be brief and plain-spoken; do not pad with disclaimers.",
        "Get to the point; avoid 'honestly / I mean / kind of' openers.",
    ],
    "rambling": [
        "You may take a slightly winding path as long as you still answer clearly.",
        "Natural spoken rhythm is fine; don't sound like a template.",
    ],
}


def _pick_system_prompt(
    rng: Optional[random.Random] = None,
    voice_register: str = "conversational",
    *,
    strict_demographic_voice: bool = False,
) -> str:
    r = rng or random
    pool: List[str] = list(_SYSTEM_PROMPTS)
    if strict_demographic_voice and pool:
        filtered = [p for p in pool if not _system_prompt_mandates_casual_fillers(p)]
        if filtered:
            pool = filtered
    base = r.choice(pool)
    reg = (voice_register or "conversational").strip() or "conversational"
    frags = _VOICE_REGISTER_FRAGMENTS.get(reg) or _VOICE_REGISTER_FRAGMENTS["conversational"]
    if frags and r.random() < 0.88:
        base = f"{base} {r.choice(frags)}"
    return base


# ---------------------------------------------------------------------------
# Universal scale-type detection — options alone determine scale type
# ---------------------------------------------------------------------------

def infer_scale_type(options: List[str]) -> str:
    """Infer scale type from answer options. Works for any survey domain."""
    if not options:
        return "open_text"
    if all(o.isdigit() for o in options):
        return "numeric"
    freq_terms = {"never", "rarely", "sometimes", "often", "daily", "weekly"}
    if any(o.lower() in freq_terms for o in options):
        return "frequency"
    # Also check frequency-style phrases (e.g. "1-2 per week", "multiple per day")
    freq_phrases = {"1-2 per week", "3-4 per week", "multiple per day", "1-2 per month", "2-3 per week"}
    if any(o.lower() in freq_phrases for o in options):
        return "frequency"
    if len(options) == 2:
        return "categorical"
    return "likert"


def _extract_numeric_legend_from_question(question: str) -> Dict[str, str]:
    """Parse inline numeric legends like '1 = support; 5 = oppose'."""
    pairs = re.findall(
        r"(\d{1,2})\s*[:=\-]\s*(.+?)(?=(?:\s+\d{1,2}\s*[:=\-])|[;,\n\.]|$)",
        question or "",
        flags=re.IGNORECASE,
    )
    legend: Dict[str, str] = {}
    for key, label in pairs:
        cleaned = (label or "").strip().strip('"').strip("'")
        if cleaned:
            legend[key] = cleaned
    return legend


def _resolve_numeric_label_map(
    option_labels: Optional[List[str]],
    scale_options: List[str],
    question: str,
) -> Dict[str, str]:
    """Resolve numeric option -> meaning mapping from options or question text."""
    if not scale_options or not all(str(o).isdigit() for o in scale_options):
        return {}

    mapping: Dict[str, str] = {}
    labels = [str(o).strip() for o in (option_labels or []) if str(o).strip()]

    if labels:
        if len(labels) == len(scale_options) and not any(
            re.match(r"^\d{1,2}\s*[:=\-]", lbl) for lbl in labels
        ):
            for opt, lbl in zip(scale_options, labels):
                mapping[str(opt)] = lbl
        else:
            for lbl in labels:
                m = re.match(r"^(\d{1,2})\s*[:=\-]\s*(.+)$", lbl)
                if not m:
                    continue
                key = m.group(1).strip()
                text = m.group(2).strip().strip('"').strip("'")
                if key in scale_options and text:
                    mapping[key] = text

    if not mapping:
        mapping = _extract_numeric_legend_from_question(question)

    return {k: v for k, v in mapping.items() if k in scale_options and v}


def _format_numeric_legend(label_map: Dict[str, str]) -> str:
    if not label_map:
        return ""
    ordered = sorted(label_map.keys(), key=lambda x: int(x))
    return "; ".join(f"{k} = {label_map[k]}" for k in ordered)


# ---------------------------------------------------------------------------
# Topic-aware persona anchor injection — only when question is lifestyle-related
# ---------------------------------------------------------------------------

def _load_lifestyle_keywords() -> List[str]:
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.lifestyle_keywords:
            return cfg.lifestyle_keywords
    except Exception:
        pass
    return ["food", "diet", "exercise", "shopping", "travel", "media"]


LIFESTYLE_KEYWORDS: List[str] = _load_lifestyle_keywords()

ANCHOR_EXCLUDE_KEYWORDS: List[str] = [
    "network international", "payment provider", "bank", "fintech",
    "transaction", "payfast", "pay fast",
]


def allow_persona_anchor(question: str) -> bool:
    """Return True only when the question topic warrants lifestyle anchors."""
    q = question.lower()
    if any(k in q for k in ANCHOR_EXCLUDE_KEYWORDS):
        return False
    return any(k in q for k in LIFESTYLE_KEYWORDS)


# ---------------------------------------------------------------------------
# Frequency interpretation map for stronger consistency instructions
# ---------------------------------------------------------------------------

def _load_frequency_interpretation() -> Dict[str, str]:
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.frequency_interpretation:
            action = "use this service"
            return {k: v.format(action=action) for k, v in cfg.frequency_interpretation.items()}
    except Exception:
        pass
    return {}


_FREQUENCY_INTERPRETATION: Dict[str, str] = _load_frequency_interpretation()

# ---------------------------------------------------------------------------
# Instruction-block variants for the user-prompt — rotated to break template
# ---------------------------------------------------------------------------

_INSTRUCTION_VARIANTS: List[str] = [
    (
        "Talk about this like you'd tell a friend. Don't organize your thoughts -- just say what comes to mind.\n"
        "You selected \"{sampled_option}\" ({sampled_prob_pct}) — that means {interpretation}.\n"
        "Start with some context from your life, then work your way to the answer. Repeat yourself if that's natural.\n"
        "It's fine to trail off or change direction mid-sentence."
    ),
    (
        "Say what comes to mind first. Don't plan your answer.\n"
        "Your answer is \"{sampled_option}\" ({sampled_prob_pct}) — {interpretation}.\n"
        "Mention something real from your day or your routine. Be messy, be you."
    ),
    (
        "Answer like you're voice-noting a friend about this.\n"
        "You chose \"{sampled_option}\" ({sampled_prob_pct}) — meaning {interpretation}. Stick to that.\n"
        "Include a real detail from your life. It's OK to be imperfect."
    ),
    (
        "Think out loud. Start with a feeling or situation, then get to your point.\n"
        "\"{sampled_option}\" ({sampled_prob_pct}) is your answer — {interpretation}.\n"
        "Don't try to sound balanced or smart. Just talk."
    ),
]

_MICRO_INSTRUCTION_VARIANTS: List[str] = [
    (
        "Super short answer — like texting.\n"
        "Your answer is \"{sampled_option}\" ({sampled_prob_pct}) — {interpretation}. "
        "Just say it quick."
    ),
    (
        "Under 5 words. No need to explain.\n"
        "You chose \"{sampled_option}\" ({sampled_prob_pct}). Say it your way."
    ),
]

# ---------------------------------------------------------------------------
# Universal scale-type instruction templates — no domain assumptions
# ---------------------------------------------------------------------------

_NUMERIC_INSTRUCTION_VARIANTS: List[str] = [
    (
        "Your answer must include the score: {sampled_option}.\n"
        "Briefly explain why you chose that score.\n"
        "Do not introduce unrelated lifestyle topics unless they are relevant.\n"
        "{semantic_guardrail}"
    ),
    (
        "You selected {sampled_option}. State that score and explain in 1-2 short sentences.\n"
        "Keep your explanation focused on the question topic.\n"
        "{semantic_guardrail}"
    ),
]

_NUMERIC_MICRO_VARIANTS: List[str] = [
    (
        "Your answer must include the score: {sampled_option}. "
        "One short sentence is enough.\n"
        "{semantic_guardrail}"
    ),
]

# Hidden-state: stance-only, no numbers in answer. CRITICAL RULE first (D.5).
_CRITICAL_RULE_NO_NUMBERS: str = (
    "CRITICAL RULE:\n"
    "- Do NOT include any number (1, 2, 3, 4, 5).\n"
    "- Do NOT mention 'score', 'rating', or 'scale'.\n"
    "- If you violate this, the answer is invalid.\n\n"
)

_NUMERIC_HIDDEN_STATE_VARIANTS: List[str] = [
    (
        _CRITICAL_RULE_NO_NUMBERS
        + "Your true stance is exactly: **{mapped_scale_meaning}**.\n"
        "Write a natural, 1-2 sentence response explaining why you feel that way. "
        "Use your demographics, income, and lifestyle in your explanation. "
        "Speak like a human talking to a friend or leaving a casual comment online."
    ),
    (
        _CRITICAL_RULE_NO_NUMBERS
        + "Your stance is: **{mapped_scale_meaning}**.\n"
        "In 1-2 sentences, explain your view naturally. Speak naturally as this person."
    ),
]

_NUMERIC_HIDDEN_STATE_MICRO_VARIANTS: List[str] = [
    (
        _CRITICAL_RULE_NO_NUMBERS
        + "Your stance is: **{mapped_scale_meaning}**.\n"
        "Reply in one short, natural sentence."
    ),
]

_LIKERT_INSTRUCTION_VARIANTS: List[str] = [
    (
        _CRITICAL_RULE_NO_NUMBERS
        + 'Your stance is: "{sampled_option}".\n'
        "Write 1-2 sentences explaining why, in natural language."
    ),
    (
        _CRITICAL_RULE_NO_NUMBERS
        + 'You chose "{sampled_option}". Briefly explain why in natural language.'
    ),
]

_FREQUENCY_INSTRUCTION_VARIANTS: List[str] = [
    (
        'Your answer means: "{sampled_option}". '
        "That means {interpretation}.\n"
        "Describe briefly how often this happens in your routine."
    ),
    (
        'You selected "{sampled_option}" ({sampled_prob_pct}) — {interpretation}. '
        "Stick to that. Do NOT describe a different frequency.\n"
        "Output only the answer."
    ),
]

_CATEGORICAL_INSTRUCTION_VARIANTS: List[str] = [
    (
        'Your answer is "{sampled_option}".\n'
        "Explain briefly why."
    ),
    (
        'You chose "{sampled_option}". Give a short justification.'
    ),
]

_OPEN_TEXT_INSTRUCTION_VARIANTS: List[str] = [
    (
        "Answer the question naturally as this person.\n"
        "Do not provide a score.\n"
        "Do not invent a scale.\n"
        "Write 1–3 sentences."
    ),
    (
        "Respond in your own words as this person.\n"
        "No ratings, no numbers. Just a brief natural answer in 1–3 sentences."
    ),
]

_DURATION_INSTRUCTION_VARIANTS: List[str] = [
    (
        "Answer how long this has been true. Your answer should be a duration like "
        "'2 years', 'around 5 years', 'about a decade'. Briefly mention how it started or why. "
        "Do NOT provide a score or rating."
    ),
    (
        "Respond with a time span — e.g. '3 years', 'maybe 5 years', 'close to a decade'. "
        "No scores or ratings. Add a short context if natural."
    ),
    (
        "Give a duration answer — e.g. 'About 5 years now', 'Roughly 4 years', 'Maybe 3 years'. "
        "Do not invent a scale or provide a numeric score."
    ),
]


def _is_duration_question(question: str) -> bool:
    """Detect tenure/duration questions for open-ended time-span answers."""
    q = question.lower()
    return any(
        p in q
        for p in ("how long", "how many years", "for how long", "since when")
    )


# ---------------------------------------------------------------------------
# Archetype-specific narrative guidance — creates natural behavioral clusters
# ---------------------------------------------------------------------------

def _load_archetype_hints() -> Dict[str, str]:
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.archetype_hints:
            return cfg.archetype_hints
    except Exception:
        pass
    return {"default": ""}


ARCHETYPE_HINTS: Dict[str, str] = _load_archetype_hints()

def _load_cultural_hints() -> Dict[str, str]:
    try:
        from config.domain import get_domain_config
        cfg = get_domain_config()
        if cfg.cultural_hints:
            return cfg.cultural_hints
    except Exception:
        pass
    return {}


CULTURAL_BEHAVIOR_HINTS: Dict[str, str] = _load_cultural_hints()


def _stance_category(option: str) -> str:
    """Map sampled_option to stance category for semantic content enforcement.
    Returns: strong_support | support | neutral | oppose | strong_oppose
    """
    o = option.lower().strip()
    if "strong" in o and ("support" in o or "agree" in o or "favor" in o):
        return "strong_support"
    if "strong" in o and ("oppose" in o or "disagree" in o or "against" in o):
        return "strong_oppose"
    if o in ("neutral", "no opinion", "undecided", "neither", "not sure", "mixed", "depends"):
        return "neutral"
    if "support" in o or "agree" in o or "favor" in o:
        return "support"
    if "oppose" in o or "disagree" in o or "against" in o:
        return "oppose"
    return "neutral"


def _violates_strong_stance(text: str) -> bool:
    """True if text contains hedging/contradiction phrases that violate strong stance."""
    t = text.lower()
    banned = ["but", "however", "on the other hand", "on one hand", "depends", "mixed feelings"]
    return any(phrase in t for phrase in banned)


def _text_expresses_stance(text: str, stance_cat: str) -> bool:
    """True if text contains language that expresses the given stance (for decision-text consistency)."""
    t = text.lower()
    if stance_cat == "strong_support":
        return any(
            w in t for w in (
                "support", "agree", "favor", "back", "smart move", "right move",
                "great", "absolutely", "fully support", "strongly support", "all for it",
            )
        )
    if stance_cat == "strong_oppose":
        return any(
            w in t for w in (
                "oppose", "against", "disagree", "ridiculous", "bad idea", "wrong",
                "terrible", "absolutely not", "strongly oppose", "no way",
            )
        )
    if stance_cat == "support":
        return any(w in t for w in ("support", "agree", "for it", "back it"))
    if stance_cat == "oppose":
        return any(w in t for w in ("oppose", "against", "disagree", "not for"))
    return True  # neutral or unknown: allow


def _income_suggests_budget_pressure(income: Any) -> bool:
    s = str(income or "").strip().lower()
    if not s:
        return False
    if s.startswith("<"):
        return True
    if "<10k" in s or "under 10" in s or "below 10" in s:
        return True
    if re.search(r"\b9\s*k\b", s):
        return True
    return False


def _strict_demographic_voice_cohort(persona: Persona, rhetorical_habit: str) -> bool:
    habit = (rhetorical_habit or "direct").strip().lower()
    if habit in {"rambling", "storytelling", "verbose", "narrative", "emotional_lead"}:
        return False
    age_s = str(getattr(persona, "age", "") or "")
    m = re.search(r"\b(\d{2})\b", age_s)
    if m and int(m.group(1)) >= 50:
        return True
    if _income_suggests_budget_pressure(getattr(persona, "income", None)):
        return True
    return False


def _demographic_voice_instructions(persona: Persona, rhetorical_habit: str) -> str:
    """Tighten filler and framing from age/income without overriding explicit rambling styles."""
    habit = (rhetorical_habit or "direct").strip().lower()
    loose = habit in {"rambling", "storytelling", "verbose", "narrative", "emotional_lead"}
    parts: List[str] = []
    age_s = str(getattr(persona, "age", "") or "")
    m = re.search(r"\b(\d{2})\b", age_s)
    age_floor = int(m.group(1)) if m else 0
    if age_floor >= 50 and not loose:
        parts.append(
            "Speak with measured practicality. Do NOT start sentences with: Well, / Yep, / Yeah, / "
            'Honestly, / Like, (filler) / You know,. '
            'Strictly avoid discourse fillers as openers or crutches: "honestly", "I mean", '
            '"you know", and "like" (filler, not "I like food").'
        )
    inc = str(getattr(persona, "income", "") or "").strip()
    if _income_suggests_budget_pressure(inc):
        parts.append(
            "Your household is budget-constrained: focus on costs and tradeoffs; keep sentences short "
            "and direct. Avoid luxury or hobby-upscale framing unless it clearly fits."
        )
    if not parts:
        return ""
    return "\nVOICE (demographics):\n" + " ".join(parts)


def _persona_context(persona: Persona) -> dict:
    """Build a dict of persona fields for template filling."""
    pa = persona.personal_anchors
    return {
        "hobby": pa.hobby,
        "cuisine_preference": pa.cuisine_preference,
        "diet": pa.diet,
        "health_focus": pa.health_focus,
        "work_schedule": pa.work_schedule,
        "commute_method": pa.commute_method,
        "typical_dinner_time": pa.typical_dinner_time,
        "location": persona.location,
        "income": persona.income,
        "household_size": persona.household_size,
    }


def build_agent_prompt(
    persona: Persona,
    question: str,
    sampled_option: str,
    distribution: Dict[str, float],
    memories: List[str],
    rng: Optional[random.Random] = None,
    *,
    consistency_warning: Optional[str] = None,
    used_openings: Optional[set] = None,
    structured_context: Optional[Dict[str, Any]] = None,
    tone_override: Optional[str] = None,
    simulation_context: Optional[Dict[str, Any]] = None,
    option_labels: Optional[List[str]] = None,
    response_contract: Optional[Dict[str, Any]] = None,
    turn_understanding: Optional[Dict[str, Any]] = None,
) -> str:
    r = ensure_py_rng(rng, key=f"prompt:{persona.agent_id}:{question}:{sampled_option}")
    pa = persona.personal_anchors
    options = list(distribution.keys())
    scale_type = str((turn_understanding or {}).get("scale_type") or infer_scale_type(options))
    contract_mode = (response_contract or {}).get("expression_mode", "")
    if contract_mode == "open_expression":
        scale_type = "open_text"
    # For empty options (open-ended), check if it's a duration question
    if not options and scale_type == "open_text" and _is_duration_question(question):
        scale_type = "duration"
    eff_understanding = turn_understanding or build_turn_understanding_rules(question)
    imode_for_rel = str(
        (response_contract or {}).get("interaction_mode")
        or eff_understanding.get("interaction_mode")
        or "survey"
    )
    rel_policy = build_relevance_from_turn_understanding(
        question, eff_understanding, interaction_mode=imode_for_rel
    )
    allow_anchors = rel_policy.include_lifestyle_anchors

    try:
        from config.domain import get_domain_config
        _currency = get_domain_config().currency
    except Exception:
        _currency = "USD"
    persona_block = ""
    if rel_policy.include_core_demographics:
        persona_block = (
            f"Age group: {persona.age}. Nationality: {persona.nationality}. Location: {persona.location}.\n"
            f"Income band: {_currency} {persona.income}/month. Occupation: {persona.occupation}. "
            f"Household size: {persona.household_size}."
        )
    if rel_policy.include_family and persona.family.spouse:
        persona_block += f" Spouse: yes, children: {persona.family.children}."
    if rel_policy.include_mobility:
        persona_block += f"\nCar: {'yes' if persona.mobility.car else 'no'}, metro: {persona.mobility.metro_usage}."

    if rel_policy.include_biography and getattr(persona, "life_path", None) and persona.life_path.biography:
        persona_block += f"\nBackground: {persona.life_path.biography}"

    # Append lifestyle fields only when relevance policy allows
    if rel_policy.include_lifestyle_anchors:
        persona_block += (
            f"\nCuisine preference: {pa.cuisine_preference}. Diet: {pa.diet}. Hobby: {pa.hobby}.\n"
            f"Work schedule: {pa.work_schedule}. Dinner time: {pa.typical_dinner_time}. "
            f"Commute: {pa.commute_method}. Health focus: {pa.health_focus}."
        )

    # Archetype and cultural hints only when lifestyle tier is active
    archetype_block = ""
    cultural_block = ""
    if rel_policy.include_archetype_cultural:
        archetype = pa.archetype if hasattr(pa, "archetype") else "default"
        archetype_hint = ARCHETYPE_HINTS.get(archetype, "")
        if archetype_hint:
            archetype_block = f"\nARCHETYPE MINDSET:\n{archetype_hint}\n"
        cultural_hint = CULTURAL_BEHAVIOR_HINTS.get(persona.nationality, "")
        if cultural_hint:
            cultural_block = f"\nCULTURAL CONTEXT:\n{cultural_hint}\n"

    if rel_policy.include_behavior_floats:
        behavior_desc = (
            f"convenience={persona.lifestyle.convenience_preference:.2f}, "
            f"service_pref={persona.lifestyle.primary_service_preference:.2f}, "
            f"price_sensitivity={persona.lifestyle.price_sensitivity:.2f}."
        )
        if scale_type not in ("open_text", "duration"):
            behavior_desc += f" Sampled response: \"{sampled_option}\"."
    else:
        behavior_desc = "Behavior preference detail omitted for this question type."

    _topic = str(eff_understanding.get("topic") or "")
    memories_use = filter_memories_for_topic(memories, _topic, max_items=5)
    memory_block = (
        "No relevant memories."
        if not memories_use
        else "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories_use[:5])
    )

    ctx = _persona_context(persona)

    # Use persistent style profile from persona (with ~20% random deviation)
    ns = persona.personal_anchors.narrative_style
    _profile = NarrativeStyleProfile(
        verbosity=ns.verbosity,
        preferred_tone=ns.preferred_tone,
        preferred_style=ns.preferred_style,
        slang_level=ns.slang_level,
        grammar_quality=ns.grammar_quality,
        voice_register=getattr(ns, "voice_register", "conversational") or "conversational",
        rhetorical_habit=getattr(ns, "rhetorical_habit", "direct") or "direct",
        avoid_phrases=tuple(getattr(ns, "avoid_phrases", []) or ()),
    )
    style = pick_style_from_profile(_profile, rng=rng)
    structure = pick_sentence_structure(rng=rng)
    opening = pick_opening_deduplicated(
        ctx,
        used_openings=used_openings,
        rng=rng,
        profile=_profile,
        income_band=getattr(persona, "income", None),
    )
    anchor_name, anchor_value = pick_persona_anchor(ctx, rng=rng)
    length = pick_length_from_profile(_profile, rng=rng)

    # Response compression curve: as turns increase, shift toward shorter answers
    _turn_count = 0
    _fatigue = 0.0
    _emotional_state = "neutral"
    _decision_latency = "normal"
    if response_contract:
        _turn_count = int(response_contract.get("_turn_count", 0) or 0)
        _fatigue = float(response_contract.get("_fatigue", 0.0) or 0.0)
        _emotional_state = str(response_contract.get("_emotional_state", "neutral") or "neutral")
        _decision_latency = str(response_contract.get("decision_latency", "normal") or "normal")
    _settings = _get_settings()
    if getattr(_settings, "enable_compression_curve", True) and _turn_count > 2:
        import math
        verbosity_levels = ["long", "medium", "short", "micro"]
        current_idx = verbosity_levels.index(length) if length in verbosity_levels else 1
        compression = 1.0 - math.exp(-_turn_count / 6.0)
        shift = int(compression * 2)
        new_idx = min(len(verbosity_levels) - 1, current_idx + shift)
        length = verbosity_levels[new_idx]
    if _fatigue > 0.5:
        if length == "long":
            length = "medium"
        elif length == "medium" and _fatigue > 0.7:
            length = "short"

    tone = tone_override if tone_override else pick_tone_from_profile(_profile, rng=rng)
    if _emotional_state == "annoyed" and not tone_override:
        tone = r.choice(["terse", "blunt", "skeptical"])
    elif _emotional_state == "enthusiastic" and not tone_override:
        tone = r.choice(["casual", "emotional_practical"])
    elif _emotional_state == "bored" and not tone_override:
        tone = r.choice(["lazy", "distracted", "terse"])

    # Decision latency modulates response style
    if _decision_latency == "instant":
        if length in ("long", "medium"):
            length = "short"
    elif _decision_latency == "deliberate":
        if length == "micro":
            length = "short"

    qmk_for_shape = str(eff_understanding.get("question_model_key_candidate") or "")
    if qmk_for_shape and r.random() < 0.72:
        sampled_len = sample_response_shape(qmk_for_shape, imode_for_rel, r)
        if sampled_len in ("micro", "short", "medium", "long"):
            length = sampled_len

    include_anchor = allow_anchors and length != "micro" and r.random() < 0.55
    style_instruction = build_style_instruction(
        style, structure, opening, anchor_name, anchor_value,
        length=length, tone=tone, include_anchor=include_anchor,
        voice_line=format_voice_instruction_line(_profile),
        avoid_line=format_avoid_phrases_line(_profile),
    )
    if tone_override and scale_type not in ("open_text", "duration"):
        style_instruction += "\nTone affects wording only, not the answer content. Do not change the selected answer."

    _demo_voice = _demographic_voice_instructions(persona, _profile.rhetorical_habit)
    if _demo_voice:
        style_instruction += _demo_voice
    style_instruction += (
        "\nVOICE LIMIT: Use at most one conversational hedge in your whole answer "
        '(e.g. "I guess", "maybe", "kind of", "honestly", "I mean", "you know").'
    )

    # Scale-type driven instruction selection
    interpretation = _FREQUENCY_INTERPRETATION.get(sampled_option, sampled_option)
    sampled_prob = distribution.get(sampled_option, 0.0)

    if scale_type == "open_text":
        instruction = r.choice(_OPEN_TEXT_INSTRUCTION_VARIANTS)
        if response_contract:
            latent_stance = str(response_contract.get("latent_stance", "")).strip()
            conf = str(response_contract.get("confidence_band", "")).strip()
            if latent_stance:
                instruction += (
                    f"\nInternal stance anchor: {latent_stance}."
                    " Express this naturally in words only."
                    " Do NOT mention ratings, scales, or numeric scores."
                )
            if conf == "low":
                instruction += (
                    "\nUse mild uncertainty language since your confidence is low, "
                    "but express it in at most one short phrase (consistent with the one-hedge cap)."
                )
            elif conf == "high":
                instruction += "\nSpeak clearly and decisively, but still naturally."
    elif scale_type == "duration":
        instruction = r.choice(_DURATION_INSTRUCTION_VARIANTS)
    elif scale_type == "numeric":
        numeric_label_map = _resolve_numeric_label_map(option_labels, options, question)
        sampled_label = numeric_label_map.get(sampled_option, "")
        if sampled_label:
            variants = (
                _NUMERIC_HIDDEN_STATE_MICRO_VARIANTS if length == "micro"
                else _NUMERIC_HIDDEN_STATE_VARIANTS
            )
            instruction = r.choice(variants).format(mapped_scale_meaning=sampled_label)
        else:
            semantic_guardrail = ""
            variants = _NUMERIC_MICRO_VARIANTS if length == "micro" else _NUMERIC_INSTRUCTION_VARIANTS
            instruction = r.choice(variants).format(
                sampled_option=sampled_option,
                semantic_guardrail=semantic_guardrail,
            )
    elif scale_type == "likert":
        variants = _LIKERT_INSTRUCTION_VARIANTS
        instruction = r.choice(variants).format(sampled_option=sampled_option)
    elif scale_type == "frequency":
        sorted_dist = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        top_option, top_prob = sorted_dist[0]
        variants = _MICRO_INSTRUCTION_VARIANTS if length == "micro" else _FREQUENCY_INSTRUCTION_VARIANTS
        instruction = r.choice(variants).format(
            sampled_option=sampled_option,
            interpretation=interpretation,
            top_option=top_option,
            top_prob_pct=f"{top_prob:.0%}",
            sampled_prob_pct=f"{sampled_prob:.0%}",
        )
    else:  # categorical
        variants = _CATEGORICAL_INSTRUCTION_VARIANTS
        instruction = r.choice(variants).format(sampled_option=sampled_option)

    if response_contract:
        belief_stmts = response_contract.get("belief_statements") or []
        if rel_policy.include_beliefs and belief_stmts:
            instruction += "\nYOUR BELIEFS:\n" + "\n".join(f"- {s}" for s in belief_stmts) + "\n"
        psummary = str(response_contract.get("personality_summary", "")).strip()
        if rel_policy.include_personality_summary and psummary:
            instruction += f"\nPERSONALITY: {psummary}\n"
        tradeoff = str(response_contract.get("tradeoff_guidance", "")).strip()
        if rel_policy.include_tradeoff and tradeoff:
            instruction += f"\n{tradeoff}\n"

        interaction_mode = str(response_contract.get("interaction_mode", "")).strip()
        dominant_factor = str(response_contract.get("dominant_factor", "")).strip()
        runner_up = str(response_contract.get("runner_up_option", "")).strip()
        narrative_guidance = str(response_contract.get("narrative_guidance", "")).strip()
        if interaction_mode == "conversation":
            instruction += (
                "\nThis is regular conversation, not a scored survey turn."
                " Answer naturally and briefly, like a real person."
                " Do NOT invent options, ratings, or survey framing."
            )
        elif interaction_mode == "qualitative_interview":
            instruction += (
                "\nThis is a qualitative interview turn."
                " Answer in first person with grounded details, routines, examples, or feelings when relevant."
                " Do NOT invent ratings, scales, or option labels unless explicitly asked."
            )
        if rel_policy.include_dominant_factor and dominant_factor:
            instruction += (
                f"\nYour answer should sound like it is mainly driven by this factor: {dominant_factor}."
                " Do not explain your answer as generic uncertainty if this factor is strong."
            )
        if rel_policy.include_runner_up and runner_up and runner_up != sampled_option:
            instruction += (
                f"\nThe closest alternative was \"{runner_up}\", but your final answer is still \"{sampled_option}\"."
                " Mention tension only if confidence is low or medium."
            )
        if rel_policy.include_narrative_guidance and narrative_guidance:
            instruction += f"\nNarrative stance hint: {narrative_guidance}"

    # Stance confidence -> HARD constraints + semantic alignment (plan: output_quality_refined)
    stance_confidence = 0.5
    if scale_type not in ("open_text", "duration") and distribution:
        prob = distribution.get(sampled_option)
        if prob is None:
            prob = max(distribution.values(), default=0.5) if distribution else 0.5
        stance_confidence = float(prob)
        if stance_confidence >= 0.6:
            instruction += (
                "\nYour decision is strong. Speak decisively: Do NOT hedge and Do NOT present both sides. "
                "State a clear, single-sided opinion. Do NOT use 'on one hand', 'it depends', or similar phrases. "
                "Do NOT contradict your stance. Do NOT include opposing arguments if your stance is strong. "
                "If your stance is strong, express it with conviction and emotion."
            )
        elif stance_confidence >= 0.4:
            instruction += (
                "\nYour decision is moderate. You can slightly contradict yourself -- real people do. "
                "Start with one thought, then partially walk it back or add a 'but' -- this is natural."
            )
        else:
            instruction += "\nYour decision is uncertain; use tentative language."

        # Semantic content enforcement (redundant for compliance)
        option_for_stance = sampled_option
        if scale_type == "numeric":
            _numeric_map = _resolve_numeric_label_map(option_labels, options, question)
            if _numeric_map.get(sampled_option):
                option_for_stance = _numeric_map.get(sampled_option, sampled_option)
        stance_cat = _stance_category(option_for_stance)
        if stance_cat == "strong_support":
            instruction += (
                "\nYour stance is: STRONGLY SUPPORT. You MUST clearly express strong support. "
                "Do NOT express neutrality or mixed feelings. Do NOT weaken your stance. Express it with conviction and emotion."
            )
        elif stance_cat == "strong_oppose":
            instruction += (
                "\nYour stance is: STRONGLY OPPOSE. You MUST clearly express strong opposition. "
                "Do NOT express neutrality or mixed feelings. Do NOT weaken your stance. Express it with conviction and emotion."
            )
        elif stance_cat == "support":
            instruction += "\nYou MUST express support clearly (do not hedge)."
        elif stance_cat == "oppose":
            instruction += "\nYou MUST express opposition clearly (do not hedge)."
        elif stance_cat == "neutral":
            instruction += "\nYou may express uncertainty or mixed feelings."

    # Global: anti-perfection + anti-policy-analyst + context-first + emotion-logic
    _anti_perfection = (
        "\nYour answer should NOT sound like it was written by an AI. "
        "Real people don't speak in perfect sentences. They trail off, repeat themselves, "
        "mix feelings with facts, and sometimes slightly contradict themselves. Do that."
    )
    instruction += _anti_perfection

    _bflags = _get_settings()
    if getattr(_bflags, "enable_context_first_instruction", True) and scale_type not in ("open_text", "duration") and r.random() < 0.40:
        instruction += (
            "\nBefore giving your answer, briefly set the scene -- "
            "mention what's happening in your life, your routine, or a recent experience. THEN give your opinion."
        )
    if getattr(_bflags, "enable_emotion_logic_mixing", True) and scale_type not in ("open_text", "duration") and r.random() < 0.30:
        instruction += (
            "\nStart with how this makes you FEEL, then explain why practically. "
            "Example: 'Honestly it frustrates me because the prices keep going up and my salary hasn't changed.'"
        )

    if scale_type not in ("open_text", "duration"):
        instruction += (
            "\nDo NOT try to sound balanced or politically neutral. Respond as a real person with a clear opinion. "
            "Your response MUST contain a complete opinion and be at least 8 words. Do not output only a filler or opening phrase. "
            "Do NOT overanalyze. Respond quickly and instinctively like a real person. "
            "Do NOT justify your opinion like an expert. Speak casually."
        )
        # Randomized voice for population diversity
        _voice_variants = [
            "Respond like a casual comment online.",
            "Respond like you're talking to a friend.",
            "Respond naturally in your own voice.",
        ]
        instruction += "\n" + r.choice(_voice_variants)

    warn_block = ""
    if consistency_warning:
        warn_block = f"\nWARNING: {consistency_warning}\n"

    # Demographic plausibility warning (skip for open_text/duration — no sampled option)
    if scale_type not in ("open_text", "duration"):
        demo_warning = validate_demographic_plausibility(persona, sampled_option)
        if demo_warning:
            warn_block = f"\nWARNING: {demo_warning}. Adjust your answer to be realistic.\n" + warn_block

    # Writing quality instruction from persistent style profile
    writing_quality_block = ""
    if _profile.slang_level > 0.55:
        writing_quality_block += "Use casual slang, abbreviations, and informal language. "
    elif _profile.slang_level < 0.20:
        writing_quality_block += "Use proper, clean language — no slang. "
    if _profile.grammar_quality < 0.35:
        writing_quality_block += "Write with imperfect grammar — fragments, run-ons, missing punctuation are OK. "
    elif _profile.grammar_quality > 0.75:
        writing_quality_block += "Use complete, well-formed sentences. "

    # Structured context from multi-question survey state
    agent_state_block = ""
    if structured_context:
        summary = structured_context.get("recent_answers_summary", "")
        ctx_lines = []
        for k, v in structured_context.items():
            if k == "recent_answers_summary" or v is None:
                continue
            ctx_lines.append(f"  {k}: {v}")
        if ctx_lines or summary:
            agent_state_block = "\nAGENT STATE (from prior survey answers):\n"
            if summary:
                agent_state_block += f"  Summary: {summary}\n"
            agent_state_block += "\n".join(ctx_lines) + "\n"

    # Simulation dynamics block (activation, media, life events)
    sim_block = ""
    if simulation_context:
        parts = []
        act_level = simulation_context.get("activation")
        if act_level is not None and float(act_level) > 0.2:
            if float(act_level) > 0.7:
                parts.append("You are currently emotionally activated — feeling strongly about recent events.")
            elif float(act_level) > 0.4:
                parts.append("Recent events have caught your attention and you have moderate emotional engagement.")
        recent_media = simulation_context.get("recent_media_frames", [])
        if recent_media:
            frames_str = "; ".join(str(f) for f in recent_media[:3])
            parts.append(f"Recent news/media you noticed: {frames_str}")
        recent_events = simulation_context.get("recent_life_events", [])
        if recent_events:
            events_str = ", ".join(str(e) for e in recent_events[:3])
            parts.append(f"Recent life changes: {events_str}")
        sim_day = simulation_context.get("day")
        if sim_day is not None:
            parts.append(f"Current simulation day: {sim_day}")
        if parts:
            sim_block = "\nSIMULATION CONTEXT:\n" + "\n".join(f"- {p}" for p in parts) + "\n"

    return f"""PERSONA:
{persona_block}
{archetype_block}{cultural_block}
BEHAVIOR MODEL:
{behavior_desc}
{agent_state_block}{sim_block}
MEMORY:
{memory_block}

QUESTION:
{question}

NARRATIVE STYLE:
{style_instruction}
{writing_quality_block}
{warn_block}
INSTRUCTION:
{instruction}"""


# ---------------------------------------------------------------------------
# Per-agent temperature jitter
# ---------------------------------------------------------------------------

def _agent_temperature(
    base_temp: float,
    rng: Optional[random.Random] = None,
    personality: Optional[Any] = None,
) -> float:
    """Add uniform jitter plus openness/conscientiousness tilt (clamped)."""
    r = rng or random
    jitter = (r.random() - 0.5) * 0.30  # +/- 0.15
    trait_adj = 0.0
    if personality is not None:
        o = float(getattr(personality, "openness_to_experience", 0.5) or 0.5)
        c = float(getattr(personality, "conscientiousness", 0.5) or 0.5)
        trait_adj = (o - 0.5) * 0.38 - (c - 0.5) * 0.18
    return max(0.3, min(1.5, base_temp + jitter + trait_adj))


# ---------------------------------------------------------------------------
# LLM-as-judge evaluation prompt
# ---------------------------------------------------------------------------

def _build_judge_system() -> str:
    try:
        from config.domain import get_domain_config
        city = get_domain_config().city_name
    except Exception:
        city = "a simulated city"
    return (
        f"You evaluate survey responses from a simulated {city} resident. "
        "Score from 1 to 5 (integer only) for each criterion. "
        "Be strict but fair. Respond with valid JSON only."
    )


JUDGE_SYSTEM = _build_judge_system()


def _deterministic_alignment_repair(
    sampled_option: str,
    scale: List[str],
    option_label_map: Optional[Dict[str, str]] = None,
) -> str:
    """Build a deterministic fallback sentence anchored to sampled option."""
    if not scale:
        return "It depends on the situation for me."

    scale_type = infer_scale_type(scale)
    label_map = option_label_map or {}
    if scale_type == "frequency":
        return f"I would say {sampled_option}; that best matches my routine."
    if scale_type == "numeric":
        mapped = label_map.get(sampled_option)
        if mapped:
            return f"My stance is {mapped}, and that is what I mean here."
        return f"I would choose {sampled_option} on this scale."
    if scale_type == "likert":
        return f"My answer is {sampled_option}, and that reflects my actual view."
    if scale_type == "categorical":
        return f"I choose {sampled_option} because it fits my situation best."
    return f"My answer is {sampled_option}."


def build_judge_prompt(
    persona_summary: str,
    question: str,
    response: str,
) -> str:
    return f"""Persona:
{persona_summary}

Question:
{question}

Response:
{response}

Score (1-5) for:
- realism: Does it sound like a real person?
- persona_consistency: Is it consistent with the persona?
- cultural_plausibility: Is it believable for the persona's cultural context?

Return JSON: {{"realism": N, "persona_consistency": N, "cultural_plausibility": N}}"""


# ---------------------------------------------------------------------------
# Persona compression for display/APIs
# ---------------------------------------------------------------------------

def compress_persona_for_display(persona: Persona, max_sentences: int = 4) -> str:
    """Extended natural language summary for APIs or UI."""
    return persona.to_compressed_summary()


# ---------------------------------------------------------------------------
# LLM mini-judge for semantic consistency (fallback when heuristic is ambiguous)
# ---------------------------------------------------------------------------

async def _judge_narrative_consistency(
    narrative: str,
    sampled_option: str,
    client: Any,
) -> bool:
    """Cheap LLM call to verify narrative matches the selected option.

    Universal prompt — works for numeric, likert, frequency, categorical.
    Returns True if consistent, False if contradictory.
    Uses gpt-4o-mini at temperature=0 for deterministic yes/no.
    """
    prompt = (
        f'The respondent selected option "{sampled_option}".\n\n'
        f"Does the explanation clearly match that answer?\n\n"
        f'Respond YES or NO.\n\n'
        f'Written answer: "{narrative}"'
    )
    try:
        result = await client.chat(
            [{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=3,
        )
        return "yes" in result.strip().lower()
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Reasoner with retry-on-banned, consistency check, LLM judge, and temp jitter
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3


async def reasoner_via_llm(
    persona: Persona,
    question: str,
    sampled_answer: str,
    distribution: Dict[str, float],
    memories: List[str],
    *,
    used_openings: Optional[set] = None,
    tone_override: Optional[str] = None,
    simulation_context: Optional[Dict[str, Any]] = None,
    option_labels: Optional[List[str]] = None,
    response_contract: Optional[Dict[str, Any]] = None,
    turn_understanding: Optional[Dict[str, Any]] = None,
    diagnostics_enabled: bool = False,
) -> Any:
    """Async reasoner with vague-answer bypass, retry on banned patterns,
    narrative-option mismatch, and post-hoc hedging injection.

    Pipeline:
    0. Vague-answer gate: ~22% of agents skip LLM entirely → short human answer
    1. Banned pattern check (fast regex)
    2. Heuristic keyword/indirect-pattern check
    3. LLM mini-judge fallback (only when heuristic is ambiguous AND retries remain)
    4. Post-hoc hedging injection for extra human texture
    """
    from config.settings import get_settings
    from llm.client import get_llm_client

    settings = get_settings()
    rng = ensure_py_rng(
        None,
        key=f"reasoner:{persona.agent_id}:{question}:{sampled_answer}",
    )

    # Gate: vague/terse answer bypass — probability varies by agent's style profile
    ns = persona.personal_anchors.narrative_style
    _profile = NarrativeStyleProfile(
        verbosity=ns.verbosity,
        preferred_tone=ns.preferred_tone,
        preferred_style=ns.preferred_style,
        slang_level=ns.slang_level,
        grammar_quality=ns.grammar_quality,
        voice_register=getattr(ns, "voice_register", "conversational") or "conversational",
        rhetorical_habit=getattr(ns, "rhetorical_habit", "direct") or "direct",
        avoid_phrases=tuple(getattr(ns, "avoid_phrases", []) or ()),
    )
    # Skip vague-answer bypass for open_text (no predefined option)
    is_open_text = not distribution
    _min_vague_words = 5
    stance_confidence_early = float(distribution.get(sampled_answer, 0.5)) if distribution else 0.5
    _fatigue_v = float((response_contract or {}).get("_fatigue", 0.0) or 0.0)
    try:
        from agents.behavior_controller import BehaviorController as _BCV

        _vbudget = _BCV.compute_budget(
            fatigue=_fatigue_v,
            confidence_band=compute_confidence_band(stance_confidence_early),
            grammar_quality=getattr(_profile, "grammar_quality", 0.5) if _profile else 0.5,
            settings=settings,
        )
    except ImportError:
        _vbudget = None
    alignment_meta: Dict[str, Any] = {
        "heuristic_consistent": None,
        "judge_consistent": None,
        "repaired": False,
        "hard_block_applied": False,
        "final_consistent": None,
    }
    if not is_open_text:
        base_vp = vague_answer_probability_for_profile(_profile)
        vague_prob = (
            0.5 * base_vp + 0.5 * float(_vbudget.vague_prob)
            if _vbudget is not None
            else base_vp
        )
        vague_prob = max(0.0, min(1.0, float(vague_prob)))
        if rng.random() < vague_prob:
            for _ in range(3):  # try up to 3 draws for a long enough vague answer
                vague = pick_vague_answer(sampled_answer, rng)
                if vague and len(vague.split()) >= _min_vague_words:
                    if diagnostics_enabled:
                        return {"answer": vague, "alignment": alignment_meta}
                    return vague
            # no long enough vague answer: fall through to LLM

    client = get_llm_client()
    _strict_voice = _strict_demographic_voice_cohort(persona, _profile.rhetorical_habit)
    system_prompt = _pick_system_prompt(
        rng,
        voice_register=_profile.voice_register,
        strict_demographic_voice=_strict_voice,
    )
    system_prompt += (
        " Hard rule: use at most one conversational hedge in your answer "
        '(e.g. "honestly", "I mean", "you know", "like" as filler); keep wording tight '
        "because light editing may run after you answer."
    )
    temperature = _agent_temperature(
        settings.llm_temperature, rng, personality=persona.personality,
    )
    scale = list(distribution.keys())
    numeric_label_map = _resolve_numeric_label_map(option_labels, scale, question)

    consistency_warning: Optional[str] = None
    stance_confidence = float(distribution.get(sampled_answer, 0.5)) if distribution else 0.5
    option_for_stance = numeric_label_map.get(sampled_answer) or sampled_answer
    stance_cat = _stance_category(option_for_stance)
    dynamic_tone_override = tone_override
    if dynamic_tone_override is None:
        contract_tone: Optional[str] = None
        if response_contract and response_contract.get("tone_selected"):
            contract_tone = str(response_contract.get("tone_selected") or "").strip()
        has_tradeoff = bool(
            response_contract and str(response_contract.get("tradeoff_guidance", "") or "").strip()
        )
        if contract_tone:
            if has_tradeoff and rng.random() < 0.62:
                dynamic_tone_override = contract_tone
            elif rng.random() < 0.5:
                dynamic_tone_override = contract_tone
            else:
                dynamic_tone_override = None
        else:
            cband = compute_confidence_band(stance_confidence)
            dynamic_tone_override = tone_for_confidence_band(cband, rng=rng)

    # Fatigue for behavior budget (mirrors build_agent_prompt; must exist in this scope)
    _fatigue = 0.0
    if response_contract:
        _fatigue = float(response_contract.get("_fatigue", 0.0) or 0.0)

    for attempt in range(_MAX_RETRIES + 1):
        prompt = build_agent_prompt(
            persona, question, sampled_answer, distribution, memories,
            rng=rng, consistency_warning=consistency_warning,
            used_openings=used_openings,
            tone_override=dynamic_tone_override,
            simulation_context=simulation_context,
            option_labels=option_labels,
            response_contract=response_contract,
            turn_understanding=turn_understanding,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        result = await client.chat(
            messages,
            temperature=temperature,
            max_tokens=int(getattr(settings, "llm_reasoner_max_tokens", 288)),
        )

        # Min-length guard: reject very short LLM output (orphaned filler)
        if not is_open_text and len(result.split()) < 8 and attempt < _MAX_RETRIES:
            consistency_warning = (
                "Your answer was too short. Write a complete 1-2 sentence opinion (at least 8 words)."
            )
            continue

        if is_banned_pattern(result) and attempt < _MAX_RETRIES:
            continue

        # Duration questions: reject score/rating leakage
        if is_open_text and _is_duration_question(question):
            if contains_duration_anti_pattern(result) and attempt < _MAX_RETRIES:
                consistency_warning = (
                    "This is a duration question. Do NOT provide a score or rating. "
                    "Answer with a time span like '5 years' or 'about a decade'."
                )
                continue

        # Skip consistency checks for open_text (no option to validate against)
        if not is_open_text:
            consistent, detected = validate_narrative_consistency(
                result, sampled_answer, scale,
                option_label_map=numeric_label_map,
            )
            alignment_meta["heuristic_consistent"] = bool(consistent)
            if not consistent and attempt < _MAX_RETRIES:
                label_hint = ""
                if sampled_answer in numeric_label_map:
                    label_hint = (
                        f' On this scale, "{sampled_answer}" means '
                        f'"{numeric_label_map[sampled_answer]}".'
                    )
                consistency_warning = (
                    f"Your previous answer contradicted \"{sampled_answer}\". "
                    f"You MUST clearly reflect \"{sampled_answer}\" in your answer."
                    f"{label_hint}"
                )
                continue

            if consistent and attempt < _MAX_RETRIES:
                judge_ok = await _judge_narrative_consistency(
                    result, sampled_answer, client,
                )
                alignment_meta["judge_consistent"] = bool(judge_ok)
                if not judge_ok:
                    consistency_warning = (
                        f"A consistency check found your answer does not clearly "
                        f"match \"{sampled_answer}\". Rewrite to clearly reflect "
                        f"\"{sampled_answer}\"."
                    )
                    continue

            # Hard post-generation filter: only very strong stances block hedging
            if stance_confidence >= 0.80 and _violates_strong_stance(result) and attempt < _MAX_RETRIES:
                consistency_warning = (
                    "Your answer contradicted a strong stance. Do NOT use 'but', 'however', "
                    "'on one hand', 'on the other hand', or 'depends'. State a single-sided opinion only."
                )
                continue

            # Decision-text consistency: narrative must express the chosen stance
            if stance_cat in ("strong_support", "strong_oppose", "support", "oppose"):
                if not _text_expresses_stance(result, stance_cat) and attempt < _MAX_RETRIES:
                    consistency_warning = (
                        f"Your answer did not clearly express \"{option_for_stance}\". "
                        "You MUST state your position clearly (e.g. support/oppose/agree/against)."
                    )
                    continue

        # Post-hoc processing chain for human texture (feature-flag aware)
        _pp_settings = _get_settings()
        _pp_log: List[str] = []

        try:
            from agents.behavior_controller import BehaviorController
            _budget = BehaviorController.compute_budget(
                fatigue=_fatigue,
                confidence_band=compute_confidence_band(stance_confidence),
                grammar_quality=getattr(_profile, "grammar_quality", 0.5) if _profile else 0.5,
                settings=_pp_settings,
            )
        except ImportError:
            _budget = None

        _transforms_fired = 0
        _max_transforms = _budget.max_transforms if _budget else 6

        def _try_transform(name: str, fn, *args, flag_attr: str = "", **kwargs) -> str:
            nonlocal _transforms_fired
            if _transforms_fired >= _max_transforms:
                return args[0] if args else ""
            if flag_attr and not getattr(_pp_settings, flag_attr, True):
                return args[0] if args else ""
            before = args[0] if args else ""
            out = fn(*args, **kwargs)
            if out != before:
                _pp_log.append(name)
                _transforms_fired += 1
            return out

        _conf_band = compute_confidence_band(stance_confidence)
        _b = _budget

        def _run_hedging(t: str) -> str:
            kw = {"hedge_probability": _b.hedge_prob} if _b else {}
            kw["profile"] = _profile
            return _try_transform("hedging", maybe_add_hedging, t, rng, **kw)

        def _run_thinking(t: str) -> str:
            kw = {"marker_probability": _b.thinking_marker_prob} if _b else {}
            return _try_transform(
                "thinking_markers", inject_thinking_markers, t, _profile, rng,
                flag_attr="enable_thinking_markers", **kw,
            )

        def _run_redundancy(t: str) -> str:
            kw = {"redundancy_probability": _b.redundancy_prob} if _b else {}
            return _try_transform(
                "redundancy", maybe_add_redundancy, t, rng,
                flag_attr="enable_redundancy_injection", **kw,
            )

        def _run_micro(t: str) -> str:
            kw = {"contradiction_probability": _b.micro_contradiction_prob} if _b else {}
            kw["agent_id"] = persona.agent_id
            kw["persona"] = persona
            return _try_transform(
                "micro_contradiction", maybe_add_micro_contradiction, t,
                confidence_band=_conf_band, rng=rng,
                flag_attr="enable_micro_contradiction", **kw,
            )

        def _run_fragment(t: str) -> str:
            kw = {"fragment_probability": _b.fragment_prob} if _b else {}
            out = _try_transform("fragmentize", maybe_fragmentize, t, _profile, rng, **kw)
            if out != t and _fragment_looks_dangling(out):
                return t
            return out

        def _run_polish(t: str) -> str:
            kw = {"polish_apply_probability": _b.polish_prob} if _b else {}
            return _try_transform("degrade_polish", degrade_polish, t, rng, **kw)

        _texture_steps = [
            ("hedging", _run_hedging),
            ("thinking_markers", _run_thinking),
            ("redundancy", _run_redundancy),
            ("micro_contradiction", _run_micro),
        ]
        _tail_steps = [
            ("fragmentize", _run_fragment),
            ("degrade_polish", _run_polish),
        ]
        _steps = list(_texture_steps)
        if _budget is not None:
            tail = list(_tail_steps)
            rng.shuffle(tail)
            _steps.extend(tail)
        else:
            _steps.extend(_tail_steps)
        for _name, _fn in _steps:
            result = _fn(result)

        result = trim_weak_terminal_suffix(result)

        # Final min-length guard: post-hoc steps can truncate; never return < 8 words
        if not is_open_text and len(result.split()) < 8 and attempt < _MAX_RETRIES:
            consistency_warning = (
                "Your answer was too short after editing. Write a complete 1-2 sentence opinion (at least 8 words)."
            )
            continue

        break

    # Hard final gate: never return contradictory structured narrative.
    if not is_open_text:
        final_consistent, _ = validate_narrative_consistency(
            result,
            sampled_answer,
            scale,
            option_label_map=numeric_label_map,
        )
        alignment_meta["final_consistent"] = bool(final_consistent)
        if not final_consistent:
            result = _deterministic_alignment_repair(
                sampled_option=sampled_answer,
                scale=scale,
                option_label_map=numeric_label_map,
            )
            repaired_consistent, _ = validate_narrative_consistency(
                result,
                sampled_answer,
                scale,
                option_label_map=numeric_label_map,
            )
            alignment_meta["repaired"] = True
            alignment_meta["hard_block_applied"] = True
            alignment_meta["final_consistent"] = bool(repaired_consistent)

    if diagnostics_enabled:
        return {"answer": result, "alignment": alignment_meta, "pp_log": _pp_log}
    return {"answer": result, "pp_log": _pp_log}
