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
    degrade_polish,
    maybe_add_hedging,
    maybe_fragmentize,
    pick_vague_answer,
    should_use_vague_answer,
    validate_demographic_plausibility,
)
from population.personas import Persona


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


def _pick_system_prompt(rng: Optional[random.Random] = None) -> str:
    r = rng or random
    return r.choice(_SYSTEM_PROMPTS)


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
        "Answer as this person in 1-3 sentences.\n"
        "Use at least one personal detail (hobby, cuisine, schedule, diet, etc.) in the explanation.\n"
        "Avoid generic phrases like \"I prefer to cook at home for my family.\" Use a specific personal context instead.\n"
        "Your probability distribution strongly favors \"{top_option}\" ({top_prob_pct}). "
        "You selected \"{sampled_option}\" ({sampled_prob_pct}). "
        "Your narrative MUST describe \"{sampled_option}\" frequency — that means {interpretation}.\n"
        "Do NOT describe \"{top_option}\" ordering patterns if it differs from your selection.\n"
        "Reply with only the answer, no meta-commentary."
    ),
    (
        "Respond in 1-3 sentences as this exact person.\n"
        "Mention something specific about your life — a food, a place, a habit.\n"
        "Your distribution top choice is \"{top_option}\" ({top_prob_pct}), but "
        "your selected answer is \"{sampled_option}\" ({sampled_prob_pct}). "
        "{interpretation}. Make sure your narrative matches that frequency — don't contradict it.\n"
        "Just give the answer, nothing else."
    ),
    (
        "Give a 1-3 sentence answer that sounds like something this person would actually say.\n"
        "Include a concrete detail from your daily routine.\n"
        "You chose \"{sampled_option}\" ({sampled_prob_pct}) — meaning {interpretation}. Stick to that.\n"
        "The distribution favors \"{top_option}\" ({top_prob_pct}) — do NOT describe that frequency unless it matches your selection.\n"
        "No preamble, no commentary, just the answer."
    ),
    (
        "Write 1-3 sentences as this person.\n"
        "Ground your answer in a real detail — your commute, your kitchen, your schedule.\n"
        "\"{sampled_option}\" ({sampled_prob_pct}) is your answer — {interpretation}. "
        "Everything you say must align with that frequency.\n"
        "Do NOT describe \"{top_option}\" frequency if it differs from \"{sampled_option}\".\n"
        "Output only the answer."
    ),
]

_MICRO_INSTRUCTION_VARIANTS: List[str] = [
    (
        "Give the shortest possible answer — a few words at most.\n"
        "Your answer is \"{sampled_option}\" ({sampled_prob_pct}) — {interpretation}. "
        "Just say it briefly, the way you'd text a friend.\n"
        "Output only the answer."
    ),
    (
        "Answer in under 5 words. No explanation needed.\n"
        "You chose \"{sampled_option}\" ({sampled_prob_pct}). Say it your way — super brief.\n"
        "Output only the answer."
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

_LIKERT_INSTRUCTION_VARIANTS: List[str] = [
    (
        'Your answer corresponds to: "{sampled_option}".\n'
        "Explain your stance in 1-2 short sentences."
    ),
    (
        'You chose "{sampled_option}". Briefly explain why.'
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
) -> str:
    r = rng or random.Random()
    pa = persona.personal_anchors
    options = list(distribution.keys())
    scale_type = infer_scale_type(options)
    # For empty options (open-ended), check if it's a duration question
    if not options and _is_duration_question(question):
        scale_type = "duration"
    allow_anchors = allow_persona_anchor(question)

    try:
        from config.domain import get_domain_config
        _currency = get_domain_config().currency
    except Exception:
        _currency = "USD"
    persona_block = (
        f"Age group: {persona.age}. Nationality: {persona.nationality}. Location: {persona.location}.\n"
        f"Income band: {_currency} {persona.income}/month. Occupation: {persona.occupation}. "
        f"Household size: {persona.household_size}."
    )
    if persona.family.spouse:
        persona_block += f" Spouse: yes, children: {persona.family.children}."
    persona_block += f"\nCar: {'yes' if persona.mobility.car else 'no'}, metro: {persona.mobility.metro_usage}."

    # Append lifestyle fields only when question topic warrants
    if allow_anchors:
        persona_block += (
            f"\nCuisine preference: {pa.cuisine_preference}. Diet: {pa.diet}. Hobby: {pa.hobby}.\n"
            f"Work schedule: {pa.work_schedule}. Dinner time: {pa.typical_dinner_time}. "
            f"Commute: {pa.commute_method}. Health focus: {pa.health_focus}."
        )

    # Archetype and cultural hints only for lifestyle-related questions
    archetype_block = ""
    cultural_block = ""
    if allow_anchors:
        archetype = pa.archetype if hasattr(pa, "archetype") else "default"
        archetype_hint = ARCHETYPE_HINTS.get(archetype, "")
        if archetype_hint:
            archetype_block = f"\nARCHETYPE MINDSET:\n{archetype_hint}\n"
        cultural_hint = CULTURAL_BEHAVIOR_HINTS.get(persona.nationality, "")
        if cultural_hint:
            cultural_block = f"\nCULTURAL CONTEXT:\n{cultural_hint}\n"

    behavior_desc = (
        f"convenience={persona.lifestyle.convenience_preference:.2f}, "
        f"service_pref={persona.lifestyle.primary_service_preference:.2f}, "
        f"price_sensitivity={persona.lifestyle.price_sensitivity:.2f}."
    )
    if scale_type not in ("open_text", "duration"):
        behavior_desc += f" Sampled response: \"{sampled_option}\"."

    memory_block = (
        "No relevant memories."
        if not memories
        else "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories[:5])
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
    )
    style = pick_style_from_profile(_profile, rng=rng)
    structure = pick_sentence_structure(rng=rng)
    opening = pick_opening_deduplicated(ctx, used_openings=used_openings, rng=rng)
    anchor_name, anchor_value = pick_persona_anchor(ctx, rng=rng)
    length = pick_length_from_profile(_profile, rng=rng)
    tone = tone_override if tone_override else pick_tone_from_profile(_profile, rng=rng)
    include_anchor = allow_anchors and length != "micro" and r.random() < 0.55
    style_instruction = build_style_instruction(
        style, structure, opening, anchor_name, anchor_value,
        length=length, tone=tone, include_anchor=include_anchor,
    )
    if tone_override and scale_type not in ("open_text", "duration"):
        style_instruction += "\nTone affects wording only, not the answer content. Do not change the selected answer."

    # Scale-type driven instruction selection
    interpretation = _FREQUENCY_INTERPRETATION.get(sampled_option, sampled_option)
    sampled_prob = distribution.get(sampled_option, 0.0)

    if scale_type == "open_text":
        instruction = r.choice(_OPEN_TEXT_INSTRUCTION_VARIANTS)
    elif scale_type == "duration":
        instruction = r.choice(_DURATION_INSTRUCTION_VARIANTS)
    elif scale_type == "numeric":
        numeric_label_map = _resolve_numeric_label_map(option_labels, options, question)
        sampled_label = numeric_label_map.get(sampled_option, "")
        semantic_guardrail = ""
        if sampled_label:
            legend = _format_numeric_legend(numeric_label_map)
            semantic_guardrail = (
                f"IMPORTANT SCALE SEMANTICS: {legend}.\n"
                f'You selected option "{sampled_option}", which means "{sampled_label}".\n'
                "Your explanation must justify this exact stance. Do not invert the meaning."
            )
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

def _agent_temperature(base_temp: float, rng: Optional[random.Random] = None) -> float:
    """Add uniform jitter to the base LLM temperature so agents vary in creativity."""
    r = rng or random
    jitter = (r.random() - 0.5) * 0.30  # +/- 0.15
    return max(0.3, min(1.5, base_temp + jitter))


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
) -> str:
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
    rng = random.Random()

    # Gate: vague/terse answer bypass — probability varies by agent's style profile
    ns = persona.personal_anchors.narrative_style
    _profile = NarrativeStyleProfile(
        verbosity=ns.verbosity,
        preferred_tone=ns.preferred_tone,
        preferred_style=ns.preferred_style,
        slang_level=ns.slang_level,
        grammar_quality=ns.grammar_quality,
    )
    # Skip vague-answer bypass for open_text (no predefined option)
    is_open_text = not distribution
    if not is_open_text:
        vague_prob = vague_answer_probability_for_profile(_profile)
        if rng.random() < vague_prob:
            vague = pick_vague_answer(sampled_answer, rng)
            if vague:
                return vague

    client = get_llm_client()
    system_prompt = _pick_system_prompt(rng)
    temperature = _agent_temperature(settings.llm_temperature, rng)
    scale = list(distribution.keys())
    numeric_label_map = _resolve_numeric_label_map(option_labels, scale, question)

    consistency_warning: Optional[str] = None

    for attempt in range(_MAX_RETRIES + 1):
        prompt = build_agent_prompt(
            persona, question, sampled_answer, distribution, memories,
            rng=rng, consistency_warning=consistency_warning,
            used_openings=used_openings,
            tone_override=tone_override,
            simulation_context=simulation_context,
            option_labels=option_labels,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        result = await client.chat(
            messages,
            temperature=temperature,
            max_tokens=256,
        )

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
                if not judge_ok:
                    consistency_warning = (
                        f"A consistency check found your answer does not clearly "
                        f"match \"{sampled_answer}\". Rewrite to clearly reflect "
                        f"\"{sampled_answer}\"."
                    )
                    continue

        # Post-hoc hedging for human texture
        result = maybe_add_hedging(result, rng)
        result = maybe_fragmentize(result, _profile, rng)
        result = degrade_polish(result, rng)
        return result
    result = maybe_add_hedging(result, rng)
    result = maybe_fragmentize(result, _profile, rng)
    result = degrade_polish(result, rng)
    return result
