"""
Narrative Diversity Engine: styles, sentence openings, grammatical structures,
tone modifiers, and a banned-pattern filter to eliminate repetitive AI-style
writing.

Each agent has a persistent ``NarrativeStyleProfile`` assigned during
population synthesis.  The profile determines the agent's typical verbosity,
tone, slang level, and grammar quality.  Per-response picks still happen
but are biased ~80% toward the persistent profile, with ~20% random
deviation to mirror natural human variation.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Persistent Narrative Style Profile
# ---------------------------------------------------------------------------

@dataclass
class NarrativeStyleProfile:
    """Persistent writing-style identity for one agent.

    Assigned once during population synthesis and reused across all survey
    responses.  Each field controls one axis of narrative variation.
    """
    verbosity: str          # "micro", "short", "medium", "long"
    preferred_tone: str     # one of TONE_MODIFIERS
    preferred_style: str    # one of NARRATIVE_STYLES
    slang_level: float      # 0.0 = formal, 1.0 = heavy slang
    grammar_quality: float  # 0.0 = fragments/typos, 1.0 = proper grammar


def derive_narrative_style_profile(
    age: str,
    income: str,
    occupation: str,
    nationality: str,
    rng: random.Random,
) -> NarrativeStyleProfile:
    """Deterministically derive a style profile from demographics.

    Young, low-income agents trend toward micro/short, high slang, low grammar.
    Older, professional agents trend toward medium/long, low slang, high grammar.
    Randomness is injected via rng so the mapping is probabilistic, not rigid.
    """
    # Verbosity: age + occupation driven
    if age == "18-24":
        verb_weights = {"micro": 0.40, "short": 0.35, "medium": 0.20, "long": 0.05}
    elif age in ("25-34",):
        verb_weights = {"micro": 0.25, "short": 0.30, "medium": 0.30, "long": 0.15}
    elif occupation in ("managerial", "professional"):
        verb_weights = {"micro": 0.12, "short": 0.22, "medium": 0.38, "long": 0.28}
    else:
        verb_weights = {"micro": 0.20, "short": 0.30, "medium": 0.30, "long": 0.20}
    verbosity = rng.choices(
        list(verb_weights.keys()), weights=list(verb_weights.values()), k=1
    )[0]

    # Tone: nationality + age + occupation
    if age == "18-24":
        tone_pool = ["casual", "terse", "lazy", "humorous", "distracted", "blunt"]
    elif occupation == "managerial":
        tone_pool = ["matter_of_fact", "reflective", "casual", "blunt"]
    elif nationality in ("Western",):
        tone_pool = ["casual", "humorous", "matter_of_fact", "blunt", "terse"]
    elif nationality in ("Indian", "Pakistani", "Filipino"):
        tone_pool = ["casual", "matter_of_fact", "reflective", "terse"]
    else:
        tone_pool = ["casual", "matter_of_fact", "terse", "reflective"]
    preferred_tone = rng.choice(tone_pool)

    # Style: income + family context
    if income in ("<10k", "10-25k"):
        style_pool = ["constraint", "economic", "practical", "habit", "routine"]
    elif income == "50k+":
        style_pool = ["preference", "social", "spontaneous", "nostalgia", "routine"]
    else:
        style_pool = ["routine", "preference", "habit", "family", "health", "contrast"]
    preferred_style = rng.choice(style_pool)

    # Slang level: age-driven with noise
    base_slang = {"18-24": 0.70, "25-34": 0.45, "35-44": 0.25, "45-54": 0.15, "55+": 0.10}
    slang = base_slang.get(age, 0.30) + rng.uniform(-0.15, 0.15)
    slang = max(0.0, min(1.0, slang))

    # Grammar quality: occupation + income driven
    base_grammar = 0.50
    if occupation in ("managerial", "professional", "technical"):
        base_grammar += 0.20
    if income in ("50k+", "25-50k"):
        base_grammar += 0.10
    if age == "18-24":
        base_grammar -= 0.15
    grammar = base_grammar + rng.uniform(-0.12, 0.12)
    grammar = max(0.0, min(1.0, grammar))

    return NarrativeStyleProfile(
        verbosity=verbosity,
        preferred_tone=preferred_tone,
        preferred_style=preferred_style,
        slang_level=slang,
        grammar_quality=grammar,
    )


# ---------------------------------------------------------------------------
# 14 narrative styles (the "voice" of the response)
# Weights are flattened so no single style dominates.
# ---------------------------------------------------------------------------

NARRATIVE_STYLES: List[str] = [
    "routine",
    "preference",
    "constraint",
    "family",
    "health",
    "social",
    "habit",
    "contrast",
    "nostalgia",
    "economic",
    "spontaneous",
    "cultural",
    "practical",
    "emotional",
]

STYLE_WEIGHTS: Dict[str, float] = {
    "routine": 0.12,
    "preference": 0.11,
    "constraint": 0.10,
    "family": 0.08,
    "health": 0.07,
    "social": 0.07,
    "habit": 0.07,
    "contrast": 0.07,
    "nostalgia": 0.06,
    "economic": 0.06,
    "spontaneous": 0.05,
    "cultural": 0.05,
    "practical": 0.05,
    "emotional": 0.04,
}

# ---------------------------------------------------------------------------
# Tone modifiers — a second axis of variation layered on top of style
# ---------------------------------------------------------------------------

TONE_MODIFIERS: List[str] = [
    "casual",
    "matter_of_fact",
    "reflective",
    "humorous",
    "terse",
    "blunt",
    "lazy",
    "distracted",
    "skeptical",
    "practical",
]

TONE_WEIGHTS: Dict[str, float] = {
    "casual": 0.25,
    "matter_of_fact": 0.16,
    "reflective": 0.12,
    "humorous": 0.08,
    "terse": 0.15,
    "blunt": 0.10,
    "lazy": 0.08,
    "distracted": 0.06,
    "skeptical": 0.04,
    "practical": 0.04,
}

TONE_HINTS: Dict[str, str] = {
    "casual": "Use a relaxed, conversational tone — like texting a friend.",
    "matter_of_fact": "Be direct and factual, no filler words.",
    "reflective": "Sound thoughtful, as if you're pausing to think about it.",
    "humorous": "Add a touch of dry humor or self-awareness.",
    "terse": "Keep it blunt and minimal — you're in a hurry.",
    "blunt": "Sound like you couldn't care less about the survey. Minimal effort, no fluff.",
    "lazy": "Answer like you're half-asleep filling this out. Minimal words, maybe trailing off.",
    "distracted": "Answer like you're doing something else at the same time. Slightly unfocused.",
    "skeptical": "Sound slightly doubtful or questioning — not convinced it's that simple.",
    "practical": "Focus on what actually works for you — no fluff, just outcomes.",
}

LENGTH_DISTRIBUTION: Dict[str, float] = {
    "micro": 0.32,
    "short": 0.30,
    "medium": 0.26,
    "long": 0.12,
}

LENGTH_HINTS: Dict[str, str] = {
    "micro": "Answer in 1-5 words only. Example: 'Mostly weekends.' or 'A couple times a week.'",
    "short": "Keep your answer to 1 sentence.",
    "medium": "Answer in 2-3 sentences.",
    "long": "Answer in 3-4 sentences with detail.",
}

# ---------------------------------------------------------------------------
# ~65 diverse sentence openings (none starting with banned patterns)
# ---------------------------------------------------------------------------

OPENING_PATTERNS: List[str] = [
    # --- messy human / fragmentary (30+) — heaviest category ---
    "Depends really.",
    "Hard to say but",
    "Not sure actually, maybe I",
    "Hmm I think I",
    "Ugh, probably I",
    "Like maybe",
    "It varies honestly but I",
    "Good question, I",
    "Lol I",
    "Tbh I",
    "Idk, maybe I",
    "I dunno, like I",
    "Umm so basically I",
    "So like I",
    "Sometimes. Other times I",
    "It really depends on",
    "I guess I",
    "Umm I",
    "Haha I",
    "Honestly no idea but I",
    "Oh man, I",
    "Hard to say tbh I",
    "Wait let me think... I",
    "I think maybe I",
    "Errr I",
    "I mean I",
    "Hmm probably I",
    "Not gonna lie I",
    "Kinda depends but I",
    "Bruh I",
    "Uhhh I",
    "I guess probably I",
    # --- direct answer / minimal (18) ---
    "Probably about",
    "I'd say around",
    "Roughly",
    "More or less",
    "Give or take,",
    "Not that much.",
    "Quite a bit actually.",
    "A fair amount.",
    "Barely.",
    "All the time.",
    "Way too often.",
    "Less than you'd think.",
    "Not as much as I used to.",
    "A lot more than I should.",
    "Enough.",
    "Too much.",
    "Not enough lol.",
    "Depends.",
    # --- minimal fragments (12) ---
    "Mm.",
    "Yeah.",
    "Not much.",
    "Eh.",
    "Yep.",
    "Nope.",
    "Sure.",
    "Yeah no.",
    "Meh.",
    "Kinda.",
    "Not really.",
    "Yeah sorta.",
    # --- colloquial / terse (10) ---
    "Nah, I",
    "Yeah, pretty much I",
    "Rarely, if ever.",
    "Honestly? Not much.",
    "Pretty often actually.",
    "Yeah a lot.",
    "Not really no.",
    "Couple times.",
    "Like every day.",
    "All the time honestly.",
    # --- casual / non-sequitur (12) ---
    "Man,",
    "Yo,",
    "So yeah,",
    "I mean look,",
    "Ha, well,",
    "Right so",
    "Anyway,",
    "OK honestly,",
    "Lmao,",
    "Oof,",
    "Welp,",
    "OK so like,",
    # --- deflection / question (10) ---
    "Why does everyone ask this,",
    "Funny you ask,",
    "You know what, I",
    "Thing is, I",
    "OK so",
    "Real talk,",
    "Here's the thing,",
    "Put it this way,",
    "I don't even know, I",
    "That's a weird question but I",
    # --- situational / conditional (10) ---
    "When I'm lazy, I",
    "If it's been a rough day, I",
    "On a good week, I",
    "Depending on my mood, I",
    "If I had to guess, I",
    "On busy weeks, I",
    "When there's nothing in the fridge, I",
    "If money wasn't an issue, I'd",
    "End of the month I",
    "If I'm tired I",
    # --- routine (10 — casual only, no polished templates) ---
    "Most days, I",
    "Honestly, I",
    "Around dinner time, I",
    "These days I",
    "Every Friday, I",
    "Lately, I",
    "On weekends, I",
    "Midweek, I",
    "Sunday is the one day I",
    "On my days off, I",
    # --- personal context (8 — grounded, not essay-like) ---
    "My {diet} thing means I",
    "Money-wise, I",
    "Delivery fees add up, so I",
    "It's cheaper to",
    "The kids always want",
    "Cooking for one is",
    "My trainer says I",
    "Health-wise, I",
    # --- emotional / spontaneous (6) ---
    "I won't lie,",
    "Look, I",
    "If I'm being real,",
    "Funny thing is, I",
    "Not gonna overthink this,",
    "Compared to last year, I",
]

# ---------------------------------------------------------------------------
# 8 grammatical sentence structures
# ---------------------------------------------------------------------------

SENTENCE_STRUCTURES: List[str] = [
    "reason_then_behavior",
    "behavior_then_explanation",
    "context_behavior_detail",
    "contrast",
    "anecdote",
    "question_then_answer",
    "direct_statement",
    "conditional",
]

# ---------------------------------------------------------------------------
# Banned starts (patterns that signal "AI-style" repetition)
# ---------------------------------------------------------------------------

BANNED_STARTS: List[str] = [
    r"^With my\b",
    r"^Since I\b",
    r"^As someone who\b",
    r"^As someone\b",
    r"^As a\b",
    r"^Being a\b",
    r"^Given that I\b",
    r"^Considering my\b",
    r"^I usually order\b",
    r"^I prefer to cook\b",
    r"^I tend to\b",
    r"^I find that\b",
    r"^In my experience\b",
    r"^Balancing my\b",
    r"^Having a\b",
    r"^Due to my\b",
    r"^It's worth noting\b",
    r"^I would say\b",
    r"^I believe\b",
    r"^In terms of\b",
    r"^After a long day\b",
    r"^When I get home\b",
    r"^Most evenings\b",
    r"^By the time\b",
    r"^After work\b",
]

_BANNED_COMPILED = [re.compile(p, re.IGNORECASE) for p in BANNED_STARTS]

BANNED_PHRASES_ANYWHERE: List[str] = [
    "after a long day", "when i get home", "most evenings",
    "by the time", "as someone who", "with my busy schedule",
    "given my", "considering my", "being a", "in my experience",
    "after work,", "with my schedule", "since i work",
    "given that i", "due to my", "as a busy", "as a working",
    "back home we always", "the way my week goes",
    "between work and everything", "coming from a",
    "my family back home", "the thing about",
    "score:", "scored it", "rating:", "out of 5", "out of 10",
    "i give it a", "my score", "my rating", "i scored",
]


# ---------------------------------------------------------------------------
# Duration answer validation — reject score/rating leakage in tenure questions
# ---------------------------------------------------------------------------

_DURATION_ANTI_PATTERNS = re.compile(
    r"\b(score|rating|rate)\s*(of\s*)?\d|\b\d\s*(out of|\/\s*)\d|"
    r"give it a\s*\d|i'd (give|rate)\s*(it\s*)?\d",
    re.IGNORECASE
)

_DURATION_VALID_PATTERN = re.compile(
    r"\b(\d+)\s*(year|years|yr|yrs|month|months|decade)s?\b",
    re.IGNORECASE
)


def contains_duration_anti_pattern(text: str) -> bool:
    """True if text contains score/rating leakage (invalid for duration questions)."""
    return bool(_DURATION_ANTI_PATTERNS.search(text))


def validate_duration_answer(text: str) -> bool:
    """True if text contains a plausible duration (years, months, decade)."""
    return bool(_DURATION_VALID_PATTERN.search(text))


# Score/rating leakage (real humans don't say "I give it a 5" or "Score: 3")
_SCORE_LEAKAGE = re.compile(
    r"\b(score|rating|rate)\s*:?\s*\d|\bscored\s+it\s+\d|\b(out of|/)\s*\d\b|"
    r"\bgive\s+it\s+a\s+\d|\bi\s+give\s+it\s+\d|\b\d\s*out of\s*\d\b",
    re.IGNORECASE,
)


def is_banned_pattern(text: str) -> bool:
    """Return True if the text contains a banned AI-style pattern.

    Checks both start-of-string regex anchors and mid-sentence substring
    matches (punctuation-normalized) to catch hedging-prefixed clichés
    like ``"I mean look, with my busy schedule..."``.
    Also rejects score/rating leakage (e.g. "Score: 5", "I give it a 3").
    """
    text_stripped = text.strip()
    if any(pat.search(text_stripped) for pat in _BANNED_COMPILED):
        return True
    if _SCORE_LEAKAGE.search(text_stripped):
        return True

    normalized = re.sub(r'[^\w\s]', ' ', text_stripped.lower())
    return any(phrase in normalized for phrase in BANNED_PHRASES_ANYWHERE)


# ---------------------------------------------------------------------------
# Frequency keyword map for narrative-option consistency validation
# ---------------------------------------------------------------------------

FREQUENCY_KEYWORDS: Dict[str, List[str]] = {
    "rarely": [
        "rarely", "almost never", "hardly ever", "once in a while",
        "very seldom", "not often", "infrequently",
    ],
    "1-2 per week": [
        "once or twice a week", "1-2 times", "a couple times a week",
        "once a week", "twice a week",
    ],
    "3-4 per week": [
        "3-4 times", "three or four times", "several times a week",
        "few times a week", "most weekdays",
    ],
    "daily": [
        "daily", "every day", "each day", "every single day",
        "on a daily basis", "day in day out",
    ],
    "multiple per day": [
        "multiple times a day", "several times a day", "twice a day",
        "more than once a day", "a few times daily", "two or three times a day",
    ],
}

_CONTRADICTION_PAIRS: Dict[str, List[str]] = {
    "rarely": ["daily", "multiple per day", "3-4 per week"],
    "1-2 per week": ["multiple per day"],
    "daily": ["rarely"],
    "multiple per day": ["rarely", "1-2 per week"],
}

# Indirect signals that contradict a given option even without explicit
# frequency keywords.  Each entry maps an option to phrases that *oppose* it.
_INDIRECT_CONTRADICTIONS: Dict[str, List[str]] = {
    "daily": [
        "i cook at home", "i prepare meals", "i make my own food",
        "prefer cooking", "enjoy cooking", "love to cook",
        "don't bother with apps", "don't use delivery",
        "never order", "avoid ordering", "skip delivery",
    ],
    "multiple per day": [
        "i cook at home", "i prepare meals", "i make my own food",
        "rarely order", "don't order much", "once in a while",
    ],
    "rarely": [
        "every evening i order", "i order after work",
        "my go-to routine is delivery", "delivery is my lifeline",
        "order food daily", "can't live without delivery",
        "always ordering", "constant deliveries",
    ],
}

# Positive signals that *should* appear for a given option.  If the text
# matches a positive signal for a contradicting option, flag it.
_POSITIVE_SIGNALS: Dict[str, List[str]] = {
    "rarely": [
        "cook at home", "prepare meals", "make my own",
        "don't really order", "almost never", "once in a blue moon",
    ],
    "daily": [
        "every evening", "every night", "after work i order",
        "my go-to", "can't live without", "delivery is a lifeline",
    ],
    "multiple per day": [
        "breakfast and dinner", "lunch and dinner delivery",
        "more than once", "a few times each day",
    ],
}

_POSITIVE_STANCE_KEYWORDS = (
    "support", "agree", "favor", "favour", "approve", "back", "stability",
    "relief", "benefit", "help", "good idea", "should implement",
)
_NEGATIVE_STANCE_KEYWORDS = (
    "oppose", "disagree", "against", "reject", "harm", "hurt", "bad idea",
    "should not implement", "discourage", "risk", "concern", "worsen",
)
_NEUTRAL_STANCE_KEYWORDS = ("neutral", "mixed", "unsure", "depends")


def _infer_label_stance(label_text: str) -> Optional[str]:
    t = (label_text or "").lower()
    if any(k in t for k in _NEUTRAL_STANCE_KEYWORDS):
        return "neutral"
    pos = any(k in t for k in _POSITIVE_STANCE_KEYWORDS)
    neg = any(k in t for k in _NEGATIVE_STANCE_KEYWORDS)
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return None


def _infer_narrative_stance(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(k in t for k in _NEUTRAL_STANCE_KEYWORDS):
        return "neutral"
    pos = any(k in t for k in _POSITIVE_STANCE_KEYWORDS)
    neg = any(k in t for k in _NEGATIVE_STANCE_KEYWORDS)
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return None


def validate_numeric_consistency(text: str, score: str) -> bool:
    """Check that numeric narrative contains the sampled score.

    Reject responses like 'I'd give it a 9' when sampled score is '3'.
    """
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return False
    return score in numbers


def _is_numeric_scale(scale: List[str]) -> bool:
    """Return True if all scale options are numeric strings."""
    return bool(scale) and all(o.isdigit() for o in scale)


def validate_narrative_consistency(
    narrative: str,
    sampled_option: str,
    scale: List[str],
    option_label_map: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Optional[str]]:
    """Check whether narrative text contradicts the sampled option.

    For numeric scales: requires the score to appear in the text.
    For frequency scales: uses keyword/indirect/positive-signal checks.
    For other scales: returns (True, None) — LLM judge handles ambiguity.
    For empty scale (open_text): returns (True, None) — no option to validate.

    Returns (is_consistent, detected_contradicting_option_or_None).
    """
    if not scale:
        return True, None

    text_lower = narrative.lower()

    # Numeric scale: when using hidden-state (option_label_map), validate stance only
    if _is_numeric_scale(scale):
        if option_label_map and sampled_option in option_label_map:
            expected = _infer_label_stance(option_label_map.get(sampled_option, ""))
            observed = _infer_narrative_stance(narrative)
            if (
                expected in {"positive", "negative"}
                and observed in {"positive", "negative"}
                and expected != observed
            ):
                return False, sampled_option
            return True, None
        if not validate_numeric_consistency(narrative, sampled_option):
            return False, sampled_option
        return True, None

    # Layer 1: direct frequency keyword contradiction
    contradicting_options = _CONTRADICTION_PAIRS.get(sampled_option, [])
    for option in contradicting_options:
        keywords = FREQUENCY_KEYWORDS.get(option, [])
        for kw in keywords:
            if kw in text_lower:
                return False, option

    # Layer 2: indirect contradiction patterns for THIS option
    indirect = _INDIRECT_CONTRADICTIONS.get(sampled_option, [])
    for pattern in indirect:
        if pattern in text_lower:
            return False, sampled_option

    # Layer 3: positive signals for contradicting options
    for option in contradicting_options:
        signals = _POSITIVE_SIGNALS.get(option, [])
        for sig in signals:
            if sig in text_lower:
                return False, option

    return True, None


# ---------------------------------------------------------------------------
# Random selection helpers
# ---------------------------------------------------------------------------

def _weighted_choice(items: Dict[str, float], rng: Optional[random.Random] = None) -> str:
    """Pick one key from a {name: probability} dict using weighted sampling."""
    r = rng or random
    keys = list(items.keys())
    weights = [items[k] for k in keys]
    return r.choices(keys, weights=weights, k=1)[0]


def pick_narrative_style(rng: Optional[random.Random] = None) -> str:
    return _weighted_choice(STYLE_WEIGHTS, rng)


def pick_tone(rng: Optional[random.Random] = None) -> str:
    return _weighted_choice(TONE_WEIGHTS, rng)


def pick_response_length(rng: Optional[random.Random] = None) -> str:
    """Return 'short', 'medium', or 'long' via weighted sampling."""
    return _weighted_choice(LENGTH_DISTRIBUTION, rng)


def pick_opening(
    persona_context: Optional[dict] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Pick and fill a random opening pattern using persona details."""
    r = rng or random
    template = r.choice(OPENING_PATTERNS)
    ctx = persona_context or {}
    try:
        return template.format_map(_SafeDict(ctx))
    except Exception:
        return template


def pick_opening_deduplicated(
    persona_context: Optional[dict] = None,
    used_openings: Optional[set] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Pick an opening that hasn't been used in this batch yet.

    Falls back to a random pick if all templates are exhausted (unlikely
    with 120+ patterns in a 500-agent batch).
    """
    r = rng or random
    ctx = persona_context or {}
    used = used_openings if used_openings is not None else set()

    available = [t for t in OPENING_PATTERNS if t not in used]
    if not available:
        used.clear()
        available = list(OPENING_PATTERNS)

    template = r.choice(available)
    used.add(template)

    try:
        return template.format_map(_SafeDict(ctx))
    except Exception:
        return template


def pick_sentence_structure(rng: Optional[random.Random] = None) -> str:
    r = rng or random
    return r.choice(SENTENCE_STRUCTURES)


def pick_persona_anchor(
    persona_context: dict,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str]:
    """Pick one personal anchor to emphasize. Returns (anchor_name, anchor_value)."""
    r = rng or random
    candidates = [
        ("hobby", persona_context.get("hobby", "")),
        ("cuisine", persona_context.get("cuisine_preference", "")),
        ("diet", persona_context.get("diet", "")),
        ("health_focus", persona_context.get("health_focus", "")),
        ("work_schedule", persona_context.get("work_schedule", "")),
        ("commute_method", persona_context.get("commute_method", "")),
    ]
    candidates = [(k, v) for k, v in candidates if v]
    if not candidates:
        return ("hobby", "reading")
    return r.choice(candidates)


# ---------------------------------------------------------------------------
# Style instruction builder (injected into LLM prompt)
# ---------------------------------------------------------------------------

def build_style_instruction(
    style: str,
    structure: str,
    opening: str,
    anchor_name: str,
    anchor_value: str,
    length: str = "medium",
    tone: str = "casual",
    include_anchor: bool = True,
) -> str:
    """Build a short instruction block telling the LLM how to write this response."""
    structure_hints = {
        "reason_then_behavior": "Start with the reason, then describe the behavior.",
        "behavior_then_explanation": "State the behavior first, then explain why.",
        "context_behavior_detail": "Set the context, describe the behavior, add a personal detail.",
        "contrast": "Contrast with a past habit or what others do, then state current behavior.",
        "anecdote": "Open with a brief personal anecdote, then give your answer.",
        "question_then_answer": "Pose a rhetorical question to yourself, then answer it.",
        "direct_statement": "Jump straight into your answer with no preamble.",
        "conditional": "Give your view and a typical situation where it applies, without defaulting to 'it depends'.",
    }
    hint = structure_hints.get(structure, "Write naturally.")
    length_hint = LENGTH_HINTS.get(length, LENGTH_HINTS["medium"])
    tone_hint = TONE_HINTS.get(tone, TONE_HINTS["casual"])

    anchor_line = ""
    if include_anchor:
        anchor_line = f"Weave in your {anchor_name} ({anchor_value}) naturally.\n"

    opening_instruction = (
        f"Your FIRST WORDS must be: \"{opening}\"\n"
        if length not in ("micro",)
        else f"Begin your answer similarly to: \"{opening}\"\n"
    )

    return (
        f"Narrative voice: {style}. {hint}\n"
        f"Tone: {tone_hint}\n"
        f"{opening_instruction}"
        f"{anchor_line}"
        f"{length_hint}\n"
        f"NEVER start with 'With my', 'Since I', 'As someone who', 'As a', "
        f"'Being a', 'I tend to', 'I find that', 'In my experience', 'Balancing my', "
        f"'After a long day', 'When I get home', 'Most evenings', 'By the time', 'After work', "
        f"'Back home', 'Living in', 'Between work', 'The way my week goes', or 'Coming from'."
    )


# ---------------------------------------------------------------------------
# Profile-aware picking: biased toward the agent's persistent style (~80%)
# with ~20% random deviation for natural variation.
# ---------------------------------------------------------------------------

_PROFILE_LOYALTY = 0.80


def pick_style_from_profile(
    profile: Optional[NarrativeStyleProfile] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Pick narrative style biased by the agent's persistent profile."""
    r = rng or random
    if profile is None or r.random() > _PROFILE_LOYALTY:
        return _weighted_choice(STYLE_WEIGHTS, rng)
    return profile.preferred_style


def pick_tone_from_profile(
    profile: Optional[NarrativeStyleProfile] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Pick tone biased by the agent's persistent profile."""
    r = rng or random
    if profile is None or r.random() > _PROFILE_LOYALTY:
        return _weighted_choice(TONE_WEIGHTS, rng)
    return profile.preferred_tone


def pick_length_from_profile(
    profile: Optional[NarrativeStyleProfile] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Pick response length biased by the agent's persistent profile."""
    r = rng or random
    if profile is None or r.random() > _PROFILE_LOYALTY:
        return _weighted_choice(LENGTH_DISTRIBUTION, rng)
    return profile.verbosity


def vague_answer_probability_for_profile(
    profile: Optional[NarrativeStyleProfile] = None,
) -> float:
    """Return the probability of giving a vague/terse answer based on style.

    Micro-verbosity agents give vague answers ~50% of the time;
    long-verbosity agents almost never.
    """
    if profile is None:
        return 0.30
    probs = {"micro": 0.50, "short": 0.30, "medium": 0.15, "long": 0.05}
    return probs.get(profile.verbosity, 0.30)


class _SafeDict(dict):
    """Dict subclass that returns the key wrapped in braces for missing keys."""
    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
