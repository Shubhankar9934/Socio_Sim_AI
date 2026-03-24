"""Decision->expression response contract helpers.

Centralizes confidence banding, expected score computation, and
expression mode/tone routing so LPFG remains the sole decision authority
while the narrative layer only performs language rendering.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import get_settings


@dataclass(frozen=True)
class ResponseDecisionContract:
    """Compact payload threaded from decision to narrative layer."""

    expression_mode: str
    interaction_mode: str
    sampled_option: str
    max_prob: float
    confidence_band: str
    tone_selected: str
    expected_score: Optional[float] = None
    latent_stance: Optional[str] = None
    runner_up_option: Optional[str] = None
    dominant_factor: Optional[str] = None
    dominant_score: Optional[float] = None
    narrative_guidance: Optional[str] = None
    tradeoff_guidance: Optional[str] = None
    belief_statements: Optional[List[str]] = None
    personality_summary: Optional[str] = None
    decision_latency: str = "normal"  # "instant", "normal", "deliberate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_confidence_band(max_prob: float) -> str:
    """Map max probability to low/medium/high confidence band."""
    settings = get_settings()
    high_threshold = float(settings.decision_confident_threshold)
    medium_threshold = float(min(0.30, high_threshold))
    p = float(max(0.0, min(1.0, max_prob)))
    if p >= high_threshold:
        return "high"
    if p >= medium_threshold:
        return "medium"
    return "low"


def tone_for_confidence_band(confidence_band: str, rng=None) -> str:
    """Narrative tone mapping used by prompt layer.

    Returns a varied tone from a pool per band instead of a fixed value.
    """
    import random as _random
    r = rng or _random
    if confidence_band == "high":
        pool = ["confident", "blunt", "matter_of_fact"]
    elif confidence_band == "medium":
        pool = ["casual", "reflective", "emotional_practical"]
    else:
        pool = ["uncertain", "distracted", "lazy", "skeptical"]
    try:
        return r.choice(pool)
    except Exception:
        return pool[0]


def compute_expected_score(
    distribution: Dict[str, float],
    ordered_options: Optional[List[str]] = None,
) -> Optional[float]:
    """Compute normalized expected score in [0, 1] for ordered options."""
    if not distribution:
        return None
    opts = list(ordered_options or distribution.keys())
    if len(opts) == 1:
        return 1.0
    if len(opts) == 0:
        return None
    denom = float(max(1, len(opts) - 1))
    score = 0.0
    for idx, opt in enumerate(opts):
        score += (idx / denom) * float(distribution.get(opt, 0.0))
    return float(max(0.0, min(1.0, score)))


def latent_stance_from_score(expected_score: Optional[float], confidence_band: str) -> Optional[str]:
    """Map latent score to a coarse semantic stance for open expression."""
    if expected_score is None:
        return None
    s = float(expected_score)
    if s >= 0.70:
        return "positive_strong" if confidence_band == "high" else "positive"
    if s >= 0.55:
        return "positive"
    if s <= 0.30:
        return "negative_strong" if confidence_band == "high" else "negative"
    if s <= 0.45:
        return "negative"
    return "mixed"


def infer_expression_mode(scale_type: str, distribution: Dict[str, float]) -> str:
    """Resolve explicit expression mode for downstream language renderer."""
    if scale_type in {"open_text", "duration"} or not distribution:
        return "open_expression"
    return "structured_expression"


def _compute_tradeoff_guidance(
    sampled_option: str,
    runner_up_option: Optional[str],
    distribution: Dict[str, float],
    rhetorical_habit: str = "direct",
) -> Optional[str]:
    """Generate tradeoff guidance when agent is torn between two options."""
    if not runner_up_option or not distribution:
        return None
    sampled_prob = distribution.get(sampled_option, 0.0)
    runner_prob = distribution.get(runner_up_option, 0.0)
    if runner_prob <= 0 or abs(sampled_prob - runner_prob) >= 0.15:
        return None

    a, b = sampled_option, runner_up_option
    habit = (rhetorical_habit or "direct").strip() or "direct"
    if habit == "list_pros_cons":
        return (
            f"You lean toward \"{a}\" but \"{b}\" is still plausible — in one short sentence, "
            f"name one upside of each, then commit to \"{a}\" as your answer."
        )
    if habit == "narrative":
        return (
            f"\"{a}\" fits your situation most days, but \"{b}\" still crosses your mind — "
            f"say that in one brief beat, then land on \"{a}\"."
        )
    if habit == "emotional_lead":
        return (
            f"You're pulled between \"{a}\" and \"{b}\" — lead with the feeling, "
            f"then state you are going with \"{a}\" for this question."
        )
    return (
        f"Between \"{a}\" and \"{b}\" you side with \"{a}\" — say it plainly in one line "
        f"without a long 'I'm torn' monologue."
    )


_FACTOR_CAUSAL_LANGUAGE = {
    "personality": "your personal habits and tendencies",
    "income": "your financial situation -- talk about prices, budget, affordability",
    "social": "what people around you do -- friends, family, neighbors",
    "location": "where you live and what's available nearby",
    "memory": "your past experiences with this topic",
    "behavioral": "your daily routines and behavior patterns",
    "belief": "your core beliefs and values about this",
}


def build_response_contract(
    *,
    interaction_mode: str = "survey",
    scale_type: str,
    sampled_option: str,
    distribution: Dict[str, float],
    ordered_options: Optional[List[str]] = None,
    latent_state: Optional[Any] = None,
    decision_trace: Optional[Dict[str, Any]] = None,
    beliefs: Optional[Any] = None,
    personality_summary: Optional[str] = None,
    rhetorical_habit: str = "direct",
) -> ResponseDecisionContract:
    """Build a unified Decision->Expression contract object."""
    expression_mode = infer_expression_mode(scale_type, distribution)
    max_prob = float(max(distribution.values())) if distribution else 0.0
    confidence_band = compute_confidence_band(max_prob)
    expected_score = compute_expected_score(distribution, ordered_options)

    if expected_score is None and latent_state is not None:
        try:
            vec = np.asarray(latent_state.to_vector(), dtype=np.float64)
            if vec.size > 0:
                expected_score = float(np.clip(vec.mean(), 0.0, 1.0))
        except Exception:
            expected_score = None

    latent_stance = latent_stance_from_score(expected_score, confidence_band)
    ranked = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True) if distribution else []
    runner_up_option = ranked[1][0] if len(ranked) > 1 else None
    dominance = (decision_trace or {}).get("dominance", {}) if isinstance(decision_trace, dict) else {}
    dominant_factor = str(dominance.get("factor", "")).strip() or None
    dominant_score = float(dominance.get("raw_scalar")) if dominant_factor and dominance.get("raw_scalar") is not None else None
    guidance_map = {}
    if isinstance(decision_trace, dict):
        spec = decision_trace.get("question_spec", {})
        if isinstance(spec, dict):
            guidance_map = spec.get("narrative_guidance", {}) or {}

    tradeoff = _compute_tradeoff_guidance(
        sampled_option, runner_up_option, distribution, rhetorical_habit=rhetorical_habit,
    )

    belief_stmts: Optional[List[str]] = None
    if beliefs is not None:
        try:
            from agents.belief_network import surface_top_beliefs
            belief_stmts = surface_top_beliefs(beliefs, top_n=3)
        except Exception:
            belief_stmts = None

    if dominant_factor:
        causal = _FACTOR_CAUSAL_LANGUAGE.get(dominant_factor)
        if causal:
            dominant_factor = f"{dominant_factor} ({causal})"

    # Decision latency: how quickly the agent would decide
    decision_latency = _compute_decision_latency(confidence_band)

    return ResponseDecisionContract(
        expression_mode=expression_mode,
        interaction_mode=interaction_mode,
        sampled_option=sampled_option,
        max_prob=max_prob,
        confidence_band=confidence_band,
        tone_selected=tone_for_confidence_band(confidence_band),
        expected_score=expected_score,
        latent_stance=latent_stance,
        runner_up_option=runner_up_option,
        dominant_factor=dominant_factor,
        dominant_score=dominant_score,
        narrative_guidance=str(guidance_map.get(sampled_option, "")).strip() or None,
        tradeoff_guidance=tradeoff,
        belief_statements=belief_stmts if belief_stmts else None,
        personality_summary=personality_summary,
        decision_latency=decision_latency,
    )


def _compute_decision_latency(confidence_band: str) -> str:
    """Map confidence band to decision latency category.

    High confidence + low fatigue -> instant (shorter, punchier)
    Medium -> normal
    Low confidence -> deliberate (longer, more hedging)
    """
    if confidence_band == "high":
        return "instant"
    elif confidence_band == "low":
        return "deliberate"
    return "normal"


_STRUCTURED_SCALES = frozenset({
    "likert", "frequency", "likelihood", "nps", "policy_support", "categorical", "numeric",
})

_OPTION_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new",
    "now", "old", "see", "two", "way", "who", "boy", "did", "she", "use", "her", "many",
    "than", "them", "these", "this", "that", "with", "from", "have", "been", "will",
    "your", "into", "more", "some", "such", "only", "also", "just", "like", "over",
    "i", "a", "an", "to", "of", "in", "on", "at", "is", "it", "or", "as", "be", "we",
})


def _significant_tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return [w for w in words if len(w) > 2 and w not in _OPTION_STOPWORDS]


def _compact_option_echo(opt: str, max_len: int) -> str:
    s = (opt or "").strip()
    if len(s) <= max_len:
        return s
    cut = s[:max_len]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    cut = cut.rstrip(",;:-–— ")
    return cut + "…" if cut else s[:max_len]


def _short_option_label(opt: str, max_len: int) -> str:
    """Use the title before a spaced dash (not bare hyphens inside words)."""
    s = (opt or "").strip()
    if not s:
        return s
    for sep in (" — ", " – ", " - "):
        if sep in s:
            head = s.split(sep, 1)[0].strip()
            if len(head) >= 3:
                return head
    return _compact_option_echo(s, max_len)


def _mentions_option(
    hay: str,
    needle: str,
    option_labels: Optional[List[str]],
    *,
    long_threshold: int,
    min_token_hits: int = 3,
) -> bool:
    h = hay.lower()
    n = (needle or "").strip().lower()
    if not n:
        return False
    if len(needle) <= long_threshold:
        if n in h:
            return True
    if needle.isdigit() and option_labels:
        for lbl in option_labels:
            m = re.match(r"^(\d{1,2})\s*[:=\-]\s*(.+)$", str(lbl).strip())
            if m and m.group(1) == needle:
                meaning = m.group(2).strip().lower()
                if meaning and meaning in h:
                    return True
    tokens = _significant_tokens(needle)
    if len(needle) > long_threshold and len(tokens) >= min_token_hits:
        hits = sum(1 for t in tokens if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", h))
        need = min(min_token_hits, max(2, len(tokens) // 3))
        if hits >= need:
            return True
    if len(needle) <= long_threshold:
        for token in needle.replace("/", " ").split():
            t = token.strip().strip('"').strip("'")
            if len(t) > 2 and t.lower() in h:
                return True
    if len(needle) > long_threshold and n in h:
        return True
    return False


def enforce_survey_response(
    text: str,
    *,
    scale_type: str,
    sampled_option: str,
    option_labels: Optional[List[str]] = None,
    interaction_mode: str = "survey",
    min_words_structured: int = 6,
) -> str:
    """Final gate on survey text: punctuation, option echo, minimum substance.

    Skips strict option-echo for open-ended scales and non-survey modes.
    Long categorical options use a compact echo + token-overlap detection so
    the full survey string is not pasted at the start of every answer.
    """
    raw = (text or "").strip()
    if not raw:
        return raw

    mode = (interaction_mode or "survey").strip().lower()
    if mode != "survey":
        if not raw.endswith((".", "!", "?")):
            return raw + "."
        return raw

    st = (scale_type or "categorical").strip().lower()
    if st in ("open_text", "duration"):
        if not raw.endswith((".", "!", "?")):
            return raw + "."
        return raw

    opt = (sampled_option or "").strip()
    if not opt:
        if not raw.endswith((".", "!", "?")):
            return raw + "."
        return raw

    settings = get_settings()
    long_th = int(getattr(settings, "enforce_survey_long_option_char_threshold", 72))
    compact_max = int(getattr(settings, "enforce_survey_compact_echo_max_chars", 70))
    long_opt = len(opt) > long_th
    echo_fragment = _short_option_label(opt, compact_max) if long_opt else opt

    mentioned = _mentions_option(raw, opt, option_labels, long_threshold=long_th)

    if not mentioned:
        if long_opt:
            if echo_fragment.endswith("…"):
                raw = f"I'm going with this choice: {echo_fragment} {raw}"
            else:
                raw = f"I'd go with {echo_fragment}. {raw}"
        else:
            raw = f"{opt}. {raw}"

    if not raw.endswith((".", "!", "?")):
        raw = raw + "."

    words = raw.split()
    min_need = min_words_structured
    if st in _STRUCTURED_SCALES:
        min_need = max(min_need, 10)

    if len(words) < min_need:
        if long_opt:
            tail = " Overall that's still my choice."
        else:
            tail = f" Overall I'm still {opt}."
        raw = raw.rstrip(".!? ") + tail
        if not raw.endswith((".", "!", "?")):
            raw = raw + "."
        words = raw.split()

    if len(words) < min_words_structured:
        raw = raw.rstrip(".!? ") + " That's how I see it."

    return raw
