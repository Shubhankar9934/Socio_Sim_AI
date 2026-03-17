"""
Action type inference: classify survey questions into universal action types
using rule-based patterns with LLM fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from agents.actions import ActionTemplate

logger = logging.getLogger(__name__)

_CACHE_PATH = Path("data/action_template_cache.json")

_PATTERN_RULES = [
    (["how often", "how frequently", "per week", "per month", "times"],
     ActionTemplate("frequency", "behavior", "ordinal")),
    (["do you support", "in favor", "should the government", "agree or disagree"],
     ActionTemplate("support", "policy", "ordinal")),
    (["would you vote", "who would you", "which candidate"],
     ActionTemplate("choose", "candidate", "binary")),
    (["rate your", "how satisfied", "satisfaction", "how happy", "nps", "score"],
     ActionTemplate("rate", "experience", "continuous")),
    (["would you switch", "would you try", "willing to adopt", "start using"],
     ActionTemplate("adopt", "product", "binary")),
    (["would you stop", "quit", "cancel", "give up"],
     ActionTemplate("reject", "service", "binary")),
    (["how much would you invest", "allocate", "budget"],
     ActionTemplate("invest", "investment", "continuous")),
    (["would you move", "relocate", "migrate"],
     ActionTemplate("migrate", "location", "binary")),
    (["would you protest", "demonstrate", "rally"],
     ActionTemplate("protest", "policy", "binary")),
    (["would you comply", "follow the rule", "obey"],
     ActionTemplate("comply", "norm", "binary")),
]


def _load_cache() -> Dict[str, dict]:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: Dict[str, dict]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


_ACTION_CACHE: Dict[str, dict] = _load_cache()


def _cache_key(question: str) -> str:
    norm = " ".join(question.lower().split())
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def infer_action_type_rule(question: str) -> Optional[ActionTemplate]:
    norm = question.lower()
    for patterns, template in _PATTERN_RULES:
        if any(p in norm for p in patterns):
            return template
    return None


class ActionModelBuilder:
    """Infer action types from questions using rules + LLM fallback."""

    async def infer_action_type(
        self,
        question: str,
        options: Optional[List[str]] = None,
    ) -> ActionTemplate:
        key = _cache_key(question)
        if key in _ACTION_CACHE:
            return ActionTemplate.from_dict(_ACTION_CACHE[key])

        rule_match = infer_action_type_rule(question)
        if rule_match is not None:
            _ACTION_CACHE[key] = rule_match.to_dict()
            _save_cache(_ACTION_CACHE)
            return rule_match

        return await self._infer_via_llm(question, options, key)

    async def _infer_via_llm(
        self, question: str, options: Optional[List[str]], key: str
    ) -> ActionTemplate:
        try:
            from llm.client import get_llm_client

            action_types = ", ".join([
                "frequency", "adopt", "reject", "support", "oppose",
                "rate", "choose", "increase", "decrease", "invest",
                "migrate", "comply", "protest",
            ])
            targets = ", ".join([
                "service", "product", "policy", "candidate",
                "belief", "behavior", "location", "investment", "norm",
            ])
            opt_str = f"\nOptions: {options}" if options else ""
            prompt = (
                f'Survey question: "{question}"{opt_str}\n\n'
                f"Classify into action_type ({action_types}) and target ({targets}).\n"
                f'Respond with JSON: {{"action_type": "...", "target": "...", '
                f'"intensity_scale": "ordinal|binary|continuous"}}'
            )
            client = get_llm_client()
            resp = await client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=100,
            )
            text = resp.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            parsed = json.loads(text)
            template = ActionTemplate.from_dict(parsed)
            _ACTION_CACHE[key] = template.to_dict()
            _save_cache(_ACTION_CACHE)
            return template
        except Exception:
            fallback = ActionTemplate("choose", "behavior", "ordinal")
            _ACTION_CACHE[key] = fallback.to_dict()
            _save_cache(_ACTION_CACHE)
            return fallback
