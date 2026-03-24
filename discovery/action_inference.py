"""
Action type inference: classify survey questions into universal action types
using rule-based patterns with LLM fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.actions import ActionTemplate
from agents.intent_router import build_turn_understanding_hybrid, infer_action_template_rule

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
    action_type, target, intensity_scale = infer_action_template_rule(question)
    if not action_type or not target or not intensity_scale:
        return None
    return ActionTemplate(action_type, target, intensity_scale)


@dataclass
class ActionInferenceResult:
    template: ActionTemplate
    rule_template: Optional[Dict[str, Any]]
    hybrid_understanding: Optional[Dict[str, Any]]
    source: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template": self.template.to_dict(),
            "rule_template": self.rule_template,
            "hybrid_understanding": self.hybrid_understanding,
            "source": self.source,
            "confidence": self.confidence,
        }


class ActionModelBuilder:
    """Infer action types from questions using rules + LLM fallback."""

    def __init__(self) -> None:
        self.last_result: Optional[ActionInferenceResult] = None

    async def infer_action_type(
        self,
        question: str,
        options: Optional[List[str]] = None,
    ) -> ActionTemplate:
        result = await self.infer_action_type_result(question, options=options)
        return result.template

    async def infer_action_type_result(
        self,
        question: str,
        options: Optional[List[str]] = None,
    ) -> ActionInferenceResult:
        key = _cache_key(question)
        if key in _ACTION_CACHE:
            cached_template = ActionTemplate.from_dict(_ACTION_CACHE[key])
            self.last_result = ActionInferenceResult(
                template=cached_template,
                rule_template=None,
                hybrid_understanding=None,
                source="cache",
                confidence=0.99,
            )
            return self.last_result

        rule_match = infer_action_type_rule(question)
        rule_template = rule_match.to_dict() if rule_match is not None else None
        hybrid_understanding = await build_turn_understanding_hybrid(question, options=options)
        hybrid_template = ActionTemplate(
            action_type=hybrid_understanding.action_type_candidate or "choose",
            target=hybrid_understanding.target_candidate or "behavior",
            intensity_scale=hybrid_understanding.intensity_scale_candidate or "ordinal",
        )

        if rule_match is not None and (
            rule_match.action_type == hybrid_template.action_type
            and rule_match.target == hybrid_template.target
            and rule_match.intensity_scale == hybrid_template.intensity_scale
        ):
            _ACTION_CACHE[key] = rule_match.to_dict()
            _save_cache(_ACTION_CACHE)
            self.last_result = ActionInferenceResult(
                template=rule_match,
                rule_template=rule_template,
                hybrid_understanding=hybrid_understanding.to_dict(),
                source="rule_llm_agree",
                confidence=max(hybrid_understanding.final_confidence, 0.8),
            )
            return self.last_result

        if hybrid_understanding.llm_confidence >= 0.8:
            _ACTION_CACHE[key] = hybrid_template.to_dict()
            _save_cache(_ACTION_CACHE)
            self.last_result = ActionInferenceResult(
                template=hybrid_template,
                rule_template=rule_template,
                hybrid_understanding=hybrid_understanding.to_dict(),
                source="hybrid",
                confidence=hybrid_understanding.final_confidence,
            )
            return self.last_result

        template = await self._infer_via_llm(question, options, key)
        self.last_result = ActionInferenceResult(
            template=template,
            rule_template=rule_template,
            hybrid_understanding=hybrid_understanding.to_dict(),
            source="fallback_llm",
            confidence=max(hybrid_understanding.llm_confidence, 0.6),
        )
        return self.last_result

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
