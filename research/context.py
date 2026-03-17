"""
Research Context Integration.

Merges shared research context with archetype-specific persona context
before LLM reasoning calls.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from research.engine import ResearchContext


def enrich_archetype_context(
    archetype_context: Dict[str, Any],
    research_ctx: Optional[ResearchContext],
) -> Dict[str, Any]:
    """Merge research context into an archetype's reasoning context.

    The research_ctx is shared across ALL archetypes. Each archetype gets
    the same factual grounding but interprets it through their own persona.
    """
    enriched = dict(archetype_context)

    if research_ctx is not None:
        prompt_text = research_ctx.to_prompt_text()
        if prompt_text:
            enriched["research_context"] = prompt_text
        enriched["research_facts"] = [
            f.statement for f in research_ctx.facts[:8]
        ]

    return enriched
