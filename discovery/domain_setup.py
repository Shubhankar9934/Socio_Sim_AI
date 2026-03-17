"""
Domain auto-setup: given a name, description, and sample questions,
auto-generate a complete domain config using LLM + dimension discovery.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DomainAutoSetup:
    """Auto-generate a domain config from description and sample questions."""

    async def setup_domain(
        self,
        domain_name: str,
        description: str,
        sample_questions: List[str],
        city_name: str = "",
        currency: str = "USD",
        reference_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new domain and return the domain_id.

        Steps:
          1. Use LLM to generate domain.json skeleton
          2. Discover dimensions from questions
          3. Generate question models with LLM
          4. If reference_data provided: save reference distributions
          5. Save to data/domains/{domain_id}/
        """
        domain_id = re.sub(r"[^a-z0-9_]", "_", domain_name.lower()).strip("_")
        if not domain_id:
            domain_id = "custom"

        domain_dir = Path(f"data/domains/{domain_id}")
        domain_dir.mkdir(parents=True, exist_ok=True)

        domain_config = await self._generate_config_via_llm(
            domain_name, description, sample_questions, city_name, currency
        )

        (domain_dir / "domain.json").write_text(
            json.dumps(domain_config, indent=2), encoding="utf-8"
        )

        discovered = await self._discover_dimensions(sample_questions, domain_id)
        if discovered:
            domain_config["discovered_dimensions"] = True

        question_models = await self._generate_question_models(sample_questions)
        if question_models:
            domain_config["question_model_overrides"] = question_models
            (domain_dir / "domain.json").write_text(
                json.dumps(domain_config, indent=2), encoding="utf-8"
            )

        if reference_data:
            (domain_dir / "reference_distributions.json").write_text(
                json.dumps(reference_data, indent=2), encoding="utf-8"
            )

        demographics_stub = {
            "age": {"18-24": 0.2, "25-34": 0.3, "35-44": 0.25, "45-54": 0.15, "55+": 0.1},
            "nationality": {"local": 0.5, "other": 0.5},
            "income": {"low": 0.3, "medium": 0.4, "high": 0.3},
            "location": {"urban": 0.6, "suburban": 0.3, "rural": 0.1},
            "occupation": {"employed": 0.7, "student": 0.15, "other": 0.15},
        }
        if not (domain_dir / "demographics.json").exists():
            (domain_dir / "demographics.json").write_text(
                json.dumps(demographics_stub, indent=2), encoding="utf-8"
            )

        logger.info("Domain '%s' created at %s", domain_id, domain_dir)
        return domain_id

    async def _generate_config_via_llm(
        self,
        domain_name: str,
        description: str,
        sample_questions: List[str],
        city_name: str,
        currency: str,
    ) -> Dict[str, Any]:
        """Use LLM to generate the domain.json skeleton."""
        base_config: Dict[str, Any] = {
            "city_id": domain_name.lower().replace(" ", "_"),
            "city_name": city_name or domain_name,
            "currency": currency,
            "districts": [],
            "nationalities": [],
            "premium_areas": [],
            "topic_keywords": {},
            "domain_keywords": {},
            "services": {},
            "price_levels": {},
            "system_prompts": [],
            "archetype_hints": {},
            "cultural_hints": {},
            "frequency_interpretation": {},
            "lifestyle_keywords": [],
            "location_terms": [],
        }

        try:
            from llm.client import get_llm_client
            client = get_llm_client()

            q_str = "\n".join(f"  - {q}" for q in sample_questions[:10])
            prompt = (
                f"Generate a domain configuration for a synthetic society simulation.\n\n"
                f"Domain: {domain_name}\n"
                f"Description: {description}\n"
                f"Sample questions:\n{q_str}\n\n"
                f"Generate JSON with these fields:\n"
                f"- topic_keywords: dict of topic -> list of relevant keywords\n"
                f"- services: dict of service_name -> availability (0-1)\n"
                f"- system_prompts: list of 2-3 system prompt strings for LLM agents\n"
                f"- lifestyle_keywords: list of relevant lifestyle terms\n\n"
                f"Respond with ONLY valid JSON."
            )
            resp = await client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=500,
            )
            text = resp.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            generated = json.loads(text)
            for k, v in generated.items():
                if k in base_config:
                    base_config[k] = v
        except Exception:
            logger.warning("LLM domain config generation failed; using defaults")

        return base_config

    async def _discover_dimensions(
        self, questions: List[str], domain_id: str
    ) -> bool:
        """Run dimension discovery and save results."""
        try:
            from discovery.dimensions import DimensionDiscovery, save_discovered_dimensions
            disc = DimensionDiscovery()
            result = await disc.discover_dimensions(questions)
            if result.behavioral or result.belief:
                save_discovered_dimensions(domain_id, result)
                return True
        except Exception:
            logger.warning("Dimension discovery failed during domain setup")
        return False

    async def _generate_question_models(
        self, questions: List[str]
    ) -> Dict[str, Any]:
        """Generate question model overrides via LLM."""
        models: Dict[str, Any] = {}
        try:
            from llm.client import get_llm_client
            client = get_llm_client()

            q_str = "\n".join(f"  - {q}" for q in questions[:10])
            prompt = (
                f"For each survey question, suggest a question model with:\n"
                f"- key: snake_case identifier\n"
                f"- scale: list of answer options\n"
                f"- scale_type: frequency|likert|categorical|binary\n\n"
                f"Questions:\n{q_str}\n\n"
                f'Respond with JSON: {{"question_text": {{"key": "...", "scale": [...], "scale_type": "..."}}}}'
            )
            resp = await client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=600,
            )
            text = resp.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            models = json.loads(text)
        except Exception:
            logger.warning("Question model generation failed")
        return models
