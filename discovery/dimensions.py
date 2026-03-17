"""
Dynamic dimension discovery: embed questions, cluster, and name dimensions
using sentence-transformers + KMeans + LLM.

The 12 behavioral + 7 belief core dimensions remain the default set.
Discovery extends them with domain-specific extras when triggered.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredDimension:
    name: str
    description: str
    kind: str  # "behavioral" or "belief"
    representative_questions: List[str] = field(default_factory=list)


@dataclass
class DiscoveredDimensions:
    behavioral: List[DiscoveredDimension] = field(default_factory=list)
    belief: List[DiscoveredDimension] = field(default_factory=list)
    question_to_dimension: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def behavioral_names(self) -> List[str]:
        return [d.name for d in self.behavioral]

    @property
    def belief_names(self) -> List[str]:
        return [d.name for d in self.belief]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavioral": [
                {"name": d.name, "description": d.description,
                 "representative_questions": d.representative_questions}
                for d in self.behavioral
            ],
            "belief": [
                {"name": d.name, "description": d.description,
                 "representative_questions": d.representative_questions}
                for d in self.belief
            ],
            "question_to_dimension": self.question_to_dimension,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscoveredDimensions:
        behavioral = [
            DiscoveredDimension(name=d["name"], description=d["description"],
                                kind="behavioral",
                                representative_questions=d.get("representative_questions", []))
            for d in data.get("behavioral", [])
        ]
        belief = [
            DiscoveredDimension(name=d["name"], description=d["description"],
                                kind="belief",
                                representative_questions=d.get("representative_questions", []))
            for d in data.get("belief", [])
        ]
        return cls(
            behavioral=behavioral,
            belief=belief,
            question_to_dimension=data.get("question_to_dimension", {}),
        )


def _cache_path(domain_id: str) -> Path:
    return Path(f"data/domains/{domain_id}/discovered_dimensions.json")


def load_discovered_dimensions(domain_id: str) -> Optional[DiscoveredDimensions]:
    p = _cache_path(domain_id)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return DiscoveredDimensions.from_dict(data)
        except Exception:
            logger.warning("Failed to load discovered dimensions from %s", p)
    return None


def save_discovered_dimensions(domain_id: str, dims: DiscoveredDimensions) -> None:
    p = _cache_path(domain_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dims.to_dict(), indent=2), encoding="utf-8")


class DimensionDiscovery:
    """Discover behavioral and belief dimensions from a set of questions."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self._model_name = embedding_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required for dimension discovery. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def _cluster(
        self, embeddings: np.ndarray, n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.cluster import KMeans
        n_clusters = min(n_clusters, len(embeddings))
        if n_clusters < 1:
            return np.array([]), np.array([])
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        return labels, km.cluster_centers_

    async def _name_clusters_via_llm(
        self,
        questions: List[str],
        labels: np.ndarray,
        n_clusters: int,
        kind: str,
    ) -> List[DiscoveredDimension]:
        cluster_questions: Dict[int, List[str]] = {}
        for i, label in enumerate(labels):
            cluster_questions.setdefault(int(label), []).append(questions[i])

        dimensions: List[DiscoveredDimension] = []
        try:
            from llm.client import get_llm_client
            client = get_llm_client()
        except Exception:
            for cid in range(n_clusters):
                qs = cluster_questions.get(cid, [])
                dimensions.append(DiscoveredDimension(
                    name=f"{kind}_dim_{cid}",
                    description=f"Auto-discovered {kind} dimension {cid}",
                    kind=kind,
                    representative_questions=qs[:3],
                ))
            return dimensions

        for cid in range(n_clusters):
            qs = cluster_questions.get(cid, [])
            if not qs:
                continue
            sample = qs[:5]
            prompt = (
                f"These survey questions form a cluster:\n"
                + "\n".join(f'  - "{q}"' for q in sample)
                + f"\n\nName the underlying {kind} dimension that connects them. "
                f"Respond with JSON: {{\"name\": \"snake_case_name\", \"description\": \"one sentence\"}}"
            )
            try:
                resp = await client.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=100,
                )
                text = resp.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
                parsed = json.loads(text)
                dimensions.append(DiscoveredDimension(
                    name=parsed.get("name", f"{kind}_dim_{cid}"),
                    description=parsed.get("description", ""),
                    kind=kind,
                    representative_questions=sample,
                ))
            except Exception:
                dimensions.append(DiscoveredDimension(
                    name=f"{kind}_dim_{cid}",
                    description=f"Auto-discovered {kind} dimension {cid}",
                    kind=kind,
                    representative_questions=sample,
                ))
        return dimensions

    async def _assign_weights_via_llm(
        self,
        questions: List[str],
        dimension_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """For each question, get LLM-assigned weights over discovered dimensions."""
        result: Dict[str, Dict[str, float]] = {}
        dim_str = ", ".join(dimension_names)

        try:
            from llm.client import get_llm_client
            client = get_llm_client()
        except Exception:
            return result

        for q in questions:
            prompt = (
                f'Survey question: "{q}"\n\n'
                f"Rate relevance of each dimension from -1.0 to 1.0.\n"
                f"Dimensions: {dim_str}\n\n"
                f"Respond with ONLY a JSON object. Only include dimensions with |weight| >= 0.1."
            )
            try:
                resp = await client.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=300,
                )
                text = resp.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
                weights = {k: float(v) for k, v in json.loads(text).items()
                           if abs(float(v)) >= 0.1}
                result[q] = weights
            except Exception:
                pass
        return result

    async def discover_dimensions(
        self,
        questions: List[str],
        n_behavioral: int = 12,
        n_belief: int = 7,
    ) -> DiscoveredDimensions:
        if not questions:
            return DiscoveredDimensions()

        embeddings = self._embed(questions)

        behavioral_labels, _ = self._cluster(embeddings, n_behavioral)
        behavioral_dims = await self._name_clusters_via_llm(
            questions, behavioral_labels, min(n_behavioral, len(questions)), "behavioral"
        )

        belief_labels, _ = self._cluster(embeddings, n_belief)
        belief_dims = await self._name_clusters_via_llm(
            questions, belief_labels, min(n_belief, len(questions)), "belief"
        )

        all_dim_names = [d.name for d in behavioral_dims] + [d.name for d in belief_dims]
        q_to_dim = await self._assign_weights_via_llm(questions, all_dim_names)

        return DiscoveredDimensions(
            behavioral=behavioral_dims,
            belief=belief_dims,
            question_to_dimension=q_to_dim,
        )


def get_active_dimension_names(domain_id: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Return (behavioral_names, belief_names) including any discovered extras."""
    from agents.behavior import DIMENSION_NAMES as CORE_BEHAVIORAL
    from agents.belief_network import BELIEF_DIMENSIONS as CORE_BELIEF

    if domain_id is None:
        try:
            from config.settings import get_settings
            domain_id = get_settings().domain_id
        except Exception:
            pass

    extra_b: List[str] = []
    extra_bl: List[str] = []
    if domain_id:
        discovered = load_discovered_dimensions(domain_id)
        if discovered:
            extra_b = [n for n in discovered.behavioral_names if n not in CORE_BEHAVIORAL]
            extra_bl = [n for n in discovered.belief_names if n not in CORE_BELIEF]

    return list(CORE_BEHAVIORAL) + extra_b, list(CORE_BELIEF) + extra_bl
