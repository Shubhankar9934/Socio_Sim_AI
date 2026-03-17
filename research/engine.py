"""
Shared Research Engine.

Performs web research, extracts structured facts, and builds context documents
that are shared across all archetypes during reasoning. One research call per
question/event, not per agent.

Supports pluggable search backends (tavily, serpapi, or none for offline).
Results are cached to avoid redundant lookups.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """A single extracted fact with source provenance."""

    statement: str
    source: str = ""
    confidence: float = 0.8


@dataclass
class ResearchContext:
    """Compiled research context for a question or event."""

    query: str
    facts: List[Fact] = field(default_factory=list)
    summary: str = ""
    raw_sources: List[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Format for LLM prompt injection."""
        if not self.facts and not self.summary:
            return ""
        lines = ["[Research Context]"]
        if self.summary:
            lines.append(self.summary)
        if self.facts:
            lines.append("Key facts:")
            for f in self.facts[:10]:
                lines.append(f"  - {f.statement}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "facts": [{"statement": f.statement, "source": f.source, "confidence": f.confidence} for f in self.facts],
            "summary": self.summary,
        }


class ResearchCache:
    """Hash-keyed JSON cache for research results."""

    def __init__(self, path: str = "data/research_cache.json"):
        self._path = Path(path)
        self._cache: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._cache = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[ResearchContext]:
        data = self._cache.get(self._key(query))
        if data is None:
            return None
        facts = [Fact(**f) for f in data.get("facts", [])]
        return ResearchContext(
            query=data.get("query", query),
            facts=facts,
            summary=data.get("summary", ""),
        )

    def put(self, ctx: ResearchContext) -> None:
        self._cache[self._key(ctx.query)] = ctx.to_dict()
        try:
            self._save()
        except OSError:
            pass


class ResearchEngine:
    """Shared research engine with caching and pluggable backends."""

    def __init__(
        self,
        provider: str = "none",
        cache_path: str = "data/research_cache.json",
        openai_api_key: str = "",
    ):
        self._provider = provider
        self._cache = ResearchCache(cache_path)
        self._api_key = openai_api_key

    def research_question(self, question: str) -> ResearchContext:
        """Research a question. Returns cached result if available."""
        cached = self._cache.get(question)
        if cached is not None:
            return cached

        if self._provider == "none":
            ctx = ResearchContext(query=question, summary="No research provider configured.")
            self._cache.put(ctx)
            return ctx

        ctx = self._do_search(question)
        self._cache.put(ctx)
        return ctx

    def build_context(self, events: List[Dict[str, Any]]) -> Optional[ResearchContext]:
        """Build combined research context from a batch of events."""
        if not events:
            return None

        queries = []
        for ev in events:
            name = ev.get("name") or ev.get("type", "")
            if name:
                queries.append(name)

        if not queries:
            return None

        combined_query = "; ".join(queries[:5])
        return self.research_question(combined_query)

    def _do_search(self, query: str) -> ResearchContext:
        """Execute web search via the configured provider."""
        if self._provider == "tavily":
            return self._search_tavily(query)
        elif self._provider == "serpapi":
            return self._search_serpapi(query)
        else:
            return ResearchContext(query=query, summary=f"Unknown provider: {self._provider}")

    def _search_tavily(self, query: str) -> ResearchContext:
        """Search via Tavily API."""
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self._api_key)
            result = client.search(query, max_results=5)
            facts = []
            sources = []
            for r in result.get("results", []):
                facts.append(Fact(
                    statement=r.get("content", "")[:200],
                    source=r.get("url", ""),
                ))
                sources.append(r.get("url", ""))
            summary = result.get("answer", "")
            return ResearchContext(query=query, facts=facts, summary=summary, raw_sources=sources)
        except Exception as e:
            logger.warning("Tavily search failed: %s", e)
            return ResearchContext(query=query, summary=f"Search failed: {e}")

    def _search_serpapi(self, query: str) -> ResearchContext:
        """Search via SerpAPI."""
        try:
            import serpapi
            params = {"q": query, "api_key": self._api_key, "num": 5}
            results = serpapi.search(params)
            facts = []
            for r in results.get("organic_results", []):
                facts.append(Fact(
                    statement=r.get("snippet", "")[:200],
                    source=r.get("link", ""),
                ))
            return ResearchContext(query=query, facts=facts)
        except Exception as e:
            logger.warning("SerpAPI search failed: %s", e)
            return ResearchContext(query=query, summary=f"Search failed: {e}")
