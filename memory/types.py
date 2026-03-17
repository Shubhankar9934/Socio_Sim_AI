"""
Memory types: Episodic, Semantic, and Behavioral for agent memory store.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    BEHAVIORAL = "behavioral"


class EpisodicMemory(BaseModel):
    """Past experience: "I complained about parking last year"."""

    agent_id: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticMemory(BaseModel):
    """World knowledge: "Dubai Marina parking is difficult"."""

    agent_id: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BehavioralMemory(BaseModel):
    """Habit pattern: "orders food 3x/week"."""

    agent_id: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    weight: float = 1.0
    value: Optional[float] = None  # e.g. 3.0 for "3x per week"
    metadata: Dict[str, Any] = Field(default_factory=dict)


def memory_to_text(mem: "EpisodicMemory | SemanticMemory | BehavioralMemory") -> str:
    """Single line text for embedding and retrieval."""
    return mem.content
