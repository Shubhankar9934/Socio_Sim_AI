"""
ChromaDB-backed vector memory store per agent.
Supports add_memory, recall(query, top_k), get_recent(n), and
exponential decay of memory weights over time.
"""

import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from memory.types import BehavioralMemory, EpisodicMemory, MemoryType, SemanticMemory

# Optional ChromaDB; fallback to in-memory list
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class MemoryStore:
    """
    Vector store for agent memories. Uses ChromaDB if available and persist_dir set,
    otherwise in-memory list with simple text matching for recall.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_prefix: str = "agent_memory",
    ):
        self._persist_dir = persist_dir or os.environ.get("CHROMA_PERSIST_DIR", "")
        self._prefix = collection_prefix
        self._client: Any = None
        self._collections: Dict[str, Any] = {}
        self._in_memory: Dict[str, List[Dict[str, Any]]] = {}
        self._use_chroma = HAS_CHROMADB and bool(self._persist_dir)
        if self._use_chroma:
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

    def _collection_name(self, agent_id: str) -> str:
        # Chroma collection names: alphanumeric and underscore only
        safe_id = "".join(c if c.isalnum() or c == "_" else "_" for c in agent_id)
        return f"{self._prefix}_{safe_id}"

    def _get_or_create_collection(self, agent_id: str):
        if self._use_chroma and self._client:
            name = self._collection_name(agent_id)
            if name not in self._collections:
                self._collections[name] = self._client.get_or_create_collection(
                    name=name, metadata={"agent_id": agent_id}
                )
            return self._collections[name]
        return None

    def add_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        initial_weight: float = 1.0,
    ) -> None:
        """Add one memory for an agent."""
        meta = metadata or {}
        meta["memory_type"] = memory_type.value
        meta["value"] = value
        meta["timestamp"] = datetime.utcnow().isoformat()
        meta["weight"] = initial_weight
        doc = {"content": content, "metadata": meta}

        if self._use_chroma:
            coll = self._get_or_create_collection(agent_id)
            if coll is not None:
                # Chroma expects id, document, metadatas
                import uuid
                coll.add(
                    ids=[str(uuid.uuid4())],
                    documents=[content],
                    metadatas=[meta],
                )
                return
        # In-memory fallback
        if agent_id not in self._in_memory:
            self._in_memory[agent_id] = []
        self._in_memory[agent_id].append(doc)

    def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[str]:
        """Retrieve top_k relevant memories for the agent by semantic similarity."""
        if self._use_chroma:
            coll = self._get_or_create_collection(agent_id)
            if coll is not None:
                try:
                    results = coll.query(query_texts=[query], n_results=min(top_k, 50))
                    if results and results["documents"] and len(results["documents"]) > 0:
                        return list(results["documents"][0])
                except Exception:
                    pass
            return []
        # In-memory: keyword match weighted by memory weight (decayed)
        memories = self._in_memory.get(agent_id, [])
        q = query.lower()
        scored = []
        for m in memories:
            c = m.get("content", "").lower()
            relevance = sum(1 for w in q.split() if w in c)
            mem_weight = m.get("metadata", {}).get("weight", 1.0)
            score = relevance * mem_weight
            scored.append((score, m["content"]))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k] if s[0] > 0]

    def get_recent(self, agent_id: str, n: int = 10) -> List[str]:
        """Return n most recent memories (by timestamp in metadata)."""
        if self._use_chroma:
            coll = self._get_or_create_collection(agent_id)
            if coll is not None:
                try:
                    results = coll.get(include=["documents", "metadatas"])
                    if results and results["documents"]:
                        docs = results["documents"]
                        metas = results.get("metadatas") or []
                        combined = list(zip(docs, metas))
                        combined.sort(
                            key=lambda x: x[1].get("timestamp", ""),
                            reverse=True,
                        )
                        return [c[0] for c in combined[:n]]
                except Exception:
                    pass
            return []
        memories = self._in_memory.get(agent_id, [])
        sorted_m = sorted(
            memories,
            key=lambda m: m.get("metadata", {}).get("timestamp", ""),
            reverse=True,
        )
        return [m["content"] for m in sorted_m[:n]]


    def decay_all(
        self,
        lambda_: float = 0.05,
        prune_threshold: float = 0.01,
    ) -> None:
        """Apply exponential decay to all in-memory memory weights.

        Memories whose weight drops below *prune_threshold* are removed.
        Call once per simulation day.
        """
        factor = math.exp(-lambda_)
        for agent_id in list(self._in_memory.keys()):
            surviving = []
            for m in self._in_memory[agent_id]:
                w = m.get("metadata", {}).get("weight", 1.0) * factor
                m.setdefault("metadata", {})["weight"] = w
                if w >= prune_threshold:
                    surviving.append(m)
            self._in_memory[agent_id] = surviving


# Module-level default store (in-memory if no CHROMA_PERSIST_DIR)
_default_store: Optional[MemoryStore] = None


def get_memory_store(persist_dir: Optional[str] = None) -> MemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore(persist_dir=persist_dir)
    return _default_store
