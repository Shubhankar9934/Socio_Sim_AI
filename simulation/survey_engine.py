"""
Multi-question survey engine with persistent agent state.

Orchestrates sequential rounds where all agents answer one question per round.
Between rounds the engine runs social influence diffusion and belief updates
so that earlier answers naturally bias later decisions.

Uses the existing single-question ``run_survey()`` for each round's async
LLM batching and archetype compression.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

import numpy as np

from simulation.event_queue import EventDrivenScheduler, SimEvent
from simulation.orchestrator import run_survey


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class SurveyEngineConfig:
    """Tunables for a multi-question survey session."""

    use_archetypes: bool = False
    social_influence_between_rounds: bool = True
    summarize_every: int = 5
    social_warmup_steps: int = 2
    max_concurrent: Optional[int] = None
    social_neighbor_sample_k: int = 15
    archetype_noise_std: float = 0.05
    narrative_budget: float = 0.20
    recluster_every: int = 10
    archetype_aggregation: str = "median"
    narrative_templates_per_archetype: int = 3


# ------------------------------------------------------------------
# Round result
# ------------------------------------------------------------------

@dataclass
class RoundResult:
    round_idx: int
    question: str
    question_id: str
    responses: List[Dict[str, Any]]
    elapsed_seconds: float = 0.0


# ------------------------------------------------------------------
# Session result
# ------------------------------------------------------------------

@dataclass
class SurveySessionResult:
    session_id: str
    questions: List[str]
    rounds: List[RoundResult] = field(default_factory=list)
    total_responses: int = 0
    elapsed_seconds: float = 0.0
    status: str = "completed"


# ------------------------------------------------------------------
# Progress callback type
# ------------------------------------------------------------------

ProgressCallback = Callable[[int, int, str, str, List[Dict[str, Any]]], Coroutine[Any, Any, None]]


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class SurveyEngine:
    """Run a multi-question survey across a persistent agent population.

    Parameters
    ----------
    agents : list[dict]
        The shared ``agents_store`` list; each entry has ``persona``,
        ``state``, and optionally ``environment`` keys.
    social_graph : networkx.Graph | None
        Barabasi-Albert social network (may be None).
    config : SurveyEngineConfig
        Session configuration.
    event_scheduler : EventDrivenScheduler | None
        Optional event queue for interleaving world events between rounds.
    """

    def __init__(
        self,
        agents: List[Dict[str, Any]],
        social_graph: Optional[Any] = None,
        config: Optional[SurveyEngineConfig] = None,
        event_scheduler: Optional[EventDrivenScheduler] = None,
    ) -> None:
        self.agents = agents
        self.social_graph = social_graph
        self.config = config or SurveyEngineConfig()
        self.event_scheduler = event_scheduler
        self._progress_callbacks: List[ProgressCallback] = []
        self._session_progress: Dict[str, Any] = {}
        self._archetype_states: Optional[Dict[int, Any]] = None
        self._labels: Optional[List[int]] = None
        self._prev_centroids: Optional[np.ndarray] = None
        self._prev_inertia: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_progress(self, callback: ProgressCallback) -> None:
        """Register an async callback invoked after each round completes."""
        self._progress_callbacks.append(callback)

    async def run(
        self,
        questions: List[Dict[str, Any]],
    ) -> SurveySessionResult:
        """Execute all survey rounds via batched event scheduling.

        Instead of looping imperatively, each round and inter-round step is
        placed on the ``EventDrivenScheduler`` timeline as a ``SimEvent``.
        This lets world events, social diffusion, and survey rounds coexist
        on a single unified timeline processed in ``O(log n)`` per event.

        Parameters
        ----------
        questions : list of dicts with ``question`` and optional ``question_id``

        Returns
        -------
        SurveySessionResult
        """
        session_id = str(uuid.uuid4())
        total_rounds = len(questions)
        t0 = time.monotonic()

        self._session_progress = {
            "session_id": session_id,
            "total_rounds": total_rounds,
            "current_round": 0,
            "status": "running",
            "completed_questions": [],
        }

        rounds: List[RoundResult] = []
        total_responses = 0

        # Build archetype states once if archetype mode is enabled
        if self.config.use_archetypes and len(self.agents) > 1:
            from simulation.archetypes import build_archetype_states
            from config.settings import get_settings as _get_settings
            _s = _get_settings()
            self._archetype_states, self._labels, self._prev_centroids, self._prev_inertia = build_archetype_states(
                self.agents, _s.archetype_count,
                aggregation=self.config.archetype_aggregation,
            )

        # Use a dedicated scheduler so pre-existing external events
        # (on self.event_scheduler) are not lost.
        scheduler = EventDrivenScheduler()

        # Normalise question list
        resolved_questions: List[Dict[str, Any]] = []
        for q in questions:
            resolved_questions.append({
                "question": q["question"],
                "question_id": q.get("question_id") or str(uuid.uuid4()),
                "options": q.get("options"),
            })

        # --- Schedule batched events on a unified timeline ---------------
        for i, rq in enumerate(resolved_questions):
            scheduler.schedule(SimEvent(
                time=float(i),
                agent_id="__batch__",
                event_type="survey_round",
                payload={
                    "question": rq["question"],
                    "question_id": rq["question_id"],
                    "options": rq.get("options"),
                    "round_idx": i,
                },
            ))
            if i < total_rounds - 1:
                if self.config.social_influence_between_rounds:
                    scheduler.schedule(SimEvent(
                        time=float(i) + 0.3,
                        agent_id="__batch__",
                        event_type="social_diffusion",
                        payload={},
                    ))
                scheduler.schedule(SimEvent(
                    time=float(i) + 0.5,
                    agent_id="__batch__",
                    event_type="memory_summarization",
                    payload={"round_idx": i},
                ))

        # --- Register handlers -------------------------------------------
        async def _handle_survey_round(event: SimEvent, _agents: Dict[str, Any]) -> None:
            payload = event.payload
            round_idx = payload["round_idx"]
            question_text = payload["question"]
            question_id = payload["question_id"]

            self._session_progress["current_round"] = round_idx

            # Periodic reclustering so archetypes track evolving behaviour
            if (
                self._archetype_states is not None
                and round_idx > 0
                and self.config.recluster_every > 0
                and round_idx % self.config.recluster_every == 0
            ):
                from simulation.archetypes import build_archetype_states as _rebuild
                from config.settings import get_settings as _gs
                self._archetype_states, self._labels, self._prev_centroids, self._prev_inertia = _rebuild(
                    self.agents, _gs().archetype_count,
                    aggregation=self.config.archetype_aggregation,
                    use_latent_features=True,
                    prev_centroids=self._prev_centroids,
                    prev_inertia=self._prev_inertia,
                )

            round_result = await self._run_round(
                question_text,
                question_id,
                round_idx,
                payload.get("options"),
            )
            rounds.append(round_result)
            nonlocal total_responses
            total_responses += len(round_result.responses)
            self._session_progress["completed_questions"].append(question_id)

            await self._emit_progress(
                round_idx, total_rounds, session_id,
                question_text, round_result.responses,
            )

        async def _handle_social_diffusion(event: SimEvent, _agents: Dict[str, Any]) -> None:
            self._apply_social_influence()

        async def _handle_memory_summarization(event: SimEvent, _agents: Dict[str, Any]) -> None:
            round_idx = (event.payload or {}).get("round_idx", 0)
            if (
                self.config.summarize_every > 0
                and (round_idx + 1) % self.config.summarize_every == 0
            ):
                self._summarize_all_memories()
            # Also drain any world events on the external scheduler
            if self.event_scheduler is not None:
                await self.event_scheduler.process_until_async(
                    float(round_idx + 1), self._agents_dict(),
                )

        scheduler.register("survey_round", _handle_survey_round)
        scheduler.register("social_diffusion", _handle_social_diffusion)
        scheduler.register("memory_summarization", _handle_memory_summarization)

        # --- Process the entire timeline ---------------------------------
        agents_dict = self._agents_dict()
        await scheduler.process_all_async(agents_dict)

        elapsed = time.monotonic() - t0
        self._session_progress["status"] = "completed"

        return SurveySessionResult(
            session_id=session_id,
            questions=[rq["question"] for rq in resolved_questions],
            rounds=rounds,
            total_responses=total_responses,
            elapsed_seconds=round(elapsed, 2),
        )

    def get_progress(self) -> Dict[str, Any]:
        """Return the current session progress snapshot."""
        return dict(self._session_progress)

    # ------------------------------------------------------------------
    # Private: run one round
    # ------------------------------------------------------------------

    async def _run_round(
        self,
        question: str,
        question_id: str,
        round_idx: int,
        options: Optional[List[str]] = None,
    ) -> RoundResult:
        t0 = time.monotonic()

        if self._archetype_states is not None and self._labels is not None:
            from simulation.archetype_runner import run_archetype_round
            from simulation.archetypes import refresh_archetype_states

            if round_idx > 0:
                refresh_archetype_states(
                    self._archetype_states, self.agents, self._labels,
                    aggregation=self.config.archetype_aggregation,
                )

            responses = await run_archetype_round(
                archetype_states=self._archetype_states,
                labels=self._labels,
                agents=self.agents,
                question=question,
                question_id=question_id,
                options=options,
                max_concurrent=self.config.max_concurrent,
                noise_std=self.config.archetype_noise_std,
                narrative_budget=self.config.narrative_budget,
                narrative_templates_per_archetype=self.config.narrative_templates_per_archetype,
            )
        else:
            responses = await run_survey(
                self.agents,
                question=question,
                question_id=question_id,
                options=options,
                think_fn=None,
                use_archetypes=self.config.use_archetypes,
                max_concurrent=self.config.max_concurrent,
            )

        elapsed = time.monotonic() - t0
        return RoundResult(
            round_idx=round_idx,
            question=question,
            question_id=question_id,
            responses=responses,
            elapsed_seconds=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Private: inter-round social influence
    # ------------------------------------------------------------------

    def _apply_social_influence(self) -> None:
        """Run social diffusion -- hybrid if archetypes are active."""
        if self.social_graph is None or len(self.agents) < 2:
            return
        has_states = any(a.get("state") is not None for a in self.agents)
        if not has_states:
            return

        if self._archetype_states and self._labels:
            self._apply_hybrid_diffusion()
        else:
            self._apply_sampled_diffusion()

    def _apply_sampled_diffusion(self) -> None:
        """Diffuse on sampled neighbor subgraph (non-archetype path)."""
        from agents.vectorized import (
            build_belief_matrix,
            build_trait_matrix,
            vectorized_belief_diffusion,
            vectorized_social_influence,
            write_belief_matrix,
            write_trait_matrix,
        )
        from social.network import (
            normalize_adjacency,
            sample_neighbors_adjacency,
            to_sparse_adjacency,
        )

        sparse_adj = to_sparse_adjacency(self.social_graph)
        k = self.config.social_neighbor_sample_k

        for _ in range(self.config.social_warmup_steps):
            sampled = sample_neighbors_adjacency(sparse_adj, k=k)
            adj_norm = normalize_adjacency(sampled)
            mat = build_trait_matrix(self.agents)
            bmat = build_belief_matrix(self.agents)
            mat = vectorized_social_influence(mat, adj_norm)
            bmat = vectorized_belief_diffusion(bmat, adj_norm)
            mat = np.clip(mat, 0.0, 1.0)
            bmat = np.clip(bmat, 0.0, 1.0)
            write_trait_matrix(self.agents, mat)
            write_belief_matrix(self.agents, bmat)

    def _apply_hybrid_diffusion(self) -> None:
        """Two-phase diffusion: intra-cluster local + inter-archetype coarsened.

        Phase 1 preserves within-cluster variation by diffusing only among
        cluster members.  Phase 2 propagates influence between clusters via
        the coarsened archetype graph, then distributes deltas back to members.
        """
        from agents.vectorized import (
            build_belief_matrix,
            build_trait_matrix,
            vectorized_belief_diffusion,
            vectorized_social_influence,
            write_belief_matrix,
            write_trait_matrix,
        )
        from social.network import (
            normalize_adjacency,
            sample_neighbors_adjacency,
            to_sparse_adjacency,
        )
        from simulation.archetypes import build_archetype_graph

        sparse_adj = to_sparse_adjacency(self.social_graph)
        k = self.config.social_neighbor_sample_k
        labels = self._labels
        assert labels is not None

        # ---- Phase 1: intra-cluster diffusion ----------------------------
        import scipy.sparse as sp

        for cid, astate in self._archetype_states.items():
            member_indices = [i for i, lab in enumerate(labels) if lab == cid]
            if len(member_indices) < 2:
                continue
            sub_agents = [self.agents[i] for i in member_indices]
            idx_set = set(member_indices)
            n_sub = len(member_indices)
            local_map = {g: l for l, g in enumerate(member_indices)}

            rows, cols, data = [], [], []
            for gi in member_indices:
                li = local_map[gi]
                row_slice = sparse_adj.getrow(gi)
                for gj, w in zip(row_slice.indices, row_slice.data):
                    if gj in idx_set:
                        rows.append(li)
                        cols.append(local_map[gj])
                        data.append(w)
            if not data:
                continue
            sub_adj = sp.csr_matrix(
                (data, (rows, cols)), shape=(n_sub, n_sub),
            )
            sub_sampled = sample_neighbors_adjacency(sub_adj, k=min(k, n_sub))
            sub_norm = normalize_adjacency(sub_sampled)

            mat = build_trait_matrix(sub_agents)
            bmat = build_belief_matrix(sub_agents)
            mat = np.clip(vectorized_social_influence(mat, sub_norm), 0.0, 1.0)
            bmat = np.clip(vectorized_belief_diffusion(bmat, sub_norm), 0.0, 1.0)
            write_trait_matrix(sub_agents, mat)
            write_belief_matrix(sub_agents, bmat)

        # ---- Phase 2: inter-archetype diffusion --------------------------
        arch_graph = build_archetype_graph(
            self.social_graph, labels, len(self._archetype_states),
        )
        n_arch = len(self._archetype_states)
        if n_arch < 2:
            return

        arch_ids = sorted(self._archetype_states.keys())
        arch_id_to_row = {aid: r for r, aid in enumerate(arch_ids)}

        arch_latent = np.vstack([
            self._archetype_states[aid].latent_state.to_vector() for aid in arch_ids
        ])
        arch_belief = np.vstack([
            self._archetype_states[aid].beliefs.to_vector() for aid in arch_ids
        ])

        arch_sparse = to_sparse_adjacency(arch_graph)
        arch_norm = normalize_adjacency(arch_sparse)

        new_latent = np.clip(vectorized_social_influence(arch_latent, arch_norm), 0.0, 1.0)
        new_belief = np.clip(vectorized_belief_diffusion(arch_belief, arch_norm), 0.0, 1.0)

        latent_deltas = new_latent - arch_latent
        belief_deltas = new_belief - arch_belief

        for aid, row in arch_id_to_row.items():
            member_indices = [i for i, lab in enumerate(labels) if lab == aid]
            ld = latent_deltas[row]
            bd = belief_deltas[row]
            for mi in member_indices:
                state = self.agents[mi].get("state")
                if state and hasattr(state, "latent_state"):
                    vec = state.latent_state.to_vector() + ld
                    from agents.behavior import BehavioralLatentState
                    state.latent_state = BehavioralLatentState.from_vector(np.clip(vec, 0.0, 1.0))
                if state and hasattr(state, "beliefs"):
                    bvec = state.beliefs.to_vector() + bd
                    from agents.belief_network import BeliefNetwork
                    state.beliefs = BeliefNetwork.from_vector(np.clip(bvec, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Private: memory summarization
    # ------------------------------------------------------------------

    def _summarize_all_memories(self) -> None:
        for a in self.agents:
            state = a.get("state")
            if state and hasattr(state, "summarize_memory"):
                state.summarize_memory()

    # ------------------------------------------------------------------
    # Private: progress emission
    # ------------------------------------------------------------------

    async def _emit_progress(
        self,
        round_idx: int,
        total_rounds: int,
        session_id: str,
        question: str,
        responses: List[Dict[str, Any]],
    ) -> None:
        for cb in self._progress_callbacks:
            try:
                await cb(round_idx, total_rounds, session_id, question, responses)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _agents_dict(self) -> Dict[str, Any]:
        """Build agent_id -> agent dict for event handlers."""
        out: Dict[str, Any] = {}
        for a in self.agents:
            p = a.get("persona")
            if p:
                out[p.agent_id] = a
        return out
