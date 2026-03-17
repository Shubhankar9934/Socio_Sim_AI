"""
Archetype compression: KMeans clustering of agents; route LLM calls through
representatives.

The ``ArchetypeState`` dataclass holds shared state for a cluster so that
distribution computation can run per-archetype (30 brains) rather than
per-agent (500+), achieving 5-10x speed-ups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from agents.behavior import BehavioralLatentState
from agents.belief_network import BeliefNetwork
from population.personas import Persona


# ------------------------------------------------------------------
# Vector aggregation
# ------------------------------------------------------------------

def _aggregate_vectors(
    vecs: List[np.ndarray],
    method: str = "median",
) -> np.ndarray:
    """Aggregate a list of vectors into a single representative vector.

    ``"median"`` is the default because mean distorts bimodal clusters
    (e.g. 3 "rarely" + 1 "daily" produces a mean nobody holds).
    ``"trimmed_mean"`` drops top/bottom 10% -- useful for large clusters.
    ``"mean"`` is the legacy behaviour.
    """
    arr = np.vstack(vecs)
    if method == "median":
        return np.median(arr, axis=0)
    if method == "trimmed_mean":
        from scipy.stats import trim_mean
        return trim_mean(arr, proportiontocut=0.1, axis=0)
    return np.mean(arr, axis=0)


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

_LOC_SET = [
    "Dubai Marina", "Jumeirah", "Deira", "Business Bay",
    "Al Barsha", "JLT", "Downtown", "Al Karama", "JVC", "Others",
]
_NAT_SET = [
    "Emirati", "Indian", "Pakistani", "Filipino",
    "Western", "Egyptian", "Other",
]
_AGE_MAP = {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}
_INCOME_MAP = {"<10k": 0, "10-25k": 1, "25-50k": 2, "50k+": 3}


def _persona_to_feature_vector(p: Persona) -> np.ndarray:
    """Convert persona to numeric vector for clustering (11 dims)."""
    loc_idx = _LOC_SET.index(p.location) if p.location in _LOC_SET else 0
    nat_idx = _NAT_SET.index(p.nationality) if p.nationality in _NAT_SET else 0
    return np.array([
        _AGE_MAP.get(p.age, 2),
        _INCOME_MAP.get(p.income, 1),
        loc_idx,
        nat_idx,
        p.lifestyle.convenience_preference,
        p.lifestyle.primary_service_preference,
        p.lifestyle.price_sensitivity,
        p.lifestyle.tech_adoption,
        float(p.family.spouse),
        float(p.family.children) / 5.0,
        float(p.mobility.car),
    ], dtype=np.float64)


def _agent_to_feature_vector(agent: Dict[str, Any]) -> np.ndarray:
    """23-dim feature vector: 11 demographic + 12 current latent state.

    Used for reclustering after agents have evolved so that the KMeans
    input reflects *current* behaviour, not just initial demographics.
    """
    p = agent["persona"]
    base = _persona_to_feature_vector(p)
    state = agent.get("state")
    if state and hasattr(state, "latent_state"):
        latent = state.latent_state.to_vector()
        return np.concatenate([base, latent])
    return np.concatenate([base, np.full(12, 0.5)])


# ------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------

def _kmeans_init(
    prev_centroids: Optional[np.ndarray],
    n_clusters: int,
    n_features: int,
) -> Tuple[Any, int]:
    """Return ``(init, n_init)`` kwargs for KMeans.

    Warm-starts from *prev_centroids* when the shape is compatible;
    falls back to ``"k-means++"`` otherwise.
    """
    if (
        prev_centroids is not None
        and prev_centroids.ndim == 2
        and prev_centroids.shape == (n_clusters, n_features)
    ):
        return prev_centroids, 1
    return "k-means++", 10


def compute_archetypes(
    personas: List[Persona],
    n_clusters: int,
    seed: Optional[int] = None,
    prev_centroids: Optional[np.ndarray] = None,
) -> Tuple[List[int], np.ndarray, float]:
    """Cluster personas into *n_clusters*. Returns ``(labels, cluster_centers, inertia)``.

    When *prev_centroids* is provided and shape-compatible, KMeans is
    warm-started from the previous run's centroids (``n_init=1``).
    This stabilises cluster identity across reclustering rounds.
    """
    if not personas or n_clusters >= len(personas):
        return list(range(len(personas))), np.array([]), 0.0
    X = np.vstack([_persona_to_feature_vector(p) for p in personas])
    n_clusters = min(n_clusters, len(personas) - 1) if len(personas) > n_clusters else max(1, len(personas))
    init, n_init = _kmeans_init(prev_centroids, n_clusters, X.shape[1])
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init, init=init)
    labels = kmeans.fit_predict(X)
    inertia = float(kmeans.inertia_)
    return list(labels), kmeans.cluster_centers_, inertia


def choose_archetype_representative(
    personas: List[Persona],
    labels: List[int],
    cluster_id: int,
    centers: Optional[np.ndarray] = None,
    feature_matrix: Optional[np.ndarray] = None,
) -> int:
    """Index of persona closest to cluster center (Euclidean distance).

    When *feature_matrix* is provided it is used as the distance basis
    instead of recomputing persona feature vectors.  This is needed for
    latent-aware reclustering where the feature space is 23-dim.
    """
    indices = [i for i, lab in enumerate(labels) if lab == cluster_id]
    if not indices:
        return 0
    if len(indices) == 1:
        return indices[0]
    if centers is not None and cluster_id < len(centers):
        center = centers[cluster_id]
        if feature_matrix is not None:
            vecs = feature_matrix[indices]
        else:
            vecs = np.vstack([_persona_to_feature_vector(personas[i]) for i in indices])
        dists = np.linalg.norm(vecs - center, axis=1)
        return indices[int(np.argmin(dists))]
    return indices[0]


def build_archetype_map(
    personas: List[Persona],
    n_archetypes: int,
    seed: Optional[int] = None,
) -> Tuple[Dict[int, int], List[int]]:
    """Build mapping: cluster_id -> representative persona index + labels."""
    labels, centers, _ = compute_archetypes(personas, n_archetypes, seed)
    unique = set(labels)
    rep = {
        c: choose_archetype_representative(personas, labels, c, centers)
        for c in unique
    }
    return rep, labels


# ------------------------------------------------------------------
# ArchetypeState
# ------------------------------------------------------------------

@dataclass
class ArchetypeState:
    """Shared mutable state for one archetype cluster.

    Holds the aggregated latent / belief state of all members and caches
    the last round's distribution and narrative for result expansion.
    """
    archetype_id: int
    members: List[str] = field(default_factory=list)
    representative_idx: int = 0
    latent_state: BehavioralLatentState = field(default_factory=BehavioralLatentState)
    beliefs: BeliefNetwork = field(default_factory=BeliefNetwork)
    persona_template: Optional[Persona] = field(default=None, repr=False)
    last_distribution: Dict[str, float] = field(default_factory=dict)
    last_sampled: str = ""
    narrative_templates: List[str] = field(default_factory=list)

    @property
    def narrative_template(self) -> str:
        """Backward-compat: first template or empty string."""
        return self.narrative_templates[0] if self.narrative_templates else ""

    @narrative_template.setter
    def narrative_template(self, value: str) -> None:
        if self.narrative_templates:
            self.narrative_templates[0] = value
        else:
            self.narrative_templates.append(value)


_INERTIA_DRIFT_THRESHOLD = 1.25


def build_archetype_states(
    agents: List[Dict[str, Any]],
    n_archetypes: int,
    seed: Optional[int] = None,
    aggregation: str = "median",
    use_latent_features: bool = False,
    prev_centroids: Optional[np.ndarray] = None,
    prev_inertia: Optional[float] = None,
) -> Tuple[Dict[int, ArchetypeState], List[int], np.ndarray, float]:
    """Cluster agents and return per-archetype shared state.

    Parameters
    ----------
    aggregation : ``"median"`` | ``"trimmed_mean"`` | ``"mean"``
    use_latent_features : when True, clustering uses the 23-dim
        agent feature vector (demographic + current latent state)
        instead of the 11-dim persona-only vector.  Useful for
        reclustering after agents have evolved.
    prev_centroids : centroids from a previous clustering run.
        When shape-compatible, KMeans is warm-started for stable
        cluster identity across reclustering rounds.
    prev_inertia : inertia from the previous clustering run.  When
        warm-start yields new_inertia > prev_inertia * 1.25, KMeans
        is rerun with k-means++ to escape a bad local minimum.

    Returns
    -------
    archetype_states : dict mapping cluster_id -> ArchetypeState
    labels : list mapping agent index -> cluster_id
    centroids : (K, D) array of cluster centers for warm-starting
        the next reclustering call.
    inertia : within-cluster sum of squared distances
    """
    personas = [a["persona"] for a in agents]
    X: Optional[np.ndarray] = None
    inertia: float = 0.0

    if use_latent_features:
        X = np.vstack([_agent_to_feature_vector(a) for a in agents])
        if not agents or n_archetypes >= len(agents):
            labels_list: List[int] = list(range(len(agents)))
            centers = np.array([])
            inertia = 0.0
        else:
            nc = min(n_archetypes, len(agents) - 1) if len(agents) > n_archetypes else max(1, len(agents))
            init, n_init = _kmeans_init(prev_centroids, nc, X.shape[1])
            km = KMeans(n_clusters=nc, random_state=seed, n_init=n_init, init=init)
            labels_arr = km.fit_predict(X)
            labels_list = list(labels_arr)
            centers = km.cluster_centers_
            inertia = float(km.inertia_)
            if (
                prev_inertia is not None
                and prev_inertia > 0
                and inertia > prev_inertia * _INERTIA_DRIFT_THRESHOLD
            ):
                km_fallback = KMeans(
                    n_clusters=nc, random_state=seed,
                    n_init=10, init="k-means++",
                )
                labels_arr = km_fallback.fit_predict(X)
                labels_list = list(labels_arr)
                centers = km_fallback.cluster_centers_
                inertia = float(km_fallback.inertia_)
    else:
        labels_list, centers, inertia = compute_archetypes(
            personas, n_archetypes, seed, prev_centroids=prev_centroids,
        )

    unique_ids = sorted(set(labels_list))
    archetype_states: Dict[int, ArchetypeState] = {}

    for cid in unique_ids:
        member_indices = [i for i, lab in enumerate(labels_list) if lab == cid]
        member_ids = [personas[i].agent_id for i in member_indices]
        rep_idx = choose_archetype_representative(
            personas, labels_list, cid, centers,
            feature_matrix=X,
        )

        latent_vecs, belief_vecs = [], []
        for idx in member_indices:
            state = agents[idx].get("state")
            if state and hasattr(state, "latent_state"):
                latent_vecs.append(state.latent_state.to_vector())
                belief_vecs.append(state.beliefs.to_vector())

        agg_latent = BehavioralLatentState.from_vector(
            _aggregate_vectors(latent_vecs, aggregation)
        ) if latent_vecs else BehavioralLatentState()
        agg_beliefs = BeliefNetwork.from_vector(
            _aggregate_vectors(belief_vecs, aggregation)
        ) if belief_vecs else BeliefNetwork()

        archetype_states[cid] = ArchetypeState(
            archetype_id=cid,
            members=member_ids,
            representative_idx=rep_idx,
            latent_state=agg_latent,
            beliefs=agg_beliefs,
            persona_template=personas[rep_idx],
        )

    for i, lab in enumerate(labels_list):
        personas[i].meta.persona_cluster = int(lab)
        rep_idx = archetype_states[lab].representative_idx
        personas[i].meta.archetype_id = int(rep_idx)

    return archetype_states, labels_list, centers, inertia


def refresh_archetype_states(
    archetype_states: Dict[int, ArchetypeState],
    agents: List[Dict[str, Any]],
    labels: List[int],
    aggregation: str = "median",
) -> None:
    """Re-aggregate member latent/belief vectors back into archetype state.

    Called between survey rounds so archetype state reflects individual
    updates made during the last round.
    """
    for cid, astate in archetype_states.items():
        member_indices = [i for i, lab in enumerate(labels) if lab == cid]
        latent_vecs, belief_vecs = [], []
        for idx in member_indices:
            state = agents[idx].get("state")
            if state and hasattr(state, "latent_state"):
                latent_vecs.append(state.latent_state.to_vector())
                belief_vecs.append(state.beliefs.to_vector())
        if latent_vecs:
            astate.latent_state = BehavioralLatentState.from_vector(
                _aggregate_vectors(latent_vecs, aggregation)
            )
        if belief_vecs:
            astate.beliefs = BeliefNetwork.from_vector(
                _aggregate_vectors(belief_vecs, aggregation)
            )


# ------------------------------------------------------------------
# Archetype-level social graph
# ------------------------------------------------------------------

def build_archetype_graph(
    social_graph: nx.Graph,
    labels: List[int],
    n_archetypes: int,
) -> nx.Graph:
    """Build a coarsened graph where nodes are archetypes.

    Edge weight between two archetype nodes is the number of inter-cluster
    edges in the original graph, normalized by the product of cluster sizes.
    This lets social diffusion run on ~30 nodes instead of ~500.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_archetypes))

    node_to_cluster = {i: labels[i] for i in range(len(labels))}
    cluster_sizes: Dict[int, int] = {}
    for lab in labels:
        cluster_sizes[lab] = cluster_sizes.get(lab, 0) + 1

    edge_counts: Dict[Tuple[int, int], int] = {}
    for u, v in social_graph.edges():
        if u >= len(labels) or v >= len(labels):
            continue
        cu, cv = node_to_cluster[u], node_to_cluster[v]
        if cu == cv:
            continue
        key = (min(cu, cv), max(cu, cv))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    for (cu, cv), count in edge_counts.items():
        denom = max(cluster_sizes.get(cu, 1) * cluster_sizes.get(cv, 1), 1)
        weight = count / denom
        G.add_edge(cu, cv, weight=weight)

    return G
