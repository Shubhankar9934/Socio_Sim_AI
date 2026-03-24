"""
Social network: Barabasi-Albert scale-free graph with community structure.
Edges typed as friend / coworker / neighbor from location and occupation.
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from population.personas import Persona


RELATIONSHIP_TYPES = ("friend", "coworker", "neighbor", "family")

TIE_STRENGTH: Dict[str, float] = {
    "family": 1.0,
    "coworker": 0.6,
    "neighbor": 0.5,
    "friend": 0.7,
}


def _agent_ids(personas: List[Persona]) -> List[str]:
    return [p.agent_id for p in personas]


def _persona_by_id(personas: List[Persona]) -> Dict[str, Persona]:
    return {p.agent_id: p for p in personas}


def barabasi_albert_network(
    n: int,
    m: int = 4,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Create scale-free graph with n nodes, m edges per new node."""
    m = min(m, max(1, n - 1))  # barabasi_albert requires 1 <= m < n
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G


def assign_relationship_types(
    G: nx.Graph,
    personas: List[Persona],
    id_list: List[str],
    seed: Optional[int] = None,
) -> None:
    """
    Assign edge relationship type and tie strength.
    Uses location for neighbor, occupation for coworker, else friend.
    """
    rng = random.Random(seed)
    persona_by_id = _persona_by_id(personas)
    for u, v in G.edges():
        uid = id_list[u] if u < len(id_list) else str(u)
        vid = id_list[v] if v < len(id_list) else str(v)
        pu = persona_by_id.get(uid)
        pv = persona_by_id.get(vid)
        if pu and pv:
            if pu.location == pv.location:
                rel = "neighbor" if rng.random() < 0.6 else "friend"
            elif pu.occupation == pv.occupation:
                rel = "coworker" if rng.random() < 0.5 else "friend"
            else:
                rel = "friend"
        else:
            rel = rng.choice(["friend", "coworker", "neighbor"])
        G[u][v]["relationship"] = rel
        G[v][u]["relationship"] = rel
        strength = TIE_STRENGTH.get(rel, 0.5)
        G[u][v]["tie_strength"] = strength
        G[v][u]["tie_strength"] = strength


def _assign_similarity_weights(
    G: nx.Graph,
    personas: List[Persona],
    id_list: List[str],
) -> None:
    """Pre-compute and store homophily similarity on every edge."""
    from social.influence import persona_similarity

    persona_by_id = _persona_by_id(personas)
    for u, v in G.edges():
        uid = id_list[u] if u < len(id_list) else str(u)
        vid = id_list[v] if v < len(id_list) else str(v)
        pu = persona_by_id.get(uid)
        pv = persona_by_id.get(vid)
        if pu and pv:
            sim = persona_similarity(pu, pv)
        else:
            sim = 1.0
        G[u][v]["similarity"] = sim
        G[v][u]["similarity"] = sim


def _add_family_edges(
    G: nx.Graph,
    personas: List[Persona],
    id_list: List[str],
    seed: Optional[int] = None,
) -> None:
    """Link agents who could plausibly be family (spouse/children).

    For each agent with a spouse, find a compatible agent in a similar
    location/age bracket and add a high-strength family edge. This makes
    family influence flow through the social diffusion graph.
    """
    rng = random.Random(seed)
    persona_by_id = _persona_by_id(personas)
    id_to_node = {aid: i for i, aid in enumerate(id_list)}

    # Build location index for efficient matching
    location_groups: Dict[str, List[str]] = {}
    for p in personas:
        location_groups.setdefault(p.location, []).append(p.agent_id)

    paired: Set[str] = set()
    for p in personas:
        if not p.family.spouse or p.agent_id in paired:
            continue
        candidates = [
            aid for aid in location_groups.get(p.location, [])
            if aid != p.agent_id
            and aid not in paired
            and persona_by_id[aid].family.spouse
        ]
        if not candidates:
            continue
        partner_id = rng.choice(candidates)
        u = id_to_node.get(p.agent_id)
        v = id_to_node.get(partner_id)
        if u is not None and v is not None and not G.has_edge(u, v):
            G.add_edge(u, v, relationship="family", tie_strength=1.0, similarity=0.9)
        paired.add(p.agent_id)
        paired.add(partner_id)


def build_social_network(
    personas: List[Persona],
    m: int = 4,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Build Barabasi-Albert graph over agents; node i = personas[i].
    Returns graph with integer nodes 0..n-1; use id_list to map to agent_id.
    Edges carry relationship type, tie strength, and homophily similarity weight.
    """
    n = len(personas)
    if n < 2:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        return G
    G = barabasi_albert_network(n, m=m, seed=seed)
    id_list = _agent_ids(personas)
    assign_relationship_types(G, personas, id_list, seed=seed)
    _assign_similarity_weights(G, personas, id_list)
    _add_family_edges(G, personas, id_list, seed=seed)
    G.graph["agent_ids"] = id_list
    return G


def node_to_agent_id(G: nx.Graph, node: int) -> str:
    """Map graph node index to agent_id."""
    ids = G.graph.get("agent_ids", [])
    if 0 <= node < len(ids):
        return ids[node]
    return str(node)


def agent_id_to_node(G: nx.Graph, agent_id: str) -> Optional[int]:
    """Map agent_id to graph node index."""
    ids = G.graph.get("agent_ids", [])
    try:
        return ids.index(agent_id)
    except ValueError:
        return None


def neighbors(G: nx.Graph, agent_id: str) -> List[str]:
    """Return list of neighbor agent_ids for the given agent."""
    node = agent_id_to_node(G, agent_id)
    if node is None:
        return []
    return [node_to_agent_id(G, n) for n in G.neighbors(node)]


def neighbors_by_relationship(
    G: nx.Graph,
    agent_id: str,
    relationship: str,
) -> List[str]:
    """Return neighbor agent_ids with the given relationship type."""
    node = agent_id_to_node(G, agent_id)
    if node is None:
        return []
    out = []
    for n in G.neighbors(node):
        if G[node].get(n, {}).get("relationship") == relationship:
            out.append(node_to_agent_id(G, n))
    return out


def to_sparse_adjacency(G: nx.Graph, use_tie_strength: bool = True):
    """Convert NetworkX graph to a scipy sparse CSR matrix.

    Uses the pre-computed ``similarity`` edge weight so that social
    diffusion is homophily-weighted.  When ``use_tie_strength`` is True,
    the similarity weight is multiplied by the tie_strength attribute
    so family/close ties carry more diffusion weight.

    Returns scipy.sparse.csr_matrix of shape (N, N).
    """
    import scipy.sparse as sp

    n = G.number_of_nodes()
    if n == 0:
        return sp.csr_matrix((0, 0), dtype=float)
    try:
        A_sim = nx.to_scipy_sparse_array(G, weight="similarity", format="csr")
        if use_tie_strength:
            try:
                A_tie = nx.to_scipy_sparse_array(G, weight="tie_strength", format="csr")
                A = A_sim.multiply(A_tie)
            except Exception:
                A = A_sim
        else:
            A = A_sim
    except Exception:
        A = nx.to_scipy_sparse_array(G, format="csr")
    return A


def normalize_adjacency(A):
    """Row-normalize a sparse adjacency matrix so each row sums to 1.

    Pre-normalizing once avoids redundant degree division on every
    simulation step.  Isolated nodes (degree 0) get a self-loop of 1.
    """
    import numpy as np
    import scipy.sparse as sp

    degree = np.asarray(A.sum(axis=1)).ravel()
    degree = np.maximum(degree, 1.0)
    D_inv = sp.diags(1.0 / degree)
    return D_inv @ A


def sample_neighbors_adjacency(
    A,
    k: int = 15,
    seed: Optional[int] = None,
):
    """Build a sparse adjacency where each node retains at most *k* neighbors.

    For nodes with degree <= k the row is unchanged.  For higher-degree nodes
    k neighbors are sampled **with probability proportional to edge weight**
    (homophily-biased sampling).  Returns a new CSR matrix of the same shape.
    """
    import numpy as np
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    A_csr = A.tocsr()
    n = A_csr.shape[0]
    rows, cols, vals = [], [], []

    for i in range(n):
        start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
        neighbors_i = A_csr.indices[start:end]
        weights_i = A_csr.data[start:end].copy()
        deg = len(neighbors_i)

        if deg <= k:
            rows.extend([i] * deg)
            cols.extend(neighbors_i.tolist())
            vals.extend(weights_i.tolist())
        else:
            w = weights_i / (weights_i.sum() or 1.0)
            chosen = rng.choice(deg, size=k, replace=False, p=w)
            rows.extend([i] * k)
            cols.extend(neighbors_i[chosen].tolist())
            vals.extend(weights_i[chosen].tolist())

    if not rows:
        return sp.csr_matrix((n, n), dtype=float)
    return sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(n, n),
    )
