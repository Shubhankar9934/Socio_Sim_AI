# Social Module

Barabasi-Albert social network and influence metrics.

## network.py

**Purpose**: Scale-free graph with community structure and typed edges.

### Constants

| Constant | Description |
|----------|-------------|
| `RELATIONSHIP_TYPES` | friend, coworker, neighbor. |

### Functions

| Function | Description |
|----------|-------------|
| `build_social_network(personas, m, seed)` | Barabasi-Albert graph; assign relationship types (location → neighbor, occupation → coworker); assign homophily similarity weights. |
| `barabasi_albert_network(n, m, seed)` | Create scale-free graph. |
| `assign_relationship_types(G, personas, id_list, seed)` | Edge relationship type from location/occupation. |
| `node_to_agent_id(G, node)` | Map node index to agent_id. |
| `agent_id_to_node(G, agent_id)` | Map agent_id to node index. |
| `neighbors(G, agent_id)` | List of neighbor agent_ids. |
| `neighbors_by_relationship(G, agent_id, relationship)` | Neighbors with given relationship type. |
| `to_sparse_adjacency(G)` | Convert to scipy sparse CSR. Uses similarity edge weight. |
| `normalize_adjacency(A)` | Row-normalize so rows sum to 1. |
| `sample_neighbors_adjacency(A, k, seed)` | Retain at most k neighbors per node; sample by edge weight. |

---

## influence.py

**Purpose**: Social influence metrics.

### Functions

| Function | Description |
|----------|-------------|
| `fraction_friends_with_trait(graph, agent_id, trait_by_agent)` | Fraction of neighbors with trait=True. Used for friends_using_delivery. |
| `persona_similarity(p1, p2)` | Homophily similarity (0–1) for edge weights. |
