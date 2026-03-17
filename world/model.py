"""
World model: Dubai as a graph of districts (nodes) and connections (roads/metro).
"""

from typing import Dict, List, Optional

import networkx as nx

from world.districts import DEFAULT_DISTRICT_PROPERTIES, DistrictProperties, get_district


def build_city_graph() -> nx.Graph:
    """
    Dubai city graph: nodes = districts, edges = connections with distance (km).
    """
    G = nx.Graph()
    districts = list(DEFAULT_DISTRICT_PROPERTIES.keys())
    for d in districts:
        G.add_node(d, **get_district(d).__dict__)
    # Edges with approximate distances (km)
    edges = [
        ("Dubai Marina", "Jumeirah", 8),
        ("Dubai Marina", "JLT", 3),
        ("Dubai Marina", "Business Bay", 10),
        ("Jumeirah", "Downtown", 6),
        ("Deira", "Business Bay", 12),
        ("Deira", "Al Karama", 5),
        ("Business Bay", "Downtown", 2),
        ("Business Bay", "Al Barsha", 8),
        ("Al Barsha", "JVC", 6),
        ("JLT", "Business Bay", 7),
        ("Downtown", "Al Karama", 4),
        ("Al Karama", "Deira", 5),
        ("Others", "Al Barsha", 5),
        ("Others", "Deira", 10),
    ]
    for u, v, d in edges:
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v, distance_km=d)
    return G


# Module-level singleton
_city_graph: Optional[nx.Graph] = None


def get_city_graph() -> nx.Graph:
    global _city_graph
    if _city_graph is None:
        _city_graph = build_city_graph()
    return _city_graph


def districts_with_metro(G: Optional[nx.Graph] = None) -> List[str]:
    """List district names that have metro access."""
    G = G or get_city_graph()
    return [n for n in G.nodes() if G.nodes[n].get("metro_access", False)]
