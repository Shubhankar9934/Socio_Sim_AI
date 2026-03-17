"""
Life Events: personal transitions (marriage, promotion, relocation, etc.)
that reshape agent demographics, beliefs, and behavioral state over time.

Each simulation day, every agent is checked against the event catalog.
Low-probability events fire stochastically, producing realistic life
trajectories across multi-year simulations.
"""

from __future__ import annotations

import random as _stdlib_random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agents.state import AgentState
    from population.personas import Persona


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


AGE_NEXT: Dict[str, str] = {
    "18-24": "25-34",
    "25-34": "35-44",
    "35-44": "45-54",
    "45-54": "55+",
}

INCOME_UPGRADE: Dict[str, str] = {
    "<10k": "10-25k",
    "10-25k": "25-50k",
    "25-50k": "50k+",
}

INCOME_DOWNGRADE: Dict[str, str] = {
    "50k+": "25-50k",
    "25-50k": "10-25k",
    "10-25k": "<10k",
}


@dataclass
class LifeEvent:
    """One type of personal life transition."""

    name: str
    base_probability: float
    behavioral_impacts: Dict[str, float] = field(default_factory=dict)
    belief_impacts: Dict[str, float] = field(default_factory=dict)
    demographic_changes: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event Catalog
# ---------------------------------------------------------------------------

LIFE_EVENT_CATALOG: List[LifeEvent] = [
    # --- Household ---
    LifeEvent(
        name="marriage",
        base_probability=0.0008,
        behavioral_impacts={"routine_stability": 0.05, "novelty_seeking": -0.03,
                            "social_influence_susceptibility": 0.02},
        belief_impacts={"health_priority": 0.02},
        demographic_changes={"family.spouse": True},
        conditions={"family.spouse": False, "age_in": ("25-34", "35-44", "45-54")},
    ),
    LifeEvent(
        name="divorce",
        base_probability=0.0002,
        behavioral_impacts={"routine_stability": -0.06, "novelty_seeking": 0.04,
                            "financial_confidence": -0.04},
        belief_impacts={"government_trust": -0.02},
        demographic_changes={"family.spouse": False},
        conditions={"family.spouse": True},
    ),
    LifeEvent(
        name="child_birth",
        base_probability=0.0006,
        behavioral_impacts={"time_pressure": 0.08, "price_sensitivity": 0.04,
                            "convenience_seeking": 0.05, "routine_stability": 0.04},
        belief_impacts={"health_priority": 0.03, "environmental_concern": 0.02},
        demographic_changes={"family.children": "+1"},
        conditions={"family.spouse": True, "age_in": ("25-34", "35-44"),
                    "max_children": 4},
    ),

    # --- Economic ---
    LifeEvent(
        name="promotion",
        base_probability=0.0005,
        behavioral_impacts={"financial_confidence": 0.06, "price_sensitivity": -0.04,
                            "novelty_seeking": 0.02},
        belief_impacts={"innovation_curiosity": 0.02, "technology_optimism": 0.01},
        demographic_changes={"income": "<upgrade>"},
        conditions={"income_below": "50k+"},
    ),
    LifeEvent(
        name="job_loss",
        base_probability=0.0002,
        behavioral_impacts={"financial_confidence": -0.08, "price_sensitivity": 0.06,
                            "risk_aversion": 0.04, "time_pressure": -0.03},
        belief_impacts={"government_trust": -0.03, "price_consciousness": 0.04},
        demographic_changes={"income": "<downgrade>"},
        conditions={"income_above": "<10k"},
    ),
    LifeEvent(
        name="new_job",
        base_probability=0.0004,
        behavioral_impacts={"financial_confidence": 0.03, "routine_stability": -0.02,
                            "novelty_seeking": 0.03},
        belief_impacts={"innovation_curiosity": 0.01},
        demographic_changes={},
        conditions={},
    ),

    # --- Location ---
    LifeEvent(
        name="relocation",
        base_probability=0.0003,
        behavioral_impacts={"novelty_seeking": 0.03, "routine_stability": -0.04,
                            "social_influence_susceptibility": 0.02},
        belief_impacts={},
        demographic_changes={"location": "<random>"},
        conditions={},
    ),

    # --- Lifestyle ---
    LifeEvent(
        name="health_improvement",
        base_probability=0.0004,
        behavioral_impacts={"health_orientation": 0.06,
                            "environmental_consciousness": 0.02},
        belief_impacts={"health_priority": 0.04},
        demographic_changes={},
        conditions={},
    ),

    # --- Aging ---
    LifeEvent(
        name="age_transition",
        base_probability=0.00015,
        behavioral_impacts={"risk_aversion": 0.02, "routine_stability": 0.02,
                            "novelty_seeking": -0.01},
        belief_impacts={},
        demographic_changes={"age": "<next>"},
        conditions={"age_can_advance": True},
    ),
]


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def check_eligibility(persona: "Persona", state: "AgentState", event: LifeEvent) -> bool:
    """Return True if the agent meets all conditions for this event."""
    conds = event.conditions
    if not conds:
        return True

    fam = persona.family

    if "family.spouse" in conds:
        if fam.spouse != conds["family.spouse"]:
            return False

    if "age_in" in conds:
        if persona.age not in conds["age_in"]:
            return False

    if "max_children" in conds:
        if fam.children >= conds["max_children"]:
            return False

    if "income_below" in conds:
        if persona.income == conds["income_below"]:
            return False

    if "income_above" in conds:
        if persona.income == conds["income_above"]:
            return False

    if "age_can_advance" in conds:
        if persona.age not in AGE_NEXT:
            return False

    return True


# ---------------------------------------------------------------------------
# Probability adjustment
# ---------------------------------------------------------------------------

def compute_probability(
    persona: "Persona",
    state: "AgentState",
    event: LifeEvent,
    social_graph=None,
    agents: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """Adjust base_probability based on demographics and social context."""
    p = event.base_probability

    if event.name == "marriage":
        if persona.age == "25-34":
            p *= 1.5
        elif persona.age == "45-54":
            p *= 0.5

    elif event.name == "child_birth":
        if persona.age == "25-34":
            p *= 1.3
        if persona.family.children >= 2:
            p *= 0.5
        if social_graph is not None and agents is not None:
            parent_ratio = _neighbor_parent_ratio(persona, social_graph, agents)
            if parent_ratio > 0.3:
                p *= 1.4

    elif event.name == "promotion":
        if persona.occupation == "managerial":
            p *= 1.3
        elif persona.occupation == "service":
            p *= 0.7

    elif event.name == "relocation":
        if persona.income in ("25-50k", "50k+"):
            p *= 1.3

    return p


def _neighbor_parent_ratio(persona, social_graph, agents) -> float:
    """Fraction of social neighbors who have children."""
    from social.network import agent_id_to_node, node_to_agent_id

    node = agent_id_to_node(social_graph, persona.agent_id)
    if node is None:
        return 0.0

    agents_by_id = {
        a.get("persona").agent_id: a
        for a in agents
        if a.get("persona") is not None
    }

    total = 0
    parents = 0
    for n_node in social_graph.neighbors(node):
        nid = node_to_agent_id(social_graph, n_node)
        a = agents_by_id.get(nid)
        if a is None:
            continue
        p = a.get("persona")
        if p is not None:
            total += 1
            if p.family.children > 0:
                parents += 1
    return parents / max(1, total)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_life_event(
    persona: "Persona",
    state: "AgentState",
    event: LifeEvent,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Apply a life event: update behavior, beliefs, and demographics."""
    # Behavioral impacts
    if event.behavioral_impacts:
        state.latent_state.apply_event_impact(event.behavioral_impacts)

    # Belief impacts
    if event.belief_impacts:
        state.beliefs.apply_event_impact(event.belief_impacts)

    # Demographic changes
    for key, value in event.demographic_changes.items():
        _apply_demographic_change(persona, key, value, rng)


# Behavioral nudges applied to social neighbors when a life event fires.
# Values are small deltas (positive = increase).
_NEIGHBOR_CASCADES: Dict[str, Dict[str, float]] = {
    "marriage": {
        "routine_stability": 0.02,
        "social_influence_susceptibility": 0.01,
    },
    "promotion": {
        "financial_confidence": 0.02,
        "novelty_seeking": 0.01,
    },
    "child_birth": {
        "health_orientation": 0.02,
        "routine_stability": 0.02,
    },
    "relocation": {
        "novelty_seeking": 0.01,
    },
}


def cascade_to_neighbors(
    agent_id: str,
    event: "LifeEvent",
    social_graph,
    agents_by_id: Dict[str, Dict[str, Any]],
) -> None:
    """Apply small aspirational nudges to social neighbors of the agent
    who experienced the life event.

    Only a subset of high-impact events trigger cascades, and the nudges
    are intentionally small (behavioral contagion, not direct change).
    """
    nudges = _NEIGHBOR_CASCADES.get(event.name)
    if nudges is None or social_graph is None:
        return

    from social.network import agent_id_to_node, node_to_agent_id

    node = agent_id_to_node(social_graph, agent_id)
    if node is None:
        return
    for n_node in social_graph.neighbors(node):
        nid = node_to_agent_id(social_graph, n_node)
        neighbor = agents_by_id.get(nid)
        if neighbor is None:
            continue
        state = neighbor.get("state")
        if state is None or not hasattr(state, "latent_state"):
            continue
        for dim, delta in nudges.items():
            if hasattr(state.latent_state, dim):
                old = getattr(state.latent_state, dim)
                setattr(state.latent_state, dim, _clamp(old + delta))


def _apply_demographic_change(
    persona: "Persona",
    key: str,
    value: Any,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Mutate a persona field based on the change descriptor."""
    if key == "family.spouse":
        persona.family.spouse = bool(value)

    elif key == "family.children":
        if value == "+1":
            persona.family.children = min(8, persona.family.children + 1)
        elif value == "-1":
            persona.family.children = max(0, persona.family.children - 1)
        else:
            persona.family.children = int(value)

    elif key == "income":
        if value == "<upgrade>":
            persona.income = INCOME_UPGRADE.get(persona.income, persona.income)
        elif value == "<downgrade>":
            persona.income = INCOME_DOWNGRADE.get(persona.income, persona.income)
        else:
            persona.income = str(value)

    elif key == "age":
        if value == "<next>":
            persona.age = AGE_NEXT.get(persona.age, persona.age)
        else:
            persona.age = str(value)

    elif key == "location":
        if value == "<random>":
            _relocate_random(persona, rng)
        else:
            persona.location = str(value)

    elif key == "household_size":
        if value == "+1":
            current = int(persona.household_size.split("-")[0].replace("+", ""))
            persona.household_size = str(current + 1)
        else:
            persona.household_size = str(value)


def _relocate_random(
    persona: "Persona",
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Pick a new district weighted by income-appropriate distribution."""
    try:
        from config.demographics import get_demographics
        dist = get_demographics().location_given_income.get(persona.income, {})
    except Exception:
        dist = {}

    if not dist:
        return

    # Exclude current location
    options = {k: v for k, v in dist.items() if k != persona.location}
    if not options:
        return

    total = sum(options.values())
    if total <= 0:
        return

    locations = list(options.keys())
    weights = [options[loc] / total for loc in locations]

    if rng is not None:
        chosen = rng.choice(locations, p=weights)
    else:
        chosen = np.random.choice(locations, p=weights)
    persona.location = str(chosen)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_life_events(
    persona: "Persona",
    state: "AgentState",
    rng: np.random.Generator,
    social_graph=None,
    agents: Optional[List[Dict[str, Any]]] = None,
) -> List[LifeEvent]:
    """Check each catalog event for eligibility and roll dice. Return triggered events."""
    triggered: List[LifeEvent] = []
    for event in LIFE_EVENT_CATALOG:
        if not check_eligibility(persona, state, event):
            continue
        prob = compute_probability(persona, state, event, social_graph, agents)
        if rng.random() < prob:
            triggered.append(event)
    return triggered
