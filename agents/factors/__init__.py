"""
Factor modules and graph builder.

Provides ``build_factor_graph(question_model)`` which assembles a
FactorGraph with the correct factors and weights for a given QuestionModel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.factor_graph import FactorGraph
from agents.factors.behavioral import behavioral_factor
from agents.factors.belief import belief_factor
from agents.factors.income import income_factor
from agents.factors.location import location_factor
from agents.factors.memory import memory_factor
from agents.factors.personality import personality_factor
from agents.factors.social import social_factor

if TYPE_CHECKING:
    from config.question_models import QuestionModel

FACTOR_REGISTRY = {
    "personality": personality_factor,
    "income": income_factor,
    "social": social_factor,
    "location": location_factor,
    "memory": memory_factor,
    "behavioral": behavioral_factor,
    "belief": belief_factor,
}


def build_factor_graph(model: "QuestionModel") -> FactorGraph:
    """Assemble a FactorGraph from a QuestionModel's factor_weights."""
    graph = FactorGraph()
    for name, weight in model.factor_weights.items():
        fn = FACTOR_REGISTRY.get(name)
        if fn is not None:
            graph.add_factor(fn, weight)
    return graph
