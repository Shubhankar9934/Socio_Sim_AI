"""Lexical routing: grocery / inflation questions map to cost_of_living (not general)."""

from agents.intent_router import detect_domain_rule, detect_topic_rule
from agents.perception import _detect_topic


def test_grocery_prices_topic_cost_of_living():
    q = "How are you adjusting to higher grocery prices?"
    assert detect_topic_rule(q) == "cost_of_living"
    assert _detect_topic(q) == "cost_of_living"


def test_inflation_supermarket_topic():
    q = "Has inflation changed how you shop at the supermarket?"
    assert detect_topic_rule(q) == "cost_of_living"


def test_grocery_question_domain_economics_first_match():
    q = "Grocery bills and utility bills are squeezing our household budget."
    d = detect_domain_rule(q)
    assert d == "economics"
