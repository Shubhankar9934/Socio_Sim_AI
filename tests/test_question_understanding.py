from agents.perception import detect_question_model, perceive
from agents.state import AgentState
from population.synthesis import generate_population


def test_transport_how_often_maps_to_frequency_model():
    p = perceive("How often do you use ride-hailing apps like Uber or Careem?")
    qm = detect_question_model(p)
    assert p.question_type == "frequency"
    assert qm.name == "transport_usage_frequency"
    assert qm.scale == ["never", "rarely", "sometimes", "often", "very often"]


def test_likelihood_question_does_not_map_to_frequency():
    p = perceive("How likely are you to try a new fintech wallet next month?")
    qm = detect_question_model(p)
    assert p.question_type == "likelihood"
    assert "likely" in " ".join(qm.scale).lower()


def test_cost_of_living_routes_to_cost_model():
    p = perceive("How satisfied are you with the cost of living in your city?")
    qm = detect_question_model(p)
    assert p.question_type == "likert"
    assert qm.name == "cost_of_living_satisfaction"


def test_greeting_routes_to_conversation_open_text():
    p = perceive("hi")
    qm = detect_question_model(p)
    assert p.interaction_mode == "conversation"
    assert p.question_type == "open_text"
    assert p.adaptive_allowed is False
    assert qm.name == "generic_open_text"


def test_qualitative_prompt_routes_to_open_text():
    p = perceive("What do your friends say about you?")
    qm = detect_question_model(p)
    assert p.interaction_mode == "qualitative_interview"
    assert p.question_type == "open_text"
    assert p.adaptive_allowed is False
    assert qm.name == "generic_open_text"


def test_explicit_survey_still_routes_to_structured_mode_after_conversation():
    persona = generate_population(1, method="bayesian", seed=77)[0]
    state = AgentState.from_persona(persona)
    state.record_interaction("hi", "conversation")
    p = perceive("How likely are you to try a new fintech wallet next month?", state=state)
    qm = detect_question_model(p)
    assert p.interaction_mode == "survey"
    assert p.adaptive_allowed is True
    assert qm.name == "tech_adoption_likelihood"


def test_follow_up_inherits_qualitative_interview_mode():
    persona = generate_population(1, method="bayesian", seed=78)[0]
    state = AgentState.from_persona(persona)
    state.record_interaction("What do you do in your free time?", "qualitative_interview")
    p = perceive("What else?", state=state)
    qm = detect_question_model(p)
    assert p.interaction_mode == "qualitative_interview"
    assert qm.name == "generic_open_text"
