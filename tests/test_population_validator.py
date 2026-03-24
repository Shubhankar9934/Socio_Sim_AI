from population.synthesis import generate_population
from population.validator import multimodality_score


def test_multimodality_score_non_zero_for_seeded_bayesian_population():
    personas = generate_population(200, method="bayesian", seed=42, id_prefix="DXB")
    score = multimodality_score(personas)
    assert 0.0 <= score <= 1.0
    assert score > 0.0
