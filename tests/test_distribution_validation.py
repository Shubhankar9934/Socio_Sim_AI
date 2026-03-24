from evaluation.distribution_validation import validate_survey_distribution


def test_validation_includes_reference_metadata():
    responses = [
        {"sampled_option": "daily"},
        {"sampled_option": "daily"},
        {"sampled_option": "rarely"},
    ]
    report = validate_survey_distribution(
        responses,
        question_model_key="food_delivery_frequency",
    )
    assert "reference_source" in report
    assert "mapping_applied" in report

