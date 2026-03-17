"""
Distribution validation: compare observed survey distributions against
reference market data to check if the probabilistic model is calibrated.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare


from config.reference_distributions import REFERENCE_DISTRIBUTIONS, get_reference_distribution

DEFAULT_REFERENCE: Dict[str, float] = get_reference_distribution("generic_frequency")
DUBAI_FOOD_DELIVERY_REFERENCE = DEFAULT_REFERENCE


def aggregate_survey_distribution(
    responses: List[Dict[str, Any]],
    key: str = "sampled_option",
) -> Dict[str, float]:
    """Aggregate responses into observed proportions."""
    counts: Dict[str, int] = {}
    total = 0
    for r in responses:
        option = r.get(key)
        if option is not None:
            counts[option] = counts.get(option, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def compare_to_reference(
    observed: Dict[str, float],
    reference: Optional[Dict[str, float]] = None,
    significance: float = 0.05,
) -> Dict[str, Any]:
    """Compare observed vs reference distribution.

    Returns JS divergence, chi-square p-value, per-option diffs, and pass/fail.
    """
    reference = reference or DEFAULT_REFERENCE
    all_keys = sorted(set(list(observed.keys()) + list(reference.keys())))

    obs_arr = np.array([observed.get(k, 0.0) for k in all_keys])
    ref_arr = np.array([reference.get(k, 0.0) for k in all_keys])

    obs_arr = obs_arr / obs_arr.sum() if obs_arr.sum() > 0 else obs_arr
    ref_arr = ref_arr / ref_arr.sum() if ref_arr.sum() > 0 else ref_arr

    js_div = float(jensenshannon(obs_arr, ref_arr))
    js_similarity = 1.0 - js_div

    ref_for_chi = ref_arr.copy()
    ref_for_chi[ref_for_chi == 0] = 1e-10
    obs_for_chi = obs_arr * 1000
    ref_for_chi_scaled = ref_for_chi * 1000

    try:
        chi2_stat, chi2_p = chisquare(obs_for_chi, f_exp=ref_for_chi_scaled)
        chi2_p = float(chi2_p)
    except Exception:
        chi2_p = 0.0

    per_option = {
        k: {
            "observed": round(observed.get(k, 0.0), 4),
            "reference": round(reference.get(k, 0.0), 4),
            "diff": round(observed.get(k, 0.0) - reference.get(k, 0.0), 4),
        }
        for k in all_keys
    }

    passed = js_similarity >= 0.85

    return {
        "js_divergence": round(js_div, 4),
        "js_similarity": round(js_similarity, 4),
        "chi_square_p_value": round(chi2_p, 4),
        "chi_square_significant": chi2_p < significance,
        "per_option": per_option,
        "passed": passed,
    }


def validate_survey_distribution(
    responses: List[Dict[str, Any]],
    reference: Optional[Dict[str, float]] = None,
    key: str = "sampled_option",
    question_model_key: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end: aggregate responses and compare to reference.

    If no explicit reference is given but question_model_key is provided,
    auto-resolves from the reference distribution registry.
    """
    if reference is None and question_model_key:
        reference = get_reference_distribution(question_model_key) or None
    observed = aggregate_survey_distribution(responses, key=key)
    comparison = compare_to_reference(observed, reference)
    comparison["observed_distribution"] = observed
    comparison["n_responses"] = len(responses)
    if question_model_key:
        comparison["question_model_key"] = question_model_key
    return comparison
