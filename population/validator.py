"""
Validation of synthetic population vs target distributions.
KL divergence, Jensen-Shannon divergence, chi-square goodness-of-fit.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import chisquare
from scipy.stats import entropy as scipy_entropy

from config.demographics import get_demographics
from population.personas import Persona


def _distribution_from_personas(
    personas: List[Persona],
    attribute: str,
) -> Dict[str, float]:
    """Empirical distribution of attribute across personas."""
    values = []
    for p in personas:
        if attribute == "age":
            values.append(p.age)
        elif attribute == "nationality":
            values.append(p.nationality)
        elif attribute == "income":
            values.append(p.income)
        elif attribute == "location":
            values.append(p.location)
        elif attribute == "household_size":
            values.append(p.household_size)
        elif attribute == "occupation":
            values.append(p.occupation)
        else:
            raise ValueError(f"Unknown attribute: {attribute}")
    counts = pd.Series(values).value_counts(normalize=True)
    return counts.to_dict()


def _align_distributions(
    target: Dict[str, float],
    empirical: Dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (target_probs, empirical_probs) over same ordered keys."""
    keys = sorted(set(target.keys()) | set(empirical.keys()))
    target_probs = np.array([target.get(k, 0.0) for k in keys])
    empirical_probs = np.array([empirical.get(k, 0.0) for k in keys])
    # Normalize
    target_probs = target_probs / target_probs.sum()
    empirical_probs = empirical_probs / (empirical_probs.sum() or 1e-10)
    return target_probs, empirical_probs


def kl_divergence(target: Dict[str, float], empirical: Dict[str, float]) -> float:
    """KL(target || empirical). Lower is better. Returns bits if base 2."""
    p, q = _align_distributions(target, empirical)
    # Avoid zeros
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(scipy_entropy(p, q, base=2))


def jensen_shannon_divergence(
    target: Dict[str, float],
    empirical: Dict[str, float],
) -> float:
    """Jensen-Shannon divergence in [0,1]. Lower is better."""
    p, q = _align_distributions(target, empirical)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(jensenshannon(p, q))


def population_realism_score(
    personas: List[Persona],
    marginals: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """
    Aggregate realism score in [0, 1]. Higher is better.
    Uses 1 - mean(JS divergence) over attributes.
    """
    if marginals is None:
        marginals = get_demographics().get_all_marginals()
    scores = []
    for attr, target_dist in marginals.items():
        empirical = _distribution_from_personas(personas, attr)
        js = jensen_shannon_divergence(target_dist, empirical)
        scores.append(1.0 - js)
    return float(np.mean(scores)) if scores else 0.0


def chi_square_test(
    personas: List[Persona],
    attribute: str,
    target: Dict[str, float],
) -> tuple[float, float, bool]:
    """
    Chi-square goodness-of-fit. Returns (chi2, p_value, reject_null).
    reject_null=True means synthetic distribution differs significantly from target.
    """
    empirical = _distribution_from_personas(personas, attribute)
    keys = sorted(set(target.keys()) | set(empirical.keys()))
    observed = np.array([empirical.get(k, 0) * len(personas) for k in keys])
    expected = np.array([target.get(k, 0) * len(personas) for k in keys])
    expected = np.clip(expected, 0.1, None)
    chi2, p = chisquare(observed, expected)
    # Reject null (distributions differ) if p < 0.05
    return float(chi2), float(p), p < 0.05


def segment_distribution(personas: List[Persona]) -> Dict[str, float]:
    """Return the empirical distribution of population segments."""
    segments = [getattr(p.meta, "population_segment", None) or "unassigned" for p in personas]
    counts = pd.Series(segments).value_counts(normalize=True)
    return counts.to_dict()


def multimodality_score(personas: List[Persona]) -> float:
    """Estimate multimodality of the latent state distribution via GMM BIC.

    Fits 1-component and K-component Gaussian Mixture Models to the
    population's latent trait vectors and returns a 0-1 score where
    higher means stronger multimodality.  Returns 0.0 if sklearn or
    sufficient data is unavailable.
    """
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return 0.0

    from agents.personality import personality_from_persona
    from agents.behavior import init_from_persona as _init_latent

    if len(personas) < 30:
        return 0.0

    vectors = []
    for p in personas:
        traits = personality_from_persona(p)
        latent = _init_latent(p, traits)
        vectors.append(latent.to_vector())
    X = np.vstack(vectors)

    n_segments = len(set(getattr(p.meta, "population_segment", "x") for p in personas))
    k_max = max(2, min(n_segments, 6))
    covariance_types = ("diag", "tied", "spherical", "full")

    best_single_bic = np.inf
    best_multi_bic = np.inf
    best_multi_labels = None

    for cov in covariance_types:
        try:
            gmm_1 = GaussianMixture(n_components=1, covariance_type=cov, random_state=42).fit(X)
            bic_1 = float(gmm_1.bic(X))
            best_single_bic = min(best_single_bic, bic_1)
        except Exception:
            continue

        for k in range(2, k_max + 1):
            try:
                gmm_k = GaussianMixture(n_components=k, covariance_type=cov, random_state=42).fit(X)
                bic_k = float(gmm_k.bic(X))
                if bic_k < best_multi_bic:
                    best_multi_bic = bic_k
                    best_multi_labels = gmm_k.predict(X)
            except Exception:
                continue

    if not np.isfinite(best_single_bic) or not np.isfinite(best_multi_bic):
        return 0.0

    # Lower BIC is better; positive delta indicates multimodal model evidence.
    bic_delta = best_single_bic - best_multi_bic
    if bic_delta <= 0:
        return 0.0

    # Normalize by single-model scale to avoid sign/ratio pathologies.
    scale = max(abs(best_single_bic), 1.0)
    bic_score = float(np.clip(bic_delta / (0.08 * scale), 0.0, 1.0))

    if best_multi_labels is None or len(set(best_multi_labels.tolist())) < 2:
        return bic_score
    try:
        sil = float(silhouette_score(X, best_multi_labels))
        sil_score = float(np.clip((sil + 1.0) / 2.0, 0.0, 1.0))
    except Exception:
        sil_score = 0.5

    return float(np.clip(0.75 * bic_score + 0.25 * sil_score, 0.0, 1.0))


def validate_population(
    personas: List[Persona],
    realism_threshold: float = 0.85,
) -> tuple[bool, float, Dict[str, float]]:
    """
    Validate synthetic population. Returns (passed, realism_score, per_attribute_js).
    passed = realism_score >= realism_threshold.
    Includes segment distribution and multimodality as bonus diagnostics.
    """
    marginals = get_demographics().get_all_marginals()
    per_attr: Dict[str, float] = {}
    for attr, target_dist in marginals.items():
        empirical = _distribution_from_personas(personas, attr)
        js = jensen_shannon_divergence(target_dist, empirical)
        per_attr[attr] = 1.0 - js

    mm = multimodality_score(personas)
    per_attr["multimodality"] = mm

    seg_dist = segment_distribution(personas)
    per_attr["segment_entropy"] = float(scipy_entropy(list(seg_dist.values())))

    score = float(np.mean([v for k, v in per_attr.items()
                           if k not in ("multimodality", "segment_entropy")]))
    passed = score >= realism_threshold
    return passed, score, per_attr
