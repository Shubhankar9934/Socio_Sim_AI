"""
Narrative similarity detection: find duplicate/near-duplicate survey narratives
using sentence embeddings and cosine similarity.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def compute_narrative_similarity(
    narratives: List[str],
    threshold: float = 0.9,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Compute pairwise cosine similarity and report duplicate rate.

    Returns dict with duplicate_rate, duplicate_pairs, mean_similarity, etc.
    Target: duplicate_rate < 0.05 (5%).
    """
    if len(narratives) < 2:
        return {
            "duplicate_rate": 0.0,
            "duplicate_pairs": 0,
            "total_pairs": 0,
            "threshold": threshold,
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
            "flagged_pairs": [],
        }

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(model_name)
    embeddings = model.encode(narratives, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)

    n = len(narratives)
    total_pairs = n * (n - 1) // 2
    duplicate_pairs = 0
    flagged: List[Dict[str, Any]] = []

    upper_triangle = np.triu_indices(n, k=1)
    sims = sim_matrix[upper_triangle]

    duplicate_mask = sims > threshold
    duplicate_pairs = int(duplicate_mask.sum())

    if duplicate_pairs > 0:
        dup_indices = np.where(duplicate_mask)[0]
        for idx in dup_indices[:20]:
            i = int(upper_triangle[0][idx])
            j = int(upper_triangle[1][idx])
            flagged.append({
                "index_a": i,
                "index_b": j,
                "similarity": round(float(sims[idx]), 4),
            })

    np.fill_diagonal(sim_matrix, 0)

    return {
        "duplicate_rate": duplicate_pairs / total_pairs if total_pairs > 0 else 0.0,
        "duplicate_pairs": duplicate_pairs,
        "total_pairs": total_pairs,
        "threshold": threshold,
        "mean_similarity": round(float(sims.mean()), 4),
        "max_similarity": round(float(sims.max()), 4) if len(sims) > 0 else 0.0,
        "flagged_pairs": flagged,
    }
