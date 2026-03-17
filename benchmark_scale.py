"""
Scale benchmark: profile activation update, cluster detection, and bias
pipeline latency at 10K, 50K, and 100K agents.

Usage:
    python benchmark_scale.py
"""

import time
from typing import Dict, List, Tuple

import numpy as np

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _build_sparse_ba_graph(n: int, m: int = 5, seed: int = 42) -> "sp.csr_matrix":
    """Build a Barabasi-Albert-like sparse adjacency matrix."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    degrees = np.ones(n)
    for i in range(m, n):
        probs = degrees[:i] / degrees[:i].sum()
        targets = rng.choice(i, size=m, replace=False, p=probs)
        for t in targets:
            rows.extend([i, t])
            cols.extend([t, i])
            degrees[i] += 1
            degrees[t] += 1
    data = np.ones(len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _normalize_adj(adj: "sp.csr_matrix") -> "sp.csr_matrix":
    row_sums = np.array(adj.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    diag_inv = sp.diags(1.0 / row_sums)
    return diag_inv @ adj


def benchmark_activation(n: int, n_frames: int = 10) -> float:
    """Time the vectorized activation update for N agents."""
    from simulation.cascade_detector import update_activation, compute_neighbor_activation

    A = np.random.rand(n).astype(np.float64) * 0.5
    exposure = np.random.rand(n).astype(np.float64)
    emotion = np.random.rand(n).astype(np.float64)
    topic_imp = np.random.rand(n).astype(np.float64)
    alignment = np.random.rand(n).astype(np.float64) * 2 - 1
    susceptibility = np.random.rand(n).astype(np.float64)

    adj = _build_sparse_ba_graph(n)
    adj_norm = _normalize_adj(adj)

    t0 = time.perf_counter()
    neighbor_act = compute_neighbor_activation(A, adj_norm)
    _ = update_activation(A, exposure, emotion, topic_imp, alignment, neighbor_act, susceptibility)
    return time.perf_counter() - t0


def benchmark_cluster_detection(n: int) -> float:
    """Time cluster detection for N agents."""
    from simulation.cascade_detector import detect_activation_clusters

    A = np.random.rand(n).astype(np.float64)
    A[: n // 10] = 0.9
    adj = _build_sparse_ba_graph(n)

    t0 = time.perf_counter()
    _ = detect_activation_clusters(A, adj, activation_threshold=0.8, min_size_absolute=50, min_size_fraction=0.001)
    return time.perf_counter() - t0


def benchmark_bias_pipeline(n: int) -> float:
    """Time the bias pipeline for N agents (sequential, per-agent)."""
    from agents.biases import apply_all_biases

    scale = ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]
    dist = {s: 1.0 / len(scale) for s in scale}

    class FakeState:
        base_malleability = 0.5
        calcification = 0.1
        knowledge_levels = {"topic": 0.5}
        beliefs = None
        latent_state = None

    ctx = {"topic": "topic", "topic_importance": 0.5, "media_conflict": 0.3}

    t0 = time.perf_counter()
    for _ in range(min(n, 1000)):
        apply_all_biases(dist, scale, FakeState(), ctx)
    per_agent = (time.perf_counter() - t0) / min(n, 1000)
    return per_agent * n


def benchmark_attention(n: int, k: int = 10) -> float:
    """Time adaptive attention for N agents, K frames."""
    from media.attention import adaptive_attention

    A = np.random.rand(n).astype(np.float64)
    exposure = np.random.rand(n, k).astype(np.float64)
    emotion = np.random.rand(n, k).astype(np.float64)

    t0 = time.perf_counter()
    _ = adaptive_attention(A, exposure, emotion)
    return time.perf_counter() - t0


def run_benchmarks() -> None:
    scales = [10_000, 50_000, 100_000]

    print("=" * 70)
    print(f"{'JADU Scale Benchmark':^70}")
    print("=" * 70)

    for n in scales:
        print(f"\n--- {n:,} agents ---")

        t = benchmark_activation(n)
        print(f"  Activation update:    {t*1000:8.1f} ms")

        if HAS_SCIPY:
            t = benchmark_cluster_detection(n)
            print(f"  Cluster detection:    {t*1000:8.1f} ms")

        t = benchmark_attention(n)
        print(f"  Adaptive attention:   {t*1000:8.1f} ms")

        t = benchmark_bias_pipeline(n)
        print(f"  Bias pipeline (est):  {t*1000:8.1f} ms")

    print("\n" + "=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    run_benchmarks()
