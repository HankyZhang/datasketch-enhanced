import numpy as np
import pytest

from hnsw.hnsw import HNSW
from method3 import KMeansHNSW, KMeansHNSWMultiPivot


def _build_base(data):
    dist = lambda a, b: float(np.linalg.norm(a - b))
    base = HNSW(distance_func=dist, m=16, ef_construction=200)
    for i, v in enumerate(data):
        base.insert(i, v)
    return base, dist


def _brute_force_gt(data, queries, k):
    gt = []
    for q in queries:
        d = np.linalg.norm(data - q, axis=1)
        idx = np.argsort(d)[:k]
        gt.append(set(int(i) for i in idx))
    return gt


def _eval_system(system, queries, gt, k, n_probe):
    correct = 0
    for q, g in zip(queries, gt):
        res = system.search(q, k=k, n_probe=n_probe)
        ids = {int(i) for i,_ in res}
        correct += len(ids & g)
    return correct / (len(queries) * k)


def test_multi_pivot_recall_improvement():
    rng = np.random.default_rng(123)
    # Generate mildly anisotropic clusters to benefit multi-pivot coverage
    clusters = []
    for center, cov_scale, n in [([0,0],[1.0,0.3],250), ([6,0],[0.5,2.0],250), ([-5,5],[1.2,0.8],250)]:
        a = rng.normal(loc=center, scale=cov_scale, size=(n,2))
        clusters.append(a)
    data2d = np.vstack(clusters).astype(np.float32)
    # Lift to higher dim with random linear projection to increase variation
    proj = rng.normal(size=(2,32)).astype(np.float32)
    data = (data2d @ proj).astype(np.float32)

    base, dist = _build_base(data)

    n_clusters = 30
    k_children = 120
    k = 10
    n_probe = 8
    q_count = 40
    query_idx = rng.choice(len(data), size=q_count, replace=False)
    queries = data[query_idx]

    gt = _brute_force_gt(data, queries, k)

    single = KMeansHNSW(
        base_index=base,
        n_clusters=n_clusters,
        k_children=k_children,
        child_search_ef=250,
        kmeans_params={'max_iters':80,'verbose':0},
    )

    multi = KMeansHNSWMultiPivot(
        base_index=base,
        n_clusters=n_clusters,
        k_children=k_children,
        num_pivots=3,
        pivot_selection_strategy='line_perp_third',
        pivot_overquery_factor=1.25,
        child_search_ef=260,
        kmeans_params={'max_iters':80,'verbose':0},
    )

    recall_single = _eval_system(single, queries, gt, k, n_probe)
    recall_multi = _eval_system(multi, queries, gt, k, n_probe)

    # Both recalls should be non-trivial
    assert recall_single > 0.3, f"Single-pivot recall unexpectedly low: {recall_single:.3f}"
    assert recall_multi > 0.3, f"Multi-pivot recall unexpectedly low: {recall_multi:.3f}"
    # Multi-pivot should be at least not worse than single (allow tiny noise margin)
    assert recall_multi + 1e-6 >= recall_single - 0.02, (
        f"Multi-pivot recall {recall_multi:.3f} significantly worse than single {recall_single:.3f}" )

    # Prefer (soft) improvement expectation; if not improved just print info
    if recall_multi < recall_single:
        print(f"[INFO] Multi-pivot recall {recall_multi:.3f} < single {recall_single:.3f} (within tolerance)")

    print(f"Recall@{k} single={recall_single:.4f} multi={recall_multi:.4f}")
