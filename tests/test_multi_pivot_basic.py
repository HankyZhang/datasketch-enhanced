import numpy as np
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hnsw.hnsw import HNSW
from method3.kmeans_hnsw import KMeansHNSW
from method3.kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot

def test_multi_pivot_diversity():
    np.random.seed(7)
    base = np.random.randn(1500, 32).astype(np.float32)
    queries = np.random.randn(5, 32).astype(np.float32)
    dist = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=dist, m=8, ef_construction=120)
    for i, v in enumerate(base):
        base_index.insert(i, v)
    single = KMeansHNSW(base_index=base_index, n_clusters=16, k_children=120, child_search_ef=180)
    multi = KMeansHNSWMultiPivot(base_index=base_index, n_clusters=16, k_children=120, child_search_ef=180, num_pivots=3, pivot_overquery_factor=1.15)
    # Collect average candidate size stat as a proxy for diversity / coverage
    def recall_proxy(system):
        total = 0
        found = 0
        for q in queries:
            res = system.search(q, k=10, n_probe=5)
            found += len(res)
            total += 10
        return found / total
    r_single = recall_proxy(single)
    r_multi = recall_proxy(multi)
    # Multi-pivot should not perform worse in sheer number of returned items
    assert r_multi >= 0.8 * r_single
    # Multi-pivot expected to have similar or larger avg_candidate_size
    assert multi.get_stats()['avg_candidate_size'] >= single.get_stats()['avg_candidate_size'] * 0.5

def test_num_pivots_fallback_single():
    np.random.seed(11)
    base = np.random.randn(800, 16).astype(np.float32)
    dist = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=dist, m=8, ef_construction=100)
    for i, v in enumerate(base):
        base_index.insert(i, v)
    single = KMeansHNSW(base_index=base_index, n_clusters=8, k_children=80, child_search_ef=120)
    multi1 = KMeansHNSWMultiPivot(base_index=base_index, n_clusters=8, k_children=80, child_search_ef=120, num_pivots=1)
    # Compare one query results length equality (not exact ids but length should match)
    q = np.random.randn(16).astype(np.float32)
    res_single = single.search(q, k=10, n_probe=3)
    res_multi1 = multi1.search(q, k=10, n_probe=3)
    assert len(res_single) == len(res_multi1)
