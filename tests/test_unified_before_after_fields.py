import numpy as np
import time
from hnsw.hnsw import HNSW
from method3.tune_kmeans_hnsw_optimized import OptimizedKMeansHNSWMultiPivotEvaluator


def build_small_index(n=120, dim=32):
    base_vectors = np.random.randn(n, dim).astype(np.float32)
    query_vectors = base_vectors[:5].copy()
    query_ids = list(range(len(query_vectors)))
    dist = lambda a,b: np.linalg.norm(a-b)
    index = HNSW(distance_func=dist, m=8, ef_construction=64)
    for i, v in enumerate(base_vectors):
        index.insert(i, v)
    return base_vectors, query_vectors, query_ids, index, dist


def test_all_methods_before_after_present():
    base_vectors, query_vectors, query_ids, index, dist = build_small_index()
    evaluator = OptimizedKMeansHNSWMultiPivotEvaluator(base_vectors, query_vectors, query_ids, dist)
    param_grid = {
        'n_clusters': [4],
        'k_children': [20],
        'child_search_ef': [80]
    }
    evaluation_params = {
        'k_values': [5],
        'n_probe_values': [2],
        'hybrid_parent_level': 2,
        'enable_hybrid': True
    }
    adaptive_config = {
        'adaptive_k_children': False,
        'k_children_scale': 1.5,
        'k_children_min': 10,
        'k_children_max': None,
        'diversify_max_assignments': None,
        'repair_min_assignments': None,
        'overlap_sample': 10
    }
    multi_pivot_config = {
        'enabled': True,
        'num_pivots': 3,
        'pivot_selection_strategy': 'line_perp_third',
        'pivot_overquery_factor': 1.2
    }
    results = evaluator.optimized_parameter_sweep(
        index,
        param_grid,
        evaluation_params,
        adaptive_config=adaptive_config,
        multi_pivot_config=multi_pivot_config
    )
    assert results, "No sweep results returned"
    combo = results[0]
    methods = combo['methods_unified']
    # Expect 5 method families (baseline, pure kmeans, hybrid, single pivot, multi pivot)
    found_families = set()
    for key, val in methods.items():
        fam = val['method']
        found_families.add(fam)
        assert 'before_repair' in val and val['before_repair'] is not None, f"Missing before_repair for {key}"
        assert 'after_repair' in val and val['after_repair'] is not None, f"Missing after_repair for {key}"
        # basic required stats inside snapshots
        for snap_name in ['before_repair','after_repair']:
            snap = val[snap_name]
            assert 'coverage_fraction' in snap, f"coverage_fraction missing in {snap_name} for {key}"
            assert 'duplication_rate' in snap, f"duplication_rate missing in {snap_name} for {key}"
    expected = {'hnsw_baseline','pure_kmeans','hybrid_hnsw','kmeans_hnsw_single_pivot','kmeans_hnsw_multi_pivot'}
    assert expected.issubset(found_families), f"Missing method families: {expected - found_families}"
