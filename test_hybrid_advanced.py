#!/usr/bin/env python3
"""
Advanced Hybrid HNSW Comparison

Features:
1. Compare parent→child mapping methods: approx vs brute
2. Evaluate impact of diversification + repair on coverage & recall
3. Export detailed stats and recall metrics to JSON for benchmarking

Output JSON: hybrid_mapping_comparison.json
"""

import os
import sys
import json
import time
from typing import Dict, Any

import numpy as np

# Ensure local module import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hnsw_core'))
from hnsw import HNSW  # type: ignore
from hnsw_hybrid import (
    HNSWHybrid,
    HNSWEvaluator,
    create_synthetic_dataset,
    create_query_set,
)


def build_base_index(dataset: np.ndarray, query_ids, m=16, ef_construction=200):
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base = HNSW(distance_func=distance_func, m=m, ef_construction=ef_construction)
    for i, vec in enumerate(dataset):
        if i not in query_ids:  # exclude query vectors for fairness
            base.insert(i, vec)
    return base, distance_func


def evaluate_variant(name: str, hybrid: HNSWHybrid, evaluator: HNSWEvaluator, k: int, n_probe: int) -> Dict[str, Any]:
    result = evaluator.evaluate_recall(hybrid, k=k, n_probe=n_probe)
    stats = hybrid.get_stats()
    return {
        'mapping_method': stats.get('mapping_method'),
        'stats': stats,
        'recall_at_k': result['recall_at_k'],
        'avg_query_time_ms': result['avg_query_time_ms'],
        'total_correct': result['total_correct'],
        'total_expected': result['total_expected'],
        'k': k,
        'n_probe': n_probe,
        'candidate_size_avg': stats.get('avg_candidate_size'),
        'coverage_fraction': stats.get('coverage_fraction'),
        'mean_jaccard_overlap': stats.get('mean_jaccard_overlap'),
        'median_jaccard_overlap': stats.get('median_jaccard_overlap'),
    }


def main():
    print("=== Advanced Hybrid HNSW Mapping Comparison ===")
    cfg = {
        'dataset_size': 2000,
        'dim': 64,
        'n_queries': 100,
        'k': 10,
        'n_probe': 5,
        'k_children': 200,
        'parent_level': 1,
    }
    print(f"Config: {cfg}")

    # 1. Dataset & queries
    dataset = create_synthetic_dataset(cfg['dataset_size'], cfg['dim'], seed=42)
    query_vectors, query_ids = create_query_set(dataset, cfg['n_queries'], seed=123)

    # 2. Base index (shared)
    print("Building shared base HNSW index...")
    base_index, distance_func = build_base_index(dataset, query_ids, m=16, ef_construction=150)

    # Evaluator & ground truth
    evaluator = HNSWEvaluator(dataset, query_vectors, query_ids)
    ground_truth = evaluator.compute_ground_truth(k=cfg['k'], distance_func=distance_func)

    variants = {}

    # 3a. Approx mapping
    print("\n[Variant] Approx mapping")
    approx_index = HNSWHybrid(
        base_index=base_index,
        parent_level=cfg['parent_level'],
        k_children=cfg['k_children'],
        parent_child_method='approx',
        approx_ef=60,
    )
    variants['approx'] = evaluate_variant('approx', approx_index, evaluator, cfg['k'], cfg['n_probe'])

    # 3b. Brute mapping
    print("\n[Variant] Brute mapping")
    brute_index = HNSWHybrid(
        base_index=base_index,
        parent_level=cfg['parent_level'],
        k_children=cfg['k_children'],
        parent_child_method='brute',
    )
    variants['brute'] = evaluate_variant('brute', brute_index, evaluator, cfg['k'], cfg['n_probe'])

    # 3c. Approx + diversification + repair
    print("\n[Variant] Approx + Diversify + Repair")
    diversify_index = HNSWHybrid(
        base_index=base_index,
        parent_level=cfg['parent_level'],
        k_children=cfg['k_children'],
        parent_child_method='approx',
        approx_ef=60,
        diversify_max_assignments=3,
        repair_min_assignments=1,
    )
    variants['approx_diversified'] = evaluate_variant('approx_diversified', diversify_index, evaluator, cfg['k'], cfg['n_probe'])

    # 4. Comparison summary
    comp = {}
    if 'approx' in variants and 'brute' in variants:
        comp['recall_diff_brute_minus_approx'] = variants['brute']['recall_at_k'] - variants['approx']['recall_at_k']
        comp['coverage_diff_brute_minus_approx'] = variants['brute']['coverage_fraction'] - variants['approx']['coverage_fraction']
    if 'approx_diversified' in variants and 'approx' in variants:
        comp['coverage_gain_diversified'] = variants['approx_diversified']['coverage_fraction'] - variants['approx']['coverage_fraction']
        comp['recall_gain_diversified'] = variants['approx_diversified']['recall_at_k'] - variants['approx']['recall_at_k']

    output = {
        'dataset': {
            'n_vectors': cfg['dataset_size'],
            'dim': cfg['dim'],
            'n_queries': cfg['n_queries'],
        },
        'config': cfg,
        'variants': variants,
        'comparison': comp,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    out_path = os.path.join(os.path.dirname(__file__), 'hybrid_mapping_comparison.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Exported comparison JSON -> {out_path}")
    print("Summary (recall@k):")
    for k, v in variants.items():
        print(f"  {k:20s} recall={v['recall_at_k']:.4f} coverage={v['coverage_fraction']:.3f} avgCand={v['candidate_size_avg']:.1f}")

    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
