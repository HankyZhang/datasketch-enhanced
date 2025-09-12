"""Large-Scale Unoptimized Hybrid HNSW Benchmark

Purpose:
    Evaluate the *unoptimized* (no diversification / repair) HybridHNSWIndex behavior
    on large datasets (e.g. 600k vectors) across parameter grids, with focus on
    how increasing ef_construction (and thus graph quality) and k_children / n_probe
    affect recall.

Notes:
    1. This script intentionally DOES NOT use diversification or repair parameters.
    2. Ground truth on 600k * full query set brute-force is prohibitive; we sample queries
       and compute brute-force distances in batches.
    3. Expect long build times at high ef_construction; schedule accordingly.
    4. Run in a machine with sufficient RAM (>= 8GB recommended for 600k × 128 float32).

Example (PowerShell):
    python large_unoptimized_benchmark.py --dataset-size 600000 --queries 600 \
        --k-children 500 800 1000 1500 --n-probe 5 8 10 12 \
        --efc 100 150 200 300 --k 10 --out unoptimized_600k.csv --dim 128 --seed 42

Quick test (small):
    python large_unoptimized_benchmark.py --dataset-size 20000 --queries 300 --efc 100 150 --k-children 500 800 --n-probe 5 10 --quick
"""

import argparse
import csv
import os
import time
from typing import List, Dict
import numpy as np

from hnsw_hybrid_evaluation import (
    HybridHNSWIndex,
    generate_synthetic_dataset,
    split_query_set_from_dataset,
    RecallEvaluator,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-size', type=int, required=True, help='Total dataset size (e.g. 600000)')
    p.add_argument('--dim', type=int, default=128, help='Vector dimensionality')
    p.add_argument('--queries', type=int, default=600, help='Number of queries to evaluate (sampled)')
    p.add_argument('--k', type=int, default=10, help='Recall@k target')
    p.add_argument('--k-children', type=int, nargs='+', default=[800, 1000, 1500], help='List of k_children values')
    p.add_argument('--n-probe', type=int, nargs='+', default=[5, 8, 10, 12], help='List of n_probe values')
    p.add_argument('--efc', type=int, nargs='+', default=[100, 150, 200], help='List of ef_construction values')
    p.add_argument('--m', type=int, nargs='+', default=[16], help='Graph M values to test (usually one)')
    p.add_argument('--parent-level', type=int, default=2, help='Parent extraction level')
    p.add_argument('--mapping-ef', type=int, default=50, help='ef used for approx parent->child mapping')
    p.add_argument('--sample-query-eval', type=int, default=None, help='Optional further sample of queries for recall eval (subset of --queries)')
    p.add_argument('--batch-ground-truth', type=int, default=50000, help='Batch size for brute-force distance blocks when computing ground truth')
    p.add_argument('--out', default='unoptimized_large_results.csv', help='Output CSV file')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true', help='Quick mode: limit grid to first value of each list')
    return p.parse_args()


def ensure_header(path: str, header: List[str]):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(header)


def append_row(path: str, header: List[str], row: Dict[str, object]):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([row.get(h, '') for h in header])


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Adjust grids for quick mode
    k_children_values = [args.k_children[0]] if args.quick else args.k_children
    n_probe_values = [args.n_probe[0]] if args.quick else args.n_probe
    efc_values = [args.efc[0]] if args.quick else args.efc
    m_values = [args.m[0]] if args.quick else args.m

    print("=== Large Unoptimized Hybrid Benchmark ===")
    print(f"Dataset: {args.dataset_size} dim={args.dim} queries={args.queries} k={args.k}")
    print(f"Grids: k_children={k_children_values} n_probe={n_probe_values} efc={efc_values} m={m_values}")
    if args.quick:
        print("[Quick] Only first value of each list used.")

    # Generate dataset & split queries
    t0 = time.time()
    full_dataset = generate_synthetic_dataset(args.dataset_size, args.dim)
    # Use queries as withheld set for fairness
    base_dataset, query_set = split_query_set_from_dataset(full_dataset, args.queries)
    print(f"Dataset generated & split in {time.time()-t0:.1f}s | base={len(base_dataset)} queries={len(query_set)}")

    # Optionally shrink evaluation queries further
    if args.sample_query_eval and args.sample_query_eval < len(query_set):
        q_ids = list(query_set.keys())
        np.random.shuffle(q_ids)
        selected = set(q_ids[:args.sample_query_eval])
        query_set = {qid: query_set[qid] for qid in selected}
        print(f"Sampled {len(query_set)} queries for evaluation (subset).")

    evaluator = RecallEvaluator(base_dataset)

    header = [
        'dataset_size','dim','n_queries','k','m','ef_construction','k_children','n_probe',
        'mapping_ef','parent_level','recall@k','avg_query_time','base_build_time','parent_extraction_time',
        'mapping_build_time','avg_candidate_size','coverage_fraction','parent_count','covered_points','total_points',
        'overlap_unique_fraction','avg_assignment_count','mean_jaccard_overlap','median_jaccard_overlap',
        'multi_coverage_fraction','max_assignment_count','timestamp'
    ]
    ensure_header(args.out, header)

    for m in m_values:
        for efc in efc_values:
            # Build base index once per (m, efc)
            print(f"\n[Build] m={m} ef_construction={efc}")
            build_start = time.time()
            index_base = HybridHNSWIndex(k_children=0, n_probe=0)  # temporary placeholder
            index_base.build_base_index(base_dataset, m=m, ef_construction=efc)
            build_time_total = time.time() - build_start
            # Extract parents once
            index_base.extract_parent_nodes(target_level=args.parent_level)
            parent_ids = index_base.parent_ids
            print(f"Parents extracted: {len(parent_ids)}")
            # Loop over search-time params (k_children, n_probe)
            for k_children in k_children_values:
                for n_probe in n_probe_values:
                    label = f"m={m} efc={efc} kch={k_children} np={n_probe}"
                    print(f"  [Config] {label}")
                    try:
                        # Clone minimal state (reuse base index & parent sets)
                        hybrid = HybridHNSWIndex(k_children=k_children, n_probe=n_probe, parent_child_method='approx')
                        # Reuse built index & dataset
                        hybrid.base_index = index_base.base_index
                        hybrid.dataset = base_dataset
                        hybrid.parent_ids = parent_ids
                        hybrid.parent_vectors = {pid: base_dataset[pid] for pid in parent_ids}
                        if parent_ids:
                            hybrid._parent_matrix = np.stack([hybrid.parent_vectors[pid] for pid in parent_ids], axis=0)
                        hybrid.parent_extraction_time = index_base.parent_extraction_time
                        hybrid.base_build_time = index_base.base_build_time
                        # Build parent->child mapping (UNOPTIMIZED: no diversify/repair)
                        hybrid.build_parent_child_mapping(method='approx', ef=args.mapping_ef,
                                                           diversify_max_assignments=None,
                                                           repair_min_assignments=None)
                        # Warmup one query
                        first_qid, first_qvec = next(iter(query_set.items()))
                        hybrid.search(first_qvec, k=args.k)
                        # Evaluate recall
                        results = evaluator.evaluate_recall(hybrid, query_set, k=args.k,
                                                            progress_interval=max(50, len(query_set)//10))
                        stats = hybrid.stats()
                        row = {
                            'dataset_size': args.dataset_size,
                            'dim': args.dim,
                            'n_queries': len(query_set),
                            'k': args.k,
                            'm': m,
                            'ef_construction': efc,
                            'k_children': k_children,
                            'n_probe': n_probe,
                            'mapping_ef': args.mapping_ef,
                            'parent_level': args.parent_level,
                            'recall@k': results['recall@k'],
                            'avg_query_time': results['avg_query_time'],
                            'base_build_time': stats['base_build_time'],
                            'parent_extraction_time': stats['parent_extraction_time'],
                            'mapping_build_time': stats['mapping_build_time'],
                            'avg_candidate_size': stats['avg_candidate_size'],
                            'coverage_fraction': stats['coverage_fraction'],
                            'parent_count': stats['parent_count'],
                            'covered_points': stats['covered_points'],
                            'total_points': stats['total_points'],
                            'overlap_unique_fraction': stats.get('overlap_unique_fraction',''),
                            'avg_assignment_count': stats.get('avg_assignment_count',''),
                            'mean_jaccard_overlap': stats.get('mean_jaccard_overlap',''),
                            'median_jaccard_overlap': stats.get('median_jaccard_overlap',''),
                            'multi_coverage_fraction': stats.get('multi_coverage_fraction',''),
                            'max_assignment_count': stats.get('max_assignment_count',''),
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        append_row(args.out, header, row)
                        print(f"    -> recall@{args.k}={row['recall@k']:.4f} cand={row['avg_candidate_size']:.1f} cov={row['coverage_fraction']:.3f}")
                    except Exception as e:
                        print(f"    [ERROR] {label} failed: {e}")
                        # record failure
                        fail_row = {
                            'dataset_size': args.dataset_size,
                            'dim': args.dim,
                            'n_queries': len(query_set),
                            'k': args.k,
                            'm': m,
                            'ef_construction': efc,
                            'k_children': k_children,
                            'n_probe': n_probe,
                            'mapping_ef': args.mapping_ef,
                            'parent_level': args.parent_level,
                            'recall@k': 0.0,
                            'avg_query_time': 0.0,
                            'base_build_time': index_base.base_build_time,
                            'parent_extraction_time': index_base.parent_extraction_time,
                            'mapping_build_time': 0.0,
                            'avg_candidate_size': 0.0,
                            'coverage_fraction': 0.0,
                            'parent_count': len(parent_ids),
                            'covered_points': 0,
                            'total_points': len(base_dataset),
                            'overlap_unique_fraction': '',
                            'avg_assignment_count': '',
                            'mean_jaccard_overlap': '',
                            'median_jaccard_overlap': '',
                            'multi_coverage_fraction': '',
                            'max_assignment_count': '',
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        append_row(args.out, header, fail_row)
    print("\nBenchmark complete. Results ->", args.out)


if __name__ == '__main__':
    main()
