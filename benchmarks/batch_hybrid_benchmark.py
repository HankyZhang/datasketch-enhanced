"""Batch Hybrid HNSW Benchmark Runner

Generates recall & performance metrics across dataset sizes and parameter grids.
Writes intermediate CSV rows so partial results survive crashes.

Usage (PowerShell):
    python batch_hybrid_benchmark.py --out results_hybrid.csv

Optional quick mode:
    python batch_hybrid_benchmark.py --quick
"""
import argparse
import csv
import os
import sys
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
    p.add_argument('--out', default='hybrid_benchmark_results.csv', help='Output CSV file')
    p.add_argument('--quick', action='store_true', help='Use reduced grid for a fast dry run')
    p.add_argument('--k', type=int, default=10, help='Recall@k')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--sample-queries', type=int, default=None, help='Sample this many queries per config (None=all)')
    p.add_argument('--parent-level', type=int, default=2, help='Target HNSW level for parents')
    p.add_argument('--mapping-ef', type=int, default=50, help='ef for approx parent->child mapping queries')
    p.add_argument('--diversify-max', type=int, default=None, help='Cap assignments per point during initial mapping')
    p.add_argument('--repair-min', type=int, default=None, help='Ensure each point appears in at least this many parent lists (repair phase)')
    return p.parse_args()


def ensure_header(path: str, header: List[str]):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_row(path: str, row: Dict[str, object], header: List[str]):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([row.get(h, '') for h in header])


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Parameter grids
    if args.quick:
        dataset_sizes = [5000]
        k_children_list = [500, 1000]
        n_probe_list = [5, 10]
    else:
        dataset_sizes = [5000, 20000, 50000]
        k_children_list = [500, 1000, 1500]
        n_probe_list = [5, 10, 15, 20]

    mapping_methods_all = ['approx']  # brute optional limited below

    header = [
        'dataset_size','n_queries','k','k_children','n_probe','method','parent_level','mapping_ef',
        'diversify_max','repair_min',
        'recall@k','avg_query_time','base_build_time','parent_extraction_time','mapping_build_time',
        'avg_candidate_size','coverage_fraction','parent_count','covered_points','total_points',
        'overlap_unique_fraction','avg_assignment_count','mean_jaccard_overlap','median_jaccard_overlap','multi_coverage_fraction','max_assignment_count',
        'query_sampled','timestamp'
    ]
    ensure_header(args.out, header)

    print("=== Hybrid Benchmark Start ===")
    print(f"Output: {args.out}")
    print(f"Quick mode: {args.quick}")

    for ds_size in dataset_sizes:
        start_ds = time.time()
        # Derive queries count
        n_queries = min(1000, max(100, ds_size // 20))  # 5% capped at 1000
        print(f"\n--- Dataset {ds_size} vectors (queries={n_queries}) ---")
        full_dataset = generate_synthetic_dataset(ds_size, 128)
        train_dataset, query_set = split_query_set_from_dataset(full_dataset, n_queries)

        # Precompute optional sampled query IDs
        if args.sample_queries:
            q_ids = list(query_set.keys())
            np.random.shuffle(q_ids)
            sample_query_ids = q_ids[:args.sample_queries]
            print(f"Using sampled {len(sample_query_ids)} queries for evaluation")
        else:
            sample_query_ids = None

        evaluator = RecallEvaluator(train_dataset)

        parent_level = args.parent_level
        for k_children in k_children_list:
            for n_probe in n_probe_list:
                # Decide mapping methods (add brute only on smallest dataset & small k_children to limit cost)
                mapping_methods = list(mapping_methods_all)
                if ds_size == dataset_sizes[0] and k_children in (500, 1000) and not args.quick:
                    mapping_methods.append('brute')

                for method in mapping_methods:
                    cfg_label = f"ds={ds_size} kch={k_children} np={n_probe} m={method}"
                    print(f"\n[Config] {cfg_label}")
                    try:
                        index = HybridHNSWIndex(k_children=k_children, n_probe=n_probe, parent_child_method=method)
                        index.build_base_index(train_dataset)
                        index.extract_parent_nodes(target_level=parent_level)
                        index.build_parent_child_mapping(method=method, ef=args.mapping_ef,
                                                          diversify_max_assignments=args.diversify_max,
                                                          repair_min_assignments=args.repair_min)

                        # Warmup (optional)
                        first_qid, first_qvec = next(iter(query_set.items()))
                        index.search(first_qvec, k=args.k)

                        # Evaluate
                        results = evaluator.evaluate_recall(
                            index,
                            query_set,
                            k=args.k,
                            progress_interval=max(50, (len(query_set) // 10)),
                            sample_query_ids=sample_query_ids
                        )
                        stats = index.stats()
                        row = {
                            'dataset_size': ds_size,
                            'n_queries': len(query_set),
                            'k': args.k,
                            'k_children': k_children,
                            'n_probe': n_probe,
                            'method': method,
                            'parent_level': parent_level,
                            'mapping_ef': args.mapping_ef,
                            'diversify_max': args.diversify_max,
                            'repair_min': args.repair_min,
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
                            'query_sampled': sample_query_ids is not None,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        append_row(args.out, row, header)
                        print(f"[DONE] {cfg_label} recall@{args.k}={row['recall@k']:.4f} avg_q={row['avg_query_time']:.6f}s coverage={row['coverage_fraction']:.3f}")
                    except Exception as e:  # noqa
                        print(f"[ERROR] {cfg_label} failed: {e}")
                        # Append a failure row for traceability
                        fail_row = {
                            'dataset_size': ds_size,
                            'n_queries': len(query_set),
                            'k': args.k,
                            'k_children': k_children,
                            'n_probe': n_probe,
                            'method': method,
                            'parent_level': parent_level,
                            'mapping_ef': args.mapping_ef,
                            'diversify_max': args.diversify_max,
                            'repair_min': args.repair_min,
                            'recall@k': 0.0,
                            'avg_query_time': 0.0,
                            'base_build_time': 0.0,
                            'parent_extraction_time': 0.0,
                            'mapping_build_time': 0.0,
                            'avg_candidate_size': 0.0,
                            'coverage_fraction': 0.0,
                            'parent_count': 0,
                            'covered_points': 0,
                            'total_points': len(train_dataset),
                            'overlap_unique_fraction': '',
                            'avg_assignment_count': '',
                            'mean_jaccard_overlap': '',
                            'median_jaccard_overlap': '',
                            'multi_coverage_fraction': '',
                            'max_assignment_count': '',
                            'query_sampled': sample_query_ids is not None,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        append_row(args.out, fail_row, header)
        print(f"Dataset {ds_size} completed in {time.time()-start_ds:.1f}s")

    print("\n=== Benchmark Complete ===")
    print(f"Results saved to {args.out}")

if __name__ == '__main__':
    main()
