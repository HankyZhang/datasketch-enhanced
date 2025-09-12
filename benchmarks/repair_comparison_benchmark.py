"""Repair-Enabled Hybrid HNSW Benchmark (Comparison with Unoptimized Mapping)

Purpose:
    Run a single configuration (same efc / mapping_ef / k_children as prior unoptimized run)
    but WITH repair_min_assignments=1 to force coverage=1.0 and measure recall & candidate cost.

Configuration (default):
    dataset_size = 5000 (4500 indexed + 500 queries)
    ef_construction = 400
    m = 16
    parent_level = 2
    mapping_ef = 400
    k_children = 400
    n_probe list = 5 8 10 12
    k = 10

Output:
    CSV file (default: repair_5k_efc400_kch400_map400.csv) with rows per n_probe.

Usage (PowerShell):
    py -3 repair_comparison_benchmark.py --out repair_5k_efc400_kch400_map400.csv

"""
import argparse
import csv
import time
import os
import numpy as np
from hnsw_hybrid_evaluation import (
    HybridHNSWIndex,
    generate_synthetic_dataset,
    split_query_set_from_dataset,
    RecallEvaluator,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-size', type=int, default=5000)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--queries', type=int, default=500)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--m', type=int, default=16)
    p.add_argument('--efc', type=int, default=400, help='ef_construction')
    p.add_argument('--parent-level', type=int, default=2)
    p.add_argument('--mapping-ef', type=int, default=400)
    p.add_argument('--k-children', type=int, default=400)
    p.add_argument('--n-probe', type=int, nargs='+', default=[5,8,10,12])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='repair_5k_efc400_kch400_map400.csv')
    return p.parse_args()


def ensure_header(path, header):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(header)


def append_row(path, header, row):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([row.get(h, '') for h in header])


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print('=== Repair Comparison Benchmark ===')
    print(f"dataset={args.dataset_size} dim={args.dim} queries={args.queries} k={args.k}")
    print(f"efc={args.efc} m={args.m} parent_level={args.parent_level} mapping_ef={args.mapping_ef} k_children={args.k_children}")
    print(f"n_probe list={args.n_probe}")

    # Data
    t0 = time.time()
    full = generate_synthetic_dataset(args.dataset_size, args.dim)
    base_dataset, query_set = split_query_set_from_dataset(full, args.queries)
    print(f"Data ready in {time.time()-t0:.2f}s | base={len(base_dataset)} queries={len(query_set)}")

    # Build base index once
    build_start = time.time()
    base = HybridHNSWIndex(k_children=0, n_probe=0)
    base.build_base_index(base_dataset, m=args.m, ef_construction=args.efc)
    base.extract_parent_nodes(target_level=args.parent_level)
    print(f"Base index built in {time.time()-build_start:.2f}s | parents={len(base.parent_ids)}")

    evaluator = RecallEvaluator(base_dataset)

    header = [
        'dataset_size','dim','n_queries','k','m','ef_construction','k_children','n_probe',
        'mapping_ef','parent_level','recall@k','avg_query_time','base_build_time','parent_extraction_time',
        'mapping_build_time','avg_candidate_size','coverage_fraction','parent_count','covered_points','total_points',
        'overlap_unique_fraction','avg_assignment_count','mean_jaccard_overlap','median_jaccard_overlap',
        'multi_coverage_fraction','max_assignment_count','timestamp'
    ]
    ensure_header(args.out, header)

    for n_probe in args.n_probe:
        print(f"\n[Run] n_probe={n_probe} (repair enabled)")
        hybrid = HybridHNSWIndex(k_children=args.k_children, n_probe=n_probe, parent_child_method='approx')
        # Reuse base
        hybrid.base_index = base.base_index
        hybrid.dataset = base_dataset
        hybrid.parent_ids = base.parent_ids
        hybrid.parent_vectors = {pid: base_dataset[pid] for pid in base.parent_ids}
        if base.parent_ids:
            hybrid._parent_matrix = np.stack([hybrid.parent_vectors[pid] for pid in base.parent_ids], axis=0)
        hybrid.base_build_time = base.base_build_time
        hybrid.parent_extraction_time = base.parent_extraction_time

        # Build mapping WITH repair
        hybrid.build_parent_child_mapping(method='approx', ef=args.mapping_ef,
                                          diversify_max_assignments=None,
                                          repair_min_assignments=1)
        # Warmup
        first_qid, first_qvec = next(iter(query_set.items()))
        hybrid.search(first_qvec, k=args.k)
        # Evaluate
        results = evaluator.evaluate_recall(hybrid, query_set, k=args.k, progress_interval=max(50, len(query_set)//10))
        stats = hybrid.stats()
        row = {
            'dataset_size': args.dataset_size,
            'dim': args.dim,
            'n_queries': len(query_set),
            'k': args.k,
            'm': args.m,
            'ef_construction': args.efc,
            'k_children': args.k_children,
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
        print(f"  -> recall@{args.k}={row['recall@k']:.4f} cov={row['coverage_fraction']:.3f} cand={row['avg_candidate_size']:.1f}")

    print(f"\nCompleted. Results -> {args.out}")

if __name__ == '__main__':
    main()
