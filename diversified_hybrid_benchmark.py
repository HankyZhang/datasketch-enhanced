"""Diversified / Repaired Hybrid HNSW Benchmark (5k scale)

Runs the HybridHNSWIndex with diversification (cap per-point assignment) and
repair (ensure minimum assignments) to observe recall improvements vs unoptimized
baseline. Builds base index once per ef_construction, builds mapping once per
(k_children, ef_construction) then sweeps n_probe (optionally includes full
parent count) to avoid redundant mapping builds.

Example:
    py diversified_hybrid_benchmark.py --dataset-size 5000 --queries 500 \
        --efc 400 500 --k-children 800 --n-probe 5 8 10 12 \
        --diversify-max 3 --repair-min 1 --full-parent-probe \
        --out hybrid_optimized_5k.csv
"""
from __future__ import annotations
import argparse, csv, os, time
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
    p.add_argument('--dataset-size', type=int, required=True)
    p.add_argument('--queries', type=int, default=500)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--k-children', type=int, nargs='+', default=[800])
    p.add_argument('--n-probe', type=int, nargs='+', default=[5,8,10,12])
    p.add_argument('--efc', type=int, nargs='+', default=[400])
    p.add_argument('--m', type=int, default=16)
    p.add_argument('--parent-level', type=int, default=2)
    p.add_argument('--mapping-ef', type=int, default=80)
    p.add_argument('--diversify-max', type=int, default=3)
    p.add_argument('--repair-min', type=int, default=1)
    p.add_argument('--full-parent-probe', action='store_true', help='Add n_probe=parent_count evaluation')
    p.add_argument('--out', default='hybrid_optimized_results.csv')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def ensure_header(path: str, header: List[str]):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path,'w',newline='') as f:
            csv.writer(f).writerow(header)

def append_row(path: str, header: List[str], row: Dict[str, object]):
    with open(path,'a',newline='') as f:
        w = csv.writer(f)
        w.writerow([row.get(h,'') for h in header])

def main():
    args = parse_args()
    np.random.seed(args.seed)
    print("=== Diversified / Repaired Hybrid Benchmark ===")
    print(f"Dataset={args.dataset_size} queries={args.queries} k={args.k} dim={args.dim}")
    print(f"efc={args.efc} k_children={args.k_children} n_probe_base={args.n_probe} diversify_max={args.diversify_max} repair_min={args.repair_min}")

    # Data split
    full_dataset = generate_synthetic_dataset(args.dataset_size, args.dim)
    base_dataset, query_set = split_query_set_from_dataset(full_dataset, args.queries)
    evaluator = RecallEvaluator(base_dataset)
    # Pre-compute ground truth ONCE to avoid redundant O(N*Q) work per n_probe iteration
    # This was previously recomputed for every (k_children, n_probe) causing large slowdowns.
    print("Precomputing ground truth once for all n_probe evaluations...")
    evaluator.ground_truth_cache = evaluator.compute_ground_truth(query_set, k=args.k)
    print("Ground truth cached.")

    header = [
        'dataset_size','dim','n_queries','k','m','ef_construction','k_children','n_probe',
        'full_parent_probe','diversify_max','repair_min','mapping_ef','parent_level',
        'recall@k','avg_query_time','base_build_time','parent_extraction_time',
        'mapping_build_time','avg_candidate_size','coverage_fraction','parent_count',
        'covered_points','total_points','overlap_unique_fraction','avg_assignment_count',
        'mean_jaccard_overlap','median_jaccard_overlap','multi_coverage_fraction',
        'max_assignment_count','timestamp'
    ]
    ensure_header(args.out, header)

    for efc in args.efc:
        print(f"\n[Build] ef_construction={efc}")
        base = HybridHNSWIndex(k_children=0, n_probe=0)
        base.build_base_index(base_dataset, m=args.m, ef_construction=efc)
        base.extract_parent_nodes(target_level=args.parent_level)
        parent_ids = base.parent_ids
        parent_count = len(parent_ids)
        print(f"Parents: {parent_count}")

        # Prepare n_probe list (include full parent probe if requested)
        n_probe_values = list(args.n_probe)
        if args.full_parent_probe and parent_count not in n_probe_values:
            n_probe_values.append(parent_count)
            n_probe_values.sort()

        for k_children in args.k_children:
            # Build mapping once per k_children with diversification/repair
            print(f"  [Mapping] k_children={k_children}")
            hybrid = HybridHNSWIndex(k_children=k_children, n_probe=max(n_probe_values), parent_child_method='approx')
            hybrid.base_index = base.base_index
            hybrid.dataset = base_dataset
            hybrid.parent_ids = parent_ids
            hybrid.parent_vectors = {pid: base_dataset[pid] for pid in parent_ids}
            if parent_ids:
                hybrid._parent_matrix = np.stack([hybrid.parent_vectors[pid] for pid in parent_ids], axis=0)
            hybrid.parent_extraction_time = base.parent_extraction_time
            hybrid.base_build_time = base.base_build_time
            hybrid.build_parent_child_mapping(method='approx', ef=args.mapping_ef,
                                              diversify_max_assignments=args.diversify_max,
                                              repair_min_assignments=args.repair_min)
            stats_base = hybrid.stats()  # after mapping

            for n_probe in n_probe_values:
                hybrid.n_probe = n_probe
                # warmup
                first_qid, first_qvec = next(iter(query_set.items()))
                hybrid.search(first_qvec, k=args.k)
                results = evaluator.evaluate_recall(
                    hybrid,
                    query_set,
                    k=args.k,
                    progress_interval=max(100, len(query_set)//5)  # slightly less chatty
                )
                stats = hybrid.stats()
                from datetime import datetime
                row = {
                    'dataset_size': args.dataset_size,
                    'dim': args.dim,
                    'n_queries': len(query_set),
                    'k': args.k,
                    'm': args.m,
                    'ef_construction': efc,
                    'k_children': k_children,
                    'n_probe': n_probe,
                    'full_parent_probe': int(n_probe == parent_count),
                    'diversify_max': args.diversify_max,
                    'repair_min': args.repair_min,
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
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                append_row(args.out, header, row)
                print(f"    -> n_probe={n_probe} recall@{args.k}={row['recall@k']:.4f} cand={row['avg_candidate_size']:.1f} cov={row['coverage_fraction']:.3f}")

    print(f"\nBenchmark complete -> {args.out}")

if __name__ == '__main__':
    main()
