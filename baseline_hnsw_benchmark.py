"""Baseline HNSW Recall Benchmark (Matches Hybrid Benchmark Configuration)

Purpose:
    Compute baseline HNSW recall@k on the SAME synthetic dataset configuration
    used in the unoptimized hybrid benchmark (dataset split: hold out queries),
    across grids of ef_construction (build) and ef_search (query) parameters.

Output:
    CSV rows with recall@k, avg query latency, build time.

Usage (PowerShell):
    py baseline_hnsw_benchmark.py --dataset-size 5000 --queries 500 \
        --efc 200 300 400 --ef-search 50 100 150 200 300 400 --k 10 --out baseline_5k.csv

Notes:
    - Uses the same generate_synthetic_dataset + split_query_set_from_dataset functions.
    - generate_synthetic_dataset internally seeds RNG for reproducibility.
    - Brute-force ground truth over base dataset (size ~ dataset_size - queries).
"""

from __future__ import annotations
import argparse
import csv
import os
import time
from typing import Dict, List
import numpy as np

from datasketch.hnsw import HNSW
from hnsw_hybrid_evaluation import (
    generate_synthetic_dataset,
    split_query_set_from_dataset,
)


def l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y))


def compute_ground_truth(base_dataset: Dict[int, np.ndarray], queries: Dict[int, np.ndarray], k: int) -> Dict[int, List[int]]:
    """Exact brute force top-k over base dataset for each query (exclude query ids)."""
    ground = {}
    base_items = list(base_dataset.items())
    base_ids = [bid for bid, _ in base_items]
    base_mat = np.stack([vec for _, vec in base_items], axis=0)
    for i, (qid, qv) in enumerate(queries.items()):
        if (i + 1) % 100 == 0 or (i + 1) == len(queries):
            print(f"  GT {i+1}/{len(queries)}")
        dists = np.linalg.norm(base_mat - qv, axis=1)
        top_idx = np.argpartition(dists, k)[:k]
        order = top_idx[np.argsort(dists[top_idx])]
        ground[qid] = [base_ids[j] for j in order]
    return ground


def ensure_header(path: str, header: List[str]):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(header)


def append_row(path: str, header: List[str], row: Dict[str, object]):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([row.get(h, '') for h in header])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-size', type=int, required=True)
    ap.add_argument('--queries', type=int, default=500)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--efc', type=int, nargs='+', default=[200, 300, 400], help='ef_construction values')
    ap.add_argument('--ef-search', type=int, nargs='+', default=[50, 100, 150, 200, 300, 400], help='ef_search values')
    ap.add_argument('--m', type=int, default=16)
    ap.add_argument('--out', default='baseline_results.csv')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=== Baseline HNSW Benchmark ===")
    print(f"Dataset={args.dataset_size} dim={args.dim} queries={args.queries} k={args.k}")
    print(f"ef_construction grid: {args.efc}")
    print(f"ef_search grid: {args.ef_search}")

    # Dataset + split (same as hybrid benchmark fairness)
    full_dataset = generate_synthetic_dataset(args.dataset_size, args.dim)
    base_dataset, query_set = split_query_set_from_dataset(full_dataset, args.queries)
    print(f"Base size={len(base_dataset)} Query size={len(query_set)}")

    # Ground truth (once) over base dataset for all queries
    print("Computing ground truth...")
    gt = compute_ground_truth(base_dataset, query_set, args.k)
    print("Ground truth done.")

    header = [
        'dataset_size','dim','n_queries','k','m','ef_construction','ef_search',
        'recall@k','avg_query_time','build_time','base_size','timestamp'
    ]
    ensure_header(args.out, header)

    for efc in args.efc:
        print(f"\n[Build] ef_construction={efc}")
        build_start = time.time()
        index = HNSW(distance_func=l2_distance, m=args.m, ef_construction=efc)
        # bulk insert
        index.update(base_dataset)
        build_time = time.time() - build_start
        print(f"Built in {build_time:.2f}s")

        for ef_search in args.ef_search:
            print(f"  [Search] ef_search={ef_search}")
            t0 = time.time()
            recalls = []
            q_times = []
            for i, (qid, qv) in enumerate(query_set.items()):
                if (i + 1) % max(50, len(query_set)//10) == 0 or (i + 1) == len(query_set):
                    elapsed = time.time() - t0
                    print(f"    {i+1}/{len(query_set)} elapsed={elapsed:.1f}s")
                res = index.query(qv, k=args.k, ef=ef_search)
                pred_ids = [nid for nid, _ in res]
                gt_ids = gt[qid]
                # recall
                r = len(set(pred_ids) & set(gt_ids)) / args.k
                recalls.append(r)
            avg_query_time = (time.time() - t0) / len(query_set)
            recall_at_k = float(np.mean(recalls))
            print(f"    -> recall@{args.k}={recall_at_k:.4f} avg_q_time={avg_query_time*1000:.3f}ms")
            row = {
                'dataset_size': args.dataset_size,
                'dim': args.dim,
                'n_queries': len(query_set),
                'k': args.k,
                'm': args.m,
                'ef_construction': efc,
                'ef_search': ef_search,
                'recall@k': recall_at_k,
                'avg_query_time': avg_query_time,
                'build_time': build_time,
                'base_size': len(base_dataset),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            append_row(args.out, header, row)

    print(f"\nBenchmark complete -> {args.out}")


if __name__ == '__main__':
    main()
