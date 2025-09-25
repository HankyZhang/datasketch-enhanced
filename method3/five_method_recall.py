"""Five-Method Recall Comparison

Compares recall/time of five ANN / clustering hybrid approaches on one dataset:
  1. HNSW baseline (single index search)
  2. Pure KMeans (probe n clusters then brute force inside)
  3. Hybrid HNSW (parents from level, children HNSW subsets)
  4. KMeans HNSW (single pivot two-stage)
  5. Multi-Pivot KMeans HNSW (extended two-stage)

Usage (example):
  py -3 method3/five_method_recall.py --dataset-size 5000 --query-size 100 --dimension 96 \
     --n-clusters 64 --k-children 400 --child-search-ef 400 --k 10 --n-probe 10 \
     --multi-num-pivots 3 --pivot-strategy line_perp_third --pivot-overquery-factor 1.2

Output: five_method_recall_report.json with per-method stats + summary.
"""

from __future__ import annotations
import os, sys, time, json, argparse
from typing import Dict, List, Any, Tuple
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hnsw.hnsw import HNSW
from hybrid_hnsw.hnsw_hybrid import HNSWHybrid
from method3.kmeans_hnsw import KMeansHNSW
from method3.kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot
from sklearn.cluster import MiniBatchKMeans


def make_data(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n, dim)).astype(np.float32)


def compute_ground_truth(dataset: np.ndarray, queries: np.ndarray, k: int) -> List[set]:
    gt: List[set] = []
    for q in queries:
        d = np.linalg.norm(dataset - q, axis=1)
        idx = np.argsort(d)[:k]
        gt.append(set(int(i) for i in idx))
    return gt


def eval_baseline(index: HNSW, queries: np.ndarray, gt: List[set], k: int, ef: int) -> Dict[str, Any]:
    times = []
    correct = 0
    for q, g in zip(queries, gt):
        t0 = time.time()
        res = index.query(q, k=k, ef=ef)
        times.append(time.time() - t0)
        found = {int(i) for i, _ in res}
        correct += len(found & g)
    total = len(queries) * k
    return {
        'phase': 'baseline_hnsw',
        'recall_at_k': correct / total if total else 0.0,
        'total_correct': correct,
        'total_expected': total,
        'avg_query_time_ms': float(np.mean(times) * 1000),
        'std_query_time_ms': float(np.std(times) * 1000),
        'ef': ef
    }


def build_pure_kmeans(dataset: np.ndarray, n_clusters: int, seed: int) -> Tuple[MiniBatchKMeans, Dict[int, List[int]]]:
    mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=min(1024, len(dataset)), n_init=3)
    mbk.fit(dataset)
    clusters: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
    for idx, c in enumerate(mbk.labels_):
        clusters[int(c)].append(int(idx))
    return mbk, clusters


def eval_pure_kmeans(model: MiniBatchKMeans, clusters: Dict[int, List[int]], dataset: np.ndarray,
                     queries: np.ndarray, gt: List[set], k: int, n_probe: int) -> Dict[str, Any]:
    centers = model.cluster_centers_
    n_clusters = centers.shape[0]
    n_probe_eff = min(n_probe, n_clusters)
    times = []
    correct = 0
    for q, g in zip(queries, gt):
        t0 = time.time()
        dC = np.linalg.norm(centers - q, axis=1)
        probe = np.argpartition(dC, n_probe_eff - 1)[:n_probe_eff]
        probe = probe[np.argsort(dC[probe])]
        cand_ids: List[int] = []
        for c in probe:
            cand_ids.extend(clusters.get(int(c), []))
        if cand_ids:
            cand_vecs = dataset[cand_ids]
            d = np.linalg.norm(cand_vecs - q, axis=1)
            order = np.argsort(d)[:k]
            found = {cand_ids[i] for i in order}
            correct += len(found & g)
        times.append(time.time() - t0)
    total = len(queries) * k
    return {
        'phase': 'clusters_only',
        'recall_at_k': correct / total if total else 0.0,
        'total_correct': correct,
        'total_expected': total,
        'avg_query_time_ms': float(np.mean(times) * 1000),
        'std_query_time_ms': float(np.std(times) * 1000),
        'n_probe': n_probe_eff,
        'n_clusters': n_clusters
    }


def eval_two_stage(system, queries: np.ndarray, gt: List[set], k: int, n_probe: int, phase: str) -> Dict[str, Any]:
    times = []
    correct = 0
    for q, g in zip(queries, gt):
        t0 = time.time()
        res = system.search(q, k=k, n_probe=n_probe)
        times.append(time.time() - t0)
        found = {int(i) for i, _ in res}
        correct += len(found & g)
    total = len(queries) * k
    out = {
        'phase': phase,
        'recall_at_k': correct / total if total else 0.0,
        'total_correct': correct,
        'total_expected': total,
        'avg_query_time_ms': float(np.mean(times) * 1000),
        'std_query_time_ms': float(np.std(times) * 1000),
        'n_probe': n_probe
    }
    if hasattr(system, 'get_stats'):
        out['system_stats'] = system.get_stats()
    return out


def main():
    ap = argparse.ArgumentParser(description='Five-method recall comparison (single run)')
    ap.add_argument('--dataset-size', type=int, default=10000)
    ap.add_argument('--query-size', type=int, default=100)
    ap.add_argument('--dimension', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--n-clusters', type=int, default=128)
    ap.add_argument('--k-children', type=int, default=600)
    ap.add_argument('--child-search-ef', type=int, default=600)
    ap.add_argument('--n-probe', type=int, default=10)
    ap.add_argument('--baseline-ef', type=int, default=400)
    ap.add_argument('--hybrid-parent-level', type=int, default=2)
    # Multi-pivot params
    ap.add_argument('--multi-num-pivots', type=int, default=3)
    ap.add_argument('--pivot-strategy', type=str, default='line_perp_third')
    ap.add_argument('--pivot-overquery-factor', type=float, default=1.2)
    # Disable flags
    ap.add_argument('--no-baseline', action='store_true')
    ap.add_argument('--no-pure-kmeans', action='store_true')
    ap.add_argument('--no-hybrid', action='store_true')
    ap.add_argument('--no-single', action='store_true')
    ap.add_argument('--no-multi', action='store_true')
    ap.add_argument('--out', type=str, default='five_method_recall_report.json')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"📦 Generating dataset size={args.dataset_size} dim={args.dimension} queries={args.query_size}")
    dataset = make_data(args.dataset_size, args.dimension, args.seed)
    q_idx = rng.choice(args.dataset_size, size=min(args.query_size, args.dataset_size), replace=False)
    queries = dataset[q_idx]

    print('🔍 Computing ground truth...')
    t0 = time.time()
    gt = compute_ground_truth(dataset, queries, args.k)
    gt_time = time.time() - t0
    print(f'  GT done in {gt_time:.2f}s')

    print('🏗️ Building base HNSW...')
    dist = lambda a, b: float(np.linalg.norm(a - b))
    base_index = HNSW(distance_func=dist, m=16, ef_construction=200)
    for i, v in enumerate(dataset):
        base_index.insert(i, v)
        if (i + 1) % max(1000, args.dataset_size // 10) == 0:
            print(f'  Inserted {i+1}/{args.dataset_size}')

    results: Dict[str, Any] = {
        'params': vars(args),
        'ground_truth_time_s': gt_time,
        'methods': {}
    }

    # 1. Baseline HNSW
    if not args.no_baseline:
        print('\n=== [1] HNSW Baseline ===')
        base_eval = eval_baseline(base_index, queries, gt, args.k, ef=args.baseline_ef)
        results['methods']['hnsw'] = base_eval

    # KMeans clustering (build once; reused)
    model = None
    clusters = None
    if not args.no_pure_kmeans or not args.no_single or not args.no_multi:
        print('\n🏗️ Clustering (MiniBatchKMeans)...')
        t_cluster = time.time()
        model, clusters = build_pure_kmeans(dataset, args.n_clusters, args.seed)
        cluster_time = time.time() - t_cluster
    else:
        cluster_time = 0.0

    # 2. Pure KMeans
    if not args.no_pure_kmeans:
        print('\n=== [2] Pure KMeans ===')
        pk_eval = eval_pure_kmeans(model, clusters, dataset, queries, gt, args.k, args.n_probe)
        pk_eval['clustering_time_s'] = cluster_time
        results['methods']['pure_kmeans'] = pk_eval

    # 3. Hybrid HNSW
    if not args.no_hybrid:
        print('\n=== [3] Hybrid HNSW ===')
        t_h = time.time()
        hybrid = HNSWHybrid(
            base_index=base_index,
            parent_level=args.hybrid_parent_level,
            k_children=args.k_children,
            adaptive_k_children=False
        )
        build_h = time.time() - t_h
        h_eval = eval_two_stage(hybrid, queries, gt, args.k, args.n_probe, 'hybrid_hnsw_level')
        h_eval['build_time_s'] = build_h
        results['methods']['hybrid_hnsw'] = h_eval

    # 4. Single-pivot KMeans HNSW
    if not args.no_single:
        print('\n=== [4] KMeans HNSW (single pivot) ===')
        t_s = time.time()
        single = KMeansHNSW(
            base_index=base_index,
            n_clusters=args.n_clusters,
            k_children=args.k_children,
            child_search_ef=args.child_search_ef,
            kmeans_params={'n_init': 3, 'random_state': args.seed}
        )
        build_s = time.time() - t_s
        s_eval = eval_two_stage(single, queries, gt, args.k, args.n_probe, 'kmeans_hnsw_single')
        s_eval['build_time_s'] = build_s
        results['methods']['kmeans_hnsw'] = s_eval

    # 5. Multi-pivot
    if not args.no_multi:
        print('\n=== [5] Multi-Pivot KMeans HNSW ===')
        t_m = time.time()
        multi = KMeansHNSWMultiPivot(
            base_index=base_index,
            n_clusters=args.n_clusters,
            k_children=args.k_children,
            child_search_ef=args.child_search_ef,
            multi_pivot_enabled=True,
            num_pivots=args.multi_num_pivots,
            pivot_selection_strategy=args.pivot_strategy,
            pivot_overquery_factor=args.pivot_overquery_factor,
            kmeans_params={'n_init': 3, 'random_state': args.seed}
        )
        build_m = time.time() - t_m
        m_eval = eval_two_stage(multi, queries, gt, args.k, args.n_probe, 'kmeans_hnsw_multi_pivot')
        m_eval['build_time_s'] = build_m
        results['methods']['multi_pivot_kmeans_hnsw'] = m_eval

    # Summary
    summary = {}
    for name, info in results['methods'].items():
        if 'recall_at_k' in info:
            summary[name] = info['recall_at_k']
    results['recall_summary'] = summary

    out_file = args.out
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('\n=== Done ===')
    print(json.dumps({'recall_summary': summary}, indent=2, ensure_ascii=False))
    print(f'Results saved -> {out_file}')


if __name__ == '__main__':
    main()
