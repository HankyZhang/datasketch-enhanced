"""五方案召回评估脚本 (Five-Method Recall Tuner)

对比方案 (Methods Compared):
  1. 原始 HNSW 基线 (baseline HNSW)
  2. 纯 KMeans (仅聚类 + 聚类成员二阶段检索, 无 HNSW 子集合重建)
  3. Hybrid HNSW (level-based parents)
  4. KMeans HNSW (单枢纽)
  5. Multi-Pivot KMeans HNSW (多枢纽)

CLI 重点参数:
  --no-* 可关闭某个方案; 默认全部开启
  --multi-pivot / --num-pivots / --pivot-strategy / --pivot-overquery-factor 控制多枢纽
  --baseline-ef 控制 HNSW 基线 ef
  --hybrid-parent-level / --hybrid-k-children 控制 Hybrid HNSW
  --pure-n-probe 控制纯 KMeans 检索使用多少聚类 (默认沿用 --n-probe)

输出: JSON (comparison_report.json) 包含每种方法: 构建时间, recall@k, 查询时间统计, 关键内部统计.
"""

from __future__ import annotations
import os, sys, time, argparse, json
from typing import Dict, List, Any, Tuple
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hnsw.hnsw import HNSW
from hybrid_hnsw.hnsw_hybrid import HNSWHybrid
from method3.kmeans_hnsw import KMeansHNSW
from method3.kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot
from sklearn.cluster import MiniBatchKMeans


# ---------------------- Data / Ground Truth ----------------------
def create_dataset(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def brute_force_gt(dataset: np.ndarray, queries: np.ndarray, k: int) -> List[set]:
    gt: List[set] = []
    for q in queries:
        d = np.linalg.norm(dataset - q, axis=1)
        idx = np.argsort(d)[:k]
        gt.append(set(int(i) for i in idx))
    return gt


# ---------------------- Evaluation Helpers ----------------------
def eval_hnsw_baseline(index: HNSW, queries: np.ndarray, gt: List[set], k: int, ef: int) -> Dict[str, Any]:
    times = []
    correct = 0
    for q, g in zip(queries, gt):
        t0 = time.time()
        res = index.query(q, k=k, ef=ef)
        times.append(time.time() - t0)
        ids = {int(i) for i, _ in res}
        correct += len(ids & g)
    recall = correct / (len(queries) * k)
    return {
        'recall_at_k': recall,
        'avg_query_time_ms': float(np.mean(times) * 1000),
        'std_query_time_ms': float(np.std(times) * 1000),
        'total_correct': correct
    }


def eval_two_stage(system, queries: np.ndarray, gt: List[set], k: int, n_probe: int) -> Dict[str, Any]:
    times = []
    correct = 0
    for q, g in zip(queries, gt):
        t0 = time.time()
        res = system.search(q, k=k, n_probe=n_probe)
        times.append(time.time() - t0)
        ids = {int(i) for i, _ in res}
        correct += len(ids & g)
    recall = correct / (len(queries) * k)
    return {
        'recall_at_k': recall,
        'avg_query_time_ms': float(np.mean(times) * 1000),
        'std_query_time_ms': float(np.std(times) * 1000),
        'total_correct': correct,
        'system_stats': system.get_stats() if hasattr(system, 'get_stats') else {}
    }


def build_pure_kmeans(dataset: np.ndarray, n_clusters: int, kmeans_params: Dict[str, Any]) -> Tuple[MiniBatchKMeans, Dict[int, List[int]]]:
    params = kmeans_params.copy()
    if 'max_iters' in params and 'max_iter' not in params:
        params['max_iter'] = params.pop('max_iters')
    if params.get('batch_size') in (None, 0):
        params['batch_size'] = min(1024, len(dataset))
    allowed = {
        'n_clusters','init','max_iter','batch_size','verbose','random_state','tol',
        'max_no_improvement','init_size','n_init','reassignment_ratio'
    }
    mbk_params = {k: v for k, v in params.items() if k in allowed}
    mbk_params['n_clusters'] = n_clusters
    model = MiniBatchKMeans(**mbk_params)
    model.fit(dataset)
    labels = model.labels_
    clusters: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
    for idx, c in enumerate(labels):
        clusters[int(c)].append(int(idx))
    return model, clusters


def eval_pure_kmeans(model: MiniBatchKMeans, clusters: Dict[int, List[int]], dataset: np.ndarray,
                     queries: np.ndarray, gt: List[set], k: int, n_probe: int) -> Dict[str, Any]:
    centers = model.cluster_centers_
    times = []
    correct = 0
    for q, g in zip(queries, gt):
        t0 = time.time()
        d_cent = np.linalg.norm(centers - q, axis=1)
        probe_idx = np.argsort(d_cent)[:n_probe]
        candidate_ids: List[int] = []
        for ci in probe_idx:
            candidate_ids.extend(clusters.get(int(ci), []))
        if not candidate_ids:
            times.append(time.time() - t0)
            continue
        cand_vecs = dataset[candidate_ids]
        d = np.linalg.norm(cand_vecs - q, axis=1)
        order = np.argsort(d)[:k]
        ids = {candidate_ids[i] for i in order}
        correct += len(ids & g)
        times.append(time.time() - t0)
    recall = correct / (len(queries) * k)
    return {
        'recall_at_k': recall,
        'avg_query_time_ms': float(np.mean(times) * 1000),
        'std_query_time_ms': float(np.std(times) * 1000),
        'total_correct': correct,
        'avg_cluster_size': float(len(dataset) / len(clusters)) if clusters else 0.0
    }


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(description='Five-method KMeans/HNSW Multi-Pivot Recall Tuner')
    ap.add_argument('--dataset-size', type=int, default=10000)
    ap.add_argument('--query-size', type=int, default=100)
    ap.add_argument('--dimension', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--n-probe', type=int, default=10)
    ap.add_argument('--pure-n-probe', type=int, default=None, help='Overrides n-probe for pure KMeans (optional)')
    ap.add_argument('--n-clusters', type=int, default=128)
    ap.add_argument('--k-children', type=int, default=600)
    ap.add_argument('--child-search-ef', type=int, default=600)
    ap.add_argument('--baseline-ef', type=int, default=400)
    # Hybrid
    ap.add_argument('--hybrid-parent-level', type=int, default=2)
    ap.add_argument('--hybrid-k-children', type=int, default=600)
    # Multi-pivot controls
    ap.add_argument('--multi-pivot', action='store_true')
    ap.add_argument('--num-pivots', type=int, default=3)
    ap.add_argument('--pivot-strategy', type=str, default='line_perp_third', choices=['line_perp_third','max_min_distance'])
    ap.add_argument('--pivot-overquery-factor', type=float, default=1.2)
    # Enable / disable blocks
    ap.add_argument('--no-baseline', action='store_true')
    ap.add_argument('--no-pure-kmeans', action='store_true')
    ap.add_argument('--no-hybrid', action='store_true')
    ap.add_argument('--no-single-kmeans-hnsw', action='store_true')
    ap.add_argument('--no-multi-pivot', action='store_true')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"📦 生成数据集 dataset-size={args.dataset_size} dim={args.dimension} | queries={args.query_size}")
    dataset = create_dataset(args.dataset_size, args.dimension, seed=args.seed)
    query_indices = rng.choice(args.dataset_size, size=min(args.query_size, args.dataset_size), replace=False)
    queries = dataset[query_indices]

    print('🔍 计算真实值 (ground truth) ...')
    t_gt = time.time()
    gt = brute_force_gt(dataset, queries, args.k)
    gt_time = time.time() - t_gt
    print(f'  Ground truth computed in {gt_time:.2f}s')

    # Build base HNSW index
    print('🏗️ 构建 HNSW 基线索引 (base HNSW index) ...')
    distance = lambda a, b: float(np.linalg.norm(a - b))
    base_index = HNSW(distance_func=distance, m=16, ef_construction=200)
    for i, v in enumerate(dataset):
        base_index.insert(i, v)
        if (i + 1) % max(1000, args.dataset_size // 10) == 0:
            print(f'  Inserted {i + 1}/{args.dataset_size}')

    results: Dict[str, Any] = {
        'params': vars(args),
        'ground_truth_time_s': gt_time,
        'methods': {}
    }

    # 1. Baseline HNSW
    if not args.no_baseline:
        print('\n=== [1] HNSW Baseline ===')
        t0 = time.time()
        base_eval = eval_hnsw_baseline(base_index, queries, gt, args.k, ef=args.baseline_ef)
        base_eval['eval_time_s'] = time.time() - t0
        results['methods']['hnsw'] = base_eval

    # Prepare common KMeans params
    kmeans_params = {'max_iters': 80, 'n_init': 3, 'verbose': 0, 'random_state': args.seed}

    # 2. Pure KMeans
    if not args.no_pure_kmeans:
        print('\n=== [2] Pure KMeans ===')
        t0 = time.time()
        model, cluster_map = build_pure_kmeans(dataset, args.n_clusters, kmeans_params)
        build_time = time.time() - t0
        pure_eval = eval_pure_kmeans(model, cluster_map, dataset, queries, gt, args.k,
                                     n_probe=args.pure_n_probe or args.n_probe)
        pure_eval['build_time_s'] = build_time
        pure_eval['kmeans_inertia'] = float(model.inertia_)
        results['methods']['pure_kmeans'] = pure_eval
    else:
        model = None  # keep name for later reuse detection

    # 3. Hybrid HNSW
    if not args.no_hybrid:
        print('\n=== [3] Hybrid HNSW ===')
        t0 = time.time()
        hybrid = HNSWHybrid(
            base_index=base_index,
            parent_level=args.hybrid_parent_level,
            k_children=args.hybrid_k_children,
            adaptive_k_children=False
        )
        build_time = time.time() - t0
        hybrid_eval = eval_two_stage(hybrid, queries, gt, args.k, args.n_probe)
        hybrid_eval['build_time_s'] = build_time
        results['methods']['hybrid_hnsw'] = hybrid_eval

    # 4. KMeans HNSW (single pivot)
    if not args.no_single_kmeans_hnsw:
        print('\n=== [4] KMeans HNSW (single pivot) ===')
        t0 = time.time()
        kmeans_hnsw = KMeansHNSW(
            base_index=base_index,
            n_clusters=args.n_clusters,
            k_children=args.k_children,
            child_search_ef=args.child_search_ef,
            kmeans_params=kmeans_params
        )
        build_time = time.time() - t0
        single_eval = eval_two_stage(kmeans_hnsw, queries, gt, args.k, args.n_probe)
        single_eval['build_time_s'] = build_time
        results['methods']['kmeans_hnsw'] = single_eval
    else:
        kmeans_hnsw = None

    # 5. Multi-Pivot KMeans HNSW
    if not args.no_multi_pivot and args.multi_pivot:
        print('\n=== [5] Multi-Pivot KMeans HNSW ===')
        t0 = time.time()
        multi = KMeansHNSWMultiPivot(
            base_index=base_index,
            n_clusters=args.n_clusters,
            k_children=args.k_children,
            child_search_ef=args.child_search_ef,
            multi_pivot_enabled=True,
            num_pivots=args.num_pivots,
            pivot_selection_strategy=args.pivot_strategy,
            pivot_overquery_factor=args.pivot_overquery_factor,
            kmeans_params=kmeans_params
        )
        build_time = time.time() - t0
        multi_eval = eval_two_stage(multi, queries, gt, args.k, args.n_probe)
        multi_eval['build_time_s'] = build_time
        results['methods']['multi_pivot_kmeans_hnsw'] = multi_eval
    elif not args.no_multi_pivot:
        results['methods']['multi_pivot_kmeans_hnsw'] = {'skipped': 'enable --multi-pivot to build'}

    # Aggregate comparison metrics (recall summary)
    summary = {}
    for name, info in results['methods'].items():
        if 'recall_at_k' in info:
            summary[name] = info['recall_at_k']
    results['recall_summary'] = summary

    out_file = 'comparison_report.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('\n=== 完成 / Completed ===')
    print(json.dumps({'recall_summary': summary}, indent=2, ensure_ascii=False))
    print(f'保存结果 -> {out_file}')


if __name__ == '__main__':
    main()
