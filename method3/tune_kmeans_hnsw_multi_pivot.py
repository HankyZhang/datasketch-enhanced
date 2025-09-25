"""扩展: 增加多枢纽KMeans-HNSW评估 (A variant enabling multi-pivot KMeans-HNSW)

基于 `tune_kmeans_hnsw.py` 复制, 增加参数:
  --multi-pivot            启用多枢纽版本
  --num-pivots N           使用的枢纽数量 (>=1)
  --pivot-strategy STR     枢纽选择策略: line_perp_third | max_min_distance
  --pivot-overquery-factor F 每个枢纽查询的k放大系数

比较: 单枢纽 vs 多枢纽 (相同其它参数) 的召回与时间。
"""
import os, sys, time, argparse, json, numpy as np
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hnsw.hnsw import HNSW
from method3.kmeans_hnsw import KMeansHNSW
from method3.kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot


def create_dataset(n: int, dim: int, seed: int = 42):
    np.random.seed(seed)
    return np.random.randn(n, dim).astype(np.float32)


def brute_force_ground_truth(dataset: np.ndarray, queries: np.ndarray, k: int) -> Dict[int, List[int]]:
    gt = {}
    for qi, qv in enumerate(queries):
        dists = np.linalg.norm(dataset - qv, axis=1)
        idx = np.argsort(dists)[:k]
        gt[qi] = idx.tolist()
    return gt


def evaluate_system(system, queries: np.ndarray, gt: Dict[int, List[int]], k: int, n_probe: int) -> Dict[str, Any]:
    total_correct = 0
    times = []
    for qi, qv in enumerate(queries):
        t0 = time.time()
        res = system.search(qv, k=k, n_probe=n_probe)
        times.append(time.time() - t0)
        found = {nid for nid, _ in res}
        total_correct += len(found & set(gt[qi]))
    recall = total_correct / (len(queries) * k)
    return {
        'recall_at_k': recall,
        'avg_query_time_ms': float(np.mean(times) * 1000),
        'std_query_time_ms': float(np.std(times) * 1000),
        'system_stats': system.get_stats()
    }


def main():
    ap = argparse.ArgumentParser(description='Multi-pivot KMeans-HNSW tuner')
    ap.add_argument('--dataset-size', type=int, default=5000)
    ap.add_argument('--query-size', type=int, default=50)
    ap.add_argument('--dimension', type=int, default=128)
    ap.add_argument('--n-clusters', type=int, default=64)
    ap.add_argument('--k-children', type=int, default=400)
    ap.add_argument('--child-search-ef', type=int, default=400)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--n-probe', type=int, default=10)
    # multi-pivot controls
    ap.add_argument('--multi-pivot', action='store_true')
    ap.add_argument('--num-pivots', type=int, default=3)
    ap.add_argument('--pivot-strategy', type=str, default='line_perp_third', choices=['line_perp_third','max_min_distance'])
    ap.add_argument('--pivot-overquery-factor', type=float, default=1.1)
    args = ap.parse_args()

    print('Building synthetic dataset ...')
    base = create_dataset(args.dataset_size, args.dimension)
    queries = create_dataset(args.query_size, args.dimension, seed=1234)

    distance = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance, m=16, ef_construction=200)
    for i, v in enumerate(base):
        base_index.insert(i, v)
        if (i+1) % 1000 == 0:
            print(f'  Inserted {i+1}/{len(base)} vectors')

    print('Computing ground truth ...')
    gt = brute_force_ground_truth(base, queries, args.k)

    print('\n=== Single-pivot system build ===')
    sp_start = time.time()
    single_system = KMeansHNSW(
        base_index=base_index,
        n_clusters=args.n_clusters,
        k_children=args.k_children,
        child_search_ef=args.child_search_ef
    )
    sp_build = time.time() - sp_start

    print('\n=== Multi-pivot system build ===')
    mp_start = time.time()
    multi_system = KMeansHNSWMultiPivot(
        base_index=base_index,
        n_clusters=args.n_clusters,
        k_children=args.k_children,
        child_search_ef=args.child_search_ef,
        multi_pivot_enabled=args.multi_pivot,
        num_pivots=args.num_pivots,
        pivot_selection_strategy=args.pivot_strategy,
        pivot_overquery_factor=args.pivot_overquery_factor
    )
    mp_build = time.time() - mp_start

    print('\nEvaluating ...')
    single_eval = evaluate_system(single_system, queries, gt, args.k, args.n_probe)
    multi_eval = evaluate_system(multi_system, queries, gt, args.k, args.n_probe)

    report = {
        'params': vars(args),
        'single_build_time_s': sp_build,
        'multi_build_time_s': mp_build,
        'single_eval': single_eval,
        'multi_eval': multi_eval
    }
    print('\n=== Comparison ===')
    print(json.dumps(report, indent=2))
    with open('multi_pivot_tuning_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    print('Saved to multi_pivot_tuning_results.json')

if __name__ == '__main__':
    main()
