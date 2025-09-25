"""多枢纽扩展调优脚本 (Copy of tune_kmeans_hnsw + multi-pivot evaluation)

在原始参数扫描/评估框架基础上，额外加入第5种方案：Multi-Pivot KMeans HNSW。
对比 5 种方法：
  - HNSW 基线
  - 纯 K-Means (复用已有聚类)
  - Hybrid HNSW (基于层级父节点)
  - KMeans HNSW (单枢纽)
  - Multi-Pivot KMeans HNSW (多枢纽)
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from method3.kmeans_hnsw import KMeansHNSW
from method3.kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot
from hnsw.hnsw import HNSW
from hybrid_hnsw.hnsw_hybrid import HNSWHybrid
from sklearn.cluster import MiniBatchKMeans


class KMeansHNSWMultiPivotEvaluator:
    """复制自原 KMeansHNSWEvaluator，增加 multi-pivot 评估阶段。"""
    def __init__(self, dataset: np.ndarray, query_set: np.ndarray, query_ids: List[int], distance_func: callable):
        self.dataset = dataset
        self.query_set = query_set
        self.query_ids = query_ids
        self.distance_func = distance_func
        self._ground_truth_cache: Dict = {}

    # ---------- Ground Truth ----------
    def compute_ground_truth(self, k: int, exclude_query_ids: bool = True) -> Dict[int, List[Tuple[int, float]]]:
        key = (k, exclude_query_ids)
        if key in self._ground_truth_cache:
            return self._ground_truth_cache[key]
        print(f"计算真实值 queries={len(self.query_set)} k={k} ...")
        gt = {}
        for (qv, qid) in zip(self.query_set, self.query_ids):
            dists = []
            for j, dv in enumerate(self.dataset):
                if exclude_query_ids and j == qid:
                    continue
                d = self.distance_func(qv, dv)
                dists.append((d, j))
            dists.sort()
            gt[qid] = dists[:k]
        self._ground_truth_cache[key] = gt
        return gt

    # ---------- Common Recall Eval (works for single & multi-pivot) ----------
    def evaluate_recall_generic(self, system, k: int, n_probe: int, ground_truth: Dict) -> Dict[str, Any]:
        total_correct = 0
        total_expected = len(self.query_set) * k
        q_times = []
        indiv = []
        for qv, qid in zip(self.query_set, self.query_ids):
            true_ids = {nid for _, nid in ground_truth[qid]}
            t0 = time.time()
            res = system.search(qv, k=k, n_probe=n_probe)
            dt = time.time() - t0
            q_times.append(dt)
            found = {nid for nid, _ in res}
            correct = len(found & true_ids)
            total_correct += correct
            indiv.append(correct / k if k else 0.0)
        return {
            'recall_at_k': total_correct / total_expected if total_expected else 0.0,
            'total_correct': total_correct,
            'total_expected': total_expected,
            'avg_query_time_ms': float(np.mean(q_times) * 1000),
            'std_query_time_ms': float(np.std(q_times) * 1000),
            'avg_individual_recall': float(np.mean(indiv)),
            'std_individual_recall': float(np.std(indiv)),
            'n_probe': n_probe,
            'k': k
        }

    def evaluate_hnsw_baseline(self, base_index: HNSW, k: int, ef: int, ground_truth: Dict) -> Dict[str, Any]:
        total_correct = 0
        total_expected = len(self.query_set) * k
        q_times = []
        indiv = []
        for qv, qid in zip(self.query_set, self.query_ids):
            true_ids = {nid for _, nid in ground_truth[qid]}
            t0 = time.time()
            res = base_index.query(qv, k=k, ef=ef)
            dt = time.time() - t0
            q_times.append(dt)
            found = {nid for nid, _ in res}
            correct = len(found & true_ids)
            total_correct += correct
            indiv.append(correct / k if k else 0.0)
        return {
            'phase': 'baseline_hnsw',
            'ef': ef,
            'recall_at_k': total_correct / total_expected if total_expected else 0.0,
            'avg_query_time_ms': float(np.mean(q_times) * 1000),
            'std_query_time_ms': float(np.std(q_times) * 1000)
        }

    def evaluate_hybrid(self, hybrid: 'HNSWHybrid', k: int, n_probe: int, ground_truth: Dict) -> Dict[str, Any]:
        r = self.evaluate_recall_generic(hybrid, k, n_probe, ground_truth)
        r['phase'] = 'hybrid_hnsw_level'
        r['hybrid_stats'] = hybrid.get_stats()
        return r

    def evaluate_pure_kmeans_from_existing(self, kmeans_hnsw: KMeansHNSW, k: int, ground_truth: Dict, n_probe: int) -> Dict[str, Any]:
        model = kmeans_hnsw.kmeans_model
        centers = model.cluster_centers_
        labels = model.labels_
        n_clusters = centers.shape[0]
        clusters = [[] for _ in range(n_clusters)]
        dataset_vectors = kmeans_hnsw._extract_dataset_vectors()
        idx_to_orig = list(kmeans_hnsw.base_index.keys())
        for di, cid in enumerate(labels):
            clusters[cid].append((di, idx_to_orig[di]))
        total_correct = 0
        total_expected = len(self.query_set) * k
        q_times = []
        indiv = []
        n_probe_eff = min(n_probe, n_clusters)
        for qv, qid in zip(self.query_set, self.query_ids):
            t0 = time.time()
            dC = np.linalg.norm(centers - qv, axis=1)
            probe = np.argpartition(dC, n_probe_eff - 1)[:n_probe_eff]
            probe = probe[np.argsort(dC[probe])]
            cand = []
            for pc in probe:
                for ds_idx, orig_id in clusters[pc]:
                    if orig_id == qid:
                        continue
                    vec = dataset_vectors[ds_idx]
                    d = np.linalg.norm(vec - qv)
                    cand.append((d, orig_id))
            cand.sort(key=lambda x: x[0])
            found_ids = {oid for _, oid in cand[:k]}
            true_ids = {nid for _, nid in ground_truth[qid]}
            correct = len(found_ids & true_ids)
            total_correct += correct
            indiv.append(correct / k if k else 0)
            q_times.append(time.time() - t0)
        return {
            'phase': 'clusters_only',
            'recall_at_k': total_correct / total_expected if total_expected else 0.0,
            'avg_query_time_ms': float(np.mean(q_times) * 1000),
            'std_query_time_ms': float(np.std(q_times) * 1000),
            'n_probe': n_probe_eff
        }

    # ---------- Parameter Sweep (adds multi-pivot) ----------
    def parameter_sweep(self,
                        base_index: HNSW,
                        param_grid: Dict[str, List[Any]],
                        evaluation_params: Dict[str, Any],
                        max_combinations: Optional[int] = None,
                        adaptive_config: Optional[Dict[str, Any]] = None,
                        multi_pivot_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if adaptive_config is None:
            adaptive_config = {}
        if multi_pivot_config is None:
            multi_pivot_config = {'enabled': False}
        k_values = evaluation_params.get('k_values', [10])
        n_probe_values = evaluation_params.get('n_probe_values', [5, 10])
        hybrid_parent_level = evaluation_params.get('hybrid_parent_level', 2)
        enable_hybrid = evaluation_params.get('enable_hybrid', True)
        param_names = list(param_grid.keys())
        combinations = list(product(*param_grid.values()))
        if max_combinations and len(combinations) > max_combinations:
            combinations = random.sample(combinations, max_combinations)
        print(f"参数组合数量: {len(combinations)} (with multi-pivot={multi_pivot_config.get('enabled')})")
        # Precompute ground truth for each k
        gts = {k: self.compute_ground_truth(k, exclude_query_ids=False) for k in k_values}
        results = []
        for idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            print(f"\n--- 组合 {idx+1}/{len(combinations)} params={params} ---")
            phase_records = []
            try:
                # Build single-pivot KMeans HNSW
                build_start = time.time()
                kmeans_sys = KMeansHNSW(
                    base_index=base_index,
                    **params,
                    adaptive_k_children=adaptive_config.get('adaptive_k_children', False),
                    k_children_scale=adaptive_config.get('k_children_scale', 1.5),
                    k_children_min=adaptive_config.get('k_children_min', 100),
                    k_children_max=adaptive_config.get('k_children_max'),
                    diversify_max_assignments=adaptive_config.get('diversify_max_assignments'),
                    repair_min_assignments=adaptive_config.get('repair_min_assignments')
                )
                build_time = time.time() - build_start
                # Baseline HNSW
                base_ef = base_index._ef_construction
                for k in k_values:
                    b_eval = self.evaluate_hnsw_baseline(base_index, k, base_ef, gts[k])
                    phase_records.append({**b_eval, 'k': k})
                # Pure KMeans
                for k in k_values:
                    for n_probe in n_probe_values:
                        c_eval = self.evaluate_pure_kmeans_from_existing(kmeans_sys, k, gts[k], n_probe)
                        phase_records.append({**c_eval, 'k': k})
                # Hybrid HNSW
                if enable_hybrid:
                    try:
                        hybrid = HNSWHybrid(
                            base_index=base_index,
                            parent_level=hybrid_parent_level,
                            k_children=params['k_children'],
                            approx_ef=params.get('child_search_ef'),
                            adaptive_k_children=adaptive_config.get('adaptive_k_children', False),
                            k_children_scale=adaptive_config.get('k_children_scale', 1.5),
                            k_children_min=adaptive_config.get('k_children_min', 100),
                            k_children_max=adaptive_config.get('k_children_max')
                        )
                        for k in k_values:
                            for n_probe in n_probe_values:
                                h_eval = self.evaluate_hybrid(hybrid, k, n_probe, gts[k])
                                phase_records.append({**h_eval, 'k': k})
                    except Exception as he:
                        print(f"Hybrid HNSW 失败: {he}")
                # Single-pivot recall
                for k in k_values:
                    for n_probe in n_probe_values:
                        sp_eval = self.evaluate_recall_generic(kmeans_sys, k, n_probe, gts[k])
                        sp_eval['phase'] = 'kmeans_hnsw_single'
                        sp_eval['system_stats'] = kmeans_sys.get_stats()
                        phase_records.append({**sp_eval, 'k': k})
                # Multi-pivot
                if multi_pivot_config.get('enabled'):
                    mp_build_start = time.time()
                    mp_sys = KMeansHNSWMultiPivot(
                        base_index=base_index,
                        n_clusters=params['n_clusters'],
                        k_children=params['k_children'],
                        child_search_ef=params['child_search_ef'],
                        num_pivots=multi_pivot_config.get('num_pivots', 3),
                        pivot_selection_strategy=multi_pivot_config.get('pivot_selection_strategy', 'line_perp_third'),
                        pivot_overquery_factor=multi_pivot_config.get('pivot_overquery_factor', 1.2),
                        multi_pivot_enabled=True
                    )
                    mp_build_time = time.time() - mp_build_start
                    for k in k_values:
                        for n_probe in n_probe_values:
                            mp_eval = self.evaluate_recall_generic(mp_sys, k, n_probe, gts[k])
                            mp_eval['phase'] = 'kmeans_hnsw_multi_pivot'
                            mp_eval['system_stats'] = mp_sys.get_stats()
                            mp_eval['multi_pivot_build_time'] = mp_build_time
                            phase_records.append({**mp_eval, 'k': k})
                best_recall = max(r['recall_at_k'] for r in phase_records if 'recall_at_k' in r)
                results.append({
                    'parameters': params,
                    'construction_time_single_pivot': build_time,
                    'phase_evaluations': phase_records,
                    'best_recall': best_recall
                })
                print(f"组合完成 best_recall={best_recall:.4f}")
            except Exception as e:
                print(f"组合 {params} 出错: {e}")
                continue
        return results


def save_results(results: Dict[str, Any], filename: str):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, list):
            return [conv(x) for x in o]
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        return o
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conv(results), f, indent=2, ensure_ascii=False)
    print(f"结果已保存 -> {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Pivot KMeans HNSW 参数扫描与评估 (extended)")
    parser.add_argument('--dataset-size', type=int, default=8000)
    parser.add_argument('--query-size', type=int, default=60)
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--no-sift', action='store_true')
    # 基础 method3 参数 (少量保留, 其余使用默认)
    parser.add_argument('--hybrid-parent-level', type=int, default=2)
    parser.add_argument('--no-hybrid', action='store_true')
    # 自适应/多样化/修复
    parser.add_argument('--adaptive-k-children', action='store_true')
    parser.add_argument('--k-children-scale', type=float, default=1.5)
    parser.add_argument('--k-children-min', type=int, default=100)
    parser.add_argument('--k-children-max', type=int, default=None)
    parser.add_argument('--diversify-max-assignments', type=int, default=None)
    parser.add_argument('--repair-min-assignments', type=int, default=None)
    # Multi-pivot 控制
    parser.add_argument('--enable-multi-pivot', action='store_true')
    parser.add_argument('--num-pivots', type=int, default=3)
    parser.add_argument('--pivot-selection-strategy', type=str, default='line_perp_third')
    parser.add_argument('--pivot-overquery-factor', type=float, default=1.2)
    args = parser.parse_args()

    print("🔬 Multi-Pivot 扩展参数扫描开始")
    # Synthetic data (保持与原脚本风格一致, 简化 SIFT 支持)
    base_vectors = np.random.randn(args.dataset_size, args.dimension).astype(np.float32)
    query_vectors = np.random.randn(args.query_size, args.dimension).astype(np.float32)
    query_ids = list(range(len(query_vectors)))
    distance_func = lambda a, b: np.linalg.norm(a - b)

    # Base HNSW
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    for i, vec in enumerate(base_vectors):
        base_index.insert(i, vec)
        if (i + 1) % 1000 == 0:
            print(f"  插入 {i+1}/{len(base_vectors)} vectors")

    evaluator = KMeansHNSWMultiPivotEvaluator(base_vectors, query_vectors, query_ids, distance_func)

    # 设定与原 tune 脚本类似的 param_grid (简化)
    if args.dataset_size <= 2000:
        cluster_options = [10]
    elif args.dataset_size <= 5000:
        cluster_options = [16, 32]
    else:
        cluster_options = [32, 64]
    param_grid = {
        'n_clusters': cluster_options,
        'k_children': [200],
        'child_search_ef': [300]
    }
    evaluation_params = {
        'k_values': [10],
        'n_probe_values': [5, 10, 20],
        'hybrid_parent_level': args.hybrid_parent_level,
        'enable_hybrid': (not args.no_hybrid)
    }
    adaptive_config = {
        'adaptive_k_children': args.adaptive_k_children,
        'k_children_scale': args.k_children_scale,
        'k_children_min': args.k_children_min,
        'k_children_max': args.k_children_max,
        'diversify_max_assignments': args.diversify_max_assignments,
        'repair_min_assignments': args.repair_min_assignments
    }
    multi_pivot_config = {
        'enabled': args.enable_multi_pivot,
        'num_pivots': args.num_pivots,
        'pivot_selection_strategy': args.pivot_selection_strategy,
        'pivot_overquery_factor': args.pivot_overquery_factor
    }

    results = evaluator.parameter_sweep(
        base_index,
        param_grid,
        evaluation_params,
        max_combinations=None,
        adaptive_config=adaptive_config,
        multi_pivot_config=multi_pivot_config
    )

    output = {
        'sweep_results': results,
        'multi_pivot_enabled': args.enable_multi_pivot,
        'multi_pivot_config': multi_pivot_config,
        'adaptive_config': adaptive_config,
        'dataset': {
            'size': len(base_vectors),
            'query_size': len(query_vectors),
            'dimension': base_vectors.shape[1]
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    save_results(output, 'multi_pivot_parameter_sweep.json')
    print("✅ 扫描完成 (done)")
