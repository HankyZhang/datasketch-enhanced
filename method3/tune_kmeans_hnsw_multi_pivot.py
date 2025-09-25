"""
方法3参数调优：K-Means HNSW系统 + Multi-Pivot扩展 (Method 3 Parameter Tuning: K-Means HNSW System + Multi-Pivot Extension)

本模块基于 tune_kmeans_hnsw.py，额外添加了Multi-Pivot KMeans HNSW的评估和对比。
提供五种方案的全面对比评估：

1. HNSW基线 (HNSW Baseline) - 纯HNSW索引
2. 纯K-Means (Pure K-Means) - 仅使用K-Means聚类
3. Hybrid HNSW - 基于HNSW层级的混合索引
4. KMeans HNSW (单枢纽) - 单枢纽K-Means HNSW混合系统
5. Multi-Pivot KMeans HNSW (多枢纽) - 多枢纽K-Means HNSW混合系统

功能特性:
- 全面的参数扫描和优化
- 五种方案的基准对比评估
- 召回率和查询时间分析
- Multi-Pivot枢纽策略测试
- 自适应参数调整
- 结果保存和分析

This module extends tune_kmeans_hnsw.py with Multi-Pivot KMeans HNSW evaluation.
It provides comprehensive comparison evaluation for five approaches.
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

# 添加父目录到路径 (Add parent directory to path)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from method3.kmeans_hnsw import KMeansHNSW
from method3.kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot
from hnsw.hnsw import HNSW
# 引入 Hybrid HNSW 结构用于对比评估 (Import Hybrid HNSW for comparative evaluation)
from hybrid_hnsw.hnsw_hybrid import HNSWHybrid
# 使用sklearn MiniBatchKMeans作为纯k-means基线 (Switch to sklearn MiniBatchKMeans for pure k-means baseline)
from sklearn.cluster import MiniBatchKMeans


class KMeansHNSWMultiPivotEvaluator:
    """
    K-Means HNSW系统性能全面评估器 (含Multi-Pivot扩展)
    (Comprehensive evaluator for K-Means HNSW system performance with Multi-Pivot extension)
    
    此类提供了对K-Means HNSW系统的全面评估功能，包括：
    - 真实值(Ground Truth)计算
    - 召回率评估
    - 参数扫描和优化
    - 与基线方法的性能对比 (包括Multi-Pivot方案)
    """
    
    def __init__(
        self, 
        dataset: np.ndarray, 
        query_set: np.ndarray, 
        query_ids: List[int],
        distance_func: callable
    ):
        """
        初始化评估器 (Initialize the evaluator)
        
        Args:
            dataset: 完整数据集向量 (Full dataset vectors) - shape: [n_vectors, dim]
            query_set: 查询向量 (Query vectors) - shape: [n_queries, dim]  
            query_ids: 查询向量ID列表 (IDs for query vectors)
            distance_func: 用于真实值计算的距离函数 (Distance function for ground truth computation)
        """
        self.dataset = dataset
        self.query_set = query_set
        self.query_ids = query_ids
        self.distance_func = distance_func
        
        # 真实值缓存 (Ground truth cache)
        self._ground_truth_cache = {}
    
    def compute_ground_truth(
        self, 
        k: int, 
        exclude_query_ids: bool = True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        使用暴力搜索计算真实值 (Compute ground truth using brute force search)
        
        通过对每个查询向量计算与所有数据向量的距离，找出真正的k个最近邻。
        这是评估其他算法召回率的标准基准。
        
        Args:
            k: 最近邻数量 (Number of nearest neighbors)
            exclude_query_ids: 是否从结果中排除查询ID (Whether to exclude query IDs from results)
            
        Returns:
            字典：查询ID -> (距离, 数据索引)元组列表 (Dictionary mapping query_id to list of (distance, data_index) tuples)
        """
        cache_key = (k, exclude_query_ids)
        if cache_key in self._ground_truth_cache:
            return self._ground_truth_cache[cache_key]
        
        print(f"正在计算 {len(self.query_set)} 个查询的真实值 (k={k}, exclude_query_ids={exclude_query_ids})...")
        start_time = time.time()
        
        ground_truth = {}
        excluded_count = 0
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            distances = []
            
            for j, data_vector in enumerate(self.dataset):
                if exclude_query_ids and j == query_id:
                    excluded_count += 1
                    continue  # 跳过查询本身 (Skip the query itself)
                
                distance = self.distance_func(query_vector, data_vector)
                distances.append((distance, j))
            
            # 按距离排序并取前k个 (Sort by distance and take top-k)
            distances.sort()
            ground_truth[query_id] = distances[:k]
        
        elapsed = time.time() - start_time
        print(f"真实值计算完成，耗时 {elapsed:.2f}秒，排除了 {excluded_count} 个数据点")
        
        self._ground_truth_cache[cache_key] = ground_truth
        return ground_truth
    
    def evaluate_recall(
        self,
        kmeans_hnsw: KMeansHNSW,
        k: int,
        n_probe: int,
        ground_truth: Optional[Dict] = None,
        exclude_query_ids: bool = True
    ) -> Dict[str, Any]:
        """
        评估K-Means HNSW系统的召回率性能 (Evaluate recall performance of the K-Means HNSW system)
        """
        if ground_truth is None:
            ground_truth = self.compute_ground_truth(k, exclude_query_ids)
        
        print(f"正在评估 {len(self.query_set)} 个查询的召回率 (k={k}, n_probe={n_probe})...")
        start_time = time.time()
        
        total_correct = 0
        total_expected = len(self.query_set) * k
        query_times = []
        individual_recalls = []
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            # Get ground truth for this query
            true_neighbors = {neighbor_id for _, neighbor_id in ground_truth[query_id]}
            
            # Search using K-Means HNSW
            search_start = time.time()
            results = kmeans_hnsw.search(query_vector, k=k, n_probe=n_probe)
            search_time = time.time() - search_start
            query_times.append(search_time)
            
            # Count correct results
            found_neighbors = {node_id for node_id, _ in results}
            correct = len(true_neighbors & found_neighbors)
            total_correct += correct
            
            # Individual recall for this query
            individual_recall = correct / k if k > 0 else 0.0
            individual_recalls.append(individual_recall)
        
        # Calculate final metrics
        overall_recall = total_correct / total_expected
        avg_query_time = np.mean(query_times)
        std_query_time = np.std(query_times)
        total_evaluation_time = time.time() - start_time
        
        return {
            'recall_at_k': overall_recall,
            'total_correct': total_correct,
            'total_expected': total_expected,
            'individual_recalls': individual_recalls,
            'avg_individual_recall': np.mean(individual_recalls),
            'std_individual_recall': np.std(individual_recalls),
            'avg_query_time_ms': avg_query_time * 1000,
            'std_query_time_ms': std_query_time * 1000,
            'total_evaluation_time': total_evaluation_time,
            'k': k,
            'n_probe': n_probe,
            'system_stats': kmeans_hnsw.get_stats()
        }
    
    def evaluate_multi_pivot_recall(
        self,
        multi_pivot_hnsw: KMeansHNSWMultiPivot,
        k: int,
        n_probe: int,
        ground_truth: Optional[Dict] = None,
        exclude_query_ids: bool = True
    ) -> Dict[str, Any]:
        """
        评估Multi-Pivot K-Means HNSW系统的召回率性能
        (Evaluate recall performance of the Multi-Pivot K-Means HNSW system)
        """
        if ground_truth is None:
            ground_truth = self.compute_ground_truth(k, exclude_query_ids)
        
        print(f"正在评估Multi-Pivot {len(self.query_set)} 个查询的召回率 (k={k}, n_probe={n_probe})...")
        start_time = time.time()
        
        total_correct = 0
        total_expected = len(self.query_set) * k
        query_times = []
        individual_recalls = []
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            # Get ground truth for this query
            true_neighbors = {neighbor_id for _, neighbor_id in ground_truth[query_id]}
            
            # Search using Multi-Pivot K-Means HNSW
            search_start = time.time()
            results = multi_pivot_hnsw.search(query_vector, k=k, n_probe=n_probe)
            search_time = time.time() - search_start
            query_times.append(search_time)
            
            # Count correct results
            found_neighbors = {node_id for node_id, _ in results}
            correct = len(true_neighbors & found_neighbors)
            total_correct += correct
            
            # Individual recall for this query
            individual_recall = correct / k if k > 0 else 0.0
            individual_recalls.append(individual_recall)
        
        # Calculate final metrics
        overall_recall = total_correct / total_expected
        avg_query_time = np.mean(query_times)
        std_query_time = np.std(query_times)
        total_evaluation_time = time.time() - start_time
        
        return {
            'recall_at_k': overall_recall,
            'total_correct': total_correct,
            'total_expected': total_expected,
            'individual_recalls': individual_recalls,
            'avg_individual_recall': np.mean(individual_recalls),
            'std_individual_recall': np.std(individual_recalls),
            'avg_query_time_ms': avg_query_time * 1000,
            'std_query_time_ms': std_query_time * 1000,
            'total_evaluation_time': total_evaluation_time,
            'k': k,
            'n_probe': n_probe,
            'system_stats': multi_pivot_hnsw.get_stats()
        }

    def evaluate_hnsw_baseline(
        self,
        base_index: HNSW,
        k: int,
        ef: int,
        ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate recall using ONLY the base HNSW index (Phase 1)."""
        query_times = []
        total_correct = 0
        total_expected = len(self.query_set) * k
        individual_recalls = []
        
        print(f"🔍 评估HNSW基线性能 (k={k}, ef={ef})...")
        
        for query_vector, query_id in zip(self.query_set, self.query_ids):
            true_neighbors = {node_id for _, node_id in ground_truth[query_id]}
            
            t0 = time.time()
            results = base_index.query(query_vector, k=k, ef=ef)
            dt = time.time() - t0
            query_times.append(dt)
            
            found = {nid for nid, _ in results}
            correct = len(true_neighbors & found)
            total_correct += correct
            
            individual_recall = correct / k if k > 0 else 0.0
            individual_recalls.append(individual_recall)
        
        avg_recall = np.mean(individual_recalls)
        print(f"  HNSW基线召回率: {avg_recall:.4f} (总计 {total_correct}/{total_expected})")
        
        return {
            'phase': 'baseline_hnsw',
            'ef': ef,
            'recall_at_k': total_correct / total_expected if total_expected else 0.0,
            'avg_query_time_ms': float(np.mean(query_times) * 1000),
            'std_query_time_ms': float(np.std(query_times) * 1000),
            'total_correct': total_correct,
            'total_expected': total_expected,
            'individual_recalls': individual_recalls,
            'avg_individual_recall': float(np.mean(individual_recalls)),
            'std_individual_recall': float(np.std(individual_recalls))
        }

    def evaluate_hybrid_hnsw(
        self,
        hybrid_index: 'HNSWHybrid',
        k: int,
        n_probe: int,
        ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate recall for level-based Hybrid HNSW (parents from HNSW levels)."""
        query_times = []
        total_correct = 0
        total_expected = len(self.query_set) * k
        individual_recalls = []

        for query_vector, query_id in zip(self.query_set, self.query_ids):
            true_neighbors = {node_id for _, node_id in ground_truth[query_id]}
            t0 = time.time()
            results = hybrid_index.search(query_vector, k=k, n_probe=n_probe)
            dt = time.time() - t0
            query_times.append(dt)
            found = {nid for nid, _ in results}
            correct = len(true_neighbors & found)
            total_correct += correct
            individual_recalls.append(correct / k if k > 0 else 0.0)

        return {
            'phase': 'hybrid_hnsw_level',
            'n_probe': n_probe,
            'recall_at_k': total_correct / total_expected if total_expected else 0.0,
            'avg_query_time_ms': float(np.mean(query_times) * 1000),
            'std_query_time_ms': float(np.std(query_times) * 1000),
            'total_correct': total_correct,
            'total_expected': total_expected,
            'individual_recalls': individual_recalls,
            'avg_individual_recall': float(np.mean(individual_recalls)),
            'std_individual_recall': float(np.std(individual_recalls)),
            'hybrid_stats': hybrid_index.get_stats()
        }

    def _evaluate_pure_kmeans_from_existing(
        self, 
        kmeans_hnsw: KMeansHNSW, 
        k: int, 
        ground_truth: Dict, 
        n_probe: int = 1
    ) -> Dict[str, Any]:
        """
        使用KMeansHNSW内部已有的聚类结果评估纯K-Means性能
        避免重复聚类，直接复用已训练的模型和数据映射
        """
        print(f"复用现有聚类评估纯K-Means (n_clusters={kmeans_hnsw.n_clusters}, n_probe={n_probe})...")
        
        # 直接使用KMeansHNSW内部的聚类结果
        kmeans_model = kmeans_hnsw.kmeans_model
        centers = kmeans_model.cluster_centers_
        n_clusters = centers.shape[0]
        
        # 获取与聚类时相同的数据集和索引映射
        kmeans_dataset = kmeans_hnsw._extract_dataset_vectors()
        labels = kmeans_model.labels_
        
        # 构建聚类到成员的映射
        clusters = [[] for _ in range(n_clusters)]
        dataset_idx_to_original_id = list(kmeans_hnsw.base_index.keys())
        
        for dataset_idx, cluster_id in enumerate(labels):
            original_id = dataset_idx_to_original_id[dataset_idx]
            clusters[cluster_id].append((dataset_idx, original_id))
        
        query_times = []
        total_correct = 0
        total_expected = len(self.query_set) * k
        individual_recalls = []
        
        # Cap n_probe to number of clusters
        n_probe_eff = min(n_probe, n_clusters)
        
        for query_vector, query_id in zip(self.query_set, self.query_ids):
            search_start = time.time()
            
            # 计算到聚类中心的距离
            diffs = centers - query_vector
            distances_to_centroids = np.linalg.norm(diffs, axis=1)
            
            # 获取最近的n_probe个聚类中心
            probe_centroids = np.argpartition(distances_to_centroids, n_probe_eff - 1)[:n_probe_eff]
            probe_centroids = probe_centroids[np.argsort(distances_to_centroids[probe_centroids])]
            
            # 收集所有被探测聚类的成员
            all_candidates = []
            for cluster_idx in probe_centroids:
                cluster_members = clusters[cluster_idx]
                for dataset_idx, original_id in cluster_members:
                    if original_id != query_id:  # 排除查询本身
                        member_vec = kmeans_dataset[dataset_idx]
                        dist = np.linalg.norm(member_vec - query_vector)
                        all_candidates.append((dist, original_id))
            
            # 按距离排序并取top-k
            all_candidates.sort(key=lambda x: x[0])
            results = all_candidates[:k]
            found_neighbors = {original_id for _, original_id in results}
            
            search_time = time.time() - search_start
            query_times.append(search_time)
            
            # 计算召回率
            true_neighbors = {neighbor_id for _, neighbor_id in ground_truth[query_id]}
            correct = len(true_neighbors & found_neighbors)
            total_correct += correct
            individual_recalls.append(correct / k if k > 0 else 0.0)
        
        overall_recall = total_correct / total_expected if total_expected > 0 else 0.0
        
        return {
            'method': 'pure_kmeans_from_existing',
            'recall_at_k': overall_recall,
            'total_correct': total_correct,
            'total_expected': total_expected,
            'individual_recalls': individual_recalls,
            'avg_individual_recall': float(np.mean(individual_recalls)),
            'std_individual_recall': float(np.std(individual_recalls)),
            'avg_query_time_ms': float(np.mean(query_times) * 1000),
            'std_query_time_ms': float(np.std(query_times) * 1000),
            'clustering_time': 0.0,  # 没有重新聚类，时间为0
            'n_clusters': n_clusters,
            'n_probe': n_probe_eff,
            'k': k,
            'reused_existing_clustering': True
        }

    def parameter_sweep(
        self,
        base_index: HNSW,
        param_grid: Dict[str, List[Any]],
        evaluation_params: Dict[str, Any],
        max_combinations: Optional[int] = None,
        adaptive_config: Optional[Dict[str, Any]] = None,
        multi_pivot_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行全面的参数扫描优化 (含Multi-Pivot扩展)
        (Perform comprehensive parameter sweep for optimization with Multi-Pivot extension)

        通过系统性地测试不同参数组合，找到最优的K-Means HNSW配置。
        包括基线HNSW、纯K-Means、Hybrid HNSW、单枢纽KMeans-HNSW以及Multi-Pivot KMeans-HNSW的对比评估。
        """
        if adaptive_config is None:
            adaptive_config = {
                'adaptive_k_children': False,
                'k_children_scale': 1.5,
                'k_children_min': 100,
                'k_children_max': None,
                'diversify_max_assignments': None,
                'repair_min_assignments': None
            }

        if multi_pivot_config is None:
            multi_pivot_config = {
                'enabled': False,
                'num_pivots': 3,
                'pivot_selection_strategy': 'line_perp_third',
                'pivot_overquery_factor': 1.2
            }

        print("开始K-Means HNSW + Multi-Pivot 参数扫描...")
        print(f"Multi-Pivot启用状态: {multi_pivot_config.get('enabled', False)}")

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        if max_combinations and len(combinations) > max_combinations:
            print(f"限制测试 {max_combinations} 个组合，总共 {len(combinations)} 个")
            combinations = random.sample(combinations, max_combinations)
        print(f"测试 {len(combinations)} 个参数组合...")

        results: List[Dict[str, Any]] = []
        k_values = evaluation_params.get('k_values', [10])
        n_probe_values = evaluation_params.get('n_probe_values', [5, 10, 20])
        hybrid_parent_level = evaluation_params.get('hybrid_parent_level', 2)
        enable_hybrid = evaluation_params.get('enable_hybrid', True)

        # Precompute ground truths
        ground_truths: Dict[int, Dict] = {}
        for k in k_values:
            ground_truths[k] = self.compute_ground_truth(k, exclude_query_ids=False)

        for i, combination in enumerate(combinations):
            print(f"\n--- Combination {i + 1}/{len(combinations)} ---")
            params = dict(zip(param_names, combination))
            print(f"Parameters: {params}")

            try:
                phase_records: List[Dict[str, Any]] = []
                
                # Build single-pivot KMeans-HNSW system
                construction_start = time.time()
                kmeans_hnsw = KMeansHNSW(
                    base_index=base_index,
                    **params,
                    adaptive_k_children=adaptive_config['adaptive_k_children'],
                    k_children_scale=adaptive_config['k_children_scale'],
                    k_children_min=adaptive_config['k_children_min'],
                    k_children_max=adaptive_config['k_children_max'],
                    diversify_max_assignments=adaptive_config['diversify_max_assignments'],
                    repair_min_assignments=adaptive_config['repair_min_assignments']
                )
                construction_time = time.time() - construction_start
                print(f"  构建K-Means HNSW系统耗时 {construction_time:.2f}秒")
                actual_n_clusters = kmeans_hnsw.n_clusters

                # Phase 1: Baseline HNSW
                base_ef = base_index._ef_construction
                print(f"  使用base_index的ef_construction参数: {base_ef}")
                for k in k_values:
                    b_eval = self.evaluate_hnsw_baseline(base_index, k, base_ef, ground_truths[k])
                    phase_records.append({**b_eval, 'k': k})
                    print(f"  [基线HNSW] k={k} ef={base_ef} recall={b_eval['recall_at_k']:.4f} avg_time={b_eval['avg_query_time_ms']:.2f}ms")

                # Phase 2: Pure KMeans (reuse clustering)
                print(f"  使用与KMeansHNSW相同的聚类参数: n_clusters={actual_n_clusters}")
                for k in k_values:
                    for n_probe in n_probe_values:
                        c_eval = self._evaluate_pure_kmeans_from_existing(
                            kmeans_hnsw,
                            k,
                            ground_truths[k],
                            n_probe=n_probe
                        )
                        c_eval['phase'] = 'clusters_only'
                        phase_records.append({**c_eval, 'k': k})
                        print(f"  [仅K-Means聚类] k={k} n_probe={n_probe} recall={c_eval['recall_at_k']:.4f} avg_time={c_eval['avg_query_time_ms']:.2f}ms")

                # Phase 3: Level-based Hybrid HNSW
                if enable_hybrid:
                    print(f"  构建并评估Hybrid HNSW (parent_level={hybrid_parent_level}, k_children={params['k_children']})")
                    try:
                        hybrid_build_start = time.time()
                        hybrid_index = HNSWHybrid(
                            base_index=base_index,
                            parent_level=hybrid_parent_level,
                            k_children=params['k_children'],
                            approx_ef=params.get('child_search_ef'),
                            parent_child_method='approx',
                            diversify_max_assignments=adaptive_config.get('diversify_max_assignments'),
                            repair_min_assignments=adaptive_config.get('repair_min_assignments'),
                            adaptive_k_children=adaptive_config.get('adaptive_k_children', False),
                            k_children_scale=adaptive_config.get('k_children_scale', 1.5),
                            k_children_min=adaptive_config.get('k_children_min', 100),
                            k_children_max=adaptive_config.get('k_children_max')
                        )
                        hybrid_build_time = time.time() - hybrid_build_start
                        hybrid_stats = hybrid_index.get_stats()
                        print(f"    Hybrid构建统计: {hybrid_stats.get('num_parents', 0)} parents, "
                              f"{hybrid_stats.get('num_children', 0)} children, "
                              f"coverage: {hybrid_stats.get('coverage_fraction', 0):.4f}")
                        
                        for k in k_values:
                            for n_probe in n_probe_values:
                                h_eval = self.evaluate_hybrid_hnsw(hybrid_index, k, n_probe, ground_truths[k])
                                h_eval['hybrid_build_time'] = hybrid_build_time
                                h_eval['hybrid_k_children'] = hybrid_stats.get('k_children', params['k_children'])
                                phase_records.append({**h_eval, 'k': k})
                                print(f"  [Hybrid(Level)] k={k} n_probe={n_probe} recall={h_eval['recall_at_k']:.4f} avg_time={h_eval['avg_query_time_ms']:.2f}ms")
                    except Exception as he:
                        print(f"  ⚠️ Hybrid HNSW 构建或评估失败: {he}")

                # Phase 4: Single-pivot KMeans-HNSW hybrid
                for k in k_values:
                    for n_probe in n_probe_values:
                        eval_result = self.evaluate_recall(kmeans_hnsw, k, n_probe, ground_truths[k])
                        phase_records.append({**eval_result, 'phase': 'kmeans_hnsw_single_pivot'})
                        print(f"  [K-Means HNSW单枢纽] k={k} n_probe={n_probe} recall={eval_result['recall_at_k']:.4f} avg_time={eval_result['avg_query_time_ms']:.2f}ms")

                # Phase 5: Multi-pivot KMeans-HNSW hybrid
                if multi_pivot_config.get('enabled', False):
                    print(f"  构建并评估Multi-Pivot KMeans HNSW (num_pivots={multi_pivot_config.get('num_pivots', 3)})")
                    try:
                        mp_build_start = time.time()
                        multi_pivot_system = KMeansHNSWMultiPivot(
                            base_index=base_index,
                            n_clusters=params['n_clusters'],
                            k_children=params['k_children'],
                            child_search_ef=params.get('child_search_ef'),
                            # Multi-pivot specific parameters
                            num_pivots=multi_pivot_config.get('num_pivots', 3),
                            pivot_selection_strategy=multi_pivot_config.get('pivot_selection_strategy', 'line_perp_third'),
                            pivot_overquery_factor=multi_pivot_config.get('pivot_overquery_factor', 1.2),
                            multi_pivot_enabled=True,
                            store_pivot_debug=True,
                            # Adaptive/diversify/repair config
                            adaptive_k_children=adaptive_config['adaptive_k_children'],
                            k_children_scale=adaptive_config['k_children_scale'],
                            k_children_min=adaptive_config['k_children_min'],
                            k_children_max=adaptive_config['k_children_max'],
                            diversify_max_assignments=adaptive_config['diversify_max_assignments'],
                            repair_min_assignments=adaptive_config['repair_min_assignments']
                        )
                        mp_build_time = time.time() - mp_build_start
                        mp_stats = multi_pivot_system.get_stats()
                        print(f"    Multi-Pivot构建统计: n_clusters={mp_stats.get('n_clusters', 0)}, "
                              f"avg_children={mp_stats.get('avg_children_per_cluster', 0):.1f}")
                        
                        for k in k_values:
                            for n_probe in n_probe_values:
                                mp_eval = self.evaluate_multi_pivot_recall(multi_pivot_system, k, n_probe, ground_truths[k])
                                mp_eval['phase'] = 'kmeans_hnsw_multi_pivot'
                                mp_eval['multi_pivot_build_time'] = mp_build_time
                                mp_eval['multi_pivot_config'] = multi_pivot_config
                                phase_records.append({**mp_eval, 'k': k})
                                print(f"  [K-Means HNSW多枢纽] k={k} n_probe={n_probe} recall={mp_eval['recall_at_k']:.4f} avg_time={mp_eval['avg_query_time_ms']:.2f}ms")
                    except Exception as me:
                        print(f"  ⚠️ Multi-Pivot KMeans HNSW 构建或评估失败: {me}")

                combination_results = {
                    'parameters': params,
                    'construction_time': construction_time,
                    'phase_evaluations': phase_records,
                    'multi_pivot_enabled': multi_pivot_config.get('enabled', False)
                }
                results.append(combination_results)
                best_recall = max(r['recall_at_k'] for r in phase_records if 'recall_at_k' in r)
                print(f"  此组合最佳召回率: {best_recall:.4f}")
                
            except Exception as e:
                print(f"❌ 参数组合 {params} 出错: {e}")
                continue

        print(f"\n🎯 参数扫描完成！测试了 {len(results)} 个组合")
        print(f"    包含Multi-Pivot评估: {multi_pivot_config.get('enabled', False)}")
        return results


def save_results(results: Dict[str, Any], filename: str):
    """Save evaluation results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filename, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {filename}")


def load_sift_data():
    """
    加载SIFT数据集用于评估 (Load SIFT dataset for evaluation)
    
    SIFT (Scale-Invariant Feature Transform) 是计算机视觉领域的经典特征描述符。
    该数据集包含100万个128维的特征向量，常用于相似性搜索算法的基准测试。
    
    Returns:
        tuple: (base_vectors, query_vectors) 或 (None, None) 如果加载失败
    """
    sift_dir = os.path.join(os.path.dirname(__file__), '..', 'sift')
    
    try:
        def read_fvecs(path: str, max_vectors: Optional[int] = None) -> np.ndarray:
            """
            读取.fvecs文件 (FAISS格式)。每个向量存储为：int32维度 + 维度个float32值。
            
            Read .fvecs file (FAISS format). Each vector stored as: int32 dim + dim float32.
            """
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            raw = np.fromfile(path, dtype=np.int32)
            if raw.size == 0:
                raise ValueError(f"空的fvecs文件: {path}")
            dim = raw[0]
            if dim <= 0 or dim > 4096:
                raise ValueError(f"不合理的向量维度 {dim}，解析自 {path}")
            record_size = dim + 1
            count = raw.size // record_size
            raw = raw.reshape(count, record_size)
            vecs = raw[:, 1:].astype(np.float32)
            if max_vectors is not None and count > max_vectors:
                vecs = vecs[:max_vectors]
            return vecs

        base_path = os.path.join(sift_dir, 'sift_base.fvecs')
        query_path = os.path.join(sift_dir, 'sift_query.fvecs')

        # 为调优演示限制数量以保持合理的运行时间
        base_vectors = read_fvecs(base_path, max_vectors=50000)
        query_vectors = read_fvecs(query_path, max_vectors=1000)

        print(f"已加载SIFT数据: {base_vectors.shape[0]} 个基础向量, "
              f"{query_vectors.shape[0]} 个查询向量, 维度 {base_vectors.shape[1]}")

        return base_vectors, query_vectors
    
    except Exception as e:
        print(f"加载SIFT数据时出错: {e}")
        print("改用合成数据...")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Means HNSW + Multi-Pivot参数调优和评估")
    
    # 数据集选项
    parser.add_argument('--dataset-size', type=int, default=10000, 
                        help='使用的基础向量数量 (默认: 10000)')
    parser.add_argument('--query-size', type=int, default=50, 
                        help='使用的查询向量数量 (默认: 50)')
    parser.add_argument('--dimension', type=int, default=128, 
                        help='合成数据的向量维度')
    parser.add_argument('--no-sift', action='store_true', 
                        help='强制使用合成数据，即使SIFT文件存在')
    
    # 自适应/多样化/修复选项
    parser.add_argument('--adaptive-k-children', action='store_true', 
                        help='启用基于平均聚类大小的自适应k_children')
    parser.add_argument('--k-children-scale', type=float, default=1.5, 
                        help='自适应k_children的缩放因子 (默认1.5)')
    parser.add_argument('--k-children-min', type=int, default=100, 
                        help='自适应时的最小k_children')
    parser.add_argument('--k-children-max', type=int, default=None, 
                        help='自适应时的最大k_children (可选)')
    parser.add_argument('--diversify-max-assignments', type=int, default=None, 
                        help='每个子节点的最大分配数 (启用多样化)')
    parser.add_argument('--repair-min-assignments', type=int, default=None, 
                        help='构建修复期间每个子节点的最小分配数')
    parser.add_argument('--hybrid-parent-level', type=int, default=2,
                        help='Hybrid HNSW 父节点层级 (默认:2)')
    parser.add_argument('--no-hybrid', action='store_true',
                        help='禁用Hybrid HNSW评估')
    
    # Multi-pivot 特定选项
    parser.add_argument('--enable-multi-pivot', action='store_true',
                        help='启用Multi-Pivot KMeans HNSW评估')
    parser.add_argument('--num-pivots', type=int, default=3,
                        help='每个聚类的枢纽点数量 (默认: 3)')
    parser.add_argument('--pivot-selection-strategy', type=str, default='line_perp_third',
                        choices=['line_perp_third', 'max_min_distance'],
                        help='枢纽点选择策略')
    parser.add_argument('--pivot-overquery-factor', type=float, default=1.2,
                        help='枢纽查询的过度查询因子 (默认: 1.2)')
    
    args = parser.parse_args()

    print("🔬 K-Means HNSW + Multi-Pivot参数调优和评估系统")
    print(f"📊 请求的数据集大小: {args.dataset_size}, 查询大小: {args.query_size}")
    print(f"🎯 Multi-Pivot启用状态: {args.enable_multi_pivot}")
    
    # 尝试加载SIFT数据，失败则使用合成数据
    base_vectors, query_vectors = (None, None)
    if not args.no_sift:
        base_vectors, query_vectors = load_sift_data()
    
    if base_vectors is None:
        # 创建合成数据
        print("🎲 创建合成数据集...")
        base_vectors = np.random.randn(max(args.dataset_size, 10000), args.dimension).astype(np.float32)
        query_vectors = np.random.randn(max(args.query_size, 100), args.dimension).astype(np.float32)
    
    # 切片到请求的大小
    if len(base_vectors) > args.dataset_size:
        base_vectors = base_vectors[:args.dataset_size]
    if len(query_vectors) > args.query_size:
        query_vectors = query_vectors[:args.query_size]
    print(f"📈 使用基础向量: {len(base_vectors)} | 查询: {len(query_vectors)} | 维度: {base_vectors.shape[1]}")
    query_ids = list(range(len(query_vectors)))
    
    # 距离函数
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # 构建基础HNSW索引
    print("🏗️  构建基础HNSW索引...")
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    for i, vector in enumerate(base_vectors):
        base_index.insert(i, vector)
        if (i + 1) % 1000 == 0:
            print(f"  Inserted {i + 1}/{len(base_vectors)} vectors")
    
    print(f"Base HNSW index built with {len(base_index)} vectors")
    
    # Initialize evaluator
    evaluator = KMeansHNSWMultiPivotEvaluator(base_vectors, query_vectors, query_ids, distance_func)
    
    # Define parameter grid for sweep
    if args.dataset_size <= 2000:
        cluster_options = [10]
    elif args.dataset_size <= 5000:
        cluster_options = [16, 32]
    else:
        cluster_options = [32, 64, 128]

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
    
    # 准备自适应配置
    adaptive_config = {
        'adaptive_k_children': args.adaptive_k_children,
        'k_children_scale': args.k_children_scale,
        'k_children_min': args.k_children_min,
        'k_children_max': args.k_children_max,
        'diversify_max_assignments': args.diversify_max_assignments,
        'repair_min_assignments': args.repair_min_assignments
    }
    
    # 准备Multi-Pivot配置
    multi_pivot_config = {
        'enabled': args.enable_multi_pivot,
        'num_pivots': args.num_pivots,
        'pivot_selection_strategy': args.pivot_selection_strategy,
        'pivot_overquery_factor': args.pivot_overquery_factor
    }
    
    # Perform parameter sweep
    print("\nStarting parameter sweep...")
    max_combos = 9 if len(cluster_options) > 1 else None
    
    sweep_results = evaluator.parameter_sweep(
        base_index,
        param_grid,
        evaluation_params,
        max_combinations=max_combos,
        adaptive_config=adaptive_config,
        multi_pivot_config=multi_pivot_config
    )
    
    # Save results
    if sweep_results:
        demo_result = sweep_results[0]
        demo_params = demo_result['parameters']
        print(f"\nUsing first parameter combination for demonstration: {demo_params}")
        print("\n🎯 Parameter sweep completed! All comparisons (including Multi-Pivot) are available in sweep_results.")

        results = {
            'sweep_results': sweep_results,
            'demo_parameters': demo_params,
            'multi_pivot_config': multi_pivot_config,
            'adaptive_config': adaptive_config,
            'evaluation_info': {
                'dataset_size': len(base_vectors),
                'query_size': len(query_vectors),
                'dimension': base_vectors.shape[1],
                'multi_pivot_enabled': args.enable_multi_pivot,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        save_results(results, 'multi_pivot_parameter_sweep.json')
        
    print(f"\n✅ Multi-Pivot parameter tuning completed!")
    if args.enable_multi_pivot:
        print("🎯 Five-method comparison results saved:")
        print("   1. HNSW基线 (HNSW Baseline)")
        print("   2. 纯K-Means (Pure K-Means)")
        print("   3. Hybrid HNSW")  
        print("   4. KMeans HNSW (单枢纽)")
        print("   5. Multi-Pivot KMeans HNSW (多枢纽)")
    else:
        print("💡 提示: 使用 --enable-multi-pivot 启用Multi-Pivot方案的对比评估")
