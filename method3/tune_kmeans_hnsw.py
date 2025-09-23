"""
方法3参数调优：K-Means HNSW系统 (Method 3 Parameter Tuning: K-Means HNSW System)

本模块提供K-Means HNSW系统的参数调优和优化功能。
包含全面的评估、参数扫描和性能分析。

功能特性:
- 全面的参数扫描和优化
- 基准对比评估 (HNSW基线、纯K-Means、K-Means HNSW)
- 召回率和查询时间分析
- 自适应参数调整
- 结果保存和分析

This module provides parameter tuning and optimization for the K-Means HNSW system.
It includes comprehensive evaluation, parameter sweeps, and performance analysis.
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
from hnsw.hnsw import HNSW
# 使用sklearn MiniBatchKMeans作为纯k-means基线 (Switch to sklearn MiniBatchKMeans for pure k-means baseline)
from sklearn.cluster import MiniBatchKMeans


class KMeansHNSWEvaluator:
    """
    K-Means HNSW系统性能全面评估器 (Comprehensive evaluator for K-Means HNSW system performance)
    
    此类提供了对K-Means HNSW系统的全面评估功能，包括：
    - 真实值(Ground Truth)计算
    - 召回率评估
    - 参数扫描和优化
    - 与基线方法的性能对比
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
            字典：查询ID -> (邻居ID, 距离)元组列表 (Dictionary mapping query_id to list of (neighbor_id, distance) tuples)
        """
        cache_key = (k, exclude_query_ids)
        if cache_key in self._ground_truth_cache:
            return self._ground_truth_cache[cache_key]
        
        print(f"正在计算 {len(self.query_set)} 个查询的真实值 (k={k})... (Computing ground truth for {len(self.query_set)} queries)")
        start_time = time.time()
        
        ground_truth = {}
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            distances = []
            
            for j, data_vector in enumerate(self.dataset):
                if exclude_query_ids and j == query_id:
                    continue  # 跳过查询本身 (Skip the query itself)
                
                distance = self.distance_func(query_vector, data_vector)
                distances.append((distance, j))
            
            # 按距离排序并取前k个 (Sort by distance and take top-k)
            distances.sort()
            ground_truth[query_id] = distances[:k]
            
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(self.query_set)} 个查询 (Processed {i + 1}/{len(self.query_set)} queries)")
        
        elapsed = time.time() - start_time
        print(f"真实值计算完成，耗时 {elapsed:.2f}秒 (Ground truth computed in {elapsed:.2f}s)")
        
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
        
        计算系统在给定参数下的召回率、查询时间等性能指标。
        召回率 = 找到的真实邻居数 / 应该找到的邻居数
        
        Args:
            kmeans_hnsw: 要评估的K-Means HNSW系统 (The K-Means HNSW system to evaluate)
            k: 返回结果数量 (Number of results to return)
            n_probe: 探测的聚类中心数量 (Number of centroids to probe)
            ground_truth: 预计算的真实值(可选) (Precomputed ground truth, optional)
            exclude_query_ids: 是否从评估中排除查询ID (Whether to exclude query IDs from evaluation)
            
        Returns:
            包含召回率指标和性能数据的字典 (Dictionary containing recall metrics and performance data)
        """
        if ground_truth is None:
            ground_truth = self.compute_ground_truth(k, exclude_query_ids)
        
        print(f"正在评估 {len(self.query_set)} 个查询的召回率 (k={k}, n_probe={n_probe})... (Evaluating recall)")
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
            
            if (i + 1) % 20 == 0:
                current_recall = total_correct / ((i + 1) * k)
                print(f"  Processed {i + 1}/{len(self.query_set)} queries, "
                      f"current recall: {current_recall:.4f}")
        
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

    # -------------------- Phase-Specific Evaluations --------------------
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
        for query_vector, query_id in zip(self.query_set, self.query_ids):
            true_neighbors = {nid for _, nid in ground_truth[query_id]}
            t0 = time.time()
            results = base_index.query(query_vector, k=k, ef=ef)
            dt = time.time() - t0
            query_times.append(dt)
            found = {nid for nid, _ in results}
            total_correct += len(true_neighbors & found)
        return {
            'phase': 'baseline_hnsw',
            'ef': ef,
            'recall_at_k': total_correct / total_expected if total_expected else 0.0,
            'avg_query_time_ms': float(np.mean(query_times) * 1000),
            'total_correct': total_correct,
            'total_expected': total_expected
        }

    def evaluate_clusters_only(
        self,
        kmeans_model: Any,
        dataset: np.ndarray,
        k: int,
        ground_truth: Dict,
        base_index: Optional[HNSW] = None
    ) -> Dict[str, Any]:
        """
        评估仅使用K-Means聚类的搜索性能 (Evaluate recall using ONLY KMeans clusters)
        
        策略：对每个查询找到最近的聚类中心，在该聚类的成员中按L2距离选择top-k
        
        Args:
            kmeans_model: 已训练的K-Means模型
            dataset: 聚类时使用的数据集
            k: 返回的近邻数量
            ground_truth: 真实值
            base_index: 基础HNSW索引，用于获取原始ID映射
        """
        if not hasattr(kmeans_model, 'cluster_centers_') or not hasattr(kmeans_model, 'labels_'):
            raise ValueError("KMeans model must be fitted with cluster_centers_ and labels_ available")
        
        centers = kmeans_model.cluster_centers_
        labels = kmeans_model.labels_
        n_clusters = centers.shape[0]
        
        # 构建聚类到索引的逆向映射 (Build inverse index: cluster -> indices)
        clusters = [[] for _ in range(n_clusters)]
        
        # 如果提供了base_index，建立dataset索引到原始ID的映射
        if base_index is not None:
            dataset_idx_to_original_id = list(base_index.keys())
            for dataset_idx, cluster_id in enumerate(labels):
                original_id = dataset_idx_to_original_id[dataset_idx]
                clusters[cluster_id].append((dataset_idx, original_id))
        else:
            # 假设dataset索引就是原始ID
            for dataset_idx, cluster_id in enumerate(labels):
                clusters[cluster_id].append((dataset_idx, dataset_idx))
        
        query_times = []
        total_correct = 0
        total_expected = len(self.query_set) * k
        individual_recalls = []
        
        for qvec, qid in zip(self.query_set, self.query_ids):
            t0 = time.time()
            
            # 找到最近的聚类中心
            d2c = np.linalg.norm(centers - qvec, axis=1)
            cidx = int(np.argmin(d2c))
            cluster_members = clusters[cidx]
            
            if cluster_members:
                # 计算查询向量到聚类成员的距离
                distances_with_ids = []
                for dataset_idx, original_id in cluster_members:
                    if original_id != qid:  # 排除查询本身
                        member_vec = dataset[dataset_idx]
                        dist = np.linalg.norm(member_vec - qvec)
                        distances_with_ids.append((dist, original_id))
                
                # 按距离排序并取top-k
                distances_with_ids.sort(key=lambda x: x[0])
                results = distances_with_ids[:k]
                found = {original_id for _, original_id in results}
            else:
                found = set()
            
            query_times.append(time.time() - t0)
            
            # 计算召回率
            true_neighbors = {nid for _, nid in ground_truth[qid]}
            correct = len(true_neighbors & found)
            total_correct += correct
            individual_recalls.append(correct / k if k else 0.0)
        
        overall = total_correct / total_expected if total_expected else 0.0
        return {
            'phase': 'clusters_only',
            'recall_at_k': overall,
            'avg_query_time_ms': float(np.mean(query_times) * 1000),
            'std_query_time_ms': float(np.std(query_times) * 1000),
            'total_correct': total_correct,
            'total_expected': total_expected,
            'avg_individual_recall': float(np.mean(individual_recalls)),
            'n_clusters': n_clusters
        }
    
    def parameter_sweep(
        self,
        base_index: HNSW,
        param_grid: Dict[str, List[Any]],
        evaluation_params: Dict[str, Any],
        max_combinations: Optional[int] = None,
        adaptive_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行全面的参数扫描优化 (Perform comprehensive parameter sweep for optimization)
        
        通过系统性地测试不同参数组合，找到最优的K-Means HNSW配置。
        包括基线HNSW、纯K-Means和K-Means HNSW的对比评估。
        
        Args:
            base_index: 使用的基础HNSW索引 (Base HNSW index to use)
            param_grid: 参数及其测试值的字典 (Dictionary of parameters and their values to test)
            evaluation_params: 评估参数 (Parameters for evaluation) - k值, n_probe值等
            max_combinations: 最大测试组合数 (Maximum number of combinations to test)
            adaptive_config: 自适应配置参数 (Adaptive configuration parameters, optional)
            
        Returns:
            每个参数组合的评估结果列表 (List of evaluation results for each parameter combination)
        """
        # 设置默认的自适应配置 (Set default adaptive configuration)
        if adaptive_config is None:
            adaptive_config = {
                'adaptive_k_children': False,
                'k_children_scale': 1.5,
                'k_children_min': 100,
                'k_children_max': None,
                'diversify_max_assignments': None,
                'repair_min_assignments': None
            }
        
        print("开始K-Means HNSW参数扫描... (Starting parameter sweep for K-Means HNSW)")
        
        # 生成所有参数组合 (Generate all parameter combinations)
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        if max_combinations and len(combinations) > max_combinations:
            print(f"限制测试 {max_combinations} 个组合，总共 {len(combinations)} 个 (Limiting to {max_combinations} combinations out of {len(combinations)})")
            combinations = random.sample(combinations, max_combinations)
        
        print(f"测试 {len(combinations)} 个参数组合... (Testing {len(combinations)} parameter combinations)")
        
        results = []
        k_values = evaluation_params.get('k_values', [10])
        n_probe_values = evaluation_params.get('n_probe_values', [5, 10, 20])
        
        # 预计算所有k值的真实值 (Precompute ground truth for all k values)
        ground_truths = {}
        for k in k_values:
            ground_truths[k] = self.compute_ground_truth(k)

        for i, combination in enumerate(combinations):
            print(f"\n--- Combination {i + 1}/{len(combinations)} ---")

            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            print(f"Parameters: {params}")

            try:
                phase_records = []
                
                # Phase 2: 构建完整的K-Means HNSW系统 (Build full K-Means HNSW system)
                # 先构建系统以获取实际使用的参数
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
                print(f"  构建K-Means HNSW系统耗时 {construction_time:.2f}秒 (Built KMeansHNSW system in {construction_time:.2f}s)")
                
                # 从构建的系统中提取实际使用的参数以确保一致性
                actual_n_clusters = kmeans_hnsw.n_clusters
                actual_child_search_ef = kmeans_hnsw.child_search_ef
                
                # Phase 1: 基线HNSW评估 - 使用与KMeansHNSW第一阶段相同的ef参数
                print(f"  使用与KMeansHNSW第一阶段相同的ef参数: {actual_child_search_ef}")
                for k in k_values:
                    b_eval = self.evaluate_hnsw_baseline(base_index, k, actual_child_search_ef, ground_truths[k])
                    phase_records.append({**b_eval, 'k': k})
                    print(f"  [基线HNSW/Baseline HNSW] k={k} ef={actual_child_search_ef} recall={b_eval['recall_at_k']:.4f} avg_time={b_eval['avg_query_time_ms']:.2f}ms")

                # Phase 3: K-Means聚类单独评估 - 使用与KMeansHNSW相同的聚类参数
                print(f"  使用与KMeansHNSW相同的聚类参数: n_clusters={actual_n_clusters}")
                kmeans_dataset = kmeans_hnsw._extract_dataset_vectors()
                for k in k_values:
                    c_eval = self.evaluate_clusters_only(
                        kmeans_hnsw.kmeans_model,
                        kmeans_dataset,  # 使用与聚类时相同的数据集
                        k,
                        ground_truths[k],
                        kmeans_hnsw.base_index  # 传递base_index以获取正确的索引映射
                    )
                    phase_records.append({**c_eval, 'k': k})
                    print(f"  [仅K-Means聚类/Clusters Only] k={k} recall={c_eval['recall_at_k']:.4f} avg_time={c_eval['avg_query_time_ms']:.2f}ms")

                # Phase 4: K-Means HNSW混合系统完整评估 (Full K-Means HNSW hybrid system evaluation)
                for k in k_values:
                    for n_probe in n_probe_values:
                        eval_result = self.evaluate_recall(kmeans_hnsw, k, n_probe, ground_truths[k])
                        phase_records.append({**eval_result, 'phase': 'kmeans_hnsw_hybrid'})
                        print(f"  [K-Means HNSW混合/Hybrid] k={k} n_probe={n_probe} recall={eval_result['recall_at_k']:.4f} avg_time={eval_result['avg_query_time_ms']:.2f}ms")

                # 收集此参数组合的所有评估结果 (Collect all evaluation results for this parameter combination)
                combination_results = {
                    'parameters': params,
                    'construction_time': construction_time,
                    'phase_evaluations': phase_records
                }
                results.append(combination_results)
                
                # 显示此组合的最佳召回率 (Show best recall for this combination)
                best_recall = max(r['recall_at_k'] for r in phase_records if 'recall_at_k' in r)
                print(f"  此组合最佳召回率 (Best recall): {best_recall:.4f}")
                
            except Exception as e:
                print(f"❌ 参数组合 {params} 出错: {e} (Error with combination)")
                continue
        
        print(f"\n🎯 参数扫描完成！测试了 {len(results)} 个组合 (Parameter sweep completed. Tested {len(results)} combinations)")
        return results
    
    def find_optimal_parameters(
        self,
        sweep_results: List[Dict[str, Any]],
        optimization_target: str = 'recall_at_k',
        constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal parameters from sweep results.
        
        Args:
            sweep_results: Results from parameter_sweep()
            optimization_target: Metric to optimize ('recall_at_k', 'avg_query_time_ms', etc.)
            constraints: Constraints on other metrics (e.g., {'avg_query_time_ms': 50.0})
            
        Returns:
            Dictionary containing optimal parameters and their performance
        """
        print(f"Finding optimal parameters optimizing for {optimization_target}...")
        
        best_result = None
        best_value = -float('inf') if 'recall' in optimization_target else float('inf')
        
        for result in sweep_results:
            for evaluation in result.get('phase_evaluations', []):
                # Check constraints
                if constraints:
                    violates_constraint = False
                    for constraint_metric, constraint_value in constraints.items():
                        if constraint_metric in evaluation:
                            if evaluation[constraint_metric] > constraint_value:
                                violates_constraint = True
                                break
                    if violates_constraint:
                        continue
                
                # Check if this is better
                current_value = evaluation.get(optimization_target)
                if current_value is None:
                    continue
                
                is_better = (
                    (current_value > best_value and 'recall' in optimization_target) or
                    (current_value < best_value and 'time' in optimization_target)
                )
                
                if is_better:
                    best_value = current_value
                    best_result = {
                        'parameters': result['parameters'],
                        'performance': evaluation,
                        'construction_time': result['construction_time']
                    }
        
        if best_result:
            print(f"Optimal parameters found:")
            print(f"  Parameters: {best_result['parameters']}")
            print(f"  {optimization_target}: {best_value:.4f}")
            print(f"  Construction time: {best_result['construction_time']:.2f}s")
        else:
            print("No valid parameters found satisfying constraints.")
        
        return best_result or {}
    
    def compare_with_baselines(
        self,
        kmeans_hnsw: KMeansHNSW,
        base_index: HNSW,
        k: int = 10,
        n_probe: int = 10,
        ef_values: List[int] = None
    ) -> Dict[str, Any]:
        """Compare K-Means HNSW performance with baseline HNSW and pure K-means.

        确保参数一致性：
        - HNSW基线使用与KMeansHNSW第一阶段相同的ef参数
        - 纯K-Means使用与KMeansHNSW相同的聚类参数
        """
        # 从KMeansHNSW系统获取实际使用的参数
        actual_n_clusters = kmeans_hnsw.n_clusters
        actual_child_search_ef = kmeans_hnsw.child_search_ef
        
        if ef_values is None:
            # 使用与KMeansHNSW第一阶段相同的ef参数，以及一些对比值
            ef_values = [actual_child_search_ef, 50, 100, 200, 400]
            # 去重并排序
            ef_values = sorted(set(ef_values))

        print(f"Comparing K-Means HNSW with baseline HNSW and pure K-means...")
        print(f"KMeansHNSW参数: n_clusters={actual_n_clusters}, child_search_ef={actual_child_search_ef}")

        ground_truth = self.compute_ground_truth(k)

        # K-Means HNSW two-stage
        kmeans_result = self.evaluate_recall(kmeans_hnsw, k, n_probe, ground_truth)

        # Pure k-means - 直接使用KMeansHNSW内部的聚类结果，避免重复聚类
        print(f"Evaluating pure K-means clustering using EXISTING KMeansHNSW clustering (n_clusters={actual_n_clusters}, n_probe={n_probe})...")
        kmeans_clustering_result = self._evaluate_pure_kmeans_from_existing(
            kmeans_hnsw,  # 直接传递KMeansHNSW对象
            k, 
            ground_truth,
            n_probe=n_probe
        )

        # Baseline HNSW - 直接使用KMeansHNSW内部的base_index，避免重复查询
        baseline_results = []
        for ef in ef_values:
            print(f"Evaluating baseline HNSW with ef={ef}{'[SAME as KMeansHNSW]' if ef == actual_child_search_ef else ''}...")
            # 使用现有的evaluate_hnsw_baseline方法避免代码重复
            baseline_result = self.evaluate_hnsw_baseline(kmeans_hnsw.base_index, k, ef, ground_truth)
            # 转换结果格式以匹配预期的输出格式
            baseline_results.append({
                'method': 'baseline_hnsw',
                'ef': ef,
                'is_matching_kmeans_ef': (ef == actual_child_search_ef),
                'recall_at_k': baseline_result['recall_at_k'],
                'avg_query_time_ms': baseline_result['avg_query_time_ms'],
                'total_correct': baseline_result['total_correct'],
                'total_expected': baseline_result['total_expected']
            })

        return {
            'kmeans_hnsw': kmeans_result,
            'pure_kmeans': kmeans_clustering_result,
            'baseline_hnsw': baseline_results,
            'parameter_consistency': {
                'kmeans_n_clusters': actual_n_clusters,
                'kmeans_child_search_ef': actual_child_search_ef,
                'pure_kmeans_n_clusters': kmeans_clustering_result['n_clusters'],
                'baseline_ef_matching_kmeans': actual_child_search_ef
            },
            'comparison_summary': {
                'kmeans_hnsw_recall': kmeans_result['recall_at_k'],
                'kmeans_hnsw_time_ms': kmeans_result['avg_query_time_ms'],
                'pure_kmeans_recall': kmeans_clustering_result['recall_at_k'],
                'pure_kmeans_time_ms': kmeans_clustering_result['avg_query_time_ms'],
                'best_baseline_recall': max(r['recall_at_k'] for r in baseline_results) if baseline_results else 0.0,
                'best_baseline_time_ms': min(r['avg_query_time_ms'] for r in baseline_results) if baseline_results else 0.0,
                'matching_ef_baseline_recall': next((r['recall_at_k'] for r in baseline_results if r['is_matching_kmeans_ef']), 0.0),
                'matching_ef_baseline_time_ms': next((r['avg_query_time_ms'] for r in baseline_results if r['is_matching_kmeans_ef']), 0.0)
            }
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
        
        Evaluate pure K-Means using existing clustering results from KMeansHNSW.
        Avoids redundant clustering by reusing trained model and data mappings.
        """
        print(f"Reusing existing clustering from KMeansHNSW (n_clusters={kmeans_hnsw.n_clusters}, n_probe={n_probe})...")
        
        # 直接使用KMeansHNSW内部的聚类结果
        kmeans_model = kmeans_hnsw.kmeans_model
        centers = kmeans_model.cluster_centers_
        n_clusters = centers.shape[0]
        
        # 获取与聚类时相同的数据集和索引映射
        kmeans_dataset = kmeans_hnsw._extract_dataset_vectors()
        labels = kmeans_model.labels_
        
        # 构建聚类到成员的映射 (与evaluate_clusters_only类似的逻辑)
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
            此实现通过先读取int32头部来避免解析错误。
            
            Read .fvecs file (FAISS format). Each vector stored as: int32 dim + dim float32.
            This implementation avoids mis-parsing by reading int32 header first.
            """
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            raw = np.fromfile(path, dtype=np.int32)
            if raw.size == 0:
                raise ValueError(f"空的fvecs文件: {path} (Empty fvecs file)")
            dim = raw[0]
            if dim <= 0 or dim > 4096:
                raise ValueError(f"不合理的向量维度 {dim}，解析自 {path} (Unreasonable vector dimension)")
            record_size = dim + 1
            count = raw.size // record_size
            raw = raw.reshape(count, record_size)
            vecs = raw[:, 1:].astype(np.float32)
            if max_vectors is not None and count > max_vectors:
                vecs = vecs[:max_vectors]
            return vecs

        base_path = os.path.join(sift_dir, 'sift_base.fvecs')
        query_path = os.path.join(sift_dir, 'sift_query.fvecs')

        # 为调优演示限制数量以保持合理的运行时间 (Limit for tuning demo to keep runtime reasonable)
        base_vectors = read_fvecs(base_path, max_vectors=50000)
        query_vectors = read_fvecs(query_path, max_vectors=1000)

        print(f"已加载SIFT数据: {base_vectors.shape[0]} 个基础向量, "
              f"{query_vectors.shape[0]} 个查询向量, 维度 {base_vectors.shape[1]}")
        print(f"Loaded SIFT data: {base_vectors.shape[0]} base vectors, "
              f"{query_vectors.shape[0]} query vectors, dimension {base_vectors.shape[1]}")

        return base_vectors, query_vectors
    
    except Exception as e:
        print(f"加载SIFT数据时出错: {e} (Error loading SIFT data)")
        print("改用合成数据... (Using synthetic data instead)")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Means HNSW参数调优和评估 (K-Means HNSW Parameter Tuning and Evaluation)")
    
    # 数据集选项 (Dataset options)
    parser.add_argument('--dataset-size', type=int, default=10000, 
                        help='使用的基础向量数量 (默认: 10000) (Number of base vectors to use)')
    parser.add_argument('--query-size', type=int, default=50, 
                        help='使用的查询向量数量 (默认: 50) (Number of query vectors to use)')
    parser.add_argument('--dimension', type=int, default=128, 
                        help='合成数据的向量维度 (如果未加载SIFT) (Vector dimensionality for synthetic data)')
    parser.add_argument('--no-sift', action='store_true', 
                        help='强制使用合成数据，即使SIFT文件存在 (Force synthetic data even if SIFT files exist)')
    
    # 自适应/多样化/修复选项 (Adaptive/diversification/repair options)
    parser.add_argument('--adaptive-k-children', action='store_true', 
                        help='启用基于平均聚类大小的自适应k_children (Enable adaptive k_children based on avg cluster size)')
    parser.add_argument('--k-children-scale', type=float, default=1.5, 
                        help='自适应k_children的缩放因子 (默认1.5) (Scale factor for adaptive k_children)')
    parser.add_argument('--k-children-min', type=int, default=100, 
                        help='自适应时的最小k_children (Minimum k_children when adaptive)')
    parser.add_argument('--k-children-max', type=int, default=None, 
                        help='自适应时的最大k_children (可选) (Maximum k_children when adaptive)')
    parser.add_argument('--diversify-max-assignments', type=int, default=None, 
                        help='每个子节点的最大分配数 (启用多样化) (Max assignments per child - enable diversification)')
    parser.add_argument('--repair-min-assignments', type=int, default=None, 
                        help='构建修复期间每个子节点的最小分配数 (需要多样化) (Min assignments per child during build repair)')
    parser.add_argument('--manual-repair', action='store_true', 
                        help='在最优构建后运行手动修复 (Run manual repair after optimal build)')
    parser.add_argument('--manual-repair-min', type=int, default=None, 
                        help='手动修复的最小分配数 (Min assignments for manual repair)')
    args = parser.parse_args()

    print("🔬 K-Means HNSW参数调优和评估系统 (K-Means HNSW Parameter Tuning and Evaluation)")
    print(f"📊 请求的数据集大小: {args.dataset_size}, 查询大小: {args.query_size}")
    print(f"   Requested dataset size: {args.dataset_size}, query size: {args.query_size}")
    
    # 尝试加载SIFT数据，失败则使用合成数据 (Try to load SIFT data, fall back to synthetic unless disabled)
    base_vectors, query_vectors = (None, None)
    if not args.no_sift:
        base_vectors, query_vectors = load_sift_data()
    
    if base_vectors is None:
        # 创建合成数据 (Create synthetic data)
        print("🎲 创建合成数据集... (Creating synthetic dataset)")
        base_vectors = np.random.randn(max(args.dataset_size, 10000), args.dimension).astype(np.float32)
        query_vectors = np.random.randn(max(args.query_size, 100), args.dimension).astype(np.float32)
    
    # 切片到请求的大小 (按可用量限制) (Slice to requested sizes)
    if len(base_vectors) > args.dataset_size:
        base_vectors = base_vectors[:args.dataset_size]
    if len(query_vectors) > args.query_size:
        query_vectors = query_vectors[:args.query_size]
    print(f"📈 使用基础向量: {len(base_vectors)} | 查询: {len(query_vectors)} | 维度: {base_vectors.shape[1]}")
    print(f"   Using base vectors: {len(base_vectors)} | queries: {len(query_vectors)} | dim: {base_vectors.shape[1]}")
    query_ids = list(range(len(query_vectors)))
    
    # 距离函数 (Distance function)
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # 构建基础HNSW索引 (Build base HNSW index)
    print("🏗️  构建基础HNSW索引... (Building base HNSW index)")
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=100)
    
    for i, vector in enumerate(base_vectors):
        base_index.insert(i, vector)
        if (i + 1) % 1000 == 0:
            print(f"  Inserted {i + 1}/{len(base_vectors)} vectors")
    
    print(f"Base HNSW index built with {len(base_index)} vectors")
    
    # Initialize evaluator
    evaluator = KMeansHNSWEvaluator(base_vectors, query_vectors, query_ids, distance_func)
    
    # Define parameter grid for sweep
    # Adjust default cluster count heuristics for larger datasets: scale choices
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
        'n_probe_values': [5, 10, 20]
    }
    
    # Perform parameter sweep
    print("\nStarting parameter sweep...")
    # Limit combinations to keep runtime sane on large sets
    max_combos = 9 if len(cluster_options) > 1 else None
    
    # 准备自适应配置 (Prepare adaptive configuration)
    adaptive_config = {
        'adaptive_k_children': args.adaptive_k_children,
        'k_children_scale': args.k_children_scale,
        'k_children_min': args.k_children_min,
        'k_children_max': args.k_children_max,
        'diversify_max_assignments': args.diversify_max_assignments,
        'repair_min_assignments': args.repair_min_assignments
    }
    
    sweep_results = evaluator.parameter_sweep(
        base_index,
        param_grid,
        evaluation_params,
        max_combinations=max_combos,
        adaptive_config=adaptive_config
    )
    
    # Find optimal parameters
    optimal = evaluator.find_optimal_parameters(
        sweep_results,
        optimization_target='recall_at_k',
        constraints={'avg_query_time_ms': 100.0}  # Max 100ms per query
    )
    
    if optimal:
        # Build system with optimal parameters and compare with baseline
        print("\nBuilding system with optimal parameters...")
        optimal_kmeans_hnsw = KMeansHNSW(
            base_index=base_index,
            **optimal['parameters'],
            adaptive_k_children=args.adaptive_k_children,
            k_children_scale=args.k_children_scale,
            k_children_min=args.k_children_min,
            k_children_max=args.k_children_max,
            diversify_max_assignments=args.diversify_max_assignments,
            repair_min_assignments=args.repair_min_assignments
        )

        if args.manual_repair:
            manual_min = args.manual_repair_min or args.repair_min_assignments or 1
            print(f"\nManual repair step: ensuring each node has at least {manual_min} assignments...")
            repair_stats = optimal_kmeans_hnsw.run_repair(min_assignments=manual_min)
            print(f"Manual repair completed. Coverage={repair_stats['coverage_fraction']:.3f}")
        
        comparison = evaluator.compare_with_baselines(
            optimal_kmeans_hnsw,
            base_index,
            k=10,
            n_probe=10
        )
        
        print("\nComparison Results:")
        print(f"K-Means HNSW: Recall={comparison['comparison_summary']['kmeans_hnsw_recall']:.4f}, "
              f"Time={comparison['comparison_summary']['kmeans_hnsw_time_ms']:.2f}ms")
        print(f"Pure K-Means: Recall={comparison['comparison_summary']['pure_kmeans_recall']:.4f}, "
              f"Time={comparison['comparison_summary']['pure_kmeans_time_ms']:.2f}ms")
        print(f"Best Baseline: Recall={comparison['comparison_summary']['best_baseline_recall']:.4f}, "
              f"Time={comparison['comparison_summary']['best_baseline_time_ms']:.2f}ms")
        
        # 显示参数一致性信息
        param_consistency = comparison['parameter_consistency']
        print(f"\n📊 Parameter Consistency (参数一致性):")
        print(f"  KMeansHNSW n_clusters: {param_consistency['kmeans_n_clusters']}")
        print(f"  Pure K-Means n_clusters: {param_consistency['pure_kmeans_n_clusters']} {'✓' if param_consistency['kmeans_n_clusters'] == param_consistency['pure_kmeans_n_clusters'] else '✗'}")
        print(f"  KMeansHNSW child_search_ef: {param_consistency['kmeans_child_search_ef']}")
        print(f"  Baseline HNSW ef (matching): {param_consistency['baseline_ef_matching_kmeans']}")
        
        # 显示使用相同ef参数的HNSW基线性能
        matching_ef_recall = comparison['comparison_summary']['matching_ef_baseline_recall']
        matching_ef_time = comparison['comparison_summary']['matching_ef_baseline_time_ms']
        print(f"  Baseline HNSW (same ef): Recall={matching_ef_recall:.4f}, Time={matching_ef_time:.2f}ms")
        
        # Additional detailed output for pure K-means
        pure_kmeans_result = comparison['pure_kmeans']
        print(f"\nDetailed Pure K-Means Results:")
        print(f"  Overall Recall@{pure_kmeans_result['k']}: {pure_kmeans_result['recall_at_k']:.4f}")
        print(f"  Average Individual Recall: {pure_kmeans_result['avg_individual_recall']:.4f}")
        print(f"  Correct/Expected: {pure_kmeans_result['total_correct']}/{pure_kmeans_result['total_expected']}")
        print(f"  Clustering Time: {pure_kmeans_result['clustering_time']:.2f}s")
        print(f"  Average Query Time: {pure_kmeans_result['avg_query_time_ms']:.2f}ms")
        
        # Save results
        results = {
            'sweep_results': sweep_results,
            'optimal_parameters': optimal,
            'baseline_comparison': comparison,
            'adaptive_config': {
                'adaptive_k_children': args.adaptive_k_children,
                'k_children_scale': args.k_children_scale,
                'k_children_min': args.k_children_min,
                'k_children_max': args.k_children_max,
                'diversify_max_assignments': args.diversify_max_assignments,
                'repair_min_assignments': args.repair_min_assignments,
                'manual_repair': args.manual_repair,
                'manual_repair_min': args.manual_repair_min
            },
            'evaluation_info': {
                'dataset_size': len(base_vectors),
                'query_size': len(query_vectors),
                'dimension': base_vectors.shape[1],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        save_results(results, 'method3_tuning_results.json')
        
    print("\nParameter tuning completed!")
