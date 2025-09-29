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
import traceback
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from itertools import product

# 添加父目录到路径 (Add parent directory to path)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from method3.kmeans_hnsw import KMeansHNSW
from hnsw.hnsw import HNSW
# 引入 Hybrid HNSW 结构用于对比评估 (Import Hybrid HNSW for comparative evaluation)
from hybrid_hnsw.hnsw_hybrid import HNSWHybrid
# 使用sklearn MiniBatchKMeans作为纯k-means基线 (Switch to sklearn MiniBatchKMeans for pure k-means baseline)
from sklearn.cluster import MiniBatchKMeans


class KMeansHNSWMultiPivot:
    """Multi-Pivot K-Means HNSW系统 - 集成到v1.py中
    
    与KMeansHNSW共享相同的HNSW索引和K-Means聚类，仅在子节点查找策略上不同。
    使用多个枢纽点来丰富每个质心的子节点选择。
    """
    
    def __init__(
        self,
        base_index: HNSW,
        n_clusters: int = 100,
        k_children: int = 800,
        child_search_ef: Optional[int] = None,
        # Multi-pivot specific parameters
        num_pivots: int = 3,
        pivot_selection_strategy: str = 'line_perp_third',
        pivot_overquery_factor: float = 1.2,
        # Adaptive/repair options (same as KMeansHNSW)
        adaptive_k_children: bool = False,
        k_children_scale: float = 1.5,
        k_children_min: int = 100,
        k_children_max: Optional[int] = None,
        diversify_max_assignments: Optional[int] = None,
        repair_min_assignments: Optional[int] = None,
        # Shared K-Means support (新增共享支持)
        shared_kmeans_model: Optional[MiniBatchKMeans] = None,
        shared_dataset_vectors: Optional[np.ndarray] = None
    ):
        self.base_index = base_index
        self.n_clusters = n_clusters
        self.k_children = k_children
        self.distance_func = base_index._distance_func
        
        # Multi-pivot parameters
        self.num_pivots = max(1, num_pivots)
        self.pivot_selection_strategy = pivot_selection_strategy
        self.pivot_overquery_factor = max(1.0, pivot_overquery_factor)
        
        # Adaptive/repair options
        self.adaptive_k_children = adaptive_k_children
        self.k_children_scale = k_children_scale
        self.k_children_min = k_children_min
        self.k_children_max = k_children_max
        self.diversify_max_assignments = diversify_max_assignments
        self.repair_min_assignments = repair_min_assignments
        
        # Shared K-Means support (新增共享支持)
        self.shared_kmeans_model = shared_kmeans_model
        self.shared_dataset_vectors = shared_dataset_vectors
        
        # Child search ef
        if child_search_ef is None:
            min_ef = max(k_children + 50, int(k_children * 1.2))
            self.child_search_ef = min_ef
        else:
            self.child_search_ef = child_search_ef
        
        # Core components (will be built)
        self.kmeans_model = None
        self.centroids = None
        self.centroid_ids = []
        self.parent_child_map = {}
        self.child_vectors = {}
        self._centroid_matrix = None
        
        # Stats tracking
        self.stats = {
            'method': 'multi_pivot_kmeans_hnsw',
            'n_clusters': n_clusters,
            'k_children': k_children,
            'child_search_ef': self.child_search_ef,
            'num_pivots': self.num_pivots,
            'pivot_strategy': self.pivot_selection_strategy,
            'pivot_overquery_factor': self.pivot_overquery_factor,
            'shared_kmeans_used': shared_kmeans_model is not None,
            'shared_data_used': shared_dataset_vectors is not None
        }
        self.search_times = []
        
        # Build the system
        self._build_system()
    
    def _build_system(self):
        """构建多枢纽K-Means HNSW系统"""
        shared_info = ""
        if self.shared_kmeans_model is not None:
            shared_info += " (共享K-Means模型)"
        if self.shared_dataset_vectors is not None:
            shared_info += " (共享数据向量)"
            
        print(f"Building Multi-Pivot K-Means HNSW system with {self.n_clusters} clusters, {self.num_pivots} pivots{shared_info}...")
        
        # Step 1: Extract vectors from HNSW index
        self._extract_dataset_vectors()
        
        # Step 2: Perform K-Means clustering
        self._perform_kmeans_clustering()
        
        # Step 3: Assign children using multi-pivot strategy
        self._assign_children_via_multi_pivot()
        
        # Step 4: Build centroid index for fast search
        self._build_centroid_index()
        
        print(f"Multi-Pivot K-Means HNSW system built with {len(self.parent_child_map)} centroids")
    
    def _extract_dataset_vectors(self):
        """从HNSW索引提取向量数据 (支持共享数据向量)"""
        if self.shared_dataset_vectors is not None:
            print("  Using shared dataset vectors...")
            self.dataset_vectors = self.shared_dataset_vectors
            return
        
        dataset_vectors = []
        for node_id, node in self.base_index._nodes.items():
            vector = node.point
            if vector is not None:
                dataset_vectors.append(vector)
        self.dataset_vectors = np.array(dataset_vectors)
    
    def _perform_kmeans_clustering(self):
        """执行K-Means聚类 (支持共享模型)"""
        # Multi-Pivot必须使用共享的K-Means模型 (Multi-Pivot must use shared K-Means model)
        if self.shared_kmeans_model is None:
            raise ValueError("Multi-Pivot KMeans HNSW requires a shared_kmeans_model. "
                           "Please provide a pre-trained MiniBatchKMeans model.")
        
        print("  Using shared MiniBatchKMeans model...")
        self.kmeans_model = self.shared_kmeans_model
        self.centroids = self.kmeans_model.cluster_centers_
        self.n_clusters = self.centroids.shape[0]
        self.centroid_ids = [f"centroid_{i}" for i in range(self.n_clusters)]
        
        # 自适应调整k_children (基于共享模型的聚类数量)
        if self.adaptive_k_children:
            avg_cluster_size = len(self.dataset_vectors) / self.n_clusters
            adaptive_k = int(avg_cluster_size * self.k_children_scale)
            adaptive_k = max(self.k_children_min, adaptive_k)
            if self.k_children_max:
                adaptive_k = min(self.k_children_max, adaptive_k)
            self.k_children = adaptive_k
            print(f"  自适应调整k_children: {self.k_children} (平均聚类大小: {avg_cluster_size:.1f})")
        
        print(f"Shared K-Means clustering loaded with {self.n_clusters} clusters")
    
    def _assign_children_via_multi_pivot(self):
        """使用多枢纽策略分配子节点"""
        print(f"Assigning children via multi-pivot HNSW (pivots={self.num_pivots}, ef={self.child_search_ef})...")
        
        if self.num_pivots == 1:
            # 退化到单枢纽
            self._assign_children_single_pivot()
            return
        
        k_each_query = max(self.k_children, int(self.k_children * self.pivot_overquery_factor))
        assignment_counts = {} if self.repair_min_assignments else None
        
        for idx_c, centroid_id in enumerate(self.centroid_ids):
            centroid_vector = self.centroids[idx_c]
            
            # 第一个枢纽：质心本身
            pivots = [centroid_vector]
            
            # 第一次查询：从质心开始
            neighbors_A = self.base_index.query(centroid_vector, k=k_each_query, ef=self.child_search_ef)
            S_A = [node_id for node_id, _ in neighbors_A]
            all_candidates = set(S_A)
            
            # 存储向量用于后续计算
            for node_id in S_A:
                if node_id not in self.child_vectors:
                    if node_id in self.base_index._nodes:
                        self.child_vectors[node_id] = self.base_index._nodes[node_id].point
            
            if not S_A:
                self.parent_child_map[centroid_id] = []
                continue
            
            # 添加更多枢纽
            for pivot_idx in range(1, self.num_pivots):
                if pivot_idx == 1:
                    # 第二个枢纽：距离质心最远的点
                    farthest_node, farthest_vec = self._find_farthest_from_centroid(centroid_vector, S_A)
                    if farthest_vec is not None:
                        pivots.append(farthest_vec)
                elif pivot_idx == 2 and self.pivot_selection_strategy == 'line_perp_third':
                    # 第三个枢纽：垂直距离最大的点
                    perp_vec = self._find_perpendicular_pivot(pivots[0], pivots[1], all_candidates)
                    if perp_vec is not None:
                        pivots.append(perp_vec)
                else:
                    # 后续枢纽：最大最小距离策略
                    max_min_vec = self._find_max_min_distance_pivot(pivots, all_candidates)
                    if max_min_vec is not None:
                        pivots.append(max_min_vec)
                    else:
                        break
                
                # 从新枢纽查询更多候选
                if len(pivots) > pivot_idx:
                    new_neighbors = self.base_index.query(pivots[-1], k=k_each_query, ef=self.child_search_ef)
                    for node_id, _ in new_neighbors:
                        all_candidates.add(node_id)
                        if node_id not in self.child_vectors:
                            if node_id in self.base_index._nodes:
                                self.child_vectors[node_id] = self.base_index._nodes[node_id].point
            
            # 从所有候选中选择最佳的k_children个
            selected_children = self._select_best_children_from_candidates(
                list(all_candidates), pivots, self.k_children
            )
            
            # 应用多样化过滤
            if self.diversify_max_assignments and assignment_counts is not None:
                selected_children = self._apply_diversify_filter(
                    selected_children, assignment_counts, self.diversify_max_assignments
                )
            
            self.parent_child_map[centroid_id] = selected_children
            
            # 更新分配计数
            if assignment_counts is not None:
                for child_id in selected_children:
                    assignment_counts[child_id] = assignment_counts.get(child_id, 0) + 1
            
            if (idx_c + 1) % 10 == 0:
                print(f"  Processed {idx_c + 1}/{self.n_clusters} centroids")
        
        # 修复分配
        if self.repair_min_assignments and assignment_counts is not None:
            self._repair_child_assignments(assignment_counts)
        
        # 更新统计信息
        total_children = sum(len(children) for children in self.parent_child_map.values())
        self.stats['num_children'] = len(self.child_vectors)
        self.stats['avg_children_per_centroid'] = total_children / max(1, self.n_clusters)
        print(f"Multi-pivot assignment completed: {total_children} assignments, {len(self.child_vectors)} unique children")
    
    def _assign_children_single_pivot(self):
        """单枢纽分配 (退化情况)"""
        print("Using single-pivot assignment (fallback)...")
        assignment_counts = {} if self.repair_min_assignments else None
        
        for idx_c, centroid_id in enumerate(self.centroid_ids):
            centroid_vector = self.centroids[idx_c]
            results = self.base_index.query(centroid_vector, k=self.k_children, ef=self.child_search_ef)
            children = [node_id for node_id, _ in results]
            
            # 应用多样化过滤
            if self.diversify_max_assignments and assignment_counts is not None:
                children = self._apply_diversify_filter(
                    children, assignment_counts, self.diversify_max_assignments
                )
            
            self.parent_child_map[centroid_id] = children
            
            # 存储子节点向量
            for child_id in children:
                if child_id not in self.child_vectors:
                    self.child_vectors[child_id] = self.base_index[child_id]
            
            # 更新分配计数
            if assignment_counts is not None:
                for child_id in children:
                    assignment_counts[child_id] = assignment_counts.get(child_id, 0) + 1
        
        # 修复分配
        if self.repair_min_assignments and assignment_counts is not None:
            self._repair_child_assignments(assignment_counts)
    
    def _find_farthest_from_centroid(self, centroid_vector, candidates):
        """找到距离质心最远的候选点"""
        max_distance = -1
        farthest_node = None
        farthest_vec = None
        
        for node_id in candidates:
            if node_id in self.base_index._nodes:
                node_vector = self.base_index._nodes[node_id].point
            else:
                continue
            distance = self.distance_func(centroid_vector, node_vector)
            if distance > max_distance:
                max_distance = distance
                farthest_node = node_id
                farthest_vec = node_vector
        
        return farthest_node, farthest_vec
    
    def _find_perpendicular_pivot(self, pivot_a, pivot_b, candidates):
        """找到垂直于A-B线段距离最大的点"""
        ab_vector = pivot_b - pivot_a
        ab_norm_sq = np.dot(ab_vector, ab_vector)
        
        if ab_norm_sq < 1e-12:
            return None
        
        max_perp_distance = -1
        best_vector = None
        
        for node_id in candidates:
            node_vector = self.child_vectors.get(node_id)
            if node_vector is None:
                node_vector = self.base_index._nodes[node_id].point
            
            # 计算垂直距离
            ac_vector = node_vector - pivot_a
            projection_coeff = np.dot(ac_vector, ab_vector) / ab_norm_sq
            projection = projection_coeff * ab_vector
            perpendicular = ac_vector - projection
            perp_distance = np.linalg.norm(perpendicular)
            
            if perp_distance > max_perp_distance:
                max_perp_distance = perp_distance
                best_vector = node_vector
        
        return best_vector
    
    def _find_max_min_distance_pivot(self, existing_pivots, candidates):
        """找到与现有枢纽最小距离最大的候选点"""
        best_score = -1
        best_vector = None
        
        for node_id in candidates:
            node_vector = self.child_vectors.get(node_id)
            if node_vector is None:
                node_vector = self.base_index._nodes[node_id].point
            
            # 计算到所有现有枢纽的最小距离
            min_distance = min(
                self.distance_func(node_vector, pivot) for pivot in existing_pivots
            )
            
            if min_distance > best_score:
                best_score = min_distance
                best_vector = node_vector
        
        return best_vector
    
    def _select_best_children_from_candidates(self, candidate_ids, pivots, k_children):
        """从候选节点中选择最优的k_children个子节点"""
        if len(candidate_ids) <= k_children:
            return candidate_ids
        
        # 计算每个候选点到最近枢纽的距离
        scores = []
        for node_id in candidate_ids:
            node_vector = self.child_vectors.get(node_id)
            if node_vector is None:
                node_vector = self.base_index._nodes[node_id].point
            
            min_distance = min(
                self.distance_func(node_vector, pivot) for pivot in pivots
            )
            scores.append((min_distance, node_id))
        
        # 选择距离最小的k_children个
        scores.sort()
        return [node_id for _, node_id in scores[:k_children]]
    
    def _apply_diversify_filter(self, children, assignment_counts, max_assignments):
        """应用多样化过滤器"""
        filtered_children = []
        for child_id in children:
            current_count = assignment_counts.get(child_id, 0)
            if current_count < max_assignments:
                filtered_children.append(child_id)
        return filtered_children
    
    def _repair_child_assignments(self, assignment_counts):
        """修复子节点分配以确保最小覆盖"""
        print(f"Multi-Pivot Repair phase: ensuring minimum {self.repair_min_assignments} assignments...")
        
        all_base_nodes = set(self.base_index.keys())
        assigned_nodes = set(assignment_counts.keys())
        unassigned_nodes = all_base_nodes - assigned_nodes
        
        under_assigned = {
            node_id for node_id, count in assignment_counts.items()
            if count < self.repair_min_assignments
        }
        under_assigned.update(unassigned_nodes)
        
        print(f"  Found {len(under_assigned)} under-assigned nodes ({len(unassigned_nodes)} completely unassigned)")
        
        for node_id in under_assigned:
            try:
                node_vector = self.base_index._nodes[node_id].point
                
                # 找到最近的质心并分配
                distances = []
                for i, centroid_vector in enumerate(self.centroids):
                    distance = self.distance_func(node_vector, centroid_vector)
                    distances.append((distance, self.centroid_ids[i]))
                
                distances.sort()
                current_assignments = assignment_counts.get(node_id, 0)
                needed_assignments = max(0, self.repair_min_assignments - current_assignments)
                
                for _, centroid_id in distances[:needed_assignments]:
                    if node_id not in self.parent_child_map[centroid_id]:
                        self.parent_child_map[centroid_id].append(node_id)
                        self.child_vectors[node_id] = node_vector
                        assignment_counts[node_id] = assignment_counts.get(node_id, 0) + 1
            except Exception as e:
                print(f"    ⚠️ Failed to repair node {node_id}: {e}")
                continue
        
        final_assigned = set(assignment_counts.keys())
        coverage = len(final_assigned) / len(all_base_nodes) if all_base_nodes else 0.0
        self.stats['coverage_fraction'] = coverage
        print(f"  Multi-Pivot repair completed. Final coverage: {coverage:.3f} ({len(final_assigned)}/{len(all_base_nodes)} nodes)")
    
    def _build_centroid_index(self):
        """构建质心索引用于快速搜索"""
        if self.centroids is None:
            raise ValueError("Centroids not computed yet")
        self._centroid_matrix = self.centroids.copy()
    
    def search(self, query_vector, k=10, n_probe=10):
        """两阶段搜索：质心搜索 → 子节点搜索"""
        start_time = time.time()
        
        # Stage 1: 找到最近的质心
        closest_centroids = self._stage1_centroid_search(query_vector, n_probe)
        
        # Stage 2: 在选定质心的子节点中搜索
        results = self._stage2_child_search(query_vector, closest_centroids, k)
        
        # 记录搜索时间
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)
        
        return results
    
    def _stage1_centroid_search(self, query_vector, n_probe):
        """Stage 1: 找到最近的K-Means质心"""
        distances = np.linalg.norm(self._centroid_matrix - query_vector, axis=1)
        closest_indices = np.argsort(distances)[:n_probe]
        return [(self.centroid_ids[i], distances[i]) for i in closest_indices]
    
    def _stage2_child_search(self, query_vector, closest_centroids, k):
        """Stage 2: 在子节点中搜索"""
        # 收集候选子节点
        candidate_children = set()
        for centroid_id, _ in closest_centroids:
            children = self.parent_child_map.get(centroid_id, [])
            candidate_children.update(children)
        
        if not candidate_children:
            return []
        
        # 计算距离并排序
        candidate_scores = []
        for child_id in candidate_children:
            child_vector = self.child_vectors.get(child_id)
            if child_vector is not None:
                distance = self.distance_func(query_vector, child_vector)
                candidate_scores.append((distance, child_id))
        
        # 排序并返回格式为 (child_id, distance) 的结果
        candidate_scores.sort()
        return [(child_id, distance) for distance, child_id in candidate_scores[:k]]
    
    def get_stats(self):
        """获取统计信息"""
        stats = self.stats.copy()
        if self.search_times:
            stats['avg_search_time_ms'] = float(np.mean(self.search_times))
            stats['std_search_time_ms'] = float(np.std(self.search_times))
        return stats


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
        
        注意：在当前实现中，查询向量和数据集向量是独立生成的，因此
        exclude_query_ids 参数通常应设为 False，除非查询向量是从数据集中采样的。
        
        Args:
            k: 最近邻数量 (Number of nearest neighbors)
            exclude_query_ids: 是否从结果中排除查询ID (Whether to exclude query IDs from results)
                              仅当查询向量是数据集子集时才有意义 (Only meaningful when queries are subset of dataset)
            
        Returns:
            字典：查询ID -> (距离, 数据索引)元组列表 (Dictionary mapping query_id to list of (distance, data_index) tuples)
        """
        cache_key = (k, exclude_query_ids)
        if cache_key in self._ground_truth_cache:
            return self._ground_truth_cache[cache_key]
        
        print(f"正在计算 {len(self.query_set)} 个查询的真实值 (k={k}, exclude_query_ids={exclude_query_ids})...")
        print(f"Computing ground truth for {len(self.query_set)} queries against {len(self.dataset)} data points")
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
            
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(self.query_set)} 个查询 (Processed {i + 1}/{len(self.query_set)} queries)")
        
        elapsed = time.time() - start_time
        if exclude_query_ids and excluded_count == 0:
            print(f"⚠️  警告：exclude_query_ids=True但没有排除任何数据点。查询向量可能不在数据集中。")
            print(f"   Warning: exclude_query_ids=True but no data points were excluded. Query vectors may not be in dataset.")
        
        print(f"真实值计算完成，耗时 {elapsed:.2f}秒，排除了 {excluded_count} 个数据点")
        print(f"Ground truth computed in {elapsed:.2f}s, excluded {excluded_count} data points")
        
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
            'system_stats': multi_pivot_hnsw.get_stats()
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
        individual_recalls = []
        
        print(f"🔍 评估HNSW基线性能 (k={k}, ef={ef})...")
        
        for query_vector, query_id in zip(self.query_set, self.query_ids):
            # Ground truth format: {query_id: [(distance, node_id), ...]}
            # Extract the node_ids (which are data indices) from ground truth
            true_neighbors = {node_id for _, node_id in ground_truth[query_id]}
            
            t0 = time.time()
            results = base_index.query(query_vector, k=k, ef=ef)
            dt = time.time() - t0
            query_times.append(dt)
            
            # HNSW query returns [(node_id, distance), ...]
            found = {nid for nid, _ in results}
            correct = len(true_neighbors & found)
            total_correct += correct
            
            # Individual recall for this query
            individual_recall = correct / k if k > 0 else 0.0
            individual_recalls.append(individual_recall)
            
            # Debug info for first few queries
            if query_id < 3:
                print(f"  Query {query_id}: found {len(found)} results, {correct}/{k} correct, recall={individual_recall:.4f}")
                print(f"    True neighbors (first 5): {list(true_neighbors)[:5]}")
                print(f"    Found neighbors (first 5): {list(found)[:5]}")
        
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
                'repair_min_assignments': 1
            }

        if multi_pivot_config is None:
            multi_pivot_config = {
                'enabled': False,
                'num_pivots': 3,
                'pivot_selection_strategy': 'line_perp_third',
                'pivot_overquery_factor': 1.2
            }

        print("🔬================== 五方法对比评估系统 ==================")
        print("📊 评估流程: HNSW → K-Means → Hybrid HNSW → KMeans HNSW → Multi-Pivot KMeans HNSW")
        print(f"🎯 Multi-Pivot启用状态: {multi_pivot_config.get('enabled', False)}")
        print("================================================================")

        # ========== 步骤1: 准备参数组合 ==========
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        if max_combinations and len(combinations) > max_combinations:
            print(f"限制测试 {max_combinations} 个组合，总共 {len(combinations)} 个")
            combinations = random.sample(combinations, max_combinations)
        print(f"📋 将测试 {len(combinations)} 个参数组合")

        results: List[Dict[str, Any]] = []
        k_values = evaluation_params.get('k_values', [10])
        n_probe_values = evaluation_params.get('n_probe_values', [5, 10, 20])
        hybrid_parent_level = evaluation_params.get('hybrid_parent_level', 2)
        enable_hybrid = evaluation_params.get('enable_hybrid', True)

        # ========== 步骤2: 预计算真实值 (Ground Truth) ==========
        print(f"\n🎯 步骤2: 预计算真实值 (k_values: {k_values})")
        ground_truths: Dict[int, Dict] = {}
        for k in k_values:
            ground_truths[k] = self.compute_ground_truth(k, exclude_query_ids=False)
        
        # ========== 步骤3: 预训练共享K-Means模型 ==========
        print(f"\n🤖 步骤3: 预训练共享K-Means模型以避免重复计算")
        shared_dataset_vectors = []
        for node_id, node in base_index._nodes.items():
            if node.point is not None:
                shared_dataset_vectors.append(node.point)
        shared_dataset_vectors = np.array(shared_dataset_vectors)
        print(f"   提取了 {len(shared_dataset_vectors)} 个数据向量")
        
        # 为每个n_clusters值预训练K-Means模型
        shared_kmeans_models: Dict[int, MiniBatchKMeans] = {}
        unique_n_clusters = set(params[param_names.index('n_clusters')] for params in combinations)
        
        for n_clusters in unique_n_clusters:
            print(f"   预训练K-Means模型 (n_clusters={n_clusters})...")
            actual_clusters = min(n_clusters, len(shared_dataset_vectors))
            kmeans_model = MiniBatchKMeans(
                n_clusters=actual_clusters,
                random_state=42,
                max_iter=100,
                batch_size=min(100, len(shared_dataset_vectors))
            )
            kmeans_model.fit(shared_dataset_vectors)
            shared_kmeans_models[n_clusters] = kmeans_model
            print(f"     ✅ 完成: {actual_clusters} clusters, inertia={kmeans_model.inertia_:.2f}")
        
        print(f"✅ 共享K-Means模型预训练完成 ({len(shared_kmeans_models)} 个模型)")
        print(f"💡 K-Means模型将被所有方法重用，确保公平对比")

        # ========== 步骤4: 开始参数组合评估 ==========
        for i, combination in enumerate(combinations):
            print(f"\n🔬 =========== 参数组合 {i + 1}/{len(combinations)} ===========")
            params = dict(zip(param_names, combination))
            print(f"📝 当前参数: {params}")

            try:
                phase_records: List[Dict[str, Any]] = []
                
                # 获取当前组合的共享K-Means模型
                current_n_clusters = params['n_clusters']
                shared_model = shared_kmeans_models[current_n_clusters]
                print(f"🤖 使用预训练的K-Means模型 (n_clusters={current_n_clusters})")
                
                # ========== 方法1: HNSW基线 ==========
                print(f"\n📊 方法1: HNSW基线评估")
                base_ef = base_index._ef_construction
                print(f"   参数: ef={base_ef}")
                for k in k_values:
                    b_eval = self.evaluate_hnsw_baseline(base_index, k, base_ef, ground_truths[k])
                    phase_records.append({**b_eval, 'k': k})
                    print(f"   ✅ k={k}: recall={b_eval['recall_at_k']:.4f}, 时间={b_eval['avg_query_time_ms']:.2f}ms")

                # ========== 方法2: 纯K-Means聚类 ==========
                print(f"\n📊 方法2: 纯K-Means聚类评估")
                print(f"   参数: n_clusters={current_n_clusters}, n_probe={n_probe_values}")
                for k in k_values:
                    for n_probe in n_probe_values:
                        c_eval = self._evaluate_pure_kmeans_from_existing_shared(
                            shared_model, shared_dataset_vectors, base_index,
                            k, ground_truths[k], n_probe=n_probe
                        )
                        c_eval['phase'] = 'clusters_only'
                        phase_records.append({**c_eval, 'k': k})
                        print(f"   ✅ k={k} n_probe={n_probe}: recall={c_eval['recall_at_k']:.4f}, 时间={c_eval['avg_query_time_ms']:.2f}ms")

                # ========== 方法3: Hybrid HNSW ==========
                if enable_hybrid:
                    print(f"\n📊 方法3: Hybrid HNSW评估")
                    print(f"   参数: parent_level={hybrid_parent_level}, k_children={params['k_children']}")
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
                        
                        print(f"   构建完成: {hybrid_stats.get('num_parents', 0)} parents, "
                              f"{hybrid_stats.get('num_children', 0)} children, "
                              f"coverage: {hybrid_stats.get('coverage_fraction', 0):.4f}")
                        
                        for k in k_values:
                            for n_probe in n_probe_values:
                                h_eval = self.evaluate_hybrid_hnsw(hybrid_index, k, n_probe, ground_truths[k])
                                h_eval['hybrid_build_time'] = hybrid_build_time
                                h_eval['hybrid_k_children'] = hybrid_stats.get('k_children', params['k_children'])
                                phase_records.append({**h_eval, 'k': k})
                                print(f"   ✅ k={k} n_probe={n_probe}: recall={h_eval['recall_at_k']:.4f}, 时间={h_eval['avg_query_time_ms']:.2f}ms")
                    except Exception as he:
                        print(f"   ❌ Hybrid HNSW 评估失败: {he}")

                # ========== 方法4: KMeans HNSW (单枢纽) ==========
                print(f"\n📊 方法4: KMeans HNSW (单枢纽)评估")
                print(f"   参数: n_clusters={current_n_clusters}, k_children={params['k_children']}")
                
                construction_start = time.time()
                kmeans_hnsw = KMeansHNSW(
                    base_index=base_index,
                    **params,
                    adaptive_k_children=adaptive_config['adaptive_k_children'],
                    k_children_scale=adaptive_config['k_children_scale'],
                    k_children_min=adaptive_config['k_children_min'],
                    k_children_max=adaptive_config['k_children_max'],
                    diversify_max_assignments=adaptive_config['diversify_max_assignments'],
                    repair_min_assignments=adaptive_config['repair_min_assignments'],
                    shared_kmeans_model=shared_model,
                    shared_dataset_vectors=shared_dataset_vectors
                )
                construction_time = time.time() - construction_start
                print(f"   构建完成 (耗时: {construction_time:.2f}秒)")
                actual_n_clusters = kmeans_hnsw.n_clusters

                for k in k_values:
                    for n_probe in n_probe_values:
                        eval_result = self.evaluate_recall(kmeans_hnsw, k, n_probe, ground_truths[k])
                        phase_records.append({**eval_result, 'phase': 'kmeans_hnsw_single_pivot'})
                        print(f"   ✅ k={k} n_probe={n_probe}: recall={eval_result['recall_at_k']:.4f}, 时间={eval_result['avg_query_time_ms']:.2f}ms")




                # ========== 方法5: Multi-Pivot KMeans HNSW ==========
                if multi_pivot_config.get('enabled', False):
                    print(f"\n📊 方法5: Multi-Pivot KMeans HNSW评估")
                    print(f"   参数: pivots={multi_pivot_config.get('num_pivots', 3)}, "
                          f"strategy={multi_pivot_config.get('pivot_selection_strategy', 'line_perp_third')}")
                    try:
                        multi_pivot_start = time.time()
                        multi_pivot_hnsw = KMeansHNSWMultiPivot(
                            base_index=base_index,
                            **params,
                            num_pivots=multi_pivot_config.get('num_pivots', 3),
                            pivot_selection_strategy=multi_pivot_config.get('pivot_selection_strategy', 'line_perp_third'),
                            pivot_overquery_factor=multi_pivot_config.get('pivot_overquery_factor', 1.2),
                            adaptive_k_children=adaptive_config['adaptive_k_children'],
                            k_children_scale=adaptive_config['k_children_scale'],
                            k_children_min=adaptive_config['k_children_min'],
                            k_children_max=adaptive_config['k_children_max'],
                            diversify_max_assignments=adaptive_config['diversify_max_assignments'],
                            repair_min_assignments=adaptive_config['repair_min_assignments'],
                            shared_kmeans_model=shared_model,
                            shared_dataset_vectors=shared_dataset_vectors
                        )
                        multi_pivot_build_time = time.time() - multi_pivot_start
                        print(f"   构建完成 (耗时: {multi_pivot_build_time:.2f}秒)")
                        
                        for k in k_values:
                            for n_probe in n_probe_values:
                                mp_eval_result = self.evaluate_multi_pivot_recall(multi_pivot_hnsw, k, n_probe, ground_truths[k])
                                phase_records.append({**mp_eval_result, 'phase': 'kmeans_hnsw_multi_pivot', 'multi_pivot_build_time': multi_pivot_build_time})
                                print(f"   ✅ k={k} n_probe={n_probe}: recall={mp_eval_result['recall_at_k']:.4f}, 时间={mp_eval_result['avg_query_time_ms']:.2f}ms")
                    
                    except Exception as mp_e:
                        print(f"   ❌ Multi-Pivot KMeans HNSW 评估失败: {mp_e}")
                        traceback.print_exc()

                # ========== 组合总结 ==========
                print(f"\n📈 参数组合 {i + 1} 评估完成!")
                best_recall = max(r['recall_at_k'] for r in phase_records if 'recall_at_k' in r)
                methods_tested = len(set(r.get('phase', r.get('method', 'unknown')) for r in phase_records))
                print(f"   测试了 {methods_tested} 种方法，最佳召回率: {best_recall:.4f}")
                
                combination_results = {
                    'parameters': params,
                    'construction_time': construction_time if 'construction_time' in locals() else 0.0,
                    'phase_evaluations': phase_records,
                    'multi_pivot_enabled': multi_pivot_config.get('enabled', False),
                    'best_recall': best_recall,
                    'methods_count': methods_tested
                }
                results.append(combination_results)
                
            except Exception as e:
                print(f"❌ 参数组合 {params} 评估出错: {e}")
                traceback.print_exc()
                continue

        # ========== 最终总结 ==========
        print(f"\n� ================== 五方法对比评估完成 ==================")
        print(f"📊 总计测试: {len(results)} 个参数组合")
        print(f"🎯 Multi-Pivot启用: {multi_pivot_config.get('enabled', False)}")
        
        if results:
            overall_best = max(results, key=lambda x: x.get('best_recall', 0))
            print(f"🥇 全局最佳召回率: {overall_best.get('best_recall', 0):.4f}")
            print(f"🔧 最佳参数组合: {overall_best.get('parameters', {})}")
        
        print(f"================================================================")
        return results
    

    
    def _evaluate_pure_kmeans_from_existing_shared(
        self, 
        kmeans_model: MiniBatchKMeans, 
        dataset_vectors: np.ndarray,
        base_index: HNSW,
        k: int, 
        ground_truth: Dict, 
        n_probe: int = 1
    ) -> Dict[str, Any]:
        """
        使用共享K-Means模型直接评估纯K-Means性能
        (Evaluate pure K-Means using shared model directly)
        """
        print(f"    使用共享K-Means模型进行评估 (n_clusters={kmeans_model.n_clusters}, n_probe={n_probe})")
        
        # 获取聚类中心和数据标签
        centers = kmeans_model.cluster_centers_
        n_clusters = centers.shape[0]
        labels = kmeans_model.predict(dataset_vectors)
        
        # 构建聚类到成员的映射
        clusters = [[] for _ in range(n_clusters)]
        dataset_idx_to_original_id = list(base_index.keys())
        
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
                        member_vec = dataset_vectors[dataset_idx]
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
            'method': 'pure_kmeans_shared_model',
            'recall_at_k': overall_recall,
            'total_correct': total_correct,
            'total_expected': total_expected,
            'individual_recalls': individual_recalls,
            'avg_individual_recall': float(np.mean(individual_recalls)),
            'std_individual_recall': float(np.std(individual_recalls)),
            'avg_query_time_ms': float(np.mean(query_times) * 1000),
            'std_query_time_ms': float(np.std(query_times) * 1000),
            'clustering_time': 0.0,  # 使用共享模型，时间为0
            'n_clusters': n_clusters,
            'n_probe': n_probe_eff,
            'k': k,
            'used_shared_model': True
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
    parser.add_argument('--hybrid-parent-level', type=int, default=2,
                        help='Hybrid HNSW 父节点层级 (默认:2) (Parent level for level-based Hybrid HNSW)')
    parser.add_argument('--no-hybrid', action='store_true',
                        help='禁用Hybrid HNSW评估 (Disable Hybrid HNSW evaluation)')
    
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
    # 基线 HNSW ef 固定为 200（用户指定逻辑）
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
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
    'n_probe_values': [5, 10, 20],
    'hybrid_parent_level': args.hybrid_parent_level,
    'enable_hybrid': (not args.no_hybrid)
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
        'repair_min_assignments': args.repair_min_assignments if args.repair_min_assignments is not None else 1
    }
    
    # 准备Multi-Pivot配置
    multi_pivot_config = {
        'enabled': args.enable_multi_pivot,
        'num_pivots': args.num_pivots,
        'pivot_selection_strategy': args.pivot_selection_strategy,
        'pivot_overquery_factor': args.pivot_overquery_factor
    }
    
    sweep_results = evaluator.parameter_sweep(
        base_index,
        param_grid,
        evaluation_params,
        max_combinations=max_combos,
        adaptive_config=adaptive_config,
        multi_pivot_config=multi_pivot_config
    )
    
    # 使用第一个参数组合进行演示 (Use first parameter combination for demonstration)
    if sweep_results:
        # 取第一个扫描结果作为演示参数
        demo_result = sweep_results[0]
        demo_params = demo_result['parameters']
        print(f"\nUsing first parameter combination for demonstration: {demo_params}")
        print("\n🎯 Parameter sweep completed! All comparisons are available in sweep_results.")

        # Save results
        results = {
            'sweep_results': sweep_results,
            'demo_parameters': demo_params,
            'multi_pivot_config': multi_pivot_config,
            'adaptive_config': {
                'adaptive_k_children': args.adaptive_k_children,
                'k_children_scale': args.k_children_scale,
                'k_children_min': args.k_children_min,
                'k_children_max': args.k_children_max,
                'diversify_max_assignments': args.diversify_max_assignments,
                'repair_min_assignments': args.repair_min_assignments,
                # manual repair parameters removed
            },
            'evaluation_info': {
                'dataset_size': len(base_vectors),
                'query_size': len(query_vectors),
                'dimension': base_vectors.shape[1],
                'multi_pivot_enabled': args.enable_multi_pivot,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        save_results(results, 'method3_tuning_results.json')
        
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
    
    print("Results saved to method3_tuning_results.json")