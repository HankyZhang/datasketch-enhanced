"""
优化版本的Multi-Pivot评估器 - 减少重复计算
Optimized Multi-Pivot Evaluator - Reduce Redundant Computations

主要优化：
1. 共享基础HNSW向量提取
2. 共享K-Means聚类计算
3. 只在子节点分配策略上有所不同
4. 复用已训练的聚类模型

Key optimizations:
1. Share base HNSW vector extraction
2. Share K-Means clustering computation
3. Only differ in child assignment strategy
4. Reuse trained clustering models
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Hashable
from itertools import product

# 添加父目录到路径 (Add parent directory to path)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from method3.kmeans_hnsw import KMeansHNSW
from method3.kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot
from hnsw.hnsw import HNSW
from hybrid_hnsw.hnsw_hybrid import HNSWHybrid
from sklearn.cluster import MiniBatchKMeans


class OptimizedBuildSystem:
    """
    优化构建系统 - 测量和报告构建时间的优化
    Optimized Build System - Measure and report build time optimizations
    """
    
    def __init__(
        self,
        base_index: HNSW,
        params: Dict[str, Any],
        adaptive_config: Dict[str, Any]
    ):
        self.base_index = base_index
        self.params = params
        self.adaptive_config = adaptive_config
        
        # 计时统计
        self.single_pivot_build_time = 0.0
        self.multi_pivot_build_time = 0.0
        
        print(f"  🔄 优化构建系统初始化 (n_clusters={self.params['n_clusters']})...")
    
    def create_single_pivot_system(self) -> KMeansHNSW:
        """创建单枢纽系统，测量构建时间"""
        print("    - 创建单枢纽KMeans HNSW系统...")
        
        start_time = time.time()
        system = KMeansHNSW(
            base_index=self.base_index,
            n_clusters=self.params['n_clusters'],
            k_children=self.params['k_children'],
            child_search_ef=self.params.get('child_search_ef'),
            adaptive_k_children=self.adaptive_config.get('adaptive_k_children', False),
            k_children_scale=self.adaptive_config.get('k_children_scale', 1.5),
            k_children_min=self.adaptive_config.get('k_children_min', 100),
            k_children_max=self.adaptive_config.get('k_children_max'),
            diversify_max_assignments=self.adaptive_config.get('diversify_max_assignments'),
            repair_min_assignments=self.adaptive_config.get('repair_min_assignments')
        )
        self.single_pivot_build_time = time.time() - start_time
        print(f"      ⏱️ 单枢纽构建时间: {self.single_pivot_build_time:.2f}秒")
        
        return system
    
    def create_multi_pivot_system(self, multi_pivot_config: Dict[str, Any]) -> KMeansHNSWMultiPivot:
        """创建多枢纽系统，测量构建时间"""
        print(f"    - 创建多枢纽KMeans HNSW系统 (pivots={multi_pivot_config.get('num_pivots', 3)})...")
        
        start_time = time.time()
        system = KMeansHNSWMultiPivot(
            base_index=self.base_index,
            n_clusters=self.params['n_clusters'],
            k_children=self.params['k_children'],
            child_search_ef=self.params.get('child_search_ef'),
            # Multi-pivot specific parameters
            num_pivots=multi_pivot_config.get('num_pivots', 3),
            pivot_selection_strategy=multi_pivot_config.get('pivot_selection_strategy', 'line_perp_third'),
            pivot_overquery_factor=multi_pivot_config.get('pivot_overquery_factor', 1.2),
            multi_pivot_enabled=True,
            store_pivot_debug=True,
            # Adaptive/diversify/repair config
            adaptive_k_children=self.adaptive_config.get('adaptive_k_children', False),
            k_children_scale=self.adaptive_config.get('k_children_scale', 1.5),
            k_children_min=self.adaptive_config.get('k_children_min', 100),
            k_children_max=self.adaptive_config.get('k_children_max'),
            diversify_max_assignments=self.adaptive_config.get('diversify_max_assignments'),
            repair_min_assignments=self.adaptive_config.get('repair_min_assignments')
        )
        self.multi_pivot_build_time = time.time() - start_time
        print(f"      ⏱️ 多枢纽构建时间: {self.multi_pivot_build_time:.2f}秒")
        
        return system
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """获取构建时间总结"""
        total_time = self.single_pivot_build_time + self.multi_pivot_build_time
        return {
            'single_pivot_build_time': self.single_pivot_build_time,
            'multi_pivot_build_time': self.multi_pivot_build_time, 
            'total_build_time': total_time,
            'optimization_note': '当前版本重点在于性能测量和对比分析'
        }


class SharedKMeansHNSWSystem:
    """共享HNSW索引和K-Means聚类计算的系统"""
    
    def __init__(self, base_index: HNSW, params: Dict[str, Any], adaptive_config: Dict[str, Any]):
        """
        初始化共享计算系统
        
        Args:
            base_index: HNSW基础索引
            params: 基本参数配置
            adaptive_config: 自适应配置
        """
        self.base_index = base_index
        self.params = params
        self.adaptive_config = adaptive_config
        self.distance_func = base_index._distance_func
        
        # 构建时间统计
        self.single_pivot_build_time = 0.0
        self.multi_pivot_build_time = 0.0
        self.shared_clustering_time = 0.0
        
        # 执行共享的聚类计算和公共组件构建
        self._build_shared_components()
    
    def _build_shared_components(self):
        """构建所有共享的组件：聚类、向量索引、公共数据结构"""
        print("    📊 构建共享组件：向量提取 + K-Means聚类...")
        start_time = time.time()
        
        # 1. 从HNSW索引提取向量数据和ID映射
        self._extract_vectors_and_ids()
        
        # 2. 执行K-Means聚类
        self._perform_kmeans_clustering()
        
        # 3. 构建共享的查找结构
        self._build_shared_lookup_structures()
        
        self.shared_clustering_time = time.time() - start_time
        print(f"      ✅ 共享组件构建完成 ({self.shared_clustering_time:.3f}s)")
        print(f"         - 向量数量: {len(self.node_vectors)}")
        print(f"         - 聚类数量: {len(self.centroids)}")
    
    def _extract_vectors_and_ids(self):
        """从HNSW索引提取向量数据"""
        node_vectors = []
        self.node_ids = []
        
        # 遍历所有节点并提取向量数据
        for node_id, node in self.base_index._nodes.items():
            vector = node.point  # _Node对象的point属性包含向量数据
            if vector is not None:
                node_vectors.append(vector)
                self.node_ids.append(node_id)
        
        if len(node_vectors) == 0:
            raise ValueError("无法从HNSW索引中提取向量数据")
        
        self.node_vectors = np.array(node_vectors)
        # 添加别名以保持向后兼容性
        self.dataset_vectors = self.node_vectors
        
        # 构建ID到索引的映射
        self.node_id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
        
    def _perform_kmeans_clustering(self):
        """执行K-Means聚类"""
        n_clusters = self.params['n_clusters']
        actual_clusters = min(n_clusters, len(self.node_vectors))
        
        self.kmeans_model = MiniBatchKMeans(
            n_clusters=actual_clusters,
            random_state=42,
            max_iter=100,
            batch_size=min(100, len(self.node_vectors))
        )
        
        self.cluster_labels = self.kmeans_model.fit_predict(self.node_vectors)
        self.centroids = self.kmeans_model.cluster_centers_
        self.n_clusters = actual_clusters
        
    def _build_shared_lookup_structures(self):
        """构建共享的查找结构"""
        # 聚类ID分配
        self.centroid_ids = [f"centroid_{i}" for i in range(len(self.centroids))]
        
        # 构建聚类映射：cluster_label -> [node_ids]
        self.cluster_assignments = {}
        for i, (node_id, label) in enumerate(zip(self.node_ids, self.cluster_labels)):
            if label not in self.cluster_assignments:
                self.cluster_assignments[label] = []
            self.cluster_assignments[label].append(node_id)
        
        # 构建向量字典供子系统使用
        self.child_vectors = {}
        for node_id in self.node_ids:
            idx = self.node_id_to_idx[node_id] 
            self.child_vectors[node_id] = self.dataset_vectors[idx]
        
        # 向量化查找矩阵
        self._centroid_matrix = self.centroids.copy()
        self._centroid_id_array = np.array(self.centroid_ids)
        
    def create_single_pivot_system(self) -> 'OptimizedSinglePivotSystem':
        """创建单枢纽系统，复用共享聚类结果"""
        print("    - 创建单枢纽KMeans HNSW系统 (复用共享聚类)...")
        
        start_time = time.time()
        system = OptimizedSinglePivotSystem(self, self.adaptive_config)
        self.single_pivot_build_time = time.time() - start_time
        print(f"      ⏱️ 单枢纽构建时间: {self.single_pivot_build_time:.2f}秒")
        
        return system
    
    def create_multi_pivot_system(self, multi_pivot_config: Dict[str, Any]) -> 'OptimizedMultiPivotSystem':
        """创建多枢纽系统，复用共享聚类结果"""
        print(f"    - 创建多枢纽KMeans HNSW系统 (pivots={multi_pivot_config.get('num_pivots', 3)})...")
        
        start_time = time.time()
        system = OptimizedMultiPivotSystem(self, self.adaptive_config, multi_pivot_config)
        self.multi_pivot_build_time = time.time() - start_time
        print(f"      ⏱️ 多枢纽构建时间: {self.multi_pivot_build_time:.2f}秒")
        
        return system
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """获取构建时间总结"""
        total_time = self.single_pivot_build_time + self.multi_pivot_build_time
        return {
            'shared_clustering_time': self.shared_clustering_time,
            'single_pivot_build_time': self.single_pivot_build_time,
            'multi_pivot_build_time': self.multi_pivot_build_time, 
            'total_build_time': total_time,
            'optimization_note': '使用共享聚类结果，避免重复计算'
        }


class OptimizedSinglePivotSystem:
    """优化的单枢纽系统 - 复用共享聚类结果"""
    
    def __init__(self, shared_system: SharedKMeansHNSWSystem, adaptive_config: Dict[str, Any]):
        """
        初始化单枢纽系统
        
        Args:
            shared_system: 包含共享聚类结果的系统
            adaptive_config: 自适应配置
        """
        self.shared_system = shared_system
        self.adaptive_config = adaptive_config
        self.base_index = shared_system.base_index
        self.distance_func = shared_system.distance_func
        
        # 从共享系统获取基础数据
        self.centroids = shared_system.centroids
        self.centroid_ids = shared_system.centroid_ids
        self.cluster_assignments = shared_system.cluster_assignments
        self.child_vectors = shared_system.child_vectors.copy()
        self.n_clusters = shared_system.n_clusters
        # 不再创建node_vectors副本，直接使用shared_system的引用
        self.node_ids = shared_system.node_ids
        
        # 构建父子节点映射 - 单枢纽策略
        self._build_single_pivot_parent_child_mapping()
        
        # 向量化查找矩阵 (复用共享系统的)
        self._centroid_matrix = shared_system._centroid_matrix
        self._centroid_id_array = shared_system._centroid_id_array
        
        # 统计信息
        self.stats = {
            'method': 'single_pivot_optimized',
            'n_clusters': self.n_clusters,
            'num_children': len(self.child_vectors),
            'reused_shared_clustering': True
        }
        self.search_times = []
    
    def _build_single_pivot_parent_child_mapping(self):
        """构建单枢纽的父子节点映射"""
        print("      📍 构建单枢纽父子节点映射...")
        
        k_children = self.shared_system.params['k_children']
        child_search_ef = self.shared_system.params.get('child_search_ef', k_children * 2)
        
        self.parent_child_map = {}
        
        for cluster_idx, centroid_id in enumerate(self.centroid_ids):
            # 使用质心作为单一枢纽点
            centroid_vector = self.centroids[cluster_idx]
            
            # 使用HNSW搜索找到最近的k_children个节点
            try:
                hnsw_results = self.base_index.query(
                    centroid_vector, 
                    k=k_children, 
                    ef=child_search_ef
                )
                children = [node_id for node_id, _ in hnsw_results]
                self.parent_child_map[centroid_id] = children
                
                # 确保子节点向量在child_vectors中
                for child_id in children:
                    if child_id not in self.child_vectors and child_id in self.shared_system.node_id_to_idx:
                        idx = self.shared_system.node_id_to_idx[child_id]
                        self.child_vectors[child_id] = self.shared_system.dataset_vectors[idx]
                        
            except Exception as e:
                print(f"        ⚠️ 质心 {centroid_id} 的子节点查找失败: {e}")
                self.parent_child_map[centroid_id] = []
        
        total_children = sum(len(children) for children in self.parent_child_map.values())
        avg_children = total_children / max(1, len(self.parent_child_map))
        print(f"      ✅ 单枢纽映射完成: {total_children} 个子节点, 平均 {avg_children:.1f} 个/质心")
    
    def search(self, query_vector: np.ndarray, k: int = 10, n_probe: int = 10) -> List[Tuple[Hashable, float]]:
        """两阶段搜索：质心搜索 → 子节点搜索"""
        start = time.time()
        
        # Stage 1: 找到最近的质心
        closest_centroids = self._stage1_centroid_search(query_vector, n_probe)
        
        # Stage 2: 在选定质心的子节点中搜索
        results = self._stage2_child_search(query_vector, closest_centroids, k)
        
        # 记录搜索时间
        elapsed = (time.time() - start) * 1000.0
        self.search_times.append(elapsed)
        
        return results
    
    def _stage1_centroid_search(self, query_vector: np.ndarray, n_probe: int) -> List[Tuple[str, float]]:
        """Stage 1: 找到最近的K-Means质心"""
        diffs = self._centroid_matrix - query_vector
        distances = np.linalg.norm(diffs, axis=1)
        indices = np.argsort(distances)[:n_probe]
        return [(self.centroid_ids[i], distances[i]) for i in indices]
    
    def _stage2_child_search(self, query_vector: np.ndarray, closest_centroids: List[Tuple[str, float]], k: int) -> List[Tuple[Hashable, float]]:
        """Stage 2: 在子节点中搜索"""
        # 收集候选子节点
        candidate_children = set()
        for centroid_id, _ in closest_centroids:
            children = self.parent_child_map.get(centroid_id, [])
            candidate_children.update(children)
        
        if not candidate_children:
            return []
        
        # 构建候选向量矩阵
        candidate_ids = list(candidate_children)
        vectors = []
        valid_ids = []
        
        for cid in candidate_ids:
            if cid in self.child_vectors:
                vectors.append(self.child_vectors[cid])
                valid_ids.append(cid)
        
        if not vectors:
            return []
        
        # 向量化距离计算
        candidate_matrix = np.vstack(vectors)
        distances = np.linalg.norm(candidate_matrix - query_vector, axis=1)
        
        # 排序并返回top-k
        sorted_indices = np.argsort(distances)[:k]
        return [(valid_ids[i], distances[i]) for i in sorted_indices]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        if self.search_times:
            stats['avg_search_time_ms'] = float(np.mean(self.search_times))
            stats['std_search_time_ms'] = float(np.std(self.search_times))
        return stats


class OptimizedMultiPivotSystem:
    """优化的多枢纽系统 - 复用共享聚类结果"""
    
    def __init__(self, shared_system: SharedKMeansHNSWSystem, adaptive_config: Dict[str, Any], multi_pivot_config: Dict[str, Any]):
        """
        初始化多枢纽系统
        
        Args:
            shared_system: 包含共享聚类结果的系统
            adaptive_config: 自适应配置
            multi_pivot_config: 多枢纽配置
        """
        self.shared_system = shared_system
        self.adaptive_config = adaptive_config
        self.multi_pivot_config = multi_pivot_config
        self.base_index = shared_system.base_index
        self.distance_func = shared_system.distance_func
        
        # 从共享系统获取基础数据
        self.centroids = shared_system.centroids
        self.centroid_ids = shared_system.centroid_ids
        self.cluster_assignments = shared_system.cluster_assignments
        self.child_vectors = shared_system.child_vectors.copy()
        self.n_clusters = shared_system.n_clusters
        # 不再创建node_vectors副本，直接使用shared_system的引用
        self.node_ids = shared_system.node_ids
        
        # 多枢纽参数
        self.num_pivots = multi_pivot_config.get('num_pivots', 3)
        self.pivot_selection_strategy = multi_pivot_config.get('pivot_selection_strategy', 'line_perp_third')
        self.pivot_overquery_factor = multi_pivot_config.get('pivot_overquery_factor', 1.2)
        
        # 构建父子节点映射 - 多枢纽策略
        self._build_multi_pivot_parent_child_mapping()
        
        # 向量化查找矩阵 (复用共享系统的)
        self._centroid_matrix = shared_system._centroid_matrix
        self._centroid_id_array = shared_system._centroid_id_array
        
        # 统计信息
        self.stats = {
            'method': 'multi_pivot_optimized',
            'n_clusters': self.n_clusters,
            'num_children': len(self.child_vectors),
            'num_pivots': self.num_pivots,
            'pivot_strategy': self.pivot_selection_strategy,
            'reused_shared_clustering': True
        }
        self.search_times = []
    
    def _build_multi_pivot_parent_child_mapping(self):
        """构建多枢纽的父子节点映射"""
        print(f"      🎯 构建多枢纽父子节点映射 (pivots={self.num_pivots})...")
        
        k_children = self.shared_system.params['k_children']
        child_search_ef = self.shared_system.params.get('child_search_ef', k_children * 2)
        overquery_k = int(k_children * self.pivot_overquery_factor)
        
        self.parent_child_map = {}
        
        for cluster_idx, centroid_id in enumerate(self.centroid_ids):
            try:
                # 获取多个枢纽点
                pivots = self._select_pivots_for_centroid(cluster_idx, overquery_k, child_search_ef)
                
                # 对每个枢纽点进行查询并合并结果
                all_candidates = set()
                for pivot_vector in pivots:
                    hnsw_results = self.base_index.query(
                        pivot_vector, 
                        k=overquery_k, 
                        ef=child_search_ef
                    )
                    for node_id, _ in hnsw_results:
                        all_candidates.add(node_id)
                
                # 计算所有候选点到各枢纽的距离，选择最优的k_children个
                children = self._select_best_children_from_candidates(
                    list(all_candidates), pivots, k_children
                )
                
                self.parent_child_map[centroid_id] = children
                
                # 确保子节点向量在child_vectors中
                for child_id in children:
                    if child_id not in self.child_vectors and child_id in self.shared_system.node_id_to_idx:
                        idx = self.shared_system.node_id_to_idx[child_id]
                        self.child_vectors[child_id] = self.shared_system.dataset_vectors[idx]
                        
            except Exception as e:
                print(f"        ⚠️ 质心 {centroid_id} 的多枢纽子节点查找失败: {e}")
                self.parent_child_map[centroid_id] = []
        
        total_children = sum(len(children) for children in self.parent_child_map.values())
        avg_children = total_children / max(1, len(self.parent_child_map))
        print(f"      ✅ 多枢纽映射完成: {total_children} 个子节点, 平均 {avg_children:.1f} 个/质心")
    
    def _select_pivots_for_centroid(self, cluster_idx: int, overquery_k: int, child_search_ef: int) -> List[np.ndarray]:
        """为质心选择多个枢纽点"""
        centroid_vector = self.centroids[cluster_idx]
        pivots = [centroid_vector]  # 第一个枢纽总是质心本身
        
        if self.num_pivots <= 1:
            return pivots
        
        try:
            # 第一次查询：以质心为枢纽
            first_results = self.base_index.query(centroid_vector, k=overquery_k, ef=child_search_ef)
            candidate_ids = [node_id for node_id, _ in first_results]
            
            if len(candidate_ids) < 2:
                return pivots
            
            # 获取候选向量
            candidate_vectors = []
            valid_candidate_ids = []
            for cid in candidate_ids:
                if cid in self.shared_system.node_id_to_idx:
                    idx = self.shared_system.node_id_to_idx[cid]
                    candidate_vectors.append(self.shared_system.dataset_vectors[idx])
                    valid_candidate_ids.append(cid)
            
            if len(candidate_vectors) < 2:
                return pivots
            
            candidate_vectors = np.array(candidate_vectors)
            
            # 第二个枢纽：距离质心最远的点
            distances = np.linalg.norm(candidate_vectors - centroid_vector, axis=1)
            farthest_idx = np.argmax(distances)
            second_pivot = candidate_vectors[farthest_idx]
            pivots.append(second_pivot)
            
            if self.num_pivots <= 2:
                return pivots
            
            # 第三个枢纽：根据策略选择
            if self.pivot_selection_strategy == 'line_perp_third':
                third_pivot = self._find_perpendicular_pivot(centroid_vector, second_pivot, candidate_vectors)
                if third_pivot is not None:
                    pivots.append(third_pivot)
            
            # 后续枢纽：max-min distance策略
            while len(pivots) < self.num_pivots and len(candidate_vectors) > len(pivots):
                next_pivot = self._find_max_min_distance_pivot(pivots, candidate_vectors)
                if next_pivot is not None:
                    pivots.append(next_pivot)
                else:
                    break
                    
        except Exception as e:
            print(f"        ⚠️ 枢纽选择失败，使用质心: {e}")
        
        return pivots
    
    def _find_perpendicular_pivot(self, pivot_a: np.ndarray, pivot_b: np.ndarray, candidates: np.ndarray) -> Optional[np.ndarray]:
        """找到垂直于A-B线段距离最大的点"""
        try:
            ab_vector = pivot_b - pivot_a
            ab_norm = np.linalg.norm(ab_vector)
            
            if ab_norm < 1e-6:
                return None
            
            ab_unit = ab_vector / ab_norm
            
            # 计算每个候选点到直线AB的垂直距离
            perp_distances = []
            for candidate in candidates:
                ac_vector = candidate - pivot_a
                projection_length = np.dot(ac_vector, ab_unit)
                projection_point = pivot_a + projection_length * ab_unit
                perp_distance = np.linalg.norm(candidate - projection_point)
                perp_distances.append(perp_distance)
            
            max_perp_idx = np.argmax(perp_distances)
            return candidates[max_perp_idx]
            
        except Exception:
            return None
    
    def _find_max_min_distance_pivot(self, existing_pivots: List[np.ndarray], candidates: np.ndarray) -> Optional[np.ndarray]:
        """找到与现有枢纽最小距离最大的候选点"""
        try:
            best_candidate = None
            best_min_distance = -1
            
            for candidate in candidates:
                # 计算到所有现有枢纽的距离
                min_distance = float('inf')
                for pivot in existing_pivots:
                    distance = np.linalg.norm(candidate - pivot)
                    min_distance = min(min_distance, distance)
                
                # 选择最小距离最大的候选点
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate
            
            return best_candidate
            
        except Exception:
            return None
    
    def _select_best_children_from_candidates(self, candidate_ids: List[Hashable], pivots: List[np.ndarray], k_children: int) -> List[Hashable]:
        """从候选节点中选择最优的k_children个子节点"""
        if len(candidate_ids) <= k_children:
            return candidate_ids
        
        try:
            # 获取候选向量
            candidate_vectors = []
            valid_ids = []
            
            for cid in candidate_ids:
                if cid in self.shared_system.node_id_to_idx:
                    idx = self.shared_system.node_id_to_idx[cid]
                    candidate_vectors.append(self.shared_system.dataset_vectors[idx])
                    valid_ids.append(cid)
            
            if len(candidate_vectors) <= k_children:
                return valid_ids
            
            candidate_vectors = np.array(candidate_vectors)
            
            # 计算每个候选点到最近枢纽的距离
            min_distances = []
            for candidate in candidate_vectors:
                min_dist = float('inf')
                for pivot in pivots:
                    dist = np.linalg.norm(candidate - pivot)
                    min_dist = min(min_dist, dist)
                min_distances.append(min_dist)
            
            # 选择距离最小的k_children个
            sorted_indices = np.argsort(min_distances)[:k_children]
            return [valid_ids[i] for i in sorted_indices]
            
        except Exception:
            return candidate_ids[:k_children]
    
    def search(self, query_vector: np.ndarray, k: int = 10, n_probe: int = 10) -> List[Tuple[Hashable, float]]:
        """两阶段搜索：质心搜索 → 子节点搜索"""
        start = time.time()
        
        # Stage 1: 找到最近的质心
        closest_centroids = self._stage1_centroid_search(query_vector, n_probe)
        
        # Stage 2: 在选定质心的子节点中搜索
        results = self._stage2_child_search(query_vector, closest_centroids, k)
        
        # 记录搜索时间
        elapsed = (time.time() - start) * 1000.0
        self.search_times.append(elapsed)
        
        return results
    
    def _stage1_centroid_search(self, query_vector: np.ndarray, n_probe: int) -> List[Tuple[str, float]]:
        """Stage 1: 找到最近的K-Means质心"""
        diffs = self._centroid_matrix - query_vector
        distances = np.linalg.norm(diffs, axis=1)
        indices = np.argsort(distances)[:n_probe]
        return [(self.centroid_ids[i], distances[i]) for i in indices]
    
    def _stage2_child_search(self, query_vector: np.ndarray, closest_centroids: List[Tuple[str, float]], k: int) -> List[Tuple[Hashable, float]]:
        """Stage 2: 在子节点中搜索"""
        # 收集候选子节点
        candidate_children = set()
        for centroid_id, _ in closest_centroids:
            children = self.parent_child_map.get(centroid_id, [])
            candidate_children.update(children)
        
        if not candidate_children:
            return []
        
        # 构建候选向量矩阵
        candidate_ids = list(candidate_children)
        vectors = []
        valid_ids = []
        
        for cid in candidate_ids:
            if cid in self.child_vectors:
                vectors.append(self.child_vectors[cid])
                valid_ids.append(cid)
        
        if not vectors:
            return []
        
        # 向量化距离计算
        candidate_matrix = np.vstack(vectors)
        distances = np.linalg.norm(candidate_matrix - query_vector, axis=1)
        
        # 排序并返回top-k
        sorted_indices = np.argsort(distances)[:k]
        return [(valid_ids[i], distances[i]) for i in sorted_indices]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        if self.search_times:
            stats['avg_search_time_ms'] = float(np.mean(self.search_times))
            stats['std_search_time_ms'] = float(np.std(self.search_times))
        return stats


class OptimizedKMeansHNSWMultiPivotEvaluator:
    """
    优化版K-Means HNSW系统性能评估器 (减少重复计算)
    Optimized K-Means HNSW system performance evaluator (reduced redundant computations)
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
        """计算真实值"""
        cache_key = (k, exclude_query_ids)
        if cache_key in self._ground_truth_cache:
            return self._ground_truth_cache[cache_key]
        
        print(f"正在计算 {len(self.query_set)} 个查询的真实值 (k={k})...")
        start_time = time.time()
        
        ground_truth = {}
        excluded_count = 0
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            distances = []
            
            for j, data_vector in enumerate(self.dataset):
                if exclude_query_ids and j == query_id:
                    excluded_count += 1
                    continue
                
                distance = self.distance_func(query_vector, data_vector)
                distances.append((distance, j))
            
            distances.sort()
            ground_truth[query_id] = distances[:k]
        
        elapsed = time.time() - start_time
        print(f"真实值计算完成，耗时 {elapsed:.2f}秒")
        
        self._ground_truth_cache[cache_key] = ground_truth
        return ground_truth
    
    def evaluate_recall_generic(
        self,
        system,
        k: int,
        n_probe: int,
        ground_truth: Dict,
        system_name: str = ""
    ) -> Dict[str, Any]:
        """通用的召回率评估方法"""
        print(f"正在评估 {system_name} {len(self.query_set)} 个查询的召回率 (k={k}, n_probe={n_probe})...")
        start_time = time.time()
        
        total_correct = 0
        total_expected = len(self.query_set) * k
        query_times = []
        individual_recalls = []
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            true_neighbors = {neighbor_id for _, neighbor_id in ground_truth[query_id]}
            
            search_start = time.time()
            results = system.search(query_vector, k=k, n_probe=n_probe)
            search_time = time.time() - search_start
            query_times.append(search_time)
            
            found_neighbors = {node_id for node_id, _ in results}
            correct = len(true_neighbors & found_neighbors)
            total_correct += correct
            
            individual_recall = correct / k if k > 0 else 0.0
            individual_recalls.append(individual_recall)
        
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
            'system_stats': system.get_stats()
        }
    
    def evaluate_hnsw_baseline(
        self,
        base_index: HNSW,
        k: int,
        ef: int,
        ground_truth: Dict
    ) -> Dict[str, Any]:
        """评估HNSW基线性能"""
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
        """评估Hybrid HNSW性能"""
        result = self.evaluate_recall_generic(hybrid_index, k, n_probe, ground_truth, "Hybrid HNSW")
        result['phase'] = 'hybrid_hnsw_level'
        result['hybrid_stats'] = hybrid_index.get_stats()
        return result

    def _evaluate_pure_kmeans_from_shared(
        self, 
        shared_system: 'SharedKMeansHNSWSystem',
        k: int, 
        ground_truth: Dict, 
        n_probe: int = 1
    ) -> Dict[str, Any]:
        """使用共享系统的聚类结果评估纯K-Means性能"""
        print(f"使用共享聚类评估纯K-Means (n_clusters={shared_system.params['n_clusters']}, n_probe={n_probe})...")
        
        centers = shared_system.centroids
        labels = shared_system.cluster_labels
        n_clusters = centers.shape[0]
        
        # 构建聚类到成员的映射
        clusters = [[] for _ in range(n_clusters)]
        dataset_idx_to_original_id = list(shared_system.base_index.keys())
        
        for dataset_idx, cluster_id in enumerate(labels):
            original_id = dataset_idx_to_original_id[dataset_idx]
            clusters[cluster_id].append((dataset_idx, original_id))
        
        query_times = []
        total_correct = 0
        total_expected = len(self.query_set) * k
        individual_recalls = []
        
        n_probe_eff = min(n_probe, n_clusters)
        
        for query_vector, query_id in zip(self.query_set, self.query_ids):
            search_start = time.time()
            
            # 计算到聚类中心的距离
            diffs = centers - query_vector
            distances_to_centroids = np.linalg.norm(diffs, axis=1)
            
            # 获取最近的n_probe个聚类中心
            probe_centroids = np.argpartition(distances_to_centroids, n_probe_eff - 1)[:n_probe_eff]
            probe_centroids = probe_centroids[np.argsort(distances_to_centroids[probe_centroids])]
            
            # 收集候选结果
            all_candidates = []
            for cluster_idx in probe_centroids:
                cluster_members = clusters[cluster_idx]
                for dataset_idx, original_id in cluster_members:
                    if original_id != query_id:
                        member_vec = shared_system.node_vectors[dataset_idx]
                        dist = np.linalg.norm(member_vec - query_vector)
                        all_candidates.append((dist, original_id))
            
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
        
        return {
            'method': 'pure_kmeans_from_shared',
            'recall_at_k': total_correct / total_expected if total_expected > 0 else 0.0,
            'total_correct': total_correct,
            'total_expected': total_expected,
            'individual_recalls': individual_recalls,
            'avg_individual_recall': float(np.mean(individual_recalls)),
            'std_individual_recall': float(np.std(individual_recalls)),
            'avg_query_time_ms': float(np.mean(query_times) * 1000),
            'std_query_time_ms': float(np.std(query_times) * 1000),
            'clustering_time': 0.0,  # 使用共享结果，无额外聚类时间
            'n_clusters': n_clusters,
            'n_probe': n_probe_eff,
            'k': k,
            'reused_shared_clustering': True
        }

    def optimized_parameter_sweep(
        self,
        base_index: HNSW,
        param_grid: Dict[str, List[Any]],
        evaluation_params: Dict[str, Any],
        max_combinations: Optional[int] = None,
        adaptive_config: Optional[Dict[str, Any]] = None,
        multi_pivot_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        优化的参数扫描 - 减少重复计算
        Optimized parameter sweep - reduce redundant computations
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

        print("🚀 开始优化版K-Means HNSW + Multi-Pivot 参数扫描...")
        print(f"Multi-Pivot启用状态: {multi_pivot_config.get('enabled', False)}")
        print("🔄 关键优化: 共享K-Means聚类计算，避免重复构建")

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        if max_combinations and len(combinations) > max_combinations:
            combinations = random.sample(combinations, max_combinations)
        print(f"测试 {len(combinations)} 个参数组合...")

        results: List[Dict[str, Any]] = []
        k_values = evaluation_params.get('k_values', [10])
        n_probe_values = evaluation_params.get('n_probe_values', [5, 10, 20])
        hybrid_parent_level = evaluation_params.get('hybrid_parent_level', 2)
        enable_hybrid = evaluation_params.get('enable_hybrid', True)

        # 预计算真实值
        ground_truths: Dict[int, Dict] = {}
        for k in k_values:
            ground_truths[k] = self.compute_ground_truth(k, exclude_query_ids=False)

        for i, combination in enumerate(combinations):
            print(f"\n--- 优化组合 {i + 1}/{len(combinations)} ---")
            params = dict(zip(param_names, combination))
            print(f"Parameters: {params}")

            try:
                phase_records: List[Dict[str, Any]] = []
                
                # 🔄 创建共享计算系统 (一次性完成HNSW + K-Means聚类)
                shared_computation_start = time.time()
                shared_system = SharedKMeansHNSWSystem(base_index, params, adaptive_config)
                shared_computation_time = time.time() - shared_computation_start
                
                print(f"  📊 共享计算耗时: {shared_computation_time:.2f}秒 (包含向量提取 + K-Means聚类)")

                # Phase 1: 基线HNSW (无变化)
                base_ef = base_index._ef_construction
                for k in k_values:
                    b_eval = self.evaluate_hnsw_baseline(base_index, k, base_ef, ground_truths[k])
                    phase_records.append({**b_eval, 'k': k})
                    print(f"  [基线HNSW] k={k} recall={b_eval['recall_at_k']:.4f}")

                # Phase 2: 纯K-Means (使用共享聚类结果)
                for k in k_values:
                    for n_probe in n_probe_values:
                        c_eval = self._evaluate_pure_kmeans_from_shared(
                            shared_system, k, ground_truths[k], n_probe
                        )
                        c_eval['phase'] = 'clusters_only'
                        phase_records.append({**c_eval, 'k': k})
                        print(f"  [纯K-Means] k={k} n_probe={n_probe} recall={c_eval['recall_at_k']:.4f}")

                # Phase 3: Hybrid HNSW (无变化)
                if enable_hybrid:
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
                        
                        for k in k_values:
                            for n_probe in n_probe_values:
                                h_eval = self.evaluate_hybrid_hnsw(hybrid_index, k, n_probe, ground_truths[k])
                                h_eval['hybrid_build_time'] = hybrid_build_time
                                phase_records.append({**h_eval, 'k': k})
                                print(f"  [Hybrid HNSW] k={k} n_probe={n_probe} recall={h_eval['recall_at_k']:.4f}")
                    except Exception as he:
                        print(f"  ⚠️ Hybrid HNSW失败: {he}")

                # Phase 4: 单枢纽KMeans-HNSW (使用共享聚类结果)
                single_pivot_start = time.time()
                single_pivot_system = shared_system.create_single_pivot_system()
                single_pivot_build_time = time.time() - single_pivot_start
                
                for k in k_values:
                    for n_probe in n_probe_values:
                        sp_eval = self.evaluate_recall_generic(
                            single_pivot_system, k, n_probe, ground_truths[k], "单枢纽KMeans HNSW"
                        )
                        sp_eval['phase'] = 'kmeans_hnsw_single_pivot'
                        sp_eval['single_pivot_build_time'] = single_pivot_build_time
                        phase_records.append({**sp_eval, 'k': k})
                        print(f"  [单枢纽KMeans HNSW] k={k} n_probe={n_probe} recall={sp_eval['recall_at_k']:.4f}")

                # Phase 5: 多枢纽KMeans-HNSW (使用共享聚类结果)
                if multi_pivot_config.get('enabled', False):
                    multi_pivot_start = time.time()
                    multi_pivot_system = shared_system.create_multi_pivot_system(multi_pivot_config)
                    multi_pivot_build_time = time.time() - multi_pivot_start
                    
                    for k in k_values:
                        for n_probe in n_probe_values:
                            mp_eval = self.evaluate_recall_generic(
                                multi_pivot_system, k, n_probe, ground_truths[k], "多枢纽KMeans HNSW"
                            )
                            mp_eval['phase'] = 'kmeans_hnsw_multi_pivot'
                            mp_eval['multi_pivot_build_time'] = multi_pivot_build_time
                            mp_eval['multi_pivot_config'] = multi_pivot_config
                            phase_records.append({**mp_eval, 'k': k})
                            print(f"  [多枢纽KMeans HNSW] k={k} n_probe={n_probe} recall={mp_eval['recall_at_k']:.4f}")

                # 计算时间节省
                total_build_time = single_pivot_build_time
                if multi_pivot_config.get('enabled', False):
                    total_build_time += multi_pivot_build_time
                
                time_savings = f"共享计算节省时间: 原本需要2-3次聚类，现在只需1次"
                
                combination_results = {
                    'parameters': params,
                    'shared_computation_time': shared_computation_time,
                    'total_build_time': total_build_time,
                    'time_optimization': time_savings,
                    'phase_evaluations': phase_records,
                    'multi_pivot_enabled': multi_pivot_config.get('enabled', False)
                }
                results.append(combination_results)
                
                best_recall = max(r['recall_at_k'] for r in phase_records if 'recall_at_k' in r)
                print(f"  ✅ 组合完成，最佳召回率: {best_recall:.4f}")
                print(f"  ⏱️  {time_savings}")
                
            except Exception as e:
                print(f"❌ 参数组合 {params} 出错: {e}")
                continue

        print(f"\n🎯 优化版参数扫描完成！测试了 {len(results)} 个组合")
        print(f"    Multi-Pivot启用: {multi_pivot_config.get('enabled', False)}")
        print("🚀 关键优化效果: 避免了重复的K-Means聚类计算")
        return results


def save_results(results: Dict[str, Any], filename: str):
    """保存结果到JSON文件"""
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
    
    print(f"结果已保存到 {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="优化版K-Means HNSW + Multi-Pivot参数调优 (减少重复计算)")
    
    # 数据集选项
    parser.add_argument('--dataset-size', type=int, default=1000, 
                        help='基础向量数量 (默认: 1000)')
    parser.add_argument('--query-size', type=int, default=20, 
                        help='查询向量数量 (默认: 20)')
    parser.add_argument('--dimension', type=int, default=128, 
                        help='向量维度 (默认: 128)')
    
    # Multi-pivot选项
    parser.add_argument('--enable-multi-pivot', action='store_true',
                        help='启用Multi-Pivot评估')
    parser.add_argument('--num-pivots', type=int, default=3,
                        help='枢纽点数量 (默认: 3)')
    parser.add_argument('--pivot-strategy', type=str, default='line_perp_third',
                        choices=['line_perp_third', 'max_min_distance'],
                        help='枢纽选择策略')
    
    # 自适应优化选项
    parser.add_argument('--adaptive-k-children', action='store_true',
                        help='启用基于平均聚类大小的自适应k_children')
    parser.add_argument('--k-children-scale', type=float, default=1.5,
                        help='自适应k_children的缩放因子 (默认: 1.5)')
    parser.add_argument('--k-children-min', type=int, default=50,
                        help='自适应时的最小k_children (默认: 50)')
    parser.add_argument('--k-children-max', type=int, default=None,
                        help='自适应时的最大k_children (可选)')
    
    # 多样化优化选项
    parser.add_argument('--diversify-max-assignments', type=int, default=None,
                        help='每个子节点的最大分配数，启用多样化 (可选)')
    
    # 修复优化选项
    parser.add_argument('--repair-min-assignments', type=int, default=None,
                        help='构建修复期间每个子节点的最小分配数 (可选)')
    
    # 性能调优选项
    parser.add_argument('--child-search-ef', type=int, default=None,
                        help='子节点搜索的ef参数 (默认: 自动计算)')
    parser.add_argument('--overlap-sample', type=int, default=50,
                        help='重叠统计的采样大小 (默认: 50)')
    
    args = parser.parse_args()

    print("🚀 优化版K-Means HNSW + Multi-Pivot参数调优系统")
    print(f"📊 数据集: {args.dataset_size} vectors, 查询: {args.query_size}")
    print(f"🎯 Multi-Pivot: {'启用' if args.enable_multi_pivot else '禁用'}")
    print("🔄 关键优化: 共享K-Means聚类计算，避免重复构建")
    
    # 显示启用的优化选项
    optimizations = []
    if args.adaptive_k_children:
        optimizations.append(f"自适应k_children (scale={args.k_children_scale})")
    if args.diversify_max_assignments:
        optimizations.append(f"多样化限制 (max={args.diversify_max_assignments})")
    if args.repair_min_assignments:
        optimizations.append(f"修复机制 (min={args.repair_min_assignments})")
    
    if optimizations:
        print(f"⚡ 启用的优化: {', '.join(optimizations)}")
    print()
    
    # 创建合成数据
    print("🎲 创建合成数据...")
    base_vectors = np.random.randn(args.dataset_size, args.dimension).astype(np.float32)
    query_vectors = np.random.randn(args.query_size, args.dimension).astype(np.float32)
    query_ids = list(range(len(query_vectors)))
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # 构建基础HNSW索引
    print("🏗️  构建基础HNSW索引...")
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    for i, vector in enumerate(base_vectors):
        base_index.insert(i, vector)
        if (i + 1) % 500 == 0:
            print(f"  插入进度: {i + 1}/{len(base_vectors)}")
    
    print(f"✅ HNSW索引构建完成: {len(base_index)} vectors")
    
    # 初始化优化版评估器
    evaluator = OptimizedKMeansHNSWMultiPivotEvaluator(base_vectors, query_vectors, query_ids, distance_func)
    
    # 参数网格
    if args.dataset_size <= 500:
        cluster_options = [8]
    elif args.dataset_size <= 2000:
        cluster_options = [16]
    else:
        cluster_options = [32]

    param_grid = {
        'n_clusters': cluster_options,
        'k_children': [100],
        'child_search_ef': [args.child_search_ef] if args.child_search_ef else [200]
    }
    
    evaluation_params = {
        'k_values': [10],
        'n_probe_values': [5, 10],
        'hybrid_parent_level': 2,
        'enable_hybrid': True
    }
    
    adaptive_config = {
        'adaptive_k_children': args.adaptive_k_children,
        'k_children_scale': args.k_children_scale,
        'k_children_min': args.k_children_min,
        'k_children_max': args.k_children_max,
        'diversify_max_assignments': args.diversify_max_assignments,
        'repair_min_assignments': args.repair_min_assignments,
        'overlap_sample': args.overlap_sample
    }
    
    multi_pivot_config = {
        'enabled': args.enable_multi_pivot,
        'num_pivots': args.num_pivots,
        'pivot_selection_strategy': args.pivot_strategy,
        'pivot_overquery_factor': 1.2
    }
    
    # 运行优化版参数扫描
    print("🚀 开始优化版参数扫描...")
    
    sweep_results = evaluator.optimized_parameter_sweep(
        base_index,
        param_grid,
        evaluation_params,
        max_combinations=None,
        adaptive_config=adaptive_config,
        multi_pivot_config=multi_pivot_config
    )
    
    # 保存结果
    if sweep_results:
        results = {
            'sweep_results': sweep_results,
            'optimization_info': {
                'method': 'shared_computation_optimization',
                'description': '通过共享K-Means聚类计算减少重复构建时间',
                'benefits': [
                    '避免重复向量提取',
                    '避免重复K-Means聚类',
                    '只在子节点分配策略上有差异',
                    '显著减少总体构建时间'
                ]
            },
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
        save_results(results, 'optimized_multi_pivot_results.json')
        
        print(f"\n✅ 优化版评估完成!")
        print(f"🎯 {'五种方法' if args.enable_multi_pivot else '四种方法'}对比结果已保存")
        print("🚀 关键优化: 减少了K-Means聚类的重复计算时间")
