"""
简化的K-Means HNSW性能比较工具
=================================

这是一个用于比较三种向量搜索方法性能的简化工具：
1. 纯HNSW索引 - 基于分层图的近似最近邻搜索
2. 纯K-Means聚类 - 基于聚类的向量搜索
3. K-Means HNSW混合系统 - 结合K-Means和HNSW的两阶段搜索

主要功能：
- 自动构建三种不同的索引结构
- 计算ground truth作为性能基准
- 评估召回率(recall@k)和查询响应时间
- 自动寻找最优参数组合
- 生成详细的性能比较报告

适用场景：
- 向量数据库性能评估
- 搜索算法选型
- 参数调优实验
- 学术研究和benchmarking

作者：基于tune_kmeans_hnsw.py简化而来
版本：1.0
"""

import os
import sys
import time
import json
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import MiniBatchKMeans

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from method3.kmeans_hnsw import KMeansHNSW
from hnsw.hnsw import HNSW


class SimpleComparator:
    """
    简化的三种方法性能比较器
    
    这个类封装了三种向量搜索方法的构建、评估和比较功能：
    1. HNSW - 分层可导航小世界图算法
    2. K-Means - 基于聚类的搜索算法  
    3. K-Means HNSW - 混合两阶段搜索算法
    
    使用流程：
    1. 初始化比较器（传入数据集和查询集）
    2. 调用run_full_comparison()执行完整比较
    3. 调用print_results()显示结果
    
    性能指标：
    - 召回率(Recall@K)：返回结果中真正近邻的比例
    - 查询时间：单次查询的平均响应时间
    - 构建时间：索引构建所需的时间
    """
    
    def __init__(self, dataset: np.ndarray, queries: np.ndarray, distance_func: callable):
        """
        初始化比较器
        
        Args:
            dataset: 数据集向量 [n_vectors, dim]
            queries: 查询向量 [n_queries, dim]
            distance_func: 距离函数
        """
        self.dataset = dataset
        self.queries = queries
        self.distance_func = distance_func
        self.dimension = dataset.shape[1]
        
        print(f"数据集: {len(dataset)}个向量, 查询: {len(queries)}个向量, 维度: {self.dimension}")
    
    def compute_ground_truth(self, k: int) -> List[List[int]]:
        """
        计算真实的最近邻作为基准(Ground Truth)
        
        使用暴力搜索计算每个查询向量的真实k近邻，作为评估其他算法性能的基准。
        这是最准确但计算复杂度最高的方法(O(n*m*d))，其中：
        - n: 数据集大小
        - m: 查询数量  
        - d: 向量维度
        
        Args:
            k: 需要返回的最近邻数量
            
        Returns:
            List[List[int]]: 每个查询对应的k个最近邻索引列表
            
        注意：
            这个过程可能很耗时，特别是对于大数据集。
            计算复杂度为O(n*m*d)，其中n是数据集大小，m是查询数量。
        """
        print(f"计算ground truth (k={k})...")
        start_time = time.time()
        
        ground_truth = []
        for i, query in enumerate(self.queries):
            distances = []
            for j, data_point in enumerate(self.dataset):
                dist = self.distance_func(query, data_point)
                distances.append((dist, j))
            
            # 按距离排序，取前k个最近邻
            distances.sort()
            ground_truth.append([idx for _, idx in distances[:k]])
            
            # 显示进度，避免用户等待焦虑
            if (i + 1) % 20 == 0:
                print(f"  已处理 {i + 1}/{len(self.queries)} 个查询")
        
        print(f"Ground truth计算完成，耗时: {time.time() - start_time:.2f}s")
        return ground_truth
    
    def build_hnsw(self, m: int = 16, ef_construction: int = 200) -> HNSW:
        """
        构建HNSW(分层可导航小世界图)索引
        
        HNSW是一种基于图的近似最近邻搜索算法，通过构建多层图结构
        来实现高效的向量搜索。算法特点：
        - 搜索时间复杂度：O(log n)
        - 空间复杂度：O(n * M)
        - 高召回率和快速查询速度
        
        Args:
            m: 每个节点的最大连接数，影响图的连通性和查询精度
               - 较大的m值：更高精度，但更多内存消耗
               - 较小的m值：更少内存，但可能降低精度
               - 典型值：8-48
            ef_construction: 构建时的候选队列大小
               - 较大值：构建质量更好，但耗时更长
               - 较小值：构建更快，但质量可能下降
               - 典型值：100-800
        
        Returns:
            HNSW: 构建完成的HNSW索引对象
        """
        print(f"构建HNSW索引 (m={m}, ef_construction={ef_construction})...")
        start_time = time.time()
        
        hnsw = HNSW(distance_func=self.distance_func, m=m, ef_construction=ef_construction)
        
        for i, vector in enumerate(self.dataset):
            hnsw.insert(i, vector)
            # 显示构建进度，让用户了解当前状态
            if (i + 1) % 1000 == 0:
                print(f"  已插入 {i + 1}/{len(self.dataset)} 个向量")
        
        build_time = time.time() - start_time
        print(f"HNSW索引构建完成，耗时: {build_time:.2f}s")
        return hnsw
    
    def build_pure_kmeans(self, n_clusters: int = 64) -> MiniBatchKMeans:
        """
        构建纯K-Means聚类模型
        
        K-Means是一种无监督聚类算法，将数据分为k个簇。在向量搜索中：
        1. 训练阶段：将数据集聚类为k个簇
        2. 查询阶段：找到查询向量最近的簇，在该簇内进行精确搜索
        
        使用MiniBatchKMeans而非标准KMeans的原因：
        - 更适合大数据集
        - 内存效率更高
        - 训练速度更快
        - 精度损失很小
        
        Args:
            n_clusters: 聚类中心的数量
                - 影响搜索精度和速度的权衡
                - 太少：每个簇包含太多点，搜索慢
                - 太多：可能过拟合，查询质量下降
                - 经验值：sqrt(n) 到 n/100 之间
        
        Returns:
            MiniBatchKMeans: 训练完成的K-Means模型
        """
        print(f"构建K-Means聚类 (n_clusters={n_clusters})...")
        start_time = time.time()
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(1024, len(self.dataset)),
            n_init=3,
            max_iter=100,
            verbose=0
        )
        kmeans.fit(self.dataset)
        
        build_time = time.time() - start_time
        print(f"K-Means聚类完成，耗时: {build_time:.2f}s")
        return kmeans
    
    def build_kmeans_hnsw(self, hnsw: HNSW, n_clusters: int = 64, 
                         k_children: int = 200, child_search_ef: int = 300) -> KMeansHNSW:
        """
        构建K-Means HNSW混合系统
        
        这是一个创新的两阶段搜索系统，结合了K-Means和HNSW的优势：
        
        第一阶段 - 粗搜索(K-Means)：
        - 使用K-Means聚类将数据空间划分
        - 快速定位查询向量可能所在的区域
        - 时间复杂度：O(k*d)，其中k是聚类数
        
        第二阶段 - 精搜索(HNSW)：
        - 在选定的聚类区域内使用HNSW进行精确搜索
        - 避免在整个数据集上搜索，提高效率
        - 保持HNSW的高精度特性
        
        Args:
            hnsw: 已构建的基础HNSW索引
            n_clusters: K-Means聚类数量
                - 影响第一阶段的搜索速度和精度
            k_children: 每个聚类中心关联的子节点数量
                - 控制第二阶段的搜索范围
                - 较大值：更高召回率，但搜索时间增加
            child_search_ef: 子节点搜索时的ef参数
                - 控制HNSW搜索的精度
        
        Returns:
            KMeansHNSW: 构建完成的混合系统
        """
        print(f"构建K-Means HNSW混合系统 (n_clusters={n_clusters}, k_children={k_children})...")
        start_time = time.time()
        
        kmeans_hnsw = KMeansHNSW(
            base_index=hnsw,
            n_clusters=n_clusters,
            k_children=k_children,
            child_search_ef=child_search_ef
        )
        
        build_time = time.time() - start_time
        print(f"K-Means HNSW混合系统构建完成，耗时: {build_time:.2f}s")
        return kmeans_hnsw
    
    def evaluate_hnsw(self, hnsw: HNSW, ground_truth: List[List[int]], 
                     k: int, ef: int = 100) -> Dict[str, Any]:
        """评估HNSW性能"""
        print(f"评估HNSW性能 (k={k}, ef={ef})...")
        
        total_correct = 0
        query_times = []
        
        for i, (query, true_neighbors) in enumerate(zip(self.queries, ground_truth)):
            true_ids = set(true_neighbors)
            
            start_time = time.time()
            results = hnsw.query(query, k=k, ef=ef)
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            found_ids = {node_id for node_id, _ in results}
            correct = len(true_ids & found_ids)
            total_correct += correct
        
        recall = total_correct / (len(self.queries) * k)
        avg_time = np.mean(query_times) * 1000  # 转换为毫秒
        
        return {
            'method': 'HNSW',
            'recall': recall,
            'avg_query_time_ms': avg_time,
            'total_correct': total_correct,
            'total_expected': len(self.queries) * k,
            'ef': ef
        }
    
    def evaluate_pure_kmeans(self, kmeans: MiniBatchKMeans, ground_truth: List[List[int]], 
                           k: int, n_probe: int = 5) -> Dict[str, Any]:
        """评估纯K-Means性能"""
        print(f"评估纯K-Means性能 (k={k}, n_probe={n_probe})...")
        
        # 构建聚类到数据点的映射
        clusters = [[] for _ in range(kmeans.n_clusters)]
        for idx, cluster_id in enumerate(kmeans.labels_):
            clusters[cluster_id].append(idx)
        
        total_correct = 0
        query_times = []
        
        n_probe_eff = min(n_probe, kmeans.n_clusters)
        
        for i, (query, true_neighbors) in enumerate(zip(self.queries, ground_truth)):
            true_ids = set(true_neighbors)
            
            start_time = time.time()
            
            # 找到最近的n_probe个聚类中心
            distances_to_centers = np.linalg.norm(kmeans.cluster_centers_ - query, axis=1)
            closest_clusters = np.argpartition(distances_to_centers, n_probe_eff - 1)[:n_probe_eff]
            
            # 收集这些聚类中的所有点
            candidate_ids = []
            for cluster_id in closest_clusters:
                candidate_ids.extend(clusters[cluster_id])
            
            if candidate_ids:
                # 计算到候选点的距离
                candidate_vectors = self.dataset[candidate_ids]
                distances = np.linalg.norm(candidate_vectors - query, axis=1)
                
                # 排序并取前k个
                sorted_indices = np.argsort(distances)[:k]
                found_ids = {candidate_ids[idx] for idx in sorted_indices}
            else:
                found_ids = set()
            
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            correct = len(true_ids & found_ids)
            total_correct += correct
        
        recall = total_correct / (len(self.queries) * k)
        avg_time = np.mean(query_times) * 1000  # 转换为毫秒
        
        return {
            'method': 'Pure K-Means',
            'recall': recall,
            'avg_query_time_ms': avg_time,
            'total_correct': total_correct,
            'total_expected': len(self.queries) * k,
            'n_probe': n_probe_eff,
            'n_clusters': kmeans.n_clusters
        }
    
    def evaluate_kmeans_hnsw(self, kmeans_hnsw: KMeansHNSW, 
                           ground_truth: List[List[int]], 
                           k: int, n_probe: int = 5) -> Dict[str, Any]:
        """评估K-Means HNSW性能"""
        print(f"评估K-Means HNSW性能 (k={k}, n_probe={n_probe})...")
        
        total_correct = 0
        query_times = []
        
        for i, (query, true_neighbors) in enumerate(zip(self.queries, ground_truth)):
            true_ids = set(true_neighbors)
            
            start_time = time.time()
            results = kmeans_hnsw.search(query, k=k, n_probe=n_probe)
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            found_ids = {node_id for node_id, _ in results}
            correct = len(true_ids & found_ids)
            total_correct += correct
        
        recall = total_correct / (len(self.queries) * k)
        avg_time = np.mean(query_times) * 1000  # 转换为毫秒
        
        return {
            'method': 'K-Means HNSW',
            'recall': recall,
            'avg_query_time_ms': avg_time,
            'total_correct': total_correct,
            'total_expected': len(self.queries) * k,
            'n_probe': n_probe,
            'system_stats': kmeans_hnsw.get_stats()
        }
    
    def find_best_parameters(self, hnsw: HNSW, n_clusters_options: List[int] = None) -> Dict[str, Any]:
        """
        寻找最优参数组合
        
        这个方法会测试不同的聚类数量(n_clusters)和探测数量(n_probe)组合，
        通过评估召回率来找到最优的参数设置。
        
        优化策略：
        1. 根据数据集大小自动选择聚类数范围
        2. 对每个聚类数测试多个n_probe值
        3. 选择召回率最高的参数组合
        
        参数选择经验：
        - 小数据集(≤2000): 聚类数10-20
        - 中等数据集(≤5000): 聚类数32-64  
        - 大数据集(>5000): 聚类数64-128
        
        Args:
            hnsw: 已构建的HNSW索引
            n_clusters_options: 可选的聚类数列表，如果为None则自动选择
            
        Returns:
            Dict[str, Any]: 包含最优参数和所有测试结果的字典
        """
        if n_clusters_options is None:
            # 根据数据集大小自动选择聚类数
            if len(self.dataset) <= 2000:
                n_clusters_options = [10, 20]
            elif len(self.dataset) <= 5000:
                n_clusters_options = [32, 64]
            else:
                n_clusters_options = [64, 128]
        
        print(f"开始参数优化，测试聚类数: {n_clusters_options}")
        
        best_params = None
        best_recall = 0.0
        results_summary = []
        
        # 计算ground truth (用于优化的k值)
        k_for_optimization = 10
        ground_truth = self.compute_ground_truth(k_for_optimization)
        
        for n_clusters in n_clusters_options:
            print(f"\n测试 n_clusters={n_clusters}")
            
            # 构建K-Means HNSW系统
            kmeans_hnsw = self.build_kmeans_hnsw(hnsw, n_clusters=n_clusters)
            
            # 评估不同n_probe值
            for n_probe in [5, 10]:
                result = self.evaluate_kmeans_hnsw(kmeans_hnsw, ground_truth, k_for_optimization, n_probe)
                result['n_clusters'] = n_clusters
                results_summary.append(result)
                
                # 更新最佳参数
                if result['recall'] > best_recall:
                    best_recall = result['recall']
                    best_params = {
                        'n_clusters': n_clusters,
                        'n_probe': n_probe,
                        'recall': result['recall'],
                        'avg_query_time_ms': result['avg_query_time_ms']
                    }
                
                print(f"  n_probe={n_probe}: recall={result['recall']:.4f}, "
                      f"time={result['avg_query_time_ms']:.2f}ms")
        
        print(f"\n最优参数: {best_params}")
        
        return {
            'best_params': best_params,
            'all_results': results_summary
        }
    
    def run_full_comparison(self, k: int = 10, n_clusters: int = 64) -> Dict[str, Any]:
        """
        运行完整的三方法比较实验
        
        这是主要的评估函数，执行完整的性能比较流程：
        
        1. 数据准备阶段：
           - 计算ground truth（真实最近邻）
           
        2. 索引构建阶段：
           - 构建HNSW索引
           - 训练K-Means模型
           - 构建K-Means HNSW混合系统
           
        3. 参数优化阶段：
           - 自动寻找最优参数组合
           - 基于召回率进行优化
           
        4. 性能评估阶段：
           - 测试不同参数下的性能
           - 收集召回率和查询时间数据
           
        Args:
            k: 返回的最近邻数量，影响召回率计算
            n_clusters: K-Means聚类数量，影响聚类质量
            
        Returns:
            Dict[str, Any]: 包含所有评估结果的完整报告
            
        时间复杂度：
            - Ground truth: O(n*m*d)  
            - HNSW构建: O(n*log(n)*d)
            - K-Means训练: O(k*n*d*i)
            其中 n=数据集大小, m=查询数量, d=维度, k=聚类数, i=迭代次数
        """
        print("=" * 60)
        print("开始三种方法的完整性能比较")
        print("=" * 60)
        
        # 1. 计算ground truth
        ground_truth = self.compute_ground_truth(k)
        
        # 2. 构建所有索引
        hnsw = self.build_hnsw()
        kmeans = self.build_pure_kmeans(n_clusters)
        
        # 3. 参数优化
        optimization_result = self.find_best_parameters(hnsw, [n_clusters])
        best_params = optimization_result['best_params']
        
        # 4. 使用最优参数构建最终系统
        kmeans_hnsw = self.build_kmeans_hnsw(hnsw, 
                                           n_clusters=best_params['n_clusters'],
                                           k_children=200, 
                                           child_search_ef=300)
        
        # 5. 最终性能评估
        results = []
        
        # HNSW评估 (多个ef值)
        for ef in [50, 100, 200]:
            result = self.evaluate_hnsw(hnsw, ground_truth, k, ef)
            results.append(result)
        
        # 纯K-Means评估 (多个n_probe值)
        for n_probe in [1, 5, 10]:
            result = self.evaluate_pure_kmeans(kmeans, ground_truth, k, n_probe)
            results.append(result)
        
        # K-Means HNSW评估 (多个n_probe值)
        for n_probe in [1, 5, 10]:
            result = self.evaluate_kmeans_hnsw(kmeans_hnsw, ground_truth, k, n_probe)
            results.append(result)
        
        return {
            'results': results,
            'optimization': optimization_result,
            'config': {
                'k': k,
                'n_clusters': n_clusters,
                'dataset_size': len(self.dataset),
                'query_size': len(self.queries),
                'dimension': self.dimension
            }
        }
    
    def print_results(self, comparison_results: Dict[str, Any]):
        """
        打印格式化的比较结果
        
        以用户友好的方式展示性能比较结果，包括：
        - 实验配置信息
        - 各算法在不同参数下的表现
        - 最佳性能算法推荐
        
        输出格式：
        - 按算法分组显示结果
        - 突出显示关键性能指标
        - 提供最佳选择建议
        
        Args:
            comparison_results: run_full_comparison()返回的结果字典
            
        显示信息包括：
        - 召回率：算法准确性的核心指标
        - 查询时间：算法效率的重要指标  
        - 参数设置：帮助理解性能差异
        """
        print("\n" + "=" * 80)
        print("性能比较结果")
        print("=" * 80)
        
        results = comparison_results['results']
        config = comparison_results['config']
        
        print(f"配置: k={config['k']}, 聚类数={config['n_clusters']}")
        print(f"数据集大小: {config['dataset_size']}, 查询数量: {config['query_size']}")
        print()
        
        # 按方法分组显示
        methods = {}
        for result in results:
            method = result['method']
            if method not in methods:
                methods[method] = []
            methods[method].append(result)
        
        for method_name, method_results in methods.items():
            print(f"{method_name}:")
            for result in method_results:
                param_str = ""
                if 'ef' in result:
                    param_str = f"ef={result['ef']}"
                elif 'n_probe' in result:
                    param_str = f"n_probe={result['n_probe']}"
                
                print(f"  {param_str:12} 召回率: {result['recall']:.4f}, "
                      f"平均查询时间: {result['avg_query_time_ms']:.2f}ms")
            print()
        
        # 显示最佳结果和算法推荐
        best_result = max(results, key=lambda x: x['recall'])
        print(f"🏆 最佳召回率: {best_result['method']} - {best_result['recall']:.4f} "
              f"(查询时间: {best_result['avg_query_time_ms']:.2f}ms)")
        
        # 提供算法选择建议
        print(f"\n💡 算法选择建议:")
        fastest = min(results, key=lambda x: x['avg_query_time_ms'])
        print(f"   速度优先: {fastest['method']} ({fastest['avg_query_time_ms']:.2f}ms)")
        print(f"   精度优先: {best_result['method']} (召回率 {best_result['recall']:.4f})")
        
        # 计算平衡性推荐（召回率 × 速度倒数的加权）
        balanced_scores = []
        for result in results:
            if result['avg_query_time_ms'] > 0:  # 避免除零
                balance_score = result['recall'] * (1000 / result['avg_query_time_ms'])
                balanced_scores.append((balance_score, result))
        
        if balanced_scores:
            best_balanced = max(balanced_scores, key=lambda x: x[0])[1]
            print(f"   平衡推荐: {best_balanced['method']} "
                  f"(召回率 {best_balanced['recall']:.4f}, "
                  f"时间 {best_balanced['avg_query_time_ms']:.2f}ms)")


def load_sift_data(max_base: int = 10000, max_query: int = 100):
    """
    加载SIFT数据集
    
    SIFT(Scale-Invariant Feature Transform)是计算机视觉领域的经典特征描述子，
    常用于向量搜索算法的性能评估。该数据集包含：
    - 基础向量：用于构建索引的向量集合
    - 查询向量：用于测试搜索性能的查询集合
    - 128维浮点向量
    
    数据格式：
    - .fvecs文件：FAISS标准格式
    - 每个向量前有一个int32表示维度
    - 随后是dim个float32数值
    
    Args:
        max_base: 最大基础向量数量，用于控制内存使用
        max_query: 最大查询向量数量，用于控制测试规模
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (基础向量, 查询向量) 或 (None, None)
        
    注意：
        如果SIFT文件不存在或读取失败，函数会返回None，
        调用者应该fallback到合成数据。
    """
    sift_dir = os.path.join(os.path.dirname(__file__), '..', 'sift')
    
    try:
        def read_fvecs(path: str, max_vectors: int = None) -> np.ndarray:
            """读取.fvecs文件"""
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            raw = np.fromfile(path, dtype=np.int32)
            if raw.size == 0:
                raise ValueError(f"Empty file: {path}")
            dim = raw[0]
            record_size = dim + 1
            count = raw.size // record_size
            raw = raw.reshape(count, record_size)
            vecs = raw[:, 1:].astype(np.float32)
            if max_vectors and count > max_vectors:
                vecs = vecs[:max_vectors]
            return vecs
        
        base_path = os.path.join(sift_dir, 'sift_base.fvecs')
        query_path = os.path.join(sift_dir, 'sift_query.fvecs')
        
        base_vectors = read_fvecs(base_path, max_base)
        query_vectors = read_fvecs(query_path, max_query)
        
        print(f"成功加载SIFT数据: {len(base_vectors)}个基础向量, {len(query_vectors)}个查询向量")
        return base_vectors, query_vectors
        
    except Exception as e:
        print(f"加载SIFT数据失败: {e}")
        return None, None


def create_synthetic_data(n_base: int = 10000, n_query: int = 100, dim: int = 128):
    """
    创建合成数据集
    
    当SIFT数据不可用时，生成随机的高斯分布向量数据。
    合成数据的特点：
    - 使用标准正态分布生成
    - 固定随机种子确保可重现性
    - 向量间距离符合高维空间的统计特性
    
    优点：
    - 不依赖外部数据文件
    - 可以灵活控制数据规模和维度
    - 生成速度快
    
    局限：
    - 缺乏真实数据的复杂分布特性
    - 可能无法反映实际应用场景
    
    Args:
        n_base: 基础向量数量
        n_query: 查询向量数量  
        dim: 向量维度
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (基础向量, 查询向量)
    """
    print(f"创建合成数据: {n_base}个基础向量, {n_query}个查询向量, 维度={dim}")
    
    np.random.seed(42)
    base_vectors = np.random.randn(n_base, dim).astype(np.float32)
    query_vectors = np.random.randn(n_query, dim).astype(np.float32)
    
    return base_vectors, query_vectors


def save_results(results: Dict[str, Any], filename: str):
    """
    保存结果到JSON文件
    
    将评估结果序列化并保存到文件，便于后续分析和比较。
    处理了numpy数据类型的序列化问题，确保JSON兼容性。
    
    保存的数据包括：
    - 所有算法的性能指标
    - 参数优化结果
    - 实验配置信息
    - 时间戳信息
    
    Args:
        results: 包含评估结果的字典
        filename: 保存文件名（建议使用.json扩展名）
        
    注意：
        函数会自动处理numpy数组和标量的类型转换，
        确保所有数据都能正确序列化为JSON格式。
    """
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
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="K-Means HNSW 性能比较工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python simple_comparison.py                          # 默认参数快速测试
  python simple_comparison.py --use-sift               # 使用SIFT数据集
  python simple_comparison.py --dataset-size 20000     # 指定数据集大小
  python simple_comparison.py --save-results           # 保存结果到文件
  
算法说明:
  HNSW        - 分层可导航小世界图，高精度但内存消耗大
  K-Means     - 基于聚类的搜索，速度快但精度中等  
  Mixed       - 混合算法，平衡精度和速度
        """)
    
    parser.add_argument('--dataset-size', type=int, default=5000, 
                       help='数据集大小（向量数量）')
    parser.add_argument('--query-size', type=int, default=50, 
                       help='查询向量数量')
    parser.add_argument('--dimension', type=int, default=128, 
                       help='向量维度(合成数据)')
    parser.add_argument('--k', type=int, default=10, 
                       help='返回的最近邻数量')
    parser.add_argument('--n-clusters', type=int, default=64, 
                       help='K-Means聚类数')
    parser.add_argument('--use-sift', action='store_true', 
                       help='使用SIFT数据集')
    parser.add_argument('--save-results', action='store_true', 
                       help='保存结果到文件')
    
    args = parser.parse_args()
    
    # 距离函数（欧几里得距离）
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # 加载数据
    if args.use_sift:
        print(f"\n📊 尝试加载SIFT数据集...")
        base_vectors, query_vectors = load_sift_data(args.dataset_size, args.query_size)
        if base_vectors is None:
            print("❌ SIFT数据加载失败，使用合成数据")
            base_vectors, query_vectors = create_synthetic_data(
                args.dataset_size, args.query_size, args.dimension
            )
        else:
            print("✅ SIFT数据加载成功")
    else:
        print(f"\n🎲 生成合成数据集...")
        base_vectors, query_vectors = create_synthetic_data(
            args.dataset_size, args.query_size, args.dimension
        )
    
    print(f"\n🔧 实验配置:")
    print(f"   数据集大小: {len(base_vectors):,} 向量")
    print(f"   查询数量: {len(query_vectors):,} 个")
    print(f"   向量维度: {base_vectors.shape[1]} 维")
    print(f"   召回率评估: Recall@{args.k}")
    print(f"   聚类数量: {args.n_clusters}")
    
    # 创建比较器实例并运行完整比较
    comparator = SimpleComparator(base_vectors, query_vectors, distance_func)
    results = comparator.run_full_comparison(k=args.k, n_clusters=args.n_clusters)
    comparator.print_results(results)
    
    # 保存结果到JSON文件（如果指定）
    if args.save_results:
        results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save_results(results, 'simple_comparison_results.json')
    
    print("\n🎉 算法比较完成！")
