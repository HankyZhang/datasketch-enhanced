"""
单枢纽召回率下降问题验证测试
"""
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from method3.kmeans_hnsw import KMeansHNSW
from method3.tune_kmeans_hnsw_optimized import SharedKMeansHNSWSystem, OptimizedSinglePivotSystem
from hnsw.hnsw import HNSW

def test_single_pivot_comparison():
    """对比原始版本和优化版本的单枢纽性能"""
    print("🔬 单枢纽召回率对比测试")
    
    # 创建测试数据
    np.random.seed(42)
    n_data = 5000
    n_queries = 50
    dim = 128
    
    dataset = np.random.randn(n_data, dim).astype(np.float32)
    query_vectors = np.random.randn(n_queries, dim).astype(np.float32)
    query_ids = list(range(n_queries))
    
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # 构建基础HNSW索引
    print("构建基础HNSW索引...")
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    for i, vector in enumerate(dataset):
        base_index.insert(i, vector)
    
    # 测试参数
    test_params = {
        'n_clusters': 32,
        'k_children': 200,
        'child_search_ef': 300
    }
    
    print(f"\n测试参数: {test_params}")
    
    # === 测试1: 原始版本（无优化） ===
    print("\n=== 测试原始版本（无diversify/repair）===")
    original_system = KMeansHNSW(
        base_index=base_index,
        n_clusters=test_params['n_clusters'],
        k_children=test_params['k_children'],
        child_search_ef=test_params['child_search_ef'],
        diversify_max_assignments=None,  # 禁用diversify
        repair_min_assignments=None      # 禁用repair
    )
    
    # 收集原始版本统计
    original_stats = original_system.get_stats()
    print(f"原始版本统计:")
    print(f"  - 子节点总数: {original_stats['num_children']}")
    print(f"  - 平均子节点/质心: {original_stats['avg_children_per_centroid']:.1f}")
    print(f"  - 覆盖率: {original_stats['coverage_fraction']:.3f}")
    
    # === 测试2: 优化版本（无diversify/repair）===
    print("\n=== 测试优化版本（无diversify/repair）===")
    
    params = {
        'n_clusters': test_params['n_clusters'],
        'k_children': test_params['k_children'],
        'child_search_ef': test_params['child_search_ef']
    }
    
    adaptive_config_clean = {
        'diversify_max_assignments': None,  # 禁用diversify
        'repair_min_assignments': None      # 禁用repair
    }
    
    shared_system = SharedKMeansHNSWSystem(
        base_index=base_index,
        params=params,
        adaptive_config=adaptive_config_clean
    )
    
    optimized_system_clean = OptimizedSinglePivotSystem(
        shared_system=shared_system,
        adaptive_config=adaptive_config_clean
    )
    
    optimized_stats_clean = optimized_system_clean.get_stats()
    print(f"优化版本统计（无优化）:")
    print(f"  - 子节点总数: {optimized_stats_clean['num_children']}")
    print(f"  - 平均子节点/质心: {optimized_stats_clean['num_children'] / test_params['n_clusters']:.1f}")
    
    # === 测试3: 优化版本（带diversify）===
    print("\n=== 测试优化版本（带diversify限制）===")
    adaptive_config_diversify = {
        'diversify_max_assignments': 3,  # 启用diversify
        'repair_min_assignments': None
    }
    
    # 需要重新创建shared_system，因为adaptive_config在初始化时就固定了
    shared_system_diversify = SharedKMeansHNSWSystem(
        base_index=base_index,
        params=params,
        adaptive_config=adaptive_config_diversify
    )
    
    optimized_system_diversify = OptimizedSinglePivotSystem(
        shared_system=shared_system_diversify,
        adaptive_config=adaptive_config_diversify
    )
    
    optimized_stats_diversify = optimized_system_diversify.get_stats()
    print(f"优化版本统计（带diversify=3）:")
    print(f"  - 子节点总数: {optimized_stats_diversify['num_children']}")
    print(f"  - 平均子节点/质心: {optimized_stats_diversify['num_children'] / test_params['n_clusters']:.1f}")
    
    # === 召回率测试 ===
    print("\n=== 召回率对比测试 ===")
    k = 10
    n_probe = 5
    
    def evaluate_recall(system, name):
        """简单的召回率评估"""
        print(f"\n评估 {name}...")
        
        # 计算ground truth (简化版)
        total_correct = 0
        total_queries = min(10, len(query_vectors))  # 只测试前10个查询以节省时间
        
        for i in range(total_queries):
            query_vector = query_vectors[i]
            
            # Ground truth: 暴力搜索
            gt_distances = []
            for j, data_vector in enumerate(dataset):
                dist = distance_func(query_vector, data_vector)
                gt_distances.append((dist, j))
            gt_distances.sort()
            gt_neighbors = {node_id for _, node_id in gt_distances[:k]}
            
            # 系统搜索结果
            results = system.search(query_vector, k=k, n_probe=n_probe)
            found_neighbors = {node_id for node_id, _ in results}
            
            # 计算召回率
            correct = len(gt_neighbors & found_neighbors)
            total_correct += correct
            
            if i < 3:  # 只打印前3个查询的详细信息
                print(f"  查询 {i}: 找到 {len(results)} 结果, {correct}/{k} 正确, 召回率={correct/k:.3f}")
        
        overall_recall = total_correct / (total_queries * k)
        print(f"  {name} 总体召回率: {overall_recall:.3f}")
        return overall_recall
    
    # 评估三个系统
    recall_original = evaluate_recall(original_system, "原始版本")
    recall_optimized_clean = evaluate_recall(optimized_system_clean, "优化版本(无优化)")
    recall_optimized_diversify = evaluate_recall(optimized_system_diversify, "优化版本(diversify=3)")
    
    # === 分析结果 ===
    print("\n" + "="*60)
    print("📊 对比分析结果")
    print("="*60)
    print(f"原始版本召回率:              {recall_original:.3f}")
    print(f"优化版本(无优化)召回率:      {recall_optimized_clean:.3f}")
    print(f"优化版本(diversify=3)召回率:  {recall_optimized_diversify:.3f}")
    
    print(f"\n子节点数量对比:")
    print(f"原始版本:              {original_stats['num_children']}")
    print(f"优化版本(无优化):      {optimized_stats_clean['num_children']}")
    print(f"优化版本(diversify=3): {optimized_stats_diversify['num_children']}")
    
    # 分析差异
    if recall_optimized_clean < recall_original:
        print(f"\n⚠️  发现问题: 即使不启用diversify，优化版本召回率仍然较低")
        print(f"   召回率差异: {recall_original - recall_optimized_clean:.3f}")
        print(f"   可能原因: 向量获取方式差异或其他实现差异")
    
    if recall_optimized_diversify < recall_optimized_clean:
        print(f"\n⚠️  确认问题: diversify参数显著降低了召回率")
        print(f"   召回率下降: {recall_optimized_clean - recall_optimized_diversify:.3f}")
        print(f"   子节点减少: {optimized_stats_clean['num_children'] - optimized_stats_diversify['num_children']}")

if __name__ == "__main__":
    test_single_pivot_comparison()
