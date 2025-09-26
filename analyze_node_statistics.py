"""
节点统计分析 - 比较不同方法的节点分配情况
分析KMeansHNSW、KMeansHNSW Multi-Pivot、HybridHNSW在repair前后的节点统计
"""
import numpy as np
import sys
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from method3.kmeans_hnsw import KMeansHNSW
from method3.kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot
from method3.tune_kmeans_hnsw import KMeansHNSWEvaluator
from hnsw.hnsw import HNSW
from hybrid_hnsw.hnsw_hybrid import HNSWHybrid

def analyze_node_statistics():
    """详细分析不同系统的节点分配统计"""
    print("📊 节点统计分析 - 比较不同方法的节点分配情况")
    print("="*70)
    
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
    print("🏗️ 构建基础HNSW索引...")
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    for i, vector in enumerate(dataset):
        base_index.insert(i, vector)
    
    print(f"✅ 基础HNSW索引: {len(base_index)} 个节点")
    
    # 测试参数
    test_params = {
        'n_clusters': 32,
        'k_children': 200,
        'child_search_ef': 300
    }
    
    print(f"\n🔧 测试参数: {test_params}")
    print(f"📊 数据集大小: {n_data}, 查询数量: {n_queries}")
    
    # === 1. KMeansHNSW (原始单枢纽) ===
    print(f"\n{'='*70}")
    print("1️⃣ KMeansHNSW (原始单枢纽系统)")
    print("="*70)
    
    # 无repair版本
    print("\n📋 测试: 无repair/diversify")
    kmeans_hnsw_clean = KMeansHNSW(
        base_index=base_index,
        n_clusters=test_params['n_clusters'],
        k_children=test_params['k_children'],
        child_search_ef=test_params['child_search_ef'],
        diversify_max_assignments=None,
        repair_min_assignments=None
    )
    
    stats_clean = analyze_system_nodes(kmeans_hnsw_clean, "KMeansHNSW (无repair)")
    
    # 带repair版本
    print("\n📋 测试: 带repair=1")
    kmeans_hnsw_repair = KMeansHNSW(
        base_index=base_index,
        n_clusters=test_params['n_clusters'],
        k_children=test_params['k_children'],
        child_search_ef=test_params['child_search_ef'],
        diversify_max_assignments=None,
        repair_min_assignments=1
    )
    
    stats_repair = analyze_system_nodes(kmeans_hnsw_repair, "KMeansHNSW (repair=1)")
    
    # === 2. KMeansHNSW Multi-Pivot ===
    print(f"\n{'='*70}")
    print("2️⃣ KMeansHNSW Multi-Pivot (多枢纽系统)")
    print("="*70)
    
    # 无repair版本
    print("\n📋 测试: 无repair/diversify")
    try:
        multi_pivot_clean = KMeansHNSWMultiPivot(
            base_index=base_index,
            n_clusters=test_params['n_clusters'],
            k_children=test_params['k_children'],
            child_search_ef=test_params['child_search_ef'],
            num_pivots=3,
            pivot_selection_strategy='line_perp_third',
            diversify_max_assignments=None,
            repair_min_assignments=None
        )
        
        multi_stats_clean = analyze_system_nodes(multi_pivot_clean, "Multi-Pivot (无repair)")
    except Exception as e:
        print(f"❌ Multi-Pivot (无repair) 构建失败: {e}")
        multi_stats_clean = None
    
    # 带repair版本
    print("\n📋 测试: 带repair=1")
    try:
        multi_pivot_repair = KMeansHNSWMultiPivot(
            base_index=base_index,
            n_clusters=test_params['n_clusters'],
            k_children=test_params['k_children'],
            child_search_ef=test_params['child_search_ef'],
            num_pivots=3,
            pivot_selection_strategy='line_perp_third',
            diversify_max_assignments=None,
            repair_min_assignments=1
        )
        
        multi_stats_repair = analyze_system_nodes(multi_pivot_repair, "Multi-Pivot (repair=1)")
    except Exception as e:
        print(f"❌ Multi-Pivot (repair=1) 构建失败: {e}")
        multi_stats_repair = None
    
    # === 3. Hybrid HNSW ===
    print(f"\n{'='*70}")
    print("3️⃣ Hybrid HNSW (层级系统)")
    print("="*70)
    
    try:
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=2,
            k_children=test_params['k_children'],
            child_search_ef=test_params['child_search_ef']
        )
        
        hybrid_stats = analyze_hybrid_nodes(hybrid_index, "Hybrid HNSW")
    except Exception as e:
        print(f"❌ Hybrid HNSW 构建失败: {e}")
        hybrid_stats = None
    
    # === 4. 召回率测试 ===
    print(f"\n{'='*70}")
    print("4️⃣ 召回率对比测试")
    print("="*70)
    
    evaluator = KMeansHNSWEvaluator(dataset, query_vectors, query_ids, distance_func)
    ground_truth = evaluator.compute_ground_truth(k=10, exclude_query_ids=False)
    
    recall_results = {}
    
    # 测试KMeansHNSW
    if kmeans_hnsw_clean:
        recall_clean = evaluator.evaluate_recall(kmeans_hnsw_clean, k=10, n_probe=5, ground_truth=ground_truth)
        recall_results['KMeansHNSW (无repair)'] = recall_clean['recall_at_k']
        print(f"📊 KMeansHNSW (无repair) 召回率: {recall_clean['recall_at_k']:.4f}")
    
    if kmeans_hnsw_repair:
        recall_repair = evaluator.evaluate_recall(kmeans_hnsw_repair, k=10, n_probe=5, ground_truth=ground_truth)
        recall_results['KMeansHNSW (repair=1)'] = recall_repair['recall_at_k']
        print(f"📊 KMeansHNSW (repair=1) 召回率: {recall_repair['recall_at_k']:.4f}")
    
    # 测试Multi-Pivot
    if multi_pivot_clean:
        recall_multi_clean = evaluator.evaluate_recall(multi_pivot_clean, k=10, n_probe=5, ground_truth=ground_truth)
        recall_results['Multi-Pivot (无repair)'] = recall_multi_clean['recall_at_k']
        print(f"📊 Multi-Pivot (无repair) 召回率: {recall_multi_clean['recall_at_k']:.4f}")
    
    if multi_pivot_repair:
        recall_multi_repair = evaluator.evaluate_recall(multi_pivot_repair, k=10, n_probe=5, ground_truth=ground_truth)
        recall_results['Multi-Pivot (repair=1)'] = recall_multi_repair['recall_at_k']
        print(f"📊 Multi-Pivot (repair=1) 召回率: {recall_multi_repair['recall_at_k']:.4f}")
    
    # 测试Hybrid HNSW
    if hybrid_index:
        recall_hybrid = evaluator.evaluate_hybrid_hnsw(hybrid_index, k=10, n_probe=5, ground_truth=ground_truth)
        recall_results['Hybrid HNSW'] = recall_hybrid['recall_at_k']
        print(f"📊 Hybrid HNSW 召回率: {recall_hybrid['recall_at_k']:.4f}")
    
    # === 5. 综合对比表格 ===
    print(f"\n{'='*70}")
    print("5️⃣ 综合统计对比表")
    print("="*70)
    
    create_comparison_table([
        (stats_clean, "KMeansHNSW (无repair)"),
        (stats_repair, "KMeansHNSW (repair=1)"),
        (multi_stats_clean, "Multi-Pivot (无repair)"),
        (multi_stats_repair, "Multi-Pivot (repair=1)"),
        (hybrid_stats, "Hybrid HNSW")
    ], recall_results, n_data)

def analyze_system_nodes(system, name: str) -> Dict[str, Any]:
    """分析系统的节点分配统计"""
    print(f"\n🔍 分析 {name}...")
    
    # 获取基本统计
    stats = system.get_stats()
    total_assigned_nodes = stats.get('num_children', 0)
    coverage_fraction = stats.get('coverage_fraction', 0.0)
    avg_children_per_centroid = stats.get('avg_children_per_centroid', 0.0)
    
    # 分析去重情况
    all_assigned_nodes = set()
    duplicate_count = 0
    total_assignments = 0
    
    if hasattr(system, 'parent_child_map'):
        for centroid_id, children in system.parent_child_map.items():
            for child_id in children:
                total_assignments += 1
                if child_id in all_assigned_nodes:
                    duplicate_count += 1
                else:
                    all_assigned_nodes.add(child_id)
    
    unique_assigned_nodes = len(all_assigned_nodes)
    duplication_rate = duplicate_count / total_assignments if total_assignments > 0 else 0.0
    
    result = {
        'name': name,
        'total_assignments': total_assignments,
        'unique_nodes': unique_assigned_nodes,
        'duplicate_assignments': duplicate_count,
        'duplication_rate': duplication_rate,
        'coverage_fraction': coverage_fraction,
        'avg_children_per_centroid': avg_children_per_centroid,
        'reported_num_children': total_assigned_nodes
    }
    
    print(f"  📊 总分配数: {total_assignments}")
    print(f"  🎯 去重后节点数: {unique_assigned_nodes}")
    print(f"  🔄 重复分配数: {duplicate_count}")
    print(f"  📈 重复率: {duplication_rate:.3f}")
    print(f"  📐 覆盖率: {coverage_fraction:.3f}")
    print(f"  ⚖️ 平均子节点/质心: {avg_children_per_centroid:.1f}")
    
    return result

def analyze_hybrid_nodes(system, name: str) -> Dict[str, Any]:
    """分析Hybrid HNSW的节点统计"""
    print(f"\n🔍 分析 {name}...")
    
    try:
        stats = system.get_stats()
        total_assigned_nodes = stats.get('num_children', 0)
        coverage_fraction = stats.get('coverage_fraction', 0.0)
        
        # Hybrid HNSW的特殊统计
        num_parents = stats.get('num_parents', 0)
        
        # 分析去重情况
        all_assigned_nodes = set()
        duplicate_count = 0
        total_assignments = 0
        
        if hasattr(system, 'parent_child_map'):
            for parent_id, children in system.parent_child_map.items():
                for child_id in children:
                    total_assignments += 1
                    if child_id in all_assigned_nodes:
                        duplicate_count += 1
                    else:
                        all_assigned_nodes.add(child_id)
        
        unique_assigned_nodes = len(all_assigned_nodes)
        duplication_rate = duplicate_count / total_assignments if total_assignments > 0 else 0.0
        
        result = {
            'name': name,
            'total_assignments': total_assignments,
            'unique_nodes': unique_assigned_nodes,
            'duplicate_assignments': duplicate_count,
            'duplication_rate': duplication_rate,
            'coverage_fraction': coverage_fraction,
            'num_parents': num_parents,
            'avg_children_per_parent': total_assignments / num_parents if num_parents > 0 else 0.0,
            'reported_num_children': total_assigned_nodes
        }
        
        print(f"  📊 父节点数: {num_parents}")
        print(f"  📊 总分配数: {total_assignments}")
        print(f"  🎯 去重后节点数: {unique_assigned_nodes}")
        print(f"  🔄 重复分配数: {duplicate_count}")
        print(f"  📈 重复率: {duplication_rate:.3f}")
        print(f"  📐 覆盖率: {coverage_fraction:.3f}")
        print(f"  ⚖️ 平均子节点/父节点: {result['avg_children_per_parent']:.1f}")
        
        return result
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return {
            'name': name,
            'error': str(e)
        }

def create_comparison_table(stats_list, recall_results, total_nodes):
    """创建对比表格"""
    print("\n📋 详细对比表格:")
    print("-" * 120)
    print(f"{'方法':<25} {'总分配':<8} {'去重后':<8} {'重复数':<8} {'重复率':<8} {'覆盖率':<8} {'召回率':<8}")
    print("-" * 120)
    
    for stats, name in stats_list:
        if stats and 'error' not in stats:
            recall = recall_results.get(name, 0.0)
            print(f"{name:<25} "
                  f"{stats['total_assignments']:<8} "
                  f"{stats['unique_nodes']:<8} "
                  f"{stats['duplicate_assignments']:<8} "
                  f"{stats['duplication_rate']:<8.3f} "
                  f"{stats['coverage_fraction']:<8.3f} "
                  f"{recall:<8.4f}")
    
    print("-" * 120)
    print(f"总数据集节点数: {total_nodes}")
    
    # 分析问题
    print(f"\n🔍 问题分析:")
    print(f"1. 覆盖率低可能导致召回率下降")
    print(f"2. 重复分配高表示效率浪费")
    print(f"3. repair机制应该提高覆盖率")
    print(f"4. Multi-Pivot理论上应该比单枢纽更好")

if __name__ == "__main__":
    analyze_node_statistics()
