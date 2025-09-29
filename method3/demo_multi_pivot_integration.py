#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Pivot集成使用示例
演示如何在tune_kmeans_hnsw_backup.py中使用Multi-Pivot功能
"""

import os
import sys
import numpy as np
import argparse

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from method3.tune_kmeans_hnsw_backup import KMeansHNSWEvaluator
from method3.kmeans_hnsw import KMeansHNSW
from hnsw.hnsw import HNSW

def demo_multi_pivot_integration():
    """演示Multi-Pivot集成功能"""
    print("=" * 60)
    print("Multi-Pivot集成演示")
    print("=" * 60)
    
    # 创建演示数据
    print("\n1. 创建演示数据...")
    np.random.seed(42)
    dataset_size = 200
    query_size = 20
    dimension = 64
    
    base_vectors = np.random.randn(dataset_size, dimension).astype(np.float32)
    query_vectors = np.random.randn(query_size, dimension).astype(np.float32)
    query_ids = list(range(query_size))
    
    print(f"   - 数据集大小: {dataset_size}")
    print(f"   - 查询数量: {query_size}")
    print(f"   - 向量维度: {dimension}")
    
    # 构建基础HNSW索引
    print("\n2. 构建基础HNSW索引...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=100)
    
    for i, vector in enumerate(base_vectors):
        base_index.insert(i, vector)
        if (i + 1) % 50 == 0:
            print(f"   - 已插入 {i + 1}/{dataset_size} 个向量")
    
    print(f"   - 基础索引构建完成，包含 {len(base_index)} 个向量")
    
    # 创建评估器
    print("\n3. 创建评估器...")
    evaluator = KMeansHNSWEvaluator(base_vectors, query_vectors, query_ids, distance_func)
    
    # 构建KMeansHNSW系统
    print("\n4. 构建KMeansHNSW系统...")
    kmeans_hnsw = KMeansHNSW(
        base_index=base_index,
        n_clusters=10,
        k_children=50,
        child_search_ef=80
    )
    print(f"   - KMeansHNSW构建完成，{kmeans_hnsw.n_clusters} 个聚类")
    
    # 计算ground truth
    print("\n5. 计算Ground Truth...")
    k_eval = 10
    try:
        # 暂时禁用中文输出以避免编码问题
        import sys
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        ground_truth = evaluator.compute_ground_truth(k=k_eval, exclude_query_ids=False)
        sys.stdout.close()
        sys.stdout = old_stdout
        print(f"   - Ground Truth计算完成 (k={k_eval})")
    except Exception as e:
        if old_stdout:
            sys.stdout = old_stdout
        print(f"   - Ground Truth计算失败: {e}")
        return
    
    # 运行不同的评估方法
    print("\n6. 运行多种评估方法...")
    n_probe = 5
    
    # 6.1 原始K-Means (单pivot)
    print(f"\n6.1 单pivot K-Means评估...")
    try:
        single_result = evaluator._evaluate_pure_kmeans_from_existing(
            kmeans_hnsw=kmeans_hnsw,
            k=k_eval,
            ground_truth=ground_truth,
            n_probe=n_probe
        )
        print(f"   - 召回率: {single_result['recall_at_k']:.4f}")
        print(f"   - 平均查询时间: {single_result['avg_query_time_ms']:.2f}ms")
    except Exception as e:
        print(f"   - 单pivot评估失败: {e}")
        return
    
    # 6.2 Multi-Pivot K-Means (2个pivot)
    print(f"\n6.2 Multi-Pivot K-Means评估 (2个pivot)...")
    try:
        mp2_result = evaluator._evaluate_multi_pivot_kmeans_from_existing(
            kmeans_hnsw=kmeans_hnsw,
            k=k_eval,
            ground_truth=ground_truth,
            n_probe=n_probe,
            num_pivots=2,
            pivot_selection_strategy='line_perp_third',
            pivot_overquery_factor=1.2
        )
        print(f"   - 召回率: {mp2_result['recall_at_k']:.4f}")
        print(f"   - 平均查询时间: {mp2_result['avg_query_time_ms']:.2f}ms")
        print(f"   - Pivot数量: {mp2_result['num_pivots']}")
        print(f"   - 选择策略: {mp2_result['pivot_selection_strategy']}")
    except Exception as e:
        print(f"   - Multi-Pivot (2个pivot) 评估失败: {e}")
        return
    
    # 6.3 Multi-Pivot K-Means (3个pivot)
    print(f"\n6.3 Multi-Pivot K-Means评估 (3个pivot)...")
    try:
        mp3_result = evaluator._evaluate_multi_pivot_kmeans_from_existing(
            kmeans_hnsw=kmeans_hnsw,
            k=k_eval,
            ground_truth=ground_truth,
            n_probe=n_probe,
            num_pivots=3,
            pivot_selection_strategy='line_perp_third',
            pivot_overquery_factor=1.3
        )
        print(f"   - 召回率: {mp3_result['recall_at_k']:.4f}")
        print(f"   - 平均查询时间: {mp3_result['avg_query_time_ms']:.2f}ms")
        print(f"   - Pivot数量: {mp3_result['num_pivots']}")
    except Exception as e:
        print(f"   - Multi-Pivot (3个pivot) 评估失败: {e}")
        return
    
    # 6.4 Multi-Pivot K-Means with max_min_distance strategy
    print(f"\n6.4 Multi-Pivot K-Means评估 (max_min_distance策略)...")
    try:
        mp_mmd_result = evaluator._evaluate_multi_pivot_kmeans_from_existing(
            kmeans_hnsw=kmeans_hnsw,
            k=k_eval,
            ground_truth=ground_truth,
            n_probe=n_probe,
            num_pivots=3,
            pivot_selection_strategy='max_min_distance',
            pivot_overquery_factor=1.2
        )
        print(f"   - 召回率: {mp_mmd_result['recall_at_k']:.4f}")
        print(f"   - 平均查询时间: {mp_mmd_result['avg_query_time_ms']:.2f}ms")
        print(f"   - 选择策略: {mp_mmd_result['pivot_selection_strategy']}")
    except Exception as e:
        print(f"   - Multi-Pivot (max_min_distance) 评估失败: {e}")
        return
    
    # 结果对比
    print("\n7. 结果对比分析...")
    print("-" * 60)
    print(f"{'方法':<25} {'召回率':<10} {'查询时间(ms)':<12} {'改进':<10}")
    print("-" * 60)
    
    base_recall = single_result['recall_at_k']
    base_time = single_result['avg_query_time_ms']
    
    methods = [
        ("单Pivot K-Means", single_result),
        ("Multi-Pivot (2个pivot)", mp2_result),
        ("Multi-Pivot (3个pivot)", mp3_result),
        ("Multi-Pivot (max_min_dist)", mp_mmd_result)
    ]
    
    for method_name, result in methods:
        recall = result['recall_at_k']
        time_ms = result['avg_query_time_ms']
        improvement = recall - base_recall
        print(f"{method_name:<25} {recall:<10.4f} {time_ms:<12.2f} {improvement:+.4f}")
    
    print("-" * 60)
    
    # 找出最佳方法
    best_method = max(methods[1:], key=lambda x: x[1]['recall_at_k'])  # 排除基线
    print(f"\n最佳Multi-Pivot方法: {best_method[0]}")
    print(f"召回率改进: {best_method[1]['recall_at_k'] - base_recall:+.4f}")
    
    print("\n=" * 60)
    print("Multi-Pivot集成演示完成!")
    print("=" * 60)

def demo_parameter_sweep_with_multi_pivot():
    """演示带Multi-Pivot的参数扫描"""
    print("\n" + "=" * 60)
    print("Multi-Pivot参数扫描演示")
    print("=" * 60)
    
    # 创建小规模数据用于快速演示
    print("\n1. 创建小规模演示数据...")
    np.random.seed(42)
    dataset_size = 100
    query_size = 10
    dimension = 32
    
    base_vectors = np.random.randn(dataset_size, dimension).astype(np.float32)
    query_vectors = np.random.randn(query_size, dimension).astype(np.float32)
    query_ids = list(range(query_size))
    
    # 构建基础HNSW索引
    print("\n2. 构建基础HNSW索引...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=8, ef_construction=50)
    
    for i, vector in enumerate(base_vectors):
        base_index.insert(i, vector)
    
    print(f"   - 基础索引构建完成")
    
    # 创建评估器
    evaluator = KMeansHNSWEvaluator(base_vectors, query_vectors, query_ids, distance_func)
    
    # 定义参数网格
    param_grid = {
        'n_clusters': [5],
        'k_children': [20],
        'child_search_ef': [30]
    }
    
    evaluation_params = {
        'k_values': [5],
        'n_probe_values': [3]
    }
    
    # 配置Multi-Pivot
    adaptive_config = {
        'adaptive_k_children': False,
        'k_children_scale': 1.5,
        'k_children_min': 10,
        'k_children_max': None,
        'diversify_max_assignments': None,
        'repair_min_assignments': None,
        'multi_pivot_config': {
            'enabled': True,
            'num_pivots': 2,
            'pivot_selection_strategy': 'line_perp_third',
            'pivot_overquery_factor': 1.2
        }
    }
    
    print("\n3. 运行参数扫描（包含Multi-Pivot）...")
    try:
        results = evaluator.parameter_sweep(
            base_index=base_index,
            param_grid=param_grid,
            evaluation_params=evaluation_params,
            max_combinations=1,
            adaptive_config=adaptive_config
        )
        
        if results:
            result = results[0]
            phase_evaluations = result['phase_evaluations']
            
            print(f"\n4. 参数扫描结果分析...")
            print("-" * 50)
            print(f"{'阶段':<20} {'召回率':<10} {'查询时间(ms)':<12}")
            print("-" * 50)
            
            for eval_result in phase_evaluations:
                phase = eval_result.get('phase', eval_result.get('method', 'unknown'))
                recall = eval_result.get('recall_at_k', 0)
                time_ms = eval_result.get('avg_query_time_ms', 0)
                print(f"{phase:<20} {recall:<10.4f} {time_ms:<12.2f}")
            
            print("-" * 50)
            
            # 检查是否包含Multi-Pivot结果
            multi_pivot_found = any('multi_pivot' in str(eval_result.get('phase', eval_result.get('method', ''))) 
                                  for eval_result in phase_evaluations)
            
            if multi_pivot_found:
                print("\n✓ Multi-Pivot阶段成功包含在参数扫描中!")
            else:
                print("\n✗ 未找到Multi-Pivot阶段")
        else:
            print("   - 参数扫描未返回结果")
            
    except Exception as e:
        print(f"   - 参数扫描失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Pivot集成演示")
    parser.add_argument('--demo-type', type=str, default='both', 
                       choices=['basic', 'sweep', 'both'],
                       help='演示类型: basic(基础演示), sweep(参数扫描), both(两者)')
    
    args = parser.parse_args()
    
    try:
        if args.demo_type in ['basic', 'both']:
            demo_multi_pivot_integration()
            
        if args.demo_type in ['sweep', 'both']:
            demo_parameter_sweep_with_multi_pivot()
            
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
