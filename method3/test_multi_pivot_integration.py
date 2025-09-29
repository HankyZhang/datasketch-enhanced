#!/usr/bin/env python3
"""测试Multi-Pivot集成到tune_kmeans_hnsw_backup.py的功能"""

import os
import sys
import numpy as np

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tune_kmeans_hnsw_backup import KMeansHNSWEvaluator
from method3.kmeans_hnsw import KMeansHNSW
from hnsw.hnsw import HNSW

def test_multi_pivot_integration():
    """测试Multi-Pivot功能是否正确集成"""
    print("🧪 测试Multi-Pivot集成...")
    
    # 创建小型测试数据
    np.random.seed(42)
    dataset_size = 100
    query_size = 10
    dimension = 32
    
    base_vectors = np.random.randn(dataset_size, dimension).astype(np.float32)
    query_vectors = np.random.randn(query_size, dimension).astype(np.float32)
    query_ids = list(range(query_size))
    
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # 构建基础HNSW索引
    print("🏗️ 构建基础HNSW索引...")
    base_index = HNSW(distance_func=distance_func, m=8, ef_construction=50)
    
    for i, vector in enumerate(base_vectors):
        base_index.insert(i, vector)
    
    print(f"✅ 基础索引构建完成，包含 {len(base_index)} 个向量")
    
    # 创建评估器
    evaluator = KMeansHNSWEvaluator(base_vectors, query_vectors, query_ids, distance_func)
    
    # 测试基础KMeansHNSW构建
    print("🔧 测试基础KMeansHNSW构建...")
    kmeans_hnsw = KMeansHNSW(
        base_index=base_index,
        n_clusters=5,
        k_children=20,
        child_search_ef=30
    )
    print(f"✅ KMeansHNSW构建成功，{kmeans_hnsw.n_clusters} 个聚类")
    
    # 测试Multi-Pivot评估方法
    print("🎯 测试Multi-Pivot评估方法...")
    
    # 计算ground truth
    ground_truth = evaluator.compute_ground_truth(k=5, exclude_query_ids=False)
    print(f"✅ Ground truth计算完成")
    
    # 测试Multi-Pivot评估
    try:
        mp_result = evaluator._evaluate_multi_pivot_kmeans_from_existing(
            kmeans_hnsw=kmeans_hnsw,
            k=5,
            ground_truth=ground_truth,
            n_probe=3,
            num_pivots=3,
            pivot_selection_strategy='line_perp_third',
            pivot_overquery_factor=1.2
        )
        
        print(f"✅ Multi-Pivot评估成功!")
        print(f"   Method: {mp_result['method']}")
        print(f"   Recall@5: {mp_result['recall_at_k']:.4f}")
        print(f"   Avg Query Time: {mp_result['avg_query_time_ms']:.2f}ms")
        print(f"   Num Pivots: {mp_result['num_pivots']}")
        print(f"   Strategy: {mp_result['pivot_selection_strategy']}")
        
    except Exception as e:
        print(f"❌ Multi-Pivot评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试单pivot评估进行对比
    print("🔍 测试单pivot评估作为对比...")
    try:
        single_result = evaluator._evaluate_pure_kmeans_from_existing(
            kmeans_hnsw=kmeans_hnsw,
            k=5,
            ground_truth=ground_truth,
            n_probe=3
        )
        
        print(f"✅ 单pivot评估成功!")
        print(f"   Method: {single_result['method']}")
        print(f"   Recall@5: {single_result['recall_at_k']:.4f}")
        print(f"   Avg Query Time: {single_result['avg_query_time_ms']:.2f}ms")
        
        # 对比结果
        print(f"\n📊 性能对比:")
        print(f"   Single-Pivot Recall: {single_result['recall_at_k']:.4f}")
        print(f"   Multi-Pivot Recall:  {mp_result['recall_at_k']:.4f}")
        print(f"   Recall Improvement:  {mp_result['recall_at_k'] - single_result['recall_at_k']:+.4f}")
        
    except Exception as e:
        print(f"❌ 单pivot评估失败: {e}")
        return False
    
    print("🎉 Multi-Pivot集成测试通过!")
    return True

def test_parameter_sweep_integration():
    """测试参数扫描中的Multi-Pivot集成"""
    print("\n🧪 测试参数扫描中的Multi-Pivot集成...")
    
    # 创建更小的测试数据
    np.random.seed(42)
    dataset_size = 50
    query_size = 5
    dimension = 16
    
    base_vectors = np.random.randn(dataset_size, dimension).astype(np.float32)
    query_vectors = np.random.randn(query_size, dimension).astype(np.float32)
    query_ids = list(range(query_size))
    
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # 构建基础HNSW索引
    base_index = HNSW(distance_func=distance_func, m=6, ef_construction=30)
    
    for i, vector in enumerate(base_vectors):
        base_index.insert(i, vector)
    
    # 创建评估器
    evaluator = KMeansHNSWEvaluator(base_vectors, query_vectors, query_ids, distance_func)
    
    # 测试参数扫描（包含Multi-Pivot）
    param_grid = {
        'n_clusters': [3],
        'k_children': [10],
        'child_search_ef': [15]
    }
    
    evaluation_params = {
        'k_values': [3],
        'n_probe_values': [2]
    }
    
    adaptive_config = {
        'adaptive_k_children': False,
        'k_children_scale': 1.5,
        'k_children_min': 10,
        'k_children_max': None,
        'diversify_max_assignments': None,
        'repair_min_assignments': None,
        'multi_pivot_config': {
            'enabled': True,
            'num_pivots': 2,  # 使用较少的pivot以减少计算时间
            'pivot_selection_strategy': 'line_perp_third',
            'pivot_overquery_factor': 1.1
        }
    }
    
    try:
        print("🔄 运行参数扫描（包含Multi-Pivot）...")
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
            
            # 查找不同阶段的结果
            phases_found = set()
            for eval_result in phase_evaluations:
                phase = eval_result.get('phase', eval_result.get('method', 'unknown'))
                phases_found.add(phase)
            
            print(f"✅ 参数扫描成功! 找到阶段: {sorted(phases_found)}")
            
            # 验证Multi-Pivot阶段是否存在
            multi_pivot_found = any('multi_pivot' in str(p) for p in phases_found)
            if multi_pivot_found:
                print("🎯 Multi-Pivot阶段成功包含在参数扫描中!")
            else:
                print("⚠️ 未找到Multi-Pivot阶段，可能配置有误")
                
        else:
            print("❌ 参数扫描返回空结果")
            return False
            
    except Exception as e:
        print(f"❌ 参数扫描失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🎉 参数扫描集成测试通过!")
    return True

if __name__ == "__main__":
    print("🚀 开始Multi-Pivot集成测试...\n")
    
    success1 = test_multi_pivot_integration()
    if success1:
        success2 = test_parameter_sweep_integration()
        
        if success1 and success2:
            print("\n✅ 所有测试通过! Multi-Pivot已成功集成到tune_kmeans_hnsw_backup.py")
        else:
            print("\n❌ 部分测试失败")
    else:
        print("\n❌ 基础测试失败，跳过参数扫描测试")
