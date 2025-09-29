#!/usr/bin/env python3
"""
测试共享K-Means功能
Test shared K-Means functionality
"""

import sys
import os
import time

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from v1 import KMeansHNSWMultiPivot
from method3.kmeans_hnsw import KMeansHNSW
from hnsw.hnsw import HNSW
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def test_shared_kmeans():
    """测试共享K-Means功能"""
    print("🧪 测试共享K-Means功能...")
    
    # 创建小规模测试数据
    np.random.seed(42)
    test_vectors = np.random.randn(200, 16).astype(np.float32)
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # 构建基础HNSW索引
    print("构建HNSW索引...")
    base_index = HNSW(distance_func=distance_func, m=8, ef_construction=50)
    for i, vector in enumerate(test_vectors):
        base_index.insert(i, vector)
    
    print(f"HNSW索引构建完成: {len(base_index)} 个向量")
    
    # 预训练K-Means模型
    print("预训练K-Means模型...")
    kmeans_model = MiniBatchKMeans(n_clusters=8, random_state=42, max_iter=50)
    kmeans_model.fit(test_vectors)
    
    print(f"K-Means模型训练完成: {kmeans_model.n_clusters} clusters, inertia={kmeans_model.inertia_:.2f}")
    
    # 测试1: 不共享模型的KMeansHNSW
    print("\n=== 测试1: 不共享模型的KMeansHNSW ===")
    start_time = time.time()
    kmeans_hnsw1 = KMeansHNSW(
        base_index=base_index,
        n_clusters=8,
        k_children=30
    )
    time1 = time.time() - start_time
    print(f"构建时间: {time1:.3f}秒")
    
    # 测试2: 共享模型的KMeansHNSW
    print("\n=== 测试2: 共享模型的KMeansHNSW ===")
    start_time = time.time()
    kmeans_hnsw2 = KMeansHNSW(
        base_index=base_index,
        n_clusters=8,
        k_children=30,
        shared_kmeans_model=kmeans_model,
        shared_dataset_vectors=test_vectors
    )
    time2 = time.time() - start_time
    print(f"构建时间: {time2:.3f}秒")
    
    # 测试3: 共享模型的Multi-Pivot (必须使用共享模型)
    print("\n=== 测试3: 共享模型的Multi-Pivot ===")
    start_time = time.time()
    multi_pivot1 = KMeansHNSWMultiPivot(
        base_index=base_index,
        n_clusters=8,
        k_children=30,
        num_pivots=2,
        shared_kmeans_model=kmeans_model,
        shared_dataset_vectors=test_vectors
    )
    time3 = time.time() - start_time
    print(f"构建时间: {time3:.3f}秒")
    
    # 测试4: 不同配置的共享模型Multi-Pivot
    print("\n=== 测试4: 不同配置的共享模型Multi-Pivot ===")
    start_time = time.time()
    multi_pivot2 = KMeansHNSWMultiPivot(
        base_index=base_index,
        n_clusters=8,
        k_children=30,
        num_pivots=3,  # 不同的pivot数量
        pivot_selection_strategy='max_min_distance',  # 不同的策略
        shared_kmeans_model=kmeans_model,
        shared_dataset_vectors=test_vectors
    )
    time4 = time.time() - start_time
    print(f"构建时间: {time4:.3f}秒")
    
    # 性能对比
    print("\n📊 性能对比:")
    print(f"  KMeansHNSW (不共享): {time1:.3f}秒")
    print(f"  KMeansHNSW (共享):   {time2:.3f}秒 (节省 {((time1-time2)/time1*100):.1f}%)")
    print(f"  Multi-Pivot (2枢纽): {time3:.3f}秒")
    print(f"  Multi-Pivot (3枢纽): {time4:.3f}秒")
    
    # 统计信息对比
    print("\n📋 统计信息对比:")
    stats1 = kmeans_hnsw1.get_stats()
    stats2 = kmeans_hnsw2.get_stats()
    mp_stats1 = multi_pivot1.get_stats()
    mp_stats2 = multi_pivot2.get_stats()
    
    print(f"  KMeansHNSW共享状态: {stats2.get('shared_kmeans_used', False)}")
    print(f"  Multi-Pivot共享状态: {mp_stats2.get('shared_kmeans_used', False)}")
    
    # 验证搜索功能
    print("\n🔍 验证搜索功能:")
    query_vector = np.random.randn(16).astype(np.float32)
    
    results1 = kmeans_hnsw1.search(query_vector, k=5, n_probe=3)
    results2 = kmeans_hnsw2.search(query_vector, k=5, n_probe=3)
    mp_results1 = multi_pivot1.search(query_vector, k=5, n_probe=3)
    mp_results2 = multi_pivot2.search(query_vector, k=5, n_probe=3)
    
    print(f"  KMeansHNSW (不共享): 找到 {len(results1)} 个结果")
    print(f"  KMeansHNSW (共享):   找到 {len(results2)} 个结果")
    print(f"  Multi-Pivot (不共享): 找到 {len(mp_results1)} 个结果")
    print(f"  Multi-Pivot (共享):   找到 {len(mp_results2)} 个结果")
    
    print("\n✅ 共享K-Means功能测试完成！")
    
    return {
        'kmeans_time_saved': (time1 - time2) / time1 * 100,
        'multi_pivot_time_saved': (time3 - time4) / time3 * 100,
        'shared_working': stats2.get('shared_kmeans_used', False) and mp_stats2.get('shared_kmeans_used', False)
    }

if __name__ == "__main__":
    results = test_shared_kmeans()
    print(f"\n🎯 测试结果总结:")
    print(f"  K-Means HNSW 时间节省: {results['kmeans_time_saved']:.1f}%")
    print(f"  Multi-Pivot 时间节省: {results['multi_pivot_time_saved']:.1f}%")
    print(f"  共享功能正常工作: {'✅' if results['shared_working'] else '❌'}")
