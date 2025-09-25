#!/usr/bin/env python
"""简单测试multi-pivot模块导入"""

import os
import sys
import time
import numpy as np

print("开始测试...")

try:
    # 添加父目录到路径
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    print("路径添加成功")
    
    # 测试基本导入
    from hnsw.hnsw import HNSW
    print("HNSW导入成功")
    
    from method3.kmeans_hnsw import KMeansHNSW
    print("KMeansHNSW导入成功")
    
    from method3.kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot  
    print("KMeansHNSWMultiPivot导入成功")
    
    # 创建测试数据
    base_vectors = np.random.randn(100, 32).astype(np.float32)
    query_vectors = np.random.randn(5, 32).astype(np.float32)
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    print("测试数据创建成功")
    
    # 测试基础HNSW
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=50)
    for i, vec in enumerate(base_vectors):
        base_index.insert(i, vec)
    print(f"基础HNSW索引构建成功，包含{len(base_index)}个向量")
    
    # 测试KMeansHNSW
    kmeans_hnsw = KMeansHNSW(
        base_index=base_index,
        n_clusters=5,
        k_children=20
    )
    print("单枢纽KMeansHNSW构建成功")
    
    # 测试Multi-Pivot
    multi_pivot = KMeansHNSWMultiPivot(
        base_index=base_index,
        n_clusters=5,
        k_children=20,
        num_pivots=3,
        multi_pivot_enabled=True
    )
    print("Multi-Pivot KMeansHNSW构建成功")
    
    # 测试搜索
    query = query_vectors[0]
    
    results1 = kmeans_hnsw.search(query, k=5, n_probe=2)
    print(f"单枢纽搜索结果: {len(results1)} 个")
    
    results2 = multi_pivot.search(query, k=5, n_probe=2)
    print(f"多枢纽搜索结果: {len(results2)} 个")
    
    print("✅ 所有测试通过！")

except Exception as e:
    import traceback
    print(f"❌ 测试失败: {e}")
    traceback.print_exc()
