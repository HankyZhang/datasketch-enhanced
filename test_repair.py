#!/usr/bin/env python3
"""测试HybridHNSW repair功能"""

import sys, os
sys.path.append('.')
from method3.tune_kmeans_hnsw import *
import numpy as np

print('测试HybridHNSW repair功能...')

# 创建小测试数据
np.random.seed(42)
data = np.random.randn(100, 32).astype(np.float32)

# 构建基础HNSW
distance_func = lambda x, y: np.linalg.norm(x - y)
base_index = HNSW(distance_func=distance_func, m=8, ef_construction=50)
for i, vec in enumerate(data):
    base_index.insert(i, vec)

print(f'构建了{len(base_index)}个向量的基础HNSW索引')

# 测试1: 不使用repair
print('\n=== 测试1: 不使用repair ===')
hybrid1 = HNSWHybrid(
    base_index=base_index,
    parent_level=1, 
    k_children=20,
    diversify_max_assignments=None,
    repair_min_assignments=None
)
stats1 = hybrid1.get_stats()
print(f'Coverage: {stats1.get("coverage_fraction", 0):.3f}')
print(f'Parents: {stats1.get("num_parents", 0)}')
print(f'Children: {stats1.get("num_children", 0)}')

# 测试2: 使用repair
print('\n=== 测试2: 使用repair ===')
hybrid2 = HNSWHybrid(
    base_index=base_index,
    parent_level=1, 
    k_children=20,
    diversify_max_assignments=5,
    repair_min_assignments=2
)
stats2 = hybrid2.get_stats()
print(f'Coverage: {stats2.get("coverage_fraction", 0):.3f}')
print(f'Parents: {stats2.get("num_parents", 0)}')
print(f'Children: {stats2.get("num_children", 0)}')

print('\n=== 对比结果 ===')
print(f'Coverage改善: {stats2.get("coverage_fraction", 0) - stats1.get("coverage_fraction", 0):.3f}')
print(f'Children数量变化: {stats2.get("num_children", 0) - stats1.get("num_children", 0)}')
