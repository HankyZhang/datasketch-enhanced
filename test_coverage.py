#!/usr/bin/env python3
"""测试不同Coverage情况"""

import sys, os
sys.path.append('..')
from method3.tune_kmeans_hnsw import *
import numpy as np

print('测试不同的Coverage情况...')
np.random.seed(42)
data = np.random.randn(100, 32).astype(np.float32)

distance_func = lambda x, y: np.linalg.norm(x - y)
base_index = HNSW(distance_func=distance_func, m=8, ef_construction=50)
for i, vec in enumerate(data):
    base_index.insert(i, vec)

print(f'构建了{len(base_index)}个向量的基础HNSW索引')

# 测试1: 使用较高的parent level，可能parent数量少
print('\n=== 测试1: parent_level=3 (可能parent少) ===')
hybrid1 = HNSWHybrid(
    base_index=base_index,
    parent_level=3,  # 较高level，parent可能很少
    k_children=50,
    repair_min_assignments=None  # 不使用repair
)
stats1 = hybrid1.get_stats()
print(f'Parents: {stats1.get("num_parents", 0)}, Coverage: {stats1.get("coverage_fraction", 0):.3f}')

# 测试2: 同样配置但启用repair
print('\n=== 测试2: 同样配置 + repair ===')
hybrid2 = HNSWHybrid(
    base_index=base_index,
    parent_level=3,
    k_children=50,
    repair_min_assignments=2  # 启用repair
)
stats2 = hybrid2.get_stats()
print(f'Parents: {stats2.get("num_parents", 0)}, Coverage: {stats2.get("coverage_fraction", 0):.3f}')

# 测试3: 使用较低level，parent多一些
print('\n=== 测试3: parent_level=1 (parent多) ===')
hybrid3 = HNSWHybrid(
    base_index=base_index,
    parent_level=1,
    k_children=50,
    repair_min_assignments=None  # 不使用repair
)
stats3 = hybrid3.get_stats()
print(f'Parents: {stats3.get("num_parents", 0)}, Coverage: {stats3.get("coverage_fraction", 0):.3f}')

# 测试4: 极端情况 - 很少的k_children
print('\n=== 测试4: 很少的k_children ===')
hybrid4 = HNSWHybrid(
    base_index=base_index,
    parent_level=2,
    k_children=10,  # 很少的k_children
    repair_min_assignments=None
)
stats4 = hybrid4.get_stats()
print(f'Parents: {stats4.get("num_parents", 0)}, Coverage: {stats4.get("coverage_fraction", 0):.3f}')

# 测试5: 同样的极端情况但启用repair
print('\n=== 测试5: 很少的k_children + repair ===')
hybrid5 = HNSWHybrid(
    base_index=base_index,
    parent_level=2,
    k_children=10,
    repair_min_assignments=2  # 启用repair
)
stats5 = hybrid5.get_stats()
print(f'Parents: {stats5.get("num_parents", 0)}, Coverage: {stats5.get("coverage_fraction", 0):.3f}')

print('\n=== 总结 ===')
print('Coverage < 1.0 的主要原因:')
print('1. 没有启用repair功能')
print('2. Parent数量太少无法覆盖所有子节点')
print('3. k_children太小，每个parent覆盖的子节点有限')
print('4. 使用repair功能应该能让Coverage达到1.0')
