#!/usr/bin/env python3
"""专门测试repair=1在10000数据上的效果对比"""

import sys, os
sys.path.append('.')
sys.path.append('./method3')
from hybrid_hnsw.hnsw_hybrid import HNSWHybrid
from hnsw.hnsw import HNSW
import numpy as np
import time

print('=== 测试repair=1在10000数据上的效果 ===')

# 创建10000个128维向量
np.random.seed(42)
n_vectors = 10000
dimensions = 128
data = np.random.randn(n_vectors, dimensions).astype(np.float32)

print(f'生成了{n_vectors}个{dimensions}维向量')

# 构建基础HNSW索引
print('构建基础HNSW索引...')
start_time = time.time()
distance_func = lambda x, y: np.linalg.norm(x - y)
base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)

for i, vec in enumerate(data):
    base_index.insert(i, vec)
    if (i + 1) % 2000 == 0:
        print(f'  已插入 {i + 1}/{n_vectors} 向量')

build_time = time.time() - start_time
print(f'基础HNSW构建完成，用时: {build_time:.2f}秒')
print(f'索引大小: {len(base_index)}个向量')

# 测试参数
parent_level = 2
k_children = 500

print(f'\n使用配置: parent_level={parent_level}, k_children={k_children}')

# 测试1: 不使用repair
print('\n=== 测试1: 不使用repair ===')
start_time = time.time()
hybrid1 = HNSWHybrid(
    base_index=base_index,
    parent_level=parent_level, 
    k_children=k_children,
    diversify_max_assignments=None,
    repair_min_assignments=None
)
hybrid1_time = time.time() - start_time

stats1 = hybrid1.get_stats()
print(f'构建时间: {hybrid1_time:.2f}秒')
print(f'Coverage: {stats1.get("coverage_fraction", 0):.3f}')
print(f'Raw coverage: {stats1.get("raw_coverage_fraction", 0):.3f}')
print(f'Parents: {stats1.get("num_parents", 0)}')
print(f'Children: {stats1.get("num_children", 0)}')
print(f'未覆盖向量: {stats1.get("uncovered_count", 0)}')
print(f'平均children/parent: {stats1.get("avg_children_per_parent", 0):.1f}')

# 测试2: 使用repair=1
print('\n=== 测试2: 使用repair=1 ===')
start_time = time.time()
hybrid2 = HNSWHybrid(
    base_index=base_index,
    parent_level=parent_level, 
    k_children=k_children,
    diversify_max_assignments=None,
    repair_min_assignments=1
)
hybrid2_time = time.time() - start_time

stats2 = hybrid2.get_stats()
print(f'构建时间: {hybrid2_time:.2f}秒')
print(f'Coverage: {stats2.get("coverage_fraction", 0):.3f}')
print(f'Raw coverage: {stats2.get("raw_coverage_fraction", 0):.3f}')
print(f'Parents: {stats2.get("num_parents", 0)}')
print(f'Children: {stats2.get("num_children", 0)}')
print(f'未覆盖向量: {stats2.get("uncovered_count", 0)}')
print(f'平均children/parent: {stats2.get("avg_children_per_parent", 0):.1f}')

# 测试3: 使用diversify + repair=1
print('\n=== 测试3: 使用diversify + repair=1 ===')
start_time = time.time()
hybrid3 = HNSWHybrid(
    base_index=base_index,
    parent_level=parent_level, 
    k_children=k_children,
    diversify_max_assignments=3,
    repair_min_assignments=1
)
hybrid3_time = time.time() - start_time

stats3 = hybrid3.get_stats()
print(f'构建时间: {hybrid3_time:.2f}秒')
print(f'Coverage: {stats3.get("coverage_fraction", 0):.3f}')
print(f'Raw coverage: {stats3.get("raw_coverage_fraction", 0):.3f}')
print(f'Parents: {stats3.get("num_parents", 0)}')
print(f'Children: {stats3.get("num_children", 0)}')
print(f'未覆盖向量: {stats3.get("uncovered_count", 0)}')
print(f'平均children/parent: {stats3.get("avg_children_per_parent", 0):.1f}')

# 比较结果
print('\n=== 结果对比 ===')
coverage_improvement_1 = stats2.get("coverage_fraction", 0) - stats1.get("coverage_fraction", 0)
coverage_improvement_2 = stats3.get("coverage_fraction", 0) - stats1.get("coverage_fraction", 0)
raw_coverage_improvement_1 = stats2.get("raw_coverage_fraction", 0) - stats1.get("raw_coverage_fraction", 0)
raw_coverage_improvement_2 = stats3.get("raw_coverage_fraction", 0) - stats1.get("raw_coverage_fraction", 0)

print(f'Coverage提升:')
print(f'  无repair → repair=1: {coverage_improvement_1:.3f}')
print(f'  无repair → diversify+repair=1: {coverage_improvement_2:.3f}')
print(f'Raw Coverage提升:')
print(f'  无repair → repair=1: {raw_coverage_improvement_1:.3f}')
print(f'  无repair → diversify+repair=1: {raw_coverage_improvement_2:.3f}')

print(f'构建时间对比:')
print(f'  无repair: {hybrid1_time:.2f}秒')
print(f'  repair=1: {hybrid2_time:.2f}秒 (+{hybrid2_time - hybrid1_time:.2f}秒)')
print(f'  diversify+repair=1: {hybrid3_time:.2f}秒 (+{hybrid3_time - hybrid1_time:.2f}秒)')

print(f'未覆盖向量数量:')
print(f'  无repair: {stats1.get("uncovered_count", 0)}')
print(f'  repair=1: {stats2.get("uncovered_count", 0)}')
print(f'  diversify+repair=1: {stats3.get("uncovered_count", 0)}')

print('\n=== 结论 ===')
if coverage_improvement_1 > 0.001:
    print('✅ repair=1 有效提升了coverage')
else:
    print('❌ repair=1 对coverage提升有限')

if stats2.get("uncovered_count", 0) < stats1.get("uncovered_count", 0):
    print('✅ repair=1 有效减少了未覆盖向量数量')
else:
    print('❌ repair=1 对未覆盖向量数量改善有限')
