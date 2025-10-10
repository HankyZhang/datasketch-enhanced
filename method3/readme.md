# Method3: K-Means + HNSW 混合检索系统

## 项目概述

Method3 是一个基于 K-Means 聚类和 HNSW 图索引的两阶段检索系统，专门用于大规模向量相似性搜索。

## 核心技术

### 算法架构
```
查询向量 → K-Means质心距离计算 → 选择最近的n_probe个质心 → 在子节点中精确搜索 → 返回Top-k结果
```

### 主要组件
- **基础HNSW索引**：提供快速图搜索能力
- **K-Means聚类**：将数据分组，生成质心作为父节点
- **两阶段检索**：粗选质心 + 精确搜索的组合策略

## 核心文件

| 文件 | 功能 | 推荐使用 |
|------|------|---------|
| `v1.py` | **主评估脚本** - 包含完整的五方法对比评估 | ⭐ 推荐 |
| `kmeans_hnsw.py` | 单枢纽K-Means HNSW实现 | 基础实现 |
| `tune_kmeans_hnsw.py` | 原始参数调优脚本 | 可选 |

## 快速开始

### 基本使用
```bash
# 运行完整评估（推荐）
python method3/v1.py --dataset-size 10000 --query-size 100

# 启用Multi-Pivot多枢纽策略
python method3/v1.py --enable-multi-pivot --num-pivots 3

# 使用自适应参数
python method3/v1.py --adaptive-k-children --repair-min-assignments 2
```

### 编程接口
```python
from method3.kmeans_hnsw import KMeansHNSW
from hnsw.hnsw import HNSW
import numpy as np

# 构建基础索引
distance_func = lambda x, y: np.linalg.norm(x - y)
base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)

# 插入数据
for i, vector in enumerate(dataset):
    base_index.insert(i, vector)

# 创建K-Means HNSW系统
kmeans_hnsw = KMeansHNSW(
    base_index=base_index,
    n_clusters=50,
    k_children=1000
)

# 执行查询
results = kmeans_hnsw.search(query_vector, k=10, n_probe=5)
```

## 核心参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `n_clusters` | K-Means聚类数量 | 32-128 |
| `k_children` | 每个质心的子节点数 | 800-2000 |
| `n_probe` | 查询时探测的质心数 | 5-20 |
| `child_search_ef` | 子节点搜索的ef参数 | >= k_children * 1.3 |

## Multi-Pivot多枢纽策略

### 工作原理
- **第一个枢纽**：K-Means质心
- **第二个枢纽**：距质心最远的点
- **第三个枢纽**：垂直距离最大的点
- 通过多个枢纽收集更丰富的候选节点，提升召回率

### 使用方法
```python
# 从v1.py导入Multi-Pivot实现
from v1 import KMeansHNSWMultiPivot

multi_pivot_hnsw = KMeansHNSWMultiPivot(
    base_index=base_index,
    n_clusters=50,
    k_children=1000,
    num_pivots=3,
    pivot_selection_strategy='line_perp_third',
    pivot_overquery_factor=1.2
)
```

## 评估系统

v1.py 提供五种方法的全面对比：

1. **HNSW基线** - 纯HNSW性能
2. **纯K-Means** - 仅聚类方法
3. **Hybrid HNSW** - 基于层级的混合方法
4. **单枢纽KMeans** - 标准K-Means HNSW
5. **Multi-Pivot** - 多枢纽增强版

### 性能指标
- **召回率** (Recall@k)：检索准确性
- **查询时间**：单次检索耗时
- **覆盖率**：节点覆盖程度
- **构建时间**：索引构建耗时

## 高级功能

### 自适应参数
- `--adaptive-k-children`：根据聚类大小自动调整子节点数
- `--repair-min-assignments`：保证最小节点分配数
- `--diversify-max-assignments`：限制重复分配

### 优化选项
- 共享K-Means模型避免重复计算
- 内存优化的MiniBatch聚类
- 并行化的距离计算

## 性能特征

### 优势
- **高召回率**：聚类+图搜索的组合策略
- **可扩展性**：支持大规模数据集
- **参数灵活**：丰富的调优选项
- **多策略对比**：一次运行评估多种方法

### 适用场景
- 图像检索系统
- 文档相似性搜索  
- 推荐系统
- 大规模特征匹配

## 系统要求

- Python 3.7+
- NumPy
- scikit-learn
- 内存：8GB以上（取决于数据规模）
