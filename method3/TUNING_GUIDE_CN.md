# 方法3：K-Means HNSW 参数调优指南

## `tune_kmeans_hnsw.py` 文件概述

这个文件是**方法3（K-Means HNSW）的参数调优和评估框架**。它提供了全面的工具来：

1. **评估性能** - 评估K-Means HNSW系统的性能
2. **优化参数** - 通过系统化参数扫描进行优化
3. **基准对比** - 与基础HNSW性能进行比较
4. **生成详细报告** - 提供指标和时间统计

## 主要类和函数

### 1. `KMeansHNSWEvaluator` 类

核心评估类，包含以下关键方法：

```python
# 初始化评估器
evaluator = KMeansHNSWEvaluator(dataset, query_vectors, query_ids, distance_func)

# 计算真实答案（暴力搜索）
ground_truth = evaluator.compute_ground_truth(k=10)

# 评估召回率性能
results = evaluator.evaluate_recall(kmeans_hnsw, k=10, n_probe=10)

# 参数扫描优化
sweep_results = evaluator.parameter_sweep(base_index, param_grid, eval_params)

# 找到最优参数
optimal = evaluator.find_optimal_parameters(sweep_results, 'recall_at_k')

# 与基准HNSW比较
comparison = evaluator.compare_with_baselines(kmeans_hnsw, base_index)
```

### 2. 工具函数

- `save_results()`: 将结果保存为JSON文件
- `load_sift_data()`: 加载SIFT数据集（如果可用）

## 需要调整的关键参数

### 1. **测试数据大小**（最重要）

您可以在几个地方更改测试数据大小：

#### **在这里更改数据大小：**

**位置1 - 合成数据大小**（第463-464行）：
```python
# 当前：10K基础向量，100个查询
base_vectors = np.random.randn(10000, 128).astype(np.float32)  # 修改10000
query_vectors = np.random.randn(100, 128).astype(np.float32)   # 修改100
```

**位置2 - 查询子集**（第467行）：
```python
query_vectors = query_vectors[:100]  # 将100改为所需数量
```

**位置3 - SIFT数据子集**（您可以在SIFT加载后添加）：
```python
# 在加载SIFT数据后添加此代码以使用子集
base_vectors = base_vectors[:50000]  # 使用前50K而不是完整的1M
query_vectors = query_vectors[:200]  # 使用200个查询而不是10K
```

### 2. **优化参数网格**

#### **可调整参数：**

**K-Means HNSW系统参数：**
```python
param_grid = {
    'n_clusters': [20, 50, 100],           # K-Means聚类数量
    'k_children': [500, 1000, 2000],       # 每个聚类的子节点数
    'child_search_ef': [100, 200, 400]     # HNSW搜索宽度
}
```

**评估参数：**
```python
evaluation_params = {
    'k_values': [10],                      # 要测试的Recall@k值
    'n_probe_values': [5, 10, 20]         # 要探测的聚类数量
}
```

### 3. **参数组合数量**

将 `max_combinations=9` 改为测试更多组合（当前网格有3×3×3 = 27个总组合）。

## 如何使用文件

### **方法1：作为脚本运行**
```bash
cd method3
python tune_kmeans_hnsw.py
```

### **方法2：自定义并运行**

使用提供的 `custom_tuning.py` 文件，配置选项如下：

```python
# ========== 配置 ==========
# 数据大小设置
DATASET_SIZE = 5000      # 基础向量数量（修改这个！）
QUERY_SIZE = 50          # 查询数量（修改这个！）
DIMENSION = 128          # 向量维度

# 参数网格（调整这些！）
PARAM_GRID = {
    'n_clusters': [10, 25, 50],           # K-Means聚类
    'k_children': [200, 500, 1000],       # 每个聚类的子节点
    'child_search_ef': [50, 100, 200]     # HNSW搜索宽度
}

EVAL_PARAMS = {
    'k_values': [10],                     # Recall@k
    'n_probe_values': [3, 5, 10]         # 要探测的聚类
}

MAX_COMBINATIONS = 12    # 测试27个组合中的12个
USE_SIFT_DATA = False    # 设为True使用SIFT，False使用合成数据
```

## 参数建议

### **针对不同数据集大小：**

| 数据集大小 | n_clusters | k_children | 查询数 | 组合数 |
|-------------|------------|------------|---------|--------------|
| 1K-5K       | [5, 10, 20] | [100, 200, 500] | 20-50 | 9-12 |
| 10K-50K     | [20, 50, 100] | [500, 1000, 2000] | 50-100 | 12-18 |
| 100K-500K   | [50, 100, 200] | [1000, 2000, 5000] | 100-200 | 18-27 |
| 1M+         | [100, 200, 500] | [2000, 5000, 10000] | 200-500 | 27+ |

### **关键指导原则：**

1. **n_clusters**: 约为数据集大小的1-5%
2. **k_children**: n_clusters的10-50倍
3. **child_search_ef**: k_children的1-4倍
4. **n_probe**: n_clusters的10-50%

## 快速使用示例

### **小型测试（快速）**
```python
# 在custom_tuning.py中修改：
DATASET_SIZE = 1000
QUERY_SIZE = 20
PARAM_GRID = {
    'n_clusters': [5, 10],
    'k_children': [100, 200], 
    'child_search_ef': [50, 100]
}
MAX_COMBINATIONS = 4
```

### **中等评估**
```python
DATASET_SIZE = 10000
QUERY_SIZE = 100
PARAM_GRID = {
    'n_clusters': [20, 50, 100],
    'k_children': [500, 1000],
    'child_search_ef': [100, 200, 400]
}
MAX_COMBINATIONS = 12
```

### **完整SIFT评估**
```python
USE_SIFT_DATA = True
DATASET_SIZE = 100000  # 从SIFT 1M中使用100K
QUERY_SIZE = 500
MAX_COMBINATIONS = 18
```

## 输出和结果

脚本生成：
1. **控制台输出** - 进度和结果
2. **JSON文件** - 详细指标
3. **最优参数** - 针对您的数据集
4. **基准比较** - 显示与标准HNSW的性能对比

## 主要函数详解

### `compute_ground_truth(k, exclude_query_ids=True)`
- 使用暴力搜索计算真实最近邻
- 用作召回率计算的参考
- `k`: 要找到的邻居数量
- `exclude_query_ids`: 是否从结果中排除查询点

### `evaluate_recall(kmeans_hnsw, k, n_probe, ground_truth=None)`
- 评估K-Means HNSW系统的recall@k性能
- 将搜索结果与真实答案比较
- 返回包括时间统计在内的详细指标

### `parameter_sweep(base_index, param_grid, evaluation_params, max_combinations=None)`
- 系统化测试多个参数组合
- 为每个组合构建K-Means HNSW系统
- 评估不同设置下的性能

### `find_optimal_parameters(sweep_results, optimization_target='recall_at_k', constraints=None)`
- 从扫描结果中找到最佳参数
- 可以优化召回率、速度或其他指标
- 支持约束条件（例如，最大查询时间）

### `compare_with_baselines(kmeans_hnsw, base_index, k=10, n_probe=10, ef_values=None)`
- 将K-Means HNSW与标准HNSW比较
- 为基准测试不同的ef值
- 提供性能比较
