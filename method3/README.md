# 方法3：基于 K-Means 的双阶段 HNSW 系统

本目录实现项目的“方法3”方案：使用 K-Means 聚类 + HNSW 的双阶段近似最近邻检索（ANN）系统，兼顾召回率与查询效率。

## 概述

方法3包含三个阶段：

1. **阶段1**：基础 HNSW 索引（`base_index` 复用）
2. **阶段2**：K-Means 聚类产生父节点 + 通过 HNSW 搜索填充每个父节点的子集合
3. **阶段3**：查询时先选最近的若干父（centroids）→ 在其子集合内精排

### 关键创新点

相较“方法2”直接使用 HNSW 上层 level 作为父集合，本方案使用 K-Means 质心作为父节点，带来：
- 更平衡的聚类规模（减少极大/极小父簇的不均衡）
- 更贴合数据真实分布的父节点表示
- 父节点数量可独立调节（不受 HNSW 层级数量限制）
- 得益于聚类结构的召回提升潜力

## 架构流程

```
查询向量 (Query)
     ↓
阶段1：计算到全部 K-Means 质心的距离（快速）
     ↓
阶段2：选择前 n_probe 个质心，合并其子集合并在其中做精排（HNSW 子集合预先按近邻填充）
     ↓
返回 Top-k 结果
```

## 文件结构

- `kmeans_hnsw.py`：核心实现 `KMeansHNSW`
- `tune_kmeans_hnsw.py`：参数扫描与评估脚本
- `example_usage.py`：用法示例（若存在）
- `__init__.py`：包导出
- `README.md`：当前文档

## 快速上手示例

```python
from method3 import KMeansHNSW
from hnsw.hnsw import HNSW
import numpy as np

# 构建基础 HNSW 索引
distance_func = lambda x, y: np.linalg.norm(x - y)
base_index = HNSW(distance_func=distance_func)

# 插入数据
for i, vector in enumerate(dataset):
    base_index.insert(i, vector)

# 创建 K-Means HNSW 系统
kmeans_hnsw = KMeansHNSW(
    base_index=base_index,
    n_clusters=50,
    k_children=1000
)

# 查询
results = kmeans_hnsw.search(query_vector, k=10, n_probe=5)
```

## 关键参数说明

### 系统构建相关
- `n_clusters`：K-Means 聚类质心数量（即父节点数）
- `k_children`：每个父节点希望填充的子节点数（通过 HNSW 搜索近邻获得）
- `child_search_ef`：填充子集合时的 HNSW 搜索宽度（可自动或手动指定）

### 查询相关
- `k`：返回的近邻数量
- `n_probe`：查询阶段要探测的父（质心）数量

### 高级选项
- `diversify_max_assignments`：限制同一子节点被不同父节点重复分配的次数（负载均衡）
- `repair_min_assignments`：在构建阶段保证每个子节点最少分配次数（提升覆盖率）
- `include_centroids_in_results`：是否把父质心本身也作为候选返回
- `kmeans_params`：自定义 K-Means 训练参数（迭代次数、初始化方式等）

## 性能特征

### 优势
- **聚类更均衡**：减轻某些父集合“过大”带来的查询放大
- **父节点可控**：可精细调节父集合粒度
- **结构清晰**：分离“全局聚类”与“局部精排”职责
- **召回较佳**：聚类 + HNSW 组合兼顾覆盖与精度

### 取舍
- **构建时间增加**：K-Means 训练有额外成本
- **内存开销更高**：需保存质心 + 父子映射
- **参数敏感**：`n_clusters / k_children / n_probe` 需调优

## 评估框架

`KMeansHNSWEvaluator` 提供：召回评估、参数扫描、基线对比等。

```python
from method3 import KMeansHNSWEvaluator

evaluator = KMeansHNSWEvaluator(dataset, queries, query_ids, distance_func)
gt = evaluator.compute_ground_truth(k=10, exclude_query_ids=False)
recall_stats = evaluator.evaluate_recall(kmeans_hnsw, k=10, n_probe=10, ground_truth=gt)
```

还支持：
- `parameter_sweep(...)`：网格/组合扫描
- 与基础 HNSW / 纯 K-Means / Level-based Hybrid 的多方案对比

## 与现有框架集成

完全复用：
- SIFT 向量加载与格式
- 公共距离函数接口
- 真实值与召回指标计算逻辑

## 示例与演示

运行示例（如存在脚本）：
```bash
cd method3
python example_usage.py
```

## 参数调优脚本

使用：
```bash
cd method3
python tune_kmeans_hnsw.py
```
功能：
- 加载或生成数据
- 多参数组合扫描
- 输出各方案召回/时间指标
- 保存结果 JSON 便于后续分析

## 预期性能（示意）

（具体取决于数据与参数）
- Recall@10：约 0.85–0.95
- 查询耗时：5–50 ms / query
- 构建耗时：约基础 HNSW 的 2–3 倍
- 内存：~1.5× 基础 HNSW（质心 + 映射）

## 可能的未来改进

- 分层 / 递归 K-Means（多级父集合）
- 在线增量更新（动态数据集）
- GPU 加速聚类与批量距离计算
- 近似 / 采样 K-Means 以降低构建时间

---

如需与 Level-based Hybrid（`HNSWHybrid`）对比，可在调优脚本中同时启用 Hybrid 评估，以统一评估指标观察差异。

## 与 Level-based Hybrid (`HNSWHybrid`) 快速对比示例

下面示例演示：同一基础数据上分别构建 `HNSWHybrid` 与 `KMeansHNSW`，用同一批查询比较召回与耗时（示意代码，可按需裁剪）。

```python
import numpy as np
from hnsw.hnsw import HNSW
from hybrid_hnsw import HNSWHybrid
from method3 import KMeansHNSW
import time

dim = 128
n_base = 10000
n_query = 50
np.random.seed(42)
base_vectors = np.random.randn(n_base, dim).astype(np.float32)
query_vectors = np.random.randn(n_query, dim).astype(np.float32)

dist = lambda a, b: np.linalg.norm(a - b)

# 1) 构建基础 HNSW
base = HNSW(distance_func=dist, m=16, ef_construction=200)
for i, v in enumerate(base_vectors):
     base.insert(i, v)

# 2) 构建 Level-based Hybrid
hybrid = HNSWHybrid(
     base_index=base,
     parent_level=2,
     k_children=800,
     approx_ef=300,
     diversify_max_assignments=None,
     repair_min_assignments=2  # 确保较高覆盖率
)

# 3) 构建 KMeans + HNSW (Method3)
kmeans_hybrid = KMeansHNSW(
     base_index=base,
     n_clusters=64,
     k_children=800,
     child_search_ef=300,
     repair_min_assignments=2
)

def brute_force_gt(q, k=10):
     dists = np.linalg.norm(base_vectors - q, axis=1)
     idx = np.argsort(dists)[:k]
     return set(idx)

def eval_system(search_fn, label, k=10, n_probe=12):
     correct = 0
     times = []
     for q in query_vectors:
          gt = brute_force_gt(q, k)
          t0 = time.time()
          results = search_fn(q, k, n_probe)
          times.append(time.time() - t0)
          found = {nid for nid, _ in results}
          correct += len(found & gt)
     recall = correct / (len(query_vectors) * k)
     print(f"{label}: recall@{k}={recall:.4f}, avg_query_time={np.mean(times)*1000:.2f}ms")

# Hybrid 使用 search(query, k, n_probe)
eval_system(lambda q, k, n: hybrid.search(q, k=k, n_probe=n), "LevelHybrid")

# KMeansHNSW 使用 search(query, k, n_probe)
eval_system(lambda q, k, n: kmeans_hybrid.search(q, k=k, n_probe=n), "KMeansHNSW")

print("Hybrid coverage:", hybrid.get_stats().get('coverage_fraction'))
print("KMeans coverage:", kmeans_hybrid.get_stats().get('coverage_fraction'))
```

对比要点：
- `HNSWHybrid` 的父集合来自 HNSW 第 `parent_level` 层，结构受层级节点数量影响；
- `KMeansHNSW` 的父集合来自聚类，可灵活指定 `n_clusters`；
- 若目标是更均衡的父→子分布或可控父节点规模，优先尝试 KMeans 方案；
- 若希望少一次聚类成本或利用现成层级结构，可用 Level-based Hybrid；
- `repair_min_assignments` 与 `diversify_max_assignments` 两方案语义一致，可统一实验配置；
- 召回差异受：父节点覆盖度、`k_children` 质量、`n_probe`、以及底层 HNSW 参数共同影响。
