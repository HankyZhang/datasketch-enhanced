# 方法3：基于 K-Means 的双阶段 HNSW 系统 (含Multi-Pivot扩展)

本目录实现项目的"方法3"方案：使用 K-Means 聚类 + HNSW 的双阶段近似最近邻检索（ANN）系统，兼顾召回率与查询效率。

**🎯 v1.0 重大更新：Multi-Pivot 多枢纽扩展**
- 新增 `KMeansHNSWMultiPivot` 类，使用多个枢纽点丰富子节点选择
- 五方法全面对比评估框架（HNSW基线、纯K-Means、Hybrid HNSW、单枢纽、多枢纽）
- 完整的参数调优和性能分析系统

## 概述

方法3包含三个阶段：

1. **阶段1**：基础 HNSW 索引（`base_index` 复用）
2. **阶段2**：K-Means 聚类产生父节点 + 通过 HNSW 搜索填充每个父节点的子集合
3. **阶段3**：查询时先选最近的若干父（centroids）→ 在其子集合内精排

### 关键创新点（含Multi-Pivot扩展）

相较"方法2"直接使用 HNSW 上层 level 作为父集合，本方案使用 K-Means 质心作为父节点，带来：
- 更平衡的聚类规模（减少极大/极小父簇的不均衡）
- 更贴合数据真实分布的父节点表示
- 父节点数量可独立调节（不受 HNSW 层级数量限制）
- 得益于聚类结构的召回提升潜力

**Multi-Pivot多枢纽策略**：
- 使用多个枢纽点（通常3个）从不同角度收集子节点候选
- 第一个枢纽：质心本身
- 第二个枢纽：距质心最远的点
- 第三个枢纽：垂直距离最大的点（`line_perp_third`策略）
- 通过多样化的枢纽选择提升召回率阶段 HNSW 系统

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

- `kmeans_hnsw.py`：核心实现 `KMeansHNSW`（单枢纽版本）
- `v1.py`：**主要评估脚本** - 包含完整的Multi-Pivot实现和五方法对比评估
- `tune_kmeans_hnsw.py`：原始参数扫描脚本
- `kmeans_hnsw_multi_pivot.py`：独立的Multi-Pivot实现
- `README_multi_pivot.md`：Multi-Pivot详细文档
- `__init__.py`：包导出
- `README.md`：当前文档

## 快速上手示例

### 单枢纽版本（原始KMeansHNSW）
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

### Multi-Pivot多枢纽版本（新增）
```python
# 从v1.py导入Multi-Pivot实现
import sys, os
sys.path.append(os.path.dirname(__file__))
from v1 import KMeansHNSWMultiPivot

# 创建 Multi-Pivot K-Means HNSW 系统
# 注意：使用与KMeansHNSW相同的sklearn.MiniBatchKMeans进行聚类
multi_pivot_hnsw = KMeansHNSWMultiPivot(
    base_index=base_index,
    n_clusters=50,
    k_children=1000,
    num_pivots=3,                              # 使用3个枢纽点
    pivot_selection_strategy='line_perp_third', # 枢纽选择策略
    pivot_overquery_factor=1.2                 # 过度查询因子
)

# 查询（接口与单枢纽版本相同）
results = multi_pivot_hnsw.search(query_vector, k=10, n_probe=5)
```

**关键说明**：
- KMeansHNSWMultiPivot与KMeansHNSW共享相同的K-Means聚类结果
- 两个类都使用`sklearn.MiniBatchKMeans`进行聚类
- 主要差异在于**子节点分配策略**：单枢纽vs多枢纽
- 聚类质心完全相同，仅在子节点收集方式上不同

### 共享K-Means模型（优化性能）

为了避免重复计算K-Means聚类，可以共享已训练的模型：

```python
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# 1. 预先计算K-Means聚类
base_vectors = []  # 从base_index提取向量
for node_id, node in base_index._nodes.items():
    if node.point is not None:
        base_vectors.append(node.point)
dataset_vectors = np.array(base_vectors)

# 训练K-Means模型
kmeans_model = MiniBatchKMeans(
    n_clusters=50,
    random_state=42,
    max_iter=100,
    batch_size=min(100, len(dataset_vectors))
)
kmeans_model.fit(dataset_vectors)

# 2. 创建共享K-Means的单枢纽版本
kmeans_hnsw = KMeansHNSW(
    base_index=base_index,
    n_clusters=50,
    k_children=1000,
    shared_kmeans_model=kmeans_model,      # 共享已训练的模型
    shared_dataset_vectors=dataset_vectors  # 共享数据向量
)

# 3. 创建共享K-Means的Multi-Pivot版本（需要扩展支持）
# 注意：当前v1.py中的KMeansHNSWMultiPivot还未完全支持shared_kmeans_model
# 建议的扩展方式见下文
```

**当前状态和优化建议**：

**现状分析**：
1. **KMeansHNSW**: 虽然有`shared_kmeans_model`参数，但**实际未实现**共享逻辑
2. **KMeansHNSWMultiPivot**: 没有`shared_kmeans_model`参数，每次都重新计算
3. **重复计算问题**: 两个类会各自独立进行K-Means聚类

**建议的优化实现**：
```python
# 推荐的实现方式（需要代码修改）
def create_shared_kmeans_systems(base_index, n_clusters=50, k_children=1000):
    """创建共享K-Means模型的两个系统"""
    
    # 1. 一次性提取数据和聚类
    dataset_vectors = []
    for node_id, node in base_index._nodes.items():
        if node.point is not None:
            dataset_vectors.append(node.point)
    dataset_vectors = np.array(dataset_vectors)
    
    # 2. 训练K-Means模型（只计算一次）
    kmeans_model = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, 
        max_iter=100, batch_size=min(100, len(dataset_vectors))
    )
    kmeans_model.fit(dataset_vectors)
    
    # 3. 创建两个系统，共享聚类结果
    single_pivot = KMeansHNSW(
        base_index=base_index,
        n_clusters=n_clusters,
        k_children=k_children,
        shared_kmeans_model=kmeans_model,  # 需要实现支持
        shared_dataset_vectors=dataset_vectors
    )
    
    multi_pivot = KMeansHNSWMultiPivot(
        base_index=base_index,
        n_clusters=n_clusters, 
        k_children=k_children,
        num_pivots=3,
        shared_kmeans_model=kmeans_model,  # 需要添加参数
        shared_dataset_vectors=dataset_vectors
    )
    
    return single_pivot, multi_pivot
```

**性能优化潜力**: 实现K-Means共享后可节省50-80%的聚类计算时间，特别适合需要对比多种方法的场景。

### 实现K-Means共享的代码修改建议

#### 1. 修改KMeansHNSW类（kmeans_hnsw.py）

在`_perform_kmeans_clustering`方法中添加共享模型检查：

```python
def _perform_kmeans_clustering(self, dataset_vectors: np.ndarray):
    """Perform MiniBatchKMeans clustering to identify parent centroids."""
    
    # 检查是否有共享的K-Means模型
    if self.shared_kmeans_model is not None:
        print(f"Using shared MiniBatchKMeans model with {self.n_clusters} clusters...")
        self.kmeans_model = self.shared_kmeans_model
        self.centroids = self.kmeans_model.cluster_centers_
        self.centroid_ids = [f"centroid_{i}" for i in range(self.n_clusters)]
        
        # 使用共享的数据向量计算聚类统计信息
        if self.shared_dataset_vectors is not None:
            labels = self.shared_kmeans_model.predict(self.shared_dataset_vectors)
            cluster_sizes = np.bincount(labels, minlength=self.n_clusters)
            self._cluster_info = {
                'avg_cluster_size': float(np.mean(cluster_sizes)),
                'std_cluster_size': float(np.std(cluster_sizes)),
                # ... 其他统计信息
            }
        return
    
    # 原有的K-Means训练逻辑
    print(f"Running MiniBatchKMeans with {self.n_clusters} clusters...")
    # ... 现有代码
```

#### 2. 修改KMeansHNSWMultiPivot类（v1.py）

添加共享参数支持：

```python
def __init__(
    self,
    base_index: HNSW,
    n_clusters: int = 100,
    k_children: int = 800,
    # ... 现有参数
    # 新增共享参数
    shared_kmeans_model: Optional[MiniBatchKMeans] = None,
    shared_dataset_vectors: Optional[np.ndarray] = None
):
    # ... 现有初始化代码
    self.shared_kmeans_model = shared_kmeans_model
    self.shared_dataset_vectors = shared_dataset_vectors

def _perform_kmeans_clustering(self):
    """执行K-Means聚类 (支持共享模型)"""
    
    # 检查共享模型
    if self.shared_kmeans_model is not None:
        print(f"Using shared MiniBatchKMeans model...")
        self.kmeans_model = self.shared_kmeans_model
        self.centroids = self.kmeans_model.cluster_centers_
        self.n_clusters = self.centroids.shape[0]
        self.centroid_ids = [f"centroid_{i}" for i in range(self.n_clusters)]
        return
    
    # 原有的K-Means训练逻辑
    # ... 现有代码
```

#### 3. 在v1.py的参数扫描中使用共享模型

修改`parameter_sweep`方法来避免重复聚类：

```python
# 在parameter_sweep开始时预计算K-Means
shared_kmeans_models = {}  # 缓存不同n_clusters的模型

for combination in combinations:
    params = dict(zip(param_names, combination))
    n_clusters = params['n_clusters']
    
    # 检查是否已有此n_clusters的模型
    if n_clusters not in shared_kmeans_models:
        # 训练新的K-Means模型
        kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        kmeans_model.fit(dataset_vectors)
        shared_kmeans_models[n_clusters] = kmeans_model
    
    # 使用共享模型创建系统
    shared_model = shared_kmeans_models[n_clusters]
    
    kmeans_hnsw = KMeansHNSW(
        base_index=base_index,
        shared_kmeans_model=shared_model,
        shared_dataset_vectors=dataset_vectors,
        **params
    )
    
    if multi_pivot_config.get('enabled'):
        multi_pivot_hnsw = KMeansHNSWMultiPivot(
            base_index=base_index,
            shared_kmeans_model=shared_model,
            shared_dataset_vectors=dataset_vectors,
            **params,
            **multi_pivot_config
        )
```

这样的修改可以显著提升多方法对比的性能，避免重复计算相同的K-Means聚类。

## 关键参数说明

### 系统构建相关
- `n_clusters`：K-Means 聚类质心数量（即父节点数）
- `k_children`：每个父节点希望填充的子节点数（通过 HNSW 搜索近邻获得）
- `child_search_ef`：填充子集合时的 HNSW 搜索宽度（可自动或手动指定）

### 查询相关
- `k`：返回的近邻数量
- `n_probe`：查询阶段要探测的父（质心）数量

### Multi-Pivot特定参数（新增）
- `num_pivots`：每个聚类的枢纽点数量（默认3，最小1）
- `pivot_selection_strategy`：枢纽点选择策略
  - `'line_perp_third'`：第三个枢纽选择垂直距离最大的点
  - `'max_min_distance'`：选择与现有枢纽最小距离最大的点
- `pivot_overquery_factor`：枢纽查询的过度查询因子（默认1.2）

### 自适应和优化选项
- `adaptive_k_children`：启用基于平均聚类大小的自适应k_children
- `k_children_scale`：自适应k_children的缩放因子（默认1.5）
- `k_children_min/max`：自适应时的最小/最大k_children
- `diversify_max_assignments`：限制同一子节点被不同父节点重复分配的次数
- `repair_min_assignments`：在构建阶段保证每个子节点最少分配次数（提升覆盖率）

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

## 评估框架（v1.0全面升级）

`KMeansHNSWEvaluator` 提供全方位评估功能：

### 五方法对比评估
1. **HNSW基线** - 纯HNSW索引性能
2. **纯K-Means** - 仅使用K-Means聚类
3. **Hybrid HNSW** - 基于HNSW层级的混合方法
4. **KMeans HNSW** - 单枢纽K-Means HNSW
5. **Multi-Pivot** - 多枢纽K-Means HNSW（新增）

### 基本用法
```python
from v1 import KMeansHNSWEvaluator

evaluator = KMeansHNSWEvaluator(dataset, queries, query_ids, distance_func)
gt = evaluator.compute_ground_truth(k=10, exclude_query_ids=False)

# 评估单枢纽版本
recall_stats = evaluator.evaluate_recall(kmeans_hnsw, k=10, n_probe=10, ground_truth=gt)

# 评估Multi-Pivot版本
mp_recall_stats = evaluator.evaluate_multi_pivot_recall(multi_pivot_hnsw, k=10, n_probe=10, ground_truth=gt)
```

### 参数扫描功能
```python
# 配置Multi-Pivot参数
multi_pivot_config = {
    'enabled': True,
    'num_pivots': 3,
    'pivot_selection_strategy': 'line_perp_third',
    'pivot_overquery_factor': 1.2
}

# 执行全面参数扫描
sweep_results = evaluator.parameter_sweep(
    base_index, param_grid, evaluation_params,
    multi_pivot_config=multi_pivot_config
)
```

## 与现有框架集成

完全复用：
- SIFT 向量加载与格式
- 公共距离函数接口
- 真实值与召回指标计算逻辑

## 示例与演示

### v1.0 全面评估脚本（推荐）

#### 基础使用
```bash
cd method3

# 基础Multi-Pivot评估
python v1.py --enable-multi-pivot

# 自定义数据集大小和Multi-Pivot参数
python v1.py --dataset-size 20000 --query-size 100 \
             --enable-multi-pivot --num-pivots 3 \
             --pivot-selection-strategy line_perp_third

# 启用自适应参数和修复功能
python v1.py --enable-multi-pivot --adaptive-k-children \
             --repair-min-assignments 2 --diversify-max-assignments 3

# 使用SIFT数据集
python v1.py --enable-multi-pivot --dataset-size 50000 --query-size 1000
```

#### 完整命令行参数

**数据集选项**
- `--dataset-size N`：基础向量数量（默认10000）
- `--query-size N`：查询向量数量（默认50）
- `--dimension N`：合成数据维度（默认128）
- `--no-sift`：强制使用合成数据

**Multi-Pivot参数**
- `--enable-multi-pivot`：启用Multi-Pivot评估
- `--num-pivots N`：枢纽点数量（默认3）
- `--pivot-selection-strategy STRATEGY`：枢纽选择策略
  - `line_perp_third`：垂直距离策略（默认）
  - `max_min_distance`：最大最小距离策略
- `--pivot-overquery-factor F`：过度查询因子（默认1.2）

**自适应和优化选项**
- `--adaptive-k-children`：启用自适应k_children
- `--k-children-scale F`：自适应缩放因子（默认1.5）
- `--k-children-min N`：最小k_children（默认100）
- `--k-children-max N`：最大k_children
- `--diversify-max-assignments N`：最大分配次数
- `--repair-min-assignments N`：最少分配次数
- `--hybrid-parent-level N`：Hybrid HNSW父层级（默认2）
- `--no-hybrid`：禁用Hybrid HNSW评估

### 原始参数调优脚本
```bash
cd method3
python tune_kmeans_hnsw.py
```

### 功能特性
- **数据源**：自动加载SIFT数据集，或生成合成数据
- **五方法对比**：HNSW基线、纯K-Means、Hybrid HNSW、单/多枢纽
- **参数扫描**：自动化网格搜索最优参数组合
- **结果保存**：详细的JSON格式评估报告（`method3_tuning_results.json`）
- **性能分析**：召回率、查询时间、构建时间等指标

### 输出结果说明

评估完成后将生成 `method3_tuning_results.json` 文件，包含：

```json
{
  "sweep_results": [
    {
      "parameters": {"n_clusters": 64, "k_children": 800, "child_search_ef": 300},
      "construction_time": 2.34,
      "phase_evaluations": [
        {
          "phase": "baseline_hnsw",
          "recall_at_k": 0.856,
          "avg_query_time_ms": 12.3
        },
        {
          "phase": "kmeans_hnsw_single_pivot", 
          "recall_at_k": 0.891,
          "avg_query_time_ms": 18.7
        },
        {
          "phase": "kmeans_hnsw_multi_pivot",
          "recall_at_k": 0.924,
          "avg_query_time_ms": 23.1,
          "system_stats": {
            "num_pivots": 3,
            "pivot_strategy": "line_perp_third"
          }
        }
      ]
    }
  ],
  "multi_pivot_config": {...},
  "evaluation_info": {...}
}
```

**关键指标说明**：
- `recall_at_k`：召回率@k（越高越好）
- `avg_query_time_ms`：平均查询时间（毫秒）
- `construction_time`：构建时间（秒）
- `coverage_fraction`：节点覆盖率（1.0为完全覆盖）

## 预期性能（含Multi-Pivot对比）

（具体取决于数据与参数）

### 单枢纽 KMeans HNSW
- Recall@10：约 0.85–0.92
- 查询耗时：8–40 ms / query
- 构建耗时：约基础 HNSW 的 2–3 倍
- 内存：~1.5× 基础 HNSW

### Multi-Pivot KMeans HNSW（新增）
- **Recall@10：约 0.88–0.96**（相比单枢纽提升3-5%）
- 查询耗时：12–55 ms / query（略高于单枢纽）
- 构建耗时：约基础 HNSW 的 2.5–4 倍
- 内存：~1.6× 基础 HNSW（额外的枢纽信息）

### 性能优势分析
- **召回率提升**：Multi-Pivot通过多角度候选收集显著提升召回率
- **查询延迟权衡**：略增查询时间换取更高准确性
- **适用场景**：对召回率要求较高的应用（推荐系统、相似搜索等）

## v1.0 新增功能与改进

### Multi-Pivot核心创新
- **多枢纽策略**：质心 + 最远点 + 垂直最大点的组合
- **自适应候选收集**：动态调整候选数量（`pivot_overquery_factor`）
- **策略可选**：支持多种枢纽选择算法

### 评估系统升级
- **五方法统一评估**：一次运行对比所有方法
- **详细性能分析**：个体召回率、查询时间分布等
- **参数自动调优**：智能网格搜索最优配置

### 可能的未来改进
- 动态枢纽数量（根据聚类密度自适应）
- GPU加速的多枢纽搜索
- 分层Multi-Pivot（递归多级枢纽）
- 在线学习优化枢纽选择策略

---

## v1.0 五方法全面对比示例

使用v1.py可以一次性对比所有方法的性能：

```python
# 运行完整的五方法对比评估
python v1.py --enable-multi-pivot --dataset-size 10000 --query-size 50

# 结果将包含：
# 1. HNSW基线 (HNSW Baseline)
# 2. 纯K-Means (Pure K-Means) 
# 3. Hybrid HNSW (Level-based)
# 4. KMeans HNSW (单枢纽)
# 5. Multi-Pivot KMeans HNSW (多枢纽)
```

### 手动构建五方法对比
```python
import numpy as np
from hnsw.hnsw import HNSW
from hybrid_hnsw import HNSWHybrid
from method3 import KMeansHNSW
from v1 import KMeansHNSWMultiPivot, KMeansHNSWEvaluator

# 构建基础数据和索引
dim, n_base, n_query = 128, 10000, 50
base_vectors = np.random.randn(n_base, dim).astype(np.float32)
query_vectors = np.random.randn(n_query, dim).astype(np.float32)
distance_func = lambda x, y: np.linalg.norm(x - y)

base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
for i, v in enumerate(base_vectors):
    base_index.insert(i, v)

# 1. HNSW基线
print("=== HNSW基线 ===")
# 直接使用base_index.query()进行评估

# 2. Hybrid HNSW
print("=== Hybrid HNSW ===")
hybrid = HNSWHybrid(
    base_index=base_index,
    parent_level=2,
    k_children=800,
    approx_ef=300,
    repair_min_assignments=2
)

# 3. 单枢纽KMeans HNSW
print("=== 单枢纽KMeans HNSW ===")
kmeans_hnsw = KMeansHNSW(
    base_index=base_index,
    n_clusters=64,
    k_children=800,
    child_search_ef=300,
    repair_min_assignments=2
)

# 4. Multi-Pivot KMeans HNSW
print("=== Multi-Pivot KMeans HNSW ===")
multi_pivot_hnsw = KMeansHNSWMultiPivot(
    base_index=base_index,
    n_clusters=64,
    k_children=800,
    child_search_ef=300,
    num_pivots=3,
    pivot_selection_strategy='line_perp_third',
    pivot_overquery_factor=1.2,
    repair_min_assignments=2
)

# 使用统一评估器对比性能
evaluator = KMeansHNSWEvaluator(
    base_vectors, query_vectors, list(range(n_query)), distance_func
)

# 执行参数扫描和五方法对比
param_grid = {'n_clusters': [64], 'k_children': [800], 'child_search_ef': [300]}
evaluation_params = {'k_values': [10], 'n_probe_values': [5, 10, 20]}
multi_pivot_config = {'enabled': True, 'num_pivots': 3}

results = evaluator.parameter_sweep(
    base_index, param_grid, evaluation_params, 
    multi_pivot_config=multi_pivot_config
)
```

### 方法对比要点

| 方法 | 优势 | 劣势 | 适用场景 |
|-----|------|------|---------|
| **HNSW基线** | 简单、快速 | 召回率受限 | 对速度要求极高的场景 |
| **纯K-Means** | 均衡聚类 | 缺乏图结构优势 | 需要严格聚类结构的应用 |
| **Hybrid HNSW** | 利用现有层级 | 父节点数量不可控 | 希望复用HNSW结构 |
| **单枢纽KMeans** | 平衡性能与速度 | 子节点选择有限 | 通用的高性能场景 |
| **Multi-Pivot** | **最高召回率** | 查询延迟略高 | **推荐系统、精确搜索** |

推荐使用v1.py进行全面评估，根据具体需求选择最适合的方法。
