# HNSW 实现分析 - Datasketch 项目

本文档提供了对 datasketch 项目中所有 HNSW（分层导航小世界图）相关代码的全面分析。

## 概述

datasketch 项目包含一个完整的 HNSW 实现，用于近似最近邻搜索，支持多种距离函数，包括欧几里得距离、Jaccard 距离和基于 MinHash 的相似性。该实现包含三个主要组件：

1. **核心实现** (`datasketch/hnsw.py`)
2. **基准测试代码** (`benchmark/indexes/jaccard/hnsw.py`)
3. **单元测试** (`test/test_hnsw.py`)

---

## 1. 核心 HNSW 实现 (`datasketch/hnsw.py`)

### 目的
基于 Yu. A. Malkov 和 D. A. Yashunin (2016) 的论文"使用分层导航小世界图进行高效且鲁棒的近似最近邻搜索"实现 HNSW 算法。

### 关键类

#### 支持类

**`_Layer`**
- 表示 HNSW 索引中的图层
- 将键映射到邻居字典：`{neighbor_key: distance}`
- 提供类似字典的接口（`__getitem__`、`__setitem__` 等）
- 关键方法：
  - `get_reverse_edges(key)`：查找指向给定键的所有节点
  - `copy()`：创建层的深度复制

**`_LayerWithReversedEdges`**
- `_Layer` 的增强版本，维护反向边映射
- 支持更快的硬删除操作
- 当邻居发生变化时自动更新反向边
- 内存使用量更高，但删除性能更好

**`_Node`**
- 表示图中的单个节点
- 包含：key（标识符）、point（numpy 数组）、is_deleted（软删除标志）
- 支持相等性比较和哈希

#### 主要 HNSW 类

**构造器参数：**
```python
HNSW(
    distance_func,      # 函数：(np.ndarray, np.ndarray) -> float
    m=16,              # 每个节点的邻居数量
    ef_construction=200, # 构建过程中考虑的邻居数量
    m0=None,           # 第 0 层的邻居数量（默认：2*m）
    seed=None,         # 随机种子
    reversed_edges=False # 是否维护反向边
)
```

### 核心方法

#### 插入操作
- **`insert(key, new_point, ef=None, level=None)`**：添加/更新一个点
- **`__setitem__(key, value)`**：使用 `index[key] = point` 语法插入的别名
- **`update(other)`**：从映射或另一个 HNSW 批量插入
- **`setdefault(key, default)`**：如果键不存在则插入

#### 查询操作
- **`query(query_point, k=None, ef=None)`**：查找 k 个最近邻
- **`__getitem__(key)`**：通过键检索点
- **`get(key, default=None)`**：带默认值的安全检索
- **`__contains__(key)`**：检查键是否存在（且未被软删除）

#### 删除操作
- **`remove(key, hard=False, ef=None)`**：删除点（软删除或硬删除）
- **`__delitem__(key)`**：使用 `del index[key]` 语法软删除的别名
- **`pop(key, default=None, hard=False)`**：删除并返回点
- **`popitem(last=True, hard=False)`**：删除并返回任意项
- **`clean(ef=None)`**：硬删除所有软删除的点
- **`clear()`**：删除所有点

#### 实用操作
- **`copy()`**：创建索引的深度复制
- **`merge(other)`**：通过合并两个索引创建新索引
- **`__len__()`**：未删除点的数量
- **`keys()`、`values()`、`items()`**：迭代器方法

### 使用示例

#### 欧几里得距离示例
```python
from datasketch.hnsw import HNSW
import numpy as np

# 创建随机数据
data = np.random.random_sample((1000, 10))

# 创建索引
index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

# 插入点
for i, d in enumerate(data):
    index.insert(i, d)

# 查询 10 个最近邻
neighbors = index.query(data[0], k=10)
# 返回：[(key, distance), ...]
```

#### Jaccard 距离示例
```python
# 用整数数组表示集合的 Jaccard 距离
data = np.random.randint(0, 100, size=(1000, 10))

jaccard_distance = lambda x, y: (
    1.0 - float(len(np.intersect1d(x, y, assume_unique=False)))
    / float(len(np.union1d(x, y)))
)

index = HNSW(distance_func=jaccard_distance)
for i, d in enumerate(data):
    index[i] = d  # 替代插入语法

neighbors = index.query(data[0], k=10)
```

### 内部算法详细信息

#### 多层结构
- 较高层的节点数量呈指数级减少
- 层级分配：`level = int(-log(random()) * level_mult)`
- 入口点维护在最高层

#### 搜索算法
- **`_search_ef1()`**：单个最近邻的贪婪搜索（较高层）
- **`_search_base_layer()`**：使用 ef 候选的束搜索（第 0 层）
- **`_heuristic_prune()`**：基于距离多样性的邻居选择

#### 删除策略
- **软删除**：标记为已删除，但保留在图中用于遍历
- **硬删除**：完全删除，修复邻居连接
- 删除当前入口点时重新分配入口点

---

## 2. 基准测试实现 (`benchmark/indexes/jaccard/hnsw.py`)

### 目的
提供基准测试函数来比较 HNSW 与其他方法在 Jaccard 相似性搜索任务中的性能。

### 关键函数

#### `search_nswlib_jaccard_topk(index_data, query_data, index_params, k)`
**目的**：与 nmslib 的 HNSW 实现进行基准比较
- 将集合转换为空格分隔的字符串以兼容 nmslib
- 在 nmslib 中使用 `jaccard_sparse` 空间
- 返回精确的 Jaccard 相似性以便公平比较

**参数：**
- `index_data`：(sets, keys, _, cache) 元组
- `query_data`：(sets, keys, _) 元组
- `index_params`：nmslib 参数（例如，`{'efConstruction': 200}`）
- `k`：要检索的邻居数量

#### `search_hnsw_jaccard_topk(index_data, query_data, index_params, k)`
**目的**：使用直接 Jaccard 距离对 datasketch HNSW 进行基准测试
- 使用原始集合而不进行转换
- 利用 `compute_jaccard_distance` 函数
- 将返回的距离转换回相似性

**工作流程：**
1. 使用 Jaccard 距离函数构建 HNSW 索引
2. 插入所有索引集合
3. 使用每个查询集合进行查询
4. 将距离转换为相似性：`similarity = 1.0 - distance`

#### `search_hnsw_minhash_jaccard_topk(index_data, query_data, index_params, k)`
**目的**：使用 MinHash 近似对 HNSW 进行基准测试
- 使用 MinHash 签名而不是原始集合
- 对大集合更高效（降维）
- 对检索到的候选项计算精确 Jaccard 以确保准确性

**工作流程：**
1. 为所有集合生成 MinHash 签名
2. 使用 MinHash 距离函数构建 HNSW 索引
3. 使用 MinHash 签名进行查询
4. 检索原始集合并计算精确 Jaccard 相似性
5. 按精确相似性重新排序

### 基准测试指标
所有函数返回：
- **索引指标**：构建时间、预处理时间
- **查询结果**：(query_key, [(result_key, similarity), ...])
- **查询时间**：用于 QPS 计算的个别查询持续时间

---

## 3. 单元测试 (`test/test_hnsw.py`)

### 目的
全面的测试套件，验证不同距离函数和使用模式下的 HNSW 功能。

### 测试类

#### `TestHNSW` - 基本 L2 距离测试
**距离函数**：`l2_distance(x, y) = np.linalg.norm(x - y)`

**关键测试方法：**
- **`test_search()`**：基本索引和查询功能
- **`test_upsert()`**：向现有索引添加新点
- **`test_update()`**：通过 `update()` 方法进行批量更新
- **`test_merge()`**：合并两个独立的索引
- **`test_pickle()`**：序列化/反序列化
- **`test_copy()`**：深度复制行为和独立性
- **`test_soft_remove_and_pop_and_clean()`**：软删除工作流程
- **`test_hard_remove_and_pop_and_clean()`**：硬删除工作流程
- **`test_popitem_last()`**：LIFO 删除行为
- **`test_popitem_first()`**：FIFO 删除行为
- **`test_clear()`**：完全索引清除

#### `TestHNSWLayerWithReversedEdges`
**目的**：使用 `reversed_edges=True` 测试相同功能
- 继承 `TestHNSW` 的所有测试
- 验证反向边优化不会破坏功能
- 确保更快的硬删除性能

#### `TestHNSWJaccard`
**距离函数**：`jaccard_distance(x, y) = 1.0 - |intersect(x,y)| / |union(x,y)|`
**数据类型**：表示集合的整数数组

**特化：**
- 重写 `_create_random_points()` 生成整数数组
- 针对 Jaccard 距离的自定义搜索验证
- 测试离散/分类数据处理

#### `TestHNSWMinHashJaccard`
**距离函数**：`minhash_jaccard_distance(x, y) = 1.0 - x.jaccard(y)`
**数据类型**：MinHash 对象

**工作流程：**
- 生成整数集合，转换为 MinHash 签名
- 测试近似相似性搜索
- 验证 MinHash 与 HNSW 的集成

### 测试实用工具

#### `_create_random_points(n=100, dim=10)`
- 为每个测试类生成适当的测试数据
- L2：`np.random.rand(n, dim)`
- Jaccard：`np.random.randint(0, high, (n, dim))`
- MinHash：`MinHash.bulk(sets, num_perm=128)`

#### `_insert_points(index, points, keys=None)`
- 测试两种插入方法：`insert()` 和 `[]` 赋值
- 验证入口点设置、包含性、检索
- 检查顺序保持和长度更新

#### `_search_index(index, queries, k=10)`
- 验证搜索结果正确按距离排序
- 确保图连通性（能找到足够的邻居）
- 测试不同距离函数的搜索功能

### 测试执行
```bash
# 运行所有 HNSW 测试
python -m pytest test/test_hnsw.py -v

# 运行特定测试类
python -m pytest test/test_hnsw.py::TestHNSWJaccard -v

# 运行覆盖率测试
python -m pytest test/test_hnsw.py --cov=datasketch.hnsw
```

---

## 算法特征

### 性能特征
- **时间复杂度**：
  - 插入：平均情况 O(log N * M)
  - 查询：平均情况 O(log N * ef)
  - 空间：O(N * M)，其中 M 是平均度数

### 调优参数
- **`m`**：更高的值 → 更好的召回率、更慢的插入、更多内存
- **`ef_construction`**：更高的值 → 更好的索引质量、更慢的构建
- **`ef`（查询时间）**：更高的值 → 更好的召回率、更慢的查询

### 距离函数要求
- 必须是对称的：`d(x,y) = d(y,x)`
- 必须满足三角不等式以获得最佳性能
- 对于相同的点应返回 0

### 适用场景
- **高维向量**（嵌入、特征）
- **大型数据集**（百万级以上点）精确搜索太慢
- **实时应用**需要快速近似最近邻搜索
- **自定义相似性度量**超出标准欧几里得距离

---

## 集成示例

### 与不同数据类型的使用

#### 文本嵌入
```python
from sentence_transformers import SentenceTransformer
from datasketch.hnsw import HNSW
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["示例文本 1", "示例文本 2", ...]
embeddings = model.encode(texts)

index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
for i, emb in enumerate(embeddings):
    index[i] = emb

# 查找相似文本
query_embedding = model.encode(["查询文本"])
similar = index.query(query_embedding[0], k=5)
```

#### 使用 MinHash 的集合相似性
```python
from datasketch import MinHash
from datasketch.hnsw import HNSW

def create_minhash_from_set(s, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for item in s:
        m.update(str(item).encode('utf8'))
    return m

sets = [set(np.random.randint(0, 1000, 50)) for _ in range(10000)]
minhashes = [create_minhash_from_set(s) for s in sets]

index = HNSW(distance_func=lambda x, y: 1.0 - x.jaccard(y))
for i, mh in enumerate(minhashes):
    index[i] = mh

# 查找相似集合
query_minhash = create_minhash_from_set(sets[0])
similar = index.query(query_minhash, k=10)
```

---

## 性能考虑

### 内存使用
- **标准层**：每条边约 16-32 字节
- **反向边**：每条边额外约 8 字节用于反向映射
- **节点存储**：每个节点约 24 字节 + 点大小

### 优化提示
1. **批量插入**比单个插入更高效
2. **预设数据结构大小**当最终大小已知时
3. **使用适当的 `m` 值**：大多数应用为 16-48
4. **调优 `ef_construction`**：根据召回率要求为 200-800
5. **考虑软删除与硬删除**基于更新模式

### 线程考虑
- **不是线程安全的**对于并发修改
- **只读查询**可以安全并行化
- **使用独立索引**每个线程用于并发更新

本文档提供了 datasketch 项目中 HNSW 实现的完整概述，涵盖了使用模式和内部实现细节。
