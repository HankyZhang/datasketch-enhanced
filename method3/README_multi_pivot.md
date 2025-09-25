# 多枢纽（Multi-Pivot）K-Means + HNSW 说明

本文件说明 `kmeans_hnsw_multi_pivot.py` 中多枢纽扩展的设计思想、流程、参数与使用示例。它是在“方法3”单枢纽（质心）版本基础上，引入 **多个由质心派生的查询枢纽 (pivots)**，以提升父→子集合构建阶段的覆盖度与多样性。

---
## 背景动机
单枢纽版本：每个 K-Means 质心仅用一次 HNSW 查询获取近邻作为该父节点的子集合；如果某簇内部结构拉长或存在多方向延展，单一点的近邻可能覆盖不足。

多枢纽思想：围绕同一质心，派生出若干代表不同方向/形状的枢纽点，每个枢纽独立在底层 HNSW 上查询；再把所有结果合并并截断，获得更“面状”而非“点状”局部覆盖。

---
## 枢纽选择逻辑（默认三枢纽示例）
记：
- 质心向量 A（Pivot0）
- 第一次查询结果集合 S_A
- 第二次查询结果集合 S_B（基于枢纽 B）
- 第三次查询结果集合 S_C（基于枢纽 C）

流程（`pivot_selection_strategy = 'line_perp_third'`）：
1. A = 质心 (centroid)
2. 用 A 在基础 HNSW 上查询，得到 S_A
3. 选 B：在 S_A 中距 A **最远** 的点（最大欧氏距离）
4. 用 B 查询，得到 S_B
5. 选 C：在 (S_A ∪ S_B) 中选取到直线 AB **垂直距离最大** 的点
   - 对候选 X：设  \*v = B - A\*  ，将 X 投影到 AB 方向：
     - `coeff = ((X - A) · v) / (v · v)`
     - `proj = coeff * v`
     - 垂直残差：`perp = (X - A) - proj`
     - 垂距：`||perp||`
   - 取垂距最大的点为 C
   - 若 `||v||²` 极小（A≈B 退化），回退为“在候选中选距 A 最远”
6. 用 C 查询，得到 S_C
7. 合并 U = S_A ∪ S_B ∪ S_C
8. 对 U 中每个节点 u 计算：`score(u) = min( d(u, A), d(u, B), d(u, C) )`
9. 按 score 从小到大排序，取前 `k_children` 作为该质心的子集合

> 说明：排序方式是“更靠近任一枢纽优先”（收紧式）。如果想强调分散性，可在未来增加 `children_rank_mode = {'closest','diverse'}` 来改成按 `score` 逆序或其他多样性指标。

---
## 通用化到更多枢纽
- Pivot0：质心 A
- Pivot1：`S_A` 中距 A 最远（延展主径向）
- Pivot2：策略判定：若 `line_perp_third`，使用“最大垂距”获得与 AB 垂直方向覆盖；否则进入“通用贪心”
- Pivot i (i≥3，或当未启用 line_perp_third 的第三个)：在候选集合的并集中，贪心选择 **最小到现有所有 pivots 的距离** 最大的点（max-min-distance），实现广义“球面外扩”

候选池：当前已执行过的所有 pivot 查询结果集合的并集（不会额外遍历全集）。

---
## 关键参数
| 参数 | 说明 | 典型设置 |
|------|------|----------|
| `num_pivots` | 每个质心尝试的枢纽总数（≥1） | 2~4 |
| `pivot_selection_strategy` | 枢纽选择策略；当前支持 `'line_perp_third'`（第三枢纽走垂距） | `line_perp_third` |
| `pivot_overquery_factor` | 每个枢纽查询的 k 相对 `k_children` 放大倍数，合并截断前扩展候选 | 1.1~1.5 |
| `k_children` | 最终保留的子节点数 | 300~1000（视数据规模） |
| `child_search_ef` | 构建阶段 HNSW ef（未传则按经验 >=1.5*k_children） | >= k_children*1.3 |
| `diversify_max_assignments` | 限制同一底层节点被不同质心重复分配次数 | 可选 |
| `repair_min_assignments` | 修复：保证每个底层节点至少被若干父引用 | 1~3 |
| `pivot_debug` (内部字典) | 记录每个质心的 pivot id / 类型 / 集合规模 | 调试用途 |

---
## 复杂度与影响
- 构建阶段：每个质心执行 `num_pivots` 次 HNSW 查询；时间近似线性放大。
- 合并 + 评分：合并规模约 `num_pivots * k_children * pivot_overquery_factor` 的去重集合。
- 好处：
  - 增强簇内“形状”覆盖（尤其拉长或多分支分布）
  - 有望提升召回或减少对更大 `k_children` 的需求
- 代价：
  - 构建时间 / 内存增加
  - 需谨慎调节 `pivot_overquery_factor` 避免过多冗余点

---
## 使用示例
```python
from hnsw.hnsw import HNSW
from method3.kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot
import numpy as np

# 假设已经有基础数据 dataset (N, d)
N, d = 20000, 64
dataset = np.random.randn(N, d).astype(np.float32)

# 构建基础 HNSW 索引
dist = lambda a,b: np.linalg.norm(a-b)
base = HNSW(distance_func=dist, m=16, ef_construction=200)
for i, v in enumerate(dataset):
    base.insert(i, v)

mp = KMeansHNSWMultiPivot(
    base_index=base,
    n_clusters=64,
    k_children=600,
    num_pivots=3,
    pivot_selection_strategy='line_perp_third',
    pivot_overquery_factor=1.3,
    child_search_ef=900,
    repair_min_assignments=2
)

query = dataset[0]
results = mp.search(query, k=10, n_probe=8)
print(results[:5])
print(mp.get_pivot_debug()[list(mp.get_pivot_debug().keys())[0]])
```

---
## 调参建议
| 目标 | 建议策略 |
|------|----------|
| 提升召回 | 增大 `num_pivots` (2→3)、适度提高 `pivot_overquery_factor`、增大 `n_probe` |
| 降低构建耗时 | 减小 `num_pivots` 或 `pivot_overquery_factor`，控制 `n_clusters` |
| 降低内存 | 减少 `k_children`；必要时添加多样性排序（未来改进）避免冗余重复点 |
| 覆盖未分配节点 | 设置 `repair_min_assignments` ≥ 1 |
| 过多重复分配 | 设置 `diversify_max_assignments` (如 4–6) |

---
## 可能的未来扩展
1. `children_rank_mode='diverse'`：按最大最小距离或加权距离排序以提升多样性。
2. 自适应选择是否执行第三枢纽（基于第二枢纽增量覆盖收益）。
3. 将“垂距”推广到 PCA 主轴空间中的多正交方向探索。
4. 训练时统计每个新增枢纽带来的“新增节点比例”用于早停。
5. 学习型策略：用历史查询反馈（召回 / 点击）微调 pivot 选择顺序。

---
## 与单枢纽版本对比速览
| 维度 | 单枢纽 | 多枢纽 |
|------|--------|--------|
| 覆盖度 | 中 | 更高 |
| 构建耗时 | 基准 | × `num_pivots` 放大（近似） |
| 内存 | 基准 | 略增（更多候选去重后仍截断） |
| 实现复杂度 | 低 | 中 |
| 可解释性 | 简单 | 需查看 pivot_debug |

---
## 调试与验证
- 通过 `get_pivot_debug()` 查看：`pivot_ids`, `pivot_types`, `sets_sizes`, `union_size`, `final_size`。
- 若发现第三枢纽类型不是 `max_perp_AB`，可能触发了退化 fallback（说明 A 与 B 距离极近）。
- 可临时打印每个 pivots 的 pairwise 距离，确认多样性是否足够。

---
## 注意
当前主实现文件若出现参数或构造异常（例如早期损坏的 `__init__` 定义），可参考 `kmeans_hnsw_multi_pivot_new.py` 的干净结构进行修复与合并；本 README 描述的是“目标/正确”逻辑。

---
## 快速检查清单
- [x] A = 质心
- [x] B = S_A 中距 A 最远
- [x] C = (S_A ∪ S_B) 中垂距 AB 最大（退化则回退）
- [x] 各 pivot 独立查询 → 合并去重
- [x] 以最小到任一 pivot 的距离升序截断到 k_children

---
如需英文版或希望添加多样性排序模式，可在 issue / 需求中提出。
