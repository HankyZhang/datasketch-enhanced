<div align="center">

# HNSW + K-Means / 多 Pivot 混合双阶段检索系统（中文版）

🚀 一个模块化的实验平台：在纯 Python HNSW 实现之上，对多种“父节点选择 + 两阶段检索”策略进行构建、对比与调优。

[English Version](README.md)

</div>

---

## 0. 快速概览（TL;DR）

| 组件 | 作用 | 价值 |
|------|------|------|
| `hnsw.HNSW` | 核心 HNSW 图索引 | 基线 ANN 搜索 |
| `hybrid_hnsw.HNSWHybrid` | 直接使用某一 HNSW 层级作为父层 | 复用层级结构，零聚类成本 |
| `method3` (KMeans + HNSW) | 用 K-Means 质心当父节点，再用 HNSW 填充孩子 | 聚类更均衡，召回潜力高 |
| 多 Pivot 策略 | 一个质心扩展出多个代表点进行打分 | 增加候选多样性，改善覆盖 |
| `tune_kmeans_hnsw_optimized.py` | 统一评估脚本（single/multi/hybrid/baseline） | 网格扫描+指标输出 |

---

## 1. 动机

两阶段 ANN（粗召回 + 精排）相比单层或平面结构能在保持高召回的同时降低搜索成本。本仓库把不同“父节点选择”方式（HNSW 层级、K-Means 聚类、多 Pivot、混合层级 fanout）统一成策略接口，方便公平比较：

核心问题：
1. 聚类（K-Means） vs 结构层级（HNSW Level） 谁的父集合更有效？
2. 多 Pivot 是否显著提升召回 / 时间的帕累托表现？
3. 重复分配（duplication）与覆盖率（coverage）和召回的相关性？
4. fanout / k_children 何时增大已经收益递减？

---

## 2. 架构概览

```
          +--------------------+
          |   Base HNSW Graph  |
          +--------------------+
                    |
        (Parent Selection Strategy)
          |   |            |
          |   |            +--> Hybrid (level parents)
          |   +------------+--> KMeans (centroid parents)
          +----------------+--> Multi-Pivot (centroid + extra pivots)
                    |
           Parent → Child Assignments
                    |
               TwoStageIndex
                    |
            Query (向量) 进入
                    |
          1) 粗阶段：计算到所有父的距离
          2) 选前 n_probe 个父
          3) 合并其孩子集合
          4) 精排返回 Top-k
```

分层抽象：
1. SharedContext：提取节点向量 + 可选一次性 K-Means。
2. Strategy：`prepare(shared)`（全局预处理，可选） + `assign_children`（逐父截断）。
3. TwoStageIndex：统一构建（循环父节点 → 分配孩子 → 可选修复）+ 统一检索（父筛选→候选集合→精排）。
4. Evaluator：真值构造 + 召回/时间统计 + 相关性分析。

---

## 3. 策略细节

| 策略 | 父集合来源 | 子集合来源 | 说明 |
|------|------------|------------|------|
| SinglePivot | K-Means 质心 | 聚类成员（截断） | 最简单、稳定 |
| MultiPivot | 质心 + 远点 + 垂直点 + max-min | HNSW 近邻候选后再打分 | 提升多样性 |
| Hybrid（层级） | 指定 HNSW 层 L 所有节点 | 每个父做 fanout 查询 | 不采样，层缺失报错 |
| Baseline HNSW | 无 | 直接 HNSW query | 上界/对照 |

Hybrid 特性：
- 严格使用指定层级（无回退），`fanout` 控制每个父收集的孩子数量。
- ef 未指定时：`ef = max(fanout+10, 1.5*fanout)`。

MultiPivot 流程：
1. 初始 pivot = 质心。
2. 加入最远点。
3. 尝试选择“垂直”点（最大垂距）。
4. 剩余用 max-min 迭代补满。
5. 候选按与所有 pivot 的最小距离排序截断。

---

## 4. 核心指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| recall_at_k | 总体召回 | (所有命中数)/(查询数*k) |
| avg_individual_recall | 平均单查询召回 | 每个查询命中率取均值 |
| duplication_rate | 重复分配率 | 重复孩子数 / 全部分配次数 |
| coverage_fraction | 覆盖率 | 唯一被覆盖孩子数 / 节点总数 |
| avg_query_time_ms | 平均查询耗时 | search 计时均值 |
| best_recall | 网格最优召回 | 从 sweep 里取最大 |

脚本还给出 duplication / coverage 与 best_recall 的 Pearson 相关系数。

---

## 5. 统一评估脚本

文件：`method3/tune_kmeans_hnsw_optimized.py`

功能：
* 合成数据生成
* 构造 SharedContext（一次 K-Means）
* 策略枚举（`--strategy single|multi|hybrid|all`）
* `k` 与 `n_probe` 网格扫描
* 基线 HNSW 评估（`--baseline-ef`）
* 导出 JSON + 两个 CSV
* 相关性分析

示例：
```bash
python method3/tune_kmeans_hnsw_optimized.py \
  --dataset-size 20000 \
  --query-size 100 \
  --dimension 128 \
  --n-clusters 64 \
  --k-children 400 \
  --strategy all \
  --k-list 10,20 \
  --n-probe-list 4,8,12 \
  --baseline-ef 400 \
  --out exp.json
```

输出：
* `exp.json` 全量结构化结果
* `exp_evaluations.csv` 每 (method,k,n_probe) 指标
* `exp_methods_summary.csv` 每方法摘要

---

## 6. 参数参考

| 参数 | 适用 | 描述 | 常见范围 |
|------|------|------|----------|
| `--n-clusters` | single/multi | K-Means 质心数 | 16–1024 |
| `--k-children` | 所有策略 | 每父保留子节点上限 | 100–2000 |
| `--num-pivots` | multi | Pivot 数量 | 2–5 |
| `--pivot-strategy` | multi | 第三个 pivot 策略 | line_perp_third / max_min_distance |
| `--hybrid-fanout` | hybrid | 每父查询孩子数量 | 32–512 |
| `--baseline-ef` | baseline | 基线 HNSW ef | 100–2000 |
| `--k-list` | 评估 | 召回 k 值列表 | 10,20,50 |
| `--n-probe-list` | 评估 | 查询时探测父数量 | 4–64 |
| `--repair-min` | all | 强制最少分配次数 | 1–3 |

调优提示：
- 低召回优先增大 `n_probe`；
- 覆盖率低且重复率不高 → 增大 `k_children` / `fanout`；
- 重复率高而召回提升停滞 → 尝试 multi-pivot；
- 高延迟 → 减小 `n_probe * k_children` 乘积。

---

## 7. 数据流步骤

1. 从 HNSW 提取所有点向量。
2. K-Means（如使用）生成质心/聚类成员。
3. Strategy 可在 `prepare` 阶段改写 centroids（如 Hybrid）或预先查询孩子。
4. 构建：逐父执行 `assign_children` → 形成 parent_child 映射。
5. 查询：计算 query→centroid 距离 → 选 top n_probe → 合并孩子集合去重 → 精排。
6. 评估：与真值比对 → 统计指标。

---

## 8. 扩展新策略

```python
class MyStrategy(BaseAssignmentStrategy):
    name = "my_strategy"
    def prepare(self, shared):
        # 可选：全局预处理
        ...
    def assign_children(self, cluster_id, centroid_vec, shared, k_children, child_search_ef):
        return [...]
```
注册：在评估脚本中 append 到 strategies 列表即可。

创意建议：
- 基于局部密度自适应 k_children
- 基于节点度数/中心性重新分配
- 轻量学习路由（MLP 预测父集合）

---

## 9. 常见问题排查

| 现象 | 可能原因 | 解决 |
|------|----------|------|
| Hybrid 报 ValueError | 指定层级不存在或为空 | 换层级或重建 HNSW（加深层） |
| 召回很低 | n_probe 太小 / k_children 太小 | 先增大 n_probe，再调 k_children |
| 延迟过高 | 候选集合过大 | 降低 n_probe 或 k_children |
| 覆盖率低 | fanout / k_children 不足 | 增大 fanout 或开启 repair |
| duplication 高 | 父集合高度重叠 | 用 multi-pivot 或加 diversify 逻辑 |

---

## 10. FAQ

**Q: 为什么要两个召回指标？**  
A: `recall_at_k` 是总体；`avg_individual_recall` 反映查询离散度。

**Q: Hybrid 不采样是为什么？**  
A: 保证实验结果只受层级结构影响，排除随机性。

**Q: 可以换距离度量吗？**  
A: 可以，构建 HNSW 时传入自定义 `distance_func`。

**Q: K-Means 会不会重复运行？**  
A: 只在 SharedContext 中执行一次；Hybrid 会在 prepare 阶段覆盖 centroids。

---

## 11. 性能优化建议

- 调整 MiniBatchKMeans `batch_size` 以适配内存。
- 统一设定随机种子保证复现。
- Ground Truth 只对每个 k 计算一次（脚本已实现）。
- 大规模时可考虑 Numba / Cython / GPU 加速（未来）。
- 避免极端 `n_probe * k_children` 组合（>2e5）。

---

## 12. Roadmap（建议）

| 优先级 | 特性 | 价值 |
|--------|------|------|
| 高 | 自适应 n_probe（基于距离 gap） | 动态性能/召回折中 |
| 高 | 索引持久化（保存/加载） | 复用构建结果 |
| 中 | 重复限制（每子节点最大分配次数） | 控制 duplication |
| 中 | GPU 聚类 & 批量距离 | 百万级扩展 |
| 中 | 延迟 P50/P95 指标 | SLA 评估 |
| 低 | 覆盖/召回可视化工具 | 分析更直观 |
| 低 | 学习式父路由 | 潜在召回增益 |

---

## 13. 仓库结构快速参考

```
 datasketch-enhanced/
 ├── hnsw/               # HNSW 核心实现
 ├── hybrid_hnsw/        # 层级 Hybrid 封装
 ├── method3/            # K-Means + 多策略评估
 ├── docs/               # 算法与设计文档
 ├── optimized_hnsw/     # 未来性能优化位
 ├── sift/               # SIFT 示例数据
 └── tests/              # 测试
```

---

## 14. 许可证与引用

MIT License（见 [LICENSE](LICENSE)）。若用于学术研究请引用本仓库。HNSW 原论文：*Malkov & Yashunin, 2016.*

---

## 15. 致谢

感谢开源 ANN 社区的研究与实践启发，欢迎 Issue / PR / Benchmark 贡献。

---

## 16. 最简代码片段

Baseline HNSW：
```python
from hnsw.hnsw import HNSW
import numpy as np
f = lambda a,b: np.linalg.norm(a-b)
idx = HNSW(distance_func=f, m=16, ef_construction=200)
X = np.random.randn(1000,64).astype(np.float32)
for i,v in enumerate(X):
    idx.insert(i,v)
print(idx.query(X[0], k=10, ef=200))
```

运行统一评估：
```bash
python method3/tune_kmeans_hnsw_optimized.py --dataset-size 5000 --query-size 50 \
  --n-clusters 32 --k-children 300 --strategy all --k-list 10 --n-probe-list 4,8,12
```

---

祝实验顺利！🔍
