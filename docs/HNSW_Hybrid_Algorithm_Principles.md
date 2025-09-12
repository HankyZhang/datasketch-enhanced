# HNSW Hybrid Two-Stage Retrieval System - Algorithm Principles (Updated)

> 本文档已更新，反映当前 `HybridHNSIndex` / `HybridHNSWIndex` 实际实现中新加入的：父→子映射双模式 (approx / brute)、Diversification（多样化限制）、Repair（覆盖修复）以及 Overlap 重叠统计指标。原始“原则”版本保留核心思想，这里补充准确的运行机理与约束分析。

## 1. 总体概念回顾

两阶段 (Parent → Children) 结构的目标：用一个小得多的父集合做“粗定位”，再在合并后的子候选集合中做“精排序”，以降低在线距离计算次数。然而实践中**有效候选独立覆盖度**往往成为瓶颈：若父列表高度重叠，理论候选上限 `n_probe * k_children` 被显著压缩，召回率受限。因此最新实现加入结构性控制与诊断指标。

```
Raw Theoretical Candidate Upper Bound:  n_probe * k_children
Effective Unique Candidates:            |Union_{p in ProbedParents} ChildList(p)|
```
我们关注的核心是：`Effective Unique Candidates / (n_probe * k_children)` 的折损原因：
1. 少量“密集”点反复出现（高 assignment count）
2. 父节点之间语义位置接近 → 子列表高 Jaccard 重叠
3. 部分长尾点完全未进入任何父列表 → 覆盖缺口 (coverage gap)

## 2. 架构与数据流

| 阶段 | 输入 | 处理 | 关键输出 | 关键参数 |
|------|------|------|----------|----------|
| Base Index 构建 | 原始向量全集 | 标准 HNSW 构建 | 多层图 | m, ef_construction |
| 父节点提取 | HNSW 图 | 选定层节点收集 | 父节点集合 P | parent_level |
| 父→子映射 | 父节点 + 数据集 | approx 或 brute + (Diversify + Repair) | parent_child_map | k_children, ef, diversify_max_assignments, repair_min_assignments |
| 查询阶段 Stage1 | 查询向量 q | 向量化父距离 | Top n_probe 父集合 | n_probe |
| 查询阶段 Stage2 | 父集合 & 映射 | 合并去重 + 批量距离 | Top k 结果 | k |
| 诊断/统计 | 映射 & 查询日志 | 覆盖 / 重叠 / 分配分布 | stats() 字典 | sample_pairs |

## 3. 父→子映射策略

### 3.1 Approx vs Brute

| 模式 | 流程 | 复杂度 | 适用 |
|------|------|--------|------|
| approx | 对父向量执行 HNSW 查询 (ef 控制临时扩展) 取前 k_children+1 | O(P * log N) 近似 | 大数据主用 |
| brute  | 父向量与全集批量 L2 计算排序取最邻近 | O(P * N) 精确 | 小规模/精确分析 |

### 3.2 Diversification（多样化限制）

问题：若单个数据点在多数父列表出现，unique coverage 降低。

策略：遍历父节点的初始候选序列时：
```
accepted = []
skipped = []
for cid in raw_child_ids:
    if len(accepted) == k_children: break
    if assignment_count[cid] < DIVERSIFY_MAX:   # 尚未超过阈值
        accept(cid); assignment_count[cid]+=1
    else:
        skipped.append(cid)
回填：若 accepted 不足 k_children，用 skipped 头部补齐
```
效果：抑制高频点的“霸占”行为，但可能牺牲局部最优近邻质量，需要与 k_children、n_probe 联动调参。

### 3.3 Repair（覆盖修复）

问题：长尾点可能未进入任何父列表 → 查询无法命中 → 召回上限受限。

策略：统计最终 assignment_count；对出现次数 < R_MIN 的点：计算其到所有父节点的距离，按最近父逐一插入（允许父列表长度溢出 k_children）。

权衡：保证全局最小覆盖，增加少量尾部溢出；可选限制溢出上限（后续可扩展）。

### 3.4 Overlap / Coverage 指标

构建后采样父列表对，计算 Jaccard：
```
J(L_i, L_j) = |L_i ∩ L_j| / |L_i ∪ L_j|
```
主要输出：
| 指标 | 含义 | 影响 |
|------|------|------|
| overlap_unique_fraction | 至少被一个父列表覆盖的点数 / 全集 | 低 → 召回上限低 |
| avg_assignment_count | 覆盖点平均出现次数 | 高且 unique_fraction 不升 → 冗余浪费 |
| multi_coverage_fraction | 出现次数 >1 的点比例 | 评估冗余集中度 |
| mean/median_jaccard_overlap | 父列表重叠程度 | 高 → 有效候选折损 |
| max_assignment_count | 单点最大重复次数 | 识别“热点”主导现象 |

### 3.5 有效候选规模与召回上界

设：
```
T = n_probe * k_children            # 理论上限
U = UniqueCandidatesAmongProbed     # 实际去重后
ρ = U / T                           # 利用率
```
在随机均匀分布假设下，召回率随 ρ 增大单调上升；在集中分布时即便增加 n_probe / k_children，如果 overlap 不下降，ρ 仍然受限 → 需要 Diversify。

## 4. 更新后的伪代码

### 4.1 映射构建（含 Diversify + Repair）
```text
for each parent p in Parents:
    raw = QueryChildren(p, mode=approx|brute, want=k_children+1)
    raw = raw - {p}
    if diversify_max_assignments is None:
        child_list = raw[:k_children]; inc counts
    else:
        accepted, skipped = [], []
        for cid in raw:
            if len(accepted) == k_children: break
            if assignment_count[cid] < diversify_max: accept(cid); inc
            else: skipped.append(cid)
        if len(accepted) < k_children:
            backfill = skipped[: k_children - len(accepted)]
            accept all backfill (inc counts)
        child_list = accepted (+backfill)
    parent_child_map[p] = child_list

if repair_min_assignments:
    deficit = {cid | assignment_count[cid] < repair_min}
    for cid in deficit:
        need = repair_min - assignment_count[cid]
        order = sort_parents_by_distance_to(cid)
        for pid in order while need>0:
            if cid not in parent_child_map[pid]:
                parent_child_map[pid].append(cid); inc; need--
```

### 4.2 查询流程
```text
parents = TopNProbeParents(q, n_probe)   # 向量化 L2 argpartition
pool = Union( parents ∪ child lists )    # set 合并去重
scores = DistanceBatch(q, pool)
return TopK(scores, k)
```

## 5. 参数交互与调优策略（更新）

| 情况 | 观测指标 | 调整建议 |
|------|----------|----------|
| 覆盖率低 (overlap_unique_fraction 低) | unique_fraction < 0.6 | 增大 k_children 或 启用 repair_min=1 |
| 重叠高 (mean_jaccard_overlap 高) | mean_jaccard > 0.35 | 启用/降低 diversify_max (例如 5→4→3) |
| 高频点主导 (max_assignment_count 远高于均值) | max >> avg | 启用或进一步降低 diversify_max |
| 召回仍然不足但 ρ 已高 | avg_candidate_size 接近 T | 增加 n_probe 或提高底层 base index 质量（m / ef_construction）|
| 构建过慢 | mapping_build_time 高 | 改用 approx、降低 k_children、减少 repair_min |

调优顺序建议：
1. 先获得稳定父集合 (parent_level=2 或低一级回退)
2. 设定 baseline：approx + k_children=1000 + n_probe=10
3. 查看 overlap stats；若重叠高 → 设置 diversify_max=4；复测
4. 若出现覆盖缺失 → repair_min=1
5. 增加 n_probe 评估边际收益
6. 必要时提高 k_children 或尝试 brute 小规模校准

## 6. 复杂度与新增操作影响

| 操作 | 原始复杂度 | 新增影响 |
|------|------------|----------|
| 父提取 | O(P) | 不变 |
| approx 子列表 | O(P * log N) | 与 ef 成正比 |
| brute 子列表 | O(P * N) | 精确/昂贵，仅分析用 |
| Diversify 过滤 | O(P * k_children) | 轻量，线性扫描 + 计数 |
| Repair | O(D_deficit * P) 距离排序 | 若 deficit 小则可接受 |
| Overlap 采样 | O(sample_pairs * k_children) | 默认 200 对，低成本 |

## 7. 局限再评估（相对旧文档新增）

| 局限 | 描述 | 缓解策略 |
|------|------|----------|
| 过度多样化 | 过早放弃真实最近邻 | 调高 diversify_max 或关闭，多观测 recall 曲线 |
| Repair 溢出 | 列表长度 > k_children 不均衡 | 未来可增加二次裁剪或软上限 |
| 采样 Jaccard 方差 | 父数多时 sample_pairs 200 不稳定 | 自适应增大（按 sqrt(P)）|
| 构建统计延迟 | 每次重建都计算 overlap | 缓存 + 允许 lazy 计算模式 |

## 8. 未来方向（扩展版）

1. 自适应 Diversify：基于局部密度自调每点允许出现次数
2. 分层 Repair：优先补贴密度稀疏区域（基于父距离排名权重）
3. 候选再平衡：对溢出父列表做局部截断并迁移冗余指派
4. 结构学习：用聚类/量化代替直接用 HNSW 层级作为父集合
5. 多指标联合：将 overlap 与查询真实 miss 的 ground truth 差异做回归，预测调参方向

## 9. 总结 (Updated)

新版 Hybrid 实现不再仅仅依赖“更多父列表 + 更多子候选”线性扩张召回，而是提供：
1. 结构控制（Diversify / Repair）→ 直接作用于覆盖与冗余
2. 诊断指标（Jaccard / assignment 分布 / unique fraction）→ 将“低召回”分解为“少覆盖 vs. 重叠过高”
3. 双模式映射（approx/brute）→ 在成本可控前提下做质量校准

因此调优已从黑盒试错转向“数据驱动的结构解释”。

> 实现细节请参考：`HNSW_Hybrid_Technical_Implementation.md` 中的构建与指标章节；本文聚焦算法动机与理论约束。

---
*本原则文档已同步 2025-09 实现。若后续加入分层聚类/量化父节点或自适应参数，请在 §3 / §5 / §6 / §8 扩展。*
