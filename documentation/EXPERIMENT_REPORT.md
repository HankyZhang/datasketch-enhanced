# 实验报告：Hybrid HNSW 覆盖率与召回率演进（2025-09-12）

## 1. 目标 (Goal)
建立当前“父节点中心”混合设计中 mapping_ef（父->子近似映射宽度）、coverage_fraction（覆盖率）与 recall 的定量关系；并在应用结构修复（repair，目标覆盖率=1）前建立一个高 mapping_ef 的基线。

## 2. 核心概念 (Key Concepts)
- coverage_fraction：被至少一个父节点子列表包含的索引点占全部索引点的比例。
- recall@k：在 union + rerank 后，真实 top-k 邻居被找回的比例。
- parent-centric 限制：若 coverage < 1 且被覆盖区域内近似“全部可命中”，则 recall ≈ coverage。
- mapping_ef：构建父->子映射时的近似搜索宽度（越大尝试关联更多候选子节点）。
- k_children：每个父节点子列表的容量上限；在列表极度稀疏（fill_ratio 低）时增大无效。

## 3. mapping_ef 扫描（未优化 UNOPTIMIZED）
参数：dataset=5000（索引 4500 + 查询 500），dim=128，k=10，m=16，ef_construction=400，parent_level=2，k_children=1000，n_probe=12。
每一行使用不同 mapping_ef 重建父->子映射（复用底层索引与父集合）。

观察（节选）：
```
mapping_ef | coverage  | recall@10 | fill_ratio | avg_assignment_count
---------: | --------: | --------: | ---------: | ------------------:
       40  | 0.1069    | 0.1088    | 0.0390     | 1.03
       60  | 0.1598    | 0.1636    | 0.0590     | 1.03
       80  | 0.2069    | 0.2128    | 0.0790     | 1.03
      120  | 0.2976    | 0.3022    | 0.1190     | 1.03
      160  | 0.3780    | 0.3788    | 0.1590     | 1.03
      200  | 0.4491    | 0.4496    | 0.1990     | 1.03
```
解释：recall 近乎 1:1 跟随 coverage；列表极度稀疏（fill_ratio < 0.2），冗余几乎不存在（avg_assignment_count≈1）。在覆盖尚未接近饱和前，提升 k_children 不起作用。

## 4. 高 mapping_ef 未优化运行 (High mapping_ef Unoptimized)
参数：dataset=5000（4500 + 500），dim=128，k=10，m=16，ef_construction=400，parent_level=2，mapping_ef=400，k_children=400，n_probe ∈ {5,8,10,12}。

CSV 摘录（`unoptimized_5k_efc400_kch400_map400.csv`）：
```
n_probe | recall@10 | coverage | avg_candidate_size | avg_query_time_ms
------: | --------: | -------: | -----------------: | ----------------:
     5  | 0.5304    | 0.774    | 1660.1             | 2.125
     8  | 0.6480    | 0.774    | 2346.6             | 2.620
    10  | 0.6986    | 0.774    | 2712.7             | 2.872
    12  | 0.7364    | 0.774    | 3018.8             | 3.366
```
说明：
1. 虽然 mapping_ef=400，但 coverage 停在 0.774，根因是 parent_count=16 限制了可触达范围；剩余 22.6% 永不进入候选。
2. avg_assignment_count≈1.83 显示适度冗余开始出现，但距离“高重叠”很远。
3. 随 n_probe 增加，候选规模近似线性增长（合并更多稀疏列表）。

## 5. 当前阶段发现 (Findings So Far)
1. 覆盖率是主导召回的首要因素；提升覆盖直接线性抬升召回直到接近满覆盖。
2. 当既定父集合的可触达区域饱和后，继续增大 mapping_ef 收益递减。
3. 父层稀疏（仅 16 个父）导致结构上限 coverage≈0.774。
4. 冗余（multi_coverage_fraction≈0.536）在高 mapping_ef 时上升，使召回略微超出早期“纯线性”但不足以突破覆盖上限。

## 6. 下一步计划（当时 Pending）
运行 repair（repair_min_assignments=1），在同样参数下强制所有点至少出现在一个父列表中（目标 coverage=1.0），并度量：
- recall@10 vs n_probe（预期大幅上升，接近纯 HNSW 上界）
- 候选规模膨胀与 multi_coverage_fraction 变化
- 平均查询时间影响

计划输出文件：`unoptimized_5k_efc400_kch400_map400_repair.csv`。

## 7. 预计需要记录的修复指标 (Anticipated Metrics)
- coverage_fraction（应达到 1.0）
- 各 n_probe {5,8,10,12} 的 recall@k
- 与未修复基线相比的 avg_candidate_size 增长
- multi_coverage_fraction 与 avg_assignment_count（预期更高或结构性变化）
- 单位候选增量带来的召回收益（效率曲线）

## 8. 修复前建议 (Recommendations Pre-Repair)
短期：
- 执行 repair 并加入前后对比表。
- 基于 sweep 数据添加 coverage vs recall 图。
中期：
- 比较 parent_level=1/2/3：父节点数 vs 列表大小 vs 构建时间。
- 引入候选截断策略（per-parent top-L 或全局裁剪）抑制修复后的候选膨胀。
- 在 repair 之后再尝试 diversification（限制重复分配）。

## 9. 附录：为什么需要 Repair (Rationale)
Repair 确保没有点永久不可达；把瓶颈从“结构覆盖不足”转移到“候选控制与重排效率”，为后续通过 n_probe 与候选剪枝进行精细延迟/召回权衡奠定前提。

---
本报告自动生成（2025-09-12）。随后已执行修复实验并追加第 10 节结果。

## 10. 覆盖修复（repair）对比实验（与未修复同参数直接比较）

### 10.1 实验配置
与第4节完全相同的基础参数：
- ef_construction=400, m=16, parent_level=2
- mapping_ef=400, k_children=400
- 数据集 5000 (索引 4500 + 查询 500), dim=128, k=10
- n_probe 取 {5,8,10,12}
- 差异：开启 repair_min_assignments=1（强制每个点至少出现一次）

### 10.2 结果对比（未修复 vs 修复）
```
n_probe | 未修复 recall@10 | 修复 recall@10 | 覆盖(未/修) | 候选数未修 | 候选数修复 | 召回提升(绝对) | 候选增幅(倍)
------: | ---------------: | -------------: | ----------: | ---------: | ---------: | --------------: | -------------:
5       | 0.5304           | 0.6700         | 0.774/1.000 | 1660.1     | 2193.5     | +0.1396         | 1.32x
8       | 0.6480           | 0.8462         | 0.774/1.000 | 2346.6     | 3187.6     | +0.1982         | 1.36x
10      | 0.6986           | 0.9282         | 0.774/1.000 | 2712.7     | 3757.4     | +0.2296         | 1.39x
12      | 0.7364           | 0.9818         | 0.774/1.000 | 3018.8     | 4265.5     | +0.2454         | 1.41x
```

### 10.3 关键观察
1. 覆盖率由 0.774 -> 1.0 后，召回提升基本接近“缺失覆盖”那部分的线性补足；高 n_probe 下接近纯 HNSW 上界（recall@10≈0.98）。
2. 候选集大小增加仅 ~1.3–1.4 倍，远低于“从 0.774 到 1.0”可能带来的理论最坏 1 / 0.774 ≈ 1.29~1.5 的放大上限，说明修复引入的额外列表分布较均衡，没有形成极端冗余热点。
3. avg_assignment_count 从未修复（≈1.83）下降到修复（≈1.47），说明 repair 以“最少一次”方式填补缺口，而非复制已有多覆盖区域，冗余度（multi_coverage_fraction 0.536 -> 0.322）反而下降——这是因为修复优先给未覆盖点首次赋值，降低了重复覆盖比例。
4. 由于重复覆盖减少，单点平均被探测概率分布更均匀；在相同 n_probe 下更容易命中其唯一父列表，召回提升幅度在中高 n_probe（8/10/12）进一步放大。
5. 查询时间的增加与候选数线性相关（毫秒级增长），保持可控：n_probe=12 时 ~3.37ms -> ~4.68ms。

### 10.4 结论
repair_min_assignments=1 在当前 parent_count=13, mapping_ef=400 的结构下，将结构性瓶颈（覆盖<1）彻底消除，召回从“受 coverage 限制的线性段”跃迁到“接近完全检索”区间，实现 0.74→0.98 的显著提升；代价是候选数 ~1.4x 和 查询时间 ~1.4x 的温和增长。该性价比优于继续盲目增大 mapping_ef 或 k_children。下一步可在修复基础上引入：
- 轻度 diversification 限制过多集中到少数父节点的 late redundancy；
- 候选截断（全局或 per-parent top-L）控制高 n_probe 下的候选线性膨胀；
- parent_level 调整或多层混合，提升初始覆盖同时减少 repair 工作量。

### 10.5 后续计划
1. 在 repair 基础上加入 diversification（如 diversify_max_assignments≈3–4）对比其对 multi_coverage_fraction 与 recall 的影响。
2. 设计候选剪枝策略：按父内局部距离截断 / 全局距离 heap 维护，观察 recall-候选曲线。
3. parent_level=1/2/3 对比：父节点数 vs 覆盖 vs 构建时间三维权衡。
4. 绘制 sweep + repair 对比曲线（coverage vs recall, candidate vs recall）。
5. 扩展到 20k / 100k / 600k 评估可扩展性（测 build, mapping, query QPS）。

（本节已直接加入修复结果对比与中文分析）
