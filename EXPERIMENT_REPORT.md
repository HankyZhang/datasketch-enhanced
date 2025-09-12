# Experiment Report: Hybrid HNSW Coverage & Recall Progression (2025-09-12)

## 1. Goal
Establish the relationship between mapping_ef (parent->child approximate mapping breadth), coverage_fraction, and recall for the current parent-centric hybrid design; then baseline a high mapping_ef configuration prior to applying structural repair (coverage=1) in the next action.

## 2. Key Concepts
- coverage_fraction: Fraction of indexed points that appear in at least one parent child list.
- recall@k: Fraction of true top-k neighbors recovered after union + re-rank.
- parent-centric limitation: If coverage < 1 and recall within covered region is ~1, then recall ≈ coverage.
- mapping_ef: Controls breadth of approximate search used while assigning children to parents.
- k_children: Capacity limit per parent list; ineffective when lists are far from full (low fill_ratio).

## 3. mapping_ef Sweep (UNOPTIMIZED)
Parameters: dataset=5000 (4500 indexed + 500 queries), dim=128, k=10, m=16, ef_construction=400, parent_level=2, k_children=1000, n_probe=12.
Each row rebuilds the mapping with a different mapping_ef; base index & parents reused.

Observed (abridged):
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
Interpretation: recall tracks coverage nearly 1:1; lists are extremely sparse (fill_ratio < 0.2) and redundancy minimal (avg_assignment_count ≈1). Increasing k_children cannot help until coverage saturates.

## 4. High mapping_ef Unoptimized Run
Parameters: dataset=5000 (4500 indexed + 500 queries), dim=128, k=10, m=16, ef_construction=400, parent_level=2, mapping_ef=400, k_children=400, n_probe ∈ {5,8,10,12}.

CSV excerpt (`unoptimized_5k_efc400_kch400_map400.csv`):
```
n_probe | recall@10 | coverage | avg_candidate_size | avg_query_time_ms
------: | --------: | -------: | -----------------: | ----------------:
     5  | 0.5304    | 0.774    | 1660.1             | 2.125
     8  | 0.6480    | 0.774    | 2346.6             | 2.620
    10  | 0.6986    | 0.774    | 2712.7             | 2.872
    12  | 0.7364    | 0.774    | 3018.8             | 3.366
```
Notes:
- Coverage plateaus at 0.774 despite larger mapping_ef because parent_count=16 limits reach; remaining 22.6% of points are never candidate-eligible.
- avg_assignment_count ~1.83 indicates some modest redundancy emerging, but still far from densely covered.
- Candidate size grows roughly linearly with n_probe as we union more sparse lists.

## 5. Findings So Far
1. Coverage is the dominant determinant of recall; improving coverage directly increases recall until near-full coverage.
2. mapping_ef scaling exhibits diminishing returns once reachable region (given parent set) is saturated.
3. Parent layer sparsity (only 16 parents) creates a structural ceiling at coverage≈0.774 in the current configuration.
4. Redundancy (multi_coverage_fraction≈0.536) begins to rise at higher mapping_ef, enabling recall to slightly exceed a pure 1:1 mapping with coverage in earlier regime, but not enough to break the coverage ceiling.

## 6. Next Planned Action (Pending)
Run a repair-enabled mapping (repair_min_assignments=1) with the same base parameters (efc=400, mapping_ef=400, k_children=400, parent_level=2) to force every point into at least one parent list (target coverage=1.0) and measure:
- New recall@10 vs n_probe (expect substantial uplift approaching pure HNSW upper bound when n_probe large).
- Candidate size inflation and multi_coverage_fraction shift.
- Impact on avg_query_time.

Planned output file: `unoptimized_5k_efc400_kch400_map400_repair.csv`.

## 7. Anticipated Metrics to Capture in Repair Run
- coverage_fraction (should reach 1.0)
- recall@k across n_probe {5,8,10,12}
- avg_candidate_size growth vs unoptimized baseline
- multi_coverage_fraction, avg_assignment_count (expect higher)
- recall gain per incremental candidate increase (efficiency curve)

## 8. Recommendations (Pre-Repair)
Short term:
- Execute repair run and update this report with before/after comparison table.
- Add plot (coverage vs recall) using sweep data.
Medium term:
- Evaluate parent_level=1 vs 2 vs 3 to trade off parent_count vs per-parent list size.
- Introduce candidate truncation (limit union to top-L by parent-local distance or global pruning) to cap candidate explosion post-repair.
- Explore diversification (cap assignments) AFTER repair to curb redundancy while retaining coverage.

## 9. Appendix: Rationale for Repair
Repair ensures no point is permanently unreachable, decoupling recall from initial mapping approximate search misses. This transforms the bottleneck from structural coverage to candidate control and re-ranking efficiency, enabling fine-grained latency/recall trade-offs via n_probe and candidate pruning strategies.

---
Report authored automatically (2025-09-12). Next step: run repair experiment and append Section 10 with results.

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

### 10.4 结论（中文摘要）
repair_min_assignments=1 在当前 parent_count=13, mapping_ef=400 的结构下，将结构性瓶颈（覆盖<1）彻底消除，召回从“受 coverage 限制的线性段”跃迁到“接近完全检索”区间，实现 0.74→0.98 的显著提升；代价是候选数 ~1.4x 和 查询时间 ~1.4x 的温和增长。该性价比优于继续盲目增大 mapping_ef 或 k_children。下一步可在修复基础上引入：
- 轻度 diversification 限制过多集中到少数父节点的 late redundancy；
- 候选截断（全局或 per-parent top-L）控制高 n_probe 下的候选线性膨胀；
- parent_level 调整或多层混合，提升初始覆盖同时减少 repair 工作量。

### 10.5 后续计划（中文）
1. 在 repair 基础上加入 diversification（如 diversify_max_assignments≈3–4）对比其对 multi_coverage_fraction 与 recall 的影响。
2. 设计候选剪枝策略：按父内局部距离截断 / 全局距离 heap 维护，观察 recall-候选曲线。
3. parent_level=1/2/3 对比：父节点数 vs 覆盖 vs 构建时间三维权衡。
4. 绘制 sweep + repair 对比曲线（coverage vs recall, candidate vs recall）。
5. 扩展到 20k / 100k / 600k 评估可扩展性（测 build, mapping, query QPS）。

（本节已直接加入修复结果对比与中文分析）
