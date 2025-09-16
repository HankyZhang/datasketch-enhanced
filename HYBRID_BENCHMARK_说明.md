## Hybrid HNSW Benchmark 说明

本文件说明当前仅保留的测试脚本 `hybrid_benchmark_1k_1m.py` 的测试原理、参数含义与运行输出。

### 1. 测试目的
在 SIFT 数据集（1K 小规模 + 1M 全量）上评估 Hybrid HNSW（两级父子结构）的召回率 (Recall) 与查询延迟 (Latency)，并分析不同参数（k_children、parent_level、n_probe、diversify/repair 变体）对性能与效果的影响。

### 2. Hybrid HNSW 核心思想
1. 先对底层向量集合构建一个基础 HNSW 索引（全部向量）。
2. 再根据 `parent_level` 选取上层节点作为 “父节点集合”(parents)。这些父节点覆盖原向量空间的粗粒度区域。
3. 每个父节点分配若干子节点 ID（原向量 ID），数量由 `k_children` 控制。
4. 查询流程：
   - 通过父层快速找到若干候选父节点（数量由 `n_probe` 控制）。
   - 汇集这些父节点下的子向量集合（候选集合）。
   - 在候选集合上做精排（直接距离计算或近似 HNSW 访问，当前实现为直接求距离 Top-K）。

### 3. 关键参数
| 参数 | 作用 | 说明 |
| ---- | ---- | ---- |
| SMALL_BASE / SMALL_QUERIES | 小规模数据大小 | 用于快速验证 (1K)。|
| LARGE_BASE / LARGE_QUERIES | 大规模数据大小 | 1M 主测试；LARGE_QUERIES 可抽样减少测评时间。|
| K_GT | Ground Truth 深度 | 小规模用暴力计算；大规模直接读取官方 SIFT `sift_groundtruth.ivecs` 前 K_GT。必须 >= 评估使用的最大 k。|
| K_EVAL | 每次搜索返回的候选结果数 | 用于计算 recall@10 / recall@100。|
| parent_level | 父层层数 | 控制父节点数量（层数越高，父节点越少）。|
| k_children | 每个父节点分配的子向量数量 | 可手动设置或自动推导。|
| AUTO_K_CHILDREN | 自动模式开关 | 自动按数据规模生成 k_children 候选列表。1M: 0.5%、1%、1.5% + sqrt(N)。|
| diversify_max_assignments | Diversify 限制 | 限制同一父节点吸纳的子向量分布，提升覆盖度。0 表示关闭。|
| repair_min_assignments | Repair 最小分配 | 针对分配过少的父节点做“修复”填充。0 表示关闭。|
| n_probe | 探测父节点数量 | 实际会根据父节点数量自适应裁剪/扩展。|
| N_PROBE_EXTRA_FRACTIONS | 额外高比例采样 | 例如 0.6,0.75,0.9 对应父节点数的 60%、75%、90%。|
| N_PROBE_INCLUDE_FULL | 是否允许完全扫描父节点 | =1 时可出现 n_probe == num_parents。|

### 4. n_probe 自适应策略
1. 读取用户请求的 n_probe 列表 (如 1,2,3,5,10,20)。
2. 若某值 >= 父节点总数，则降为 `num_parents - 1`（除非允许 FULL）。
3. 自动追加固定比例目标：20%、30%、40%、50%。
4. 若设置了 `N_PROBE_EXTRA_FRACTIONS` 再追加更高比例 (60%、75%、90% 等)。
5. 若 `N_PROBE_INCLUDE_FULL=1` 则追加等于父节点总数的值。
6. 去重并排序后统一执行评测。

### 5. k_children 自动推导
| 规模 | 规则 |
| ---- | ---- |
| N ≤ 5K | 使用 5%、9%、12%（带上下限裁剪）|
| N > 5K | 使用 0.5%、1%、1.5%（上限 15K）并追加 sqrt(N)（若不重复且在范围内）|

这样在 1M 数据上典型得到 k_children ≈ [1000(√N), 5000, 10000, 15000]（排序去重后）。

### 6. 指标解释
| 指标 | 含义 |
| ---- | ---- |
| recall_at_10 / recall_at_100 | 召回率，预测结果前 K 与真实前 K 的重合比例 |
| avg_query_ms / std_query_ms | 单查询耗时均值 / 标准差（毫秒）|
| coverage | 覆盖率（候选构建阶段被分配到某些父节点的子集合覆盖全局的比例）|
| avg_candidate_size | 平均候选集合大小（影响精排成本）|
| approx_ef | Hybrid 内部近似阶段的 ef（如果定义）|
| num_parents | 当前 parent_level 下的父节点总数 |

### 7. 测试流程
1. 读取向量：`sift_base.fvecs`、`sift_query.fvecs` 与官方 `sift_groundtruth.ivecs`。
2. 小规模部分 (1K)：暴力计算 ground truth（K_GT 深度）。
3. 构建基础 HNSW：m=16, ef_construction=200；数据量 ≥ 50K 时使用批量插入减少 Python 循环开销。
4. 针对 (方法 × parent_level × k_children × 变体 × n_probe) 组合：
   - 构建 Hybrid，记录构建耗时。
   - 对抽样的查询集执行检索，统计指标。
5. 汇总写入 `hybrid_benchmark_1k_1m.json`，输出最佳召回和最快配置摘要。

### 8. 变体 (variant)
组合 diversification (divX) 与 repair (repY)：
| 标签 | 条件 | 目的 |
| ---- | ---- | ---- |
| base | 无 | 基线 |
| div3 | diversify_max_assignments=3 | 降低单父节点过度集中，提高覆盖度 |
| repX | repair_min_assignments=X | 保障冷门父节点最小子向量数量 |
（可组合为 div3+rep1 等）

### 9. 性能权衡
- 更高 k_children：提升召回 / 提高索引构建与内存成本。
- 更高 n_probe：更全面的父节点覆盖 / 增加查询延迟。
- diversification：可能略微增加构建时间，改善 long-tail 覆盖。
- repair：在父节点稀疏场景下提升稳定性。

### 10. 输出文件结构示例
```
{
  "config": { ... },
  "results": [
     {
       "scale": "1M",
       "method": "approx",
       "parent_level": 2,
       "k_children": 10000,
       "variant": "div3",
       "n_probe": 500,  
       "recall_at_10": 0.45,
       "recall_at_100": 0.70,
       "avg_query_ms": 8.23,
       ...
     }
  ],
  "summary": {
     "best_1M_recall10": { ... },
     "best_1M_speed": { ... }
  },
  "timestamp": "2025-09-16 12:34:56"
}
```
（数值示例仅为格式说明。）

### 11. 快速运行示例（PowerShell）
```powershell
$env:SMALL_BASE="1000"
$env:SMALL_QUERIES="100"
$env:LARGE_BASE="1000000"
$env:LARGE_QUERIES="300"   # 可调小加快测试
$env:K_CHILDREN=""          # 为空并启用 AUTO => 自动模式
$env:AUTO_K_CHILDREN="1"
$env:PARENT_LEVELS="2"
$env:DIV_MAX_ASSIGNMENTS="0,3"
$env:REPAIR_MIN_ASSIGNMENTS="0"
$env:N_PROBE="1,2,3,5,10,20"
$env:N_PROBE_EXTRA_FRACTIONS="0.6,0.75,0.9"
$env:N_PROBE_INCLUDE_FULL="0"
py hybrid_benchmark_1k_1m.py
```

### 12. 常见调整建议
| 目标 | 建议调整 |
| ---- | ---- |
| 提升召回 | 增加 k_children / n_probe / 启用 diversification |
| 降低延迟 | 减少 n_probe / 降低 k_children / 减少 LARGE_QUERIES 样本 |
| 控制内存 | 降低最大 k_children 上限或关闭高比例 n_probe |

### 13. 注意事项
- FULL 探测（n_probe == num_parents）仅在需要上界评估时开启。
- LARGE_QUERIES 过大将显著增加总测试时间；建议先用 100~300 验证趋势。
- 若需要 recall@K 超过 K_GT，必须增大 K_GT 并重新生成/加载 GT。

---
如需添加新的评估指标或输出格式，可在 `evaluate_hybrid` 中扩展。
