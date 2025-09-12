## HNSW 混合两阶段检索 —— 算法原理（代码导出版）

本文档基于 `hnsw_core/hnsw_hybrid.py` 真实实现代码抽取与整理，系统阐述混合（Hybrid）两阶段检索结构的算法思想与工程逻辑。

### 1. 问题场景与动机
标准 HNSW 在每次查询时执行分层图的贪心/局部扩展导航。混合方案将高层（如第 2 层）重新解释为一层“粗粒度聚类中心”（父层），并为每个父节点预先计算一组大的局部近邻集合（子层）。查询阶段仅在选出的少量父节点的子集合并空间内做精排，从而避免全图遍历，同时通过 `n_probe` 与 `k_children` 参数调节召回/成本平衡。

### 2. 核心思想
相对于每次查询都逐层导航，改为：
1. 选定层级 `parent_level`，提取该层全部节点作为父节点集合。
2. 对每个父节点，离线执行一次 k-NN（k = `k_children`），生成其子节点列表。
3. 查询时仅执行两个暴力阶段：
   - 阶段 1：计算查询向量到全部父节点的距离，选出前 `n_probe` 个父节点。
   - 阶段 2：合并这些父节点的子节点集合，暴力计算距离并取 Top-K 作为最终结果。

### 3. 代码与概念映射
| 概念 | 代码字段 / 方法 | 说明 |
|------|----------------|------|
| 基础 HNSW 索引 | `self.base_index` | 外部已构建完成的原始索引。 |
| 父层级参数 | `parent_level` | 构造函数传入。 |
| 父节点提取 | `_extract_parent_nodes()` | 遍历目标层，跳过删除节点。 |
| 父向量存储 | `self.parent_vectors` | 在 `_precompute_child_mappings()` 中填充。 |
| 父→子映射 | `self.parent_child_map` | dict: parent_id → list(child_ids)。 |
| 子向量缓存 | `self.child_vectors` | 去重后缓存，避免重复取值。 |
| 预计算主流程 | `_precompute_child_mappings()` | 每个父节点：`base_index.query`。 |
| 阶段 1 搜索 | `_stage1_coarse_search()` | 父向量全量暴力 + 排序。 |
| 阶段 2 搜索 | `_stage2_fine_search()` | 合并子节点集合，距离排序。 |
| 对外查询接口 | `search()` | 串联阶段 1 + 2，输出 `(node_id, distance)`。 |
| 构建统计 | `self.stats` / `get_stats()` | 记录父/子数量与时间。 |

### 4. 构建流程（伪代码）
```
build_hybrid(base_index, parent_level, k_children):
    parents = 所选层未删除节点列表
    for parent in parents:
        vec_p = base_index[parent]
        parent_vectors[parent] = vec_p
        neighbors = base_index.query(vec_p, k = k_children)
        children = [nid for nid,_ in neighbors if nid != parent]
        parent_child_map[parent] = children
        将每个子节点向量缓存（若未出现）
```

复杂度（构建阶段）：
- 设 P = 父节点数；`k_children = Ck`。
- 每个父节点一次近似 k-NN 查询；整体约 P 次。
- 额外内存：O(P + U)，U 为全局唯一子节点数（通常远小于 P * Ck，因重叠）。

### 5. 查询两阶段（伪代码）
```
search(query, k, n_probe):
    parent_scores = [(dist(query, vec_p), parent_id) for parent_id]
    top_parents = 取前 n_probe 个父节点
    candidate_ids = ∪ parent_child_map[parent]  (父集合并)
    ranked = [(dist(query, child_vec), child_id) for child_id in candidate_ids]
    排序 ranked
    返回前 k
```

要点：
- 父节点本身不强制加入候选（除非因交叉被别的父节点列为子节点）。
- 阶段 1 规模仅为 P（千级别），可直接暴力。
- 阶段 2 候选规模 ≈ `n_probe * k_children` 去重后（典型几千到数万）。

### 6. 参数含义与取舍
| 参数 | 作用 | 取舍 |
|------|------|------|
| `parent_level` | 控制父节点粒度（层越高越少） | 过高：过粗召回不足；过低：父数过大导致阶段 1 变慢 |
| `k_children` | 每个父的子邻域宽度 | 增大提升召回，增加内存与阶段 2 计算 |
| `n_probe` | 参与精排的父节点数 | 增大提升召回，线性扩大候选集 |
| `k` | 最终返回数 | 只影响结果截断 |

调参建议：`k_children` 先取 500–2000；逐步升高 `n_probe`，观察召回提升边际；再微调 `parent_level` 控制 P。P 建议 < 1 万。

### 7. 正确性与边界处理
| 方面 | 当前行为 | 可改进点 |
|------|----------|----------|
| 删除节点 | 父提取时跳过 | 子集合也可二次过滤确认 |
| 重复子节点 | 使用 set 并集去重 | 可统计出现频次作诊断 |
| 自身排除 | 父不出现在自己的子列表 | 可加开关允许回退 |
| 空结构 | 无父节点返回空列表 | 可抛出异常更显式 |
| 查询点在索引中 | 可能出现在子集合 | 若需排除，可在结果阶段过滤 |

### 8. 复杂度总结
设：
- P = 父节点数
- C = 平均子节点数（≤ `k_children`）
- O = 重叠系数 (0 < O ≤ 1)
- Q = `n_probe`

阶段 1：O(P * d) 距离（d = 维度）  
阶段 2 候选规模 ≈ Q * C * O'  
阶段 2 暴力：O(Q * C * O' * d + 排序)；排序 O(M log M)，M 为候选数  
相对全量暴力 O(N * d)，M ≪ N。

### 9. 召回机理
- 丢失主要来自： (a) 目标近邻未落入任何被探测父节点的子集合；(b) 精排阶段距离计算本身无损（使用精确距离），故排序误差极小。
- 提高 `n_probe` → 覆盖面扩大；提高 `k_children` → 每个局部更密。
- 父子邻域重叠提供边界补偿机制。

### 10. 与原生 HNSW 的关系
| 原生 HNSW | 混合改造 |
|-----------|----------|
| 每次查询多层贪心导航 | 预先抽取一层代表集合 |
| 在线逐步扩展 | 离线预展开 + 查询时直接暴力局部 |
| `ef` 调节搜索宽度 | `n_probe` + `k_children` 双参数调节 |
| 在线使用图边结构 | 图结构离线用于生成父/子候选池 |

### 11. 局限
1. 阶段 1 仍为 O(P) 暴力（尚未矢量化/ANN 加速）。  
2. 子向量缓存存在额外内存，可改为惰性取值。  
3. 静态快照，不适合高频增量插入（需增量更新映射）。  
4. 无自适应权重，父节点密度差异未体现。  
5. 高重叠父节点可能造成子集合冗余。  

### 12. 可提升方向
| 领域 | 改进思路 |
|------|----------|
| 阶段 1 加速 | 构建父矩阵 + 矢量化 L2 / 小型 ANN |
| 内存优化 | 子节点仅存 ID，向量访问延迟加载 |
| 自适应探测 | 根据父距离分布动态分配探测数 |
| 动态维护 | 批量插入后后台刷新映射 |
| 混合重排 | 组合父距离与子距离做加权评分 |

### 13. 最小参考伪代码
```
class HNSWHybrid:
    def __init__(..., parent_level, k_children):
        parents = self._extract_parent_nodes()
        for p in parents:
            vec_p = base_index[p]
            self.parent_vectors[p] = vec_p
            nn = base_index.query(vec_p, k=k_children)
            self.parent_child_map[p] = [i for i,_ in nn if i != p]
            缓存子向量

    def search(q, k, n_probe):
        parents = 前 n_probe 个 (按 dist(q, parent_vec))
        cand_ids = 并集(parent_child_map[p] for p in parents)
        scored = [(dist(q, child_vectors[c]), c) for c in cand_ids]
        返回最小 k
```

### 14. 实战调参流程
1. 选 `parent_level` 使 P 约 1K–5K。
2. 设 `k_children` = 500 作为起点。
3. 扫描 `n_probe` ∈ {2,4,6,8,10,15,20}，找到召回曲线趋缓点。
4. 若召回上限偏低，提高 `k_children`，再少量复扫 `n_probe`。
5. 记录 (recall@K, candidate_size, latency) 做帕累托筛选。

### 15. 总结
该混合算法将“分层导航 + 在线扩展”的问题重写为： (1) 一次性抽取粗粒度代表集合；(2) 离线展开局部邻域；(3) 查询时执行两个受控规模的暴力计算。其本质是把部分搜索成本前移，换取查询阶段可解释、可调、可分析的延迟与召回权衡。

---
由 `hnsw_hybrid.py` 代码结构自动生成（2025年9月）。
