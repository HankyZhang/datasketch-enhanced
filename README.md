<div align="center">

# HNSW + K-Means / Multi-Pivot Hybrid Two-Stage Retrieval System

🚀 A modular experimental playground for building, comparing, and tuning multi-strategy two‑stage Approximate Nearest Neighbor (ANN) systems on top of a pure-Python HNSW implementation.

English | [中文版本](README_zh.md)

</div>

---

## 0. TL;DR

| Component | What it Does | Why it Exists |
|-----------|--------------|---------------|
| `hnsw.HNSW` | Core hierarchical navigable small world index | Fast baseline ANN |
| `hybrid_hnsw.HNSWHybrid` | Uses one HNSW level as parent layer → children via local expansion | Leverages structural hierarchy |
| `method3` (KMeans + HNSW) | Learns centroids (parents) via MiniBatchKMeans then fills children via HNSW | Balances clusters & recall |
| Multi-Pivot Strategy | Multiple pivots per centroid (far / perpendicular / max-min) | Increases diversity & coverage |
| `tune_kmeans_hnsw_optimized.py` | Unified evaluator (single / multi / hybrid / baseline) | Systematic experiment sweeps |

---

## 1. Motivation

Two-stage ANN designs (coarse partition → local refinement) reduce search cost vs flat indices while targeting high recall. This repo unifies several parent selection paradigms (HNSW level reuse, K-Means clustering, multi-pivot expansion, direct hybrid level fanout) behind a simple strategy contract, enabling apples-to-apples evaluation with shared metrics: recall, coverage, duplication, latency.

Key research questions addressed:
1. How does centroid balance (K-Means) compare with structural levels (HNSW layer) for candidate routing?
2. Does adding multiple pivots significantly improve recall/time Pareto front vs single pivot?
3. What duplication vs coverage trade-offs correlate with recall improvements?
4. How large must fanout / k_children be before diminishing returns dominate?

---

## 2. Architecture Overview

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
                        Query (vector) enters
                                        |
                    1) Coarse: distance to parents
                    2) Select top n_probe parents
                    3) Merge their child lists
                    4) Final re-rank in candidate set
```

Abstraction layers:
1. SharedContext: Extracts all node vectors + performs (optional) K-Means once.
2. Strategy: Implements `prepare(shared)` (optional global preprocessing) + `assign_children(...)` (per parent final trimming).
3. TwoStageIndex: Uniform build (loop parents → assign children → optional repair) + uniform search (`n_probe` parents → candidate union → exact re-rank).
4. Evaluator: Brute-force ground truth, recall/time measurement, correlation analysis.

---

## 3. Strategies In Detail

| Strategy | Parents Origin | Child Source | Notes |
|----------|----------------|--------------|-------|
| SinglePivot | K-Means centroid | Cluster members (distance-trim) | Lowest variance / simplest |
| MultiPivot | Centroid + far + perpendicular + max-min | HNSW neighbor overquery then pivot scoring | Diversity boosts candidate quality |
| Hybrid (Level-Based) | All nodes at HNSW level L | Local HNSW queries (fanout per parent) | No sampling fallback (strict) |
| Baseline HNSW | N/A | Direct HNSW query | Upper recall bound for given ef |

Hybrid specifics:
- Uses exact layer nodes (`_graphs[L]`) as parent set. If layer missing → raises.
- For each parent runs `query(parent_vec, k=fanout, ef=max(fanout+10,1.5*fanout))` to populate children.

MultiPivot selection flow:
1. Start w/ centroid.
2. Add farthest candidate.
3. Optionally perpendicular-extreme.
4. Fill remaining pivots via greedy max-min distance.
5. Score each candidate by min distance to any pivot.

---

## 4. Key Metrics

| Metric | Meaning | How Computed |
|--------|---------|--------------|
| recall_at_k | Total correct / (queries * k) | Intersection of result IDs vs GT top-k |
| avg_individual_recall | Mean per-query recall | Avg of (hits/k) per query |
| duplication_rate | (Duplicate assignments) / (total assignments) | Parents assigning same child multiple times |
| coverage_fraction | Unique assigned children / total base nodes | Measures reach of parent layer |
| avg_query_time_ms | Mean search latency | Time around TwoStageIndex.search |
| best_recall | Max recall across (k, n_probe) grid | From sweeps |

Correlation section computes Pearson between duplication / coverage vs best_recall to study scaling behavior.

---

## 5. Unified Evaluation Script

File: `method3/tune_kmeans_hnsw_optimized.py`

Features:
* Synthetic dataset generation
* SharedContext build (one clustering pass)
* Strategy enumeration (`--strategy single|multi|hybrid|all`)
* Grid sweep over `--k-list` and `--n-probe-list`
* Baseline HNSW recall (configurable `--baseline-ef`)
* CSV exports: per-evaluation + method summary
* Correlation analysis (duplication ↔ recall / coverage ↔ recall)

Example:
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
    --out experiment_results.json
```

Outputs:
* JSON (`experiment_results.json`): full structured metrics
* CSV (`experiment_results_evaluations.csv`): flattened per (method,k,n_probe)
* CSV (`experiment_results_methods_summary.csv`): one row per method

---

## 6. Parameters Reference

| CLI Flag | Applies | Description | Typical Range |
|----------|--------|-------------|---------------|
| `--n-clusters` | single/multi | K-Means parents count | 16–1024 |
| `--k-children` | all strategies | Target children kept per parent | 100–2000 |
| `--num-pivots` | multi | Number of pivots for scoring | 2–5 |
| `--pivot-strategy` | multi | Pivot diversity heuristic | line_perp_third / max_min_distance |
| `--hybrid-fanout` | hybrid | Children gathered per level parent | 32–512 |
| `--baseline-ef` | baseline | ef for raw HNSW query | 100–2000 |
| `--k-list` | evaluation | k recall values swept | 10,20,50 |
| `--n-probe-list` | evaluation | Parents probed at query time | 4–64 |
| `--repair-min` | all | Ensure each child assigned ≥ this count | 1–3 |

Internal (strategy):
* `fanout`: hybrid child query breadth (affects coverage vs build cost).
* `child_search_ef`: over-query width when filling centroid children (auto if None).

Tuning heuristics:
* Raise `n_probe` first if recall plateaus early at small k.
* Increase `k_children` if coverage_fraction << 1 and recall saturates.
* Reduce duplication via multi-pivot diversity if duplication_rate very high (>0.6) without recall gain.

---

## 7. Internal Data Flow

1. Extract all node vectors from `HNSW` → `SharedContext.node_vectors`.
2. Run MiniBatchKMeans (unless using level-based hybrid which overrides later).
3. For each strategy:
     - (optional) `prepare`: may replace centroids (hybrid) or pre-query children.
     - Build: loop centroids → `assign_children` → store parent→child list.
4. Search: compute distances query→centroids → top `n_probe` → union children → re-rank exact.
5. Evaluation: compare with brute-force GT, accumulate stats.

---

## 8. Extending the System

Add a new parent strategy:
```python
class MyStrategy(BaseAssignmentStrategy):
        name = "my_strategy"
        def prepare(self, shared):
                # optional global preprocessing
                ...
        def assign_children(self, cluster_id, centroid_vec, shared, k_children, child_search_ef):
                # return list[int]
                return [...]
```
Then register in the evaluator script similar to existing ones.

Potential ideas:
* Graph-aware repartition (leverage degree / centrality)
* Density-adaptive k_children (fewer children in dense regions)
* Learned routing (train lightweight network to select parents)

---

## 9. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Hybrid raises ValueError (level missing) | Requested HNSW level has no nodes | Use different layer or rebuild with higher max level |
| Very low recall | n_probe too small or k_children too low | Increase n_probe first, then k_children |
| High latency | Candidate set too large | Reduce k_children or n_probe |
| coverage_fraction << 1 | Fanout/assign insufficient | Increase fanout or enable repair_min |
| duplication_rate very high | Overlapping parents | Try multi-pivot or diversification logic (future) |

---

## 10. FAQ

Q: Why compute both recall_at_k and avg_individual_recall?  
A: Former is global hit ratio; latter reveals per-query variability (robustness).

Q: Why strict level usage in Hybrid (no sampling fallback)?  
A: Ensures experiments isolate structural layer effect without sampling noise.

Q: Can I plug a different distance metric?  
A: Yes, pass custom `distance_func` when constructing `HNSW` (any symmetric metric expected).

Q: How to persist / reload index?  
A: (Not implemented yet) — serialize `_nodes` and layer adjacency if needed.

Q: Does K-Means run every strategy?  
A: Only once (SharedContext). Hybrid may override centroids during prepare.

---

## 11. Performance Tips

* Use MiniBatchKMeans (already default) for large N; adjust `batch_size` if memory constrained.
* Warm start random seeds for reproducibility (`--seed`).
* For large parameter sweeps, isolate CPU cost: precompute ground truth for each k once (already implemented).
* If Python becomes bottleneck, vectorize distance computations (NumPy already used) or consider Numba acceleration.
* Avoid extremely large `k_children * n_probe` product (>200k) per query to keep latency predictable.

---

## 12. Roadmap (Suggested)

| Priority | Feature | Benefit |
|----------|---------|---------|
| High | Adaptive n_probe based on centroid distance gap | Dynamic speed/recall trade-off |
| High | Persistence (save/load HNSW + parent mapping) | Reuse built indices |
| Medium | Diversification limiter (max per child) in unified script | Lower duplication_rate |
| Medium | GPU-accelerated clustering & distance | Scale to >10M vectors |
| Medium | Percentile latency metrics (P50/P95) | Better SLO tracking |
| Low | Visualization (coverage vs recall curves) | Faster insight |
| Low | Learned routing (MLP gating) | Potential recall gain |

---

## 13. Repository Layout (Quick Reference)

```
datasketch-enhanced/
├── hnsw/                # Core HNSW implementation
├── hybrid_hnsw/         # Level-based hybrid wrapper
├── method3/             # K-Means + multi-pivot strategies + evaluators
├── docs/                # Algorithm & design documents
├── optimized_hnsw/      # Placeholder for performance variants
├── sift/                # Sample SIFT dataset (fvecs/ivecs) assets
└── tests/               # Automated tests
```

---

## 14. Licensing & Attribution

MIT Licensed (see [LICENSE](LICENSE)). Please cite the repository if used in research. Based on concepts from the original HNSW paper: *Malkov & Yashunin, 2016.*

---

## 15. Acknowledgments

Thanks to open-source ANN research community. Contributions (issues, PRs, benchmarks) are welcome.

---

## 16. Minimal Code Snippets

Baseline HNSW:
```python
from hnsw.hnsw import HNSW
import numpy as np
dist = lambda a,b: np.linalg.norm(a-b)
index = HNSW(distance_func=dist, m=16, ef_construction=200)
data = np.random.randn(1000,64).astype(np.float32)
for i,v in enumerate(data):
        index.insert(i,v)
results = index.query(data[0], k=10, ef=200)
```

Run unified evaluator:
```bash
python method3/tune_kmeans_hnsw_optimized.py --dataset-size 5000 --query-size 50 \
    --n-clusters 32 --k-children 300 --strategy all --k-list 10 --n-probe-list 4,8,12
```

---

Happy experimenting! 🔍
