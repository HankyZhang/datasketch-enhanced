## HNSW Hybrid Two-Stage Retrieval – Algorithm Principles (Code-Derived)

This document distills the algorithmic principles of the hybrid two-stage retrieval system directly from the implementation in `hnsw_core/hnsw_hybrid.py`.

### 1. Problem Setting & Motivation
Standard HNSW performs hierarchical graph navigation per query. The hybrid approach repurposes one higher layer (e.g. level 2) as a coarse clustering layer ("parent layer"), then precomputes large local neighborhoods ("child layer") for each parent. Query-time search is restricted to the union of children for the top-n parent candidates, reducing full-graph traversal cost while enabling tunable recall via `n_probe` and `k_children`.

### 2. Core Idea
Instead of walking the full multi-level HNSW for every query:
1. Extract parent node IDs from a chosen level `parent_level`.
2. For each parent, pre-run a k-NN over the base index to materialize its top `k_children` neighbors → child candidate pool.
3. At query time, only two brute-force phases occur:
   - Stage 1: Distance from query to all parents → select `n_probe` closest.
   - Stage 2: Merge children from those parents, compute distances, return top-K.

### 3. Mapping Code → Concept
| Concept | Code Attribute / Method | Notes |
|---------|-------------------------|-------|
| Base HNSW index | `self.base_index` | Provided externally; already built. |
| Parent level selection | `parent_level` | Passed into constructor. |
| Parent extraction | `_extract_parent_nodes()` | Iterates target layer, filters deleted nodes. |
| Parent vectors store | `self.parent_vectors` | Filled during `_precompute_child_mappings()`. |
| Parent→children map | `self.parent_child_map` | Dict: parent_id → list(child_ids). |
| Child vector cache | `self.child_vectors` | Avoids repeat fetch. Only stored once globally. |
| Precomputation driver | `_precompute_child_mappings()` | Runs per parent: `base_index.query(parent_vector, k_children)`. |
| Stage 1 search | `_stage1_coarse_search()` | Brute force over parent vectors (distance, parent_id) sorting. |
| Stage 2 search | `_stage2_fine_search()` | Set-union child IDs from selected parents, distance ranking. |
| Public search API | `search()` | Calls Stage 1 then Stage 2, returns `(node_id, distance)`. |
| Statistics | `self.stats` + `get_stats()` | Construction timing and counts. |

### 4. Construction Pipeline
Pseudocode (simplified):
```
build_hybrid(base_index, parent_level, k_children):
    parents = extract all node_ids in base_index._graphs[parent_level] not deleted
    for parent in parents:
        parent_vec = base_index[parent]
        parent_vectors[parent] = parent_vec
        neighbors = base_index.query(parent_vec, k = k_children)
        children = [nid for nid,_ in neighbors if nid != parent]
        parent_child_map[parent] = children
        cache each child vector once in child_vectors
```

Complexity (construction):
- Let P = number of parents, `k_children = Ck`.
- Each parent invokes a base index `query` cost ≈ O(log N) * internal factors (depends on HNSW params) but logically treated as approximate k-NN cost.
- Memory: O(P + U) vectors cached where U ≈ |unique children| ≤ P * Ck (typically far less due to overlap).

### 5. Two-Stage Query Algorithm
Pseudocode:
```
search(query, k, n_probe):
    # Stage 1
    parent_scores = [(dist(query, parent_vec), parent_id) for parent_id]
    top_parents = take n_probe smallest distances
    # Stage 2
    candidate_ids = union(parent_child_map[parent] for parent in top_parents)
    ranked = [(dist(query, child_vec), child_id) for child_id in candidate_ids]
    sort ranked
    return top k as (child_id, distance)
```

Observations:
- Parent nodes themselves are NOT forcibly reinserted into final candidate set (unless they appear as a child of another parent due to overlap).
- Stage 1 is pure brute force over a reduced set (thousands vs millions) → deterministic.
- Stage 2 brute force operates on merged child list size: `≈ n_probe * k_children` minus overlaps (often a few thousand → tens of thousands).

### 6. Parameter Roles
| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| `parent_level` | Higher level → fewer parents (coarser clusters) | Too high: under-segmentation; too low: large P slows Stage 1 |
| `k_children` | Breadth of each local region | Higher improves recall, increases memory + Stage 2 cost |
| `n_probe` | Number of parent regions explored | Higher improves recall, increases candidate size |
| `k` | Output neighbors | Affects only final truncation |

Tuning heuristic: hold `k_children` moderate (e.g. 500–2000); increase `n_probe` until diminishing recall returns; adjust `parent_level` to keep parent count manageable (<10K preferred).

### 7. Correctness & Edge Handling
| Aspect | Current Behavior | Potential Improvement |
|--------|------------------|-----------------------|
| Deleted nodes | Skipped in parent extraction | Also validate children against deletion at query time. |
| Duplicate children | Set union removes duplicates | Maintain frequency for diagnostics if desired. |
| Self-inclusion | Parent excluded from its own child list | Optionally add flag to include for fallback. |
| Empty structures | If no parents, search returns empty list | Add explicit guard & error message. |
| Query vector also in index | May appear as child → valid recall | If excluded ground truth style needed, filter by ID. |

### 8. Complexity Summary
Let:
- P = number of parents
- C = average children per parent (`k_children` or less)
- O = overlap factor (0 < O ≤ 1; 1 = no overlap)
- Q = `n_probe`

Stage 1: O(P * d) distance ops (d = dimension).  
Stage 2 candidate size ≈ Q * C * O'.  
Stage 2 brute force: O(Q * C * O' * d + sort) → sorting cost O(M log M) with M = candidate count.  
Compared to baseline full brute force O(N * d), typical M ≪ N.

### 9. Recall Dynamics
- Miss risk comes from (a) correct neighbor not in any probed parent’s child list or (b) mis-ranked in Stage 2 due to distance noise only (exact distances used, so (b) minimal).
- Increasing `n_probe` widens region coverage; increasing `k_children` densifies each region.
- Overlap between parent neighborhoods acts as a safety net to capture boundary points.

### 10. Relation to Original HNSW
| Original HNSW | Hybrid Adaptation |
|---------------|------------------|
| Multi-level greedy routing per query | Pre-extract one upper level once |
| Dynamic navigation cost each query | Fixed precomputation + cheap brute force over reduced sets |
| ef parameter tunes search breadth | `n_probe` + `k_children` jointly tune recall |
| Graph edges exploited online | Graph used offline to derive parent/child pools |

### 11. Limitations
1. Stage 1 is still O(P) brute force (no acceleration structure yet).  
2. Memory overhead for storing child vectors (can be mitigated by referencing base index directly).  
3. Static snapshot: does not handle dynamic insertions efficiently (would require incremental updates to parent mappings).  
4. No adaptive weighting—every parent equally eligible regardless of density.  
5. Potential redundancy if two parents have near-identical child lists.

### 12. Improvement Opportunities
| Area | Enhancement |
|------|-------------|
| Stage 1 speed | Build a separate compact NumPy matrix + vectorized distance or small ANN over parents |
| Memory | Store only child IDs; fetch vectors lazily from base index |
| Adaptive probing | Allocate more probes to ambiguous (close-distance) parents |
| Dynamic maintenance | Background refresh of parent-child maps after batch inserts |
| Hybrid scoring | Combine parent distance + child distance for re-ranking confidence |

### 13. Minimal Reference Pseudocode
```
class HNSWHybrid:
    def __init__(..., parent_level, k_children):
        parents = self._extract_parent_nodes()
        for p in parents:
            vec_p = base_index[p]
            self.parent_vectors[p] = vec_p
            nn = base_index.query(vec_p, k=k_children)
            self.parent_child_map[p] = [i for i,_ in nn if i != p]
            cache child vectors

    def search(q, k, n_probe):
        parents = top n_probe by dist(q, parent_vectors)
        cand_ids = union(parent_child_map[p] for p in parents)
        scored = [(dist(q, child_vectors[c]), c) for c in cand_ids]
        return k smallest scored
```

### 14. Practical Tuning Flow
1. Choose `parent_level` so P ~ 1K–5K.
2. Set `k_children` = 500 (warm start).
3. Sweep `n_probe` in {2,4,6,8,10,15,20} → pick point where recall curve flattens.
4. Increase `k_children` if recall ceiling too low; re-sweep `n_probe` lightly.
5. Record (recall@K, candidate_size, latency) triples for Pareto selection.

### 15. Summary
The hybrid algorithm converts a hierarchical navigation problem into: (1) one-off extraction of a coarse representative layer; (2) offline expansion of local neighborhoods; (3) lightweight two-tier brute-force filtering at query time. This shifts computation from per-query graph traversal to amortized precomputation, enabling controlled recall-latency trade-offs with transparent, easily profiled components.

---
Generated automatically from code structure in `hnsw_hybrid.py` (September 2025).
