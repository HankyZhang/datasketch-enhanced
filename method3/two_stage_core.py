"""
Unified Two-Stage Retrieval Core

Goal: eliminate duplicated logic across HybridHNSW, KMeansHNSW, KMeansHNSWMultiPivot.
Focus: shared pipeline + pluggable parent selection + optional multi-pivot expansion.

Design Principles:
- Single data class TwoStageParents describing parent nodes (ids, vectors)
- Strategy objects/functions: parent_selector(base_index, **cfg) -> TwoStageParents
    * hnsw_level: use existing higher HNSW level nodes as parents
    * kmeans: run MiniBatchKMeans to derive centroids as virtual parents
- Child assignment unified: for each parent vector, run HNSW query to collect candidates
  then (optional) multi-pivot: expand candidate set by extra pivot queries derived from parent neighborhood
- Optional diversification + repair (minimal) kept simple

This module is intentionally minimal (thin abstraction) so existing evaluators can gradually migrate.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Hashable, Callable, Optional, Tuple
import numpy as np
from sklearn.cluster import MiniBatchKMeans

try:
    from hnsw.hnsw import HNSW  # type: ignore
except Exception:  # fallback placeholder
    HNSW = object  # type: ignore

@dataclass
class TwoStageParents:
    parent_ids: List[Hashable]
    parent_vectors: np.ndarray  # shape (P, D)


def select_parents_hnsw_level(base_index: HNSW, level: int) -> TwoStageParents:
    if level >= len(base_index._graphs):  # fallback highest
        level = len(base_index._graphs) - 1
    ids = [nid for nid in base_index._graphs[level] if nid in base_index and not base_index._nodes[nid].is_deleted]
    vecs = np.vstack([base_index[nid] for nid in ids]) if ids else np.zeros((0, base_index._dim), dtype=np.float32)
    return TwoStageParents(ids, vecs)


def select_parents_kmeans(base_index: HNSW, n_clusters: int, random_state: int = 42) -> TwoStageParents:
    # extract vectors
    vectors = [base_index[nid] for nid in base_index if nid in base_index]
    data = np.vstack(vectors)
    mbk = MiniBatchKMeans(n_clusters=min(n_clusters, len(data)), random_state=random_state, batch_size=min(1024, len(data)))
    mbk.fit(data)
    centroids = mbk.cluster_centers_
    parent_ids = [f"centroid_{i}" for i in range(len(centroids))]
    return TwoStageParents(parent_ids, centroids)


class TwoStageRetriever:
    def __init__(
        self,
        base_index: HNSW,
        parent_selector: Callable[..., TwoStageParents],
        parent_selector_kwargs: Dict,
        k_children: int = 200,
        child_search_ef: Optional[int] = None,
        diversify_max_assignments: Optional[int] = None,
        repair_min_assignments: Optional[int] = 1,
        multi_pivot: bool = False,
        num_pivots: int = 3,
        pivot_overquery_factor: float = 1.5,
    ):
        self.base_index = base_index
        self.k_children = k_children
        self.diversify_max_assignments = diversify_max_assignments
        self.repair_min_assignments = repair_min_assignments
        self.multi_pivot = multi_pivot and num_pivots > 1
        self.num_pivots = num_pivots
        self.pivot_overquery_factor = pivot_overquery_factor
        self.distance = base_index._distance_func

        # ef heuristic
        if child_search_ef is None:
            min_ef = max(k_children + 50, int(k_children * 1.2))
            adaptive = min(int(len(base_index)*0.1), k_children*2)
            self.child_search_ef = max(min_ef, adaptive)
        else:
            self.child_search_ef = child_search_ef

        # select parents
        parents = parent_selector(base_index, **parent_selector_kwargs)
        self.parent_ids = parents.parent_ids
        self.parent_vectors = parents.parent_vectors
        self.parent_child_map: Dict[Hashable, List[Hashable]] = {}
        self.child_vectors: Dict[Hashable, np.ndarray] = {}
        self._build_children()

    def _query_nodes(self, vec, k, ef):
        try:
            return self.base_index.query(vec, k=k, ef=ef)
        except TypeError:
            return self.base_index.query(vec, k=k)

    def _build_children(self):
        from math import ceil
        over_k = int(min(self.k_children * 4, ceil(self.k_children * self.pivot_overquery_factor)))
        need_counts = (self.diversify_max_assignments is not None) or (self.repair_min_assignments is not None)
        assignment_counts: Dict[Hashable,int] = {} if need_counts else None
        for pid, pvec in zip(self.parent_ids, self.parent_vectors):
            candidate_ids = set()
            # pivot 0
            pivots = [pvec]
            if self.multi_pivot:
                # gather initial neighborhood to derive pivots (simple farthest logic)
                first = self._query_nodes(pvec, k=over_k, ef=self.child_search_ef)
                ordered = [nid for nid,_ in first if nid in self.base_index]
                vecs = [self.base_index[nid] for nid in ordered]
                if len(vecs) >= 2:
                    # pick farthest from centroid
                    dists = np.linalg.norm(np.vstack(vecs) - pvec, axis=1)
                    far_idx = int(np.argmax(dists))
                    pivots.append(vecs[far_idx])
                # third pivot: farthest from existing pivots by min-distance
                if len(vecs) >= 3 and self.num_pivots >= 3:
                    pv_mat = np.vstack(pivots)
                    best_vec = None; best_score = -1
                    for v in vecs:
                        md = np.min(np.linalg.norm(pv_mat - v, axis=1))
                        if md > best_score:
                            best_score = md; best_vec = v
                    if best_vec is not None:
                        pivots.append(best_vec)
                # extra pivots if needed
                while len(pivots) < self.num_pivots and len(vecs) > len(pivots):
                    pv_mat = np.vstack(pivots)
                    best_vec = None; best_score = -1
                    for v in vecs:
                        md = np.min(np.linalg.norm(pv_mat - v, axis=1))
                        if md > best_score:
                            best_score = md; best_vec = v
                    if best_vec is None: break
                    pivots.append(best_vec)
            # collect candidates from pivots
            for pv in pivots:
                res = self._query_nodes(pv, k=over_k, ef=self.child_search_ef)
                for nid,_ in res:
                    candidate_ids.add(nid)
            # score candidates by distance to closest pivot
            pivot_mat = np.vstack(pivots)
            scored: List[Tuple[float, Hashable]] = []
            for nid in candidate_ids:
                vec = self.base_index[nid]
                d = np.min(np.linalg.norm(pivot_mat - vec, axis=1))
                scored.append((d, nid))
            scored.sort()
            top = [nid for _, nid in scored[:self.k_children]]
            # diversification optional (simple cap on occurrences)
            if assignment_counts is not None and self.diversify_max_assignments is not None:
                filtered = []
                for nid in top:
                    c = assignment_counts.get(nid,0)
                    if c < self.diversify_max_assignments:
                        filtered.append(nid); assignment_counts[nid]=c+1
                top = filtered
            else:
                if assignment_counts is not None:
                    for nid in top:
                        assignment_counts[nid]=assignment_counts.get(nid,0)+1
            self.parent_child_map[pid]=top
            for nid in top:
                if nid not in self.child_vectors:
                    self.child_vectors[nid]=self.base_index[nid]
        # repair minimal
        if self.repair_min_assignments and assignment_counts is not None:
            all_nodes = set(self.base_index.keys())
            assigned = set(assignment_counts.keys())
            unassigned = all_nodes - assigned
            for nid in unassigned:
                # assign to closest parent
                vec = self.base_index[nid]
                dists = np.linalg.norm(self.parent_vectors - vec, axis=1)
                idx = int(np.argmin(dists))
                pid = self.parent_ids[idx]
                if nid not in self.parent_child_map[pid]:
                    self.parent_child_map[pid].append(nid)
                    self.child_vectors[nid]=vec
        # stats
        self.coverage_fraction = len(self.child_vectors)/len(self.base_index) if len(self.base_index)>0 else 0.0

    def search(self, query_vec: np.ndarray, k: int = 10, n_probe: int = 10):
        # stage 1: select parents
        diffs = self.parent_vectors - query_vec
        dists = np.linalg.norm(diffs, axis=1)
        probe_idx = np.argsort(dists)[:n_probe]
        candidate_ids = set()
        for idx in probe_idx:
            for cid in self.parent_child_map.get(self.parent_ids[idx], []):
                candidate_ids.add(cid)
        scored = []
        for nid in candidate_ids:
            vec = self.child_vectors[nid]
            d = self.distance(vec, query_vec)
            scored.append((d, nid))
        scored.sort()
        return [(nid, d) for d, nid in scored[:k]]
