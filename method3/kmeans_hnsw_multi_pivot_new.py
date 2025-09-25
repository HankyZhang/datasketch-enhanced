from __future__ import annotations

"""Temporary duplicate clean multi-pivot implementation.
Once validated, original corrupted file can be replaced.
"""

import os, sys, time
from typing import Dict, List, Tuple, Set, Optional, Hashable, Callable
import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hnsw.hnsw import HNSW  # type: ignore


class KMeansHNSWMultiPivotClean:
    def __init__(self, base_index: HNSW, n_clusters: int = 50, k_children: int = 500,
                 distance_func: Optional[Callable] = None, num_pivots: int = 3,
                 pivot_selection_strategy: str = 'line_perp_third', child_search_ef: Optional[int] = None,
                 pivot_overquery_factor: float = 1.2):
        self.base_index = base_index
        self.n_clusters = n_clusters
        self.k_children = k_children
        self.distance_func = distance_func or base_index._distance_func
        self.num_pivots = max(1, num_pivots)
        self.pivot_selection_strategy = pivot_selection_strategy
        self.pivot_overquery_factor = max(1.0, pivot_overquery_factor)
        if child_search_ef is None:
            self.child_search_ef = max(k_children + 50, int(k_children * 1.5))
        else:
            self.child_search_ef = child_search_ef
        self.kmeans_model: Optional[MiniBatchKMeans] = None
        self.centroids: Optional[np.ndarray] = None
        self.centroid_ids: List[str] = []
        self.parent_child_map: Dict[str, List[Hashable]] = {}
        self.child_vectors: Dict[Hashable, np.ndarray] = {}
        self._pivot_debug: Dict[str, Dict] = {}
        self._build()

    def _build(self):
        data = [self.base_index[k] for k in self.base_index]
        data = np.vstack(data)
        self._run_kmeans(data)
        self._assign_children_multi()

    def _run_kmeans(self, data: np.ndarray):
        self.kmeans_model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=min(512, len(data)))
        self.kmeans_model.fit(data)
        self.centroids = self.kmeans_model.cluster_centers_
        self.centroid_ids = [f'centroid_{i}' for i in range(self.n_clusters)]

    def _assign_children_multi(self):
        if self.num_pivots == 1:
            for i,cid in enumerate(self.centroid_ids):
                cvec = self.centroids[i]
                neigh = self.base_index.query(cvec, k=self.k_children, ef=self.child_search_ef)
                kids = [nid for nid,_ in neigh]
                self.parent_child_map[cid] = kids
                for nid in kids:
                    self.child_vectors[nid] = self.base_index[nid]
            return
        k_each = max(self.k_children, int(self.k_children * self.pivot_overquery_factor))
        eps = 1e-12
        for i,cid in enumerate(self.centroid_ids):
            A = self.centroids[i]
            pivots = [A]; pivot_ids=[cid]; pivot_types=['centroid']
            nA = self.base_index.query(A, k=k_each, ef=self.child_search_ef)
            S_A = [nid for nid,_ in nA]
            if not S_A:
                self.parent_child_map[cid] = []
                continue
            for nid in S_A: self.child_vectors[nid] = self.base_index[nid]
            neighbor_sets=[S_A]
            for p_idx in range(1, self.num_pivots):
                union = list({nid for s in neighbor_sets for nid in s})
                if not union: break
                if p_idx == 1:
                    dists = [(self.distance_func(A, self.base_index[nid]), nid) for nid in S_A]
                    dists.sort(reverse=True)
                    chosen_id = dists[0][1]; chosen_vec = self.base_index[chosen_id]; kind='farthest_from_A'
                elif p_idx==2 and self.pivot_selection_strategy=='line_perp_third':
                    B = pivots[1]; v=B-A; vn=float(np.dot(v,v))
                    if vn < eps:
                        remain=[nid for nid in union if nid not in pivot_ids]
                        if not remain: break
                        dists=[(self.distance_func(A,self.base_index[nid]),nid) for nid in remain]
                        dists.sort(reverse=True)
                        chosen_id=dists[0][1]; chosen_vec=self.base_index[chosen_id]; kind='fallback_max_dist_A'
                    else:
                        best=-1.0; chosen_id=pivot_ids[-1]; chosen_vec=pivots[-1]
                        for nid in union:
                            if nid in pivot_ids: continue
                            X=self.base_index[nid]; coeff=np.dot(X-A,v)/vn; perp=(X-A)-coeff*v; pd=np.linalg.norm(perp)
                            if pd>best: best=pd; chosen_id=nid; chosen_vec=X
                        kind='max_perp_AB'
                else:
                    best=-1.0; chosen_id=None; chosen_vec=None
                    for nid in union:
                        if nid in pivot_ids: continue
                        X=self.base_index[nid]; score=min(self.distance_func(X,pv) for pv in pivots)
                        if score>best: best=score; chosen_id=nid; chosen_vec=X
                    if chosen_id is None: break
                    kind='max_min_distance'
                pivots.append(chosen_vec); pivot_ids.append(chosen_id); pivot_types.append(kind)
                n_new=self.base_index.query(chosen_vec,k=k_each,ef=self.child_search_ef)
                S_new=[nid for nid,_ in n_new]
                for nid in S_new: self.child_vectors[nid]=self.base_index[nid]
                neighbor_sets.append(S_new)
            union_ids=list({nid for s in neighbor_sets for nid in s})
            scores=[]
            for nid in union_ids:
                vec=self.child_vectors[nid]
                dmin=min(self.distance_func(vec,pv) for pv in pivots)
                scores.append((dmin,nid))
            scores.sort(); selected=[nid for _,nid in scores[:self.k_children]]
            self.parent_child_map[cid]=selected
            self._pivot_debug[cid]={'pivot_ids':pivot_ids,'pivot_types':pivot_types,'sets_sizes':[len(s) for s in neighbor_sets],'union_size':len(union_ids),'final_size':len(selected),'num_pivots_used':len(pivots)}

    def search(self, q: np.ndarray, k: int = 10, n_probe: int = 10) -> List[Tuple[Hashable,float]]:
        if self.centroids is None: raise ValueError('Not built')
        diffs=self.centroids-q
        d=np.linalg.norm(diffs,axis=1)
        idx=np.argsort(d)[:n_probe]
        cand=set()
        for i in idx: cand.update(self.parent_child_map.get(self.centroid_ids[i],[]))
        ids=list(cand)
        if not ids: return []
        mat=np.vstack([self.child_vectors[i] for i in ids])
        dist=np.linalg.norm(mat-q,axis=1)
        order=np.argsort(dist)[:k]
        return [(ids[i], dist[i]) for i in order]

    def get_pivot_debug(self):
        return self._pivot_debug


if __name__ == '__main__':
    rng=np.random.default_rng(0)
    data=rng.normal(size=(1500,32)).astype(np.float32)
    base=HNSW(distance_func=lambda a,b: np.linalg.norm(a-b), m=16, ef_construction=200)
    for i,v in enumerate(data): base.insert(i,v)
    mp=KMeansHNSWMultiPivotClean(base_index=base, n_clusters=30, k_children=300)
    q=data[0]
    print('Sample search', mp.search(q, k=5))
    print('Pivot debug entries', len(mp.get_pivot_debug()))