"""Multi-Pivot Extension of K-Means-based Two-Stage HNSW System.

Generalizes `kmeans_hnsw.py` by querying multiple pivots (centroid-derived points)
per centroid to enlarge/diversify child assignments.

Pivot logic (num_pivots >= 1):
  Pivot 0 (A): centroid itself.
  Pivot 1 (B): farthest from A inside first query result set S_A.
  Pivot 2 (C) if strategy == 'line_perp_third': point in union(S_A,S_B) with
        maximum perpendicular distance to line AB (满足: 垂直于A-B方向上的最远点C).
        Else uses generic diversity rule below.
  Pivot i>=3 (or i>=2 for 'max_min_distance'): greedily choose candidate
        maximizing its minimum distance to already chosen pivots.

All pivot queries results are unified then ranked by minimum distance to any pivot;
top k_children kept as children for that centroid.

If num_pivots == 1 behavior reduces to single-pivot baseline.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple, Set, Optional, Hashable, Callable
from collections import defaultdict

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hnsw.hnsw import HNSW


class KMeansHNSWMultiPivot:
    """Multi-pivot variant of the K-Means + HNSW two-stage search system.

    Combines richer statistics / repair logic (原始多功能版本) 与 精简多枢纽逻辑 (clean 版本)。
    主要特性：
      - 多枢纽选择 (A:质心, B:距A最远, C:垂距AB最大 或 通用贪心, 后续: max-min-distance)
      - 可选单枢纽退化 (num_pivots=1 或 multi_pivot_enabled=False)
      - 自适应 k_children / 分配多样性限制 / 最小分配修复
      - 枢纽调试信息 (store_pivot_debug)
    """

    def __init__(
        self,
        base_index: HNSW,
        n_clusters: int = 100,
        k_children: int = 800,
        distance_func: Optional[Callable] = None,
        child_search_ef: Optional[int] = None,
        # --- multi-pivot specific ---
        num_pivots: int = 3,
        pivot_selection_strategy: str = 'line_perp_third',
        pivot_overquery_factor: float = 1.2,
        multi_pivot_enabled: bool = True,
        store_pivot_debug: bool = True,
        # --- generic method3 options ---
        include_centroids_in_results: bool = False,
        diversify_max_assignments: Optional[int] = None,
        repair_min_assignments: Optional[int] = None,
        overlap_sample: int = 50,
        # --- adaptive children ---
        adaptive_k_children: bool = False,
        k_children_scale: float = 1.5,
        k_children_min: int = 100,
        k_children_max: Optional[int] = None,
        # --- kmeans params ---
        kmeans_params: Optional[Dict] = None,
    ):
        # Base
        self.base_index = base_index
        self.n_clusters = n_clusters
        self.k_children = k_children
        self.distance_func = distance_func or base_index._distance_func

        # Adaptive children
        self.adaptive_k_children = adaptive_k_children
        self.k_children_scale = k_children_scale
        self.k_children_min = k_children_min
        self.k_children_max = k_children_max

        # Multi-pivot config
        self.num_pivots = max(1, num_pivots)
        self.pivot_selection_strategy = pivot_selection_strategy
        self.pivot_overquery_factor = max(1.0, pivot_overquery_factor)
        self.multi_pivot_enabled = multi_pivot_enabled and (self.num_pivots > 1)
        self.store_pivot_debug = store_pivot_debug

        # Other options
        self.include_centroids_in_results = include_centroids_in_results
        self.diversify_max_assignments = diversify_max_assignments
        self.repair_min_assignments = repair_min_assignments
        self.overlap_sample = overlap_sample

        # child_search_ef heuristic
        if child_search_ef is None:
            dataset_size = len(base_index)
            min_ef = max(k_children + 50, int(k_children * 1.2))
            adaptive_ef = min(int(dataset_size * 0.12), int(k_children * 2.2))
            self.child_search_ef = max(min_ef, adaptive_ef)
        else:
            self.child_search_ef = child_search_ef

        # KMeans params adaptation similar to single-pivot version
        default_kmeans_params = {
            'max_iters': 100,
            'tol': 1e-3,
            'n_init': 3,
            'init': 'k-means++',
            'random_state': 42,
            'verbose': 0,
            'batch_size': None,
        }
        if kmeans_params:
            default_kmeans_params.update(kmeans_params)
        self.kmeans_params = default_kmeans_params

        # Containers
        self.kmeans_model: Optional[MiniBatchKMeans] = None
        self._cluster_info: Dict = {}
        self.centroids: Optional[np.ndarray] = None
        self.centroid_ids: List[str] = []
        self.parent_child_map: Dict[str, List[Hashable]] = {}
        self.child_vectors: Dict[Hashable, np.ndarray] = {}
        self._centroid_matrix: Optional[np.ndarray] = None
        self._centroid_id_array: Optional[np.ndarray] = None

        # Stats tracking
        self.stats = {
            'n_clusters': n_clusters,
            'k_children': k_children,
            'child_search_ef': self.child_search_ef,
            'adaptive_k_children': adaptive_k_children,
            'num_pivots': self.num_pivots,
            'pivot_overquery_factor': self.pivot_overquery_factor,
            'pivot_selection_strategy': self.pivot_selection_strategy,
            'multi_pivot_enabled': self.multi_pivot_enabled,
            'kmeans_fit_time': 0.0,
            'child_mapping_time': 0.0,
            'total_construction_time': 0.0,
            'num_children': 0,
            'avg_children_per_centroid': 0.0,
            'coverage_fraction': 0.0,
            'avg_search_time_ms': 0.0,
            'avg_candidate_size': 0.0,
        }
        self.search_times: List[float] = []
        self.candidate_sizes: List[int] = []
        self._overlap_stats: Dict = {}
        self._pivot_debug: Dict[str, Dict] = {}

        # Build system
        self._build_kmeans_hnsw_system()

    # --------------------- Build Steps ---------------------
    def _build_kmeans_hnsw_system(self):
        print(f"Building Multi-Pivot K-Means HNSW system with {self.n_clusters} clusters...")
        start_time = time.time()
        dataset_vectors = self._extract_dataset_vectors()
        print(f"Extracted {len(dataset_vectors)} vectors from base HNSW index")

        t0 = time.time()
        self._perform_kmeans_clustering(dataset_vectors)
        self.stats['kmeans_fit_time'] = time.time() - t0
        print(f"K-Means clustering completed in {self.stats['kmeans_fit_time']:.2f}s")

        if self.adaptive_k_children:
            avg_cluster_size = len(dataset_vectors) / max(1, self.n_clusters)
            adaptive_value = int(avg_cluster_size * self.k_children_scale)
            adaptive_value = max(self.k_children_min, adaptive_value)
            if self.k_children_max is not None:
                adaptive_value = min(self.k_children_max, adaptive_value)
            if adaptive_value != self.k_children:
                print(f"Adaptive k_children adjustment: {self.k_children} -> {adaptive_value}")
                self.k_children = adaptive_value
                self.stats['k_children'] = self.k_children

        t1 = time.time()
        if self.multi_pivot_enabled:
            self._assign_children_via_hnsw_multi_pivot()
        else:
            self._assign_children_via_hnsw_single_pivot()
        self.stats['child_mapping_time'] = time.time() - t1
        print(f"Child assignment completed in {self.stats['child_mapping_time']:.2f}s")

        self._build_centroid_index()
        self.stats['total_construction_time'] = time.time() - start_time
        self._compute_mapping_diagnostics()
        self._update_child_stats()
        print(f"Multi-Pivot K-Means HNSW system built in {self.stats['total_construction_time']:.2f}s")

    def _extract_dataset_vectors(self) -> np.ndarray:
        vectors = []
        for key in self.base_index:
            if key in self.base_index:
                vectors.append(self.base_index[key])
        if not vectors:
            raise ValueError("No vectors found in base HNSW index")
        return np.vstack(vectors)

    def _perform_kmeans_clustering(self, dataset_vectors: np.ndarray):
        print(f"Running MiniBatchKMeans with {self.n_clusters} clusters...")
        params = self.kmeans_params.copy()
        if 'max_iters' in params and 'max_iter' not in params:
            params['max_iter'] = params.pop('max_iters')
        if params.get('batch_size') in (None, 0):
            params['batch_size'] = min(1024, len(dataset_vectors))
        valid_keys = {
            'n_clusters', 'init', 'max_iter', 'batch_size', 'verbose', 'compute_labels',
            'random_state', 'tol', 'max_no_improvement', 'init_size', 'n_init',
            'reassignment_ratio'
        }
        mbk_params = {k: v for k, v in params.items() if k in valid_keys}
        mbk_params['n_clusters'] = self.n_clusters
        self.kmeans_model = MiniBatchKMeans(**mbk_params)
        self.kmeans_model.fit(dataset_vectors)
        self.centroids = self.kmeans_model.cluster_centers_
        self.centroid_ids = [f"centroid_{i}" for i in range(self.n_clusters)]
        labels = getattr(self.kmeans_model, 'labels_', None)
        if labels is not None:
            cluster_sizes = np.bincount(labels, minlength=self.n_clusters)
            self._cluster_info = {
                'avg_cluster_size': float(np.mean(cluster_sizes)),
                'std_cluster_size': float(np.std(cluster_sizes)),
                'min_cluster_size': int(np.min(cluster_sizes)),
                'max_cluster_size': int(np.max(cluster_sizes)),
                'inertia': float(self.kmeans_model.inertia_),
                'n_iterations': int(getattr(self.kmeans_model, 'n_iter_', 0)),
            }
            print(
                f"MiniBatchKMeans inertia: {self._cluster_info['inertia']:.2f}; "
                f"Cluster sizes - Avg: {self._cluster_info['avg_cluster_size']:.1f}, "
                f"Min: {self._cluster_info['min_cluster_size']}, "
                f"Max: {self._cluster_info['max_cluster_size']}"
            )
        else:
            print(f"MiniBatchKMeans inertia: {self.kmeans_model.inertia_:.2f}")

    # -------------------- Child Assignment (Single & Multi Pivot) --------------------
    def _assign_children_via_hnsw_single_pivot(self):
        print(f"Assigning children via single-pivot HNSW (ef={self.child_search_ef})...")
        need_counts = (self.diversify_max_assignments is not None) or (self.repair_min_assignments is not None)
        assignment_counts = defaultdict(int) if need_counts else None
        for i, centroid_id in enumerate(self.centroid_ids):
            centroid_vector = self.centroids[i]
            neighbors = self.base_index.query(centroid_vector, k=self.k_children, ef=self.child_search_ef)
            children = []
            for node_id, distance in neighbors:
                if self.diversify_max_assignments is None:
                    children.append(node_id)
                    self.child_vectors[node_id] = self.base_index[node_id]
                    if assignment_counts is not None:
                        assignment_counts[node_id] += 1
                else:
                    if assignment_counts[node_id] < self.diversify_max_assignments:
                        children.append(node_id)
                        assignment_counts[node_id] += 1
                        self.child_vectors[node_id] = self.base_index[node_id]
            self.parent_child_map[centroid_id] = children
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Processed {i + 1}/{self.n_clusters} centroids, children={len(children)}")
        if self.repair_min_assignments:
            if assignment_counts is None:
                from collections import defaultdict as _dd
                assignment_counts = _dd(int)
                for child_list in self.parent_child_map.values():
                    for cid in child_list:
                        assignment_counts[cid] += 1
            if self.diversify_max_assignments is None:
                print(f"(Info) Diversification disabled but repair_min_assignments={self.repair_min_assignments}; running repair...")
            self._repair_child_assignments(assignment_counts)

    def _assign_children_via_hnsw_multi_pivot(self):
        print(f"Assigning children via multi-pivot HNSW (ef={self.child_search_ef}, num_pivots={self.num_pivots})...")
        if self.num_pivots == 1:
            self._assign_children_via_hnsw_single_pivot()
            return

        k_each_query = max(self.k_children, int(self.k_children * self.pivot_overquery_factor))
        eps = 1e-12

        for idx_c, centroid_id in enumerate(self.centroid_ids):
            A_vec = self.centroids[idx_c]
            pivots = [A_vec]
            pivot_ids = [centroid_id]
            pivot_types = ['centroid']

            neighbors_sets = []
            neighbors_A = self.base_index.query(A_vec, k=k_each_query, ef=self.child_search_ef)
            S_A = [nid for nid, _ in neighbors_A]
            neighbors_sets.append(S_A)
            if not S_A:
                self.parent_child_map[centroid_id] = []
                continue
            for nid in S_A:
                self.child_vectors[nid] = self.base_index[nid]

            for p_idx in range(1, self.num_pivots):
                union_candidates = list({nid for s in neighbors_sets for nid in s})
                if not union_candidates:
                    break
                if p_idx == 1:
                    # B: farthest from A
                    dist_list = []
                    for nid in S_A:
                        vec = self.base_index[nid]
                        dist_list.append((self.distance_func(A_vec, vec), nid))
                    dist_list.sort(reverse=True)
                    chosen_id = dist_list[0][1]
                    chosen_vec = self.base_index[chosen_id]
                    kind = 'farthest_from_A'
                elif p_idx == 2 and self.pivot_selection_strategy == 'line_perp_third':
                    B_vec = pivots[1]
                    v = B_vec - A_vec
                    v_norm_sq = float(np.dot(v, v))
                    if v_norm_sq < eps:
                        # fallback to farthest from A excluding existing pivots
                        dist_list = []
                        for nid in union_candidates:
                            if nid in pivot_ids:
                                continue
                            vec = self.base_index[nid]
                            dist_list.append((self.distance_func(A_vec, vec), nid))
                        dist_list.sort(reverse=True)
                        if dist_list:
                            chosen_id = dist_list[0][1]
                            chosen_vec = self.base_index[chosen_id]
                        else:
                            chosen_id = pivot_ids[-1]
                            chosen_vec = pivots[-1]
                        kind = 'fallback_max_dist_A'
                    else:
                        max_perp = -1.0
                        chosen_id = pivot_ids[-1]
                        chosen_vec = pivots[-1]
                        for nid in union_candidates:
                            if nid in pivot_ids:
                                continue
                            X = self.base_index[nid]
                            diffA = X - A_vec
                            coeff = np.dot(diffA, v) / v_norm_sq
                            proj = coeff * v
                            perp = diffA - proj
                            pd = np.linalg.norm(perp)
                            if pd > max_perp:
                                max_perp = pd
                                chosen_id = nid
                                chosen_vec = X
                        kind = 'max_perp_AB'
                else:
                    # generic: maximize min distance to previous pivots
                    best_score = -1.0
                    chosen_id = None
                    chosen_vec = None
                    for nid in union_candidates:
                        if nid in pivot_ids:
                            continue
                        vec = self.base_index[nid]
                        score = min(self.distance_func(vec, pv) for pv in pivots)
                        if score > best_score:
                            best_score = score
                            chosen_id = nid
                            chosen_vec = vec
                    if chosen_id is None:
                        break
                    kind = 'max_min_distance'

                pivots.append(chosen_vec)
                pivot_ids.append(chosen_id)
                pivot_types.append(kind)

                neighbors_new = self.base_index.query(chosen_vec, k=k_each_query, ef=self.child_search_ef)
                S_new = [nid for nid, _ in neighbors_new]
                for nid in S_new:
                    self.child_vectors[nid] = self.base_index[nid]
                neighbors_sets.append(S_new)

            # unify
            union_ids = list({nid for s in neighbors_sets for nid in s})
            scores = []
            for nid in union_ids:
                vec = self.child_vectors[nid]
                d_min = min(self.distance_func(vec, pv) for pv in pivots)
                scores.append((d_min, nid))
            scores.sort()
            selected = [nid for _, nid in scores[:self.k_children]]
            self.parent_child_map[centroid_id] = selected

            if self.store_pivot_debug:
                self._pivot_debug[centroid_id] = {
                    'pivot_ids': pivot_ids,
                    'pivot_types': pivot_types,
                    'sets_sizes': [len(s) for s in neighbors_sets],
                    'union_size': len(union_ids),
                    'final_size': len(selected),
                    'num_pivots_used': len(pivots)
                }

            if (idx_c + 1) % 10 == 0 or idx_c == 0:
                print(f"Centroid {idx_c+1}/{self.n_clusters}: pivots={len(pivots)} union={len(union_ids)} -> kept={len(selected)}")

        if self.repair_min_assignments:
            from collections import defaultdict as _dd
            counts = _dd(int)
            for lst in self.parent_child_map.values():
                for cid in lst:
                    counts[cid] += 1
            self._repair_child_assignments(counts)

    # -------------------- Repair & Stats --------------------
    def _repair_child_assignments(self, assignment_counts: Dict[Hashable, int]):
        print(f"Repair phase: ensuring minimum {self.repair_min_assignments} assignments...")
        all_base_nodes = set(self.base_index.keys())
        assigned_nodes = set(assignment_counts.keys())
        unassigned_nodes = all_base_nodes - assigned_nodes
        under_assigned = {node_id for node_id, count in assignment_counts.items() if count < self.repair_min_assignments}
        under_assigned.update(unassigned_nodes)
        print(f"Found {len(under_assigned)} under-assigned nodes ({len(unassigned_nodes)} completely unassigned)")
        for node_id in under_assigned:
            current_assignments = assignment_counts.get(node_id, 0)
            needed = self.repair_min_assignments - current_assignments
            if needed <= 0:
                continue
            node_vector = self.base_index[node_id]
            centroid_distances = []
            for i, centroid_vector in enumerate(self.centroids):
                distance = self.distance_func(node_vector, centroid_vector)
                centroid_distances.append((distance, self.centroid_ids[i]))
            centroid_distances.sort()
            for j in range(min(needed, len(centroid_distances))):
                _, centroid_id = centroid_distances[j]
                if node_id not in self.parent_child_map[centroid_id]:
                    self.parent_child_map[centroid_id].append(node_id)
                    self.child_vectors[node_id] = self.base_index[node_id]
                    assignment_counts[node_id] += 1
        final_assigned = set(assignment_counts.keys())
        coverage = len(final_assigned) / len(all_base_nodes)
        self.stats['coverage_fraction'] = coverage
        self._update_child_stats()
        print(
            f"Repair completed. Final coverage: {coverage:.3f} ({len(final_assigned)}/{len(all_base_nodes)})"
        )

    def _build_centroid_index(self):
        if self.centroids is None:
            raise ValueError("Centroids not computed yet")
        self._centroid_matrix = self.centroids.copy()
        self._centroid_id_array = np.array(self.centroid_ids)
        print(f"Built centroid index with shape {self._centroid_matrix.shape}")

    # -------------------- Search (reuse original 2-stage idea) --------------------
    def search(self, query_vector: np.ndarray, k: int = 10, n_probe: int = 10) -> List[Tuple[Hashable, float]]:
        if self.centroids is None:
            raise ValueError("System not built yet")
        start = time.time()
        closest_centroids = self._stage1_centroid_search(query_vector, n_probe)
        results = self._stage2_child_search(query_vector, closest_centroids, k)
        elapsed = (time.time() - start) * 1000.0
        self.search_times.append(elapsed)
        self.stats['avg_search_time_ms'] = float(np.mean(self.search_times))
        return results

    def _stage1_centroid_search(self, query_vector: np.ndarray, n_probe: int) -> List[Tuple[str, float]]:
        if self._centroid_matrix is not None:
            diffs = self._centroid_matrix - query_vector
            distances = np.linalg.norm(diffs, axis=1)
            indices = np.argsort(distances)[:n_probe]
            return [(self.centroid_ids[i], distances[i]) for i in indices]
        centroid_distances = []
        for i, centroid_vector in enumerate(self.centroids):
            distance = self.distance_func(query_vector, centroid_vector)
            centroid_distances.append((distance, self.centroid_ids[i]))
        centroid_distances.sort()
        return [(cid, dist) for dist, cid in centroid_distances[:n_probe]]

    def _stage2_child_search(self, query_vector: np.ndarray, closest_centroids: List[Tuple[str, float]], k: int) -> List[Tuple[Hashable, float]]:
        candidate_children: Set[Hashable] = set()
        for centroid_id, _ in closest_centroids:
            children = self.parent_child_map.get(centroid_id, [])
            candidate_children.update(children)
        if self.include_centroids_in_results:
            for centroid_id, _ in closest_centroids:
                idx = self.centroid_ids.index(centroid_id)
                self.child_vectors[centroid_id] = self.centroids[idx]
                candidate_children.add(centroid_id)
        if not candidate_children:
            return []
        candidate_ids = list(candidate_children)
        vectors = []
        valid_ids = []
        for cid in candidate_ids:
            if cid in self.child_vectors:
                vectors.append(self.child_vectors[cid])
                valid_ids.append(cid)
        if not vectors:
            return []
        candidate_matrix = np.vstack(vectors)
        diffs = candidate_matrix - query_vector
        distances = np.linalg.norm(diffs, axis=1)
        indices = np.argsort(distances)[:k]
        results = [(valid_ids[i], distances[i]) for i in indices]
        self.candidate_sizes.append(len(valid_ids))
        self.stats['avg_candidate_size'] = float(np.mean(self.candidate_sizes))
        return results

    # -------------------- Stats / Info --------------------
    def _update_child_stats(self):
        all_children = set()
        for lst in self.parent_child_map.values():
            all_children.update(lst)
        self.stats['num_children'] = len(all_children)
        self.stats['avg_children_per_centroid'] = (
            sum(len(v) for v in self.parent_child_map.values()) / self.n_clusters if self.n_clusters else 0.0
        )

    def _compute_mapping_diagnostics(self):
        if not self.parent_child_map:
            self.stats['coverage_fraction'] = 0.0
            return
        all_children_sets = [set(children) for children in self.parent_child_map.values() if children]
        if not all_children_sets:
            self.stats['coverage_fraction'] = 0.0
            return
        union_all = set().union(*all_children_sets)
        total_base_nodes = len(self.base_index)
        coverage_fraction = len(union_all) / total_base_nodes if total_base_nodes > 0 else 0.0
        self.stats['coverage_fraction'] = coverage_fraction
        overlaps = []
        if len(all_children_sets) > 1:
            import random
            sample_pairs = min(self.overlap_sample, len(all_children_sets) * (len(all_children_sets) - 1) // 2)
            sampled_indices = random.sample(range(len(all_children_sets)), min(len(all_children_sets), 2 * int(np.sqrt(sample_pairs))))
            for i in range(len(sampled_indices)):
                for j in range(i + 1, len(sampled_indices)):
                    set_i = all_children_sets[sampled_indices[i]]
                    set_j = all_children_sets[sampled_indices[j]]
                    if set_i and set_j:
                        jaccard = len(set_i & set_j) / len(set_i | set_j)
                        overlaps.append(jaccard)
        if overlaps:
            self._overlap_stats = {
                'avg_jaccard_overlap': float(np.mean(overlaps)),
                'std_jaccard_overlap': float(np.std(overlaps)),
                'max_jaccard_overlap': float(np.max(overlaps)),
                'overlap_samples': len(overlaps)
            }
        else:
            self._overlap_stats = {
                'avg_jaccard_overlap': 0.0,
                'std_jaccard_overlap': 0.0,
                'max_jaccard_overlap': 0.0,
                'overlap_samples': 0
            }

    def get_stats(self) -> Dict:
        self._update_child_stats()
        stats = self.stats.copy()
        stats.update(self._overlap_stats)
        if self._cluster_info:
            stats.update({
                'kmeans_inertia': self._cluster_info.get('inertia'),
                'kmeans_iterations': self._cluster_info.get('n_iterations'),
                'cluster_size_stats': {
                    'avg': self._cluster_info.get('avg_cluster_size'),
                    'std': self._cluster_info.get('std_cluster_size'),
                    'min': self._cluster_info.get('min_cluster_size'),
                    'max': self._cluster_info.get('max_cluster_size')
                }
            })
        return stats

    def get_pivot_debug(self) -> Dict[str, Dict]:
        return self._pivot_debug

    # -------------------- Manual Repair API --------------------
    def run_repair(self, min_assignments: int, verbose: bool = True) -> Dict[str, float]:
        if not self.parent_child_map:
            raise RuntimeError("Parent-child map is empty; build the system first.")
        from collections import defaultdict as _dd
        assignment_counts = _dd(int)
        for centroid_id, children in self.parent_child_map.items():
            for child in children:
                assignment_counts[child] += 1
        self.repair_min_assignments = min_assignments
        if verbose:
            print(f"Running manual repair to guarantee >= {min_assignments} assignments per node...")
        self._repair_child_assignments(assignment_counts)
        self._compute_mapping_diagnostics()
        self._update_child_stats()
        return {
            'coverage_fraction': self.stats.get('coverage_fraction', 0.0),
            'avg_children_per_centroid': self.stats.get('avg_children_per_centroid', 0.0)
        }


# -------------------- Utility (same as original) --------------------

def create_synthetic_dataset(n_vectors: int, dim: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randn(n_vectors, dim).astype(np.float32)


def create_query_set(dataset: np.ndarray, n_queries: int, seed: int = 123):
    np.random.seed(seed)
    n_dataset = len(dataset)
    query_indices = np.random.choice(n_dataset, size=n_queries, replace=False)
    query_vectors = dataset[query_indices]
    return query_vectors, query_indices.tolist()


if __name__ == "__main__":
    print("Multi-Pivot K-Means HNSW System - Example Usage")
    dataset = create_synthetic_dataset(2000, 64)
    query_vectors, query_ids = create_query_set(dataset, 10)
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    for i, vector in enumerate(dataset):
        if i not in query_ids:
            base_index.insert(i, vector)
    print(f"Base HNSW index built with {len(base_index)} vectors")
    mp_system = KMeansHNSWMultiPivot(
        base_index=base_index,
        n_clusters=30,
        k_children=300,
        pivot_overquery_factor=1.2,
        kmeans_params={'verbose': 0}
    )
    qv = query_vectors[0]
    results = mp_system.search(qv, k=10, n_probe=5)
    print("Search results:")
    for r in results:
        print(r)
    print("Pivot debug (first 3 entries):")
    for k, v in list(mp_system.get_pivot_debug().items())[:3]:
        print(k, v)
