"""
HNSW Hybrid Two-Stage Retrieval System Implementation
=====================================================

This module implements a hybrid HNSW index structure with parent-child layers
for improved recall evaluation in plaintext environment.

Core Concept:
- Stage 1 (Parent Layer): Extract high-level nodes as cluster centers
- Stage 2 (Child Layer): Pre-computed neighbor sets for fine-grained search

Author: HankyZhang
Date: September 2025
"""

import numpy as np
import time
import pickle
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from datasketch.hnsw import HNSW


class HybridHNSWIndex:
    """Hybrid HNSW implementation with two-stage parent-child retrieval system.

    Added instrumentation for build timings, coverage, candidate sizes.
    """

    def __init__(self, distance_func=None, k_children: int = 1000, n_probe: int = 10, parent_child_method: str = 'approx'):
        # Basic parameters
        self.distance_func = distance_func or self._l2_distance
        self.k_children = k_children
        self.n_probe = n_probe
        self.parent_child_method = parent_child_method

        # Core data containers
        self.base_index = None
        self.parent_ids = []              # type: List[int]
        self.parent_child_map = {}        # type: Dict[int, List[int]]
        self.parent_vectors = {}          # type: Dict[int, np.ndarray]
        self._parent_matrix = None        # cached parent vectors matrix
        self._parent_id_index = {}        # parent id -> row index
        self.dataset = {}                 # type: Dict[int, np.ndarray]

        # Timing & metrics
        self.base_build_time = 0.0
        self.parent_extraction_time = 0.0
        self.mapping_build_time = 0.0
        self.search_times = []            # type: List[float]
        self.candidate_sizes = []         # type: List[int]

        # Overlap stats cache (filled after mapping build)
        self._overlap_stats = {}

    # --- Core helpers ---
    def _l2_distance(self, x, y):
        return np.linalg.norm(x - y)

    # --- Build Phase ---
    def build_base_index(self, dataset: Dict[int, np.ndarray], m: int = 16, ef_construction: int = 200):
        print(f"Building base HNSW index with {len(dataset)} vectors...")
        start = time.time()
        self.dataset = dataset
        self.base_index = HNSW(distance_func=self.distance_func, m=m, ef_construction=ef_construction)
        self.base_index.update(dataset)
        self.base_build_time = time.time() - start
        print(f"Base index built in {self.base_build_time:.2f}s")
        return self.base_index

    def extract_parent_nodes(self, target_level: int = 2):
        if self.base_index is None:
            raise ValueError("Base index must be built first")
        if target_level >= len(self.base_index._graphs):
            target_level = len(self.base_index._graphs) - 1
            print(f"Adjusted target_level to {target_level}")
        if target_level < 0:
            target_level = 0
        t0 = time.time()
        layer = self.base_index._graphs[target_level]
        self.parent_ids = list(layer._graph.keys())
        self.parent_vectors = {pid: self.dataset[pid] for pid in self.parent_ids}
        if self.parent_ids:
            self._parent_matrix = np.stack([self.parent_vectors[pid] for pid in self.parent_ids], axis=0)
            self._parent_id_index = {pid: i for i, pid in enumerate(self.parent_ids)}
        else:
            self._parent_matrix = None
            self._parent_id_index = {}
        self.parent_extraction_time = time.time() - t0
        print(f"Extracted {len(self.parent_ids)} parents in {self.parent_extraction_time:.2f}s (level={target_level})")
        return self.parent_ids

    def build_parent_child_mapping(
        self,
        method: str = None,
        ef: int = 50,
        brute_force_batch: int = 4096,
        diversify_max_assignments: int = None,
        repair_min_assignments: int = None,
        repair_log_limit: int = 10,
    ):
        """Build mapping from parents to child candidate lists.

        Args:
            method: 'approx' (default) or 'brute'
            ef: ef parameter when using approx method
            brute_force_batch: batch size for brute force distance blocks
            diversify_max_assignments: If set, greedy cap on how many parent lists a point can appear in during initial assignment (encourages breadth / reduces heavy overlap).
            repair_min_assignments: If set, second pass ensures every point appears in at least this many parent lists (can exceed k_children for some parents to satisfy constraint).
            repair_log_limit: Max number of individual repair additions to log (avoid spam).
        """
        method = method or self.parent_child_method
        if method not in {"approx", "brute"}:
            raise ValueError("method must be 'approx' or 'brute'")
        if not self.parent_ids:
            raise ValueError("Parent nodes must be extracted first")
        print(
            f"Building parent-child mapping method={method} k_children={self.k_children} "
            f"parents={len(self.parent_ids)} diversify_max={diversify_max_assignments} repair_min={repair_min_assignments}"
        )
        start = time.time()
        self.parent_child_map = {}
        assignment_counts = defaultdict(int)  # global counts used for diversification
        data_ids = list(self.dataset.keys())
        if method == 'brute':
            data_matrix = np.stack([self.dataset[i] for i in data_ids], axis=0)
        total = len(self.parent_ids)
        for i, pid in enumerate(self.parent_ids):
            if i % max(1, total // 10) == 0 or i == total - 1:
                pct = (i + 1) / total * 100
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed else 0
                eta = (total - i - 1) / rate if rate else 0
                print(f"  [{i+1}/{total}] {pct:5.1f}% elapsed={elapsed:.1f}s eta={eta:.1f}s")
            pvec = self.dataset[pid]
            # Raw candidate generation
            if method == 'approx':
                neighbors = self.base_index.query(pvec, k=self.k_children + 1, ef=ef)
                raw_child_ids = [nid for nid, _ in neighbors if nid != pid]
            else:  # brute
                dists = []
                for start_b in range(0, len(data_ids), brute_force_batch):
                    batch_ids = data_ids[start_b:start_b + brute_force_batch]
                    batch_vecs = data_matrix[start_b:start_b + brute_force_batch]
                    batch_d = np.linalg.norm(batch_vecs - pvec, axis=1)
                    dists.extend(zip(batch_ids, batch_d))
                dists.sort(key=lambda x: x[1])
                raw_child_ids = [cid for cid, _ in dists if cid != pid]
            # Diversification pass
            if diversify_max_assignments is not None:
                selected = []
                skipped = []
                for cid in raw_child_ids:
                    if len(selected) >= self.k_children:
                        break
                    if assignment_counts[cid] < diversify_max_assignments:
                        selected.append(cid)
                        assignment_counts[cid] += 1
                    else:
                        skipped.append(cid)
                if len(selected) < self.k_children:
                    needed = self.k_children - len(selected)
                    backfill = skipped[:needed]
                    for cid in backfill:
                        assignment_counts[cid] += 1
                    selected.extend(backfill)
                child_ids = selected
            else:
                child_ids = raw_child_ids[: self.k_children]
                for cid in child_ids:
                    assignment_counts[cid] += 1
            self.parent_child_map[pid] = child_ids

        # Optional repair to guarantee minimum assignments per point
        if repair_min_assignments is not None:
            print(f"Repair phase: ensuring each point appears in >= {repair_min_assignments} parent lists...")
            # Recompute assignment counts from final diversified lists (safer)
            assignment_counts = defaultdict(int)
            for plist in self.parent_child_map.values():
                for cid in plist:
                    assignment_counts[cid] += 1
            deficit_points = [cid for cid in self.dataset.keys() if assignment_counts[cid] < repair_min_assignments]
            print(f"  Points below threshold: {len(deficit_points)}")
            # Parent vectors matrix for fast distance ranking
            if self._parent_matrix is not None:
                parent_matrix = self._parent_matrix
            else:
                parent_matrix = np.stack([self.dataset[pid] for pid in self.parent_ids], axis=0)
            repair_logs = 0
            for cid in deficit_points:
                needed = repair_min_assignments - assignment_counts[cid]
                if needed <= 0:
                    continue
                cvec = self.dataset[cid]
                # Compute distances to all parents then add to nearest parents missing this child
                dists = np.linalg.norm(parent_matrix - cvec, axis=1)
                order = np.argsort(dists)
                for idx in order:
                    if needed <= 0:
                        break
                    pid = self.parent_ids[idx]
                    plist = self.parent_child_map[pid]
                    if cid in plist:
                        continue
                    plist.append(cid)  # allow overflow beyond k_children; keeps operation simple
                    assignment_counts[cid] += 1
                    needed -= 1
                    if repair_logs < repair_log_limit:
                        print(f"    Repaired child {cid} -> parent {pid}")
                        repair_logs += 1
            if repair_logs >= repair_log_limit:
                print("    ... repair log truncated ...")
            print("Repair phase complete")

        self.mapping_build_time = time.time() - start
        # Compute and cache overlap stats
        self._overlap_stats = self.mapping_overlap_stats()
        print(f"Parent-child mapping built in {self.mapping_build_time:.2f}s")
        return self.parent_child_map

    def mapping_overlap_stats(self, sample_pairs: int = 200) -> Dict[str, float]:
        """Compute overlap / redundancy statistics across parent child lists.

        Args:
            sample_pairs: Max number of random parent pairs to sample for Jaccard overlap
        """
        if not self.parent_child_map:
            return {
                'overlap_unique_fraction': 0.0,
                'avg_assignment_count': 0.0,
                'mean_jaccard_overlap': 0.0,
                'median_jaccard_overlap': 0.0,
                'multi_coverage_fraction': 0.0,
                'max_assignment_count': 0.0,
            }
        parent_ids = list(self.parent_child_map.keys())
        lists = [self.parent_child_map[pid] for pid in parent_ids]
        total_assignments = sum(len(lst) for lst in lists)
        assign_counts = defaultdict(int)
        for lst in lists:
            for cid in lst:
                assign_counts[cid] += 1
        unique_children = len(assign_counts)
        dataset_size = len(self.dataset) if self.dataset else 0
        multi_cov = sum(1 for v in assign_counts.values() if v > 1)
        multi_coverage_fraction = multi_cov / unique_children if unique_children else 0.0
        avg_assignment = total_assignments / unique_children if unique_children else 0.0
        max_assignment = max(assign_counts.values()) if assign_counts else 0
        # Jaccard sampling
        import random
        pair_indices = []
        P = len(parent_ids)
        if P <= 1:
            mean_j = median_j = 0.0
        else:
            if P * (P - 1) // 2 <= sample_pairs:
                # use all pairs
                for i in range(P):
                    for j in range(i + 1, P):
                        pair_indices.append((i, j))
            else:
                seen = set()
                while len(pair_indices) < sample_pairs:
                    a = random.randrange(P)
                    b = random.randrange(P)
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    pair_indices.append((a, b))
            jaccards = []
            for a, b in pair_indices:
                la = lists[a]
                lb = lists[b]
                sa = set(la)
                sb = set(lb)
                inter = len(sa & sb)
                union = len(sa | sb)
                j = inter / union if union else 0.0
                jaccards.append(j)
            if jaccards:
                mean_j = float(np.mean(jaccards))
                median_j = float(np.median(jaccards))
            else:
                mean_j = median_j = 0.0
        stats = {
            'overlap_unique_fraction': unique_children / dataset_size if dataset_size else 0.0,
            'avg_assignment_count': avg_assignment,
            'mean_jaccard_overlap': mean_j,
            'median_jaccard_overlap': median_j,
            'multi_coverage_fraction': multi_coverage_fraction,
            'max_assignment_count': float(max_assignment),
        }
        return stats

    # --- Query Phase ---
    def _find_closest_parents(self, qvec: np.ndarray) -> List[int]:
        if self._parent_matrix is None or not self.parent_ids:
            return []
        diffs = self._parent_matrix - qvec
        dists = np.linalg.norm(diffs, axis=1)
        if self.n_probe < len(dists):
            idx = np.argpartition(dists, self.n_probe)[:self.n_probe]
            idx = idx[np.argsort(dists[idx])]
        else:
            idx = np.argsort(dists)
        return [self.parent_ids[i] for i in idx[:self.n_probe]]

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        start = time.time()
        parents = self._find_closest_parents(query_vector)
        cids = set(parents)
        for pid in parents:
            cids.update(self.parent_child_map.get(pid, ()))
        if not cids:
            return []
        candidates = list(cids)
        mat = np.stack([self.dataset[c] for c in candidates], axis=0)
        dists = np.linalg.norm(mat - query_vector, axis=1)
        if k < len(dists):
            top = np.argpartition(dists, k)[:k]
            order = np.argsort(dists[top])
            top = top[order]
        else:
            top = np.argsort(dists)
        results = [(candidates[i], float(dists[i])) for i in top[:k]]
        self.search_times.append(time.time() - start)
        self.candidate_sizes.append(len(cids))
        return results

    # --- Metrics ---
    def coverage(self) -> Dict[str, float]:
        if not self.dataset or not self.parent_child_map:
            return {
                'coverage_fraction': 0.0,
                'covered_points': 0,
                'total_points': len(self.dataset),
                'parent_count': len(self.parent_ids)
            }
        covered: Set[int] = set()
        for children in self.parent_child_map.values():
            covered.update(children)
        total = len(self.dataset)
        return {
            'coverage_fraction': len(covered) / total if total else 0.0,
            'covered_points': len(covered),
            'total_points': total,
            'parent_count': len(self.parent_ids)
        }

    def stats(self) -> Dict[str, float]:
        cov = self.coverage()
        merged = {
            'base_build_time': self.base_build_time,
            'parent_extraction_time': self.parent_extraction_time,
            'mapping_build_time': self.mapping_build_time,
            'avg_search_time': float(np.mean(self.search_times)) if self.search_times else 0.0,
            'avg_candidate_size': float(np.mean(self.candidate_sizes)) if self.candidate_sizes else 0.0,
            **cov,
        }
        # add overlap stats if available
        if self._overlap_stats:
            merged.update(self._overlap_stats)
        return merged


class RecallEvaluator:
    """
    Evaluator for measuring recall performance of the hybrid HNSW system.
    """
    
    def __init__(self, dataset: Dict[int, np.ndarray], distance_func=None):
        """
        Initialize the evaluator.
        
        Args:
            dataset: Complete dataset for evaluation
            distance_func: Distance function for similarity computation
        """
        self.dataset = dataset
        self.distance_func = distance_func or self._l2_distance
        self.ground_truth_cache = {}
        
    def _l2_distance(self, x, y):
        """Default L2 distance function."""
        return np.linalg.norm(x - y)
    
    def compute_ground_truth(self, query_vectors: Dict[int, np.ndarray], k: int = 10):
        """
        Compute ground truth using brute force search.
        
        Args:
            query_vectors: Dictionary of query vectors
            k: Number of nearest neighbors to find
            
        Returns:
            Dictionary mapping query IDs to ground truth results
        """
        print(f"Computing ground truth for {len(query_vectors)} queries...")
        ground_truth = {}
        
        for i, (query_id, query_vector) in enumerate(query_vectors.items()):
            if i % 1000 == 0:
                print(f"Processing query {i+1}/{len(query_vectors)}")
            
            # Brute force search
            distances = []
            for data_id, data_vector in self.dataset.items():
                if data_id != query_id:  # Exclude query itself
                    distance = self.distance_func(query_vector, data_vector)
                    distances.append((data_id, distance))
            
            # Sort and take top k
            distances.sort(key=lambda x: x[1])
            ground_truth[query_id] = [node_id for node_id, _ in distances[:k]]
        
        self.ground_truth_cache = ground_truth
        print("Ground truth computation completed")
        return ground_truth
    
    def evaluate_recall(self,
                        hybrid_index: HybridHNSWIndex,
                        query_vectors: Dict[int, np.ndarray],
                        k: int = 10,
                        progress_interval: int = 100,
                        progress_callback=None,
                        sample_query_ids: List[int] = None) -> Dict[str, float]:
        """
        Evaluate recall performance of the hybrid index.
        
        Args:
            hybrid_index: The hybrid HNSW index to evaluate
            query_vectors: Dictionary of query vectors
            k: Number of nearest neighbors to evaluate
            progress_interval: Print partial stats every N queries
            progress_callback: Optional callable(dict) invoked on progress
            sample_query_ids: Optional subset of query IDs to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if sample_query_ids is not None:
            eval_items = [(qid, query_vectors[qid]) for qid in sample_query_ids if qid in query_vectors]
        else:
            eval_items = list(query_vectors.items())
        total_q = len(eval_items)
        print(f"Evaluating recall@{k} for {total_q} queries (progress every {progress_interval})...")
        
        # Get ground truth
        if not self.ground_truth_cache:
            # If sampling, ground truth still across full dataset for selected queries only
            ground_truth_input = {qid: qv for qid, qv in eval_items}
            ground_truth = self.compute_ground_truth(ground_truth_input, k)
        else:
            ground_truth = self.ground_truth_cache
        
        total_correct = 0
        total_possible = 0
        query_times = []
        
        for i, (query_id, query_vector) in enumerate(eval_items):
            if i % 1000 == 0 and i > 0:
                print(f"  Processing query {i+1}/{total_q}")
            
            # Get hybrid search results
            start_time = time.time()
            predicted_results = hybrid_index.search(query_vector, k)
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # Extract predicted IDs
            predicted_ids = [node_id for node_id, _ in predicted_results]
            true_ids = ground_truth[query_id]
            
            # Count correct predictions
            correct = len(set(predicted_ids) & set(true_ids))
            total_correct += correct
            total_possible += k

            if (i + 1) % progress_interval == 0 or (i + 1) == total_q:
                partial_recall = total_correct / total_possible if total_possible else 0.0
                avg_qt = float(np.mean(query_times)) if query_times else 0.0
                progress_payload = {
                    'phase': 'recall_eval',
                    'processed_queries': i + 1,
                    'total_queries': total_q,
                    'partial_recall@k': partial_recall,
                    'avg_query_time_so_far': avg_qt,
                    'k': k,
                    'k_children': hybrid_index.k_children,
                    'n_probe': hybrid_index.n_probe
                }
                print(f"    [Progress] {i+1}/{total_q} partial_recall@{k}={partial_recall:.4f} avg_q_time={avg_qt:.6f}s")
                if progress_callback:
                    try:
                        progress_callback(progress_payload)
                    except Exception as e:  # noqa
                        print(f"[WARN] progress_callback error: {e}")
        
        # Calculate metrics
        recall = total_correct / total_possible if total_possible > 0 else 0.0
        avg_query_time = np.mean(query_times)
        
        results = {
            'recall@k': recall,
            'k': k,
            'total_queries': total_q,
            'avg_query_time': avg_query_time,
            'total_correct': total_correct,
            'total_possible': total_possible,
            'k_children': hybrid_index.k_children,
            'n_probe': hybrid_index.n_probe
        }
        
        print(f"Recall@{k}: {recall:.4f}")
        print(f"Average query time: {avg_query_time:.6f} seconds")
        
        return results


def generate_synthetic_dataset(n_vectors: int = 60000, dim: int = 128) -> Dict[int, np.ndarray]:
    """
    Generate a synthetic dataset for testing.
    
    Args:
        n_vectors: Number of vectors to generate
        dim: Dimensionality of vectors
        
    Returns:
        Dictionary mapping IDs to vectors
    """
    print(f"Generating synthetic dataset with {n_vectors} vectors of dimension {dim}...")
    np.random.seed(42)  # For reproducibility
    
    dataset = {}
    for i in range(n_vectors):
        vector = np.random.randn(dim).astype(np.float32)
        # Normalize to unit length for better clustering
        vector = vector / np.linalg.norm(vector)
        dataset[i] = vector
    
    print("Dataset generation completed")
    return dataset


def create_query_set(dataset: Dict[int, np.ndarray], n_queries: int = 1000) -> Dict[int, np.ndarray]:
    """(LEGACY) Sample queries directly from dataset WITHOUT removal.

    NOTE: This keeps queries inside the base index and is kept only for backward
    compatibility with older scripts. New code SHOULD use
    :func:`split_query_set_from_dataset` to ensure queries are excluded from
    the index build for fair evaluation (Project Guide Phase 2 requirement).
    """
    print("[WARN] create_query_set() keeps queries in dataset. Prefer split_query_set_from_dataset().")
    all_ids = list(dataset.keys())
    if n_queries > len(all_ids):
        n_queries = len(all_ids)
    np.random.seed(123)
    query_ids = np.random.choice(all_ids, size=n_queries, replace=False)
    return {qid: dataset[qid] for qid in query_ids}


def split_query_set_from_dataset(dataset: Dict[int, np.ndarray], n_queries: int = 1000,
                                 seed: int = 123) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Split dataset into (dataset_without_queries, query_set) for fair evaluation.

    Ensures query vectors are NOT inserted into the base index, aligning with
    the project specification: query set must not be "seen" during index
    construction. (Project Action Guide Phase 2)

    Args:
        dataset: Full dataset mapping id->vector
        n_queries: Number of queries to sample
        seed: RNG seed for reproducibility

    Returns:
        (dataset_no_queries, query_set)
    """
    total = len(dataset)
    if n_queries >= total:
        raise ValueError("n_queries must be smaller than dataset size")
    rng = np.random.default_rng(seed)
    all_ids = list(dataset.keys())
    query_ids = set(rng.choice(all_ids, size=n_queries, replace=False))
    query_set = {qid: dataset[qid] for qid in query_ids}
    dataset_no_queries = {did: vec for did, vec in dataset.items() if did not in query_ids}
    print(f"Split dataset: {len(dataset_no_queries)} training vectors, {len(query_set)} query vectors (removed from index)")
    return dataset_no_queries, query_set


if __name__ == "__main__":
    # Configuration
    DATASET_SIZE = 60000  # Adjust as needed
    VECTOR_DIM = 128
    N_QUERIES = 1000
    K = 10

    print("=== HNSW Hybrid Two-Stage Retrieval System Evaluation (Fair Split) ===")
    print(f"Full dataset size: {DATASET_SIZE}")
    print(f"Vector dimension: {VECTOR_DIM}")
    print(f"Number of queries (held-out): {N_QUERIES}")
    print(f"k for evaluation: {K}")
    print()

    # Phase 1: Generate dataset
    full_dataset = generate_synthetic_dataset(DATASET_SIZE, VECTOR_DIM)
    train_dataset, query_set = split_query_set_from_dataset(full_dataset, N_QUERIES)

    # Phase 2: Build and evaluate hybrid index
    print("\n=== Building Hybrid HNSW Index ===")
    hybrid_index = HybridHNSWIndex(k_children=1000, n_probe=10)

    # Build base index ONLY on training portion
    hybrid_index.build_base_index(train_dataset)

    # Extract parent nodes
    hybrid_index.extract_parent_nodes(target_level=2)

    # Build parent-child mapping
    hybrid_index.build_parent_child_mapping()

    # Phase 3: Evaluate recall
    print("\n=== Evaluating Recall Performance ===")
    evaluator = RecallEvaluator(train_dataset)  # ground truth over train set
    results = evaluator.evaluate_recall(hybrid_index, query_set, k=K)

    # Print final results
    print("\n=== Final Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")
