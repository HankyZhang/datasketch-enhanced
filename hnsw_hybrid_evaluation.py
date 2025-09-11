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
        self.distance_func = distance_func or self._l2_distance
        self.k_children = k_children
        self.n_probe = n_probe
        self.parent_child_method = parent_child_method

        # Core data
        self.base_index = None
        self.parent_ids: List[int] = []
        self.parent_child_map: Dict[int, List[int]] = {}
        self.parent_vectors: Dict[int, np.ndarray] = {}
        self._parent_matrix = None
        self._parent_id_index: Dict[int, int] = {}
        self.dataset: Dict[int, np.ndarray] = {}

        # Metrics
        self.base_build_time = 0.0
        self.parent_extraction_time = 0.0
        self.mapping_build_time = 0.0
        self.search_times: List[float] = []
        self.candidate_sizes: List[int] = []

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

    def build_parent_child_mapping(self, method: str = None, ef: int = 50, brute_force_batch: int = 4096):
        method = method or self.parent_child_method
        if method not in {"approx", "brute"}:
            raise ValueError("method must be 'approx' or 'brute'")
        if not self.parent_ids:
            raise ValueError("Parent nodes must be extracted first")
        print(f"Building parent-child mapping method={method} k_children={self.k_children} parents={len(self.parent_ids)}")
        start = time.time()
        self.parent_child_map = {}
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
            if method == 'approx':
                neighbors = self.base_index.query(pvec, k=self.k_children + 1, ef=ef)
                child_ids = [nid for nid, _ in neighbors if nid != pid][:self.k_children]
            else:  # brute
                dists = []
                for start_b in range(0, len(data_ids), brute_force_batch):
                    batch_ids = data_ids[start_b:start_b + brute_force_batch]
                    batch_vecs = data_matrix[start_b:start_b + brute_force_batch]
                    batch_d = np.linalg.norm(batch_vecs - pvec, axis=1)
                    dists.extend(zip(batch_ids, batch_d))
                dists.sort(key=lambda x: x[1])
                child_ids = [cid for cid, _ in dists if cid != pid][:self.k_children]
            self.parent_child_map[pid] = child_ids
        self.mapping_build_time = time.time() - start
        print(f"Parent-child mapping built in {self.mapping_build_time:.2f}s")
        return self.parent_child_map

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
        return {
            'base_build_time': self.base_build_time,
            'parent_extraction_time': self.parent_extraction_time,
            'mapping_build_time': self.mapping_build_time,
            'avg_search_time': float(np.mean(self.search_times)) if self.search_times else 0.0,
            'avg_candidate_size': float(np.mean(self.candidate_sizes)) if self.candidate_sizes else 0.0,
            **cov
        }


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
                       k: int = 10) -> Dict[str, float]:
        """
        Evaluate recall performance of the hybrid index.
        
        Args:
            hybrid_index: The hybrid HNSW index to evaluate
            query_vectors: Dictionary of query vectors
            k: Number of nearest neighbors to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"Evaluating recall@{k} for {len(query_vectors)} queries...")
        
        # Get ground truth
        if not self.ground_truth_cache:
            ground_truth = self.compute_ground_truth(query_vectors, k)
        else:
            ground_truth = self.ground_truth_cache
        
        total_correct = 0
        total_possible = 0
        query_times = []
        
        for i, (query_id, query_vector) in enumerate(query_vectors.items()):
            if i % 1000 == 0:
                print(f"Evaluating query {i+1}/{len(query_vectors)}")
            
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
        
        # Calculate metrics
        recall = total_correct / total_possible if total_possible > 0 else 0.0
        avg_query_time = np.mean(query_times)
        
        results = {
            'recall@k': recall,
            'k': k,
            'total_queries': len(query_vectors),
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
