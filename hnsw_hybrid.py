"""
HNSW Hybrid Two-Stage Retrieval System

This module implements a hybrid HNSW index structure that transforms a standard HNSW
into a two-stage retrieval system for improved recall evaluation in plaintext environments.

The system consists of:
1. Parent Layer (Coarse Filtering): Nodes from higher HNSW levels as cluster centers
2. Child Layer (Fine Filtering): Precomputed neighbor sets for each parent node

Author: AI Assistant
Date: 2024
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Set, Optional, Hashable
from collections import defaultdict
import heapq
from datasketch.hnsw import HNSW


class HNSWHybrid:
    """
    Hybrid HNSW index that implements a two-stage retrieval system.
    
    This class extracts parent nodes from higher levels of a base HNSW index
    and precomputes child node mappings for efficient two-stage search.
    """
    
    def __init__(
        self,
        base_index: HNSW,
        parent_level: int = 2,
        k_children: int = 1000,
        distance_func: Optional[callable] = None
    ):
        """
        Initialize the hybrid HNSW index.
        
        Args:
            base_index: The base HNSW index to extract structure from
            parent_level: The HNSW level to extract parent nodes from (default: 2)
            k_children: Number of child nodes to precompute for each parent
            distance_func: Distance function (uses base index's if None)
        """
        self.base_index = base_index
        self.parent_level = parent_level
        self.k_children = k_children
        self.distance_func = distance_func or base_index._distance_func
        
        # Parent-child mapping: {parent_id: [child_id_1, child_id_2, ...]}
        self.parent_child_map: Dict[Hashable, List[Hashable]] = {}
        
        # Parent node vectors for Stage 1 search
        self.parent_vectors: Dict[Hashable, np.ndarray] = {}
        
        # All child node vectors for Stage 2 search
        self.child_vectors: Dict[Hashable, np.ndarray] = {}
        
        # Statistics
        self.stats = {
            'num_parents': 0,
            'num_children': 0,
            'avg_children_per_parent': 0.0,
            'construction_time': 0.0
        }
        
        # Build the hybrid structure
        self._build_hybrid_structure()
    
    def _build_hybrid_structure(self):
        """Build the parent-child structure from the base HNSW index."""
        print(f"Building hybrid HNSW structure from level {self.parent_level}...")
        start_time = time.time()
        
        # Step 1: Extract parent nodes from the specified level
        parent_nodes = self._extract_parent_nodes()
        print(f"Found {len(parent_nodes)} parent nodes at level {self.parent_level}")
        
        # Step 2: Precompute child mappings for each parent
        self._precompute_child_mappings(parent_nodes)
        
        # Step 3: Build parent vector index for Stage 1 search
        self._build_parent_index()
        
        # Record statistics
        self.stats['num_parents'] = len(parent_nodes)
        self.stats['num_children'] = len(self.child_vectors)
        self.stats['avg_children_per_parent'] = (
            self.stats['num_children'] / self.stats['num_parents'] 
            if self.stats['num_parents'] > 0 else 0.0
        )
        self.stats['construction_time'] = time.time() - start_time
        
        print(f"Hybrid structure built in {self.stats['construction_time']:.2f}s")
        print(f"Statistics: {self.stats['num_parents']} parents, "
              f"{self.stats['num_children']} children, "
              f"{self.stats['avg_children_per_parent']:.1f} avg children/parent")
    
    def _extract_parent_nodes(self) -> List[Hashable]:
        """Extract parent nodes from the specified HNSW level."""
        parent_nodes = []
        
        # Check if the parent level exists in the base index
        if self.parent_level >= len(self.base_index._graphs):
            raise ValueError(f"Parent level {self.parent_level} does not exist in base index. "
                           f"Available levels: 0-{len(self.base_index._graphs)-1}")
        
        # Extract all nodes from the specified level
        target_layer = self.base_index._graphs[self.parent_level]
        for node_id in target_layer:
            # Only include non-deleted nodes
            if node_id in self.base_index and not self.base_index._nodes[node_id].is_deleted:
                parent_nodes.append(node_id)
        
        return parent_nodes
    
    def _precompute_child_mappings(self, parent_nodes: List[Hashable]):
        """Precompute child node mappings for each parent."""
        print("Precomputing child mappings...")
        
        for i, parent_id in enumerate(parent_nodes):
            if i % 100 == 0:
                print(f"Processing parent {i+1}/{len(parent_nodes)}")
            
            # Get parent vector
            parent_vector = self.base_index[parent_id]
            self.parent_vectors[parent_id] = parent_vector
            
            # Search for k_children nearest neighbors in the base index
            # Use the parent vector as a query
            neighbors = self.base_index.query(parent_vector, k=self.k_children)
            
            # Extract child node IDs (exclude the parent itself)
            child_ids = []
            for neighbor_id, distance in neighbors:
                if neighbor_id != parent_id:  # Exclude self
                    child_ids.append(neighbor_id)
                    # Store child vector if not already stored
                    if neighbor_id not in self.child_vectors:
                        self.child_vectors[neighbor_id] = self.base_index[neighbor_id]
            
            # Store the parent-child mapping
            self.parent_child_map[parent_id] = child_ids
    
    def _build_parent_index(self):
        """Build a simple index for parent vectors (Stage 1 search)."""
        # This is just storing the vectors - we'll use brute force search
        # In a more sophisticated implementation, we could use FAISS or similar
        pass
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        n_probe: int = 10
    ) -> List[Tuple[Hashable, float]]:
        """
        Perform two-stage search on the hybrid index.
        
        Args:
            query_vector: The query vector
            k: Number of results to return
            n_probe: Number of parent nodes to probe in Stage 1
            
        Returns:
            List of (node_id, distance) tuples, sorted by distance
        """
        # Stage 1: Coarse search - find closest parent nodes
        parent_candidates = self._stage1_coarse_search(query_vector, n_probe)
        
        # Stage 2: Fine search - search within child nodes of selected parents
        results = self._stage2_fine_search(query_vector, parent_candidates, k)
        
        return results
    
    def _stage1_coarse_search(
        self, 
        query_vector: np.ndarray, 
        n_probe: int
    ) -> List[Tuple[Hashable, float]]:
        """Stage 1: Find the closest parent nodes using brute force search."""
        parent_distances = []
        
        # Calculate distances to all parent nodes
        for parent_id, parent_vector in self.parent_vectors.items():
            distance = self.distance_func(query_vector, parent_vector)
            parent_distances.append((distance, parent_id))
        
        # Sort by distance and return top n_probe
        parent_distances.sort()
        return parent_distances[:n_probe]
    
    def _stage2_fine_search(
        self, 
        query_vector: np.ndarray, 
        parent_candidates: List[Tuple[Hashable, float]], 
        k: int
    ) -> List[Tuple[Hashable, float]]:
        """Stage 2: Search within child nodes of selected parents."""
        # Collect all child nodes from selected parents
        candidate_children = set()
        for distance, parent_id in parent_candidates:
            if parent_id in self.parent_child_map:
                candidate_children.update(self.parent_child_map[parent_id])
        
        # Calculate distances to all candidate children
        child_distances = []
        for child_id in candidate_children:
            if child_id in self.child_vectors:
                distance = self.distance_func(query_vector, self.child_vectors[child_id])
                child_distances.append((distance, child_id))
        
        # Sort by distance and return top k
        child_distances.sort()
        return child_distances[:k]
    
    def get_stats(self) -> Dict:
        """Get construction statistics."""
        return self.stats.copy()
    
    def get_parent_child_info(self) -> Dict:
        """Get detailed parent-child mapping information."""
        return {
            'parent_child_map': self.parent_child_map,
            'num_parents': len(self.parent_vectors),
            'num_children': len(self.child_vectors),
            'parent_level': self.parent_level,
            'k_children': self.k_children
        }


class HNSWEvaluator:
    """
    Evaluator for HNSW hybrid system recall performance.
    """
    
    def __init__(self, dataset: np.ndarray, query_set: np.ndarray, query_ids: List[Hashable]):
        """
        Initialize the evaluator.
        
        Args:
            dataset: Full dataset vectors (shape: [n_vectors, dim])
            query_set: Query vectors (shape: [n_queries, dim])
            query_ids: IDs for query vectors
        """
        self.dataset = dataset
        self.query_set = query_set
        self.query_ids = query_ids
        
        # Ground truth will be computed lazily
        self._ground_truth: Optional[Dict[Hashable, List[Tuple[Hashable, float]]]] = None
    
    def compute_ground_truth(self, k: int, distance_func: callable) -> Dict[Hashable, List[Tuple[Hashable, float]]]:
        """
        Compute ground truth using brute force search.
        
        Args:
            k: Number of nearest neighbors to find
            distance_func: Distance function to use
            
        Returns:
            Dictionary mapping query_id to list of (neighbor_id, distance) tuples
        """
        print(f"Computing ground truth for {len(self.query_set)} queries...")
        start_time = time.time()
        
        ground_truth = {}
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            if i % 1000 == 0:
                print(f"Processing query {i+1}/{len(self.query_set)}")
            
            # Calculate distances to all dataset vectors
            distances = []
            for j, dataset_vector in enumerate(self.dataset):
                distance = distance_func(query_vector, dataset_vector)
                distances.append((distance, j))
            
            # Sort by distance and take top k
            distances.sort()
            ground_truth[query_id] = distances[:k]
        
        self._ground_truth = ground_truth
        print(f"Ground truth computed in {time.time() - start_time:.2f}s")
        return ground_truth
    
    def evaluate_recall(
        self, 
        hybrid_index: HNSWHybrid, 
        k: int, 
        n_probe: int,
        ground_truth: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate recall performance of the hybrid index.
        
        Args:
            hybrid_index: The hybrid HNSW index to evaluate
            k: Number of results to return
            n_probe: Number of parent nodes to probe
            ground_truth: Precomputed ground truth (optional)
            
        Returns:
            Dictionary containing recall metrics and timing information
        """
        if ground_truth is None:
            if self._ground_truth is None:
                ground_truth = self.compute_ground_truth(k, hybrid_index.distance_func)
            else:
                ground_truth = self._ground_truth
        
        print(f"Evaluating recall for {len(self.query_set)} queries...")
        start_time = time.time()
        
        total_correct = 0
        total_expected = len(self.query_set) * k
        query_times = []
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            if i % 1000 == 0:
                print(f"Evaluating query {i+1}/{len(self.query_set)}")
            
            # Time the query
            query_start = time.time()
            results = hybrid_index.search(query_vector, k=k, n_probe=n_probe)
            query_time = time.time() - query_start
            query_times.append(query_time)
            
            # Extract result IDs
            result_ids = set(neighbor_id for neighbor_id, _ in results)
            
            # Get ground truth IDs
            gt_ids = set(neighbor_id for neighbor_id, _ in ground_truth[query_id])
            
            # Count correct matches
            correct = len(result_ids.intersection(gt_ids))
            total_correct += correct
        
        # Calculate metrics
        recall_at_k = total_correct / total_expected
        avg_query_time = np.mean(query_times)
        total_evaluation_time = time.time() - start_time
        
        return {
            'recall_at_k': recall_at_k,
            'total_correct': total_correct,
            'total_expected': total_expected,
            'avg_query_time_ms': avg_query_time * 1000,
            'total_evaluation_time': total_evaluation_time,
            'query_times': query_times,
            'k': k,
            'n_probe': n_probe,
            'hybrid_stats': hybrid_index.get_stats()
        }
    
    def parameter_sweep(
        self, 
        hybrid_index: HNSWHybrid, 
        k_values: List[int], 
        n_probe_values: List[int],
        ground_truth: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform parameter sweep to find optimal settings.
        
        Args:
            hybrid_index: The hybrid HNSW index to evaluate
            k_values: List of k values to test
            n_probe_values: List of n_probe values to test
            ground_truth: Precomputed ground truth (optional)
            
        Returns:
            List of evaluation results for each parameter combination
        """
        results = []
        
        for k in k_values:
            for n_probe in n_probe_values:
                print(f"\nEvaluating k={k}, n_probe={n_probe}")
                result = self.evaluate_recall(hybrid_index, k, n_probe, ground_truth)
                results.append(result)
        
        return results


def create_synthetic_dataset(n_vectors: int, dim: int, seed: int = 42) -> np.ndarray:
    """Create a synthetic dataset for testing."""
    np.random.seed(seed)
    return np.random.randn(n_vectors, dim).astype(np.float32)


def create_query_set(dataset: np.ndarray, n_queries: int, seed: int = 123) -> Tuple[np.ndarray, List[int]]:
    """Create a query set by sampling from the dataset."""
    np.random.seed(seed)
    n_dataset = len(dataset)
    query_indices = np.random.choice(n_dataset, size=n_queries, replace=False)
    query_vectors = dataset[query_indices]
    return query_vectors, query_indices.tolist()


if __name__ == "__main__":
    # Example usage
    print("HNSW Hybrid System - Example Usage")
    
    # Create synthetic data
    print("Creating synthetic dataset...")
    dataset = create_synthetic_dataset(10000, 128)  # 10K vectors for demo
    query_vectors, query_ids = create_query_set(dataset, 100)  # 100 queries for demo
    
    # Build base HNSW index
    print("Building base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    # Insert all vectors except queries
    for i, vector in enumerate(dataset):
        if i not in query_ids:  # Exclude query vectors from index
            base_index.insert(i, vector)
    
    # Build hybrid index
    print("Building hybrid HNSW index...")
    hybrid_index = HNSWHybrid(
        base_index=base_index,
        parent_level=2,
        k_children=500
    )
    
    # Evaluate
    print("Evaluating recall...")
    evaluator = HNSWEvaluator(dataset, query_vectors, query_ids)
    
    # Compute ground truth
    ground_truth = evaluator.compute_ground_truth(k=10, distance_func=distance_func)
    
    # Evaluate recall
    result = evaluator.evaluate_recall(hybrid_index, k=10, n_probe=5, ground_truth=ground_truth)
    
    print(f"\nResults:")
    print(f"Recall@10: {result['recall_at_k']:.4f}")
    print(f"Average query time: {result['avg_query_time_ms']:.2f} ms")
    print(f"Hybrid stats: {result['hybrid_stats']}")
