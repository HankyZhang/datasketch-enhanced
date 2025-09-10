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
    """
    Hybrid HNSW implementation with two-stage parent-child retrieval system.
    """
    
    def __init__(self, distance_func=None, k_children: int = 1000, n_probe: int = 10):
        """
        Initialize the hybrid HNSW index.
        
        Args:
            distance_func: Distance function for similarity computation
            k_children: Number of child nodes per parent (default: 1000)
            n_probe: Number of parent nodes to probe during search (default: 10)
        """
        self.distance_func = distance_func or self._l2_distance
        self.k_children = k_children
        self.n_probe = n_probe
        
        # Core data structures
        self.base_index = None
        self.parent_ids = []
        self.parent_child_map = {}
        self.parent_vectors = {}
        self.dataset = {}
        
        # Performance tracking
        self.build_time = 0
        self.search_times = []
        
    def _l2_distance(self, x, y):
        """Default L2 distance function."""
        return np.linalg.norm(x - y)
    
    def build_base_index(self, dataset: Dict[int, np.ndarray], m: int = 16, ef_construction: int = 200):
        """
        Build the base HNSW index from the dataset.
        
        Args:
            dataset: Dictionary mapping node IDs to vectors
            m: HNSW parameter for maximum connections per node
            ef_construction: HNSW parameter for construction search width
        """
        print(f"Building base HNSW index with {len(dataset)} vectors...")
        start_time = time.time()
        
        self.dataset = dataset
        self.base_index = HNSW(
            distance_func=self.distance_func,
            m=m,
            ef_construction=ef_construction
        )
        
        # Insert all vectors into base index
        self.base_index.update(dataset)
        
        build_time = time.time() - start_time
        print(f"Base index built in {build_time:.2f} seconds")
        
        return self.base_index
    
    def extract_parent_nodes(self, target_level: int = 2):
        """
        Extract parent nodes from a specific level of the HNSW graph.
        
        Args:
            target_level: Level to extract nodes from (default: 2)
        """
        print(f"Extracting parent nodes from level {target_level}...")
        
        if self.base_index is None:
            raise ValueError("Base index must be built first")
        
        self.parent_ids = []
        
        # Check if target level exists
        if target_level >= len(self.base_index._graphs):
            print(f"Warning: Level {target_level} does not exist. Available levels: 0-{len(self.base_index._graphs)-1}")
            # Use the highest available level
            target_level = len(self.base_index._graphs) - 1
            print(f"Using level {target_level} instead")
        
        if target_level < 0:
            target_level = 0
            print(f"Using level {target_level} (base level)")
        
        # Get nodes from the target level
        if target_level < len(self.base_index._graphs):
            layer = self.base_index._graphs[target_level]
            self.parent_ids = list(layer._graph.keys())
        
        # Store parent vectors for quick access
        self.parent_vectors = {
            parent_id: self.dataset[parent_id] 
            for parent_id in self.parent_ids
        }
        
        print(f"Extracted {len(self.parent_ids)} parent nodes from level {target_level}")
        return self.parent_ids
    
    def build_parent_child_mapping(self):
        """
        Build the parent-child mapping by finding k_children nearest neighbors
        for each parent node.
        """
        print(f"Building parent-child mapping with k_children={self.k_children}...")
        start_time = time.time()
        
        if not self.parent_ids:
            raise ValueError("Parent nodes must be extracted first")
        
        self.parent_child_map = {}
        
        for i, parent_id in enumerate(self.parent_ids):
            if i % 100 == 0:
                print(f"Processing parent {i+1}/{len(self.parent_ids)}")
            
            # Get parent vector
            parent_vector = self.dataset[parent_id]
            
            # Search for k_children nearest neighbors
            neighbors = self.base_index.query(parent_vector, k=self.k_children + 1)
            
            # Remove the parent itself from neighbors and store children
            child_ids = [nid for nid, _ in neighbors if nid != parent_id][:self.k_children]
            self.parent_child_map[parent_id] = child_ids
        
        mapping_time = time.time() - start_time
        print(f"Parent-child mapping built in {mapping_time:.2f} seconds")
        self.build_time = mapping_time
        
        return self.parent_child_map
    
    def _find_closest_parents(self, query_vector: np.ndarray) -> List[int]:
        """
        Find the closest parent nodes to the query vector.
        
        Args:
            query_vector: Query vector to search for
            
        Returns:
            List of closest parent node IDs
        """
        parent_distances = []
        
        for parent_id in self.parent_ids:
            parent_vector = self.parent_vectors[parent_id]
            distance = self.distance_func(query_vector, parent_vector)
            parent_distances.append((parent_id, distance))
        
        # Sort by distance and return top n_probe parents
        parent_distances.sort(key=lambda x: x[1])
        return [pid for pid, _ in parent_distances[:self.n_probe]]
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform two-stage search using the hybrid index.
        
        Args:
            query_vector: Query vector to search for
            k: Number of nearest neighbors to return
            
        Returns:
            List of (node_id, distance) tuples
        """
        start_time = time.time()
        
        # Stage 1: Find closest parent nodes
        closest_parents = self._find_closest_parents(query_vector)
        
        # Stage 2: Collect all child candidates
        candidate_ids = set()
        for parent_id in closest_parents:
            if parent_id in self.parent_child_map:
                candidate_ids.update(self.parent_child_map[parent_id])
        
        # Add parent nodes themselves as candidates
        candidate_ids.update(closest_parents)
        
        # Compute distances for all candidates
        candidate_results = []
        for candidate_id in candidate_ids:
            candidate_vector = self.dataset[candidate_id]
            distance = self.distance_func(query_vector, candidate_vector)
            candidate_results.append((candidate_id, distance))
        
        # Sort by distance and return top k
        candidate_results.sort(key=lambda x: x[1])
        search_time = time.time() - start_time
        self.search_times.append(search_time)
        
        return candidate_results[:k]


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
    """
    Create a query set by sampling from the dataset.
    
    Args:
        dataset: Complete dataset
        n_queries: Number of queries to sample
        
    Returns:
        Dictionary of query vectors
    """
    print(f"Creating query set with {n_queries} queries...")
    
    # Randomly sample query IDs
    all_ids = list(dataset.keys())
    np.random.seed(123)  # Different seed for queries
    query_ids = np.random.choice(all_ids, size=n_queries, replace=False)
    
    query_set = {qid: dataset[qid] for qid in query_ids}
    
    print("Query set creation completed")
    return query_set


if __name__ == "__main__":
    # Configuration
    DATASET_SIZE = 60000  # Start with smaller dataset for testing
    VECTOR_DIM = 128
    N_QUERIES = 1000
    K = 10
    
    print("=== HNSW Hybrid Two-Stage Retrieval System Evaluation ===")
    print(f"Dataset size: {DATASET_SIZE}")
    print(f"Vector dimension: {VECTOR_DIM}")
    print(f"Number of queries: {N_QUERIES}")
    print(f"k for evaluation: {K}")
    print()
    
    # Phase 1: Generate dataset
    dataset = generate_synthetic_dataset(DATASET_SIZE, VECTOR_DIM)
    query_set = create_query_set(dataset, N_QUERIES)
    
    # Phase 2: Build and evaluate hybrid index
    print("\n=== Building Hybrid HNSW Index ===")
    hybrid_index = HybridHNSWIndex(k_children=1000, n_probe=10)
    
    # Build base index
    hybrid_index.build_base_index(dataset)
    
    # Extract parent nodes
    hybrid_index.extract_parent_nodes(target_level=2)
    
    # Build parent-child mapping
    hybrid_index.build_parent_child_mapping()
    
    # Phase 3: Evaluate recall
    print("\n=== Evaluating Recall Performance ===")
    evaluator = RecallEvaluator(dataset)
    results = evaluator.evaluate_recall(hybrid_index, query_set, k=K)
    
    # Print final results
    print("\n=== Final Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")
