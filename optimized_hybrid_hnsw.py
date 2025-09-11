#!/usr/bin/env python3
"""
Optimized Hybrid HNSW with performance improvements
"""

import sys
import os
sys.path.append('.')

import time
import numpy as np
from typing import Dict, List, Tuple
from datasketch import HNSW
from hnsw_hybrid_evaluation import generate_synthetic_dataset, create_query_set

class OptimizedHybridHNSW:
    """
    Performance-optimized version of Hybrid HNSW
    """
    
    def __init__(self, k_children=1000, n_probe=10, distance_func=None):
        self.k_children = k_children
        self.n_probe = n_probe
        self.distance_func = distance_func or self._l2_distance
        
        # Core components
        self.dataset = None
        self.base_index = None
        self.parent_ids = []
        self.parent_vectors = {}
        self.parent_child_map = {}
        
        # Performance tracking
        self.build_time = 0
        self.search_times = []
        
    def _l2_distance(self, x, y):
        """Fast L2 distance function."""
        return np.linalg.norm(x - y)
    
    def build_base_index_optimized(self, dataset: Dict[int, np.ndarray], 
                                 m: int = 16, ef_construction: int = 100):
        """
        Build the base HNSW index with optimized parameters.
        
        Args:
            dataset: Dictionary mapping node IDs to vectors
            m: HNSW parameter (reduced for faster construction)
            ef_construction: HNSW parameter (reduced for faster construction)
        """
        print(f"Building optimized HNSW index with {len(dataset)} vectors...")
        start_time = time.time()
        
        self.dataset = dataset
        
        # Use more conservative parameters for faster construction
        self.base_index = HNSW(
            distance_func=self.distance_func,
            m=m,  # Reduced from 16 to speed up
            ef_construction=ef_construction  # Reduced from 200 to speed up
        )
        
        # Progress tracking for large datasets
        dataset_size = len(dataset)
        if dataset_size > 1000:
            print(f"Large dataset detected ({dataset_size} vectors)")
            print("Using batch processing with progress tracking...")
            
            # Process in chunks for better memory management
            chunk_size = min(1000, dataset_size // 10)
            processed = 0
            
            items = list(dataset.items())
            for i in range(0, len(items), chunk_size):
                chunk = dict(items[i:i + chunk_size])
                self.base_index.update(chunk)
                processed += len(chunk)
                
                if processed % 1000 == 0 or processed == dataset_size:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (dataset_size - processed) / rate if rate > 0 else 0
                    print(f"  Progress: {processed}/{dataset_size} vectors "
                          f"({processed/dataset_size*100:.1f}%) - "
                          f"{rate:.1f} vectors/sec - ETA: {eta:.1f}s")
        else:
            # Small dataset - use direct update
            self.base_index.update(dataset)
        
        build_time = time.time() - start_time
        self.build_time = build_time
        print(f"✅ Optimized base index built in {build_time:.2f} seconds")
        print(f"   Rate: {len(dataset)/build_time:.1f} vectors/sec")
        
        return self.base_index
    
    def extract_parent_nodes(self, target_level=2):
        """Extract parent nodes from specified HNSW level."""
        if not self.base_index:
            raise ValueError("Base index must be built first")
        
        print(f"Extracting parent nodes from level {target_level}...")
        
        # Ensure target level exists
        if target_level >= len(self.base_index._graphs):
            target_level = len(self.base_index._graphs) - 1
            print(f"Adjusted to level {target_level} (highest available)")
        
        # Extract parent IDs
        layer = self.base_index._graphs[target_level]
        self.parent_ids = list(layer._graph.keys())
        
        # Cache parent vectors for fast access
        self.parent_vectors = {
            parent_id: self.dataset[parent_id] 
            for parent_id in self.parent_ids
        }
        
        print(f"✅ Extracted {len(self.parent_ids)} parent nodes from level {target_level}")
        return self.parent_ids
    
    def build_parent_child_mapping_fast(self):
        """
        Build parent-child mapping with performance optimizations.
        """
        print(f"Building optimized parent-child mapping (k_children={self.k_children})...")
        start_time = time.time()
        
        if not self.parent_ids:
            raise ValueError("Parent nodes must be extracted first")
        
        self.parent_child_map = {}
        total_parents = len(self.parent_ids)
        
        for i, parent_id in enumerate(self.parent_ids):
            if i % max(1, total_parents // 10) == 0 or i == total_parents - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_parents - i - 1) / rate if rate > 0 else 0
                print(f"  Progress: {i+1}/{total_parents} parents "
                      f"({(i+1)/total_parents*100:.1f}%) - "
                      f"{rate:.1f} parents/sec - ETA: {eta:.1f}s")
            
            # Get parent vector
            parent_vector = self.dataset[parent_id]
            
            # Use HNSW query with reduced ef for faster search
            neighbors = self.base_index.query(parent_vector, 
                                            k=self.k_children + 1, 
                                            ef=50)  # Reduced ef for speed
            
            # Store children (excluding parent itself)
            child_ids = [nid for nid, _ in neighbors if nid != parent_id][:self.k_children]
            self.parent_child_map[parent_id] = child_ids
        
        mapping_time = time.time() - start_time
        print(f"✅ Parent-child mapping built in {mapping_time:.2f} seconds")
        
        # Calculate coverage statistics
        total_coverage = set()
        for children in self.parent_child_map.values():
            total_coverage.update(children)
        
        coverage_pct = len(total_coverage) / len(self.dataset) * 100
        avg_children = np.mean([len(children) for children in self.parent_child_map.values()])
        
        print(f"   Coverage: {coverage_pct:.1f}% ({len(total_coverage)}/{len(self.dataset)} vectors)")
        print(f"   Avg children per parent: {avg_children:.1f}")
        
        return self.parent_child_map
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform optimized two-stage search.
        """
        start_time = time.time()
        
        # Stage 1: Find closest parents (vectorized for speed)
        parent_distances = []
        for parent_id in self.parent_ids:
            parent_vector = self.parent_vectors[parent_id]
            distance = self.distance_func(query_vector, parent_vector)
            parent_distances.append((parent_id, distance))
        
        # Sort and select top n_probe parents
        parent_distances.sort(key=lambda x: x[1])
        selected_parents = [pid for pid, _ in parent_distances[:self.n_probe]]
        
        # Stage 2: Collect candidates efficiently
        candidate_ids = set()
        for parent_id in selected_parents:
            if parent_id in self.parent_child_map:
                candidate_ids.update(self.parent_child_map[parent_id])
        
        # Add parent nodes as candidates
        candidate_ids.update(selected_parents)
        
        # Vectorized distance computation for candidates
        candidates = list(candidate_ids)
        if not candidates:
            return []
        
        # Compute all distances at once for better performance
        candidate_vectors = np.array([self.dataset[cid] for cid in candidates])
        distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)
        
        # Create result pairs and sort
        candidate_results = list(zip(candidates, distances))
        candidate_results.sort(key=lambda x: x[1])
        
        search_time = time.time() - start_time
        self.search_times.append(search_time)
        
        return candidate_results[:k]

def test_optimized_performance():
    """Test the optimized implementation performance"""
    print("🚀 TESTING OPTIMIZED HYBRID HNSW PERFORMANCE")
    print("=" * 55)
    
    # Test with different sizes
    test_sizes = [1000, 5000, 10000]
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size} vectors:")
        
        # Generate test data
        dataset = generate_synthetic_dataset(size, 64)
        query_set = create_query_set(dataset, 100)
        
        # Test optimized hybrid
        print("\n🔧 Optimized Hybrid HNSW:")
        hybrid = OptimizedHybridHNSW(k_children=500, n_probe=10)
        
        # Build index
        build_start = time.time()
        hybrid.build_base_index_optimized(dataset, m=8, ef_construction=50)
        hybrid.extract_parent_nodes(target_level=2)
        hybrid.build_parent_child_mapping_fast()
        total_build_time = time.time() - build_start
        
        print(f"✅ Total build time: {total_build_time:.2f}s ({size/total_build_time:.1f} vectors/sec)")
        
        # Test search performance
        search_times = []
        for i, (qid, query_vector) in enumerate(list(query_set.items())[:10]):
            start = time.time()
            results = hybrid.search(query_vector, k=10)
            search_time = time.time() - start
            search_times.append(search_time)
            
            if i == 0:
                print(f"   Sample results: {[rid for rid, _ in results[:5]]}")
        
        avg_search_time = np.mean(search_times) * 1000  # Convert to ms
        print(f"✅ Average search time: {avg_search_time:.2f}ms")
        print(f"   Search throughput: {1000/avg_search_time:.1f} queries/sec")

if __name__ == "__main__":
    test_optimized_performance()
