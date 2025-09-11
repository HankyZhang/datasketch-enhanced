#!/usr/bin/env python3
"""
Demo showing the fundamental issue with Hybrid HNSW and a simple fix
"""

import sys
import os
sys.path.append('.')

import numpy as np
from datasketch import HNSW
from hnsw_hybrid_evaluation import generate_synthetic_dataset

class FixedHybridHNSW:
    """
    Fixed version that uses consistent distance calculation
    """
    def __init__(self, k_children=200, n_probe=5):
        self.k_children = k_children
        self.n_probe = n_probe
        self.dataset = None
        self.base_index = None
        self.parent_ids = []
        self.parent_child_map = {}
        
    def build_base_index(self, dataset):
        self.dataset = dataset
        self.base_index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
        
        print(f"Building base HNSW index with {len(dataset)} vectors...")
        for i, vector in enumerate(dataset):
            self.base_index.insert(i, vector)
    
    def extract_parent_nodes(self, target_level=2):
        """Extract parent nodes from HNSW level"""
        if target_level >= len(self.base_index._graphs):
            target_level = len(self.base_index._graphs) - 1
            
        layer = self.base_index._graphs[target_level]
        self.parent_ids = list(layer._graph.keys())
        print(f"Extracted {len(self.parent_ids)} parent nodes from level {target_level}")
        return self.parent_ids
    
    def build_parent_child_mapping_original(self):
        """Original method using HNSW query (problematic)"""
        print("Building parent-child mapping (ORIGINAL method)...")
        self.parent_child_map = {}
        
        for parent_id in self.parent_ids:
            parent_vector = self.dataset[parent_id]
            # Use HNSW query - this is the problem!
            neighbors = self.base_index.query(parent_vector, k=self.k_children + 1)
            child_ids = [nid for nid, _ in neighbors if nid != parent_id][:self.k_children]
            self.parent_child_map[parent_id] = child_ids
    
    def build_parent_child_mapping_fixed(self):
        """Fixed method using consistent distance calculation"""
        print("Building parent-child mapping (FIXED method)...")
        self.parent_child_map = {}
        
        for parent_id in self.parent_ids:
            parent_vector = self.dataset[parent_id]
            
            # Calculate distances to ALL vectors directly
            distances = []
            for i, vector in enumerate(self.dataset):
                if i != parent_id:  # Exclude parent itself
                    dist = np.linalg.norm(parent_vector - vector)
                    distances.append((i, dist))
            
            # Sort by distance and take k_children closest
            distances.sort(key=lambda x: x[1])
            child_ids = [idx for idx, _ in distances[:self.k_children]]
            self.parent_child_map[parent_id] = child_ids
    
    def search(self, query_vector, k=10):
        """Search using the hybrid method"""
        # Find closest parents
        parent_distances = []
        for parent_id in self.parent_ids:
            parent_vector = self.dataset[parent_id]
            dist = np.linalg.norm(query_vector - parent_vector)
            parent_distances.append((parent_id, dist))
        
        parent_distances.sort(key=lambda x: x[1])
        selected_parents = [pid for pid, _ in parent_distances[:self.n_probe]]
        
        # Collect all candidates
        candidates = set()
        for parent_id in selected_parents:
            candidates.update(self.parent_child_map[parent_id])
        
        # Search among candidates
        candidate_distances = []
        for candidate_id in candidates:
            candidate_vector = self.dataset[candidate_id]
            dist = np.linalg.norm(query_vector - candidate_vector)
            candidate_distances.append((candidate_id, dist))
        
        candidate_distances.sort(key=lambda x: x[1])
        return candidate_distances[:k]

def test_hybrid_fix():
    print("🔍 DEMONSTRATING HYBRID HNSW ISSUE AND FIX")
    print("=" * 55)
    
    # Generate test data
    dataset = generate_synthetic_dataset(1000, 32)
    
    # Use a vector from dataset as query (should find itself)
    query_id = 131
    query_vector = dataset[query_id]
    
    print(f"Query vector ID: {query_id}")
    print("Expected: Query should find itself as nearest neighbor")
    
    # Ground truth
    distances = [(i, np.linalg.norm(query_vector - vec)) for i, vec in enumerate(dataset)]
    distances.sort(key=lambda x: x[1])
    ground_truth = [idx for idx, _ in distances[:10]]
    print(f"Ground truth top-10: {ground_truth}")
    
    # Test original hybrid
    print("\n" + "="*50)
    print("🔴 ORIGINAL HYBRID (Problematic)")
    print("="*50)
    
    hybrid_original = FixedHybridHNSW(k_children=200, n_probe=5)
    hybrid_original.build_base_index(dataset)
    hybrid_original.extract_parent_nodes(target_level=2)
    hybrid_original.build_parent_child_mapping_original()
    
    # Check coverage
    all_candidates_orig = set()
    for children in hybrid_original.parent_child_map.values():
        all_candidates_orig.update(children)
    
    coverage_orig = len(all_candidates_orig) / len(dataset) * 100
    query_in_candidates_orig = query_id in all_candidates_orig
    
    print(f"Coverage: {coverage_orig:.1f}%")
    print(f"Query {query_id} in candidates: {query_in_candidates_orig}")
    
    # Search
    results_orig = hybrid_original.search(query_vector, k=10)
    neighbors_orig = [nid for nid, _ in results_orig]
    recall_orig = len(set(neighbors_orig) & set(ground_truth)) / 10
    
    print(f"Results: {neighbors_orig}")
    print(f"Recall@10: {recall_orig:.3f} ({recall_orig*100:.1f}%)")
    
    # Test fixed hybrid
    print("\n" + "="*50)
    print("🟢 FIXED HYBRID (Consistent distances)")
    print("="*50)
    
    hybrid_fixed = FixedHybridHNSW(k_children=200, n_probe=5)
    hybrid_fixed.build_base_index(dataset)
    hybrid_fixed.extract_parent_nodes(target_level=2)
    hybrid_fixed.build_parent_child_mapping_fixed()
    
    # Check coverage
    all_candidates_fixed = set()
    for children in hybrid_fixed.parent_child_map.values():
        all_candidates_fixed.update(children)
    
    coverage_fixed = len(all_candidates_fixed) / len(dataset) * 100
    query_in_candidates_fixed = query_id in all_candidates_fixed
    
    print(f"Coverage: {coverage_fixed:.1f}%")
    print(f"Query {query_id} in candidates: {query_in_candidates_fixed}")
    
    # Search
    results_fixed = hybrid_fixed.search(query_vector, k=10)
    neighbors_fixed = [nid for nid, _ in results_fixed]
    recall_fixed = len(set(neighbors_fixed) & set(ground_truth)) / 10
    
    print(f"Results: {neighbors_fixed}")
    print(f"Recall@10: {recall_fixed:.3f} ({recall_fixed*100:.1f}%)")
    
    # Summary
    print("\n" + "="*50)
    print("📊 COMPARISON SUMMARY")
    print("="*50)
    print(f"Original Hybrid:")
    print(f"  Coverage: {coverage_orig:.1f}%")
    print(f"  Query in candidates: {query_in_candidates_orig}")
    print(f"  Recall@10: {recall_orig*100:.1f}%")
    print()
    print(f"Fixed Hybrid:")
    print(f"  Coverage: {coverage_fixed:.1f}%")
    print(f"  Query in candidates: {query_in_candidates_fixed}")
    print(f"  Recall@10: {recall_fixed*100:.1f}%")
    print()
    print(f"Improvement: {(recall_fixed - recall_orig)*100:.1f} percentage points")

if __name__ == "__main__":
    test_hybrid_fix()
