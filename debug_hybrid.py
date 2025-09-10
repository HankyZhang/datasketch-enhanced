#!/usr/bin/env python3
"""
Debug script for HNSW Hybrid System

This script helps debug issues with the hybrid system.
"""

import numpy as np
from datasketch.hnsw import HNSW
from hnsw_hybrid import HNSWHybrid, HNSWEvaluator, create_synthetic_dataset, create_query_set


def debug_hybrid_system():
    """Debug the hybrid system to identify issues."""
    print("Debugging HNSW Hybrid System")
    print("=" * 60)
    
    # Create small test dataset
    print("Creating test dataset...")
    dataset = create_synthetic_dataset(100, 32)  # Small dataset for debugging
    query_vectors, query_ids = create_query_set(dataset, 5)  # 5 queries
    
    print(f"Dataset: {dataset.shape}, Queries: {query_vectors.shape}")
    print(f"Query IDs: {query_ids}")
    
    # Build base HNSW index
    print("Building base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    # Insert vectors (excluding queries)
    for i, vector in enumerate(dataset):
        if i not in query_ids:
            base_index.insert(i, vector)
    
    print(f"Base index built: {len(base_index)} vectors, {len(base_index._graphs)} layers")
    
    # Debug layer structure
    print("\nDebugging layer structure:")
    for i, layer in enumerate(base_index._graphs):
        print(f"  Layer {i}: {len(layer)} nodes")
        if i <= 2:  # Show first few layers
            # _Layer objects are iterable, so we can get the keys this way
            layer_keys = list(layer)[:10]  # Get first 10 node keys
            print(f"    Nodes: {layer_keys}...")  # Show first 10 nodes
    
    # Build hybrid index
    print("\nBuilding hybrid HNSW index...")
    # Use level 1 since we only have 2 layers (0 and 1)
    hybrid_index = HNSWHybrid(
        base_index=base_index,
        parent_level=1,  # Use level 1 instead of 2
        k_children=50
    )
    
    print(f"Hybrid stats: {hybrid_index.get_stats()}")
    
    # Debug parent-child mapping
    print("\nDebugging parent-child mapping:")
    parent_child_info = hybrid_index.get_parent_child_info()
    print(f"  Number of parents: {parent_child_info['num_parents']}")
    print(f"  Number of children: {parent_child_info['num_children']}")
    
    for parent_id, children in list(parent_child_info['parent_child_map'].items())[:3]:
        print(f"  Parent {parent_id}: {len(children)} children")
        print(f"    First 5 children: {children[:5]}")
    
    # Test search with debugging
    print("\nTesting search with debugging...")
    query_vector = query_vectors[0]
    print(f"Query vector shape: {query_vector.shape}")
    
    # Stage 1 debug
    print("\nStage 1 - Coarse search:")
    parent_candidates = hybrid_index._stage1_coarse_search(query_vector, n_probe=3)
    print(f"  Parent candidates: {parent_candidates}")
    
    # Stage 2 debug
    print("\nStage 2 - Fine search:")
    if parent_candidates:
        candidate_children = set()
        print(f"  Parent candidates: {parent_candidates}")
        print(f"  Available parent-child map keys: {list(hybrid_index.parent_child_map.keys())}")
        
        for i, candidate in enumerate(parent_candidates):
            print(f"  Candidate {i}: {candidate}")
            print(f"    Type: {type(candidate)}")
            print(f"    Length: {len(candidate)}")
            if len(candidate) == 2:
                distance, parent_id = candidate
                print(f"    Distance: {distance}, Parent ID: {parent_id}")
                print(f"  Checking parent {parent_id}...")
                if parent_id in hybrid_index.parent_child_map:
                    children = hybrid_index.parent_child_map[parent_id]
                    candidate_children.update(children)
                    print(f"    Parent {parent_id}: {len(children)} children")
                else:
                    print(f"    Parent {parent_id}: NOT FOUND in parent_child_map!")
            else:
                print(f"    Invalid candidate format!")
        
        print(f"  Total candidate children: {len(candidate_children)}")
        
        if candidate_children:
            # Calculate distances to candidate children
            child_distances = []
            for child_id in list(candidate_children)[:10]:  # Test first 10
                if child_id in hybrid_index.child_vectors:
                    distance = hybrid_index.distance_func(query_vector, hybrid_index.child_vectors[child_id])
                    child_distances.append((distance, child_id))
                    print(f"    Child {child_id}: distance = {distance:.4f}")
            
            print(f"  Child distances calculated: {len(child_distances)}")
            
            if child_distances:
                child_distances.sort()
                print(f"  Top 3 results: {child_distances[:3]}")
            else:
                print("  No child distances calculated!")
        else:
            print("  No candidate children found!")
    else:
        print("  No parent candidates found!")
    
    # Test full search
    print("\nFull search test:")
    print("  Calling hybrid_index.search()...")
    results = hybrid_index.search(query_vector, k=5, n_probe=3)
    print(f"  Search results: {len(results)} neighbors")
    print(f"  Results: {results}")
    
    # Debug the search method step by step
    print("\nDebugging search method step by step:")
    print("  Stage 1 - Coarse search:")
    parent_candidates = hybrid_index._stage1_coarse_search(query_vector, n_probe=3)
    print(f"    Parent candidates: {parent_candidates}")
    
    print("  Stage 2 - Fine search:")
    stage2_results = hybrid_index._stage2_fine_search(query_vector, parent_candidates, k=5)
    print(f"    Stage 2 results: {len(stage2_results)} neighbors")
    print(f"    Stage 2 results: {stage2_results}")
    
    # Test with different parameters
    print("\nTesting with different parameters:")
    for n_probe in [1, 2, 3, 5]:
        results = hybrid_index.search(query_vector, k=5, n_probe=n_probe)
        print(f"  n_probe={n_probe}: {len(results)} results")
    
    print("\nDebug completed!")


if __name__ == "__main__":
    debug_hybrid_system()
