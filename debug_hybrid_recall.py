#!/usr/bin/env python3
"""
Debug Hybrid HNSW Recall Issues

This script diagnoses why hybrid HNSW is getting poor recall.
"""

import numpy as np
import time
from hnsw.hnsw import HNSW
from hybrid_hnsw.hnsw_hybrid import HNSWHybrid

def read_fvecs(filename, max_count=None):
    """Read .fvecs format files efficiently."""
    import struct
    vectors = []
    count = 0
    
    with open(filename, 'rb') as f:
        while True:
            if max_count and count >= max_count:
                break
                
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            
            vector_bytes = f.read(4 * dim)
            if len(vector_bytes) < 4 * dim:
                break
                
            vector = struct.unpack('f' * dim, vector_bytes)
            vectors.append(vector)
            count += 1
    
    return np.array(vectors, dtype=np.float32)

def debug_hybrid_mapping(hybrid_index, base_vectors):
    """Debug the parent-child mapping."""
    print("\n🔍 DEBUGGING HYBRID MAPPING")
    print("=" * 50)
    
    stats = hybrid_index.get_stats()
    num_parents = stats['num_parents']
    total_vectors = len(base_vectors)
    
    print(f"Total vectors in dataset: {total_vectors}")
    print(f"Number of parent nodes: {num_parents}")
    print(f"Expected children per parent: {total_vectors / num_parents:.1f}")
    print(f"Requested k_children: {hybrid_index.k_children}")
    
    # Check coverage
    all_children = set()
    parent_sizes = []
    
    for parent_id, children in hybrid_index.parent_child_map.items():
        parent_sizes.append(len(children))
        all_children.update(children)
    
    coverage = len(all_children) / total_vectors
    print(f"\nCoverage analysis:")
    print(f"  Unique children covered: {len(all_children)}/{total_vectors} ({coverage:.2%})")
    print(f"  Average children per parent: {np.mean(parent_sizes):.1f}")
    print(f"  Min children per parent: {min(parent_sizes)}")
    print(f"  Max children per parent: {max(parent_sizes)}")
    
    # Check if any vectors are missing
    missing_vectors = set(range(total_vectors)) - all_children
    if missing_vectors:
        print(f"  ⚠️  Missing {len(missing_vectors)} vectors from child mappings!")
    else:
        print(f"  ✅ All vectors are covered by at least one parent")
    
    return coverage, len(missing_vectors)

def test_simple_search(hybrid_index, query_vector, ground_truth_ids, k=10):
    """Test a simple search and show what's happening."""
    print(f"\n🔍 DEBUGGING SEARCH PROCESS")
    print("=" * 50)
    
    # Test different n_probe values
    for n_probe in [1, 5, 10, 20]:
        if n_probe > len(hybrid_index.parent_ids):
            continue
            
        print(f"\nTesting n_probe={n_probe}:")
        
        # Get search results
        results = hybrid_index.search(query_vector, k=100, n_probe=n_probe)
        result_ids = [rid for rid, _ in results[:k]]
        
        # Calculate recall
        recall = len(set(result_ids) & set(ground_truth_ids[:k])) / k
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  Found {len(results)} results total")
        print(f"  Top-{k} results: {result_ids[:k]}")
        print(f"  Ground truth top-{k}: {ground_truth_ids[:k].tolist()}")
        
        # Check which parents were probed
        parent_distances = []
        for pid, pvec in hybrid_index.parent_vectors.items():
            dist = np.linalg.norm(query_vector - pvec)
            parent_distances.append((dist, pid))
        parent_distances.sort()
        
        probed_parents = [pid for _, pid in parent_distances[:n_probe]]
        print(f"  Probed parents: {probed_parents}")
        
        # Check how many candidates we get from these parents
        all_candidates = set()
        for pid in probed_parents:
            if pid in hybrid_index.parent_child_map:
                children = hybrid_index.parent_child_map[pid]
                all_candidates.update(children)
                print(f"    Parent {pid}: {len(children)} children")
        
        print(f"  Total unique candidates: {len(all_candidates)}")
        
        # Check if ground truth is in candidates
        gt_in_candidates = len(set(all_candidates) & set(ground_truth_ids[:k]))
        print(f"  Ground truth in candidates: {gt_in_candidates}/{k}")

def main():
    print("🔍 Hybrid HNSW Recall Debugging")
    print("=" * 50)
    
    # Load small dataset for debugging
    base_vectors = read_fvecs("sift/sift_base.fvecs", 5000)
    query_vectors = read_fvecs("sift/sift_query.fvecs", 10)
    
    print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} queries")
    
    # Build base index
    print("\nBuilding base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    base_index.update(dataset)
    
    # Compute ground truth for first query
    query = query_vectors[0]
    distances = np.linalg.norm(base_vectors - query, axis=1)
    gt_indices = np.argsort(distances)
    
    print(f"Ground truth computed for query 0")
    
    # Test different hybrid configurations
    configs = [
        (2, 200),   # level 2, k_children=200 (reasonable)
        (2, 500),   # level 2, k_children=500 (might be too big)
        (2, 1000),  # level 2, k_children=1000 (definitely too big)
        (1, 200),   # level 1, k_children=200
    ]
    
    for parent_level, k_children in configs:
        print(f"\n{'='*60}")
        print(f"TESTING: parent_level={parent_level}, k_children={k_children}")
        print(f"{'='*60}")
        
        # Build hybrid index
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=parent_level,
            k_children=k_children,
            parent_child_method='approx'
        )
        
        # Debug the mapping
        coverage, missing = debug_hybrid_mapping(hybrid_index, base_vectors)
        
        # Test search
        test_simple_search(hybrid_index, query, gt_indices)
        
        print(f"\nSUMMARY for level={parent_level}, k_children={k_children}:")
        print(f"  Coverage: {coverage:.2%}")
        print(f"  Missing vectors: {missing}")

if __name__ == "__main__":
    main()
