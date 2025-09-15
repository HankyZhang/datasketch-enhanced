#!/usr/bin/env python3
"""
Quick SIFT Test for Hybrid HNSW

A simple script to quickly test hybrid HNSW on SIFT data
with a small subset for rapid verification.
"""

import numpy as np
import struct
import time

def read_fvecs_sample(filename: str, max_vectors: int = 1000) -> np.ndarray:
    """Read first N vectors from .fvecs file."""
    vectors = []
    with open(filename, 'rb') as f:
        for _ in range(max_vectors):
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            
            # Read vector
            vector_bytes = f.read(4 * dim)
            if len(vector_bytes) < 4 * dim:
                break
            vector = struct.unpack('f' * dim, vector_bytes)
            vectors.append(vector)
    
    return np.array(vectors, dtype=np.float32)

def read_ivecs_sample(filename: str, max_vectors: int = 100) -> np.ndarray:
    """Read first N vectors from .ivecs file."""
    vectors = []
    with open(filename, 'rb') as f:
        for _ in range(max_vectors):
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            
            # Read vector
            vector_bytes = f.read(4 * dim)
            if len(vector_bytes) < 4 * dim:
                break
            vector = struct.unpack('i' * dim, vector_bytes)
            vectors.append(vector)
    
    return np.array(vectors, dtype=np.int32)

def quick_sift_test():
    """Quick test with small SIFT subset."""
    print("🚀 Quick SIFT Test for Hybrid HNSW")
    print("=" * 40)
    
    # Load small subset
    print("Loading small SIFT subset...")
    base_vectors = read_fvecs_sample("sift/sift_base.fvecs", 5000)
    query_vectors = read_fvecs_sample("sift/sift_query.fvecs", 100)
    ground_truth = read_ivecs_sample("sift/sift_groundtruth.ivecs", 100)
    
    print(f"Loaded {len(base_vectors)} base vectors")
    print(f"Loaded {len(query_vectors)} query vectors")
    print(f"Vector dimension: {base_vectors.shape[1]}")
    print()
    
    # Import HNSW modules
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Distance function
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # Test 1: Standard HNSW
    print("=== Testing Standard HNSW ===")
    start_time = time.time()
    
    standard_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    standard_index.update(dataset)
    
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.2f}s")
    
    # Test queries
    query_times = []
    recalls = []
    
    for i in range(10):  # Test 10 queries
        query = query_vectors[i]
        gt = ground_truth[i][:10]  # Top 10 ground truth
        
        start_time = time.time()
        results = standard_index.query(query, k=10, ef=200)
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        # Calculate recall@10
        result_ids = [rid for rid, _ in results]
        recall = len(set(result_ids) & set(gt)) / 10
        recalls.append(recall)
    
    avg_query_time = np.mean(query_times) * 1000  # ms
    avg_recall = np.mean(recalls)
    
    print(f"Avg query time: {avg_query_time:.2f}ms")
    print(f"Avg recall@10: {avg_recall:.4f}")
    print()
    
    # Test 2: Hybrid HNSW
    print("=== Testing Hybrid HNSW ===")
    start_time = time.time()
    
    # Build base index (reuse the one above)
    hybrid_index = HNSWHybrid(
        base_index=standard_index,
        parent_level=2,
        k_children=500,
        parent_child_method='approx'
    )
    
    hybrid_build_time = time.time() - start_time
    print(f"Hybrid build time: {hybrid_build_time:.2f}s")
    
    # Get stats
    stats = hybrid_index.get_stats()
    print(f"Parents: {stats.get('num_parents', 0)}, Children: {stats.get('num_children', 0)}")
    
    # Test different n_probe values
    for n_probe in [1, 5, 10]:
        query_times = []
        recalls = []
        
        for i in range(10):  # Test 10 queries
            query = query_vectors[i]
            gt = ground_truth[i][:10]  # Top 10 ground truth
            
            start_time = time.time()
            results = hybrid_index.search(query, k=10, n_probe=n_probe)
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # Calculate recall@10
            result_ids = [rid for rid, _ in results]
            recall = len(set(result_ids) & set(gt)) / 10
            recalls.append(recall)
        
        avg_query_time = np.mean(query_times) * 1000  # ms
        avg_recall = np.mean(recalls)
        
        print(f"n_probe={n_probe}: Query time={avg_query_time:.2f}ms, Recall@10={avg_recall:.4f}")
    
    print("\n✅ Quick test completed!")
    print("\nTo run full evaluation, use: python sift_evaluation.py")

if __name__ == "__main__":
    quick_sift_test()
