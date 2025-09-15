#!/usr/bin/env python3
"""
Quick validation test for 1M dataset parameters before full test
"""

import numpy as np
import struct
import time

def read_fvecs(filename: str, max_count: int = None) -> np.ndarray:
    """Read .fvecs format files efficiently."""
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

def quick_validation():
    """Quick test to validate approach before full 1M test."""
    print("🔍 Quick Validation Test")
    print("Testing with 50K subset to validate approach...")
    print("=" * 50)
    
    # Load subset for validation
    base_vectors = read_fvecs("sift/sift_base.fvecs", 50000)  # 50K subset
    query_vectors = read_fvecs("sift/sift_query.fvecs", 20)    # 20 queries
    
    print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} queries")
    
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Test configuration: m=16, ef_construction=200, k_children=2000
    print("\nBuilding base HNSW (m=16, ef_construction=200)...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    
    start_time = time.time()
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    base_index.update(dataset)
    base_build_time = time.time() - start_time
    
    print(f"Base index built in {base_build_time:.2f}s")
    
    # Build hybrid with parent_level=2
    print("\nBuilding hybrid structure (parent_level=2, k_children=2000)...")
    start_time = time.time()
    hybrid_index = HNSWHybrid(
        base_index=base_index,
        parent_level=2,
        k_children=2000,
        parent_child_method='exact'
    )
    hybrid_build_time = time.time() - start_time
    
    stats = hybrid_index.get_stats()
    num_parents = stats.get('num_parents', 0)
    
    print(f"Hybrid structure built in {hybrid_build_time:.2f}s")
    print(f"Parents at level 2: {num_parents}")
    print(f"Max n_probe allowed: {int(0.5 * num_parents)}")
    
    # Quick search test
    print("\nTesting search with different n_probe values...")
    max_n_probe = int(0.5 * num_parents)
    test_n_probes = [min(n, max_n_probe) for n in [1, 2, 5, 10] if min(n, max_n_probe) >= 1]
    
    for n_probe in test_n_probes:
        start_time = time.time()
        results = hybrid_index.search(query_vectors[0], k=10, n_probe=n_probe)
        search_time = (time.time() - start_time) * 1000
        print(f"  n_probe={n_probe}: {search_time:.2f}ms, found {len(results)} results")
    
    print(f"\n✅ Validation successful!")
    print(f"Expected scaling for 1M dataset:")
    print(f"  - Build time: ~{base_build_time * 20:.0f}s (20x scaling)")
    print(f"  - Parents at level 2: ~{num_parents * 20} (estimated)")
    print(f"  - Max n_probe: ~{int(0.5 * num_parents * 20)} (estimated)")
    
    return True

if __name__ == "__main__":
    import os
    if not os.path.exists("sift"):
        print("❌ SIFT dataset not found!")
        exit(1)
    
    try:
        quick_validation()
        print("\n🚀 Ready to run full 1M test!")
        print("Run: python test_1m_level2.py")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
