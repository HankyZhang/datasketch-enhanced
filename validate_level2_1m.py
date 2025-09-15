#!/usr/bin/env python3
"""
Quick validation for 1M test approach - Level 2 constraints
Tests with smaller dataset to validate methodology before full 1M run
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

def validate_level2_approach():
    """Validate the level 2 approach with different dataset sizes."""
    print("🔍 Level 2 Validation Test")
    print("Testing parent distribution and n_probe constraints")
    print("=" * 60)
    
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Test different dataset sizes to understand scaling
    test_sizes = [10000, 50000, 100000]  # 10K, 50K, 100K
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size:,} vectors...")
        
        # Load subset
        base_vectors = read_fvecs("sift/sift_base.fvecs", size)
        
        # Test configuration: m=16, ef_construction=200
        distance_func = lambda x, y: np.linalg.norm(x - y)
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        
        # Build base index
        start_time = time.time()
        base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
        base_index.update(dataset)
        base_build_time = time.time() - start_time
        
        # Test different k_children values
        for k_children in [1000, 2000, 3000]:
            print(f"\n  Testing k_children = {k_children}...")
            
            start_time = time.time()
            hybrid_index = HNSWHybrid(
                base_index=base_index,
                parent_level=2,
                k_children=k_children,
                parent_child_method='exact'
            )
            hybrid_build_time = time.time() - start_time
            
            stats = hybrid_index.get_stats()
            num_parents = stats.get('num_parents', 0)
            max_n_probe = max(1, int(0.5 * num_parents))
            
            print(f"    Parents at level 2: {num_parents:,}")
            print(f"    Max n_probe allowed: {max_n_probe}")
            print(f"    Build time: {hybrid_build_time:.2f}s")
            print(f"    Constraint utilization: {max_n_probe / max(1, num_parents) * 100:.1f}%")
            
            # Quick search test
            if len(base_vectors) >= 1000:
                query = base_vectors[0]
                for n_probe in [1, min(5, max_n_probe), max_n_probe]:
                    start_time = time.time()
                    results = hybrid_index.search(query, k=10, n_probe=n_probe)
                    search_time = (time.time() - start_time) * 1000
                    print(f"      n_probe={n_probe}: {search_time:.2f}ms")
    
    # Estimate 1M performance
    print(f"\n🔮 1M Dataset Estimates:")
    print("-" * 40)
    
    # Based on scaling patterns
    estimated_parents_1m = num_parents * (1000000 / size)
    estimated_max_n_probe_1m = int(0.5 * estimated_parents_1m)
    estimated_build_time_1m = base_build_time * (1000000 / size) * 1.2  # 20% overhead
    
    print(f"Estimated parents at level 2: ~{estimated_parents_1m:,.0f}")
    print(f"Estimated max n_probe: ~{estimated_max_n_probe_1m:,}")
    print(f"Estimated build time: ~{estimated_build_time_1m/60:.1f} minutes")
    
    if estimated_max_n_probe_1m >= 10:
        print("✅ Good n_probe range expected for 1M dataset")
    else:
        print("⚠️  Limited n_probe range expected - may need parameter adjustment")
    
    return True

def test_parameter_sensitivity():
    """Test how parameters affect parent count at level 2."""
    print(f"\n🧪 Parameter Sensitivity Analysis")
    print("How different parameters affect level 2 parent distribution")
    print("=" * 60)
    
    # Load moderate dataset for testing
    base_vectors = read_fvecs("sift/sift_base.fvecs", 50000)
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # Test different m values
    print(f"\n📈 Effect of 'm' parameter:")
    for m in [8, 16, 24, 32]:
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        base_index = HNSW(distance_func=distance_func, m=m, ef_construction=200)
        base_index.update(dataset)
        
        hybrid_index = HNSWHybrid(base_index, parent_level=2, k_children=1000)
        stats = hybrid_index.get_stats()
        num_parents = stats.get('num_parents', 0)
        max_n_probe = int(0.5 * num_parents) if num_parents > 0 else 0
        
        print(f"  m={m:2d}: {num_parents:,} parents, max n_probe = {max_n_probe}")
    
    # Test different ef_construction values
    print(f"\n📈 Effect of 'ef_construction' parameter:")
    for ef_c in [100, 200, 400, 600]:
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        base_index = HNSW(distance_func=distance_func, m=16, ef_construction=ef_c)
        base_index.update(dataset)
        
        hybrid_index = HNSWHybrid(base_index, parent_level=2, k_children=1000)
        stats = hybrid_index.get_stats()
        num_parents = stats.get('num_parents', 0)
        max_n_probe = int(0.5 * num_parents) if num_parents > 0 else 0
        
        print(f"  ef_c={ef_c:3d}: {num_parents:,} parents, max n_probe = {max_n_probe}")

def main():
    """Main validation function."""
    print("🔬 Pre-1M Validation Suite")
    print("Validating approach before full 1M dataset test")
    print("=" * 70)
    
    try:
        # Check if SIFT data exists
        import os
        if not os.path.exists("sift/sift_base.fvecs"):
            print("❌ SIFT dataset not found!")
            return
        
        # Run validation tests
        validate_level2_approach()
        test_parameter_sensitivity()
        
        print(f"\n✅ Validation Complete!")
        print("🚀 Ready to run full 1M test: python test_1m_sift_level2.py")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
