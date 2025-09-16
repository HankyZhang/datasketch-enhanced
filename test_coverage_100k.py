#!/usr/bin/env python3
"""
Test coverage improvement: Compare baseline vs optimized (diversify_max=3, repair_min=1)
for 100K vectors to show how optimization algorithms increase child node coverage.
"""

import numpy as np
import struct
import time
from typing import Optional

def read_fvecs(filename: str, max_count: Optional[int] = None) -> np.ndarray:
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

def test_coverage_100k():
    """Test coverage for 100K vectors: baseline vs optimized."""
    print("🎯 Testing Coverage for 100K Vectors")
    print("=" * 50)
    print("Question: With 100,000 vectors, why only 74,000 child nodes?")
    print("Answer: Testing diversify_max=3, repair_min=1 optimization...")
    print()
    
    # Load 100K vectors
    print("Loading 100K SIFT vectors...")
    base_vectors = read_fvecs("sift/sift_base.fvecs", 100000)
    print(f"✅ Loaded {len(base_vectors)} base vectors")
    
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Build base index
    print("Building base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    print("  Adding vectors to index...")
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    start = time.time()
    base_index.update(dataset)
    base_time = time.time() - start
    print(f"✅ Base index built in {base_time:.1f}s")
    
    configurations = [
        {
            "name": "❌ BASELINE (No Optimization)", 
            "diversify_max": None, 
            "repair_min": None,
            "expected": "~74K child nodes (poor coverage)"
        },
        {
            "name": "✅ OPTIMIZED (diversify_max=3, repair_min=1)", 
            "diversify_max": 3, 
            "repair_min": 1,
            "expected": "~95K+ child nodes (better coverage)"
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{config['name']}")
        print("-" * 60)
        print(f"Expected: {config['expected']}")
        
        start_time = time.time()
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=2,
            k_children=1000,  # Standard k_children
            parent_child_method='approx',
            approx_ef=10000,  # Large approx_ef for high recall
            diversify_max_assignments=config['diversify_max'],
            repair_min_assignments=config['repair_min']
        )
        build_time = time.time() - start_time
        
        stats = hybrid_index.get_stats()
        
        result = {
            'name': config['name'],
            'diversify_max': config['diversify_max'],
            'repair_min': config['repair_min'],
            'build_time': build_time,
            'num_parents': stats.get('num_parents', 0),
            'num_children': stats.get('num_children', 0),
            'coverage_fraction': stats.get('coverage_fraction', 0),
            'avg_children_per_parent': stats.get('avg_children_per_parent', 0),
        }
        results.append(result)
        
        print(f"📊 RESULTS:")
        print(f"  Build time: {build_time:.1f}s")
        print(f"  Parents: {result['num_parents']:,}")
        print(f"  Children: {result['num_children']:,}")
        print(f"  Coverage: {result['coverage_fraction']:.1%}")
        print(f"  Avg children/parent: {result['avg_children_per_parent']:.0f}")
        
        # Specific answer to your question
        if config['diversify_max'] is None and config['repair_min'] is None:
            if result['num_children'] < 80000:
                print(f"  ⚠️  CONFIRMED: Only {result['num_children']:,} out of 100,000 vectors covered!")
                print(f"     This explains why you see ~74K child nodes.")
        else:
            improvement = result['num_children'] - results[0]['num_children']
            print(f"  🎉 IMPROVEMENT: +{improvement:,} more vectors covered!")
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 OPTIMIZATION IMPACT SUMMARY")
    print("=" * 60)
    
    baseline = results[0]
    optimized = results[1]
    
    coverage_improvement = optimized['coverage_fraction'] - baseline['coverage_fraction']
    children_improvement = optimized['num_children'] - baseline['num_children']
    
    print(f"Baseline coverage:  {baseline['num_children']:,} / 100,000 = {baseline['coverage_fraction']:.1%}")
    print(f"Optimized coverage: {optimized['num_children']:,} / 100,000 = {optimized['coverage_fraction']:.1%}")
    print()
    print(f"✅ Improvement: +{children_improvement:,} vectors (+{coverage_improvement:.1%})")
    print()
    
    # Answer the original question
    print("🎯 ANSWER TO YOUR QUESTION:")
    print(f"   Q: Why only 74,000 child nodes with 100,000 vectors?")
    print(f"   A: Without optimization, dense regions dominate parent-child mappings.")
    print(f"      Many vectors in sparse regions never get assigned to any parent.")
    print()
    print(f"   Q: Do diversify_max=3, repair_min=1 work?")
    if children_improvement > 10000:
        print(f"   A: YES! Optimization adds {children_improvement:,} more vectors to coverage.")
        print(f"      diversify_max=3 prevents over-concentration in dense areas.")
        print(f"      repair_min=1 ensures sparse vectors get assigned to nearest parents.")
    else:
        print(f"   A: Results inconclusive. May need different parameters or larger k_children.")

if __name__ == "__main__":
    import os
    if not os.path.exists("sift"):
        print("❌ SIFT dataset not found! Please ensure sift/ directory exists.")
    else:
        test_coverage_100k()
