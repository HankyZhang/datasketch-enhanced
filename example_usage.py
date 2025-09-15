#!/usr/bin/env python3
"""
HNSW Enhanced - Example Usage

This script demonstrates how to use the simplified HNSW package structure
with both standard HNSW and hybrid HNSW implementations.
"""

import numpy as np
import time

def example_standard_hnsw():
    """Example using standard HNSW implementation."""
    print("=== Standard HNSW Example ===")
    
    # Import standard HNSW
    from hnsw import HNSW
    
    # Create distance function
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # Initialize HNSW index
    index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    # Create test dataset
    print("Creating test dataset...")
    dataset = {i: np.random.random(64).astype(np.float32) for i in range(1000)}
    
    # Build index
    print("Building index...")
    start_time = time.time()
    index.update(dataset)
    build_time = time.time() - start_time
    print(f"Index built in {build_time:.2f} seconds")
    
    # Perform queries
    print("Performing queries...")
    query_vector = np.random.random(64).astype(np.float32)
    
    start_time = time.time()
    results = index.query(query_vector, k=10, ef=200)
    query_time = time.time() - start_time
    
    print(f"Query completed in {query_time*1000:.2f} ms")
    print(f"Found {len(results)} nearest neighbors")
    print(f"First result: ID={results[0][0]}, Distance={results[0][1]:.4f}")
    print()

def example_hybrid_hnsw():
    """Example using hybrid HNSW implementation."""
    print("=== Hybrid HNSW Example ===")
    
    # Import modules
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Create distance function
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # Create test dataset
    print("Creating test dataset...")
    dataset = {i: np.random.random(64).astype(np.float32) for i in range(2000)}
    
    # Build base HNSW index
    print("Building base HNSW index...")
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    base_index.update(dataset)
    
    # Build hybrid index
    print("Building hybrid index...")
    start_time = time.time()
    hybrid_index = HNSWHybrid(
        base_index=base_index,
        parent_level=2,
        k_children=500,
        parent_child_method='approx'
    )
    build_time = time.time() - start_time
    print(f"Hybrid index built in {build_time:.2f} seconds")
    
    # Print index statistics
    stats = hybrid_index.get_stats()
    print(f"Parent nodes: {stats.get('num_parents', 'N/A')}")
    print(f"Children nodes: {stats.get('num_children', 'N/A')}")
    print(f"Avg children/parent: {stats.get('avg_children_per_parent', 0):.1f}")
    
    # Perform queries
    print("Performing hybrid queries...")
    query_vector = np.random.random(64).astype(np.float32)
    
    start_time = time.time()
    results = hybrid_index.search(query_vector, k=10)
    query_time = time.time() - start_time
    
    print(f"Hybrid query completed in {query_time*1000:.2f} ms")
    print(f"Found {len(results)} nearest neighbors")
    if results:
        print(f"First result: ID={results[0][0]}, Distance={results[0][1]:.4f}")
    print()

def example_comparison():
    """Compare standard vs hybrid HNSW performance."""
    print("=== Performance Comparison ===")
    
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Common setup
    distance_func = lambda x, y: np.linalg.norm(x - y)
    dataset = {i: np.random.random(32).astype(np.float32) for i in range(5000)}
    query_vector = np.random.random(32).astype(np.float32)
    
    # Standard HNSW
    print("Testing standard HNSW...")
    standard_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    start_time = time.time()
    standard_index.update(dataset)
    standard_build_time = time.time() - start_time
    
    start_time = time.time()
    standard_results = standard_index.query(query_vector, k=10, ef=200)
    standard_query_time = time.time() - start_time
    
    # Hybrid HNSW
    print("Testing hybrid HNSW...")
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    base_index.update(dataset)
    
    start_time = time.time()
    hybrid_index = HNSWHybrid(base_index=base_index, parent_level=2, k_children=800)
    hybrid_build_time = time.time() - start_time
    
    start_time = time.time()
    hybrid_results = hybrid_index.search(query_vector, k=10)
    hybrid_query_time = time.time() - start_time
    
    # Results
    print("\nComparison Results:")
    print(f"Standard HNSW - Build: {standard_build_time:.2f}s, Query: {standard_query_time*1000:.2f}ms")
    print(f"Hybrid HNSW   - Build: {hybrid_build_time:.2f}s, Query: {hybrid_query_time*1000:.2f}ms")
    print(f"Standard found {len(standard_results)} results")
    print(f"Hybrid found {len(hybrid_results)} results")
    print()

def main():
    """Run all examples."""
    print("HNSW Enhanced - Usage Examples\n")
    print("This script demonstrates the simplified HNSW package structure.")
    print("=" * 60)
    print()
    
    try:
        example_standard_hnsw()
        example_hybrid_hnsw()
        example_comparison()
        
        print("✅ All examples completed successfully!")
        print("\nFor more advanced usage, check the documentation in the docs/ folder.")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
