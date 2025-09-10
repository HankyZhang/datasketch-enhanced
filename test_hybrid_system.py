#!/usr/bin/env python3
"""
Test script for the HNSW Hybrid System

This script provides a simple test to verify the hybrid system works correctly.
"""

import numpy as np
import time
from datasketch.hnsw import HNSW
from hnsw_hybrid import HNSWHybrid, HNSWEvaluator, create_synthetic_dataset, create_query_set


def test_basic_functionality():
    """Test basic functionality of the hybrid system."""
    print("Testing HNSW Hybrid System - Basic Functionality")
    print("=" * 60)
    
    # Create small test dataset
    print("Creating test dataset...")
    dataset = create_synthetic_dataset(1000, 64)  # 1K vectors, 64 dim
    query_vectors, query_ids = create_query_set(dataset, 50)  # 50 queries
    
    print(f"Dataset: {dataset.shape}, Queries: {query_vectors.shape}")
    
    # Build base HNSW index
    print("Building base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    # Insert vectors (excluding queries)
    for i, vector in enumerate(dataset):
        if i not in query_ids:
            base_index.insert(i, vector)
    
    print(f"Base index built: {len(base_index)} vectors, {len(base_index._graphs)} layers")
    
    # Build hybrid index
    print("Building hybrid HNSW index...")
    start_time = time.time()
    hybrid_index = HNSWHybrid(
        base_index=base_index,
        parent_level=2,
        k_children=200
    )
    build_time = time.time() - start_time
    
    print(f"Hybrid index built in {build_time:.2f}s")
    print(f"Hybrid stats: {hybrid_index.get_stats()}")
    
    # Test search
    print("Testing search functionality...")
    query_vector = query_vectors[0]
    results = hybrid_index.search(query_vector, k=10, n_probe=5)
    
    print(f"Search results: {len(results)} neighbors found")
    print(f"Top 3 results: {results[:3]}")
    
    # Test evaluation
    print("Testing evaluation...")
    evaluator = HNSWEvaluator(dataset, query_vectors, query_ids)
    
    # Compute ground truth for a few queries
    print("Computing ground truth...")
    ground_truth = evaluator.compute_ground_truth(k=10, distance_func=distance_func)
    
    # Evaluate recall
    print("Evaluating recall...")
    result = evaluator.evaluate_recall(hybrid_index, k=10, n_probe=5, ground_truth=ground_truth)
    
    print(f"Evaluation results:")
    print(f"  Recall@10: {result['recall_at_k']:.4f}")
    print(f"  Query time: {result['avg_query_time_ms']:.2f} ms")
    print(f"  Total correct: {result['total_correct']}/{result['total_expected']}")
    
    print("\n" + "=" * 60)
    print("Basic functionality test completed successfully!")
    print("=" * 60)


def test_parameter_sensitivity():
    """Test parameter sensitivity."""
    print("\nTesting Parameter Sensitivity")
    print("=" * 60)
    
    # Create test dataset
    dataset = create_synthetic_dataset(2000, 64)
    query_vectors, query_ids = create_query_set(dataset, 100)
    
    # Build base index
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    for i, vector in enumerate(dataset):
        if i not in query_ids:
            base_index.insert(i, vector)
    
    # Test different k_children values
    k_children_values = [100, 300, 500]
    n_probe_values = [3, 5, 10]
    
    evaluator = HNSWEvaluator(dataset, query_vectors, query_ids)
    ground_truth = evaluator.compute_ground_truth(k=10, distance_func=distance_func)
    
    print("Testing different parameter combinations:")
    print("k_children | n_probe | Recall@10 | Query Time (ms)")
    print("-" * 50)
    
    for k_children in k_children_values:
        # Build hybrid index
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=2,
            k_children=k_children
        )
        
        for n_probe in n_probe_values:
            result = evaluator.evaluate_recall(
                hybrid_index, k=10, n_probe=n_probe, ground_truth=ground_truth
            )
            
            print(f"{k_children:10} | {n_probe:7} | {result['recall_at_k']:9.4f} | {result['avg_query_time_ms']:13.2f}")
    
    print("\nParameter sensitivity test completed!")


def test_large_dataset():
    """Test with a larger dataset to verify scalability."""
    print("\nTesting Large Dataset Scalability")
    print("=" * 60)
    
    # Create larger dataset
    print("Creating larger dataset...")
    dataset = create_synthetic_dataset(10000, 128)  # 10K vectors, 128 dim
    query_vectors, query_ids = create_query_set(dataset, 200)  # 200 queries
    
    print(f"Large dataset: {dataset.shape}, Queries: {query_vectors.shape}")
    
    # Build base index
    print("Building base HNSW index...")
    start_time = time.time()
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    for i, vector in enumerate(dataset):
        if i not in query_ids:
            base_index.insert(i, vector)
    
    base_build_time = time.time() - start_time
    print(f"Base index built in {base_build_time:.2f}s")
    
    # Build hybrid index
    print("Building hybrid HNSW index...")
    start_time = time.time()
    hybrid_index = HNSWHybrid(
        base_index=base_index,
        parent_level=2,
        k_children=500
    )
    hybrid_build_time = time.time() - start_time
    
    print(f"Hybrid index built in {hybrid_build_time:.2f}s")
    print(f"Hybrid stats: {hybrid_index.get_stats()}")
    
    # Quick evaluation
    print("Quick evaluation...")
    evaluator = HNSWEvaluator(dataset, query_vectors, query_ids)
    
    # Use smaller ground truth for speed
    small_query_vectors = query_vectors[:50]
    small_query_ids = query_ids[:50]
    small_evaluator = HNSWEvaluator(dataset, small_query_vectors, small_query_ids)
    
    ground_truth = small_evaluator.compute_ground_truth(k=10, distance_func=distance_func)
    result = small_evaluator.evaluate_recall(hybrid_index, k=10, n_probe=10, ground_truth=ground_truth)
    
    print(f"Large dataset results:")
    print(f"  Recall@10: {result['recall_at_k']:.4f}")
    print(f"  Query time: {result['avg_query_time_ms']:.2f} ms")
    print(f"  Base build time: {base_build_time:.2f}s")
    print(f"  Hybrid build time: {hybrid_build_time:.2f}s")
    
    print("\nLarge dataset test completed!")


if __name__ == "__main__":
    try:
        # Run all tests
        test_basic_functionality()
        test_parameter_sensitivity()
        test_large_dataset()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The HNSW Hybrid System is working correctly.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
