"""
HNSW Hybrid Evaluation Test Runner
=================================

This script runs the complete evaluation pipeline for the hybrid HNSW system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hnsw_hybrid_evaluation import (
    HybridHNSWIndex, 
    RecallEvaluator, 
    generate_synthetic_dataset, 
    create_query_set
)
import numpy as np
import time
import json


def run_parameter_sweep():
    """
    Run parameter sweep to find optimal k_children and n_probe values.
    """
    print("=== Parameter Sweep Evaluation ===")
    
    # Configuration
    DATASET_SIZE = 10000  # Smaller for quick testing
    VECTOR_DIM = 64
    N_QUERIES = 500
    K = 10
    
    # Parameter ranges to test
    k_children_values = [500, 1000, 1500]
    n_probe_values = [5, 10, 15, 20]
    
    # Generate dataset once
    print("Generating dataset...")
    dataset = generate_synthetic_dataset(DATASET_SIZE, VECTOR_DIM)
    query_set = create_query_set(dataset, N_QUERIES)
    
    # Initialize evaluator
    evaluator = RecallEvaluator(dataset)
    
    results = []
    
    for k_children in k_children_values:
        for n_probe in n_probe_values:
            print(f"\nTesting k_children={k_children}, n_probe={n_probe}")
            
            # Build hybrid index
            hybrid_index = HybridHNSWIndex(k_children=k_children, n_probe=n_probe)
            
            # Build base index
            hybrid_index.build_base_index(dataset, m=16, ef_construction=100)
            
            # Extract parent nodes
            hybrid_index.extract_parent_nodes(target_level=1)  # Use level 1 for smaller dataset
            
            if len(hybrid_index.parent_ids) == 0:
                print(f"No parent nodes found at level 1, trying level 0...")
                hybrid_index.extract_parent_nodes(target_level=0)
            
            if len(hybrid_index.parent_ids) == 0:
                print("No parent nodes found, skipping this configuration")
                continue
            
            # Build parent-child mapping
            hybrid_index.build_parent_child_mapping()
            
            # Evaluate recall
            result = evaluator.evaluate_recall(hybrid_index, query_set, k=K)
            results.append(result)
            
            print(f"Recall@{K}: {result['recall@k']:.4f}, Avg Query Time: {result['avg_query_time']:.6f}s")
    
    return results


def run_basic_evaluation():
    """
    Run basic evaluation with default parameters.
    """
    print("=== Basic HNSW Hybrid Evaluation ===")
    
    # Configuration
    DATASET_SIZE = 20000
    VECTOR_DIM = 128
    N_QUERIES = 1000
    K = 10
    
    print(f"Dataset size: {DATASET_SIZE}")
    print(f"Vector dimension: {VECTOR_DIM}")
    print(f"Number of queries: {N_QUERIES}")
    print(f"k for evaluation: {K}")
    
    # Generate dataset
    print("\nGenerating dataset...")
    dataset = generate_synthetic_dataset(DATASET_SIZE, VECTOR_DIM)
    query_set = create_query_set(dataset, N_QUERIES)
    
    # Build hybrid index
    print("\nBuilding Hybrid HNSW Index...")
    hybrid_index = HybridHNSWIndex(k_children=1000, n_probe=10)
    
    # Build base index
    hybrid_index.build_base_index(dataset, m=16, ef_construction=200)
    
    # Extract parent nodes from level 2
    hybrid_index.extract_parent_nodes(target_level=2)
    
    # If no nodes at level 2, try level 1
    if len(hybrid_index.parent_ids) == 0:
        print("No nodes found at level 2, trying level 1...")
        hybrid_index.extract_parent_nodes(target_level=1)
    
    # If still no nodes, try level 0
    if len(hybrid_index.parent_ids) == 0:
        print("No nodes found at level 1, trying level 0...")
        hybrid_index.extract_parent_nodes(target_level=0)
    
    print(f"Found {len(hybrid_index.parent_ids)} parent nodes")
    
    if len(hybrid_index.parent_ids) == 0:
        print("ERROR: No parent nodes found at any level!")
        return None
    
    # Build parent-child mapping
    hybrid_index.build_parent_child_mapping()
    
    # Evaluate recall
    print("\nEvaluating Recall Performance...")
    evaluator = RecallEvaluator(dataset)
    results = evaluator.evaluate_recall(hybrid_index, query_set, k=K)
    
    # Print results
    print("\n=== Final Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    return results


def test_hnsw_levels():
    """
    Test what levels are available in the HNSW graph.
    """
    print("=== Testing HNSW Graph Structure ===")
    
    # Small dataset for testing
    dataset = generate_synthetic_dataset(1000, 64)
    
    # Build HNSW index
    from datasketch.hnsw import HNSW
    
    hnsw = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y), m=16, ef_construction=100)
    hnsw.update(dataset)
    
    # Analyze graph structure
    level_counts = {}
    max_level = len(hnsw._graphs) - 1
    
    print(f"Graph analysis for {len(dataset)} nodes:")
    print(f"Total layers: {len(hnsw._graphs)}")
    print(f"Maximum level: {max_level}")
    
    for level in range(len(hnsw._graphs)):
        level_counts[level] = len(hnsw._graphs[level])
        print(f"Level {level}: {level_counts[level]} nodes")
    
    return level_counts


if __name__ == "__main__":
    print("HNSW Hybrid Evaluation Test Suite")
    print("=" * 50)
    
    # Test 1: Analyze HNSW structure
    print("\nTest 1: Analyzing HNSW Graph Structure")
    level_info = test_hnsw_levels()
    
    # Test 2: Basic evaluation
    print("\nTest 2: Basic Evaluation")
    try:
        results = run_basic_evaluation()
        if results:
            print("✓ Basic evaluation completed successfully")
        else:
            print("✗ Basic evaluation failed")
    except Exception as e:
        print(f"✗ Basic evaluation failed with error: {e}")
    
    # Test 3: Parameter sweep (optional)
    print("\nTest 3: Parameter Sweep (optional)")
    try:
        sweep_results = run_parameter_sweep()
        print(f"✓ Parameter sweep completed with {len(sweep_results)} configurations")
        
        # Find best configuration
        if sweep_results:
            best_result = max(sweep_results, key=lambda x: x['recall@k'])
            print(f"Best configuration: k_children={best_result['k_children']}, n_probe={best_result['n_probe']}")
            print(f"Best recall@{best_result['k']}: {best_result['recall@k']:.4f}")
    except Exception as e:
        print(f"✗ Parameter sweep failed with error: {e}")
    
    print("\nEvaluation suite completed!")
