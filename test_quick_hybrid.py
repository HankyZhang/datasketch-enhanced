#!/usr/bin/env python3
"""
Quick Test Runner for HNSW Hybrid System
========================================

This script runs a quick test to verify the hybrid HNSW system is working correctly
before running the full evaluation pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from complete_hybrid_evaluation import ComprehensiveEvaluator, EvaluationConfig
import time


def run_quick_test():
    """
    Run a quick test with smaller parameters to verify functionality.
    """
    print("HNSW Hybrid System - Quick Test")
    print("=" * 50)
    
    # Quick test configuration
    config = EvaluationConfig(
        dataset_size=5000,    # Small dataset for quick test
        vector_dim=64,        # Smaller dimension
        n_queries=200,        # Fewer queries
        k_values=[10],        # Single k value
        k_children_values=[500],  # Single k_children value
        n_probe_values=[10],  # Single n_probe value
        save_results=True,
        results_dir="quick_test_results"
    )
    
    print(f"Test Configuration:")
    print(f"  Dataset: {config.dataset_size:,} vectors")
    print(f"  Dimension: {config.vector_dim}")
    print(f"  Queries: {config.n_queries:,}")
    print(f"  k_children: {config.k_children_values}")
    print(f"  n_probe: {config.n_probe_values}")
    print()
    
    try:
        # Run evaluation
        evaluator = ComprehensiveEvaluator(config)
        
        # Phase 1: Objectives
        print("Phase 1: Setting objectives...")
        objectives = evaluator.phase1_objectives_and_concepts()
        
        # Phase 2: Data preparation
        print("\\nPhase 2: Preparing data...")
        prep_stats = evaluator.phase2_preparation_and_baseline()
        
        # Phase 3: Build structure
        print("\\nPhase 3: Building hybrid structure...")
        hybrid_index, structure_stats = evaluator.phase3_build_parent_child_structure(
            k_children=config.k_children_values[0]
        )
        
        # Phase 4: Test search
        print("\\nPhase 4: Testing search...")
        search_stats = evaluator.phase4_implement_two_stage_search(
            hybrid_index, 
            n_probe=config.n_probe_values[0]
        )
        
        # Phase 5: Evaluate
        print("\\nPhase 5: Evaluating performance...")
        eval_results = evaluator.phase5_experimental_evaluation(
            hybrid_index, 
            k=config.k_values[0]
        )
        
        # Summary
        print("\\n" + "=" * 50)
        print("QUICK TEST RESULTS")
        print("=" * 50)
        print(f"✅ Dataset: {prep_stats['dataset_size']:,} vectors")
        print(f"✅ Parent nodes: {structure_stats['num_parents']}")
        print(f"✅ Search candidates: {search_stats['num_candidates']}")
        print(f"✅ Recall@{config.k_values[0]}: {eval_results['recall@k']:.4f}")
        print(f"✅ Query time: {eval_results['avg_query_time']:.6f}s")
        print(f"✅ Build time: {structure_stats['total_build_time']:.2f}s")
        
        success = eval_results['recall@k'] > 0.5  # Expect at least 50% recall
        
        if success:
            print("\\n🎉 QUICK TEST PASSED!")
            print("System is working correctly. Ready for full evaluation.")
        else:
            print("\\n⚠️  QUICK TEST WARNING!")
            print("Low recall detected. Check system configuration.")
        
        return success, eval_results
        
    except Exception as e:
        print(f"\\n❌ QUICK TEST FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def run_medium_test():
    """
    Run a medium-scale test to verify performance at larger scale.
    """
    print("\\n" + "=" * 50)
    print("MEDIUM SCALE TEST")
    print("=" * 50)
    
    config = EvaluationConfig(
        dataset_size=50000,   # Medium dataset
        vector_dim=128,       # Standard dimension
        n_queries=1000,       # More queries
        k_values=[5, 10, 20], # Multiple k values
        k_children_values=[1000, 1500],  # Two k_children values
        n_probe_values=[10, 15],         # Two n_probe values
        save_results=True,
        results_dir="medium_test_results"
    )
    
    print(f"Medium Test Configuration:")
    print(f"  Dataset: {config.dataset_size:,} vectors")
    print(f"  Configurations: {len(config.k_children_values) * len(config.n_probe_values) * len(config.k_values)}")
    
    try:
        evaluator = ComprehensiveEvaluator(config)
        start_time = time.time()
        
        # Run parameter sweep
        results = evaluator.run_parameter_sweep()
        analysis = evaluator.analyze_results()
        
        total_time = time.time() - start_time
        
        print(f"\\nMedium test completed in {total_time:.2f} seconds")
        print(f"Tested {len(results)} configurations")
        
        # Check performance
        best_recall = max(r['recall@k'] for r in results if r['k'] == 10)
        avg_time = sum(r['avg_query_time'] for r in results) / len(results)
        
        print(f"Best Recall@10: {best_recall:.4f}")
        print(f"Average query time: {avg_time:.6f}s")
        
        success = best_recall > 0.7  # Expect at least 70% recall for k=10
        
        if success:
            print("\\n🎉 MEDIUM TEST PASSED!")
        else:
            print("\\n⚠️  MEDIUM TEST WARNING!")
            print("Performance may need optimization.")
        
        return success, results
        
    except Exception as e:
        print(f"\\n❌ MEDIUM TEST FAILED!")
        print(f"Error: {str(e)}")
        return False, None


def main():
    """
    Run test suite to verify system functionality.
    """
    print("HNSW Hybrid System Test Suite")
    print("=" * 60)
    
    # Run quick test first
    quick_success, quick_results = run_quick_test()
    
    if not quick_success:
        print("\\nStopping due to quick test failure.")
        return False
    
    # Ask user if they want to run medium test
    print("\\nQuick test passed! Run medium test? (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            medium_success, medium_results = run_medium_test()
            
            if medium_success:
                print("\\n🎉 ALL TESTS PASSED!")
                print("System is ready for full-scale evaluation.")
                print("\\nTo run complete evaluation:")
                print("  python complete_hybrid_evaluation.py")
                return True
            else:
                print("\\n⚠️  Medium test had issues. Review configuration.")
                return False
        else:
            print("\\nSkipping medium test.")
            print("\\nTo run complete evaluation:")
            print("  python complete_hybrid_evaluation.py")
            return True
            
    except KeyboardInterrupt:
        print("\\nTest interrupted by user.")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\\nTest suite completed successfully!")
    else:
        print("\\nTest suite encountered issues.")
