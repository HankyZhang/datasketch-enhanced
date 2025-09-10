#!/usr/bin/env python3
"""
Simple test to verify the system works
"""

import sys
import os
sys.path.append('.')

from complete_hybrid_evaluation import ComprehensiveEvaluator, EvaluationConfig

def test_basic_functionality():
    """Test basic functionality"""
    print("Testing basic functionality...")
    
    config = EvaluationConfig(
        dataset_size=1000,
        vector_dim=32,
        n_queries=50,
        k_values=[10],
        k_children_values=[200],
        n_probe_values=[5],
        save_results=False
    )
    
    evaluator = ComprehensiveEvaluator(config)
    
    try:
        # Test phase 1
        evaluator.phase1_objectives_and_concepts()
        print("✅ Phase 1: OK")
        
        # Test phase 2
        evaluator.phase2_preparation_and_baseline()
        print("✅ Phase 2: OK")
        
        # Test phase 3
        hybrid_index, _ = evaluator.phase3_build_parent_child_structure(200)
        print("✅ Phase 3: OK")
        
        # Test phase 4
        evaluator.phase4_implement_two_stage_search(hybrid_index, 5)
        print("✅ Phase 4: OK")
        
        # Test phase 5
        results = evaluator.phase5_experimental_evaluation(hybrid_index, 10)
        recall = results["recall@k"]
        print(f"✅ Phase 5: OK - Recall: {recall:.4f}")
        
        print("\n🎉 All phases working correctly!")
        return True, recall
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

if __name__ == "__main__":
    success, recall = test_basic_functionality()
    if success:
        print(f"System test PASSED with recall {recall:.4f}")
    else:
        print("System test FAILED")
