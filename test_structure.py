#!/usr/bin/env python3
"""
Simple test script to verify the HNSW package structure works correctly.
"""

import sys
import os
import numpy as np

def test_hnsw_import():
    """Test importing the standard HNSW module."""
    try:
        from hnsw import HNSW
        print("✓ Standard HNSW import successful")
        return True
    except ImportError as e:
        print(f"✗ Standard HNSW import failed: {e}")
        return False

def test_hybrid_hnsw_import():
    """Test importing the hybrid HNSW module."""
    try:
        from hybrid_hnsw import HNSWHybrid
        print("✓ Hybrid HNSW import successful")
        return True
    except ImportError as e:
        print(f"✗ Hybrid HNSW import failed: {e}")
        return False

def test_optimized_hnsw_import():
    """Test importing the optimized HNSW module."""
    try:
        import optimized_hnsw
        print("✓ Optimized HNSW module import successful")
        return True
    except ImportError as e:
        print(f"✗ Optimized HNSW import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic HNSW functionality."""
    try:
        from hnsw import HNSW
        
        # Create a simple HNSW index
        distance_func = lambda x, y: np.linalg.norm(x - y)
        index = HNSW(distance_func=distance_func, m=4, ef_construction=50)
        
        # Add some test data
        test_data = {i: np.random.random(10) for i in range(20)}
        index.update(test_data)
        
        # Perform a query
        query = np.random.random(10)
        results = index.query(query, k=5)
        
        if len(results) == 5:
            print("✓ Basic HNSW functionality test passed")
            return True
        else:
            print(f"✗ Basic HNSW functionality test failed: got {len(results)} results instead of 5")
            return False
            
    except Exception as e:
        print(f"✗ Basic HNSW functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running HNSW structure verification tests...\n")
    
    tests = [
        test_hnsw_import,
        test_hybrid_hnsw_import,
        test_optimized_hnsw_import,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The HNSW structure is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
