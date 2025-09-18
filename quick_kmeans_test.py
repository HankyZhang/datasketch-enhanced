#!/usr/bin/env python3
"""
Quick K-means speed verification
"""

import numpy as np
import time

# Add paths
import sys
import os
sys.path.append('.')

def quick_test():
    print("K-means Speed Optimization Test")
    print("=" * 40)
    
    # Test data
    np.random.seed(42)
    X = np.random.randn(2000, 64).astype(np.float32)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    try:
        from kmeans.kmeans import KMeans
        
        print("Testing optimized K-means...")
        start = time.time()
        
        kmeans = KMeans(n_clusters=10, max_iters=50, verbose=True, random_state=42)
        kmeans.fit(X)
        
        elapsed = time.time() - start
        
        print(f"✅ Completed in {elapsed:.2f} seconds")
        print(f"   Inertia: {kmeans.inertia_:.2f}")
        print(f"   Iterations: {kmeans.n_iter_}")
        print(f"   Speed: {X.shape[0]/elapsed:.0f} samples/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 K-means optimizations working!")
    else:
        print("\n❌ Test failed")
