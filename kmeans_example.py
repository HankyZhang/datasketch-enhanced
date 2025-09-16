"""
Simple example of using K-means clustering with SIFT data.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kmeans import KMeans, load_sift_data


def simple_example():
    """Simple example demonstrating K-means usage."""
    print("🎯 Simple K-means Example with SIFT Data")
    print("=" * 50)
    
    try:
        # Load SIFT data
        print("Loading SIFT dataset...")
        base_vectors, learn_vectors, query_vectors, groundtruth = load_sift_data()
        
        # Use a subset of learn vectors for this example
        X = learn_vectors[:2000]  # Use 2000 samples
        print(f"Using {X.shape[0]} samples with {X.shape[1]} features")
        
        # Create K-means model
        print("\nCreating K-means model with k=20...")
        kmeans = KMeans(
            n_clusters=20,
            max_iters=200,
            n_init=3,
            random_state=42,
            verbose=True
        )
        
        # Fit the model
        print("\nFitting K-means...")
        kmeans.fit(X)
        
        # Get results
        print(f"\nResults:")
        print(f"  Final inertia: {kmeans.inertia_:.2f}")
        print(f"  Iterations: {kmeans.n_iter_}")
        print(f"  Cluster centers shape: {kmeans.cluster_centers_.shape}")
        
        # Get cluster information
        cluster_info = kmeans.get_cluster_info()
        print(f"\nCluster distribution:")
        print(f"  Average cluster size: {cluster_info['avg_cluster_size']:.1f}")
        print(f"  Largest cluster: {cluster_info['max_cluster_size']}")
        print(f"  Smallest cluster: {cluster_info['min_cluster_size']}")
        
        # Test prediction on new data
        print(f"\nTesting prediction on query vectors...")
        query_subset = query_vectors[:100]  # Use 100 query vectors
        predicted_labels = kmeans.predict(query_subset)
        
        print(f"Predicted {len(predicted_labels)} labels")
        print(f"Label distribution: {np.bincount(predicted_labels)}")
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    simple_example()
