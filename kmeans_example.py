"""Simple example using sklearn MiniBatchKMeans with (optional) SIFT data.

Legacy local KMeans implementation has been removed; this example now demonstrates
usage of `MiniBatchKMeans` for clustering large vector datasets.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.cluster import MiniBatchKMeans

# Optional legacy SIFT loader replacement (minimal) – expects sift/*.fvecs if present
def load_sift_data_or_synthetic(dim: int = 128):
    sift_dir = os.path.join(os.path.dirname(__file__), 'sift')
    base_path = os.path.join(sift_dir, 'sift_learn.fvecs')
    if os.path.exists(base_path):
        try:
            raw = np.fromfile(base_path, dtype=np.int32)
            d = raw[0]
            rec = d + 1
            count = raw.size // rec
            raw = raw.reshape(count, rec)
            learn_vectors = raw[:, 1:].astype(np.float32)
            # Provide placeholders for compatibility
            return None, learn_vectors, learn_vectors[:1000], None
        except Exception:
            pass
    # Fallback synthetic
    learn_vectors = np.random.randn(10000, dim).astype(np.float32)
    return None, learn_vectors, learn_vectors[:1000], None


def simple_example():
    """Simple example demonstrating K-means usage."""
    print("🎯 Simple MiniBatchKMeans Example")
    print("=" * 50)
    
    try:
        # Load SIFT data or synthetic
        print("Loading SIFT (or synthetic) dataset...")
        base_vectors, learn_vectors, query_vectors, groundtruth = load_sift_data_or_synthetic()

        # Use a subset of learn vectors for this example
        X = learn_vectors[:2000]  # Use 2000 samples
        print(f"Using {X.shape[0]} samples with {X.shape[1]} features")

        # Create MiniBatchKMeans model
        print("\nCreating MiniBatchKMeans model with k=20...")
        kmeans = MiniBatchKMeans(
            n_clusters=20,
            max_iter=200,
            n_init=3,
            batch_size=512,
            random_state=42,
            verbose=0
        )

        # Fit the model
        print("\nFitting MiniBatchKMeans...")
        kmeans.fit(X)

        # Get results
        print(f"\nResults:")
        print(f"  Final inertia: {kmeans.inertia_:.2f}")
        print(f"  Iterations: {kmeans.n_iter_}")
        print(f"  Cluster centers shape: {kmeans.cluster_centers_.shape}")

        # Basic cluster size stats
        labels = getattr(kmeans, 'labels_', None)
        if labels is not None:
            counts = np.bincount(labels, minlength=20)
            print(f"\nCluster distribution:")
            print(f"  Average cluster size: {counts.mean():.1f}")
            print(f"  Largest cluster: {counts.max()}")
            print(f"  Smallest cluster: {counts.min()}")

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
