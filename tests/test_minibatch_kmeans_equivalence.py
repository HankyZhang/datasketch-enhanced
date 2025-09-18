import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

def test_minibatch_kmeans_inertia_close():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=0.0, scale=0.5, size=(200, 16)),
        rng.normal(loc=5.0, scale=0.5, size=(200, 16)),
        rng.normal(loc=-4.0, scale=0.5, size=(200, 16)),
    ]).astype(np.float32)

    full = KMeans(n_clusters=3, n_init=5, max_iter=200, random_state=42)
    mb = MiniBatchKMeans(n_clusters=3, n_init=3, max_iter=200, batch_size=128, random_state=42)

    full.fit(X)
    mb.fit(X)

    # Allow small relative difference
    rel_diff = (mb.inertia_ - full.inertia_) / full.inertia_
    # Assert difference within 5%
    assert rel_diff < 0.05, f"MiniBatchKMeans inertia too high relative to full KMeans: rel_diff={rel_diff:.3f}"
