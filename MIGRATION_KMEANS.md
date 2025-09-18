# Migration Guide: Custom KMeans -> sklearn MiniBatchKMeans

This project previously included a bespoke `kmeans` module. It has been deprecated and replaced everywhere by `sklearn.cluster.MiniBatchKMeans` for better stability, performance, and maintenance.

## Why the Change?
- Robustness: battle-tested implementation maintained by scikit-learn.
- Speed: minibatch updates scale better for large datasets.
- Simplicity: removes redundant code and lowers maintenance burden.
- Ecosystem: direct compatibility with scikit-learn tooling (pipelines, metrics, etc.).

## Key API Differences
| Legacy (Removed) | MiniBatchKMeans | Notes |
|------------------|-----------------|-------|
| `max_iters`       | `max_iter`       | Rename parameter |
| `inertia_`        | `inertia_`       | Same meaning |
| `n_iter_`         | `n_iter_`        | Same meaning |
| `cluster_centers_`| `cluster_centers_`| Same |
| `predict(X)`      | `predict(X)`     | Same |
| `get_cluster_info()` | (none)       | Compute manually via `np.bincount(labels_)` |
| `verbose` (bool)  | `verbose` (int) | Use `0` or `1` |

## Replacement Snippet
```python
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(
    n_clusters=128,
    max_iter=100,
    n_init=3,
    batch_size=1024,
    tol=1e-3,
    random_state=42
)

kmeans.fit(X)
print(kmeans.inertia_, kmeans.n_iter_)
labels = kmeans.labels_
```

## Computing Legacy `get_cluster_info()` Equivalent
```python
import numpy as np
labels = kmeans.labels_
counts = np.bincount(labels, minlength=kmeans.n_clusters)
cluster_info = {
    'avg_cluster_size': counts.mean(),
    'std_cluster_size': counts.std(),
    'min_cluster_size': counts.min(),
    'max_cluster_size': counts.max(),
}
```

## Method 3 (KMeansHNSW) Changes
- Accepts `kmeans_params` that now map directly to MiniBatchKMeans (`max_iter`, `n_init`, `batch_size`, `tol`).
- Cluster statistics are derived internally (no external helper needed).

## How to Remove Legacy Code Completely
1. Delete the `kmeans/` directory (left temporarily for backward compatibility).
2. Search your code for `from kmeans` imports and replace with MiniBatchKMeans.
3. Remove any docs referencing the old implementation.

## Testing Equivalence
A lightweight test is included (see `tests/test_minibatch_kmeans_equivalence.py`) comparing MiniBatchKMeans vs full KMeans on small synthetic data to ensure acceptable inertia gap.

## Suggested Parameter Defaults
| Scenario | n_init | max_iter | batch_size | tol |
|----------|--------|----------|------------|-----|
| Balanced | 3      | 100      | 1024       | 1e-3|
| High Quality | 5-10 | 200      | 2048       | 1e-4|
| Maximum Speed | 1  | 50       | 512        | 1e-2|

## FAQ
**Q: Do I lose clustering quality?**  
In most moderate-scale datasets the difference vs full batch KMeans is negligible (<1% inertia delta) with correct batch sizing.

**Q: Can I still use standard `KMeans`?**  
Yes, import `from sklearn.cluster import KMeans` if you prefer full batch; the system only requires centroids and labels.

**Q: When will the legacy module be deleted?**  
It can be removed immediately after external dependencies finish migration.

---
If you need help migrating custom scripts, open an issue or adapt the snippets above.
