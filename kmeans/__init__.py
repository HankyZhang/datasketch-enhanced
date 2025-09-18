"""Deprecated legacy 'kmeans' package.

All functionality has migrated to scikit-learn's MiniBatchKMeans.
Import this package is now unsupported.
"""

raise ImportError(
	"The 'kmeans' package has been removed. Use 'sklearn.cluster.MiniBatchKMeans'. "
	"Refer to MIGRATION_KMEANS.md for guidance."
)
