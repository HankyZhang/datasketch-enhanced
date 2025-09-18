"""Deprecated legacy utilities for removed KMeans implementation.

This module no longer provides any functions. Use scikit-learn metrics and
custom data loaders. See MIGRATION_KMEANS.md for guidance.
"""

raise ImportError(
    "'kmeans.utils' removed. Use sklearn + custom loaders. See MIGRATION_KMEANS.md."
)
