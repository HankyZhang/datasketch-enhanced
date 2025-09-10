HNSW Algorithm
==============

The HNSW (Hierarchical Navigable Small World) algorithm is a fast and accurate approximate nearest neighbor search algorithm.

.. automodule:: datasketch.hnsw
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
------------

- **High Performance**: O(log N) search complexity
- **Dynamic Updates**: Support for insertion, deletion, and updates
- **High Accuracy**: Configurable parameters for 95%+ recall
- **Scalable**: Handles millions of data points in real-time

Algorithm Parameters
-------------------

.. py:class:: HNSW(distance_func, m=16, ef_construction=200, m0=None, seed=None, reversed_edges=False)

   :param distance_func: Function to compute distance between two vectors
   :param m: Maximum number of neighbors per node (default: 16)
   :param ef_construction: Number of candidates during construction (default: 200)
   :param m0: Maximum neighbors for layer 0 (default: 2*m)
   :param seed: Random seed for reproducible results
   :param reversed_edges: Whether to maintain reverse edges for faster deletion

Core Methods
------------

.. py:method:: insert(key, point, ef=None, level=None)

   Insert a new point into the index.

.. py:method:: query(query_point, k=None, ef=None)

   Search for k nearest neighbors.

.. py:method:: remove(key, hard=False, ef=None)

   Remove a point using soft or hard deletion.

.. py:method:: update(mapping)

   Batch insert multiple points.

Examples
--------

Basic Usage::

    from datasketch import HNSW
    import numpy as np

    # Create index
    index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

    # Insert data
    data = np.random.random((1000, 50))
    index.update({i: vec for i, vec in enumerate(data)})

    # Query
    query = np.random.random(50)
    neighbors = index.query(query, k=10)

Parameter Tuning::

    # High precision configuration
    precise_index = HNSW(
        distance_func=euclidean_distance,
        m=32,
        ef_construction=400
    )

    # Fast configuration
    fast_index = HNSW(
        distance_func=euclidean_distance,
        m=8,
        ef_construction=100
    )

For more detailed examples and Chinese documentation, see the examples directory and markdown documentation files.
