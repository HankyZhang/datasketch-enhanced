HNSW Enhanced Documentation
===========================

Welcome to HNSW Enhanced documentation!

This package provides a high-performance implementation of the HNSW (Hierarchical Navigable Small World) algorithm with comprehensive Chinese documentation and comments.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   hnsw

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install numpy
   git clone https://github.com/HankyZhang/datasketch-enhanced.git
   cd datasketch-enhanced
   pip install -e .

Basic usage:

.. code-block:: python

   from datasketch import HNSW
   import numpy as np

   # Create HNSW index
   index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

   # Add data
   data = np.random.random((1000, 10))
   index.update({i: vec for i, vec in enumerate(data)})

   # Search nearest neighbors
   query = np.random.random(10)
   neighbors = index.query(query, k=10)

Chinese Documentation
--------------------

For detailed Chinese documentation, see:

- `HNSW算法原理详解.md <../HNSW算法原理详解.md>`_
- `HNSW_代码分析_中文版.md <../HNSW_代码分析_中文版.md>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`