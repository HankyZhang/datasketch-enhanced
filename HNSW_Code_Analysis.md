# HNSW Implementation Analysis - Datasketch Project

This document provides a comprehensive analysis of all HNSW (Hierarchical Navigable Small World) related code in the datasketch project.

## Overview

The datasketch project contains a complete HNSW implementation for approximate nearest neighbor search, with support for various distance functions including Euclidean, Jaccard, and MinHash-based similarities. The implementation consists of three main components:

1. **Core Implementation** (`datasketch/hnsw.py`)
2. **Benchmarking Code** (`benchmark/indexes/jaccard/hnsw.py`) 
3. **Unit Tests** (`test/test_hnsw.py`)

---

## 1. Core HNSW Implementation (`datasketch/hnsw.py`)

### Purpose
Implements the HNSW algorithm for approximate nearest neighbor search based on the paper "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Yu. A. Malkov, D. A. Yashunin (2016).

### Key Classes

#### Supporting Classes

**`_Layer`**
- Represents a graph layer in the HNSW index
- Maps keys to neighbor dictionaries: `{neighbor_key: distance}`
- Provides dictionary-like interface (`__getitem__`, `__setitem__`, etc.)
- Key methods:
  - `get_reverse_edges(key)`: Find all nodes pointing to a given key
  - `copy()`: Create deep copy of the layer

**`_LayerWithReversedEdges`**
- Enhanced version of `_Layer` that maintains reverse edge mappings
- Enables faster hard removal operations
- Automatically updates reverse edges when neighbors change
- Higher memory usage but better performance for deletions

**`_Node`**
- Represents a single node in the graph
- Contains: key (identifier), point (numpy array), is_deleted (soft deletion flag)
- Supports equality comparison and hashing

#### Main HNSW Class

**Constructor Parameters:**
```python
HNSW(
    distance_func,      # Function: (np.ndarray, np.ndarray) -> float
    m=16,              # Number of neighbors per node
    ef_construction=200, # Neighbors considered during construction
    m0=None,           # Neighbors for level 0 (default: 2*m)
    seed=None,         # Random seed
    reversed_edges=False # Whether to maintain reverse edges
)
```

### Core Methods

#### Insertion Operations
- **`insert(key, new_point, ef=None, level=None)`**: Add/update a point
- **`__setitem__(key, value)`**: Alias for insert using `index[key] = point`
- **`update(other)`**: Batch insert from mapping or another HNSW
- **`setdefault(key, default)`**: Insert if key doesn't exist

#### Query Operations
- **`query(query_point, k=None, ef=None)`**: Find k nearest neighbors
- **`__getitem__(key)`**: Retrieve point by key
- **`get(key, default=None)`**: Safe retrieval with default
- **`__contains__(key)`**: Check if key exists (and not soft-deleted)

#### Removal Operations
- **`remove(key, hard=False, ef=None)`**: Remove point (soft or hard)
- **`__delitem__(key)`**: Alias for soft remove using `del index[key]`
- **`pop(key, default=None, hard=False)`**: Remove and return point
- **`popitem(last=True, hard=False)`**: Remove and return arbitrary item
- **`clean(ef=None)`**: Hard remove all soft-deleted points
- **`clear()`**: Remove all points

#### Utility Operations
- **`copy()`**: Create deep copy of index
- **`merge(other)`**: Create new index by merging two indices
- **`__len__()`**: Number of non-deleted points
- **`keys()`, `values()`, `items()`**: Iterator methods

### Usage Examples

#### Euclidean Distance Example
```python
from datasketch.hnsw import HNSW
import numpy as np

# Create random data
data = np.random.random_sample((1000, 10))

# Create index
index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

# Insert points
for i, d in enumerate(data):
    index.insert(i, d)

# Query for 10 nearest neighbors
neighbors = index.query(data[0], k=10)
# Returns: [(key, distance), ...]
```

#### Jaccard Distance Example
```python
# Jaccard distance for sets represented as integer arrays
data = np.random.randint(0, 100, size=(1000, 10))

jaccard_distance = lambda x, y: (
    1.0 - float(len(np.intersect1d(x, y, assume_unique=False)))
    / float(len(np.union1d(x, y)))
)

index = HNSW(distance_func=jaccard_distance)
for i, d in enumerate(data):
    index[i] = d  # Alternative insertion syntax

neighbors = index.query(data[0], k=10)
```

### Internal Algorithm Details

#### Multi-level Structure
- Higher levels have exponentially fewer nodes
- Level assignment: `level = int(-log(random()) * level_mult)`
- Entry point maintained at highest level

#### Search Algorithms
- **`_search_ef1()`**: Greedy search for single closest neighbor (higher levels)
- **`_search_base_layer()`**: Beam search with ef candidates (level 0)
- **`_heuristic_prune()`**: Neighbor selection based on distance diversity

#### Deletion Strategies
- **Soft Delete**: Mark as deleted, keep in graph for traversal
- **Hard Delete**: Remove completely, repair neighbor connections
- Entry point reassignment when deleting current entry point

---

## 2. Benchmarking Implementation (`benchmark/indexes/jaccard/hnsw.py`)

### Purpose
Provides benchmarking functions to compare HNSW performance against other methods for Jaccard similarity search tasks.

### Key Functions

#### `search_nswlib_jaccard_topk(index_data, query_data, index_params, k)`
**Purpose**: Benchmark against nmslib's HNSW implementation
- Converts sets to space-separated strings for nmslib compatibility
- Uses `jaccard_sparse` space in nmslib
- Returns exact Jaccard similarities for fair comparison

**Parameters:**
- `index_data`: (sets, keys, _, cache) tuple
- `query_data`: (sets, keys, _) tuple  
- `index_params`: nmslib parameters (e.g., `{'efConstruction': 200}`)
- `k`: Number of neighbors to retrieve

#### `search_hnsw_jaccard_topk(index_data, query_data, index_params, k)`
**Purpose**: Benchmark datasketch HNSW with direct Jaccard distance
- Uses raw sets without conversion
- Leverages `compute_jaccard_distance` function
- Converts returned distances back to similarities

**Workflow:**
1. Build HNSW index with Jaccard distance function
2. Insert all index sets
3. Query with each query set
4. Convert distances to similarities: `similarity = 1.0 - distance`

#### `search_hnsw_minhash_jaccard_topk(index_data, query_data, index_params, k)`
**Purpose**: Benchmark HNSW with MinHash approximation
- Uses MinHash signatures instead of raw sets
- More efficient for large sets (dimensionality reduction)
- Computes exact Jaccard on retrieved candidates for accuracy

**Workflow:**
1. Generate MinHash signatures for all sets
2. Build HNSW index with MinHash distance function
3. Query using MinHash signatures
4. Retrieve original sets and compute exact Jaccard similarities
5. Re-rank by exact similarities

### Benchmarking Metrics
All functions return:
- **Indexing metrics**: Build time, preprocessing time
- **Query results**: (query_key, [(result_key, similarity), ...])
- **Query times**: Individual query durations for QPS calculation

---

## 3. Unit Tests (`test/test_hnsw.py`)

### Purpose
Comprehensive test suite validating HNSW functionality across different distance functions and usage patterns.

### Test Classes

#### `TestHNSW` - Basic L2 Distance Tests
**Distance Function**: `l2_distance(x, y) = np.linalg.norm(x - y)`

**Key Test Methods:**
- **`test_search()`**: Basic indexing and querying functionality
- **`test_upsert()`**: Adding new points to existing index
- **`test_update()`**: Batch updates via `update()` method
- **`test_merge()`**: Merging two separate indices
- **`test_pickle()`**: Serialization/deserialization
- **`test_copy()`**: Deep copying behavior and independence
- **`test_soft_remove_and_pop_and_clean()`**: Soft deletion workflow
- **`test_hard_remove_and_pop_and_clean()`**: Hard deletion workflow  
- **`test_popitem_last()`**: LIFO removal behavior
- **`test_popitem_first()`**: FIFO removal behavior
- **`test_clear()`**: Complete index clearing

#### `TestHNSWLayerWithReversedEdges`
**Purpose**: Tests same functionality with `reversed_edges=True`
- Inherits all tests from `TestHNSW`
- Validates that reverse edge optimization doesn't break functionality
- Ensures faster hard removal performance

#### `TestHNSWJaccard` 
**Distance Function**: `jaccard_distance(x, y) = 1.0 - |intersect(x,y)| / |union(x,y)|`
**Data Type**: Integer arrays representing sets

**Specializations:**
- Overrides `_create_random_points()` to generate integer arrays
- Custom search validation for Jaccard distance
- Tests discrete/categorical data handling

#### `TestHNSWMinHashJaccard`
**Distance Function**: `minhash_jaccard_distance(x, y) = 1.0 - x.jaccard(y)`
**Data Type**: MinHash objects

**Workflow:**
- Generates integer sets, converts to MinHash signatures
- Tests approximate similarity search
- Validates MinHash integration with HNSW

### Testing Utilities

#### `_create_random_points(n=100, dim=10)`
- Generates test data appropriate for each test class
- L2: `np.random.rand(n, dim)`
- Jaccard: `np.random.randint(0, high, (n, dim))`
- MinHash: `MinHash.bulk(sets, num_perm=128)`

#### `_insert_points(index, points, keys=None)`
- Tests both insertion methods: `insert()` and `[]` assignment
- Validates entry point setup, containment, retrieval
- Checks ordering preservation and length updates

#### `_search_index(index, queries, k=10)`
- Validates search results are properly distance-ordered
- Ensures graph connectivity (can find enough neighbors)
- Tests search functionality across different distance functions

### Test Execution
```bash
# Run all HNSW tests
python -m pytest test/test_hnsw.py -v

# Run specific test class
python -m pytest test/test_hnsw.py::TestHNSWJaccard -v

# Run with coverage
python -m pytest test/test_hnsw.py --cov=datasketch.hnsw
```

---

## Algorithm Characteristics

### Performance Characteristics
- **Time Complexity**: 
  - Insert: O(log N * M) average case
  - Query: O(log N * ef) average case  
  - Space: O(N * M) where M is average degree

### Tuning Parameters
- **`m`**: Higher values → better recall, slower insertion, more memory
- **`ef_construction`**: Higher values → better index quality, slower construction
- **`ef` (query time)**: Higher values → better recall, slower queries

### Distance Function Requirements
- Must be symmetric: `d(x,y) = d(y,x)`
- Must satisfy triangle inequality for optimal performance
- Should return 0 for identical points

### Suitable Use Cases
- **High-dimensional vectors** (embeddings, features)
- **Large datasets** (millions+ points) where exact search is too slow
- **Real-time applications** requiring fast approximate nearest neighbor search
- **Custom similarity metrics** beyond standard Euclidean distance

---

## Integration Examples

### Using with Different Data Types

#### Text Embeddings
```python
from sentence_transformers import SentenceTransformer
from datasketch.hnsw import HNSW
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["sample text 1", "sample text 2", ...]
embeddings = model.encode(texts)

index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
for i, emb in enumerate(embeddings):
    index[i] = emb

# Find similar texts
query_embedding = model.encode(["query text"])
similar = index.query(query_embedding[0], k=5)
```

#### Set Similarity with MinHash
```python
from datasketch import MinHash
from datasketch.hnsw import HNSW

def create_minhash_from_set(s, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for item in s:
        m.update(str(item).encode('utf8'))
    return m

sets = [set(np.random.randint(0, 1000, 50)) for _ in range(10000)]
minhashes = [create_minhash_from_set(s) for s in sets]

index = HNSW(distance_func=lambda x, y: 1.0 - x.jaccard(y))
for i, mh in enumerate(minhashes):
    index[i] = mh

# Find similar sets
query_minhash = create_minhash_from_set(sets[0])
similar = index.query(query_minhash, k=10)
```

---

## Performance Considerations

### Memory Usage
- **Standard layers**: ~16-32 bytes per edge
- **Reversed edges**: Additional ~8 bytes per edge for reverse mapping
- **Node storage**: ~24 bytes + point size per node

### Optimization Tips
1. **Batch insertions** are more efficient than individual inserts
2. **Pre-size data structures** when final size is known
3. **Use appropriate `m` values**: 16-48 for most applications
4. **Tune `ef_construction`**: 200-800 depending on recall requirements
5. **Consider soft vs hard deletion** based on update patterns

### Threading Considerations
- **Not thread-safe** for concurrent modifications
- **Read-only queries** can be parallelized safely
- **Use separate indices** per thread for concurrent updates

This documentation provides a complete overview of the HNSW implementation in the datasketch project, covering both usage patterns and internal implementation details.
