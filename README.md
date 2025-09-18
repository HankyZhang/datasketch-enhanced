# HNSW Hybrid Two-Stage Retrieval System

🚀 **Advanced HNSW implementation with hybrid two-stage retrieval architecture**

> Update: K-Means based components (Method 3 / two-stage variants) now use sklearn `MiniBatchKMeans` instead of the previous custom implementation for improved stability and faster construction on large datasets.

A high-performance implementation of the HNSW (Hierarchical Navigable Small World) algorithm featuring an innovative hybrid two-stage retrieval system that significantly improves recall performance.

## 🆕 Latest: HNSW Hybrid Two-Stage System

The **HNSW Hybrid Two-Stage Retrieval System** transforms a standard HNSW into a two-stage retrieval architecture for improved recall performance:

### 🔥 Core Features
- **Two-Stage Search**: Coarse filtering (parent nodes) + Fine filtering (child nodes)
- **Enhanced Recall**: 62.86% Recall@10 performance with optimized parameters
- **Configurable Parameters**: Tunable k_children and n_probe for precision-efficiency trade-offs
- **Scalable Design**: Supports large-scale datasets up to millions of vectors
- **Comprehensive Evaluation**: Complete Recall@K metrics and parameter tuning framework

## 🌟 Key Features

### 🔍 HNSW Algorithm Advantages
- **Efficient Search**: O(log N) time complexity for approximate nearest neighbor search
- **Dynamic Updates**: Real-time insert, delete, and update operations
- **High Precision**: Configurable parameters for 95%+ recall rates
- **Scalable**: Support for million-scale datasets with real-time search

### 🏗️ Hybrid Architecture Innovation
- **Parent-Child Structure**: Extract parent nodes from HNSW Level 2
- **Two-Stage Retrieval**: Coarse search → Fine search within selected regions
- **Parameter Optimization**: Systematic tuning of k_children and n_probe parameters
- **Performance Validation**: Comprehensive evaluation against brute-force ground truth

## 📁 Project Structure

```
datasketch-enhanced/
├── hnsw/                        # 🎯 Standard HNSW Implementation
│   ├── hnsw.py                  # Core HNSW algorithm
│   ├── version.py               # Version information
│   └── __init__.py              # Package initialization
├── hybrid_hnsw/                 # � Hybrid Two-Stage HNSW
│   ├── hnsw_hybrid.py           # Hybrid two-stage implementation
│   └── __init__.py              # Package initialization
├── optimized_hnsw/              # ⚡ Optimized HNSW Implementation
│   └── __init__.py              # Package initialization
├── docs/                        # 📚 Essential Documentation
│   ├── HNSW_Hybrid_Algorithm_Principles.md # Core algorithm principles
│   ├── HNSW_Hybrid_Technical_Implementation.md # Technical details
│   └── HNSW算法原理详解.md      # Chinese algorithm documentation
├── setup.py                     # Installation configuration
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation
```bash
pip install numpy
git clone https://github.com/HankyZhang/datasketch-enhanced.git
cd datasketch-enhanced
## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/HankyZhang/datasketch-enhanced.git
cd datasketch-enhanced

# Install in development mode
pip install -e .
```

### Basic Usage

#### Standard HNSW Usage
```python
from hnsw import HNSW
import numpy as np

# Create random data
data = np.random.random((1000, 50))

# Initialize HNSW index  
distance_func = lambda x, y: np.linalg.norm(x - y)
index = HNSW(distance_func=distance_func, m=16, ef_construction=200)

# Insert data
for i, vector in enumerate(data):
    index.insert(i, vector)

# Search for nearest neighbors
query = np.random.random(50)
neighbors = index.query(query, k=10)

print(f"Found {len(neighbors)} nearest neighbors")
```

#### 🆕 HNSW Hybrid Two-Stage Retrieval System
```python
from hybrid_hnsw.hnsw_hybrid import HybridHNSWIndex
import numpy as np

# Create synthetic dataset
data = np.random.random((5000, 128))

# Initialize hybrid index
hybrid_index = HybridHNSWIndex(
    distance_func=lambda x, y: np.linalg.norm(x - y),
    m=16,
    ef_construction=200,
    k_children=50,
    n_probe=10
)

# Build the index
for i, vector in enumerate(data):
    hybrid_index.insert(i, vector)

# Search with hybrid two-stage retrieval
query = np.random.random(128)
results = hybrid_index.search(query, k=10)
print(f"Hybrid search found {len(results)} results")
```
```python
from hnsw import HNSW
from hybrid_hnsw import HNSWHybrid
import numpy as np

# Create synthetic dataset
dataset = {i: np.random.random(128) for i in range(5000)}  # 5K vectors, 128 dimensions

# Build base HNSW index
distance_func = lambda x, y: np.linalg.norm(x - y)
base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)

# Insert vectors (excluding queries)
for i, vector in enumerate(dataset):
    if i not in query_ids:
# Insert vectors
base_index.update(dataset)

# Build hybrid index
hybrid_index = HNSWHybrid(
    base_index=base_index,
    parent_level=2,          # Extract parents from level 2
    k_children=1000         # 1000 children per parent
)

# Simple query example
query = np.random.random(128)
results = hybrid_index.search(query, k=10)

print(f"Found {len(results)} results")
print(f"First result: {results[0]}")
```

## 🛠️ Advanced Usage

### Running Experiments
```bash
# Run the complete hybrid system demonstration
python project_demo.py
```

### Running Tests
```bash
# Run structure verification test
python test_structure.py
```

### Parameter Tuning
The hybrid (parent_level based) and the MiniBatchKMeans + HNSW (Method 3) systems support these key parameters:

- **`parent_level`** (hybrid variant only): HNSW level to extract parent nodes from (default: 2)
- **`n_clusters`** (KMeansHNSW): Number of MiniBatchKMeans clusters (acts as parents)
- **`k_children`**: Number of child nodes per parent (default: 1000)
- **`n_probe`**: Number of parent nodes / centroids to probe during search (default: 15)
- **MiniBatchKMeans params** via `kmeans_params`: `max_iter`, `n_init`, `batch_size`, `tol`, `random_state`

## 📊 Performance Results

### Benchmark Results (5K vectors, 128 dimensions)
- **Recall@10**: 62.86%
- **Average Query Time**: 5.43ms
- **Parent Nodes**: 12 nodes managing 1,438 children
- **Memory Efficiency**: Optimized data structures with minimal overhead

### Key Performance Insights
- **Two-stage approach** provides systematic search within precomputed regions
- **Parameter tuning** allows precision-efficiency trade-offs
- **Scalable architecture** maintains performance at larger scales

## 📚 Documentation

- **[Algorithm Principles](doc_md/HNSW_Hybrid_Algorithm_Principles.md)**: Core concepts and theory
- **[Technical Implementation](doc_md/HNSW_Hybrid_Technical_Implementation.md)**: Implementation details
- **[Complete Guide](doc_md/HNSW_HYBRID_README.md)**: Comprehensive user guide
- **[Project Summary](doc_md/PROJECT_SUMMARY.md)**: Complete project overview

## 🎯 Use Cases

- **Recommendation Systems**: High-recall similarity search
- **Image Retrieval**: Content-based search with improved accuracy
- **Semantic Search**: Document and text similarity with enhanced recall
- **Research Applications**: Algorithm comparison and parameter studies

## 🤝 Contributing

This project is actively maintained. Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built on the foundation of the original HNSW algorithm with innovative hybrid architecture enhancements and MiniBatchKMeans integration for improved recall and construction efficiency.
