# HNSW Hybrid Two-Stage Retrieval System

🚀 Advanced HNSW implementation with a level-based (HNSW parents) and K-Means based two-stage retrieval architecture.

Update highlights:
- Method 3 (K-Means + HNSW) now uses sklearn `MiniBatchKMeans` for stability & speed.
- Unified public hybrid class name: `HNSWHybrid` (old placeholder `HybridHNSWIndex` removed).
- Added optional diversification (`diversify_max_assignments`) and repair (`repair_min_assignments` / programmatic `run_repair`) mechanisms for parent→child coverage control.

## 🆕 Latest: HNSW Hybrid Two-Stage System

The **HNSW Hybrid Two-Stage Retrieval System** transforms a standard HNSW into a two-stage retrieval architecture for improved recall performance:

### 🔥 Core Features
- Two-stage search: coarse (parents) → fine (children)
- Configurable: `k_children`, `n_probe`, `approx_ef`, diversification & repair
- Coverage diagnostics: `coverage_fraction`, `raw_child_coverage`, assignment stats
- Scales to large datasets; evaluation utilities included

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
├── hybrid_hnsw/                 # 🔀 Level-based Hybrid Two-Stage HNSW
│   ├── hnsw_hybrid.py           # `HNSWHybrid` implementation
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
git clone https://github.com/HankyZhang/datasketch-enhanced.git
cd datasketch-enhanced
pip install -e .
```

### Standard HNSW Example
```python
from hnsw.hnsw import HNSW
import numpy as np

data = np.random.random((1000, 64)).astype(np.float32)
dist = lambda a, b: np.linalg.norm(a - b)
index = HNSW(distance_func=dist, m=16, ef_construction=200)
for i, v in enumerate(data):
    index.insert(i, v)
q = np.random.random(64).astype(np.float32)
print(index.query(q, k=10))
```

### Level-Based Hybrid (Parents from HNSW Level)
```python
from hnsw.hnsw import HNSW
from hybrid_hnsw import HNSWHybrid
import numpy as np

vectors = np.random.random((5000, 128)).astype(np.float32)
dist = lambda a, b: np.linalg.norm(a - b)
base = HNSW(distance_func=dist, m=16, ef_construction=200)
for i, v in enumerate(vectors):
    base.insert(i, v)

hybrid = HNSWHybrid(
    base_index=base,
    parent_level=2,
    k_children=800,
    approx_ef=300,
    diversify_max_assignments=None,
    repair_min_assignments=None
)

query = np.random.random(128).astype(np.float32)
results = hybrid.search(query, k=10, n_probe=12)
print('Top IDs:', [r[0] for r in results])
print('Coverage:', hybrid.get_stats().get('coverage_fraction'))

# (Optional) enforce coverage post-build
hybrid.run_repair(2)
print('Coverage after repair:', hybrid.get_stats().get('coverage_fraction'))
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

### Parameter Tuning (Key Parameters)
| Parameter | Applies To | Purpose |
|-----------|-----------|---------|
| `parent_level` | `HNSWHybrid` | HNSW level used as parent set |
| `n_clusters` | `KMeansHNSW` | Number of K-Means centroids (parents) |
| `k_children` | Both | Target children per parent (before diversification/repair) |
| `n_probe` | Both | Parents probed at query time |
| `approx_ef` | `HNSWHybrid` | Width of approximate child gathering |
| `child_search_ef` | `KMeansHNSW` | HNSW ef used when filling children per centroid |
| `diversify_max_assignments` | Both | Upper bound per child (load balancing) |
| `repair_min_assignments` | Both | Minimum assignments per child during build |

Repair notes:
- Setting `repair_min_assignments ≥ 2` during build generally lifts `coverage_fraction` close to 1.0.
- If omitted, you can still call `run_repair(min_assignments)` later (programmatic only) for both `HNSWHybrid` and `KMeansHNSW`.
- `coverage_fraction` counts parents + unique children; `raw_child_coverage` counts only child set.

## 📊 Performance Results

### Example Small-Scale Metrics (Illustrative)
- Recall@10 mid-range (depends on config and dataset)
- Query latency driven by `n_probe * k_children` candidate size
- Diversification smooths parent load; repair ensures coverage

Tune `k_children`, `n_probe`, and ef parameters jointly for best trade-off.

## 📚 Documentation

Core docs located in `docs/` (Chinese & English mixed):
- `HNSW_Hybrid_Algorithm_Principles.md`
- `HNSW_Hybrid_Technical_Implementation.md`

Method 3 (K-Means + HNSW) details in `method3/README.md`.

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

Built on the original HNSW algorithm with hybrid parenting strategies and MiniBatchKMeans integration for flexible two-stage recall optimization.
