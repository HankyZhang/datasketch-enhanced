# SIFT Dataset Testing Guide

This document explains how to use the SIFT1M dataset to test and benchmark the enhanced HNSW implementations.

## Dataset Overview

The SIFT1M dataset is a standard benchmark in the similarity search community:
- **Base vectors**: 1,000,000 SIFT descriptors (128 dimensions)
- **Query vectors**: 10,000 test queries
- **Ground truth**: 100 nearest neighbors for each query
- **Format**: Binary `.fvecs` (float vectors) and `.ivecs` (integer vectors)

## Quick Setup

1. **Download SIFT dataset** (if not already available):
   ```bash
   # The dataset should be in a 'sift' directory with these files:
   # sift/sift_base.fvecs      (1M base vectors)
   # sift/sift_query.fvecs     (10K query vectors) 
   # sift/sift_groundtruth.ivecs (ground truth)
   ```

2. **Run quick test**:
   ```bash
   python quick_sift_test.py
   ```

3. **Run comprehensive benchmark**:
   ```bash
   python sift_benchmark.py --size small    # 5K vectors, fast
   python sift_benchmark.py --size medium   # 50K vectors
   python sift_benchmark.py --size large    # 200K vectors
   python sift_benchmark.py --size full     # Full 1M dataset
   ```

## Understanding the Results

### Performance Metrics

- **Build Time**: Time to construct the index
- **Query Time**: Average time per query (milliseconds)
- **Recall@k**: Fraction of true k-nearest neighbors found
- **QPS**: Queries per second

### Typical Performance Characteristics

**Standard HNSW**:
- High recall (>95% @ recall@10 with proper parameters)
- Moderate build time
- Good query performance
- Parameter: `ef` controls speed/quality tradeoff

**Hybrid HNSW**:
- Much faster queries (5-50x speedup)
- Comparable build time
- Slightly lower recall but still good (>85% @ recall@10)
- Parameters: `n_probe` controls speed/quality tradeoff

### Sample Output

```
SIFT Benchmark - SMALL Dataset
==================================================

Dataset: 5000 vectors, 128 dimensions

Standard HNSW (ef=200):
  Build time: 2.45s
  Query time: 3.20ms
  Recall@1:  0.8800
  Recall@10: 0.9420

Hybrid HNSW (n_probe=10):
  Build time: 2.78s
  Query time: 0.15ms
  Recall@1:  0.8200
  Recall@10: 0.8950

🏃 Query speedup: 21.33x (Hybrid vs Standard)
```

## Parameter Tuning

### Standard HNSW Parameters

- **m**: Connectivity parameter (default: 16)
  - Higher values = better recall, slower build
  - Typical range: 8-64

- **ef_construction**: Search depth during construction (default: 200)
  - Higher values = better recall, slower build
  - Typical range: 100-800

- **ef**: Search depth during query (tunable at runtime)
  - Higher values = better recall, slower queries
  - Typical range: 50-1000

### Hybrid HNSW Parameters

- **parent_level**: Which HNSW level to use as parents (default: 2)
  - Higher values = fewer parents, faster queries
  - Typical range: 1-3

- **k_children**: Max children per parent (default: 1000)
  - Higher values = better recall, more memory
  - Typical range: 500-5000

- **n_probe**: Number of parent cells to search (tunable at runtime)
  - Higher values = better recall, slower queries
  - Typical range: 1-50

## Advanced Usage

### Custom Distance Functions

```python
# L2 distance (default)
distance_func = lambda x, y: np.linalg.norm(x - y)

# Cosine similarity
distance_func = lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Manhattan distance
distance_func = lambda x, y: np.sum(np.abs(x - y))
```

### Memory Usage Optimization

```python
# For large datasets, use iterative loading
def load_vectors_chunked(filename, chunk_size=10000):
    chunks = []
    for chunk in read_fvecs_chunked(filename, chunk_size):
        chunks.append(chunk)
    return np.vstack(chunks)
```

### Result Analysis

The benchmark saves detailed results in JSON format:

```python
import json

# Load results
with open('sift_benchmark_small.json', 'r') as f:
    results = json.load(f)

# Analyze recall vs query time tradeoffs
standard_results = results['standard_hnsw']['results_by_ef']
hybrid_results = results['hybrid_hnsw']['results_by_probe']

# Plot performance curves
import matplotlib.pyplot as plt

ef_values = list(standard_results.keys())
query_times = [standard_results[ef]['query_time_ms'] for ef in ef_values]
recalls = [standard_results[ef]['recalls'][10] for ef in ef_values]

plt.plot(query_times, recalls, 'o-', label='Standard HNSW')
plt.xlabel('Query Time (ms)')
plt.ylabel('Recall@10')
plt.legend()
plt.show()
```

## Troubleshooting

### Common Issues

1. **Low Recall Values**:
   - Increase `ef_construction` for standard HNSW
   - Increase `k_children` for hybrid HNSW
   - Use higher `ef` or `n_probe` during search

2. **Slow Queries**:
   - Decrease `ef` for standard HNSW
   - Decrease `n_probe` for hybrid HNSW
   - Consider using smaller dataset size for testing

3. **Memory Issues**:
   - Use smaller dataset sizes
   - Reduce `k_children` parameter
   - Enable chunked loading for large datasets

4. **File Format Errors**:
   - Ensure SIFT files are in correct binary format
   - Check file paths and permissions
   - Verify dataset integrity

### Performance Expectations

| Dataset Size | Standard Build | Hybrid Build | Standard Query | Hybrid Query |
|-------------|----------------|--------------|----------------|--------------|
| 5K          | 2-5s          | 3-6s         | 1-10ms        | 0.1-2ms     |
| 50K         | 20-60s        | 25-70s       | 5-30ms        | 0.5-5ms     |
| 200K        | 2-5min        | 3-6min       | 10-50ms       | 1-10ms      |
| 1M          | 10-20min      | 12-25min     | 20-100ms      | 2-20ms      |

These are approximate values and will vary based on hardware and parameters.

## Best Practices

1. **Start Small**: Always test with small datasets first
2. **Parameter Sweep**: Try different parameter combinations
3. **Monitor Memory**: Large datasets can consume significant RAM
4. **Save Results**: Use JSON output for detailed analysis
5. **Compare Fairly**: Use same parameters when comparing algorithms
6. **Profile Code**: Use timing decorators for detailed performance analysis

## Integration with Your Data

To use your own dataset instead of SIFT:

```python
# Convert your data to the right format
your_base_vectors = np.array(your_data, dtype=np.float32)
your_query_vectors = np.array(your_queries, dtype=np.float32)

# Run the same benchmark
benchmark = SIFTBenchmark("custom")
benchmark.base_vectors = your_base_vectors
benchmark.query_vectors = your_query_vectors
benchmark.ground_truth = compute_ground_truth(your_base_vectors, your_query_vectors)

standard_results = benchmark.benchmark_standard_hnsw()
hybrid_results = benchmark.benchmark_hybrid_hnsw()
```
