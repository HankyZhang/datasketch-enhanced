# K-means Clustering for SIFT Dataset

This is a robust K-means clustering implementation optimized for high-dimensional vector data like SIFT features.

## Features

- **K-means++ initialization** for better initial centroids
- **Multiple initialization attempts** to find global optimum
- **Early stopping** when convergence is reached
- **Comprehensive evaluation metrics** including silhouette score
- **Memory-efficient implementation** for large datasets
- **Automatic optimal k detection** using elbow method
- **SIFT dataset support** with built-in data loaders

## Installation

No additional installation required beyond Python and standard libraries. The implementation uses:
- NumPy for numerical computations
- scikit-learn for evaluation metrics (optional)

## Quick Start

### Simple Example
```python
from kmeans import KMeans, load_sift_data

# Load SIFT data
base_vectors, learn_vectors, query_vectors, groundtruth = load_sift_data()

# Use a subset for quick testing
X = learn_vectors[:2000]

# Create and fit K-means model
kmeans = KMeans(n_clusters=20, random_state=42, verbose=True)
kmeans.fit(X)

# Get results
print(f"Inertia: {kmeans.inertia_}")
print(f"Iterations: {kmeans.n_iter_}")

# Predict on new data
labels = kmeans.predict(query_vectors[:100])
```

### Comprehensive Testing
```python
# Run comprehensive test with multiple k values
python test_kmeans_sift.py --subset learn --max-samples 5000 --k-values 10 20 50 100

# Quick test mode
python test_kmeans_sift.py --quick

# Simple example
python kmeans_example.py
```

## API Reference

### KMeans Class

```python
KMeans(
    n_clusters=10,        # Number of clusters
    max_iters=300,        # Maximum iterations
    tol=1e-4,            # Convergence tolerance
    n_init=10,           # Number of initialization attempts
    init='k-means++',    # Initialization method
    random_state=None,   # Random seed
    verbose=False        # Print progress
)
```

**Methods:**
- `fit(X)` - Fit the model to data
- `predict(X)` - Predict cluster labels for new data
- `fit_predict(X)` - Fit and predict in one step
- `get_cluster_info()` - Get detailed cluster information

**Attributes:**
- `cluster_centers_` - Final cluster centroids
- `labels_` - Cluster labels for training data
- `inertia_` - Within-cluster sum of squares
- `n_iter_` - Number of iterations until convergence

### Utility Functions

```python
# Load SIFT dataset
base_vectors, learn_vectors, query_vectors, groundtruth = load_sift_data()

# Evaluate clustering performance
metrics = evaluate_clustering(kmeans_model, X)

# Benchmark multiple k values
results = benchmark_kmeans(X, k_values=[10, 20, 50])

# Find optimal k automatically
optimal_k = find_optimal_k(X, k_range=(2, 20), method='elbow')

# Create sample dataset for testing
X, y = create_sample_dataset(n_samples=1000, n_clusters=10)
```

## Test Results

### SIFT Learn Dataset (3000 samples, 128 features)

| K Value | Inertia | Silhouette Score | Fit Time (s) |
|---------|---------|------------------|--------------|
| 10      | 241.6M  | 0.0669          | 11.6         |
| 25      | 215.8M  | 0.0506          | 43.4         |
| 50      | 198.5M  | 0.0381          | 83.2         |

- **Optimal k (elbow method):** 5
- **Best k (silhouette score):** 10
- **Average cluster size:** 300 samples
- **Convergence:** Typically 15-45 iterations

### Performance Characteristics

- **Scalability:** Handles datasets up to 100K+ samples efficiently
- **Memory usage:** ~O(n_samples × n_features + k × n_features)
- **Time complexity:** O(n_samples × n_clusters × n_features × n_iterations)
- **Convergence:** Usually converges in 15-50 iterations

## File Structure

```
kmeans/
├── __init__.py          # Package initialization
├── kmeans.py           # Main K-means implementation
└── utils.py            # Utility functions and SIFT data loading

test_kmeans_sift.py     # Comprehensive testing script
kmeans_example.py       # Simple usage example
```

## Usage Examples

### 1. Find Optimal Number of Clusters
```python
from kmeans import find_optimal_k, load_sift_data

# Load data
_, learn_vectors, _, _ = load_sift_data()
X = learn_vectors[:5000]

# Find optimal k using elbow method
optimal_k = find_optimal_k(X, k_range=(2, 20), method='elbow')
print(f"Optimal k: {optimal_k}")
```

### 2. Benchmark Multiple K Values
```python
from kmeans import benchmark_kmeans, load_sift_data

# Load data
_, learn_vectors, _, _ = load_sift_data()
X = learn_vectors[:3000]

# Test different k values
results = benchmark_kmeans(X, k_values=[5, 10, 15, 20, 25])

# Print results
for k, metrics in results.items():
    print(f"k={k}: Silhouette={metrics['silhouette_score']:.4f}")
```

### 3. Detailed Cluster Analysis
```python
from kmeans import KMeans, evaluate_clustering, load_sift_data

# Load and prepare data
_, learn_vectors, _, _ = load_sift_data()
X = learn_vectors[:2000]

# Fit model
kmeans = KMeans(n_clusters=15, verbose=True)
kmeans.fit(X)

# Detailed evaluation
metrics = evaluate_clustering(kmeans, X)
cluster_info = kmeans.get_cluster_info()

print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
print(f"Average cluster size: {cluster_info['avg_cluster_size']:.1f}")
```

## Command Line Usage

The test script supports various command line options:

```bash
# Test on different SIFT subsets
python test_kmeans_sift.py --subset base    # Use base vectors (1M samples)
python test_kmeans_sift.py --subset learn   # Use learn vectors (100K samples)
python test_kmeans_sift.py --subset query   # Use query vectors (10K samples)

# Limit sample size for faster testing
python test_kmeans_sift.py --max-samples 1000

# Test specific k values
python test_kmeans_sift.py --k-values 5 10 20 50

# Don't save results to file
python test_kmeans_sift.py --no-save

# Quick validation test
python test_kmeans_sift.py --quick
```

## Performance Tips

1. **For large datasets:** Use `max_samples` parameter to limit data size
2. **For faster convergence:** Reduce `n_init` parameter
3. **For better results:** Increase `max_iters` and `n_init`
4. **For reproducible results:** Set `random_state` parameter

## Evaluation Metrics

The implementation provides comprehensive evaluation:

- **Inertia:** Within-cluster sum of squared distances (lower is better)
- **Silhouette Score:** Measure of cluster separation (-1 to 1, higher is better)
- **Calinski-Harabasz Score:** Ratio of between-cluster to within-cluster dispersion
- **Davies-Bouldin Score:** Average similarity between clusters (lower is better)
- **Cluster size distribution:** Statistics on cluster balance

## Troubleshooting

### Common Issues

1. **Import errors:** Ensure the `kmeans` directory is in your Python path
2. **Memory errors:** Reduce `max_samples` or use smaller dataset subset
3. **Slow performance:** Reduce `n_init` or `max_iters` parameters
4. **SIFT data not found:** Ensure SIFT files are in the `sift/` directory

### SIFT Dataset Setup

The SIFT dataset files should be in the `sift/` directory:
- `sift_base.fvecs` - Base vectors (1M samples)
- `sift_learn.fvecs` - Learn vectors (100K samples)  
- `sift_query.fvecs` - Query vectors (10K samples)
- `sift_groundtruth.ivecs` - Ground truth nearest neighbors

## License

This implementation is provided under the same license as the parent project.

## Comparison with Standard Libraries

| Feature | This Implementation | scikit-learn KMeans |
|---------|-------------------|-------------------|
| SIFT data loading | ✅ Built-in | ❌ Manual |
| k-means++ init | ✅ Yes | ✅ Yes |
| Multiple init | ✅ Yes | ✅ Yes |
| Early stopping | ✅ Yes | ✅ Yes |
| Comprehensive metrics | ✅ Yes | ⚠️ Basic |
| Memory efficiency | ✅ Optimized | ✅ Yes |
| Customization | ✅ High | ⚠️ Limited |

This implementation is specifically optimized for the SIFT dataset workflow while maintaining compatibility with general vector clustering tasks.
