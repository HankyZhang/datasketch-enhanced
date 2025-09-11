import unittest
import warnings
import sys
import os
import time
from typing import List, Tuple, Dict, Any

import numpy as np

# Import HNSW from local module
from datasketch.hnsw import HNSW

# Try to import MinHash, skip MinHash tests if not available
try:
    # Import MinHash from installed datasketch by temporarily modifying sys.path
    original_sys_path = sys.path.copy()
    # Remove current directory and any path containing this project
    project_paths = [p for p in sys.path if 'datasketch-enhanced' in p or p == '.' or p == '']
    for path in project_paths:
        if path in sys.path:
            sys.path.remove(path)
    
    from datasketch.minhash import MinHash
    sys.path[:] = original_sys_path
    MINHASH_AVAILABLE = True
except ImportError:
    sys.path[:] = original_sys_path
    MINHASH_AVAILABLE = False
    print("Warning: MinHash not available, skipping MinHash tests")


def l2_distance(x, y):
    return np.linalg.norm(x - y)


def jaccard_distance(x, y):
    return 1.0 - float(len(np.intersect1d(x, y, assume_unique=False))) / float(
        len(np.union1d(x, y))
    )


class TestHNSW(unittest.TestCase):
    def _create_random_points(self, n=100, dim=10):
        return np.random.rand(n, dim)

    def _create_index(self, vecs, keys=None):
        hnsw = HNSW(
            distance_func=l2_distance,
            m=16,
            ef_construction=100,
        )
        self._insert_points(hnsw, vecs, keys)
        return hnsw

    def _search_index(self, index, queries, k=10):
        return self._search_index_dist(index, queries, l2_distance, k)

    def _insert_points(self, index, points, keys=None):
        original_length = len(index)

        if keys is None:
            keys = list(range(len(points)))

        for i, (key, point) in enumerate(zip(keys, points)):
            # Test insert.
            if i % 2 == 0:
                index.insert(key, point)
            else:
                index[key] = point
            # Make sure the entry point is set.
            self.assertTrue(index._entry_point is not None)
            # Test contains.
            self.assertIn(key, index)
            if original_length == 0:
                self.assertNotIn(key + 1, index)
            # Test get.
            self.assertTrue(np.array_equal(index.get(key), point))
            self.assertTrue(np.array_equal(index[key], point))

        if original_length == 0:
            # Test length.
            self.assertEqual(len(index), len(points))

            # Test order.
            for key_indexed, key in zip(index, keys):
                self.assertEqual(key_indexed, key)
            for key_indexed, key in zip(index.keys(), keys):
                self.assertEqual(key_indexed, key)
            for vec_indexed, vec in zip(index.values(), points):
                self.assertTrue(np.array_equal(vec_indexed, vec))
            for (key_indexed, vec_indexed), key, vec in zip(
                index.items(), keys, points
            ):
                self.assertEqual(key_indexed, key)
                self.assertTrue(np.array_equal(vec_indexed, vec))

    def _search_index_dist(self, index, queries, distance_func, k=10):
        for i in range(len(queries)):
            results = index.query(queries[i], 10)
            # Check graph connectivity.
            self.assertEqual(len(results), 10)
            for j in range(len(results) - 1):
                self.assertLessEqual(
                    distance_func(index[results[j][0]], queries[i]),
                    distance_func(index[results[j + 1][0]], queries[i]),
                )

    def test_search(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        self._search_index(hnsw, data)

    def test_upsert(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        new_data = self._create_random_points(n=10, dim=10)
        self._insert_points(hnsw, new_data)
        self._search_index(hnsw, new_data)

    def test_update(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        new_data = self._create_random_points(n=10, dim=10)
        hnsw.update({i: new_data[i] for i in range(len(new_data))})
        self._search_index(hnsw, new_data)

    def test_merge(self):
        data1 = self._create_random_points()
        data2 = self._create_random_points()
        hnsw1 = self._create_index(data1, keys=list(range(len(data1))))
        hnsw2 = self._create_index(
            data2, keys=list(range(len(data1), len(data1) + len(data2)))
        )
        new_index = hnsw1.merge(hnsw2)
        self._search_index(new_index, data1)
        self._search_index(new_index, data2)
        for i in range(len(data1)):
            self.assertIn(i, new_index)
            self.assertTrue(np.array_equal(new_index[i], data1[i]))
        for i in range(len(data2)):
            self.assertIn(i + len(data1), new_index)
            self.assertTrue(np.array_equal(new_index[i + len(data1)], data2[i]))

    def test_pickle(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        import pickle

        hnsw2 = pickle.loads(pickle.dumps(hnsw))
        self.assertEqual(hnsw, hnsw2)

    def test_copy(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        hnsw2 = hnsw.copy()
        self.assertEqual(hnsw, hnsw2)

        hnsw.remove(0)
        self.assertTrue(0 not in hnsw)
        self.assertTrue(0 in hnsw2)

    def test_soft_remove_and_pop_and_clean(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        # Remove all points except the last one.
        for i in range(len(data) - 1):
            if i % 2 == 0:
                hnsw.remove(i)
            else:
                point = hnsw.pop(i)
                self.assertTrue(np.array_equal(point, data[i]))
            self.assertNotIn(i, hnsw)
            self.assertEqual(len(hnsw), len(data) - i - 1)
            self.assertRaises(KeyError, hnsw.pop, i)
            # Test idempotency.
            hnsw.remove(i)
            hnsw.remove(i)
            hnsw.remove(i)
            results = hnsw.query(data[i], 10)
            # Check graph connectivity.
            # self.assertEqual(len(results), min(10, len(data) - i - 1))
            expected_result_size = min(10, len(data) - i - 1)
            if len(results) != expected_result_size:
                warnings.warn(
                    f"Issue encountered at i={i} during soft remove unit test: "
                    f"expected {expected_result_size} results, "
                    f"got {len(results)} results. "
                    "Potential graph connectivity issue."
                )
                # NOTE: we are not getting the expected number of results.
                # Try hard remove all previous soft removed points.
                hnsw.clean()
                results = hnsw.query(data[i], 10)
                self.assertEqual(len(results), min(10, len(data) - i - 1))
        # Remove last point.
        hnsw.remove(len(data) - 1)
        self.assertNotIn(len(data) - 1, hnsw)
        self.assertEqual(len(hnsw), 0)
        self.assertRaises(KeyError, hnsw.pop, len(data) - 1)
        self.assertRaises(KeyError, hnsw.remove, len(data) - 1)
        # Test search on empty index.
        self.assertRaises(ValueError, hnsw.query, data[0])
        # Test clean.
        hnsw.clean()
        self.assertEqual(len(hnsw), 0)
        self.assertRaises(KeyError, hnsw.remove, 0)
        self.assertRaises(ValueError, hnsw.query, data[0])

    def test_hard_remove_and_pop_and_clean(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        # Remove all points except the last one.
        for i in range(len(data) - 1):
            if i % 2 == 0:
                hnsw.remove(i, hard=True)
            else:
                point = hnsw.pop(i, hard=True)
                self.assertTrue(np.array_equal(point, data[i]))
            self.assertNotIn(i, hnsw)
            self.assertEqual(len(hnsw), len(data) - i - 1)
            self.assertRaises(KeyError, hnsw.pop, i)
            self.assertRaises(KeyError, hnsw.remove, i)
            results = hnsw.query(data[i], 10)
            # Check graph connectivity.
            self.assertEqual(len(results), min(10, len(data) - i - 1))
        # Remove last point.
        hnsw.remove(len(data) - 1, hard=True)
        self.assertNotIn(len(data) - 1, hnsw)
        self.assertEqual(len(hnsw), 0)
        self.assertRaises(KeyError, hnsw.pop, len(data) - 1)
        self.assertRaises(KeyError, hnsw.remove, len(data) - 1)
        # Test search on empty index.
        self.assertRaises(ValueError, hnsw.query, data[0])
        # Test clean.
        hnsw.clean()
        self.assertEqual(len(hnsw), 0)
        self.assertRaises(KeyError, hnsw.remove, 0)
        self.assertRaises(ValueError, hnsw.query, data[0])

    def test_popitem_last(self):
        data = self._create_random_points()
        for hard in [True, False]:
            hnsw = self._create_index(data)
            for i in range(len(data)):
                key, point = hnsw.popitem(hard=hard)
                self.assertTrue(np.array_equal(point, data[key]))
                self.assertEqual(key, len(data) - i - 1)
                self.assertTrue(np.array_equal(point, data[len(data) - i - 1]))
                self.assertNotIn(key, hnsw)
                self.assertEqual(len(hnsw), len(data) - i - 1)
            self.assertRaises(KeyError, hnsw.popitem)

    def test_popitem_first(self):
        data = self._create_random_points()
        for hard in [True, False]:
            hnsw = self._create_index(data)
            for i in range(len(data)):
                key, point = hnsw.popitem(last=False, hard=hard)
                self.assertTrue(np.array_equal(point, data[key]))
                self.assertEqual(key, i)
                self.assertTrue(np.array_equal(point, data[i]))
                self.assertNotIn(key, hnsw)
                self.assertEqual(len(hnsw), len(data) - i - 1)
            self.assertRaises(KeyError, hnsw.popitem)

    def test_clear(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        hnsw.clear()
        self.assertEqual(len(hnsw), 0)
        self.assertRaises(StopIteration, next, iter(hnsw))
        self.assertRaises(StopIteration, next, iter(hnsw.keys()))
        self.assertRaises(StopIteration, next, iter(hnsw.values()))
        self.assertRaises(KeyError, hnsw.pop, 0)
        self.assertRaises(KeyError, hnsw.__getitem__, 0)
        self.assertRaises(KeyError, hnsw.popitem)
        self.assertRaises(ValueError, hnsw.query, data[0])


class TestHNSWRecall(unittest.TestCase):
    """Test class specifically for measuring baseline HNSW recall performance."""
    
    def setUp(self):
        """Set up test data for recall testing."""
        np.random.seed(42)  # For reproducible results
        
    def _compute_ground_truth(self, dataset: np.ndarray, queries: np.ndarray, k: int) -> Dict[int, List[int]]:
        """Compute ground truth using brute force search."""
        ground_truth = {}
        
        for query_idx, query_vector in enumerate(queries):
            distances = []
            for data_idx, data_vector in enumerate(dataset):
                dist = l2_distance(query_vector, data_vector)
                distances.append((data_idx, dist))
            
            # Sort by distance and take top k
            distances.sort(key=lambda x: x[1])
            ground_truth[query_idx] = [idx for idx, _ in distances[:k]]
            
        return ground_truth
    
    def _calculate_recall(self, hnsw_results: List[int], ground_truth: List[int]) -> float:
        """Calculate recall between HNSW results and ground truth."""
        if not ground_truth:
            return 0.0
        
        hnsw_set = set(hnsw_results)
        gt_set = set(ground_truth)
        
        intersection = hnsw_set.intersection(gt_set)
        return len(intersection) / len(gt_set)
    
    def test_baseline_hnsw_recall_small(self):
        """Test baseline HNSW recall on a small dataset."""
        # Create test dataset
        n_data = 1000
        n_queries = 50
        dim = 64
        k = 10
        
        # Generate data
        dataset = np.random.random((n_data, dim)).astype(np.float32)
        query_indices = np.random.choice(n_data, n_queries, replace=False)
        queries = dataset[query_indices]
        
        # Remove query points from dataset to avoid trivial matches
        data_mask = np.ones(n_data, dtype=bool)
        data_mask[query_indices] = False
        filtered_dataset = dataset[data_mask]
        filtered_indices = np.arange(n_data)[data_mask]
        
        print(f"\n📊 Testing Baseline HNSW Recall:")
        print(f"   Dataset size: {len(filtered_dataset)}")
        print(f"   Query count: {n_queries}")
        print(f"   Vector dimension: {dim}")
        print(f"   k: {k}")
        
        # Compute ground truth
        print("   Computing ground truth...")
        start_time = time.time()
        ground_truth = {}
        for query_idx, query_vector in enumerate(queries):
            distances = []
            for data_idx, data_vector in enumerate(filtered_dataset):
                dist = l2_distance(query_vector, data_vector)
                distances.append((filtered_indices[data_idx], dist))
            
            distances.sort(key=lambda x: x[1])
            ground_truth[query_idx] = [idx for idx, _ in distances[:k]]
        
        gt_time = time.time() - start_time
        print(f"   Ground truth computed in {gt_time:.2f}s")
        
        # Test different HNSW configurations
        test_configs = [
            {'m': 8, 'ef_construction': 100, 'ef_search': 50},
            {'m': 16, 'ef_construction': 200, 'ef_search': 100},
            {'m': 32, 'ef_construction': 400, 'ef_search': 200},
        ]
        
        results = []
        
        for config in test_configs:
            print(f"\n   Testing: m={config['m']}, ef_construction={config['ef_construction']}, ef_search={config['ef_search']}")
            
            # Build HNSW index
            start_time = time.time()
            hnsw = HNSW(
                distance_func=l2_distance,
                m=config['m'],
                ef_construction=config['ef_construction']
            )
            
            # Insert data points
            for idx, vector in zip(filtered_indices, filtered_dataset):
                hnsw.insert(idx, vector)
            
            build_time = time.time() - start_time
            print(f"     Build time: {build_time:.2f}s")
            
            # Test queries
            start_time = time.time()
            recalls = []
            
            for query_idx, query_vector in enumerate(queries):
                # Get HNSW results
                hnsw_results = hnsw.query(query_vector, k=k, ef=config['ef_search'])
                hnsw_ids = [result_id for result_id, _ in hnsw_results]
                
                # Calculate recall
                recall = self._calculate_recall(hnsw_ids, ground_truth[query_idx])
                recalls.append(recall)
            
            query_time = time.time() - start_time
            avg_recall = np.mean(recalls)
            avg_query_time = query_time / n_queries
            
            result = {
                'config': config,
                'recall@k': avg_recall,
                'avg_query_time': avg_query_time,
                'build_time': build_time,
                'std_recall': np.std(recalls)
            }
            results.append(result)
            
            print(f"     Recall@{k}: {avg_recall:.4f} ± {np.std(recalls):.4f}")
            print(f"     Avg query time: {avg_query_time*1000:.2f}ms")
            
            # Basic assertion - recall should be reasonable
            self.assertGreater(avg_recall, 0.1, f"Recall too low: {avg_recall:.4f}")
            self.assertLess(avg_query_time, 1.0, f"Query time too slow: {avg_query_time:.4f}s")
        
        # Find best configuration
        best_result = max(results, key=lambda x: x['recall@k'])
        print(f"\n   🏆 Best configuration:")
        print(f"     Config: {best_result['config']}")
        print(f"     Recall@{k}: {best_result['recall@k']:.4f}")
        print(f"     Query time: {best_result['avg_query_time']*1000:.2f}ms")
        
        # Store results for potential comparison
        self.baseline_recall_results = results
        
        return results
    
    def test_baseline_hnsw_recall_medium(self):
        """Test baseline HNSW recall on a medium dataset."""
        # Create test dataset
        n_data = 5000
        n_queries = 100
        dim = 128
        k = 10
        
        # Generate data
        dataset = np.random.random((n_data, dim)).astype(np.float32)
        query_indices = np.random.choice(n_data, n_queries, replace=False)
        queries = dataset[query_indices]
        
        # Remove query points from dataset
        data_mask = np.ones(n_data, dtype=bool)
        data_mask[query_indices] = False
        filtered_dataset = dataset[data_mask]
        filtered_indices = np.arange(n_data)[data_mask]
        
        print(f"\n📊 Testing Baseline HNSW Recall (Medium Dataset):")
        print(f"   Dataset size: {len(filtered_dataset)}")
        print(f"   Query count: {n_queries}")
        print(f"   Vector dimension: {dim}")
        print(f"   k: {k}")
        
        # Use optimized configuration
        config = {'m': 16, 'ef_construction': 200, 'ef_search': 100}
        
        # Build HNSW index
        print("   Building HNSW index...")
        start_time = time.time()
        hnsw = HNSW(
            distance_func=l2_distance,
            m=config['m'],
            ef_construction=config['ef_construction']
        )
        
        for idx, vector in zip(filtered_indices, filtered_dataset):
            hnsw.insert(idx, vector)
        
        build_time = time.time() - start_time
        print(f"   Build time: {build_time:.2f}s")
        
        # Sample ground truth computation (for performance)
        print("   Computing sample ground truth...")
        sample_size = min(20, n_queries)
        sample_queries = queries[:sample_size]
        
        start_time = time.time()
        ground_truth = {}
        for query_idx, query_vector in enumerate(sample_queries):
            distances = []
            for data_idx, data_vector in enumerate(filtered_dataset):
                dist = l2_distance(query_vector, data_vector)
                distances.append((filtered_indices[data_idx], dist))
            
            distances.sort(key=lambda x: x[1])
            ground_truth[query_idx] = [idx for idx, _ in distances[:k]]
        
        gt_time = time.time() - start_time
        print(f"   Ground truth computed in {gt_time:.2f}s")
        
        # Test HNSW queries
        start_time = time.time()
        recalls = []
        
        for query_idx, query_vector in enumerate(sample_queries):
            hnsw_results = hnsw.query(query_vector, k=k, ef=config['ef_search'])
            hnsw_ids = [result_id for result_id, _ in hnsw_results]
            
            recall = self._calculate_recall(hnsw_ids, ground_truth[query_idx])
            recalls.append(recall)
        
        query_time = time.time() - start_time
        avg_recall = np.mean(recalls)
        avg_query_time = query_time / sample_size
        
        print(f"   Results:")
        print(f"     Recall@{k}: {avg_recall:.4f} ± {np.std(recalls):.4f}")
        print(f"     Avg query time: {avg_query_time*1000:.2f}ms")
        print(f"     Total queries tested: {sample_size}")
        
        # Assertions
        self.assertGreater(avg_recall, 0.15, f"Recall too low for medium dataset: {avg_recall:.4f}")
        self.assertLess(avg_query_time, 0.1, f"Query time too slow for medium dataset: {avg_query_time:.4f}s")
        
        return {
            'dataset_size': len(filtered_dataset),
            'recall@k': avg_recall,
            'avg_query_time': avg_query_time,
            'build_time': build_time,
            'config': config
        }


class TestHNSWLayerWithReversedEdges(TestHNSW):
    def _create_index(self, vecs, keys=None):
        hnsw = HNSW(
            distance_func=l2_distance,
            m=16,
            ef_construction=100,
            reversed_edges=True,
        )
        self._insert_points(hnsw, vecs, keys)
        return hnsw


class TestHNSWJaccard(TestHNSW):
    def _create_random_points(self, high=50, n=100, dim=10):
        return np.random.randint(0, high, (n, dim))

    def _create_index(self, sets, keys=None):
        hnsw = HNSW(
            distance_func=jaccard_distance,
            m=16,
            ef_construction=100,
        )
        self._insert_points(hnsw, sets, keys)
        return hnsw

    def _search_index(self, index, queries, k=10):
        return super()._search_index_dist(index, queries, jaccard_distance, k)


def minhash_jaccard_distance(x, y) -> float:
    return 1.0 - x.jaccard(y)


@unittest.skipUnless(MINHASH_AVAILABLE, "MinHash not available")
class TestHNSWMinHashJaccard(TestHNSW):
    def _create_random_points(self, high=50, n=100, dim=10):
        sets = np.random.randint(0, high, (n, dim))
        return MinHash.bulk(sets, num_perm=128)

    def _create_index(self, minhashes, keys=None):
        hnsw = HNSW(
            distance_func=minhash_jaccard_distance,
            m=16,
            ef_construction=100,
        )
        self._insert_points(hnsw, minhashes, keys)
        return hnsw

    def _search_index(self, index, queries, k=10):
        return super()._search_index_dist(index, queries, minhash_jaccard_distance, k)


if __name__ == "__main__":
    unittest.main()
