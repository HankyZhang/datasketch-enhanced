import pytest
import numpy as np
from hnsw_hybrid_evaluation import generate_synthetic_dataset, split_query_set_from_dataset


def test_split_query_set_from_dataset_basic():
    dataset = generate_synthetic_dataset(500, 32)
    train, queries = split_query_set_from_dataset(dataset, 50, seed=999)
    assert len(dataset) == 500
    assert len(queries) == 50
    assert len(train) == 450
    # Ensure disjoint
    assert set(train.keys()).isdisjoint(set(queries.keys()))
    # Deterministic split
    train2, queries2 = split_query_set_from_dataset(dataset, 50, seed=999)
    assert list(sorted(queries.keys())) == list(sorted(queries2.keys()))


def test_split_query_set_error():
    dataset = generate_synthetic_dataset(10, 8)
    with pytest.raises(ValueError):
        split_query_set_from_dataset(dataset, 10)
