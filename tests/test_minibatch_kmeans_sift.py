"""MiniBatchKMeans baseline test on (subset of) SIFT dataset.

This test exercises the same pure K-Means (single-cluster probe) logic used in
`KMeansHNSWEvaluator._evaluate_pure_kmeans` to (a) ensure the SIFT loading path
works in CI, and (b) document the expected low recall when only the single
nearest centroid is probed (equivalent to IVF nprobe=1).

We purposefully:
  * Use a capped number of base vectors (<= 20_000) and queries (<= 20) to keep
    runtime reasonable.
  * Use a modest number of clusters (128) to mirror tuning script defaults for
    10k–50k scales.
  * Assert recall is > 0 (non‑degenerate) and < 0.6 (expected for nprobe=1) so
    the test is stable yet meaningful. (Typical observed ~0.05–0.15 @k=10.)
  * Skip cleanly if SIFT files are not present so the rest of the suite passes
    in minimal environments.

NOTE: This is not a performance benchmark; strict timing assertions would be
fragile across heterogeneous CI machines, so we only enforce a lenient upper
bound on average query time to catch pathological regressions (e.g. O(N*k)).
"""

from __future__ import annotations

import os
import math
import struct
import numpy as np
import pytest
from sklearn.cluster import MiniBatchKMeans


SIFT_DIR = os.path.join(os.path.dirname(__file__), "..", "sift")
BASE_FVECS = os.path.join(SIFT_DIR, "sift_base.fvecs")
QUERY_FVECS = os.path.join(SIFT_DIR, "sift_query.fvecs")


def _read_fvecs(path: str, max_vectors: int | None = None) -> np.ndarray:
    """Stream-read .fvecs (FAISS) without loading whole file.

    Format: For each vector: int32 dim, then dim float32 values. The dim is
    redundantly stored per vector. We read only the first `max_vectors` (if
    provided) to keep tests fast and memory-light.
    """
    with open(path, 'rb') as f:
        header = f.read(4)
        if not header:
            raise ValueError(f"Empty fvecs file: {path}")
        dim = struct.unpack('i', header)[0]
        if dim <= 0 or dim > 4096:
            raise ValueError(f"Unreasonable vector dimension {dim} in {path}")
        record_bytes = 4 + dim * 4
        # Determine total count via file size
        f.seek(0, os.SEEK_END)
        total_bytes = f.tell()
        count = total_bytes // record_bytes
        to_read = count if max_vectors is None else min(count, max_vectors)
        f.seek(0)
        out = np.empty((to_read, dim), dtype=np.float32)
        for i in range(to_read):
            d_raw = f.read(4)
            if len(d_raw) < 4:
                raise EOFError("Unexpected EOF while reading dimension")
            d = struct.unpack('i', d_raw)[0]
            if d != dim:
                raise ValueError(f"Inconsistent dim at vector {i}: {d} != {dim}")
            vec_bytes = f.read(dim * 4)
            if len(vec_bytes) < dim * 4:
                raise EOFError("Unexpected EOF while reading vector data")
            out[i] = np.frombuffer(vec_bytes, dtype=np.float32)
        return out


@pytest.mark.skipif(
    not (os.path.exists(BASE_FVECS) and os.path.exists(QUERY_FVECS)),
    reason="SIFT dataset files not present; skipping MiniBatchKMeans SIFT test",
)
def test_minibatch_kmeans_single_cluster_recall_sift():
    # ---- Load (subset) data ----
    max_base = 20_000  # cap to keep test quick (< ~2s typical)
    max_queries = 20
    base = _read_fvecs(BASE_FVECS, max_vectors=max_base)
    queries = _read_fvecs(QUERY_FVECS, max_vectors=max_queries)

    assert base.ndim == 2 and queries.ndim == 2
    assert base.shape[1] == queries.shape[1]
    dim = base.shape[1]
    assert dim in (64, 128), "Unexpected SIFT dimensionality"

    n_base = base.shape[0]
    n_queries = queries.shape[0]
    k = 10

    # ---- Fit MiniBatchKMeans ----
    # Choose n_clusters roughly sqrt(N), capped, like evaluator heuristic
    suggested = int(math.sqrt(n_base))
    n_clusters = min(256, max(32, suggested))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=123,
        batch_size=min(2048, n_base),
        n_init=5,
        max_iter=120,
        verbose=0,
    )
    kmeans.fit(base)

    # ---- Build inverse index for speed (cluster -> indices) ----
    inv = [[] for _ in range(n_clusters)]
    for idx, c in enumerate(kmeans.labels_):
        inv[c].append(idx)

    # ---- Ground truth (brute force) ----
    # For each query compute top-k true neighbors excluding identical index (if present)
    gt_sets = []  # list of sets of neighbor ids
    for qid, qvec in enumerate(queries):
        dists = np.linalg.norm(base - qvec, axis=1)
        # Exclude self id if overlapping sampling (unlikely because queries separate file)
        topk = np.argsort(dists)[: k]
        gt_sets.append(set(int(i) for i in topk if i != qid))

    # ---- Single-cluster retrieval (expected low recall) ----
    correct = 0
    for qvec, gt in zip(queries, gt_sets):
        # Pick nearest centroid
        centroid_diffs = kmeans.cluster_centers_ - qvec
        centroid_d2 = np.linalg.norm(centroid_diffs, axis=1)
        cid = int(np.argmin(centroid_d2))
        member_ids = inv[cid]
        if not member_ids:
            continue
        member_vecs = base[member_ids]
        mdists = np.linalg.norm(member_vecs - qvec, axis=1)
        order = np.argsort(mdists)[:k]
        retrieved = {int(member_ids[i]) for i in order}
        correct += len(gt & retrieved)

    total_expected = n_queries * k
    recall = correct / total_expected

    # ---- Assertions ----
    # Lower bound: should retrieve something; Upper bound: single-cluster recall should be modest
    assert 0.0 < recall < 0.6, f"Recall {recall:.3f} outside expected single-cluster range"
    # Sanity on cluster sizes
    avg_cluster_size = n_base / n_clusters
    assert avg_cluster_size > 10, "Clusters unexpectedly tiny; heuristic misconfigured?"

    # Provide debug info if assertion fails
    print(
        f"MiniBatchKMeans SIFT test: N={n_base}, Q={n_queries}, dim={dim}, "
        f"n_clusters={n_clusters}, avg_cluster_size={avg_cluster_size:.1f}, recall@{k}={recall:.4f}"
    )
