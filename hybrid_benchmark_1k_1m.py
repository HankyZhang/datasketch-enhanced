#!/usr/bin/env python3
"""
Hybrid HNSW Benchmark (1K & 100K Scales)

Goal:
  Measure hybrid HNSW recall & latency under different parameter settings at:
    - Small scale (1K base vectors) for rapid iteration (brute-force GT)
    - Large scale (100K base vectors) using provided SIFT ground truth (no brute-force)

Parameters (env overrides):
  SMALL_BASE=1000          # base vectors for small test
  SMALL_QUERIES=100        # queries for small test (subset of sift_query)
  LARGE_BASE=100000        # base vectors for large test (default 100K for faster testing)
  LARGE_QUERIES=50         # number of queries sampled for large test (<=10000)
    K_CHILDREN=500,1000      # comma list. If omitted AND AUTO_K_CHILDREN=1 -> auto scale.
    AUTO_K_CHILDREN=1        # when set (and no explicit K_CHILDREN list) derive per-scale list
        DIV_MAX_ASSIGNMENTS=0,3  # comma list; 0 disables diversification; produces variants
        REPAIR_MIN_ASSIGNMENTS=0 # comma list; 0 disables repair; combined with diversification
  PARENT_LEVELS=1,2        # comma list
  N_PROBE=1,2,3,5,10,20    # comma list (auto-pruned by num_parents)
    N_PROBE_EXTRA_FRACTIONS=0.6,0.75,0.9  # added high coverage fractions (applied after 30/40/50%)
    N_PROBE_INCLUDE_FULL=0   # set 1 to allow n_probe == num_parents (full parent scan)
  METHODS=approx           # 'approx' or 'approx,brute' (brute is costly at 1M)
  K_GT=100                 # ground truth depth (SIFT GT has 100)
  K_EVAL=100               # retrieval depth per query

Outputs:
  Prints tables per scale
  Writes JSON: hybrid_benchmark_1k_1m.json

Notes:
  * For 1M scale we DO NOT brute force distances; we load sift_groundtruth.ivecs.
  * For recall@10 & recall@100 we rely on GT top-100 list (standard practice).
  * Building hybrid with very large k_children or brute mapping at 1M can be slow.
"""

from __future__ import annotations
import os, struct, time, json, math, random
from typing import List, Dict, Any, Tuple
import numpy as np

from hnsw.hnsw import HNSW  # local module
from hybrid_hnsw.hnsw_hybrid import HNSWHybrid


# ---------------- File Readers -----------------
def read_fvecs(path: str, max_count: int | None = None) -> np.ndarray:
    """Read first max_count vectors from .fvecs (dim:int, dim*float32)."""
    out = []
    with open(path, 'rb') as f:
        n = 0
        while True:
            if max_count and n >= max_count:
                break
            head = f.read(4)
            if len(head) < 4:
                break
            dim = struct.unpack('i', head)[0]
            data = f.read(4 * dim)
            if len(data) < 4 * dim:
                break
            vec = struct.unpack('f' * dim, data)
            out.append(vec)
            n += 1
    return np.asarray(out, dtype=np.float32)


def read_ivecs(path: str, max_count: int | None = None) -> np.ndarray:
    out = []
    with open(path, 'rb') as f:
        n = 0
        while True:
            if max_count and n >= max_count:
                break
            head = f.read(4)
            if len(head) < 4:
                break
            dim = struct.unpack('i', head)[0]
            data = f.read(4 * dim)
            if len(data) < 4 * dim:
                break
            ids = struct.unpack('i' * dim, data)
            out.append(ids)
            n += 1
    return np.asarray(out, dtype=np.int32)


# ------------- Ground Truth Handling -------------
def compute_bruteforce_gt(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Brute force ground truth for small scale (vectorized chunk)."""
    print(f"[GT] Computing brute force ground truth: queries={len(queries)} k={k}")
    gt_rows = []
    for i, q in enumerate(queries):
        if i % max(1, len(queries)//10) == 0:
            print(f"  - query {i+1}/{len(queries)}")
        d = np.linalg.norm(base - q, axis=1)
        part = np.argpartition(d, k)[:k]
        part = part[np.argsort(d[part])]
        gt_rows.append(part)
    return np.vstack(gt_rows)


def load_sift_gt(k: int) -> np.ndarray:
    """Load SIFT official ground truth (each row length >= k)."""
    gt = read_ivecs('sift/sift_groundtruth.ivecs')
    if gt.shape[1] < k:
        raise ValueError(f"Ground truth only has {gt.shape[1]} < k={k}")
    return gt[:, :k]


# ------------- Evaluation -----------------
def recall_at_k(result_ids: List[int], gt_ids: np.ndarray, k: int) -> float:
    if not result_ids:
        return 0.0
    return len(set(result_ids[:k]) & set(gt_ids[:k])) / k


def evaluate_hybrid(
    hybrid: HNSWHybrid,
    queries: np.ndarray,
    gt_subset: np.ndarray,
    n_probe: int,
    k_retrieve: int,
    ks: Tuple[int, int] = (10, 100),
) -> Dict[str, Any]:
    k1, k2 = ks
    rec_k1, rec_k2 = [], []
    q_times = []
    for i, q in enumerate(queries):
        t0 = time.time()
        res = hybrid.search(q, k=k_retrieve, n_probe=n_probe)
        q_times.append((time.time() - t0) * 1000.0)
        ids = [rid for rid, _ in res]
        gt = gt_subset[i]
        rec_k1.append(recall_at_k(ids, gt, k1))
        rec_k2.append(recall_at_k(ids, gt, k2))
    stats = hybrid.get_stats()
    return {
        'recall_at_%d' % k1: float(np.mean(rec_k1)),
        'recall_at_%d' % k2: float(np.mean(rec_k2)),
        'avg_query_ms': float(np.mean(q_times)),
        'std_query_ms': float(np.std(q_times)),
        'coverage': stats.get('coverage_fraction'),
        'avg_candidate_size': stats.get('avg_candidate_size'),
        'approx_ef': getattr(hybrid, 'approx_ef', None),
        'num_parents': stats.get('num_parents'),
    }


# ------------- Benchmark Logic -----------------
def parse_list(env_name: str, default: str) -> List[int]:
    raw = os.getenv(env_name, default)
    if raw.lower() in ('', 'auto'):
        return []  # signal for auto if AUTO_K_CHILDREN enabled
    vals = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except ValueError:
            pass
    return vals


def parse_float_list(env_name: str, default: str) -> List[float]:
    raw = os.getenv(env_name, default)
    vals: List[float] = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def derive_k_children_list(dataset_size: int, num_parents: int = None) -> List[int]:
    """Adaptive k_children selection based on dataset size and number of parents.

    The key insight: k_children should ensure good coverage without excessive overlap.
    Coverage_ratio = (num_parents * k_children) / dataset_size should be 1.2-2.0 for good recall.
    """
    if dataset_size <= 5000:
        # Small datasets: use percentage-based approach
        frac_candidates = [0.05, 0.09, 0.12]
        values = []
        for f in frac_candidates:
            k = int(dataset_size * f)
            k = max(25, min(k, dataset_size - 10))
            values.append(k)
    else:
        # Large datasets: use coverage-based approach
        if num_parents:
            # Calculate k_children for different coverage ratios
            ideal_coverage_per_parent = dataset_size / num_parents
            values = []
            
            # Conservative: ~80% coverage per parent
            k1 = max(200, int(ideal_coverage_per_parent * 0.8))
            values.append(min(k1, 5000))
            
            # Balanced: ~120% coverage per parent (some overlap)
            k2 = max(300, int(ideal_coverage_per_parent * 1.2))
            values.append(min(k2, 8000))
            
            # Aggressive: ~200% coverage per parent (more overlap, better recall)
            k3 = max(500, int(ideal_coverage_per_parent * 2.0))
            values.append(min(k3, 15000))
        else:
            # Fallback to percentage if num_parents unknown
            frac_candidates = [0.005, 0.01, 0.015]
            values = []
            for f in frac_candidates:
                k = int(dataset_size * f)
                k = max(200, min(k, 15000))
                values.append(k)
    
    sqrt_k = int(math.sqrt(dataset_size))
    if dataset_size > 5000 and 200 <= sqrt_k <= 15000:
        values.append(sqrt_k)
    
    return sorted(set(values))


def run_scale(
    scale_name: str,
    base_vecs: np.ndarray,
    query_vecs: np.ndarray,
    gt: np.ndarray,
    k_children_list: List[int],
    parent_levels: List[int],
    methods: List[str],
    n_probe_list: List[int],
    k_eval: int,
    k_gt: int,
    limit_queries: int | None = None,
    auto_k: bool = False,
    diversify_values: List[int] | None = None,
    repair_values: List[int] | None = None,
    extra_probe_fracs: List[float] | None = None,
    include_full_probe: bool = False,
) -> List[Dict[str, Any]]:
    results = []
    if limit_queries and limit_queries < len(query_vecs):
        idx = np.random.choice(len(query_vecs), limit_queries, replace=False)
        query_vecs = query_vecs[idx]
        gt = gt[idx]
    distance_func = lambda x, y: np.linalg.norm(x - y)

    # Build base index (insert all vectors) – for 1M this is heavy.
    print(f"[{scale_name}] Building base HNSW (n={len(base_vecs)}) ...")
    t0 = time.time()
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    # Use batch update for large datasets to reduce Python overhead
    if len(base_vecs) >= 50000:
        dataset_dict = {i: base_vecs[i] for i in range(len(base_vecs))}
        base_index.update(dataset_dict)
        base_build = time.time() - t0
        print(f"[{scale_name}] Base index batch update done in {base_build:.2f}s")
    else:
        for i, v in enumerate(base_vecs):
            base_index.insert(i, v)
            if (i+1) % max(1, len(base_vecs)//10) == 0:
                print(f"  inserted {i+1}/{len(base_vecs)}")
        base_build = time.time() - t0
        print(f"[{scale_name}] Base index build time: {base_build:.2f}s")

    for method in methods:
        for parent_level in parent_levels:
            # Derive per-scale k_children dynamically if auto enabled
            # First, get number of parents at this level to calculate appropriate k_children
            if auto_k and not k_children_list:
                # Build a temporary count to determine parents at this level
                if parent_level >= len(base_index._graphs):
                    adjusted_level = len(base_index._graphs) - 1
                else:
                    adjusted_level = parent_level
                
                target_layer = base_index._graphs[adjusted_level]
                num_parents_at_level = len([nid for nid in target_layer 
                                          if nid in base_index and not base_index._nodes[nid].is_deleted])
                
                current_k_children_list = derive_k_children_list(len(base_vecs), num_parents_at_level)
                print(f"  Auto k_children for level {parent_level} ({num_parents_at_level} parents): {current_k_children_list}")
            else:
                current_k_children_list = k_children_list
            for k_children in current_k_children_list:
                # Variant matrix for diversify/repair
                div_list = diversify_values or [0]
                rep_list = repair_values or [0]
                for div in div_list:
                    for rep in rep_list:
                        label_suffix = []
                        if div and div > 0:
                            label_suffix.append(f"div{div}")
                        if rep and rep > 0:
                            label_suffix.append(f"rep{rep}")
                        variant_tag = '+'.join(label_suffix) if label_suffix else 'base'
                        print(f"[{scale_name}] Hybrid build method={method} level={parent_level} k_children={k_children} variant={variant_tag}")
                        t1 = time.time()
                        hybrid = HNSWHybrid(
                            base_index=base_index,
                            parent_level=parent_level,
                            k_children=k_children,
                            parent_child_method=method,
                            diversify_max_assignments=div or None,
                            repair_min_assignments=rep or None,
                        )
                        build_time = time.time() - t1
                        num_parents = hybrid.get_stats().get('num_parents', 0)
                        # Automatically adjust n_probe list according to num_parents.
                        # Previous behavior: drop any n_probe > num_parents. New behavior: cap & dedupe.
                        # Requirement: N_PROBE < num_parents (interpreted as not exceeding the parent count).
                        adjusted_n_probe: List[int] = []
                        for n in n_probe_list:
                            if num_parents <= 0:
                                adj = 1
                            elif n >= num_parents:
                                adj = num_parents - 1 if num_parents > 1 else 1
                            else:
                                adj = n
                            if adj < 1:
                                adj = 1
                            if adj not in adjusted_n_probe:
                                adjusted_n_probe.append(adj)
                        if not adjusted_n_probe:
                            adjusted_n_probe = [1]
                        # Ensure we probe sufficiently: include targets at ~30%,40%,50% of parents.
                        if num_parents > 1:
                            frac_targets = []
                            # Built-in fraction targets (large-scale guidance) now 20%,30%,40%,50%
                            for f in (0.20, 0.30, 0.40, 0.50):
                                t = int(round(num_parents * f))
                                if t >= num_parents:
                                    t = num_parents - 1
                                if t < 1:
                                    t = 1
                                frac_targets.append(t)
                            for t in frac_targets:
                                if t not in adjusted_n_probe:
                                    adjusted_n_probe.append(t)
                        # Add higher coverage fractions if configured
                        if num_parents > 1 and extra_probe_fracs:
                            for f in extra_probe_fracs:
                                if f <= 0:
                                    continue
                                t = int(round(num_parents * f))
                                if t >= num_parents:
                                    # allow equality only if include_full_probe specified
                                    if include_full_probe and t == num_parents:
                                        pass
                                    else:
                                        t = num_parents - 1 if num_parents > 1 else 1
                                if t < 1:
                                    t = 1
                                if t not in adjusted_n_probe:
                                    adjusted_n_probe.append(t)
                        if include_full_probe and num_parents > 0:
                            if num_parents not in adjusted_n_probe:
                                adjusted_n_probe.append(num_parents)
                        adjusted_n_probe = sorted(set(adjusted_n_probe))
                        if adjusted_n_probe != n_probe_list:
                            print(f"    adjusted n_probe list -> {adjusted_n_probe} (num_parents={num_parents})")
                        allowed_n_probe = adjusted_n_probe
                        for n_probe in allowed_n_probe:
                            eval_res = evaluate_hybrid(hybrid, query_vecs, gt, n_probe=n_probe, k_retrieve=k_eval)
                            row = {
                                'scale': scale_name,
                                'method': method,
                                'parent_level': parent_level,
                                'k_children': k_children,
                                'variant': variant_tag,
                                'diversify_max_assignments': div or None,
                                'repair_min_assignments': rep or None,
                                'n_probe': n_probe,
                                'k_children_mode': 'auto' if auto_k and not k_children_list else 'manual',
                                'base_build_s': base_build,
                                'hybrid_build_s': build_time,
                                **eval_res
                            }
                            results.append(row)
                            print(f"  n_probe={n_probe} R@10={row['recall_at_10']:.4f} R@100={row['recall_at_100']:.4f} {row['avg_query_ms']:.2f}ms variant={variant_tag}")
    return results


def main():  # pragma: no cover
    required = ['sift/sift_base.fvecs', 'sift/sift_query.fvecs', 'sift/sift_groundtruth.ivecs']
    for p in required:
        if not os.path.exists(p):
            print(f"❌ Missing required file: {p}")
            return 1

    # Environment config
    small_base = int(os.getenv('SMALL_BASE', '1000'))
    small_queries = int(os.getenv('SMALL_QUERIES', '100'))
    large_base = int(os.getenv('LARGE_BASE', '100000'))
    large_queries = int(os.getenv('LARGE_QUERIES', '50'))
    k_children_list = parse_list('K_CHILDREN', '5000,10000')
    auto_k_children = os.getenv('AUTO_K_CHILDREN', '0') in ('1','true','True')
    diversify_values = parse_list('DIV_MAX_ASSIGNMENTS', '0,3')
    repair_values = parse_list('REPAIR_MIN_ASSIGNMENTS', '0')
    parent_levels = parse_list('PARENT_LEVELS', '1,2')
    n_probe_list = parse_list('N_PROBE', '1,2,3,5,10,20')
    extra_probe_fracs = parse_float_list('N_PROBE_EXTRA_FRACTIONS', '0.6,0.75,0.9')
    include_full_probe = os.getenv('N_PROBE_INCLUDE_FULL', '0') in ('1','true','True')
    methods = [m.strip() for m in os.getenv('METHODS', 'approx').split(',') if m.strip()]
    k_gt = int(os.getenv('K_GT', '100'))
    k_eval = int(os.getenv('K_EVAL', '100'))

    random.seed(42)
    np.random.seed(42)

    print("=== Hybrid HNSW Benchmark (1K & 100K) ===")
    print(f"Small scale: base={small_base} queries={small_queries}")
    print(f"Large scale: base={large_base} queries={large_queries}")
    if auto_k_children and not k_children_list:
        print("k_children: AUTO (scale-dependent)")
    else:
        print(f"k_children (manual): {k_children_list}")
    print(f"Methods: {methods} parent_levels={parent_levels} n_probe={n_probe_list}")
    print(f"Extra n_probe fractions: {extra_probe_fracs} include_full={include_full_probe}")
    print(f"Diversify values: {diversify_values}  Repair values: {repair_values}")

    # Load query vectors once (10K queries total in file)
    full_queries = read_fvecs('sift/sift_query.fvecs')
    if len(full_queries) < max(small_queries, large_queries):
        raise ValueError("Not enough query vectors in file for requested counts")

    # SMALL SCALE DATA
    small_base_vecs = read_fvecs('sift/sift_base.fvecs', small_base)
    small_query_vecs = full_queries[:small_queries]
    small_gt = compute_bruteforce_gt(small_base_vecs, small_query_vecs, k_gt)

    # LARGE SCALE DATA (only read needed portion; for 1M we read all)
    large_base_vecs = read_fvecs('sift/sift_base.fvecs', large_base)
    large_query_vecs = full_queries[:large_queries]
    
    # Important: SIFT ground truth is only valid for the full 1M dataset!
    # If we're using a subset, we must compute ground truth from our subset
    if large_base >= 1000000:
        # Full dataset - use official SIFT ground truth
        sift_gt_full = load_sift_gt(k_gt)  # 10K x k_gt
        large_gt = sift_gt_full[:large_queries]
        print(f"Using official SIFT ground truth for full dataset ({large_base} vectors)")
    else:
        # Subset dataset - compute ground truth from our subset
        print(f"Computing ground truth for subset dataset ({large_base} vectors)")
        large_gt = compute_bruteforce_gt(large_base_vecs, large_query_vecs, k_gt)
        print(f"Ground truth computation completed")

    all_results: List[Dict[str, Any]] = []
    # Run small scale with full method list (brute allowed)
    all_results += run_scale('1K', small_base_vecs, small_query_vecs, small_gt, k_children_list, parent_levels, methods, n_probe_list, k_eval, k_gt, auto_k=auto_k_children, diversify_values=diversify_values, repair_values=repair_values, extra_probe_fracs=extra_probe_fracs, include_full_probe=include_full_probe)

    # For large scale, discourage brute to save time
    large_methods = [m for m in methods if m == 'approx'] or ['approx']
    all_results += run_scale('100K', large_base_vecs, large_query_vecs, large_gt, k_children_list, parent_levels, large_methods, n_probe_list, k_eval, k_gt, limit_queries=large_queries, auto_k=auto_k_children, diversify_values=diversify_values, repair_values=repair_values, extra_probe_fracs=extra_probe_fracs, include_full_probe=include_full_probe)

    # Aggregate best configs
    def best_by(metric: str, scale: str):
        subset = [r for r in all_results if r['scale'] == scale]
        return max(subset, key=lambda x: x[metric]) if subset else None

    summary = {
        'best_1K_recall10': best_by('recall_at_10', '1K'),
        'best_100K_recall10': best_by('recall_at_10', '100K'),
        'best_1K_speed': min([r for r in all_results if r['scale']=='1K'], key=lambda x: x['avg_query_ms']) if any(r['scale']=='1K' for r in all_results) else None,
        'best_100K_speed': min([r for r in all_results if r['scale']=='100K'], key=lambda x: x['avg_query_ms']) if any(r['scale']=='100K' for r in all_results) else None,
    }

    out = {
        'config': {
            'small_base': small_base,
            'small_queries': small_queries,
            'large_base': large_base,
            'large_queries': large_queries,
            'k_children_list_input': k_children_list,
            'auto_k_children': auto_k_children,
            'diversify_values': diversify_values,
            'repair_values': repair_values,
            'parent_levels': parent_levels,
            'methods_requested': methods,
            'n_probe_list': n_probe_list,
            'extra_probe_fracs': extra_probe_fracs,
            'include_full_probe': include_full_probe,
            'k_gt': k_gt,
            'k_eval': k_eval,
        },
        'results': all_results,
        'summary': summary,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('hybrid_benchmark_1k_1m.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\n✅ Written hybrid_benchmark_1k_1m.json")

    # Pretty summary print
    def pr(label, row):
        if not row:
            print(f"{label}: -")
            return
        print(f"{label}: method={row['method']} level={row['parent_level']} k_children={row['k_children']} n_probe={row['n_probe']} R@10={row['recall_at_10']:.4f} time={row['avg_query_ms']:.2f}ms")
    print("\n=== SUMMARY ===")
    pr('Best 1K Recall', summary['best_1K_recall10'])
    pr('Best 100K Recall', summary['best_100K_recall10'])
    pr('Fastest 1K', summary['best_1K_speed'])
    pr('Fastest 100K', summary['best_100K_speed'])
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
