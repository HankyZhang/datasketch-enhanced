"""Aggregate & Compare Benchmark Results

Combines:
  - Baseline HNSW (baseline_5k.csv)
  - Unoptimized Hybrid (unoptimized_5k.csv)
  - Optimized Hybrid (hybrid_optimized_5k.csv)

Outputs:
  - comparison_summary.csv : unified rows with method + key metrics
  - prints top lines per method and recall vs latency pairs

Assumes CSV schemas already produced by existing scripts.
"""
from __future__ import annotations
import csv, os, math
from typing import List, Dict

BASELINE_FILE = 'baseline_5k.csv'
UNOPT_FILE = 'unoptimized_5k.csv'
OPT_FILE = 'hybrid_optimized_5k.csv'  # legacy single optimized file
OUT_FILE = 'comparison_summary.csv'

# Additional optional optimized files automatically merged if present
OPT_ADDITIONAL = [
    'hybrid_optimized_5k_full.csv',  # full 500-query efc=400 run
    'hybrid_optimized_all.csv',      # potential manually merged superset
]


def load_csv(path: str) -> List[Dict[str,str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        return list(r)


def try_float(v: str):
    try:
        return float(v)
    except Exception:
        return math.nan


def main():
    base_rows = load_csv(BASELINE_FILE)
    unopt_rows = load_csv(UNOPT_FILE)
    # Load all optimized variants (merge & dedupe later)
    opt_rows_primary = load_csv(OPT_FILE)
    opt_variant_rows = []
    for extra in OPT_ADDITIONAL:
        rows_extra = load_csv(extra)
        if rows_extra:
            print(f"Including optimized variant file: {extra} ({len(rows_extra)} rows)")
            opt_variant_rows.extend(rows_extra)
    opt_rows = opt_rows_primary + opt_variant_rows

    if not base_rows:
        print(f"Missing baseline file {BASELINE_FILE}; abort.")
        return
    if not unopt_rows:
        print(f"Missing unoptimized hybrid file {UNOPT_FILE}; abort.")
        return
    if not opt_rows:
        print("No optimized hybrid files found; abort.")
        return

    out_header = [
        'method','ef_construction','search_width','k_children','n_probe','diversify_max','repair_min',
        'parent_count','coverage_fraction','recall@k','avg_query_time_ms','candidate_size','notes'
    ]
    seen = set()
    with open(OUT_FILE,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(out_header)

        # Baseline
        for row in base_rows:
            r = [
                'baseline',
                row.get('ef_construction',''),
                row.get('ef_search',''),
                '', '', '', '', '', '',
                row.get('recall@k',''),
                f"{try_float(row.get('avg_query_time','0'))*1000:.3f}",
                '',
                ''
            ]
            w.writerow(r)
        # Unoptimized hybrid (dedupe identical configs)
        for row in unopt_rows:
            key = (row.get('ef_construction'), row.get('k_children'), row.get('n_probe'))
            if key in seen:
                continue
            seen.add(key)
            r = [
                'hybrid_unopt',
                row.get('ef_construction',''),
                '',
                row.get('k_children',''),
                row.get('n_probe',''),
                '', '',
                row.get('parent_count',''),
                row.get('coverage_fraction',''),
                row.get('recall@k',''),
                f"{try_float(row.get('avg_query_time','0'))*1000:.3f}",
                row.get('avg_candidate_size',''),
                ''
            ]
            w.writerow(r)
        # Optimized hybrid (dedupe by key including n_probe & diversify params & full_parent flag)
        seen_opt = set()
        for row in opt_rows:
            key = (
                row.get('ef_construction',''),
                row.get('k_children',''),
                row.get('n_probe',''),
                row.get('diversify_max',''),
                row.get('repair_min',''),
                row.get('parent_count',''),
                row.get('full_parent_probe',''),
            )
            if key in seen_opt:
                continue
            seen_opt.add(key)
            r = [
                'hybrid_opt',
                row.get('ef_construction',''),
                '',
                row.get('k_children',''),
                row.get('n_probe',''),
                row.get('diversify_max',''),
                row.get('repair_min',''),
                row.get('parent_count',''),
                row.get('coverage_fraction',''),
                row.get('recall@k',''),
                f"{try_float(row.get('avg_query_time','0'))*1000:.3f}",
                row.get('avg_candidate_size',''),
                'full_parent' if row.get('full_parent_probe')=='1' else ''
            ]
            w.writerow(r)

    print(f"Wrote {OUT_FILE}")
    # Quick console insight: top recall per method (excluding full_parent flag for opt)
    def max_recall(rows, method_key, filter_fn=lambda r: True):
        best = None; best_r = -1
        for r in rows:
            if not filter_fn(r):
                continue
            rec = try_float(r.get('recall@k',''))
            if rec>best_r:
                best_r=rec; best=r
        return best_r, best

    # Build index for opt rows referencing full_parent flag
    opt_best_partial = max_recall(opt_rows,'hybrid_opt', lambda r: r.get('full_parent_probe')=='0')
    opt_best_full = max_recall(opt_rows,'hybrid_opt', lambda r: r.get('full_parent_probe')=='1')
    print("Optimized hybrid best (partial n_probe):", opt_best_partial[0])
    print("Optimized hybrid best (full parent):", opt_best_full[0])

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
