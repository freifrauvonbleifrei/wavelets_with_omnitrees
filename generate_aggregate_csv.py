#!/usr/bin/env python
"""Generate CSV data for coefficient and storage boxplots.

Reads wavelet_stats.csv and writes per-level raw-values CSVs plus an aggregate
CSV with medians/means, for two groups of methods:

Coefficient counts:
  - coeff_canonical.csv
  - coeff_downsplit.csv
  - coeff_openvdb.csv
  - coeff_aggregate.csv
  - extremal_vdb_over_ds.csv

Storage (bytes):
  - storage_openvdb.csv       (vdb_file_bytes)
  - storage_can_raw.csv       (descriptor + coefficients raw)
  - storage_can_blosc2.csv    (descriptor + coefficients blosc2)
  - storage_ds_raw.csv
  - storage_ds_blosc2.csv
  - storage_aggregate.csv

Slopes (log2(N(L+1)/N(L)) per thingi, per adjacent level pair and method):
  - slope_aggregate.csv       (metric, method, lo, hi, n, mean, median, min, max, std)

Usage:
    python generate_aggregate_csv.py [--csv wavelet_stats.csv] [--outdir .]
"""
import argparse
import os

import numpy as np
import pandas as pd


COEFF_METHODS = [
    ("can", "can_boxes", "coeff_canonical.csv"),
    ("ds", "ds_boxes", "coeff_downsplit.csv"),
    ("vdb", "vdb_total_coefficients", "coeff_openvdb.csv"),
]

STORAGE_METHODS = [
    ("vdb", "vdb_file_bytes", "storage_openvdb.csv"),
    ("can_raw", "can_raw_total_bytes", "storage_can_raw.csv"),
    ("can_b2", "can_blosc2_total_bytes", "storage_can_blosc2.csv"),
    ("ds_raw", "ds_raw_total_bytes", "storage_ds_raw.csv"),
    ("ds_b2", "ds_blosc2_total_bytes", "storage_ds_blosc2.csv"),
]

SLOPE_METRICS = {
    "boxes": {"can": "can_boxes", "ds": "ds_boxes", "vdb": "vdb_total_coefficients"},
    "raw": {
        "can": "can_raw_total_bytes",
        "ds": "ds_raw_total_bytes",
        "vdb": "vdb_file_bytes",
    },
    "blosc2": {
        "can": "can_blosc2_total_bytes",
        "ds": "ds_blosc2_total_bytes",
        "vdb": "vdb_file_bytes",
    },
}


def load_and_filter(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["level"] >= 2].copy()
    return df


def write_raw_csv(df, column, filename, outdir, integer=False):
    """Write one CSV with one column per level, containing raw values.
    Rows are aligned by ``thingi_id``: the same row in every column refers to
    the same thingi.
    """
    pivot = df.pivot_table(
        index="thingi_id", columns="level", values=column, aggfunc="first"
    ).sort_index()
    pivot.columns = [f"l{int(c)}" for c in pivot.columns]
    path = os.path.join(outdir, filename)
    if integer:
        kwargs["float_format"] = "%.0f"
    pivot.to_csv(path, **kwargs)
    return path


def write_aggregate_csv(df, methods, filename, outdir):
    levels = sorted(df["level"].unique())
    rows = []
    for level in levels:
        sub = df[df["level"] == level]
        row = {"level": level, "n": len(sub)}
        for key, col, _ in methods:
            s = sub[col].dropna()
            if not s.empty:
                row[f"{key}_mean"] = s.mean()
                row[f"{key}_median"] = s.median()
        rows.append(row)
    path = os.path.join(outdir, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_extremal_ratios(df, outdir, n_extremes=10):
    """Write CSV identifying thingies with largest/smallest vdb/ds ratio per level."""
    levels = sorted(df["level"].unique())
    all_rows = []
    for level in levels:
        sub = df[
            (df["level"] == level)
            & (df["vdb_total_coefficients"] > 0)
            & (df["ds_boxes"] > 0)
        ].copy()
        if sub.empty:
            continue
        sub["vdb_over_ds"] = sub["vdb_total_coefficients"] / sub["ds_boxes"]
        sub = sub.sort_values("vdb_over_ds")

        for _, row in sub.head(n_extremes).iterrows():
            all_rows.append(
                {
                    "level": level,
                    "thingi_id": int(row["thingi_id"]),
                    "ds_boxes": int(row["ds_boxes"]),
                    "can_boxes": int(row["can_boxes"]),
                    "vdb_total_coefficients": int(row["vdb_total_coefficients"]),
                    "vdb_over_ds": row["vdb_over_ds"],
                    "extremal": "smallest",
                }
            )
        for _, row in sub.tail(n_extremes).iterrows():
            all_rows.append(
                {
                    "level": level,
                    "thingi_id": int(row["thingi_id"]),
                    "ds_boxes": int(row["ds_boxes"]),
                    "can_boxes": int(row["can_boxes"]),
                    "vdb_total_coefficients": int(row["vdb_total_coefficients"]),
                    "vdb_over_ds": row["vdb_over_ds"],
                    "extremal": "largest",
                }
            )

    path = os.path.join(outdir, "extremal_vdb_over_ds.csv")
    pd.DataFrame(all_rows).to_csv(path, index=False)
    return path


def write_slope_aggregate(df, outdir):
    """Log2 slopes between adjacent levels, for each method & metric.

    this computes the rate of the aggregate statistic:
        rate_mean   = log2(mean(N(hi))   / mean(N(lo)))
        rate_median = log2(median(N(hi)) / median(N(lo)))

    so ``rate_mean`` is consistent with the means in ``{coeff,storage}_aggregate.csv``.
    Each level's statistic is taken over all thingies present at that level (no
    common-subset filter), matching the aggregate CSVs. Rates are NaN when the
    lower-level statistic is zero. 3 = full-grid 3D scaling, 2 = surface-like.
    """
    levels = sorted(df.level.unique())
    rows = []
    for metric, methods in SLOPE_METRICS.items():
        for lo, hi in zip(levels[:-1], levels[1:]):
            a_df = df[df.level == lo]
            b_df = df[df.level == hi]
            for key, col in methods.items():
                a, b = a_df[col].dropna(), b_df[col].dropna()
                if a.empty or b.empty:
                    continue
                mean_lo, mean_hi = a.mean(), b.mean()
                med_lo, med_hi = a.median(), b.median()
                rows.append(
                    {
                        "metric": metric,
                        "method": key,
                        "lo": int(lo),
                        "hi": int(hi),
                        "n_lo": int(len(a)),
                        "n_hi": int(len(b)),
                        "mean_lo": mean_lo,
                        "mean_hi": mean_hi,
                        "median_lo": med_lo,
                        "median_hi": med_hi,
                        "rate_mean": (
                            np.log2(mean_hi / mean_lo) if mean_lo > 0 else np.nan
                        ),
                        "rate_median": (
                            np.log2(med_hi / med_lo) if med_lo > 0 else np.nan
                        ),
                    }
                )
    path = os.path.join(outdir, "slope_aggregate.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="wavelet_stats.csv")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_and_filter(args.csv)
    print(
        f"Loaded {len(df)} rows ({df['thingi_id'].nunique()} thingies, "
        f"levels {sorted(df['level'].unique())})"
    )

    for _, col, fname in COEFF_METHODS:
        print(f"Wrote {write_raw_csv(df, col, fname, args.outdir, integer=True)}")
    for _, col, fname in STORAGE_METHODS:
        print(f"Wrote {write_raw_csv(df, col, fname, args.outdir, integer=True)}")

    print(
        f"Wrote {write_aggregate_csv(df, COEFF_METHODS, 'coeff_aggregate.csv', args.outdir)}"
    )
    print(
        f"Wrote {write_aggregate_csv(df, STORAGE_METHODS, 'storage_aggregate.csv', args.outdir)}"
    )
    print(f"Wrote {write_extremal_ratios(df, args.outdir)}")
    print(f"Wrote {write_slope_aggregate(df, args.outdir)}")


if __name__ == "__main__":
    main()
