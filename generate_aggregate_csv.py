#!/usr/bin/env python
"""Generate CSV data for coefficient boxplot: canonical vs downsplit vs OpenVDB.

Reads wavelet_stats.csv and writes per-level columns of raw coefficient counts
for each method, plus an aggregate CSV with medians/means.

Output:
  - coeff_canonical.csv   (one column per level, raw box counts)
  - coeff_downsplit.csv
  - coeff_openvdb.csv
  - coeff_aggregate.csv   (median/mean per level per method)

Usage:
    python generate_coeff_ratio_plot.py [--csv wavelet_stats.csv] [--outdir .]
"""
import argparse
import os

import numpy as np
import pandas as pd


def load_and_filter(csv_path):
    df = pd.read_csv(csv_path)
    # Keep levels >= 2 where there's actual compression
    df = df[df["level"] >= 2].copy()
    return df


def write_raw_csv(df, column, filename, outdir):
    """Write one CSV with one column per level, containing raw values."""
    levels = sorted(df["level"].unique())
    max_rows = df.groupby("level").size().max()
    data = {}
    for level in levels:
        vals = df.loc[df["level"] == level, column].values
        data[f"l{level}"] = np.concatenate([vals, [np.nan] * (max_rows - len(vals))])
    path = os.path.join(outdir, filename)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def write_aggregate_csv(df, outdir):
    levels = sorted(df["level"].unique())
    rows = []
    for level in levels:
        sub = df[df["level"] == level]
        row = {"level": level, "n": len(sub)}
        for method, col in [
            ("can", "can_boxes"),
            ("ds", "ds_boxes"),
            ("vdb", "vdb_total_coefficients"),
        ]:
            s = sub[col].dropna()
            if not s.empty:
                row[f"{method}_mean"] = s.mean()
                row[f"{method}_median"] = s.median()
        rows.append(row)
    path = os.path.join(outdir, "coeff_aggregate.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV data for coefficient boxplot."
    )
    parser.add_argument(
        "--csv",
        default="wavelet_stats.csv",
        help="Path to wavelet_stats.csv (default: wavelet_stats.csv)",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory for CSVs (default: .)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_and_filter(args.csv)
    levels = sorted(df["level"].unique())
    print(
        f"Loaded {len(df)} rows ({df['thingi_id'].nunique()} thingies, "
        f"levels {list(levels)})"
    )

    for col, fname in [
        ("can_boxes", "coeff_canonical.csv"),
        ("ds_boxes", "coeff_downsplit.csv"),
        ("vdb_total_coefficients", "coeff_openvdb.csv"),
    ]:
        p = write_raw_csv(df, col, fname, args.outdir)
        print(f"Wrote {p}")

    p = write_aggregate_csv(df, args.outdir)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
