#!/usr/bin/env python3
"""
6_inferential_tests.py — Paired t-tests (SRTM vs 3DEP) and ANOVA over CORR levels

Purpose
-------
Run inferential statistics on metrics computed by your accuracy pipeline (the file
`<AREA>/results/accuracy_metrics.csv` produced by 5_accuracy_assessment.py):

1) Paired t-test (SRTM vs 3DEP) per CORR level:
   - Per area.
   - Across all processed areas combined.
   The pairing aligns rows by (pair_ref, pair_sec, corr, replicate, n_cal), i.e.,
   the same shared calibration/validation split.

2) One-way ANOVA across CORR ∈ {RAW, TROPO, IONO, TROPO_IONO}, *within a DEM*:
   - Per area (two ANOVAs: one for SRTM, one for 3DEP if present).
   - Across all areas combined (same two ANOVAs).

You can choose which metric to test (default: 'bias_cm'), e.g. 'rmse_cm', 'mae_cm',
or 'sigma_e_cm'. Tests use only rows with method == 'least_squares' (the direct
gauge-vs-InSAR fit).

Inputs (from existing CSVs)
---------------------------
- Per-area metrics CSVs that your main script already writes:
  <areas_root>/<AREA>/results/accuracy_metrics.csv

  Expected columns (subset):
    area, pair_ref, pair_sec, dem, corr, method, replicate, n_cal,
    bias_cm, rmse_cm, mae_cm, sigma_e_cm, r, a_gain, b_offset_cm, ...

Outputs
-------
Per area:
  <areas_root>/<AREA>/results/inferential_tests.csv
    Columns: area, scope, test, metric, dem, corr, n, stat, df1, df2, p

All areas combined:
  <areas_root>/_reports/inferential_tests_all_areas.csv

Dependencies
------------
- numpy, pandas
- SciPy (optional): for exact p-values (t and F). If SciPy is missing,
  the script still computes t and F; t-test p-values use a normal approximation,
  and ANOVA p-values are reported as NaN.

How to run
----------
# All areas under the default root:
python 6_inferential_tests.py

# One area only:
python 6_inferential_tests.py --area ENP

# Choose metric, filter to a specific n_cal if desired:
python 6_inferential_tests.py --test-metric bias_cm --only-ncal 5

Notes
-----
- Pairing for the t-test requires both DEMs present for the same
  (pair_ref, pair_sec, corr, replicate, n_cal). Any rows that can't be paired
  are dropped for that comparison.
- ANOVA uses *all available rows* for the given DEM and metric (within an area
  or across all areas). You can optionally restrict to a single calibration size
  via --only-ncal if you want to fix the calibration level.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# --- Optional SciPy for accurate p-values ------------------------------------
HAVE_SCIPY = False
try:
    from scipy.stats import t as student_t, f as fisher_f
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# =============================================================================
# Small, well-documented statistical helpers
# =============================================================================
def paired_ttest_rel(x: np.ndarray, y: np.ndarray) -> Tuple[float, int, float]:
    """
    Paired t-test for H0: mean(x - y) == 0.

    Parameters
    ----------
    x, y : np.ndarray
        Paired samples (same length). Non-finite entries are dropped pairwise.

    Returns
    -------
    t_stat : float
        Student's t statistic for paired samples.
    df : int
        Degrees of freedom (n-1).
    p_two_sided : float
        Two-sided p-value. If SciPy is not installed, a normal approximation
        is used for p; if n < 2 or variance is 0, returns NaN p.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    n = int(d.size)
    if n < 2:
        return float("nan"), n - 1, float("nan")
    sd = float(d.std(ddof=1))
    if sd <= 0.0:
        return float("nan"), n - 1, float("nan")
    t = float(d.mean() / (sd / math.sqrt(n)))
    df = n - 1
    if HAVE_SCIPY:
        p = float(2.0 * student_t.sf(abs(t), df))
    else:
        # Normal approximation to t for large df
        p = float(math.erfc(abs(t) / math.sqrt(2.0)))
    return t, df, p


def anova_oneway(groups: List[np.ndarray]) -> Tuple[float, int, int, float]:
    """
    One-way ANOVA across k groups: tests if all group means are equal.

    Parameters
    ----------
    groups : list of np.ndarray
        Each array is one group's observations; non-finite values are dropped.

    Returns
    -------
    F : float
        F statistic.
    df_between : int
        Degrees of freedom between groups (k - 1).
    df_within : int
        Degrees of freedom within groups (N_total - k).
    p_value : float
        Upper-tail p-value (P(F_{dfb, dfw} >= F)). NaN if SciPy missing.
    """
    clean = [np.asarray(g, float)[np.isfinite(g)] for g in groups if len(g)]
    clean = [g for g in clean if g.size > 0]
    k = len(clean)
    n_total = sum(g.size for g in clean)
    if k < 2 or n_total <= k:
        return float("nan"), k - 1, n_total - k, float("nan")

    means = [float(g.mean()) for g in clean]
    grand = float(sum(g.sum() for g in clean) / n_total)

    ss_between = sum(g.size * (m - grand) ** 2 for g, m in zip(clean, means))
    ss_within = sum(float(((g - m) ** 2).sum()) for g, m in zip(clean, means))

    dfb = k - 1
    dfw = n_total - k
    if dfb <= 0 or dfw <= 0 or ss_within <= 0:
        return float("nan"), dfb, dfw, float("nan")

    msb = ss_between / dfb
    msw = ss_within / dfw
    F = float(msb / msw)

    if HAVE_SCIPY:
        p = float(fisher_f.sf(F, dfb, dfw))
    else:
        p = float("nan")
    return F, dfb, dfw, p


# =============================================================================
# I/O + test runners
# =============================================================================
EXPECTED_METRICS = {"bias_cm", "rmse_cm", "mae_cm", "sigma_e_cm"}

def load_area_metrics(area_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load one area's metrics CSV, if present.

    Parameters
    ----------
    area_dir : Path
        <areas_root>/<AREA> directory.

    Returns
    -------
    DataFrame or None
        Contents of <AREA>/results/accuracy_metrics.csv, or None if missing.
    """
    p = area_dir / "results" / "accuracy_metrics.csv"
    if not p.exists():
        print(f"⏭️  Missing metrics: {p}")
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"⏭️  Failed to read {p}: {e}")
        return None
    return df


def ttests_per_area(df: pd.DataFrame, area_name: str, metric: str,
                    only_ncal: Optional[int] = None) -> pd.DataFrame:
    """
    Run paired t-tests (SRTM vs 3DEP) per CORR level for a single area.

    Parameters
    ----------
    df : DataFrame
        Area metrics (as read from accuracy_metrics.csv).
    area_name : str
    metric : str
        One of EXPECTED_METRICS.
    only_ncal : int or None
        If provided, restrict rows to this calibration size (n_cal == value).

    Returns
    -------
    DataFrame
        Rows of t-test results, one per CORR level present.
    """
    # Keep least_squares only; it reflects gauge-vs-InSAR accuracy.
    data = df.copy()
    data = data[data["method"] == "least_squares"]
    if only_ncal is not None:
        data = data[data["n_cal"] == int(only_ncal)]

    if metric not in data.columns:
        print(f"  ⚠️  Metric '{metric}' missing in area {area_name}; skipping t-tests.")
        return pd.DataFrame(columns=["area","scope","test","metric","dem","corr","n","stat","df1","df2","p"])

    data = data[np.isfinite(data[metric])]

    # Align SRTM and 3DEP rows for a *paired* comparison within each CORR.
    key = ["pair_ref", "pair_sec", "corr", "replicate", "n_cal"]
    srtm = data[data["dem"] == "SRTM"][key + [metric]].rename(columns={metric: "x"})
    d3dp = data[data["dem"] == "3DEP"][key + [metric]].rename(columns={metric: "y"})
    merged = pd.merge(srtm, d3dp, on=key, how="inner")

    out_rows = []
    if not merged.empty:
        for corr in sorted(merged["corr"].unique()):
            sub = merged[merged["corr"] == corr]
            t, df_t, p = paired_ttest_rel(sub["x"].to_numpy(), sub["y"].to_numpy())
            out_rows.append({
                "area": area_name, "scope": "per-area", "test": "ttest_rel_SRTM_vs_3DEP",
                "metric": metric, "dem": "paired", "corr": corr,
                "n": int(sub.shape[0]), "stat": float(t), "df1": int(df_t), "df2": np.nan, "p": float(p),
            })
    return pd.DataFrame(out_rows)


def anova_per_area(df: pd.DataFrame, area_name: str, metric: str,
                   only_ncal: Optional[int] = None) -> pd.DataFrame:
    """
    Run one-way ANOVA across CORR levels per DEM for a single area.

    Parameters
    ----------
    df : DataFrame
    area_name : str
    metric : str
        One of EXPECTED_METRICS.
    only_ncal : int or None
        Restrict to one calibration size (optional).

    Returns
    -------
    DataFrame
        Up to two rows (SRTM + 3DEP).
    """
    data = df.copy()
    data = data[data["method"] == "least_squares"]
    if only_ncal is not None:
        data = data[data["n_cal"] == int(only_ncal)]
    if metric not in data.columns:
        return pd.DataFrame(columns=["area","scope","test","metric","dem","corr","n","stat","df1","df2","p"])

    data = data[np.isfinite(data[metric])]

    rows = []
    corr_levels = ["RAW", "TROPO", "IONO", "TROPO_IONO"]

    for dem in ("SRTM", "3DEP"):
        sub = data[data["dem"] == dem]
        if sub.empty:
            continue
        groups = [sub[sub["corr"] == c][metric].to_numpy() for c in corr_levels]
        F, dfb, dfw, p = anova_oneway(groups)
        n_tot = int(sum(len(g) for g in groups))
        rows.append({
            "area": area_name, "scope": "per-area", "test": "anova_corr_levels",
            "metric": metric, "dem": dem, "corr": "|".join(corr_levels),
            "n": n_tot, "stat": float(F), "df1": int(dfb), "df2": int(dfw), "p": float(p),
        })
    return pd.DataFrame(rows)


def ttests_all_areas(df_all: pd.DataFrame, metric: str,
                     only_ncal: Optional[int] = None) -> pd.DataFrame:
    """
    Combined paired t-tests across ALL areas, per CORR level.

    Parameters
    ----------
    df_all : DataFrame
        Concatenated per-area metrics (least_squares rows expected).
    metric : str
    only_ncal : int or None

    Returns
    -------
    DataFrame
    """
    data = df_all.copy()
    data = data[data["method"] == "least_squares"]
    if only_ncal is not None:
        data = data[data["n_cal"] == int(only_ncal)]
    if metric not in data.columns:
        return pd.DataFrame(columns=["area","scope","test","metric","dem","corr","n","stat","df1","df2","p"])
    data = data[np.isfinite(data[metric])]

    # Pairing SRTM vs 3DEP with area included in the key (so same split & same area).
    key = ["area", "pair_ref", "pair_sec", "corr", "replicate", "n_cal"]
    srtm = data[data["dem"] == "SRTM"][key + [metric]].rename(columns={metric: "x"})
    d3dp = data[data["dem"] == "3DEP"][key + [metric]].rename(columns={metric: "y"})
    merged = pd.merge(srtm, d3dp, on=key, how="inner")

    rows = []
    if not merged.empty:
        for corr in sorted(merged["corr"].unique()):
            sub = merged[merged["corr"] == corr]
            t, df_t, p = paired_ttest_rel(sub["x"].to_numpy(), sub["y"].to_numpy())
            rows.append({
                "area": "ALL_AREAS", "scope": "all-areas", "test": "ttest_rel_SRTM_vs_3DEP",
                "metric": metric, "dem": "paired", "corr": corr,
                "n": int(sub.shape[0]), "stat": float(t), "df1": int(df_t), "df2": np.nan, "p": float(p),
            })
    return pd.DataFrame(rows)


def anova_all_areas(df_all: pd.DataFrame, metric: str,
                    only_ncal: Optional[int] = None) -> pd.DataFrame:
    """
    Combined one-way ANOVAs across CORR levels, per DEM, across ALL areas.

    Parameters
    ----------
    df_all : DataFrame
    metric : str
    only_ncal : int or None

    Returns
    -------
    DataFrame
    """
    data = df_all.copy()
    data = data[data["method"] == "least_squares"]
    if only_ncal is not None:
        data = data[data["n_cal"] == int(only_ncal)]
    if metric not in data.columns:
        return pd.DataFrame(columns=["area","scope","test","metric","dem","corr","n","stat","df1","df2","p"])
    data = data[np.isfinite(data[metric])]

    rows = []
    corr_levels = ["RAW", "TROPO", "IONO", "TROPO_IONO"]

    for dem in ("SRTM", "3DEP"):
        sub = data[data["dem"] == dem]
        if sub.empty:
            continue
        groups = [sub[sub["corr"] == c][metric].to_numpy() for c in corr_levels]
        F, dfb, dfw, p = anova_oneway(groups)
        n_tot = int(sum(len(g) for g in groups))
        rows.append({
            "area": "ALL_AREAS", "scope": "all-areas", "test": "anova_corr_levels",
            "metric": metric, "dem": dem, "corr": "|".join(corr_levels),
            "n": n_tot, "stat": float(F), "df1": int(dfb), "df2": int(dfw), "p": float(p),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Driver
# =============================================================================
def main():
    """
    CLI entry point.

    Arguments
    ---------
    --areas-root : str
        Root containing per-area subfolders (default: /mnt/DATA2/bakke326l/processing/areas)
    --area : str
        Process only this AREA (subfolder name). If omitted, all subfolders are scanned.
    --test-metric : str
        Which metric to test (default: bias_cm). Choices: bias_cm, rmse_cm, mae_cm, sigma_e_cm
    --only-ncal : int
        If provided, restrict tests to rows with this calibration size (n_cal == value).

    Behavior
    --------
    - Reads per-area metrics CSVs.
    - Writes per-area inferential tests to each area's results folder.
    - Writes a combined tests CSV under <areas_root>/_reports/.
    - Prints reminders if SciPy is missing (p-values may be NaN for ANOVA).
    """
    ap = argparse.ArgumentParser(description="Inferential tests (paired t-test & ANOVA) on accuracy metrics CSVs.")
    ap.add_argument("--areas-root", type=str, default="/mnt/DATA2/bakke326l/processing/areas",
                    help="Root folder with per-area subfolders (default: %(default)s)")
    ap.add_argument("--area", type=str,
                    help="Process only this AREA (name of a subfolder under --areas-root).")
    ap.add_argument("--test-metric", type=str, default="bias_cm",
                    choices=sorted(EXPECTED_METRICS),
                    help="Metric to use in tests (default: %(default)s)")
    ap.add_argument("--only-ncal", type=int,
                    help="If set, restrict to a specific calibration size (n_cal).")
    args = ap.parse_args()

    root = Path(args.areas_root)
    if args.area:
        targets = [root / args.area]
    else:
        targets = sorted([d for d in root.iterdir() if d.is_dir()])

    all_metrics = []
    per_area_results = []

    if not HAVE_SCIPY:
        print("ℹ️  SciPy not detected — t/F statistics reported; "
              "t-test p-values use a normal approximation; ANOVA p-values will be NaN.")

    for area_dir in targets:
        area_name = area_dir.name
        df = load_area_metrics(area_dir)
        if df is None or df.empty:
            continue

        # Save for combined tests
        all_metrics.append(df.assign(area=area_name) if "area" not in df.columns else df)

        # Per-area tests
        tt = ttests_per_area(df, area_name=area_name, metric=args.test_metric, only_ncal=args.only_ncal)
        av = anova_per_area(df, area_name=area_name, metric=args.test_metric, only_ncal=args.only_ncal)
        per_area = pd.concat([tt, av], ignore_index=True) if (not tt.empty or not av.empty) else pd.DataFrame(
            columns=["area","scope","test","metric","dem","corr","n","stat","df1","df2","p"]
        )

        # Write per-area CSV
        out_csv = area_dir / "results" / "inferential_tests.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        per_area.to_csv(out_csv, index=False)
        print(f"🧪  Wrote per-area tests: {out_csv} (rows: {len(per_area)})")
        per_area_results.append(per_area)

    # Combined tests across all processed areas
    if all_metrics:
        df_all = pd.concat(all_metrics, ignore_index=True)
        t_all = ttests_all_areas(df_all, metric=args.test_metric, only_ncal=args.only_ncal)
        a_all = anova_all_areas(df_all, metric=args.test_metric, only_ncal=args.only_ncal)
        comb = pd.concat([t_all, a_all], ignore_index=True) if (not t_all.empty or not a_all.empty) else pd.DataFrame(
            columns=["area","scope","test","metric","dem","corr","n","stat","df1","df2","p"]
        )
        reports_dir = root / "_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_comb = reports_dir / "inferential_tests_all_areas.csv"
        comb.to_csv(out_comb, index=False)
        print(f"🧪  Wrote combined tests: {out_comb} (rows: {len(comb)})")
    else:
        print("⏭️  No metrics found — nothing to test.")

if __name__ == "__main__":
    main()
