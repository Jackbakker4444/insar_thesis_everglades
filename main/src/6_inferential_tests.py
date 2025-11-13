#!/usr/bin/env python3
# =============================================================================
# Dependencies (Python libraries)
# =============================================================================
# • Standard library: pathlib, argparse
# • Third-party: numpy, pandas, scipy (stats), statsmodels (AnovaRM, multitest)
#
# Recommended install (conda-forge):
#   conda install -c conda-forge numpy pandas scipy statsmodels


"""
6_inferential_tests.py — Per-pair & area/all-areas ANOVA/t-tests for corrections and DEMs
(multi-density blocks: 60%, 45%, 30%, 15%; corrections forced to SRTM)

Purpose
-------
Given per-area metrics created by 5_accuracy_assessment_dem_corr.py, perform:

A) Corrections (two parallel assessments):
   1) WITH IDW:  RAW, IONO, TROPO, TROPO_IONO, IDW
   2) NO  IDW:  RAW, IONO, TROPO, TROPO_IONO

   For EACH of the two sets, we compute:
     - Per-pair repeated-measures ANOVA (subject = block = replicatexn_cal),
       using **all densities present** (e.g., 60%, 45%, 30%, 15%).
     - Per-pair pairwise paired t-tests (Holm adjusted).
     - Area-level repeated-measures ANOVA (subject = pair; uses per-pair means).
     - Area-level ranking (WITH IDW only): mean/SD/median/IQR, ordered best→worst,
       enriched with adjusted p-values vs next/lower and vs RAW
       (two-sided difference from RAW, plus optional one-sided "improvement").
     - All-areas (global) repeated-measures ANOVA + ranking (WITH IDW only).

   Notes:
   - **Corrections analyses use DEM=SRTM only** for LS variants. Pairs without SRTM for
     the four LS corrections are skipped. IDW rows are included for the "with_idw" set.
   - All tests (ANOVA/t-tests) use the chosen metric (default: nrmse_sd).

B) DEMs (SRTM vs 3DEP), correction FIXED but SWITCHABLE (default TROPO):
   - Per-pair paired t-test (subject = block).
   - Area-level paired t-test (subject = pair; per-pair means).
   - All-areas paired t-test (subject = area::pair).
   All three use **all blocks** where both DEMs are available within the block.

Design assumptions
------------------
- Script 5 writes accuracy rows for multiple densities (e.g., 60/40, 45/55, 30/70, 15/85),
  for each replicate and variant.
- Required columns: area, pair_ref, pair_sec, dem, corr, method, replicate, n_cal, <metric>
- Lower metric (e.g., rmse_cm, nrmse_sd) is better.

Inputs
------
Per AREA:
  <areas_root>/<AREA>/results/accuracy_metrics.csv
  Required columns: area, pair_ref, pair_sec, dem, corr, method, replicate, n_cal, <metric>

Outputs
-------
Per AREA (into <areas_root>/<AREA>/results/):
  # Corrections — WITH IDW
  corrections_per_pair_anova__with_idw.csv
  corrections_per_pair_pairwise__with_idw.csv
  corrections_area_overall_anova__with_idw.csv
  corrections_area_ranking__with_idw.csv

  # Corrections — NO IDW
  corrections_per_pair_anova__no_idw.csv
  corrections_area_overall_anova__no_idw.csv

  # DEMs (suffix includes correction used)
  dem_per_pair_ttest__<CORR>.csv
  dem_area_overall_ttest__<CORR>.csv
  dem_area_summary_table__<CORR>.csv
  dem_rank_by_area__<CORR>.csv      (RMSE-based per-area DEM rank table)

Across ALL AREAS (into <areas_root>/results/):
  corrections_all_areas_anova__with_idw.csv
  corrections_all_areas_ranking__with_idw.csv
  corrections_all_areas_anova__no_idw.csv
  corrections_all_areas_rank_rmse__with_idw.csv       (RMSE-based global corrections rank)
  corrections_best_by_area__with_idw.csv
  corrections_best_by_area__no_idw.csv
  corrections_order_by_area__with_idw.csv             (best→4th per area, RMSE means)
  dem_all_areas_ttest__<CORR>.csv
  dem_all_areas_rank__<CORR>.csv                      (RMSE-based global DEM rank)
  dem_each_area_overall_ttest__<CORR>.csv

P-values vs RAW
---------------
- For corrections, p-values vs RAW are **two-sided paired t-tests** on the chosen metric
  (e.g. nrmse_sd) with Holm correction:
    - `p_vs_raw_two_sided` (unadjusted)
    - `p_vs_raw_two_sided_holm` (Holm-adjusted)
- One-sided "improvement" p-values (H1: variant < RAW) are still computed, but you can
  ignore them in the thesis tables.

How to run
----------
# All areas, default metric nrmse_sd and DEM correction = TROPO
python 6_inferential_tests.py

# Single area
python 6_inferential_tests.py --area ENP

# Use log-RMSE instead of NRMSE_SD
python 6_inferential_tests.py --metric log_rmse_cm

# DEM comparison using IONO instead of TROPO
python 6_inferential_tests.py --dem-corr IONO
"""

from __future__ import annotations
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm  # noqa: F401
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multitest import multipletests
except Exception:
    raise SystemExit("This script requires statsmodels. Install with: pip install statsmodels")

# ------------------------------ Config / Constants --------------------------------

# Force corrections ANOVA to use this DEM
CORR_DEM_ENFORCED = "SRTM"

# Default correction to use for DEM comparisons (can be overridden via --dem-corr)
DEM_CORR_DEFAULT = "TROPO"

METHOD_IDW = "idw_dhvis"
METHOD_LS  = "least_squares"

CORR_LEVELS = ["RAW", "IONO", "TROPO", "TROPO_IONO"]
ALL_VARIANTS = CORR_LEVELS + ["IDW"]

VARIANT_SETS = {
    "with_idw": ALL_VARIANTS,      # 5 levels
    "no_idw":  CORR_LEVELS,        # 4 levels
}

DEM_CORR_CHOICES = set(CORR_LEVELS)  # correction used for DEM comparisons (IDW not applicable)

# ------------------------------ Small formatting helpers -------------------------

def _fmt_pm(mean: float, sd: float) -> str:
    """Format 'mean ± sd' for table-ready RMSE strings."""
    return f"{mean:.3f} ± {sd:.3f}" if np.isfinite(mean) and np.isfinite(sd) else ""

def _fmt_med_iqr(median: float, iqr: float) -> str:
    """Format 'median [IQR]' for table-ready RMSE strings."""
    return f"{median:.3f} [{iqr:.3f}]" if np.isfinite(median) and np.isfinite(iqr) else ""

# ------------------------------ Safe stats helpers --------------------------------

def ttest_rel_safe(a, b, eps: float = 1e-12):
    """Paired t-test with a stability guard; returns ``(t, p, n)``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    n = a.size
    if n < 2:
        return np.nan, np.nan, int(n)
    d = a - b
    if n > 1 and np.nanstd(d, ddof=1) < eps:
        return 0.0, 1.0, int(n)
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p), int(n)

def one_sided_from_two_sided(t_val: float, p_two: float, n: int, alternative: str = "less") -> float:
    """Convert a two-sided p-value to a one-sided p-value using the t-statistic sign."""
    if not np.isfinite(t_val) or n < 2 or not np.isfinite(p_two):
        return np.nan
    df = max(n - 1, 1)
    if alternative == "less":
        return float(stats.t.cdf(t_val, df=df))
    elif alternative == "greater":
        return float(1.0 - stats.t.cdf(t_val, df=df))
    return np.nan

# ------------------------------ Utility helpers --------------------------------

def _pair_tag(ref: str, sec: str) -> str:
    """Build a compact pair tag ``YYYYMMDD_YYYYMMDD`` from ISO dates."""
    return f"{ref.replace('-','')}_{sec.replace('-','')}"

def _parse_block_id(block_id: str) -> tuple[int, int]:
    """Extract (replicate, n_cal) from a block_id like 'rep3_n60'."""
    m = re.match(r"rep(\d+)_n(\d+)", str(block_id))
    if not m:
        return np.nan, np.nan
    return int(m.group(1)), int(m.group(2))

def _choose_dem_for_corrections(df_pair: pd.DataFrame) -> str | None:
    """Enforce DEM=SRTM for corrections comparisons."""
    dems_present = set(df_pair["dem"].dropna().astype(str).unique().tolist())
    return "SRTM" if "SRTM" in dems_present else None

def _build_blocks_for_pair(df_pair: pd.DataFrame, dem_sel: str, metric_col: str,
                           variant_list: list[str]) -> pd.DataFrame:
    """Construct long-format data for per-pair corrections ANOVA."""
    df = df_pair.copy()
    frames = []

    # LS variants (RAW/IONO/TROPO/TROPO_IONO) — keep only the selected DEM
    ls_levels = [v for v in variant_list if v != "IDW"]
    if ls_levels:
        ls_needed = df[
            (df["method"] == METHOD_LS) &
            (df["dem"] == dem_sel) &
            (df["corr"].isin(ls_levels))
        ].copy()
        if not ls_needed.empty:
            use_cols = ["area","pair_ref","pair_sec","replicate","n_cal","corr",metric_col]
            frames.append(ls_needed[use_cols].rename(columns={metric_col: "value"}))

    # IDW baseline — not DEM-filtered (IDW rows have dem='N/A')
    if "IDW" in variant_list:
        idw_needed = df[(df["method"] == METHOD_IDW)].copy()
        if not idw_needed.empty:
            use_cols = ["area","pair_ref","pair_sec","replicate","n_cal",metric_col]
            tmp = idw_needed[use_cols].rename(columns={metric_col: "value"})
            tmp = tmp.assign(corr="IDW")
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["area","pair","block_id","variant","value"])

    # Combine and label
    cat = pd.concat(frames, ignore_index=True)
    cat["pair"] = cat.apply(lambda r: _pair_tag(r["pair_ref"], r["pair_sec"]), axis=1)
    cat["variant"] = cat["corr"].astype(str)

    # Subject ID = replicate x n_cal
    cat["block_id"] = cat.apply(lambda r: f"rep{int(r['replicate'])}_n{int(r['n_cal'])}", axis=1)

    # Keep only blocks that have ALL requested variants with FINITE values
    pv = (cat.pivot_table(index=["pair","block_id"], columns="variant", values="value",
                          aggfunc="mean", observed=False)
            .reindex(columns=variant_list))
    pv = pv.replace([np.inf, -np.inf], np.nan)
    pv = pv.dropna(axis=0, how="any")

    if pv.empty:
        return pd.DataFrame(columns=["area","pair","block_id","variant","value"])

    out = (pv.stack("variant").rename("value").reset_index()
             .merge(cat[["pair","block_id","area"]].drop_duplicates(),
                    on=["pair","block_id"], how="left"))
    out["variant"] = pd.Categorical(out["variant"], categories=variant_list, ordered=True)
    return out[["area","pair","block_id","variant","value"]]


def _anova_rm_oneway(df_long: pd.DataFrame, dv: str, subject: str, within: str):
    """Run one-way repeated-measures ANOVA via ``statsmodels.AnovaRM``."""
    if df_long.empty:
        return None
    check = df_long[[subject, within]].drop_duplicates().groupby(subject, observed=False).size()
    if check.min() < 2:
        return None
    try:
        aov = AnovaRM(df_long, depvar=dv, subject=subject, within=[within]).fit()
        return aov
    except Exception:
        return None

def _pairwise_within_subject_ttests(df_long: pd.DataFrame, subject: str, within: str,
                                    dv: str, variant_list: list[str]) -> pd.DataFrame:
    """Paired t-tests across levels of ``within``, aligned by ``subject``."""
    wide = df_long.pivot_table(index=subject, columns=within, values=dv,
                               aggfunc="mean", observed=False)
    keep_cols = [c for c in variant_list if c in wide.columns]
    wide = wide[keep_cols].dropna(axis=0, how="any")
    levels = [c for c in variant_list if c in wide.columns]
    pairs = [(a, b) for i, a in enumerate(levels) for b in levels[i+1:]]
    rows, pvals = [], []
    for a, b in pairs:
        t, p, n = ttest_rel_safe(wide[a], wide[b])
        diff = float((wide[a] - wide[b]).mean()) if n >= 1 else np.nan
        rows.append({"level_a":a, "level_b":b, "t":t, "p_raw":p, "mean_diff":diff, "n":n})
        pvals.append(p)
    if rows:
        _, p_holm, _, _ = multipletests(pvals, method="holm")
        for i, r in enumerate(rows):
            r["p_holm_two_sided"] = float(p_holm[i])
    return pd.DataFrame(rows)

def _summarize_variant_means(df_long: pd.DataFrame, subject_col: str,
                             variant_list: list[str]) -> pd.DataFrame:
    """Compute mean/SD/median/IQR per variant across subjects and rank them."""
    def iqr(x):
        q = np.nanpercentile(x, [25, 75])
        return float(q[1]-q[0])
    subj_mean = df_long.pivot_table(index=subject_col, columns="variant", values="value",
                                    aggfunc="mean", observed=False)
    keep_cols = [c for c in variant_list if c in subj_mean.columns]
    subj_mean = subj_mean[keep_cols]
    long = subj_mean.reset_index().melt(id_vars=[subject_col], var_name="variant", value_name="value")
    g = long.groupby("variant", observed=False)["value"]
    out = pd.DataFrame({
        "mean": g.mean(),
        "sd": g.std(),
        "median": g.median(),
        "iqr": g.apply(iqr),
        "n_subjects": long[subject_col].nunique()
    }).reset_index()
    out["variant"] = out["variant"].astype(str)
    out = out.sort_values("mean", ascending=True).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out)+1)
    return out

def _extract_adj_p(pw_df: pd.DataFrame, a: str, b: str) -> float:
    """Fetch Holm-adjusted two-sided p-value for ``a`` vs ``b`` from a pairwise table."""
    if pw_df is None or pw_df.empty:
        return np.nan
    row = pw_df[((pw_df["level_a"]==a) & (pw_df["level_b"]==b)) |
                ((pw_df["level_b"]==a) & (pw_df["level_a"]==b))]
    return float(row["p_holm_two_sided"].iloc[0]) if not row.empty else np.nan

def _best_second_summary(df_long: pd.DataFrame, subject_col: str,
                         variant_list: list[str]):
    """Produce ranking + summary dict for the best and second-best variants."""
    if df_long.empty:
        return pd.DataFrame(), {}
    rank_tbl = _summarize_variant_means(df_long, subject_col=subject_col, variant_list=variant_list)
    if rank_tbl.empty:
        return pd.DataFrame(), {}

    pw = _pairwise_within_subject_ttests(df_long, subject=subject_col, within="variant",
                                         dv="value", variant_list=variant_list)

    best = rank_tbl.iloc[0]["variant"]
    best_mean = float(rank_tbl.iloc[0]["mean"])

    if len(rank_tbl) >= 2:
        second = rank_tbl.iloc[1]["variant"]
        second_mean = float(rank_tbl.iloc[1]["mean"])
        p_best_vs_second = _extract_adj_p(pw, str(best), str(second))
        others = [str(v) for v in rank_tbl["variant"].tolist() if v not in {best, second}]
        p_second_vs_rest = []
        p_second_vs_third = np.nan
        if others:
            for o in others:
                p_second_vs_rest.append(_extract_adj_p(pw, str(second), o))
            if len(rank_tbl) >= 3:
                third = rank_tbl.iloc[2]["variant"]
                p_second_vs_third = _extract_adj_p(pw, str(second), str(third))
            p_second_vs_rest_min = float(np.nanmin(p_second_vs_rest)) if len(p_second_vs_rest) else np.nan
        else:
            p_second_vs_rest_min = np.nan
            p_second_vs_third = np.nan
    else:
        second = None
        second_mean = np.nan
        p_best_vs_second = np.nan
        p_second_vs_rest_min = np.nan
        p_second_vs_third = np.nan

    summary = {
        "best_variant": str(best),
        "best_mean": best_mean,
        "second_variant": (str(second) if second is not None else ""),
        "second_mean": second_mean,
        "p_best_vs_second_adj_two_sided": float(p_best_vs_second),
        "p_second_vs_rest_min_adj_two_sided": float(p_second_vs_rest_min),
        "p_second_vs_third_adj_two_sided": float(p_second_vs_third),
        "n_subjects": int(df_long[subject_col].nunique()),
    }
    return rank_tbl, summary

def _vs_raw_table(df_long: pd.DataFrame, subject_col: str,
                  variant_list: list[str]) -> pd.DataFrame:
    """
    Build a p-value table for each correction vs RAW across subjects.

    Tests are based on the analysis metric (e.g., nrmse_sd) and are:
      - Two-sided paired t-tests vs RAW  -> p_two_sided
      - Holm-adjusted two-sided          -> p_two_sided_holm

    One-sided 'improvement' p-values (H1: variant < RAW) are still provided but
    no longer used in the rank tables by default.
    """
    if df_long.empty or "RAW" not in variant_list:
        return pd.DataFrame()

    wide = df_long.pivot_table(index=subject_col, columns="variant", values="value",
                               aggfunc="mean", observed=False)
    if "RAW" not in wide.columns:
        return pd.DataFrame()

    rows = []
    p_two_list = []
    p_one_list = []

    variants = [v for v in variant_list if v != "RAW" and v in wide.columns]
    for v in variants:
        w = wide[["RAW", v]].dropna(axis=0, how="any")
        if w.shape[0] < 2:
            rows.append({"variant": v, "n_subjects": int(w.shape[0])})
            p_two_list.append(np.nan)
            p_one_list.append(np.nan)
            continue

        # diff = variant - RAW on the chosen metric (e.g., nrmse_sd)
        t, p_two, n = ttest_rel_safe(w[v], w["RAW"])
        mean_diff = float((w[v] - w["RAW"]).mean()) if n >= 1 else np.nan

        # One-sided improvement (variant < RAW) — kept for completeness
        p_one_sided = one_sided_from_two_sided(t, p_two, n, alternative="less")

        rows.append({
            "variant": v,
            "n_subjects": int(n),
            "mean_variant": float(w[v].mean()),
            "mean_raw": float(w["RAW"].mean()),
            "mean_diff_variant_minus_raw": mean_diff,
            "t": float(t),
            "p_two_sided": float(p_two),
            "p_one_sided_improve": float(p_one_sided),
        })
        p_two_list.append(p_two)
        p_one_list.append(p_one_sided)

    if rows:
        # Holm adjustment for TWO-SIDED p-values (difference from RAW)
        finite_two = [p for p in p_two_list if np.isfinite(p)]
        if finite_two:
            _, p_two_holm, _, _ = multipletests(finite_two, method="holm")
            j = 0
            for r in rows:
                if np.isfinite(r.get("p_two_sided", np.nan)):
                    r["p_two_sided_holm"] = float(p_two_holm[j]); j += 1
                else:
                    r["p_two_sided_holm"] = np.nan

        # Optional Holm for one-sided improvement (kept, but not used anymore)
        finite_one = [p for p in p_one_list if np.isfinite(p)]
        if finite_one:
            _, p_one_holm, _, _ = multipletests(finite_one, method="holm")
            j = 0
            for r in rows:
                if np.isfinite(r.get("p_one_sided_improve", np.nan)):
                    r["p_one_sided_improve_holm"] = float(p_one_holm[j]); j += 1
                else:
                    r["p_one_sided_improve_holm"] = np.nan

    return pd.DataFrame(rows)

# -------- Enriched ranking (adds p vs next/lower and vs RAW columns) --------

def _enriched_ranking(df_long: pd.DataFrame, subject_col: str,
                      variant_list: list[str]) -> pd.DataFrame:
    """Ranking + extra p-value columns (two-sided vs RAW, plus optional one-sided-improve)."""
    rank_tbl = _summarize_variant_means(df_long, subject_col=subject_col, variant_list=variant_list)
    if rank_tbl.empty:
        return rank_tbl

    pw = _pairwise_within_subject_ttests(df_long, subject=subject_col, within="variant",
                                         dv="value", variant_list=variant_list)
    vsraw = _vs_raw_table(df_long, subject_col=subject_col, variant_list=variant_list)
    vsraw = vsraw.set_index("variant") if not vsraw.empty else pd.DataFrame()

    # Compute descriptive RMSE + bias (if present)
    def _iqr_arr(a):
        a = np.asarray(a, float)
        q = np.nanpercentile(a, [25, 75]); return float(q[1]-q[0])

    rmse_sd_map, rmse_med_map, rmse_iqr_map = {}, {}, {}
    bias_mean_map, bias_sd_map = {}, {}

    if "rmse_value" in df_long.columns:
        g = df_long.groupby("variant", observed=False)["rmse_value"]
        rmse_sd_map  = g.std().to_dict()
        rmse_med_map = g.median().to_dict()
        rmse_iqr_map = {v: _iqr_arr(df_long.loc[df_long["variant"]==v, "rmse_value"].values)
                        for v in g.groups.keys()}

    if "bias_value" in df_long.columns:
        gb = df_long.groupby("variant", observed=False)["bias_value"]
        bias_mean_map = gb.mean().to_dict()
        bias_sd_map   = gb.std().to_dict()

    def _p_adj(a, b):
        return _extract_adj_p(pw, str(a), str(b))

    rows = []
    variants_ordered = rank_tbl["variant"].tolist()
    for i, v in enumerate(variants_ordered):
        p_next = np.nan
        p_min_lower = np.nan
        if i+1 < len(variants_ordered):
            next_v = variants_ordered[i+1]
            p_next = _p_adj(v, next_v)
            lower_ps = []
            for j in range(i+1, len(variants_ordered)):
                lower_ps.append(_p_adj(v, variants_ordered[j]))
            if lower_ps:
                p_min_lower = float(np.nanmin(lower_ps))

        # VS RAW: two-sided p-values (difference), plus optional one-sided
        p2_raw = np.nan
        p2_raw_holm = np.nan
        p1_raw = np.nan
        p1_raw_holm = np.nan
        if not vsraw.empty and v in vsraw.index:
            if "p_two_sided" in vsraw.columns:
                p2_raw = float(vsraw.loc[v, "p_two_sided"])
            if "p_two_sided_holm" in vsraw.columns:
                p2_raw_holm = float(vsraw.loc[v, "p_two_sided_holm"])
            if "p_one_sided_improve" in vsraw.columns:
                p1_raw = float(vsraw.loc[v, "p_one_sided_improve"])
            if "p_one_sided_improve_holm" in vsraw.columns:
                p1_raw_holm = float(vsraw.loc[v, "p_one_sided_improve_holm"])

        rows.append({
            "variant": str(v),
            "mean": float(rank_tbl.loc[i, "mean"]),
            "sd": float(rank_tbl.loc[i, "sd"]),
            "median": float(rank_tbl.loc[i, "median"]),
            "iqr": float(rank_tbl.loc[i, "iqr"]),
            "n_subjects": int(rank_tbl.loc[i, "n_subjects"]),
            "rank": int(rank_tbl.loc[i, "rank"]),
            "p_vs_next_adj_two_sided": float(p_next),
            "p_vs_lower_min_adj_two_sided": float(p_min_lower),
            # Two-sided vs RAW (difference from RAW)
            "p_vs_raw_two_sided": float(p2_raw),
            "p_vs_raw_two_sided_holm": float(p2_raw_holm),
            # Optional one-sided "improve" vs RAW
            "p_vs_raw_one_sided_improve": float(p1_raw),
            "p_vs_raw_one_sided_improve_holm": float(p1_raw_holm),
            # Added descriptors for reporting:
            "rmse_sd": float(rmse_sd_map.get(v, np.nan)),
            "rmse_median": float(rmse_med_map.get(v, np.nan)),
            "rmse_iqr": float(rmse_iqr_map.get(v, np.nan)),
            "bias_mean": float(bias_mean_map.get(v, np.nan)),
            "bias_sd": float(bias_sd_map.get(v, np.nan)),
        })

    return pd.DataFrame(rows)

# ------------------------------ Per-area runner (for one variant set) ------------------------------

def _process_corrections_for_variantset(area_name: str, df_area: pd.DataFrame,
                                        metric_col: str, variant_set_name: str,
                                        variant_list: list[str], res_dir: Path):
    """Run the corrections analysis for ONE variant set (with or without IDW).

    Uses **all blocks** (replicatexn_cal) and **SRTM** for LS corrections. Writes
    per-pair outputs and returns an area-level long table of per-pair means for
    downstream global aggregation.

    Returns
    -------
    pandas.DataFrame
        Area-level long table (columns: area, pair, variant, value, rmse_value, bias_value).
    """
    perpair_rows = []
    area_perpair_long = []
    pw_all = []

    for p in sorted(df_area["pair"].unique().tolist()):
        dfp = df_area[df_area["pair"]==p].copy()

        # Enforce SRTM for corrections LS data
        dem_sel = _choose_dem_for_corrections(dfp)
        if dem_sel is None:
            continue

        long = _build_blocks_for_pair(dfp, dem_sel, metric_col, variant_list)
        if long.empty:
            continue

        # Attach replicate/n_cal to long for RMSE/bias alignment
        _rn = long["block_id"].apply(_parse_block_id)
        long = long.assign(replicate=[r for r, _ in _rn], n_cal=[n for _, n in _rn])

        # Build RMSE (rmse_cm) and Bias (bias_cm) values for the SAME blocks/variants as in 'long'
        ls_levels = [v for v in variant_list if v != "IDW"]
        rmse_maps, bias_maps = [], []
        if ls_levels:
            ls_sub = dfp[(dfp["method"]==METHOD_LS) & (dfp["dem"]==dem_sel) & (dfp["corr"].isin(ls_levels))].copy()
            if not ls_sub.empty:
                rmse_maps.append(
                    ls_sub[["replicate","n_cal","corr","rmse_cm"]]
                    .rename(columns={"corr":"variant","rmse_cm":"rmse_value"})
                )
                if "bias_cm" in ls_sub.columns:
                    bias_maps.append(
                        ls_sub[["replicate","n_cal","corr","bias_cm"]]
                        .rename(columns={"corr":"variant","bias_cm":"bias_value"})
                    )
        if "IDW" in variant_list:
            idw_sub = dfp[(dfp["method"]==METHOD_IDW)].copy()
            if not idw_sub.empty:
                tmp = idw_sub[["replicate","n_cal","rmse_cm"]].copy()
                tmp["variant"] = "IDW"
                tmp = tmp.rename(columns={"rmse_cm":"rmse_value"})
                rmse_maps.append(tmp[["replicate","n_cal","variant","rmse_value"]])
                if "bias_cm" in idw_sub.columns:
                    tmpb = idw_sub[["replicate","n_cal","bias_cm"]].copy()
                    tmpb["variant"] = "IDW"
                    tmpb = tmpb.rename(columns={"bias_cm":"bias_value"})
                    bias_maps.append(tmpb[["replicate","n_cal","variant","bias_value"]])

        if rmse_maps:
            rmse_map = pd.concat(rmse_maps, ignore_index=True)
            long = long.merge(rmse_map, on=["replicate","n_cal","variant"], how="left")
        else:
            long["rmse_value"] = np.nan

        if bias_maps:
            bias_map = pd.concat(bias_maps, ignore_index=True)
            long = long.merge(bias_map, on=["replicate","n_cal","variant"], how="left")
        else:
            long["bias_value"] = np.nan

        # Per-pair ANOVA (subject=block)
        aov = _anova_rm_oneway(long, dv="value", subject="block_id", within="variant")
        if aov is None:
            continue

        # Pairwise comparisons (two-sided Holm)
        pw = _pairwise_within_subject_ttests(long, subject="block_id", within="variant",
                                             dv="value", variant_list=variant_list)

        # --- Attach RMSE means per variant to the pairwise table (for reporting) ---
        if pw is not None and not pw.empty:
            rmse_means_pair = long.groupby("variant", observed=False)["rmse_value"].mean()
            pw["mean_a_rmse"] = pw["level_a"].map(lambda v: float(rmse_means_pair.get(v, np.nan)))
            pw["mean_b_rmse"] = pw["level_b"].map(lambda v: float(rmse_means_pair.get(v, np.nan)))
            pw["mean_diff_rmse_a_minus_b"] = pw["mean_a_rmse"] - pw["mean_b_rmse"]

        # Means on metric (tests) and RMSE/bias (reporting)
        means = long.groupby("variant", observed=False)["value"].mean().sort_values()
        rmse_means = long.groupby("variant", observed=False)["rmse_value"].mean()
        bias_means = long.groupby("variant", observed=False)["bias_value"].mean()

        best_variant = str(means.index[0])

        long = long.assign(area=area_name)
        area_perpair_long.append(long[["area","pair","block_id","variant","value","rmse_value","bias_value"]])

        row = {
            "area": area_name,
            "pair": p,
            "variant_set": variant_set_name,
            "dem_used": dem_sel,
            "metric": metric_col,
            "n_blocks": int(long["block_id"].nunique()),
            "anova_F": float(aov.anova_table["F Value"].iloc[0]),
            "anova_df1": float(aov.anova_table["Num DF"].iloc[0]),
            "anova_df2": float(aov.anova_table["Den DF"].iloc[0]),
            "anova_p": float(aov.anova_table["Pr > F"].iloc[0]),
            "best_variant": best_variant,
            "best_mean_value": float(means.iloc[0]),
        }
        # Per-variant metric (tests) and RMSE/Bias (reporting)
        for v in variant_list:
            row[f"mean_{v}"] = float(means.get(v, np.nan))
            row[f"rmse_mean_{v}"] = float(rmse_means.get(v, np.nan))
            row[f"bias_mean_{v}"] = float(bias_means.get(v, np.nan))
        perpair_rows.append(row)

        if pw is not None and not pw.empty and variant_set_name == "with_idw":
            pw_all.append(pw.assign(area=area_name, pair=p,
                                    dem_used=dem_sel, metric=metric_col,
                                    variant_set=variant_set_name))

    # Write per-pair outputs
    tag = f"__{variant_set_name}"
    if perpair_rows:
        pd.DataFrame(perpair_rows).to_csv(res_dir / f"corrections_per_pair_anova{tag}.csv", index=False)
        # Only write pairwise table for WITH IDW (keep original behavior) — now with RMSE means attached
        if pw_all and variant_set_name == "with_idw":
            pd.concat(pw_all, ignore_index=True).to_csv(res_dir / f"corrections_per_pair_pairwise{tag}.csv", index=False)
        print(f"✅  Corrections per-pair outputs ({variant_set_name}) written in {res_dir}")
    else:
        print(f"ℹ️  No per-pair corrections ANOVA results for {area_name} ({variant_set_name}).")

    # Area-level (subject = pair) using per-pair means
    corr_area_csv   = res_dir / f"corrections_area_overall_anova{tag}.csv"
    corr_area_rank  = res_dir / f"corrections_area_ranking{tag}.csv"

    if area_perpair_long:
        long_all = pd.concat(area_perpair_long, ignore_index=True)

        # Collapse to per-pair means (metric for tests + rmse/bias for reporting)
        area_long = (long_all
                     .groupby(["area","pair","variant"], observed=False)
                     .agg(value=("value","mean"),
                          rmse_value=("rmse_value","mean"),
                          bias_value=("bias_value","mean"))
                     .reset_index())

        # ANOVA (+ explanation columns)
        aov_area = _anova_rm_oneway(area_long.rename(columns={"pair":"subject"}),
                                    dv="value", subject="subject", within="variant")
        if aov_area is not None:
            tbl = aov_area.anova_table.reset_index(drop=True).copy()
            tbl["note"] = ("One-way repeated-measures ANOVA across corrections (all densities; subjects = pairs). "
                           "Lower is better; small p implies at least one correction differs.")
            tbl["k_levels"] = int(len([v for v in variant_list if v in area_long["variant"].unique()]))
            tbl["subjects_n"] = int(area_long["pair"].nunique())

            # --- Attach RMSE & bias summaries (per variant) to the ANOVA CSV for reporting ---
            def _iqr_ser(x):
                q = np.nanpercentile(x, [25, 75]); return float(q[1]-q[0])
            rmse_g = area_long.groupby("variant", observed=False)["rmse_value"]
            bias_g = area_long.groupby("variant", observed=False)["bias_value"]
            for v in [vv for vv in variant_list if vv in area_long["variant"].unique()]:
                # RMSE descriptors
                tbl[f"rmse_mean_{v}"]   = float(rmse_g.mean().get(v, np.nan))
                tbl[f"rmse_sd_{v}"]     = float(rmse_g.std().get(v, np.nan))
                tbl[f"rmse_median_{v}"] = float(rmse_g.median().get(v, np.nan))
                tbl[f"rmse_iqr_{v}"]    = float(_iqr_ser(area_long.loc[area_long["variant"]==v, "rmse_value"]))
                tbl[f"rmse_n_pairs_{v}"]= int(area_long.loc[area_long["variant"]==v, "pair"].nunique())
                # Bias (mean + SD)
                tbl[f"bias_mean_{v}"] = float(bias_g.mean().get(v, np.nan))
                tbl[f"bias_sd_{v}"]   = float(bias_g.std().get(v, np.nan))
            tbl.to_csv(corr_area_csv, index=False)

        # Ranking (enriched) — ONLY for with_idw
        if variant_set_name == "with_idw":
            enr = _enriched_ranking(area_long.rename(columns={"pair":"subject",
                                                              "rmse_value":"rmse_value",
                                                              "bias_value":"bias_value"}),
                                    subject_col="subject", variant_list=variant_list)
            if not enr.empty:
                enr[["rank","variant","mean","sd","median","iqr","n_subjects",
                     "p_vs_next_adj_two_sided","p_vs_lower_min_adj_two_sided",
                     "p_vs_raw_two_sided","p_vs_raw_two_sided_holm",
                     "p_vs_raw_one_sided_improve","p_vs_raw_one_sided_improve_holm",
                     "rmse_sd","rmse_median","rmse_iqr","bias_mean","bias_sd"]] \
                    .to_csv(corr_area_rank, index=False)

        print(f"✅  Corrections area-level outputs ({variant_set_name}) written in {res_dir}")
        # For global aggregation (carry metric + RMSE + Bias for reporting)
        return area_long.assign(variant_set=variant_set_name)
    else:
        print(f"ℹ️  No area-level corrections data for {area_name} ({variant_set_name}).")
    return pd.DataFrame()

# ------------------------------ Whole-area process ------------------------------

def process_area(area_dir: Path, metric_col: str = "nrmse_sd",
                 dem_corr: str = DEM_CORR_DEFAULT, alpha: float = 0.05):
    """Run all requested tests for a single AREA and write outputs (all densities)."""
    area_name = area_dir.name
    res_dir = area_dir / "results"
    metrics_csv = res_dir / "accuracy_metrics.csv"
    if not metrics_csv.exists():
        print(f"⏭️  No metrics for {area_name}: {metrics_csv}")
        return {}, None

    df = pd.read_csv(metrics_csv)
    needed = {"area","pair_ref","pair_sec","dem","corr","method","replicate","n_cal",metric_col}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{metrics_csv} missing required columns: {missing}")
    df["dem"]    = df["dem"].astype(str).str.strip().str.upper()
    df["corr"]   = df["corr"].astype(str).str.strip().str.upper()
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    
    df["pair"] = df.apply(lambda r: _pair_tag(str(r["pair_ref"]), str(r["pair_sec"])), axis=1)

    corr_for_global_by_set = {}
    # --- Corrections: run twice (with IDW, without IDW)
    for set_name, variants in VARIANT_SETS.items():
        cg = _process_corrections_for_variantset(area_name, df.copy(), metric_col,
                                                 variant_set_name=set_name,
                                                 variant_list=variants,
                                                 res_dir=res_dir)
        corr_for_global_by_set[set_name] = cg

    # --- DEMs (fixed correction) — use all densities
    dem_perpair_rows = []
    dem_pairs_long = []
    corr_used = dem_corr.upper()
    if corr_used not in DEM_CORR_CHOICES:
        raise SystemExit(f"--dem-corr must be one of {sorted(DEM_CORR_CHOICES)}")

    for p in sorted(df["pair"].unique().tolist()):
        dfp = df[df["pair"]==p].copy()
        
        sub = dfp[(dfp["method"] == METHOD_LS) & (dfp["corr"] == corr_used)].copy()
        if sub.empty:
            continue

        # Keep only finite metric rows; tiny validation sets → metric = NaN are dropped here
        sub = sub[np.isfinite(sub[metric_col])].copy()
        if sub.empty:
            continue

        sub["block_id"] = sub.apply(lambda r: f"rep{int(r['replicate'])}_n{int(r['n_cal'])}", axis=1)
        have_both = sub.groupby("block_id", observed=False)["dem"].apply(lambda s: {"SRTM","3DEP"}.issubset(set(s)))
        ok_blocks = have_both[have_both].index.tolist()
        sub_ok = sub[sub["block_id"].isin(ok_blocks)].copy()
        if sub_ok.empty:
            continue

        # Metric t-test (tests)
        wide = (sub_ok.pivot_table(index="block_id", columns="dem", values=metric_col, aggfunc="mean", observed=False)
                    .reindex(columns=["SRTM","3DEP"])
                    .dropna(subset=["SRTM","3DEP"]))   
        if wide.shape[0] < 2:
            continue
        t, pval, n = ttest_rel_safe(wide["3DEP"], wide["SRTM"])
        diff = float((wide["3DEP"] - wide["SRTM"]).mean()) if n >= 1 else np.nan

        # RMSE summaries on the SAME blocks (reporting)
        wide_rmse_full = (sub_ok.pivot_table(index="block_id", columns="dem", values="rmse_cm", aggfunc="mean", observed=False)
                               .reindex(columns=["SRTM","3DEP"]))
        wide_rmse = wide_rmse_full.reindex(wide.index)  # align to metric-tested blocks

        # Bias summaries on the SAME blocks (reporting) — optional if present
        if "bias_cm" in sub_ok.columns:
            wide_bias_full = (sub_ok.pivot_table(index="block_id", columns="dem", values="bias_cm", aggfunc="mean", observed=False)
                                   .reindex(columns=["SRTM","3DEP"]))
            wide_bias = wide_bias_full.reindex(wide.index)
        else:
            wide_bias = None

        dem_perpair_rows.append({
            "area": area_name,
            "pair": dfp["pair"].iloc[0] if not dfp.empty else p,
            "metric": metric_col,
            "corr_used": corr_used,
            "n_blocks": int(wide.shape[0]),
            "mean_diff_3DEP_minus_SRTM": diff,
            "t": float(t), "p": float(pval),
            "best_dem": "3DEP" if (diff < 0) else "SRTM",
            "mean_SRTM": float(wide["SRTM"].mean()),
            "mean_3DEP": float(wide["3DEP"].mean()),
            "rmse_mean_SRTM": float(wide_rmse["SRTM"].mean()) if "SRTM" in wide_rmse else np.nan,
            "rmse_mean_3DEP": float(wide_rmse["3DEP"].mean()) if "3DEP" in wide_rmse else np.nan,
            "bias_mean_SRTM": float(wide_bias["SRTM"].mean()) if (wide_bias is not None and "SRTM" in wide_bias) else np.nan,
            "bias_mean_3DEP": float(wide_bias["3DEP"].mean()) if (wide_bias is not None and "3DEP" in wide_bias) else np.nan,
        })

        # carry both metrics to global level (pair means)
        dem_pairs_long.append(pd.DataFrame({
            "area": [area_name, area_name],
            "pair": [dfp["pair"].iloc[0] if not dfp.empty else p,
                     dfp["pair"].iloc[0] if not dfp.empty else p],
            "dem": ["SRTM","3DEP"],
            "corr_used": [corr_used, corr_used],
            "value": [float(wide["SRTM"].mean()), float(wide["3DEP"].mean())],
            "rmse_value": [float(wide_rmse["SRTM"].mean()) if "SRTM" in wide_rmse else np.nan,
                           float(wide_rmse["3DEP"].mean()) if "3DEP" in wide_rmse else np.nan],
            "bias_value": [float(wide_bias["SRTM"].mean()) if (wide_bias is not None and "SRTM" in wide_bias) else np.nan,
                           float(wide_bias["3DEP"].mean()) if (wide_bias is not None and "3DEP" in wide_bias) else np.nan]
        }))

    suffix = f"__{corr_used}"
    if dem_perpair_rows:
        pd.DataFrame(dem_perpair_rows).to_csv(res_dir / f"dem_per_pair_ttest{suffix}.csv", index=False)
        print(f"✅  DEM per-pair t-tests ({corr_used}) written: {res_dir / f'dem_per_pair_ttest{suffix}.csv'}")
    else:
        print(f"ℹ️  No per-pair DEM tests for {area_name} at correction {corr_used} (insufficient SRTM+3DEP blocks).")

    dem_area_csv = res_dir / f"dem_area_overall_ttest{suffix}.csv"
    dem_area_tbl = res_dir / f"dem_area_summary_table{suffix}.csv"
    if dem_pairs_long:
        long_dem = pd.concat(dem_pairs_long, ignore_index=True)
        wide_pairs = (long_dem.pivot_table(index="pair", columns="dem", values="value",
                                           aggfunc="mean", observed=False)
                            .reindex(columns=["SRTM","3DEP"])
                            .dropna(subset=["SRTM","3DEP"]))
        # RMSE + Bias across same pairs
        wide_pairs_rmse_full = (long_dem.pivot_table(index="pair", columns="dem", values="rmse_value",
                                                     aggfunc="mean", observed=False)
                                         .reindex(columns=["SRTM","3DEP"]))
        wide_pairs_rmse = wide_pairs_rmse_full.reindex(wide_pairs.index)

        if "bias_value" in long_dem.columns:
            wide_pairs_bias_full = (long_dem.pivot_table(index="pair", columns="dem", values="bias_value",
                                                         aggfunc="mean", observed=False)
                                             .reindex(columns=["SRTM","3DEP"]))
            wide_pairs_bias = wide_pairs_bias_full.reindex(wide_pairs.index)
        else:
            wide_pairs_bias = None

        if wide_pairs.shape[0] >= 2:
            t_area, p_area, _ = ttest_rel_safe(wide_pairs["3DEP"], wide_pairs["SRTM"])
            pd.DataFrame([{
                "area": area_name,
                "metric": metric_col,
                "corr_used": corr_used,
                "n_pairs": int(wide_pairs.shape[0]),
                "mean_diff_3DEP_minus_SRTM": float((wide_pairs["3DEP"]-wide_pairs["SRTM"]).mean()),
                "t": float(t_area), "p": float(p_area),
                "best_dem_overall": "3DEP" if (wide_pairs["3DEP"]-wide_pairs["SRTM"]).mean() < 0 else "SRTM",
                "rmse_mean_SRTM": float(wide_pairs_rmse["SRTM"].mean()) if "SRTM" in wide_pairs_rmse else np.nan,
                "rmse_mean_3DEP": float(wide_pairs_rmse["3DEP"].mean()) if "3DEP" in wide_pairs_rmse else np.nan,
                "bias_mean_SRTM": float(wide_pairs_bias["SRTM"].mean()) if (wide_pairs_bias is not None and "SRTM" in wide_pairs_bias) else np.nan,
                "bias_mean_3DEP": float(wide_pairs_bias["3DEP"].mean()) if (wide_pairs_bias is not None and "3DEP" in wide_pairs_bias) else np.nan,
            }]).to_csv(dem_area_csv, index=False)

            # Summary table per DEM with both metrics
            def _iqr_arr(a): 
                a = np.asarray(a, float)
                q = np.nanpercentile(a, [25, 75]); return float(q[1]-q[0])
            pd.DataFrame({
                "variant":["SRTM","3DEP"],
                "corr_used":[corr_used, corr_used],
                "mean":[float(wide_pairs["SRTM"].mean()), float(wide_pairs["3DEP"].mean())],
                "sd":[float(wide_pairs["SRTM"].std()), float(wide_pairs["3DEP"].std())],
                "median":[float(wide_pairs["SRTM"].median()), float(wide_pairs["3DEP"].median())],
                "iqr":[_iqr_arr(wide_pairs["SRTM"].values),
                       _iqr_arr(wide_pairs["3DEP"].values)],
                "n_pairs":[int(wide_pairs.shape[0]), int(wide_pairs.shape[0])],
                # RMSE descriptors
                "rmse_mean":[float(wide_pairs_rmse["SRTM"].mean()), float(wide_pairs_rmse["3DEP"].mean())],
                "rmse_sd":[float(wide_pairs_rmse["SRTM"].std()), float(wide_pairs_rmse["3DEP"].std())],
                "rmse_median":[float(wide_pairs_rmse["SRTM"].median()), float(wide_pairs_rmse["3DEP"].median())],
                "rmse_iqr":[_iqr_arr(wide_pairs_rmse["SRTM"].values),
                            _iqr_arr(wide_pairs_rmse["3DEP"].values)],
                # Bias descriptors (mean/SD only as requested)
                "bias_mean":[float(wide_pairs_bias["SRTM"].mean()) if wide_pairs_bias is not None else np.nan,
                             float(wide_pairs_bias["3DEP"].mean()) if wide_pairs_bias is not None else np.nan],
                "bias_sd":[float(wide_pairs_bias["SRTM"].std()) if wide_pairs_bias is not None else np.nan,
                           float(wide_pairs_bias["3DEP"].std()) if wide_pairs_bias is not None else np.nan],
            }).to_csv(dem_area_tbl, index=False)
            print(f"✅  DEM area-level t-test ({corr_used}) written: {dem_area_csv}")
            print(f"✅  DEM area summary table ({corr_used}) written: {dem_area_tbl}")

            # Per-AREA DEM rank table for RMSE (for thesis tables)
            rmse_stats = {}
            for dem_name in ["SRTM", "3DEP"]:
                vals = wide_pairs_rmse[dem_name].values
                mean = float(np.nanmean(vals))
                sd   = float(np.nanstd(vals, ddof=1))
                median = float(np.nanmedian(vals))
                q25, q75 = np.nanpercentile(vals, [25, 75])
                iqr = float(q75 - q25)
                rmse_stats[dem_name] = {
                    "mean": mean,
                    "sd": sd,
                    "median": median,
                    "iqr": iqr,
                }
            ordered = sorted(["SRTM","3DEP"], key=lambda d: rmse_stats[d]["mean"])
            dem_rank_rows = []
            for r, dem_name in enumerate(ordered, start=1):
                s = rmse_stats[dem_name]
                dem_rank_rows.append({
                    "area": area_name,
                    "rank": r,
                    "dem": dem_name,
                    "rmse_mean": s["mean"],
                    "rmse_sd": s["sd"],
                    "rmse_median": s["median"],
                    "rmse_iqr": s["iqr"],
                    "mean±SD(rmse)": _fmt_pm(s["mean"], s["sd"]),
                    "median[IQR](rmse)": _fmt_med_iqr(s["median"], s["iqr"]),
                    "p_best": float(p_area),  # same p for both in 2-level test
                })
            pd.DataFrame(dem_rank_rows).to_csv(res_dir / f"dem_rank_by_area{suffix}.csv", index=False)
        else:
            print(f"ℹ️  Not enough pairs with both DEMs in {area_name} for area-level DEM test at {corr_used}.")

    dem_for_global = None
    if dem_pairs_long:
        dem_for_global = (pd.concat(dem_pairs_long, ignore_index=True)
                          .groupby(["area","pair","dem","corr_used"], observed=False)[["value","rmse_value","bias_value"]]
                          .mean()
                          .reset_index())

    return corr_for_global_by_set, dem_for_global

# ------------------------------ CLI / Global aggregation --------------------------------

def main():
    """CLI entry point for running per-pair and area/all-areas analyses.

    Arguments
    ---------
    --areas-root : str, default "/mnt/DATA2/bakke326l/processing/areas"
        Root with per-area subfolders.
    --area : str, optional
        If provided, only process this AREA (subfolder name).
    --metric : str, default "nrmse_sd"
        Metric column to analyze (e.g., "rmse_cm", "log_rmse_cm").
    --dem-corr : str, default DEM_CORR_DEFAULT ("TROPO")
        Correction used for DEM comparisons (IDW not applicable here).
    --alpha : float, default 0.05
        Reserved for future use.
    """
    ap = argparse.ArgumentParser(description="Run per-pair and area/all-areas ANOVA/t-tests for corrections (with & without IDW) and DEMs, using all densities.")
    ap.add_argument("--areas-root", type=str, default="/mnt/DATA2/bakke326l/processing/areas",
                    help="Root containing per-area subfolders.")
    ap.add_argument("--area", type=str,
                    help="Only process this AREA (subfolder name). If omitted, process all areas.")
    ap.add_argument("--metric", type=str, default="nrmse_sd",
                    help="Metric column to analyze (e.g., rmse_cm or log_rmse_cm).")
    ap.add_argument("--dem-corr", type=str, default=DEM_CORR_DEFAULT,
                    help=f"Correction to use for DEM comparisons (default {DEM_CORR_DEFAULT}; options: {sorted(DEM_CORR_CHOICES)}).")
    ap.add_argument("--alpha", type=float, default=0.05, help="(retained for future use)")
    args = ap.parse_args()

    root = Path(args.areas_root)
    targets = [root / args.area] if args.area else sorted([p for p in root.iterdir() if p.is_dir()])

    # Accumulate for global analyses per variant set
    all_corr_by_set = {name: [] for name in VARIANT_SETS.keys()}
    all_dem  = []

    for area_dir in targets:
        cg_by_set, dg = process_area(area_dir, metric_col=args.metric, dem_corr=args.dem_corr, alpha=args.alpha)
        for set_name, dfset in cg_by_set.items():
            if dfset is not None and not dfset.empty:
                all_corr_by_set[set_name].append(dfset.assign(area=area_dir.name))
        if dg is not None and not dg.empty:
            all_dem.append(dg.assign(area=area_dir.name))

    out_root = root / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    # Across ALL AREAS — Corrections (run separately for with_idw and no_idw)
    for set_name, variants in VARIANT_SETS.items():
        tag = f"__{set_name}"
        stacks = all_corr_by_set.get(set_name, [])
        if stacks:
            corr_all = pd.concat(stacks, ignore_index=True)
            # Global subject = area::pair
            corr_all["pair_global"] = corr_all.apply(lambda r: f"{r['area']}::{r['pair']}", axis=1)

            # Global ANOVA (+ explanation columns)
            aov_all = _anova_rm_oneway(corr_all.rename(columns={"pair_global":"subject"}),
                                       dv="value", subject="subject", within="variant")
            if aov_all is not None:
                tbl = aov_all.anova_table.reset_index(drop=True).copy()
                tbl["note"] = ("One-way repeated-measures ANOVA across corrections (all densities; subjects = area::pair). "
                               "Lower is better; small p implies at least one correction differs.")
                tbl["k_levels"] = int(len([v for v in variants if v in corr_all['variant'].unique()]))
                tbl["subjects_n"] = int(corr_all["pair_global"].nunique())

                # --- Attach RMSE & bias summaries (per variant) across all subjects ---
                def _iqr_arr(a):
                    q = np.nanpercentile(a, [25, 75]); return float(q[1]-q[0])
                rmse_g = corr_all.groupby("variant", observed=False)["rmse_value"]
                bias_g = corr_all.groupby("variant", observed=False)["bias_value"] if "bias_value" in corr_all.columns else None
                for v in [vv for vv in variants if vv in corr_all["variant"].unique()]:
                    tbl[f"rmse_mean_{v}"]   = float(rmse_g.mean().get(v, np.nan))
                    tbl[f"rmse_sd_{v}"]     = float(rmse_g.std().get(v, np.nan))
                    tbl[f"rmse_median_{v}"] = float(rmse_g.median().get(v, np.nan))
                    tbl[f"rmse_iqr_{v}"]    = float(_iqr_arr(corr_all[corr_all["variant"]==v]["rmse_value"].values))
                    tbl[f"rmse_n_subjects_{v}"] = int(corr_all[corr_all["variant"]==v]["pair_global"].nunique())
                    if bias_g is not None:
                        tbl[f"bias_mean_{v}"] = float(bias_g.mean().get(v, np.nan))
                        tbl[f"bias_sd_{v}"]   = float(bias_g.std().get(v, np.nan))
                tbl.to_csv(out_root / f"corrections_all_areas_anova{tag}.csv", index=False)
                print(f"✅  All-areas corrections ANOVA ({set_name}) written: {out_root / f'corrections_all_areas_anova{tag}.csv'}")
            else:
                print(f"ℹ️  All-areas corrections ANOVA could not be computed ({set_name}).")

            # Ranking — ONLY for with_idw
            if set_name == "with_idw":
                enr = _enriched_ranking(corr_all.rename(columns={"pair_global":"subject"}),
                                        subject_col="subject", variant_list=variants)
                if not enr.empty:
                    enr[["rank","variant","mean","sd","median","iqr","n_subjects",
                         "p_vs_next_adj_two_sided","p_vs_lower_min_adj_two_sided",
                         "p_vs_raw_two_sided","p_vs_raw_two_sided_holm",
                         "p_vs_raw_one_sided_improve","p_vs_raw_one_sided_improve_holm",
                         "rmse_sd","rmse_median","rmse_iqr","bias_mean","bias_sd"]] \
                        .to_csv(out_root / f"corrections_all_areas_ranking{tag}.csv", index=False)

                # RMSE-centric global corrections rank table (for thesis tables)
                if not corr_all.empty:
                    def _iqr_arr2(a):
                        a = np.asarray(a, float)
                        q = np.nanpercentile(a, [25, 75]); return float(q[1]-q[0])
                    rmse_g = corr_all.groupby("variant", observed=False)["rmse_value"]
                    stats_tmp = []
                    for v in variants:
                        if v not in rmse_g.groups:
                            continue
                        vals = corr_all.loc[corr_all["variant"]==v, "rmse_value"].values
                        mean = float(np.nanmean(vals))
                        sd   = float(np.nanstd(vals, ddof=1))
                        med  = float(np.nanmedian(vals))
                        iqr  = _iqr_arr2(vals)
                        nsubj = int(corr_all[corr_all["variant"]==v]["pair_global"].nunique())
                        stats_tmp.append((v, mean, sd, med, iqr, nsubj))
                    stats_tmp.sort(key=lambda t: t[1])  # by mean RMSE
                    # map two-sided Holm p vs RAW from enr
                    pmap = {}
                    if "p_vs_raw_two_sided_holm" in enr.columns:
                        pmap = dict(zip(enr["variant"], enr["p_vs_raw_two_sided_holm"]))
                    rows_rmse = []
                    for i, (v, mean, sd, med, iqr, nsubj) in enumerate(stats_tmp, start=1):
                        rows_rmse.append({
                            "rank": i,
                            "corr": v,
                            "rmse_mean": mean,
                            "rmse_sd": sd,
                            "rmse_median": med,
                            "rmse_iqr": iqr,
                            "mean±SD(rmse)": _fmt_pm(mean, sd),
                            "median[IQR](rmse)": _fmt_med_iqr(med, iqr),
                            "n_subjects": nsubj,
                            "p_diff_from_RAW_two_sided_holm": float(pmap.get(v, np.nan)) if v != "RAW" else np.nan,
                        })
                    pd.DataFrame(rows_rmse).to_csv(out_root / "corrections_all_areas_rank_rmse__with_idw.csv", index=False)

            print(f"✅  All-areas corrections ranking/ANOVA ({set_name}) written in {out_root}")
        else:
            print(f"ℹ️  No corrections data accumulated across areas ({set_name}).")

    # Across ALL AREAS — DEM t-test (with chosen correction) + each-area table
    dem_root_rows = []
    if all_dem:
        dem_all = pd.concat(all_dem, ignore_index=True)
        corr_used = str(args.dem_corr).upper()
        wide_all = (dem_all[dem_all["corr_used"] == corr_used]
                    .pivot_table(index=["area","pair"], columns="dem", values="value",
                                 aggfunc="mean", observed=False)
                    .reindex(columns=["SRTM","3DEP"])
                    .dropna(subset=["SRTM","3DEP"]))
        if wide_all.shape[0] >= 2:
            t_all, p_all, _ = ttest_rel_safe(wide_all["3DEP"], wide_all["SRTM"])

            # RMSE across the same pairs
            wide_all_rmse_full = (dem_all[dem_all["corr_used"] == corr_used]
                                  .pivot_table(index=["area","pair"], columns="dem", values="rmse_value",
                                               aggfunc="mean", observed=False)
                                  .reindex(columns=["SRTM","3DEP"]))
            wide_all_rmse = wide_all_rmse_full.reindex(wide_all.index)

            # Bias across the same pairs (optional)
            if "bias_value" in dem_all.columns:
                wide_all_bias_full = (dem_all[dem_all["corr_used"] == corr_used]
                                      .pivot_table(index=["area","pair"], columns="dem", values="bias_value",
                                                   aggfunc="mean", observed=False)
                                      .reindex(columns=["SRTM","3DEP"]))
                wide_all_bias = wide_all_bias_full.reindex(wide_all.index)
            else:
                wide_all_bias = None

            # RMSE dispersion and medians/IQR
            def _iqr_arr(a):
                a = np.asarray(a, float)
                q = np.nanpercentile(a, [25, 75]); return float(q[1]-q[0])

            out_row = {
                "corr_used": corr_used,
                "n_pairs": int(wide_all.shape[0]),
                "mean_diff_3DEP_minus_SRTM": float((wide_all["3DEP"]-wide_all["SRTM"]).mean()),
                "t": float(t_all), "p": float(p_all),
                "best_dem_overall": "3DEP" if (wide_all["3DEP"]-wide_all["SRTM"]).mean() < 0 else "SRTM",
                # RMSE means + descriptors
                "rmse_mean_SRTM": float(wide_all_rmse["SRTM"].mean()),
                "rmse_mean_3DEP": float(wide_all_rmse["3DEP"].mean()),
                "rmse_sd_SRTM": float(wide_all_rmse["SRTM"].std()),
                "rmse_sd_3DEP": float(wide_all_rmse["3DEP"].std()),
                "rmse_median_SRTM": float(wide_all_rmse["SRTM"].median()),
                "rmse_median_3DEP": float(wide_all_rmse["3DEP"].median()),
                "rmse_iqr_SRTM": _iqr_arr(wide_all_rmse["SRTM"].values),
                "rmse_iqr_3DEP": _iqr_arr(wide_all_rmse["3DEP"].values),
            }

            # Bias mean + SD per DEM
            if wide_all_bias is not None:
                out_row.update({
                    "bias_mean_SRTM": float(wide_all_bias["SRTM"].mean()),
                    "bias_mean_3DEP": float(wide_all_bias["3DEP"].mean()),
                    "bias_sd_SRTM": float(wide_all_bias["SRTM"].std()),
                    "bias_sd_3DEP": float(wide_all_bias["3DEP"].std()),
                })

            pd.DataFrame([out_row]).to_csv(out_root / f"dem_all_areas_ttest__{corr_used}.csv", index=False)
            print(f"✅  All-areas DEM t-test ({corr_used}) written: {out_root / f'dem_all_areas_ttest__{corr_used}.csv'}")

            # All-areas DEM rank table based on RMSE (2 rows, SRTM vs 3DEP)
            rmse_all = {}
            for dem_name in ["SRTM","3DEP"]:
                vals = wide_all_rmse[dem_name].values
                mean = float(np.nanmean(vals))
                sd   = float(np.nanstd(vals, ddof=1))
                median = float(np.nanmedian(vals))
                q25, q75 = np.nanpercentile(vals, [25, 75])
                iqr = float(q75 - q25)
                rmse_all[dem_name] = {
                    "mean": mean,
                    "sd": sd,
                    "median": median,
                    "iqr": iqr,
                }
            ordered_dem = sorted(["SRTM","3DEP"], key=lambda d: rmse_all[d]["mean"])
            dem_rank_all_rows = []
            for r, dem_name in enumerate(ordered_dem, start=1):
                s = rmse_all[dem_name]
                dem_rank_all_rows.append({
                    "rank": r,
                    "dem": dem_name,
                    "rmse_mean": s["mean"],
                    "rmse_sd": s["sd"],
                    "rmse_median": s["median"],
                    "rmse_iqr": s["iqr"],
                    "mean±SD(rmse)": _fmt_pm(s["mean"], s["sd"]),
                    "median[IQR](rmse)": _fmt_med_iqr(s["median"], s["iqr"]),
                    "p_best": float(p_all),  # same p for both
                })
            pd.DataFrame(dem_rank_all_rows).to_csv(out_root / f"dem_all_areas_rank__{corr_used}.csv", index=False)
        else:
            print(f"ℹ️  Not enough pairs with both DEMs across areas for all-areas DEM test at {corr_used}.")

        # Each area's overall DEM t-test at root (kept; now includes RMSE mean, SD, median, IQR)
        for area_name, sub in dem_all[dem_all["corr_used"] == corr_used].groupby("area", observed=False):
            wide = (
                sub.pivot_table(
                    index="pair",
                    columns="dem",
                    values="value",
                    aggfunc="mean",
                    observed=False,
                )
                .reindex(columns=["SRTM", "3DEP"])
                .dropna(subset=["SRTM", "3DEP"])
            )
            if wide.shape[0] >= 2:
                t_a, p_a, _ = ttest_rel_safe(wide["3DEP"], wide["SRTM"])

                # RMSE aligned
                wide_rmse_full = sub.pivot_table(
                    index="pair",
                    columns="dem",
                    values="rmse_value",
                    aggfunc="mean",
                    observed=False,
                ).reindex(columns=["SRTM", "3DEP"])
                wide_rmse = wide_rmse_full.reindex(wide.index)

                # Helper for IQR
                def _iqr_arr_local(a):
                    a = np.asarray(a, float)
                    q25, q75 = np.nanpercentile(a, [25, 75])
                    return float(q75 - q25)

                vals_S = wide_rmse["SRTM"].values
                vals_3 = wide_rmse["3DEP"].values

                dem_root_rows.append({
                    "area": area_name,
                    "corr_used": corr_used,
                    "n_pairs": int(wide.shape[0]),
                    "mean_diff_3DEP_minus_SRTM": float(
                        (wide["3DEP"] - wide["SRTM"]).mean()
                    ),
                    "t": float(t_a),
                    "p": float(p_a),
                    "best_dem_overall": "3DEP"
                    if (wide["3DEP"] - wide["SRTM"]).mean() < 0
                    else "SRTM",
                    # RMSE mean (as before)
                    "rmse_mean_SRTM": float(np.nanmean(vals_S)),
                    "rmse_mean_3DEP": float(np.nanmean(vals_3)),
                    # NEW: RMSE SD
                    "rmse_sd_SRTM": float(np.nanstd(vals_S, ddof=1)),
                    "rmse_sd_3DEP": float(np.nanstd(vals_3, ddof=1)),
                    # NEW: RMSE median
                    "rmse_median_SRTM": float(np.nanmedian(vals_S)),
                    "rmse_median_3DEP": float(np.nanmedian(vals_3)),
                    # NEW: RMSE IQR
                    "rmse_iqr_SRTM": _iqr_arr_local(vals_S),
                    "rmse_iqr_3DEP": _iqr_arr_local(vals_3),
                })
        if dem_root_rows:
            pd.DataFrame(dem_root_rows).to_csv(out_root / f"dem_each_area_overall_ttest__{corr_used}.csv", index=False)
            print(f"✅  Each-area DEM overall t-test ({corr_used}) written: {out_root / f'dem_each_area_overall_ttest__{corr_used}.csv'}")
    else:
        print("ℹ️  No DEM data accumulated across areas.")

    # Best correction by area — BOTH with_idw and no_idw (all densities)
    for set_name, variants in VARIANT_SETS.items():
        stacks = all_corr_by_set.get(set_name, [])
        best_area_rows = []
        if not stacks:
            continue
        stack = pd.concat(stacks, ignore_index=True)

        for area_name, sub in stack.groupby("area", observed=False):
            # per-area, per-pair means already in columns: value (metric), rmse_value, bias_value
            area_long = (sub.groupby(["pair","variant"], observed=False)
                           .agg(value=("value","mean"),
                                rmse_value=("rmse_value","mean"),
                                bias_value=("bias_value","mean"))
                           .reset_index())

            # Ranking (by metric value) + p-vs-RAW table for best
            rank_tbl, _ = _best_second_summary(
                area_long.rename(columns={"pair": "subject"}),
                subject_col="subject",
                variant_list=variants,
            )
            if rank_tbl.empty:
                continue

            # Map RMSE means per variant (reporting)
            rmse_means = area_long.groupby("variant", observed=False)["rmse_value"].mean()

            # p vs RAW for BEST (metric-based; plain two-sided)
            vsraw = _vs_raw_table(
                area_long.rename(columns={"pair": "subject"}),
                subject_col="subject",
                variant_list=variants,
            )
            vsraw = vsraw.set_index("variant") if not vsraw.empty else pd.DataFrame()

            # Prepare top-4 entries (or fewer if fewer levels exist)
            ordered = rank_tbl.sort_values("rank")["variant"].tolist()
            topk = ordered[:min(4, len(ordered))]

            row = {"area": area_name}

            # Best variant fields (metric-based)
            row["best_variant"] = str(topk[0]) if len(topk) >= 1 else ""
            row["best_mean"] = (
                float(
                    rank_tbl.loc[rank_tbl["variant"] == row["best_variant"], "mean"].values[0]
                )
                if len(topk) >= 1
                else np.nan
            )

            # Best: RMSE mean
            row["best_rmse_mean"] = (
                float(rmse_means.get(row["best_variant"], np.nan))
                if len(topk) >= 1
                else np.nan
            )

            # PLAIN two-sided p vs RAW for the BEST (no Holm)
            if (
                not vsraw.empty
                and row["best_variant"] in vsraw.index
                and "p_two_sided" in vsraw.columns
            ):
                row["best_p_vs_raw_two_sided"] = float(
                    vsraw.loc[row["best_variant"], "p_two_sided"]
                )
            else:
                row["best_p_vs_raw_two_sided"] = np.nan

            # Second–Fourth: names + RMSE means
            for i, label in enumerate(["second", "third", "fourth"], start=1):
                v = str(topk[i]) if len(topk) > i else ""
                row[f"{label}_variant"] = v
                row[f"{label}_rmse_mean"] = (
                    float(rmse_means.get(v, np.nan)) if v else np.nan
                )

            # Number of subjects (pairs)
            row["n_subjects_pairs"] = int(area_long["pair"].nunique())

            best_area_rows.append(row)

        if best_area_rows:
            out = out_root / f"corrections_best_by_area__{set_name}.csv"
            pd.DataFrame(best_area_rows).to_csv(out, index=False)
            print(f"✅  Best correction by area ({set_name}) written: {out}")

            # Wide, RMSE-focused table for WITH IDW (best→4th corrections per area)
            if set_name == "with_idw":
                wide_rows = []
                for row in best_area_rows:
                    wide_rows.append({
                        "area": row["area"],
                        "best_corr": row.get("best_variant", ""),
                        "best_mean_rmse": row.get("best_rmse_mean", np.nan),
                        # also use plain two-sided p vs RAW here
                        "p_diff_from_RAW_two_sided": row.get("best_p_vs_raw_two_sided", np.nan),
                        "second_corr": row.get("second_variant", ""),
                        "second_mean_rmse": row.get("second_rmse_mean", np.nan),
                        "third_corr": row.get("third_variant", ""),
                        "third_mean_rmse": row.get("third_rmse_mean", np.nan),
                        "fourth_corr": row.get("fourth_variant", ""),
                        "fourth_mean_rmse": row.get("fourth_rmse_mean", np.nan),
                    })
                pd.DataFrame(wide_rows).to_csv(
                    out_root / "corrections_order_by_area__with_idw.csv",
                    index=False,
                )

if __name__ == "__main__":
    main()
