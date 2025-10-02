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
       enriched with adjusted p-values vs next/lower and vs RAW (two-sided + one-sided-improve).
     - All-areas (global) repeated-measures ANOVA + ranking (WITH IDW only).

   Notes:
   - **Corrections analyses use DEM=SRTM only** for LS variants. Pairs without SRTM for
     the four LS corrections are skipped. IDW rows are included for the "with_idw" set.

B) DEMs (SRTM vs 3DEP), correction FIXED but SWITCHABLE (default RAW):
   - Per-pair paired t-test (subject = block).
   - Area-level paired t-test (subject = pair; per-pair means).
   - All-areas paired t-test (subject = area::pair).
   All three use **all blocks** where both DEMs are available within the block.

Design assumptions
------------------
- Script 5 writes accuracy rows for multiple densities (e.g., 60/40, 45/55, 30/70, 15/85),
  for each replicate and variant.
- Required columns: area, pair_ref, pair_sec, dem, corr, method, replicate, n_cal, <metric>
- Lower metric (e.g., rmse_cm) is better.

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

Across ALL AREAS (into <areas_root>/results/):
  corrections_all_areas_anova__with_idw.csv
  corrections_all_areas_ranking__with_idw.csv
  corrections_all_areas_anova__no_idw.csv
  dem_all_areas_ttest__<CORR>.csv
  dem_each_area_overall_ttest__<CORR>.csv
  corrections_best_by_area__with_idw.csv
  corrections_best_by_area__no_idw.csv

How to run
----------
# All areas, default metric rmse_cm and DEM correction = RAW
python 6_inferential_tests.py

# Single area
python 6_inferential_tests.py --area ENP

# Use log-RMSE instead of RMSE
python 6_inferential_tests.py --metric log_rmse_cm

# DEM comparison using IONO instead of RAW
python 6_inferential_tests.py --dem-corr IONO
"""

from __future__ import annotations
from pathlib import Path
import argparse
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
DEM_CORR_DEFAULT = "RAW"

METHOD_IDW = "idw_dhvis"
METHOD_LS  = "least_squares"

CORR_LEVELS = ["RAW", "IONO", "TROPO", "TROPO_IONO"]
ALL_VARIANTS = CORR_LEVELS + ["IDW"]

VARIANT_SETS = {
    "with_idw": ALL_VARIANTS,      # 5 levels
    "no_idw":  CORR_LEVELS,        # 4 levels
}

DEM_CORR_CHOICES = set(CORR_LEVELS)  # correction used for DEM comparisons (IDW not applicable)

# ------------------------------ Safe stats helpers --------------------------------

def ttest_rel_safe(a, b, eps: float = 1e-12):
    """Paired t-test with a stability guard; returns ``(t, p, n)``.

    Parameters
    ----------
    a, b : array-like
        Two matched samples. Non-finite pairs are dropped prior to testing.
    eps : float, default 1e-12
        Threshold for deeming the difference vector nearly constant; in that
        case the function returns ``t=0, p=1`` to avoid numerical issues.

    Returns
    -------
    tuple[float, float, int]
        ``t`` value, two-sided ``p`` value, and effective sample size ``n`` after
        filtering.
    """
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
    """Convert a two-sided p-value to a one-sided p-value using the t-statistic sign.

    Parameters
    ----------
    t_val : float
        Observed t-statistic (paired design).
    p_two : float
        Reported two-sided p-value for the same test.
    n : int
        Sample size (paired). Degrees of freedom are ``df = n − 1``.
    alternative : {"less", "greater"}, default "less"
        Direction of the one-sided alternative. For corrections we use
        ``"less"`` to test H1: variant < RAW (i.e., better if lower metric).

    Returns
    -------
    float
        One-sided p-value, or NaN if inputs are invalid.
    """
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
    """Build a compact pair tag ``YYYYMMDD_YYYYMMDD`` from ISO dates.

    Examples
    --------
    ``_pair_tag("2010-03-23", "2010-05-08") -> "20100323_20100508"``
    """
    return f"{ref.replace('-','')}_{sec.replace('-','')}"

def _choose_dem_for_corrections(df_pair: pd.DataFrame) -> str | None:
    """Enforce DEM=SRTM for corrections comparisons.

    Returns ``"SRTM"`` if present in ``df_pair['dem']``; otherwise ``None`` to
    signal that this pair should be skipped in corrections analyses.
    """
    dems_present = set(df_pair["dem"].dropna().astype(str).unique().tolist())
    return "SRTM" if "SRTM" in dems_present else None

def _build_blocks_for_pair(df_pair: pd.DataFrame, dem_sel: str, metric_col: str,
                           variant_list: list[str]) -> pd.DataFrame:
    """Construct long-format data for per-pair corrections ANOVA.

    Uses **all blocks** (``replicate x n_cal``) available for the pair, but keeps
    only those blocks where **all** requested variants exist.

    Parameters
    ----------
    df_pair : pandas.DataFrame
        Subset of the metrics table for a single pair.
    dem_sel : str
        The enforced DEM to use for LS variants (typically ``"SRTM"``).
    metric_col : str
        Name of the metric column to analyze (e.g., ``"rmse_cm"``).
    variant_list : list[str]
        Ordered list of variants to include (e.g., ``["RAW","IONO","TROPO","TROPO_IONO","IDW"]``).

    Returns
    -------
    pandas.DataFrame
        Long table with columns ``['area','pair','block_id','variant','value']``.

    Notes
    -----
    • LS rows are filtered to ``dem == dem_sel``;
    • IDW rows are not DEM-filtered (their ``dem`` is ``"N/A"``).
    • ``block_id`` uniquely identifies a calibration density within a replicate.
    """
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

    # Keep only complete blocks that have all requested variants
    grp = cat.groupby(["pair","block_id"], observed=False)
    ok_blocks = grp["variant"].apply(lambda s: set(variant_list).issubset(set(s))).reset_index()
    ok_blocks = ok_blocks[ok_blocks["variant"] == True][["pair","block_id"]]  # noqa: E712

    out = cat.merge(ok_blocks, on=["pair","block_id"], how="inner")
    out = out[["area","pair","block_id","variant","value"]].dropna(subset=["value"])
    out["variant"] = pd.Categorical(out["variant"], categories=variant_list, ordered=True)
    return out


def _anova_rm_oneway(df_long: pd.DataFrame, dv: str, subject: str, within: str):
    """Run one-way repeated-measures ANOVA via ``statsmodels.AnovaRM``.

    Returns the fitted object or ``None`` if there are not enough observations
    or the model fails to fit.
    """
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
    """Paired t-tests across levels of ``within``, aligned by ``subject``.

    Produces a table with columns: ``level_a, level_b, t, p_raw, p_holm_two_sided,
    mean_diff, n``. Holm correction is applied over the tested pairs.
    """
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
    """Compute mean/SD/median/IQR per variant across subjects and rank them.

    Returns a table with columns: ``variant, mean, sd, median, iqr, n_subjects, rank``
    (lower metric values rank better).
    """
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
    """Fetch Holm-adjusted two-sided p-value for ``a`` vs ``b`` from a pairwise table.

    The input table must have columns ``level_a``, ``level_b``, ``p_holm_two_sided``.
    """
    if pw_df is None or pw_df.empty:
        return np.nan
    row = pw_df[((pw_df["level_a"]==a) & (pw_df["level_b"]==b)) |
                ((pw_df["level_b"]==a) & (pw_df["level_a"]==b))]
    return float(row["p_holm_two_sided"].iloc[0]) if not row.empty else np.nan

def _best_second_summary(df_long: pd.DataFrame, subject_col: str,
                         variant_list: list[str]):
    """Produce ranking + summary dict for the best and second-best variants.

    Returns
    -------
    (rank_table, summary_dict)
        ``rank_table`` is the full ranking; ``summary_dict`` includes keys like
        ``best_variant``, ``best_mean``, ``second_variant``, and adjusted p-values
        comparing the top entries.
    """
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
    """Build a p-value table for each correction vs RAW across subjects.

    Reports per variant (excluding RAW):
      • ``n_subjects`` used,
      • ``mean_variant``, ``mean_raw``, and ``mean_diff_variant_minus_raw`` (negative → better than RAW),
      • paired-t ``t`` and two-sided ``p`` values,
      • one-sided p for improvement (H1: variant < RAW), plus Holm-adjusted one-sided p.
    """
    if df_long.empty or "RAW" not in variant_list:
        return pd.DataFrame()

    wide = df_long.pivot_table(index=subject_col, columns="variant", values="value",
                               aggfunc="mean", observed=False)
    if "RAW" not in wide.columns:
        return pd.DataFrame()

    rows = []
    p_one = []
    variants = [v for v in variant_list if v != "RAW" and v in wide.columns]
    for v in variants:
        w = wide[["RAW", v]].dropna(axis=0, how="any")
        if w.shape[0] < 2:
            rows.append({"variant": v, "n_subjects": int(w.shape[0])})
            p_one.append(np.nan)
            continue
        t, p_two, n = ttest_rel_safe(w[v], w["RAW"])  # diff = v - RAW
        mean_diff = float((w[v] - w["RAW"]).mean()) if n >= 1 else np.nan
        p_one_sided = one_sided_from_two_sided(t, p_two, n, alternative="less")  # H1: v < RAW (better)
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
        p_one.append(p_one_sided)

    if rows:
        finite = [p for p in p_one if np.isfinite(p)]
        if finite:
            _, p_holm, _, _ = multipletests(finite, method="holm")
            j = 0
            for r in rows:
                if np.isfinite(r.get("p_one_sided_improve", np.nan)):
                    r["p_one_sided_improve_holm"] = float(p_holm[j]); j += 1
                else:
                    r["p_one_sided_improve_holm"] = np.nan

    return pd.DataFrame(rows)

# -------- Enriched ranking (adds p vs next/lower and vs RAW columns) --------

def _enriched_ranking(df_long: pd.DataFrame, subject_col: str,
                      variant_list: list[str]) -> pd.DataFrame:
    """Ranking + extra p-value columns (two-sided & one-sided-improve vs RAW)."""
    rank_tbl = _summarize_variant_means(df_long, subject_col=subject_col, variant_list=variant_list)
    if rank_tbl.empty:
        return rank_tbl

    pw = _pairwise_within_subject_ttests(df_long, subject=subject_col, within="variant",
                                         dv="value", variant_list=variant_list)
    vsraw = _vs_raw_table(df_long, subject_col=subject_col, variant_list=variant_list)
    vsraw = vsraw.set_index("variant") if not vsraw.empty else pd.DataFrame()

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

        p2_raw = np.nan
        p1_raw = np.nan
        p1_raw_holm = np.nan
        if not vsraw.empty and v in vsraw.index:
            p2_raw = float(vsraw.loc[v, "p_two_sided"]) if "p_two_sided" in vsraw.columns else np.nan
            p1_raw = float(vsraw.loc[v, "p_one_sided_improve"]) if "p_one_sided_improve" in vsraw.columns else np.nan
            p1_raw_holm = float(vsraw.loc[v, "p_one_sided_improve_holm"]) if "p_one_sided_improve_holm" in vsraw.columns else np.nan

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
            "p_vs_raw_two_sided": float(p2_raw),
            "p_vs_raw_one_sided_improve": float(p1_raw),
            "p_vs_raw_one_sided_improve_holm": float(p1_raw_holm),
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
        Area-level long table (columns: area, pair, variant, value).
    """
    perpair_rows = []
    area_perpair_long = []
    pw_all = []

    for p in sorted(df_area["pair"].unique().tolist()):
        dfp = df_area[df_area["pair"]==p].copy()

        # Enforce SRTM for corrections LS data
        dem_sel = _choose_dem_for_corrections(dfp)
        if dem_sel is None:
            # No SRTM for this pair → skip corrections analyses for this pair
            continue

        long = _build_blocks_for_pair(dfp, dem_sel, metric_col, variant_list)
        if long.empty:
            continue

        # Per-pair ANOVA (subject=block)
        aov = _anova_rm_oneway(long, dv="value", subject="block_id", within="variant")
        if aov is None:
            continue

        # Pairwise comparisons (two-sided Holm)
        pw = _pairwise_within_subject_ttests(long, subject="block_id", within="variant",
                                             dv="value", variant_list=variant_list)

        means = long.groupby("variant", observed=False)["value"].mean().sort_values()
        best_variant = str(means.index[0])

        long = long.assign(area=area_name)
        area_perpair_long.append(long)

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
        for v in variant_list:
            row[f"mean_{v}"] = float(means.get(v, np.nan))
        perpair_rows.append(row)

        pw_all.append(pw.assign(area=area_name, pair=p,
                                dem_used=dem_sel, metric=metric_col,
                                variant_set=variant_set_name))

    # Write per-pair outputs
    tag = f"__{variant_set_name}"
    if perpair_rows:
        pd.DataFrame(perpair_rows).to_csv(res_dir / f"corrections_per_pair_anova{tag}.csv", index=False)
        # Only write pairwise table for WITH IDW (keep original behavior)
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
        area_long = long_all.groupby(["area","pair","variant"], observed=False)["value"].mean().reset_index()

        # ANOVA (+ explanation columns)
        aov_area = _anova_rm_oneway(area_long.rename(columns={"pair":"subject"}),
                                    dv="value", subject="subject", within="variant")
        if aov_area is not None:
            tbl = aov_area.anova_table.reset_index(drop=True).copy()
            tbl["note"] = ("One-way repeated-measures ANOVA across corrections (all densities; subjects = pairs). "
                           "Lower is better; small p implies at least one correction differs.")
            tbl["k_levels"] = int(len([v for v in variant_list if v in area_long["variant"].unique()]))
            tbl["subjects_n"] = int(area_long["pair"].nunique())
            tbl.to_csv(corr_area_csv, index=False)

        # Ranking (enriched) — ONLY for with_idw
        if variant_set_name == "with_idw":
            enr = _enriched_ranking(area_long.rename(columns={"pair":"subject"}),
                                    subject_col="subject", variant_list=variant_list)
            if not enr.empty:
                enr[["rank","variant","mean","sd","median","iqr","n_subjects",
                     "p_vs_next_adj_two_sided","p_vs_lower_min_adj_two_sided",
                     "p_vs_raw_two_sided","p_vs_raw_one_sided_improve","p_vs_raw_one_sided_improve_holm"]] \
                    .to_csv(corr_area_rank, index=False)

        print(f"✅  Corrections area-level outputs ({variant_set_name}) written in {res_dir}")
        # For global aggregation
        return area_long.assign(variant_set=variant_set_name)
    else:
        print(f"ℹ️  No area-level corrections data for {area_name} ({variant_set_name}).")
        return pd.DataFrame()

# ------------------------------ Whole-area process ------------------------------

def process_area(area_dir: Path, metric_col: str = "rmse_cm",
                 dem_corr: str = DEM_CORR_DEFAULT, alpha: float = 0.05):
    """Run all requested tests for a single AREA and write outputs (all densities).

    Parameters
    ----------
    area_dir : pathlib.Path
        Path to the area directory containing ``results/accuracy_metrics.csv``.
    metric_col : str, default "rmse_cm"
        Metric to analyze (lower is better), e.g., ``"rmse_cm"`` or ``"log_rmse_cm"``.
    dem_corr : str, default DEM_CORR_DEFAULT
        Correction level to use for DEM (SRTM vs 3DEP) comparisons.
    alpha : float, default 0.05
        Reserved for future use (e.g., significance flagging in outputs).

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.DataFrame | None]
        ``corr_for_global_by_set``: mapping variant-set → long table for global aggregation;
        ``dem_for_global``: long table for DEM tests (or ``None`` if not available).
    """
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
        sub = dfp[(dfp["method"]==METHOD_LS) & (dfp["corr"]==corr_used)].copy()
        if sub.empty:
            continue

        sub["block_id"] = sub.apply(lambda r: f"rep{int(r['replicate'])}_n{int(r['n_cal'])}", axis=1)
        have_both = sub.groupby("block_id", observed=False)["dem"].apply(lambda s: set(s) >= {"SRTM","3DEP"})
        ok_blocks = have_both[have_both].index.tolist()
        sub_ok = sub[sub["block_id"].isin(ok_blocks)].copy()
        if sub_ok.empty:
            continue
        wide = sub_ok.pivot_table(index="block_id", columns="dem", values=metric_col, aggfunc="mean", observed=False) \
                    .dropna(subset=["SRTM","3DEP"])
        if wide.shape[0] < 2:
            continue
        t, pval, n = ttest_rel_safe(wide["3DEP"], wide["SRTM"])
        diff = float((wide["3DEP"] - wide["SRTM"]).mean()) if n >= 1 else np.nan
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
        })
        dem_pairs_long.append(pd.DataFrame({
            "area": [area_name, area_name],
            "pair": [dfp["pair"].iloc[0] if not dfp.empty else p,
                     dfp["pair"].iloc[0] if not dfp.empty else p],
            "dem": ["SRTM","3DEP"],
            "corr_used": [corr_used, corr_used],
            "value": [float(wide["SRTM"].mean()), float(wide["3DEP"].mean())]
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
        wide_pairs = long_dem.pivot_table(index="pair", columns="dem", values="value", aggfunc="mean", observed=False) \
                             .dropna(subset=["SRTM","3DEP"])
        if wide_pairs.shape[0] >= 2:
            t_area, p_area, _ = ttest_rel_safe(wide_pairs["3DEP"], wide_pairs["SRTM"])
            pd.DataFrame([{
                "area": area_name,
                "metric": metric_col,
                "corr_used": corr_used,
                "n_pairs": int(wide_pairs.shape[0]),
                "mean_diff_3DEP_minus_SRTM": float((wide_pairs["3DEP"]-wide_pairs["SRTM"]).mean()),
                "t": float(t_area), "p": float(p_area),
                "best_dem_overall": "3DEP" if (wide_pairs["3DEP"]-wide_pairs["SRTM"]).mean() < 0 else "SRTM"
            }]).to_csv(dem_area_csv, index=False)
            pd.DataFrame({
                "variant":["SRTM","3DEP"],
                "corr_used":[corr_used, corr_used],
                "mean":[float(wide_pairs["SRTM"].mean()), float(wide_pairs["3DEP"].mean())],
                "sd":[float(wide_pairs["SRTM"].std()), float(wide_pairs["3DEP"].std())],
                "median":[float(wide_pairs["SRTM"].median()), float(wide_pairs["3DEP"].median())],
                "iqr":[float(np.subtract(*np.nanpercentile(wide_pairs["SRTM"], [75,25]))),
                       float(np.subtract(*np.nanpercentile(wide_pairs["3DEP"], [75,25])))],
                "n_pairs":[int(wide_pairs.shape[0]), int(wide_pairs.shape[0])]
            }).to_csv(dem_area_tbl, index=False)
            print(f"✅  DEM area-level t-test ({corr_used}) written: {dem_area_csv}")
            print(f"✅  DEM area summary table ({corr_used}) written: {dem_area_tbl}")
        else:
            print(f"ℹ️  Not enough pairs with both DEMs in {area_name} for area-level DEM test at {corr_used}.")

    dem_for_global = None
    if dem_pairs_long:
        dem_for_global = pd.concat(dem_pairs_long, ignore_index=True) \
            .groupby(["area","pair","dem","corr_used"], observed=False)["value"].mean().reset_index()

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
    --metric : str, default "rmse_cm"
        Metric column to analyze (e.g., "rmse_cm", "log_rmse_cm").
    --dem-corr : str, default DEM_CORR_DEFAULT ("RAW")
        Correction used for DEM comparisons (IDW not applicable here).
    --alpha : float, default 0.05
        Reserved for future use.
    """
    ap = argparse.ArgumentParser(description="Run per-pair and area/all-areas ANOVA/t-tests for corrections (with & without IDW) and DEMs, using all densities.")
    ap.add_argument("--areas-root", type=str, default="/mnt/DATA2/bakke326l/processing/areas",
                    help="Root containing per-area subfolders.")
    ap.add_argument("--area", type=str,
                    help="Only process this AREA (subfolder name). If omitted, process all areas.")
    ap.add_argument("--metric", type=str, default="rmse_cm",
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
                         "p_vs_raw_two_sided","p_vs_raw_one_sided_improve","p_vs_raw_one_sided_improve_holm"]] \
                        .to_csv(out_root / f"corrections_all_areas_ranking{tag}.csv", index=False)

            print(f"✅  All-areas corrections ranking/ANOVA ({set_name}) written in {out_root}")
        else:
            print(f"ℹ️  No corrections data accumulated across areas ({set_name}).")

    # Across ALL AREAS — DEM t-test (with chosen correction) + each-area table
    dem_root_rows = []
    if all_dem:
        dem_all = pd.concat(all_dem, ignore_index=True)
        corr_used = str(args.dem_corr).upper()
        wide_all = dem_all[dem_all["corr_used"]==corr_used] \
            .pivot_table(index=["area","pair"], columns="dem", values="value", aggfunc="mean", observed=False) \
            .dropna(subset=["SRTM","3DEP"])
        if wide_all.shape[0] >= 2:
            t_all, p_all, _ = ttest_rel_safe(wide_all["3DEP"], wide_all["SRTM"])
            pd.DataFrame([{
                "corr_used": corr_used,
                "n_pairs": int(wide_all.shape[0]),
                "mean_diff_3DEP_minus_SRTM": float((wide_all["3DEP"]-wide_all["SRTM"]).mean()),
                "t": float(t_all), "p": float(p_all),
                "best_dem_overall": "3DEP" if (wide_all["3DEP"]-wide_all["SRTM"]).mean() < 0 else "SRTM"
            }]).to_csv(out_root / f"dem_all_areas_ttest__{corr_used}.csv", index=False)
            print(f"✅  All-areas DEM t-test ({corr_used}) written: {out_root / f'dem_all_areas_ttest__{corr_used}.csv'}")
        else:
            print(f"ℹ️  Not enough pairs with both DEMs across areas for all-areas DEM test at {corr_used}.")

        # Each area's overall DEM t-test at root
        for area_name, sub in dem_all[dem_all["corr_used"]==corr_used].groupby("area", observed=False):
            wide = sub.pivot_table(index="pair", columns="dem", values="value", aggfunc="mean", observed=False) \
                      .dropna(subset=["SRTM","3DEP"])
            if wide.shape[0] >= 2:
                t_a, p_a, _ = ttest_rel_safe(wide["3DEP"], wide["SRTM"])
                dem_root_rows.append({
                    "area": area_name,
                    "corr_used": corr_used,
                    "n_pairs": int(wide.shape[0]),
                    "mean_diff_3DEP_minus_SRTM": float((wide["3DEP"]-wide["SRTM"]).mean()),
                    "t": float(t_a), "p": float(p_a),
                    "best_dem_overall": "3DEP" if (wide["3DEP"]-wide["SRTM"]).mean() < 0 else "SRTM"
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
            area_long = (sub.groupby(["pair","variant"], observed=False)["value"]
                           .mean().reset_index())
            rank_tbl, best2 = _best_second_summary(area_long.rename(columns={"pair":"subject"}),
                                                   subject_col="subject", variant_list=variants)
            if best2:
                best_area_rows.append({
                    "area": area_name,
                    "best_variant": best2["best_variant"],
                    "best_mean": best2["best_mean"],
                    "second_variant": best2["second_variant"],
                    "second_mean": best2["second_mean"],
                    "p_best_vs_second_adj_two_sided": best2["p_best_vs_second_adj_two_sided"],
                    "n_subjects_pairs": best2["n_subjects"],
                })
        if best_area_rows:
            out = out_root / f"corrections_best_by_area__{set_name}.csv"
            pd.DataFrame(best_area_rows).to_csv(out, index=False)
            print(f"✅  Best correction by area ({set_name}) written: {out}")

if __name__ == "__main__":
    main()
