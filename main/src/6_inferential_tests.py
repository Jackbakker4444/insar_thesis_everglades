#!/usr/bin/env python3
"""
6_inferential_tests.py — Per-pair & area-level ANOVA/t-tests for corrections and DEMs

Purpose
-------
Given per-area metrics created by 5_accuracy_assessment.py, perform:
  A) Corrections (RAW, IONO, TROPO, TROPO_IONO, IDW), DEM fixed per pair:
     - Per-pair repeated-measures ANOVA (subject = replicate × n_cal "block").
     - Area-level repeated-measures ANOVA (subject = pair; uses per-pair means).
     - All-areas repeated-measures ANOVA (subject = area:pair).
     - NEW: Best & second-best correction with adjusted p-values (area-level and all-areas).

  B) DEMs (SRTM vs 3DEP), correction FIXED but SWITCHABLE (default TROPO_IONO):
     - Per-pair paired t-test (across blocks).
     - Area-level paired t-test (across pairs; uses per-pair means).
     - All-areas paired t-test (across all pairs in all areas).

Design assumptions
------------------
- For a given pair and DEM, your accuracy script already ensures:
  - Same gauge set and same cal/val plan across corrections.
  - Identical densities across corrections within that DEM.
- For DEM comparisons at the chosen correction, densities are equalized across DEMs.

Inputs
------
Per AREA:
  <areas_root>/<AREA>/results/accuracy_metrics.csv
  (Columns include: area, pair_ref, pair_sec, dem, corr, method, replicate, n_cal,
   rmse_cm, log_rmse_cm, valid_area_km2, area_per_gauge_km2, ...)

Outputs
-------
Per AREA into <areas_root>/<AREA>/results/:
  corrections_per_pair_anova.csv
  corrections_per_pair_pairwise.csv
  corrections_area_overall_anova.csv
  corrections_area_summary_table.csv
  NEW: corrections_area_ranking.csv
  NEW: corrections_area_best_second.csv
  dem_per_pair_ttest__<CORR>.csv
  dem_area_overall_ttest__<CORR>.csv
  dem_area_summary_table__<CORR>.csv

Across ALL AREAS into <areas_root>/results/:
  corrections_all_areas_anova.csv
  NEW: corrections_all_areas_ranking.csv
  NEW: corrections_all_areas_best_second.csv
  dem_all_areas_ttest__<CORR>.csv

How to run
----------
# All areas, default metric rmse_cm and DEM-correction = TROPO_IONO
python 6_inferential_tests.py

# Single area
python 6_inferential_tests.py --area ENP

# Use log-RMSE instead of RMSE
python 6_inferential_tests.py --metric log_rmse_cm

# DEM comparison using IONO instead of TROPO_IONO
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

# ------------------------------ Constants --------------------------------

METHOD_IDW = "idw_dhvis"
METHOD_LS  = "least_squares"
CORR_LEVELS = ["RAW", "IONO", "TROPO", "TROPO_IONO"]
ALL_VARIANTS = CORR_LEVELS + ["IDW"]
DEM_CORR_CHOICES = set(CORR_LEVELS)  # correction used for DEM comparisons (IDW not applicable)

# ------------------------------ Safe stats helpers --------------------------------

def ttest_rel_safe(a, b, eps: float = 1e-12):
    """
    Paired t-test that avoids SciPy precision-loss warnings when data are (near) identical.
    - Drops NaNs pairwise.
    - If std(diff) < eps, returns t=0, p=1, n=len(diff).
    - Else returns scipy.stats.ttest_rel(a, b).
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

# ------------------------------ Utility helpers --------------------------------

def _pair_tag(ref: str, sec: str) -> str:
    """Build a compact pair tag from ISO dates."""
    return f"{ref.replace('-','')}_{sec.replace('-','')}"

def _choose_dem_for_corrections(df_pair: pd.DataFrame) -> str:
    """
    Pick DEM used for correction comparison: prefer SRTM if present, else 3DEP.
    df_pair is metrics for a single pair across DEM/CORR/METHOD.
    """
    dems_present = sorted(df_pair["dem"].dropna().astype(str).unique().tolist())
    if "SRTM" in dems_present:
        return "SRTM"
    if "3DEP" in dems_present:
        return "3DEP"
    return dems_present[0] if dems_present else "SRTM"

def _build_blocks_for_pair(df_pair: pd.DataFrame, dem_sel: str, metric_col: str) -> pd.DataFrame:
    """
    Construct long-format data for per-pair corrections ANOVA.
    Keep only blocks (replicate, n_cal) where all 5 variants exist.
    Returns columns: ['area','pair','block_id','variant','value']
    """
    df = df_pair.copy()
    df = df[df["dem"] == dem_sel].copy()

    is_ls  = (df["method"] == METHOD_LS)
    is_idw = (df["method"] == METHOD_IDW)

    ls_needed = df[is_ls & df["corr"].isin(CORR_LEVELS)].copy()
    idw_needed = df[is_idw].copy().assign(corr="IDW")

    use_cols = ["area","pair_ref","pair_sec","replicate","n_cal","corr",metric_col]
    ls_needed = ls_needed[use_cols].rename(columns={metric_col:"value"})
    idw_needed = idw_needed[use_cols].rename(columns={metric_col:"value"})
    cat = pd.concat([ls_needed, idw_needed], ignore_index=True)

    cat["pair"] = cat.apply(lambda r: _pair_tag(r["pair_ref"], r["pair_sec"]), axis=1)
    cat["block_id"] = cat.apply(lambda r: f"rep{int(r['replicate'])}_n{int(r['n_cal'])}", axis=1)
    cat["variant"] = cat["corr"].astype(str)

    grp = cat.groupby(["pair","block_id"], observed=False)
    ok_blocks = grp["variant"].apply(lambda s: set(ALL_VARIANTS).issubset(set(s))).reset_index()
    ok_blocks = ok_blocks[ok_blocks["variant"] == True][["pair","block_id"]]  # noqa: E712

    cat_ok = cat.merge(ok_blocks, on=["pair","block_id"], how="inner")
    cat_ok = cat_ok[["area","pair","block_id","variant","value"]].dropna(subset=["value"])
    cat_ok["variant"] = pd.Categorical(cat_ok["variant"], categories=ALL_VARIANTS, ordered=True)
    return cat_ok

def _anova_rm_oneway(df_long: pd.DataFrame, dv: str, subject: str, within: str):
    """Run one-way repeated-measures ANOVA with statsmodels AnovaRM."""
    check = df_long[[subject, within]].drop_duplicates().groupby(subject, observed=False).size()
    if check.min() < 2:
        return None
    try:
        aov = AnovaRM(df_long, depvar=dv, subject=subject, within=[within]).fit()
        return aov
    except Exception:
        return None

def _pairwise_within_subject_ttests(df_long: pd.DataFrame, subject: str, within: str, dv: str):
    """
    Paired t-tests for all pairs of levels in 'within', aligned by 'subject'.
    Holm-adjust p-values. Returns DataFrame of comparisons.
    """
    wide = df_long.pivot_table(index=subject, columns=within, values=dv, aggfunc="mean", observed=False)
    wide = wide.dropna(axis=0, how="any")
    levels = [c for c in wide.columns if c in ALL_VARIANTS]
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
            r["p_holm"] = float(p_holm[i])
    return pd.DataFrame(rows)

def _summarize_variant_means(df_long: pd.DataFrame, subject_col: str) -> pd.DataFrame:
    """Mean/SD/median/IQR per variant across subjects for a readable table."""
    def iqr(x):
        q = np.nanpercentile(x, [25, 75])
        return float(q[1]-q[0])
    # Compute per-subject means first so each subject has equal weight
    subj_mean = df_long.pivot_table(index=subject_col, columns="variant", values="value", aggfunc="mean", observed=False)
    # Longify
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
    # Add rank (lower is better)
    out = out.sort_values("mean", ascending=True).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out)+1)
    return out

def _extract_adj_p(pw_df: pd.DataFrame, a: str, b: str) -> float:
    """Fetch Holm-adjusted p for a vs b from pairwise table (order agnostic)."""
    if pw_df is None or pw_df.empty:
        return np.nan
    row = pw_df[((pw_df["level_a"]==a) & (pw_df["level_b"]==b)) |
                ((pw_df["level_a"]==b) & (pw_df["level_b"]==a))]
    return float(row["p_holm"].iloc[0]) if not row.empty else np.nan

def _best_second_summary(df_long: pd.DataFrame, subject_col: str):
    """
    Build ranking + summary for best and second-best with adjusted p-values.
    Returns (ranking_df, best_second_row_dict)
    """
    if df_long.empty:
        return pd.DataFrame(), {}

    # Ranking table
    rank_tbl = _summarize_variant_means(df_long, subject_col=subject_col)
    if rank_tbl.empty:
        return pd.DataFrame(), {}

    # Pairwise tests across subjects
    pw = _pairwise_within_subject_ttests(df_long, subject=subject_col, within="variant", dv="value")

    # Identify best & second-best
    best = rank_tbl.iloc[0]["variant"]
    best_mean = float(rank_tbl.iloc[0]["mean"])
    if len(rank_tbl) >= 2:
        second = rank_tbl.iloc[1]["variant"]
        second_mean = float(rank_tbl.iloc[1]["mean"])
        p_best_vs_second = _extract_adj_p(pw, str(best), str(second))
        # Second vs everyone else excluding best
        others = [str(v) for v in rank_tbl["variant"].tolist() if v not in {best, second}]
        p_second_vs_rest = []
        p_second_vs_third = np.nan
        if others:
            for o in others:
                p_second_vs_rest.append(_extract_adj_p(pw, str(second), o))
            # If there is a third, report that too
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
        "p_best_vs_second_adj": float(p_best_vs_second),
        "p_second_vs_rest_min_adj": float(p_second_vs_rest_min),
        "p_second_vs_third_adj": float(p_second_vs_third),
        "n_subjects": int(df_long[subject_col].nunique()),
    }
    return rank_tbl, summary

# ------------------------------ Main per-area runner --------------------------------

def process_area(area_dir: Path, metric_col: str = "rmse_cm", dem_corr: str = "TROPO_IONO", alpha: float = 0.05):
    """
    Run all requested tests for a single AREA and write outputs.
    """
    area_name = area_dir.name
    res_dir = area_dir / "results"
    metrics_csv = res_dir / "accuracy_metrics.csv"
    if not metrics_csv.exists():
        print(f"⏭️  No metrics for {area_name}: {metrics_csv}")
        return None, None  # for global aggregation

    df = pd.read_csv(metrics_csv)
    needed = {"area","pair_ref","pair_sec","dem","corr","method","replicate","n_cal",metric_col}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{metrics_csv} missing required columns: {missing}")

    df["pair"] = df.apply(lambda r: _pair_tag(str(r["pair_ref"]), str(r["pair_sec"])), axis=1)

    # --------------------- A) Corrections (fixed DEM) ---------------------
    perpair_rows = []
    area_perpair_long = []
    pw_all = []
    for p in sorted(df["pair"].unique().tolist()):
        dfp = df[df["pair"]==p].copy()
        dem_sel = _choose_dem_for_corrections(dfp)
        long = _build_blocks_for_pair(dfp, dem_sel, metric_col)
        if long.empty:
            continue
        aov = _anova_rm_oneway(long, dv="value", subject="block_id", within="variant")
        if aov is None:
            continue
        pw = _pairwise_within_subject_ttests(long, subject="block_id", within="variant", dv="value")

        means = long.groupby("variant", observed=False)["value"].mean().sort_values()
        best_variant = str(means.index[0])

        long = long.assign(area=area_name)
        area_perpair_long.append(long)

        row = {
            "area": area_name,
            "pair": p,
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
        for v in ALL_VARIANTS:
            row[f"mean_{v}"] = float(means.get(v, np.nan))
        perpair_rows.append(row)

        pw_all.append(pw.assign(area=area_name, pair=p, dem_used=dem_sel, metric=metric_col))

    corr_perpair_csv = res_dir / "corrections_per_pair_anova.csv"
    if perpair_rows:
        pd.DataFrame(perpair_rows).to_csv(corr_perpair_csv, index=False)
        print(f"✅  Corrections per-pair ANOVA written: {corr_perpair_csv}")
        if pw_all:
            pd.concat(pw_all, ignore_index=True).to_csv(res_dir / "corrections_per_pair_pairwise.csv", index=False)
    else:
        print(f"ℹ️  No per-pair corrections ANOVA results for {area_name} (insufficient data).")

    # Area-level ANOVA (subject = pair), using per-pair means
    corr_area_csv = res_dir / "corrections_area_overall_anova.csv"
    corr_area_tbl = res_dir / "corrections_area_summary_table.csv"
    corr_area_rank_csv = res_dir / "corrections_area_ranking.csv"          # NEW
    corr_area_best2_csv = res_dir / "corrections_area_best_second.csv"     # NEW
    if area_perpair_long:
        long_all = pd.concat(area_perpair_long, ignore_index=True)
        # Collapse to per-pair means (subject = pair)
        area_long = long_all.groupby(["area","pair","variant"], observed=False)["value"].mean().reset_index()
        aov_area = _anova_rm_oneway(area_long.rename(columns={"pair":"subject"}),
                                    dv="value", subject="subject", within="variant")
        if aov_area is not None:
            aov_area.anova_table.reset_index(drop=True).to_csv(corr_area_csv, index=False)
        # Human-readable per-variant stats across pairs
        _summarize_variant_means(area_long.rename(columns={"pair":"subject"}), subject_col="subject") \
            [["rank","variant","mean","sd","median","iqr","n_subjects"]] \
            .to_csv(corr_area_tbl, index=False)
        print(f"✅  Corrections area-level ANOVA written: {corr_area_csv}")
        print(f"✅  Corrections area summary table written: {corr_area_tbl}")

        # NEW: Ranking & best/second with p-values (paired across pairs)
        rank_tbl, best2 = _best_second_summary(area_long.rename(columns={"pair":"subject"}),
                                               subject_col="subject")
        if not rank_tbl.empty:
            rank_tbl[["rank","variant","mean","sd","median","iqr","n_subjects"]] \
                .to_csv(corr_area_rank_csv, index=False)
        if best2:
            pd.DataFrame([{
                "area": area_name,
                **best2
            }]).to_csv(corr_area_best2_csv, index=False)
        print(f"✅  Corrections area ranking written: {corr_area_rank_csv}")
        print(f"✅  Corrections area best/second with p-values written: {corr_area_best2_csv}")
    else:
        print(f"ℹ️  No area-level corrections data for {area_name}.")

    # --------------------- B) DEMs (fixed correction = SWITCHABLE) ---------------------
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
    dem_perpair_csv = res_dir / f"dem_per_pair_ttest{suffix}.csv"
    if dem_perpair_rows:
        pd.DataFrame(dem_perpair_rows).to_csv(dem_perpair_csv, index=False)
        print(f"✅  DEM per-pair t-tests ({corr_used}) written: {dem_perpair_csv}")
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

    # Return items for global aggregation
    corr_for_global = None
    if area_perpair_long:
        # Per-pair means for corrections
        corr_for_global = long_all.groupby(["area","pair","variant"], observed=False)["value"].mean().reset_index()
    dem_for_global = None
    if dem_pairs_long:
        dem_for_global = pd.concat(dem_pairs_long, ignore_index=True).groupby(["area","pair","dem","corr_used"], observed=False)["value"].mean().reset_index()
    return corr_for_global, dem_for_global

# ------------------------------ CLI / Global aggregation --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run per-pair and area-level ANOVA/t-tests for corrections and DEMs.")
    ap.add_argument("--areas-root", type=str, default="/mnt/DATA2/bakke326l/processing/areas",
                    help="Root containing per-area subfolders.")
    ap.add_argument("--area", type=str,
                    help="Only process this AREA (subfolder name). If omitted, process all areas.")
    ap.add_argument("--metric", type=str, default="rmse_cm",
                    help="Metric column to analyze (e.g., rmse_cm or log_rmse_cm).")
    ap.add_argument("--dem-corr", type=str, default="TROPO_IONO",
                    help=f"Correction to use for DEM comparisons (one of {sorted(DEM_CORR_CHOICES)}).")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (for pairwise Holm tests).")
    args = ap.parse_args()

    root = Path(args.areas_root)
    targets = [root / args.area] if args.area else sorted([p for p in root.iterdir() if p.is_dir()])

    all_corr = []
    all_dem  = []
    for area_dir in targets:
        cg, dg = process_area(area_dir, metric_col=args.metric, dem_corr=args.dem_corr, alpha=args.alpha)
        if cg is not None and not cg.empty:
            all_corr.append(cg.assign(area=area_dir.name))
        if dg is not None and not dg.empty:
            all_dem.append(dg.assign(area=area_dir.name))

    out_root = root / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    # Across ALL AREAS — Corrections ANOVA + Ranking + Best/Second
    if all_corr:
        corr_all = pd.concat(all_corr, ignore_index=True)
        # Global subject = area::pair
        corr_all["pair_global"] = corr_all.apply(lambda r: f"{r['area']}::{r['pair']}", axis=1)
        # ANOVA
        aov_all = _anova_rm_oneway(corr_all.rename(columns={"pair_global":"subject"}),
                                   dv="value", subject="subject", within="variant")
        if aov_all is not None:
            aov_all.anova_table.reset_index(drop=True).to_csv(out_root / "corrections_all_areas_anova.csv", index=False)
            print(f"✅  All-areas corrections ANOVA written: {out_root/'corrections_all_areas_anova.csv'}")
        else:
            print("ℹ️  All-areas corrections ANOVA could not be computed (insufficient data).")
        # Ranking & best/second with p-values
        rank_tbl, best2 = _best_second_summary(corr_all.rename(columns={"pair_global":"subject"}), subject_col="subject")
        if not rank_tbl.empty:
            rank_tbl[["rank","variant","mean","sd","median","iqr","n_subjects"]] \
                .to_csv(out_root / "corrections_all_areas_ranking.csv", index=False)
        if best2:
            pd.DataFrame([best2]).to_csv(out_root / "corrections_all_areas_best_second.csv", index=False)
        print(f"✅  All-areas corrections ranking written: {out_root/'corrections_all_areas_ranking.csv'}")
        print(f"✅  All-areas corrections best/second with p-values written: {out_root/'corrections_all_areas_best_second.csv'}")
    else:
        print("ℹ️  No corrections data accumulated across areas.")

    # Across ALL AREAS — DEM t-test (with chosen correction)
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
            print(f"✅  All-areas DEM t-test ({corr_used}) written: {out_root/f'dem_all_areas_ttest__{corr_used}.csv'}")
        else:
            print(f"ℹ️  Not enough pairs with both DEMs across areas for all-areas DEM test at {corr_used}.")
    else:
        print("ℹ️  No DEM data accumulated across areas.")


if __name__ == "__main__":
    main()
