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

  B) DEMs (SRTM vs 3DEP), correction FIXED but SWITCHABLE (default TROPO_IONO):
     - Per-pair paired t-test (across blocks).
     - Area-level paired t-test (across pairs; uses per-pair means).
     - All-areas paired t-test (across all pairs in all areas).

Design assumptions
------------------
- For a given pair and DEM, your accuracy script already ensures:
  - Same gauge set and same cal/val plan across corrections.
  - Identical densities across corrections within that DEM (we rely on that).
- For DEM comparisons at the chosen correction, densities are equalized across DEMs
  by your updated accuracy script (shared DEM-pair area at that correction).

Inputs
------
Per AREA:
  <areas_root>/<AREA>/results/accuracy_metrics.csv
  (Produced by 5_accuracy_assessment.py with fields incl.:
   area, pair_ref, pair_sec, dem, corr, method, replicate, n_cal, rmse_cm, log_rmse_cm,
   valid_area_km2, area_per_gauge_km2, ...)

Outputs
-------
Per AREA into <areas_root>/<AREA>/results/:
  corrections_per_pair_anova.csv
  corrections_per_pair_pairwise.csv
  corrections_area_overall_anova.csv
  corrections_area_summary_table.csv
  dem_per_pair_ttest__<CORR>.csv
  dem_area_overall_ttest__<CORR>.csv
  dem_area_summary_table__<CORR>.csv

Across ALL AREAS into <areas_root>/results/:
  corrections_all_areas_anova.csv
  dem_all_areas_ttest__<CORR>.csv

How to run
----------
# All areas under root, default metric rmse_cm and DEM-correction = TROPO_IONO
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
except Exception:
    raise SystemExit("This script requires statsmodels. Install with: pip install statsmodels")

# ------------------------------ Constants --------------------------------

METHOD_IDW = "idw_dhvis"
METHOD_LS  = "least_squares"
CORR_LEVELS = ["RAW", "IONO", "TROPO", "TROPO_IONO"]
ALL_VARIANTS = CORR_LEVELS + ["IDW"]
DEM_CORR_CHOICES = set(CORR_LEVELS)  # correction used for DEM comparisons (IDW not applicable)

# ------------------------------ Helpers --------------------------------

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
    idw_needed = df[is_idw].copy()
    idw_needed = idw_needed.assign(corr="IDW")

    use_cols = ["area","pair_ref","pair_sec","replicate","n_cal","corr",metric_col]
    ls_needed = ls_needed[use_cols].rename(columns={metric_col:"value"})
    idw_needed = idw_needed[use_cols].rename(columns={metric_col:"value"})
    cat = pd.concat([ls_needed, idw_needed], ignore_index=True)

    cat["pair"] = cat.apply(lambda r: _pair_tag(r["pair_ref"], r["pair_sec"]), axis=1)
    cat["block_id"] = cat.apply(lambda r: f"rep{int(r['replicate'])}_n{int(r['n_cal'])}", axis=1)
    cat["variant"] = cat["corr"].astype(str)

    grp = cat.groupby(["pair","block_id"])
    ok_blocks = grp["variant"].apply(lambda s: set(ALL_VARIANTS).issubset(set(s))).reset_index()
    ok_blocks = ok_blocks[ok_blocks["variant"] == True][["pair","block_id"]]  # noqa: E712

    cat_ok = cat.merge(ok_blocks, on=["pair","block_id"], how="inner")
    cat_ok = cat_ok[["area","pair","block_id","variant","value"]].dropna(subset=["value"])
    cat_ok["variant"] = pd.Categorical(cat_ok["variant"], categories=ALL_VARIANTS, ordered=True)
    return cat_ok

def _anova_rm_oneway(df_long: pd.DataFrame, dv: str, subject: str, within: str):
    """Run one-way repeated-measures ANOVA with statsmodels AnovaRM."""
    if df_long[[subject, within]].drop_duplicates().groupby(subject).size().min() < 2:
        return None
    try:
        aov = AnovaRM(df_long, depvar=dv, subject=subject, within=[within]).fit()
        return aov
    except Exception:
        return None

def _pairwise_within_block_ttests(df_long: pd.DataFrame, subject: str, within: str, dv: str):
    """
    Paired t-tests for all pairs of levels in 'within', aligned by 'subject'.
    Holm-adjust p-values. Returns DataFrame of comparisons.
    """
    wide = df_long.pivot_table(index=subject, columns=within, values=dv, aggfunc="mean").dropna(axis=0, how="any")
    levels = [c for c in wide.columns if c in ALL_VARIANTS]
    pairs = [(a, b) for i, a in enumerate(levels) for b in levels[i+1:]]
    rows, pvals = [], []
    for a, b in pairs:
        t, p = stats.ttest_rel(wide[a], wide[b])
        diff = float((wide[a] - wide[b]).mean())
        rows.append({"level_a":a, "level_b":b, "t":float(t), "p_raw":float(p), "mean_diff":diff, "n":int(wide.shape[0])})
        pvals.append(p)
    # Holm step-down
    if rows:
        order = np.argsort(pvals)
        m = len(pvals)
        p_sorted = np.array(pvals)[order]
        adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p_sorted)
        adj_vals = np.clip(adj_sorted[np.argsort(order)], 0, 1.0)
        for i, val in enumerate(adj_vals):
            rows[i]["p_holm"] = float(min(1.0, val))
    return pd.DataFrame(rows)

def _summarize_variant_means(df_long: pd.DataFrame) -> pd.DataFrame:
    """Mean/SD/median/IQR per variant for a human-readable table."""
    def iqr(x):
        q = np.nanpercentile(x, [25, 75])
        return float(q[1]-q[0])
    g = df_long.groupby("variant")["value"]
    out = pd.DataFrame({
        "mean": g.mean(),
        "sd": g.std(),
        "median": g.median(),
        "iqr": g.apply(iqr),
        "n_subjects": df_long["pair"].nunique()
    }).reset_index()
    out["variant"] = out["variant"].astype(str)
    return out

def _area_level_long_from_pair_means(df_perpair_long: pd.DataFrame) -> pd.DataFrame:
    """Collapse blocks to per-pair means, keep (area, pair, variant, value)."""
    return df_perpair_long.groupby(["area","pair","variant"])["value"].mean().reset_index()

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
        pw = _pairwise_within_block_ttests(long, subject="block_id", within="variant", dv="value")
        means = long.groupby("variant")["value"].mean().sort_values()
        best_variant = str(means.index[0])
        sig_vs_others = []
        for other in [v for v in ALL_VARIANTS if v != best_variant]:
            row = pw[((pw["level_a"]==best_variant)&(pw["level_b"]==other)) |
                     ((pw["level_b"]==best_variant)&(pw["level_a"]==other))]
            if not row.empty:
                sig_vs_others.append(bool(row["p_holm"].iloc[0] < alpha))
        best_significant_all = all(sig_vs_others) if sig_vs_others else False

        long = long.assign(area=area_name)  # ensure area tag
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

    # Write per-pair ANOVA summary for corrections
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
    if area_perpair_long:
        long_all = pd.concat(area_perpair_long, ignore_index=True)
        area_long = _area_level_long_from_pair_means(long_all)
        aov_area = _anova_rm_oneway(area_long.rename(columns={"pair":"subject"}),
                                    dv="value", subject="subject", within="variant")
        if aov_area is not None:
            aov_area.anova_table.reset_index(drop=True).to_csv(corr_area_csv, index=False)
        _summarize_variant_means(area_long)[["variant","mean","sd","median","iqr","n_subjects"]] \
            .to_csv(corr_area_tbl, index=False)
        print(f"✅  Corrections area-level ANOVA written: {corr_area_csv}")
        print(f"✅  Corrections area summary table written: {corr_area_tbl}")
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
        have_both = sub.groupby("block_id")["dem"].apply(lambda s: set(s)>= {"SRTM","3DEP"})
        ok_blocks = have_both[have_both].index.tolist()
        sub_ok = sub[sub["block_id"].isin(ok_blocks)].copy()
        if sub_ok.empty:
            continue
        wide = sub_ok.pivot_table(index="block_id", columns="dem", values=metric_col, aggfunc="mean").dropna(subset=["SRTM","3DEP"])
        if wide.shape[0] < 2:
            continue
        t, pval = stats.ttest_rel(wide["3DEP"], wide["SRTM"])
        diff = float((wide["3DEP"] - wide["SRTM"]).mean())
        dem_perpair_rows.append({
            "area": area_name,
            "pair": dfp["pair"].iloc[0] if not dfp.empty else p,
            "metric": metric_col,
            "corr_used": corr_used,
            "n_blocks": int(wide.shape[0]),
            "mean_diff_3DEP_minus_SRTM": diff,
            "t": float(t), "p": float(pval),
            "best_dem": "3DEP" if diff < 0 else "SRTM",
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
        wide_pairs = long_dem.pivot_table(index="pair", columns="dem", values="value", aggfunc="mean").dropna(subset=["SRTM","3DEP"])
        if wide_pairs.shape[0] >= 2:
            t_area, p_area = stats.ttest_rel(wide_pairs["3DEP"], wide_pairs["SRTM"])
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
        corr_for_global = _area_level_long_from_pair_means(pd.concat(area_perpair_long, ignore_index=True)).assign(area=area_name)
    dem_for_global = None
    if dem_pairs_long:
        dem_for_global = pd.concat(dem_pairs_long, ignore_index=True).groupby(["area","pair","dem","corr_used"])["value"].mean().reset_index()
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

    # Across ALL AREAS — Corrections ANOVA
    if all_corr:
        corr_all = pd.concat(all_corr, ignore_index=True)
        corr_all["pair_global"] = corr_all.apply(lambda r: f"{r['area']}::{r['pair']}", axis=1)
        aov_all = _anova_rm_oneway(corr_all.rename(columns={"pair_global":"subject"}),
                                   dv="value", subject="subject", within="variant")
        if aov_all is not None:
            aov_all.anova_table.reset_index(drop=True).to_csv(out_root / "corrections_all_areas_anova.csv", index=False)
            print(f"✅  All-areas corrections ANOVA written: {out_root/'corrections_all_areas_anova.csv'}")
        else:
            print("ℹ️  All-areas corrections ANOVA could not be computed (insufficient data).")
    else:
        print("ℹ️  No corrections data accumulated across areas.")

    # Across ALL AREAS — DEM t-test (with chosen correction)
    if all_dem:
        dem_all = pd.concat(all_dem, ignore_index=True)
        corr_used = str(args.dem_corr).upper()
        wide_all = dem_all[dem_all["corr_used"]==corr_used] \
            .pivot_table(index=["area","pair"], columns="dem", values="value", aggfunc="mean") \
            .dropna(subset=["SRTM","3DEP"])
        if wide_all.shape[0] >= 2:
            t_all, p_all = stats.ttest_rel(wide_all["3DEP"], wide_all["SRTM"])
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
