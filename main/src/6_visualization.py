#!/usr/bin/env python3
"""
6_visualization.py
==================

Builds publication-ready figures *from* the metrics & TIFFs produced by
`5_accuracy_assessment.py`.

What this script creates
------------------------
A) Per-PAIR, per-AREA "accuracy vs density" (TROPO_IONO core set):
   • LS • SRTM — solid + translucent (5–95%) band
   • LS • 3DEP — solid + translucent band
   • IDW       — dashed + **hatched** band
   + Under the plot: calibrated TI(60%) & IDW(60%) maps.
   -> <area>/results/acc_den_pair_<PAIR>.png

B) Per-AREA, all pairs "accuracy vs density" (TROPO_IONO core set):
   • Same 3 lines; optionally show median (solid) &/or mean (dotted)
   -> <area>/results/acc_den_area_<AREA>.png

C) ALL-AREAS combined "accuracy vs density":
   -> <root>/results/acc_den_ALL_AREAS.png

D) Time-series boxplots (IDW • TROPO_IONO at target density):
   • Per-DEM and combined paired SRTM+3DEP spanning pair dates
   -> <area>/results/acc_period_*.png and <root>/results/acc_period_ALL_AREAS_*.png

NEW (requested)
---------------
E) Per-AREA **SRTM corrections** accuracy vs density:
   • LS • SRTM • IONO
   • LS • SRTM • TROPO_IONO
   • IDW • TROPO_IONO
   -> <area>/results/acc_den_area_SRTM_CORRS_<AREA>.png

F) Per-PAIR **SRTM corrections** accuracy vs density + **3-map row**:
   • LS • SRTM • IONO
   • LS • SRTM • TROPO_IONO
   • IDW • TROPO_IONO
   • Maps (smaller, three across):
       - cal_ti_60pct_SRTM_IONO_<PAIR>.tif
       - cal_ti_60pct_SRTM_TROPO_IONO_<PAIR>.tif
       - idw90_<PAIR>.tif (fallback: idw60_<PAIR>.tif)
   -> <area>/results/acc_den_pair_SRTM_CORRS_<PAIR>.png

Inputs expected
---------------
• Per-area metrics CSV (rebuilt by 5_* each run):
    <areas_root>/<AREA>/results/accuracy_metrics.csv
• Per-pair maps from 5_*:
    idw60_<PAIR>.tif         (and optionally idw90_<PAIR>.tif)
    cal_ti_60pct_<DEM>_<PAIR>.tif
    cal_ti_60pct_SRTM_IONO_<PAIR>.tif
    cal_ti_60pct_SRTM_TROPO_IONO_<PAIR>.tif

Uncertainty bands
-----------------
• Uses 5–95% (central 90%). Bands are **clipped to y-limits**; overflow amounts
  are annotated on the plot margin (↑ / ↓).

How to run
----------
python 6_visualization.py
python 6_visualization.py --area ENP
python 6_visualization.py --target-density 500
python 6_visualization.py --no-show-mean
python 6_visualization.py --no-show-median
python 6_visualization.py --idw-dem-density SRTM   # SRTM|3DEP|AUTO for IDW x-axis
"""

from __future__ import annotations
from pathlib import Path
import argparse, logging, os, re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import matplotlib as mpl
mpl.set_loglevel("warning")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
import matplotlib.dates as mdates
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

# -------------------------- Defaults / CLI -----------------------------------
AREAS_ROOT_DEFAULT = Path("/mnt/DATA2/bakke326l/processing/areas")
DEMS = ["SRTM", "3DEP"]
CORR_TROPO_IONO = "TROPO_IONO"
CORR_IONO = "IONO"
# IMPORTANT: we uppercase columns on load; constants must be uppercase:
METHOD_LS = "LEAST_SQUARES"
METHOD_IDW = "IDW_DHVIS"
CMAP_INV = "viridis_r"  # inverted: dark blue = larger

TARGET_DENSITY_DEFAULT = 500.0
IDW_DEM_DENSITY_DEFAULT = "SRTM"  # SRTM|3DEP|AUTO

# Percentile band for uncertainty shading (central band)
P_LOW  = 5.0   # lower percentile
P_HIGH = 95.0  # upper percentile

# Quiet GDAL/Rasterio spam
os.environ.setdefault("CPL_DEBUG", "NO")
for _n in ("rasterio", "rasterio._io", "rasterio.env", "rasterio._base", "matplotlib.font_manager"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# -------------------------- Small helpers ------------------------------------
def _pair_dates_from_tag(pair_tag: str) -> Tuple[str, str]:
    """'YYYYMMDD_YYYYMMDD' → ('YYYY-MM-DD','YYYY-MM-DD')."""
    a, b = pair_tag.split("_")
    return f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"

def _ensure_upper(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("dem", "corr", "method"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()
    return df

def _read_area_metrics(area_dir: Path) -> Optional[pd.DataFrame]:
    f = area_dir / "results" / "accuracy_metrics.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = _ensure_upper(df)
    if "density" not in df.columns:
        df["density"] = df["area_km2"] / df["n_cal"].astype(float)
    # normalize pair date strings and add tag
    for c in ("pair_ref", "pair_sec"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
    df["pair_tag"] = (
        df["pair_ref"].str.replace("-", "", regex=False) + "_" +
        df["pair_sec"].str.replace("-", "", regex=False)
    )
    return df

def _choose_idw_dem(df: pd.DataFrame, pref: str) -> Optional[str]:
    """Pick which DEM to use for the IDW density axis (SRTM preferred)."""
    if df is None or df.empty:
        return None
    present = sorted(df["dem"].unique().tolist())
    if pref.upper() in present:
        return pref.upper()
    for dem in DEMS:
        if dem in present:
            return dem
    return None

def _agg_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate rows (across replicates/pairs as given) into a curve indexed by n_cal:
      returns columns: n_cal, med_rmse, mean_rmse, p_low, p_high, med_density
    """
    if df.empty:
        return pd.DataFrame(columns=["n_cal","med_rmse","mean_rmse","p_low","p_high","med_density"])
    g = (df.groupby("n_cal", as_index=False)
           .agg(med_rmse=("rmse_cm", "median"),
                mean_rmse=("rmse_cm", "mean"),
                p_low=("rmse_cm", lambda x: np.nanpercentile(x, P_LOW)),
                p_high=("rmse_cm", lambda x: np.nanpercentile(x, P_HIGH)),
                med_density=("density", "median")))
    g.sort_values("med_density", inplace=True)
    return g

# ---------- Band clipping & annotation (prevents y-stretch from extremes) -----
def _init_clip_notes(ax):
    """Initialize per-axis counters for stacking clip annotations."""
    ax._clip_note_top = 0
    ax._clip_note_bot = 0

def _annotate_clip(ax, side: str, label: str, cm_excess: float, color: str):
    """Place a small note at top/bottom showing band exceedance beyond y-limits."""
    if not np.isfinite(cm_excess) or cm_excess <= 1e-6:
        return
    if side == "top":
        offset = 0.98 - 0.06 * ax._clip_note_top
        ax._clip_note_top += 1
        txt = f"↑ +{cm_excess:.2f} cm ({label})"
        ax.text(0.99, offset, txt, color=color, fontsize=8, ha="right", va="top",
                transform=ax.transAxes)
    elif side == "bottom":
        offset = 0.02 + 0.06 * ax._clip_note_bot
        ax._clip_note_bot += 1
        txt = f"↓ {cm_excess:.2f} cm below ({label})"
        ax.text(0.99, offset, txt, color=color, fontsize=8, ha="right", va="bottom",
                transform=ax.transAxes)

def _shade_band_clipped(ax, x, ylo, yhi, color, *,
                        hatched: bool = False, alpha: float = 0.18, z: int = 1,
                        y_min: float = 0.0, y_max: float = 1.0, label_for_note: str = ""):
    """
    Draw an uncertainty band, **clipped** to [y_min, y_max]. Any amount that would
    have extended beyond is summarized with a small note (↑ / ↓).
    """
    if x is None or len(x) == 0:
        return
    x = np.asarray(x)
    ylo = np.asarray(ylo, dtype=float)
    yhi = np.asarray(yhi, dtype=float)
    if ylo.size == 0 or yhi.size == 0:
        return

    top_excess = float(np.nanmax(np.maximum(0.0, yhi - y_max))) if yhi.size else 0.0
    bot_excess = float(np.nanmax(np.maximum(0.0, y_min - ylo))) if ylo.size else 0.0

    ylo_c = np.clip(ylo, y_min, y_max)
    yhi_c = np.clip(yhi, y_min, y_max)

    face = to_rgba(color, alpha)
    if hatched:
        ax.fill_between(x, ylo_c, yhi_c, facecolor=face,
                        edgecolor=to_rgba(color, 0.8),
                        hatch='//', linewidth=0.6, zorder=z)
    else:
        ax.fill_between(x, ylo_c, yhi_c, facecolor=face, edgecolor='none', zorder=z)

    _annotate_clip(ax, "top", label_for_note, top_excess, color)
    _annotate_clip(ax, "bottom", label_for_note, bot_excess, color)

def _top_axis_ticks(ax, area_ref: float, n_list: List[int], xmin: float, xmax: float):
    """Top x-axis with number of gauges (left many → right few)."""
    ax_top = ax.twiny()
    ax_top.set_xscale("log"); ax_top.set_xlim(xmin, xmax)
    n_sorted = sorted(set(int(n) for n in n_list if n > 0), reverse=True)
    dens_ticks = [area_ref / float(n) for n in n_sorted]
    keep = [(d, n) for d, n in zip(dens_ticks, n_sorted) if xmin <= d <= xmax]
    if keep:
        ax_top.set_xticks([d for d, _ in keep])
        ax_top.set_xticklabels([str(n) for _, n in keep])
    ax_top.set_xlabel("Number of calibration gauges")
    ax_top.tick_params(axis="x", labelsize=8)

# -------------------------- Map helpers --------------------------------------
def _read_tif_array(path: Path) -> Optional[np.ndarray]:
    if not path or not path.exists():
        return None
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(float)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            a = np.where(a == ds.nodata, np.nan, a)
        extent = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top)
        return a, extent

def _plot_maps_row_two(fig, axes_bottom, area_dir: Path, pair_tag: str):
    """Render the 60% calibrated TI and 60% IDW maps under the core per-pair plot."""
    ax1, ax2 = axes_bottom
    resdir = area_dir / "results"
    cal_ti = next(iter(resdir.glob(f"cal_ti_60pct_*_{pair_tag}.tif")), None)
    idw_tif = resdir / f"idw60_{pair_tag}.tif"

    arrays = []
    a1 = _read_tif_array(cal_ti) if cal_ti else None
    a2 = _read_tif_array(idw_tif) if idw_tif.exists() else None
    if a1: arrays.append(a1[0][np.isfinite(a1[0])])
    if a2: arrays.append(a2[0][np.isfinite(a2[0])])
    vmin, vmax = (0.0, 1.0)
    if arrays:
        vals = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        if vals.size:
            vmin, vmax = np.nanpercentile(vals, [2, 98])

    if a1:
        im = ax1.imshow(a1[0], extent=a1[1], origin='upper', cmap=CMAP_INV, vmin=vmin, vmax=vmax)
        ax1.set_title(f"Calibrated Interferogram (60%) ", fontsize=10)
        cb = plt.colorbar(im, ax=ax1, fraction=0.042, pad=0.02); cb.set_label("cm")
    else:
        ax1.text(0.5, 0.5, "Missing calibrated TI (60%)", ha="center", va="center"); ax1.set_axis_off()

    if a2:
        im = ax2.imshow(a2[0], extent=a2[1], origin='upper', cmap=CMAP_INV, vmin=vmin, vmax=vmax)
        ax2.set_title(f"Inverse Distance Weighted Watergauges (60%)", fontsize=10)
        cb = plt.colorbar(im, ax=ax2, fraction=0.042, pad=0.02); cb.set_label("cm")
    else:
        ax2.text(0.5, 0.5, "Missing IDW 60% TIFF", ha="center", va="center"); ax2.set_axis_off()

def _plot_maps_row_three(fig, axes_bottom, area_dir: Path, pair_tag: str):
    """
    Render three smaller maps (left→right):
      1) cal_ti_60pct_SRTM_IONO_<PAIR>.tif
      2) cal_ti_60pct_SRTM_TROPO_IONO_<PAIR>.tif
      3) idw90_<PAIR>.tif  (fallback: idw60_<PAIR>.tif)
    """
    ax1, ax2, ax3 = axes_bottom
    resdir = area_dir / "results"
    tif_iono = resdir / f"cal_ti_60pct_SRTM_IONO_{pair_tag}.tif"
    tif_ti   = resdir / f"cal_ti_60pct_SRTM_TROPO_IONO_{pair_tag}.tif"
    tif_idw = resdir / f"idw60_{pair_tag}.tif"

    items = [
        ("Interferogram SRTM Iono corrected (60%)", _read_tif_array(tif_iono), ax1, tif_iono.stem if tif_iono.exists() else None),
        ("Interferogram SRTM Tropo + Iono corrected (60%)", _read_tif_array(tif_ti), ax2, tif_ti.stem if tif_ti.exists() else None),
        ("Inverse Distance Weighted Watergauges (60%)", _read_tif_array(tif_idw), ax3, tif_idw.stem if tif_idw.exists() else None),
    ]

    arrays = [it[1][0][np.isfinite(it[1][0])] for it in items if it[1] is not None]
    vmin, vmax = (0.0, 1.0)
    if arrays:
        vals = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        if vals.size:
            vmin, vmax = np.nanpercentile(vals, [2, 98])

    for title, data, ax, stem in items:
        if data is None:
            ax.text(0.5, 0.5, f"Missing {title}", ha="center", va="center")
            ax.set_axis_off()
            continue
        arr, extent = data
        im = ax.imshow(arr, extent=extent, origin='upper', cmap=CMAP_INV, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}", fontsize=9)
        cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.02); cb.set_label("cm")

# ---------------------- Core per-pair acc_den (+maps) ------------------------
def plot_acc_den_pair(area_dir: Path, df_area: pd.DataFrame, pair_tag: str,
                      idw_dem_pref: str = IDW_DEM_DENSITY_DEFAULT):
    """Core TROPO_IONO trio: LS•SRTM, LS•3DEP, IDW (per pair) + two maps under it."""
    area_name = area_dir.name
    ref_iso, sec_iso = _pair_dates_from_tag(pair_tag)
    sub = df_area[(df_area["pair_ref"] == ref_iso) &
                  (df_area["pair_sec"] == sec_iso) &
                  (df_area["corr"] == CORR_TROPO_IONO)].copy()
    if sub.empty:
        print(f"⏭️  No TROPO_IONO rows for {area_name}:{pair_tag}; skipping per-pair plot.")
        return

    idw_dem = _choose_idw_dem(sub[sub["method"] == METHOD_IDW], idw_dem_pref) or "SRTM"

    curves = {}
    colors = {"SRTM": "#377eb8", "3DEP": "#4daf4a", "IDW": "#ff7f0e"}

    for dem in DEMS:
        s = sub[(sub["dem"] == dem) & (sub["method"] == METHOD_LS)].copy()
        if not s.empty:
            curves[f"LS_{dem}"] = _agg_curve(s)

    sidw = sub[(sub["dem"] == idw_dem) & (sub["method"] == METHOD_IDW)].copy()
    if not sidw.empty:
        curves["IDW"] = _agg_curve(sidw)

    if not curves:
        print(f"⏭️  No curves for {area_name}:{pair_tag}."); return

    # Bigger maps & closer together: 2 rows, tighter wspace, taller bottom row
    fig = plt.figure(figsize=(13.2, 8.8), dpi=140, constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[3.0, 2.6], hspace=0.28, wspace=0.06)
    ax = fig.add_subplot(gs[0, :])

    # Robust y-limit from **median** lines only
    med_max = max((g["med_rmse"].max() for g in curves.values() if not g.empty), default=1.0)
    y_min, y_max = 0.0, float(med_max * 1.15 if np.isfinite(med_max) and med_max > 0 else 1.0)
    ax.set_ylim(y_min, y_max)
    
    _init_clip_notes(ax)

    # Plot lines + clipped bands
    all_n = set()
    for key, g in curves.items():
        if key == "IDW":
            ax.plot(g["med_density"], g["med_rmse"], "--", color=colors["IDW"], lw=1.9, label="IDW")
            _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], colors["IDW"],
                                hatched=True, alpha=0.14, z=1, y_min=y_min, y_max=y_max,
                                label_for_note="IDW")
        else:
            dem = key.split("_", 1)[1]
            label = f"Interferogram • {dem}"
            ax.plot(g["med_density"], g["med_rmse"], "-", color=colors[dem], lw=1.9, label=label)
            _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], colors[dem],
                                hatched=False, alpha=0.22, z=1, y_min=y_min, y_max=y_max,
                                label_for_note=label)
        if "n_cal" in g.columns:
            all_n.update(g["n_cal"].tolist())

    xmin = min(g["med_density"].min() for g in curves.values())
    xmax = max(g["med_density"].max() for g in curves.values())
    ax.set_xscale("log"); ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)*0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("RMSE (cm)")
    ax.set_xlabel("Gauge density (km² per gauge) (log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(f"{area_name} — {pair_tag} — Error vs Density — SRTM vs 3DEP ", fontsize=12)
    ax.legend(ncols=3, fontsize=9)

    area_ref = float(sidw["area_km2"].median()) if not sidw.empty else float(sub["area_km2"].median())
    _top_axis_ticks(ax, area_ref, sorted(list(all_n), reverse=True), xmin, xmax)

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    _plot_maps_row_two(fig, (ax1, ax2), area_dir, pair_tag)

    out = area_dir / "results" / f"acc_den_pair_DEM_{pair_tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Pair acc_den+maps written: {out}")

# ---------- NEW: Per-pair SRTM corrs (IONO vs TROPO_IONO) + 3-map row --------
def plot_acc_den_pair_srtm_corrs(area_dir: Path, df_area: pd.DataFrame, pair_tag: str,
                                 idw_dem_pref: str = "SRTM"):
    """Per-pair comparison: LS•SRTM(IONO), LS•SRTM(TROPO_IONO), IDW(TROPO_IONO) + 3 maps."""
    area_name = area_dir.name
    ref_iso, sec_iso = _pair_dates_from_tag(pair_tag)
    sub = df_area[(df_area["pair_ref"] == ref_iso) &
                  (df_area["pair_sec"] == sec_iso)].copy()
    if sub.empty:
        print(f"⏭️  No rows for {area_name}:{pair_tag}; skipping SRTM-corrs pair plot.")
        return

    s_iono = sub[(sub["dem"] == "SRTM") & (sub["method"] == METHOD_LS) & (sub["corr"] == CORR_IONO)]
    s_ti   = sub[(sub["dem"] == "SRTM") & (sub["method"] == METHOD_LS) & (sub["corr"] == CORR_TROPO_IONO)]
    idw_dem = _choose_idw_dem(sub[sub["method"] == METHOD_IDW], idw_dem_pref) or "SRTM"
    s_idw  = sub[(sub["dem"] == idw_dem) & (sub["method"] == METHOD_IDW) & (sub["corr"] == CORR_TROPO_IONO)]

    curves = {}
    if not s_iono.empty: curves["SRTM_IONO"] = _agg_curve(s_iono)
    if not s_ti.empty:   curves["SRTM_TI"]   = _agg_curve(s_ti)
    if not s_idw.empty:  curves["IDW_TI"]    = _agg_curve(s_idw)

    if not curves:
        print(f"⏭️  No curves for {area_name}:{pair_tag} SRTM-corrs."); return

    # Colors: distinguish the two SRTM corrs
    c_iono = "#1b9e77"   # greenish
    c_ti   = "#e7298a"   # magenta
    c_idw  = "#ff7f0e"   # orange

    fig = plt.figure(figsize=(13.6, 9.0), dpi=140, constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[3.0, 2.4], hspace=0.28, wspace=0.06)
    ax = fig.add_subplot(gs[0, :])

    med_max = max((g["med_rmse"].max() for g in curves.values() if not g.empty), default=1.0)
    y_min, y_max = 0.0, float(med_max * 1.15 if np.isfinite(med_max) and med_max > 0 else 1.0)
    ax.set_ylim(y_min, y_max)
    _init_clip_notes(ax)

    all_n = set()
    if "SRTM_IONO" in curves:
        g = curves["SRTM_IONO"]
        ax.plot(g["med_density"], g["med_rmse"], "-", color=c_iono, lw=1.9, label="Interferogram • Iono")
        _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], c_iono,
                            hatched=False, alpha=0.22, z=1, y_min=y_min, y_max=y_max,
                            label_for_note="Interferogram Iono")
        if "n_cal" in g.columns: all_n.update(g["n_cal"].tolist())

    if "SRTM_TI" in curves:
        g = curves["SRTM_TI"]
        ax.plot(g["med_density"], g["med_rmse"], "-", color=c_ti, lw=1.9, label="Interferogram • TROPO+IONO")
        _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], c_ti,
                            hatched=False, alpha=0.22, z=1, y_min=y_min, y_max=y_max,
                            label_for_note="Interferogram Tropo + Iono")
        if "n_cal" in g.columns: all_n.update(g["n_cal"].tolist())

    if "IDW_TI" in curves:
        g = curves["IDW_TI"]
        ax.plot(g["med_density"], g["med_rmse"], "--", color=c_idw, lw=1.9, label="IDW")
        _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], c_idw,
                            hatched=True, alpha=0.14, z=1, y_min=y_min, y_max=y_max,
                            label_for_note="IDW")
        if "n_cal" in g.columns: all_n.update(g["n_cal"].tolist())

    xmin = min(g["med_density"].min() for g in curves.values())
    xmax = max(g["med_density"].max() for g in curves.values())
    ax.set_xscale("log"); ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)*0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("RMSE (cm)")
    ax.set_xlabel("Gauge density (km² per gauge) (log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(f"{area_name} — {pair_tag} — Error vs Density — With vs Without Tropospheric Correction (SRTM)", fontsize=12)
    ax.legend(ncols=3, fontsize=9)

    area_ref = float(s_idw["area_km2"].median()) if not s_idw.empty else float(sub["area_km2"].median())
    _top_axis_ticks(ax, area_ref, sorted(list(all_n), reverse=True), xmin, xmax)

    # Three smaller maps under the plot
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])
    _plot_maps_row_three(fig, (ax1, ax2, ax3), area_dir, pair_tag)

    out = area_dir / "results" / f"acc_den_pair_SRTM_CORRS_{pair_tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Pair SRTM-corrs acc_den+3maps written: {out}")

# ---------------------- Per-area, all pairs acc_den (core) -------------------
def plot_acc_den_area(area_dir: Path, df_area: pd.DataFrame, show_mean: bool, show_median: bool,
                      idw_dem_pref: str = IDW_DEM_DENSITY_DEFAULT):
    """Core TROPO_IONO trio: LS•SRTM, LS•3DEP, IDW aggregated across pairs."""
    area_name = area_dir.name
    sub = df_area[df_area["corr"] == CORR_TROPO_IONO].copy()
    if sub.empty:
        print(f"⏭️  No TROPO_IONO rows for {area_name}; skipping area acc_den.")
        return

    idw_dem = _choose_idw_dem(sub[sub["method"] == METHOD_IDW], idw_dem_pref) or "SRTM"

    curves = {}
    colors = {"SRTM": "#377eb8", "3DEP": "#4daf4a", "IDW": "#ff7f0e"}

    for dem in DEMS:
        s = sub[(sub["dem"] == dem) & (sub["method"] == METHOD_LS)]
        if not s.empty:
            curves[f"LS_{dem}"] = _agg_curve(s)

    sidw = sub[(sub["dem"] == idw_dem) & (sub["method"] == METHOD_IDW)]
    if not sidw.empty:
        curves["IDW"] = _agg_curve(sidw)

    if not curves:
        print(f"⏭️  No curves for {area_name}; skipping area acc_den.")
        return

    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=140, constrained_layout=True)

    med_max = max((g["med_rmse"].max() for g in curves.values() if not g.empty), default=1.0)
    y_min, y_max = 0.0, float(med_max * 1.15 if np.isfinite(med_max) and med_max > 0 else 1.0)
    ax.set_ylim(y_min, y_max)
    _init_clip_notes(ax)

    all_n = set()
    for key, g in curves.items():
        if key == "IDW":
            if show_median:
                ax.plot(g["med_density"], g["med_rmse"], "--", color=colors["IDW"], lw=1.9, label="IDW median")
            if show_mean:
                ax.plot(g["med_density"], g["mean_rmse"], ":", color=colors["IDW"], lw=1.7, label="IDW mean")
            _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], colors["IDW"],
                                hatched=True, alpha=0.14, z=1, y_min=y_min, y_max=y_max,
                                label_for_note="IDW")
        else:
            dem = key.split("_", 1)[1]
            if show_median:
                ax.plot(g["med_density"], g["med_rmse"], "-", color=colors[dem], lw=1.9, label=f"Interferogram • {dem} median")
            if show_mean:
                ax.plot(g["med_density"], g["mean_rmse"], ":", color=colors[dem], lw=1.7, label=f"Interferogram • {dem} mean")
            _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], colors[dem],
                                hatched=False, alpha=0.22, z=1, y_min=y_min, y_max=y_max,
                                label_for_note=f"{dem}")
        if "n_cal" in g.columns:
            all_n.update(g["n_cal"].tolist())

    xmin = min(g["med_density"].min() for g in curves.values())
    xmax = max(g["med_density"].max() for g in curves.values())
    ax.set_xscale("log"); ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)*0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("RMSE (cm)")
    ax.set_xlabel("Gauge density (km² per gauge) (log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(f"{area_name} — Accuracy vs Density — SRTM vs 3DEP — All Pairs")
    ax.legend(ncols=3, fontsize=9)

    area_ref = float(sidw["area_km2"].median()) if not sidw.empty else float(sub["area_km2"].median())
    _top_axis_ticks(ax, area_ref, sorted(list(all_n), reverse=True), xmin, xmax)

    out = area_dir / "results" / f"acc_den_area_{area_name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"📈 Area acc_den written: {out}")

# ---------- NEW: Per-area SRTM corrs (IONO vs TROPO_IONO) --------------------
def plot_acc_den_area_srtm_corrs(area_dir: Path, df_area: pd.DataFrame,
                                 show_mean: bool, show_median: bool,
                                 idw_dem_pref: str = "SRTM"):
    """Area-level comparison: LS•SRTM(IONO), LS•SRTM(TROPO_IONO), IDW(TROPO_IONO)."""
    area_name = area_dir.name
    sub = df_area.copy()
    s_iono = sub[(sub["dem"] == "SRTM") & (sub["method"] == METHOD_LS) & (sub["corr"] == CORR_IONO)]
    s_ti   = sub[(sub["dem"] == "SRTM") & (sub["method"] == METHOD_LS) & (sub["corr"] == CORR_TROPO_IONO)]
    idw_dem = _choose_idw_dem(sub[sub["method"] == METHOD_IDW], idw_dem_pref) or "SRTM"
    s_idw  = sub[(sub["dem"] == idw_dem) & (sub["method"] == METHOD_IDW) & (sub["corr"] == CORR_TROPO_IONO)]

    if s_iono.empty and s_ti.empty and s_idw.empty:
        print(f"⏭️  No SRTM-corr rows for {area_name}; skipping SRTM-corrs area acc_den.")
        return

    c_iono = "#1b9e77"
    c_ti   = "#e7298a"
    c_idw  = "#ff7f0e"

    curves = {}
    if not s_iono.empty: curves["SRTM_IONO"] = _agg_curve(s_iono)
    if not s_ti.empty:   curves["SRTM_TI"]   = _agg_curve(s_ti)
    if not s_idw.empty:  curves["IDW_TI"]    = _agg_curve(s_idw)

    fig, ax = plt.subplots(figsize=(10.8, 6.0), dpi=140, constrained_layout=True)

    med_max = max((g["med_rmse"].max() for g in curves.values() if not g.empty), default=1.0)
    y_min, y_max = 0.0, float(med_max * 1.15 if np.isfinite(med_max) and med_max > 0 else 1.0)
    ax.set_ylim(y_min, y_max)
    _init_clip_notes(ax)

    all_n = set()
    if "SRTM_IONO" in curves:
        g = curves["SRTM_IONO"]
        if show_median: ax.plot(g["med_density"], g["med_rmse"], "-", color=c_iono, lw=1.9, label="Iono median")
        if show_mean:   ax.plot(g["med_density"], g["mean_rmse"], ":", color=c_iono, lw=1.7, label="Iono mean")
        _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], c_iono,
                            hatched=False, alpha=0.22, z=1, y_min=y_min, y_max=y_max,
                            label_for_note="Iono")
        if "n_cal" in g.columns: all_n.update(g["n_cal"].tolist())

    if "SRTM_TI" in curves:
        g = curves["SRTM_TI"]
        if show_median: ax.plot(g["med_density"], g["med_rmse"], "-", color=c_ti, lw=1.9, label="Tropo + Iono median")
        if show_mean:   ax.plot(g["med_density"], g["mean_rmse"], ":", color=c_ti, lw=1.7, label="Tropo + Iono mean")
        _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], c_ti,
                            hatched=False, alpha=0.22, z=1, y_min=y_min, y_max=y_max,
                            label_for_note="Tropo + Iono")
        if "n_cal" in g.columns: all_n.update(g["n_cal"].tolist())

    if "IDW_TI" in curves:
        g = curves["IDW_TI"]
        if show_median: ax.plot(g["med_density"], g["med_rmse"], "--", color=c_idw, lw=1.9, label="IDW median")
        if show_mean:   ax.plot(g["med_density"], g["mean_rmse"], ":", color=c_idw, lw=1.7, label="IDW mean")
        _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], c_idw,
                            hatched=True, alpha=0.14, z=1, y_min=y_min, y_max=y_max,
                            label_for_note="IDW")
        if "n_cal" in g.columns: all_n.update(g["n_cal"].tolist())

    xmin = min(g["med_density"].min() for g in curves.values())
    xmax = max(g["med_density"].max() for g in curves.values())
    ax.set_xscale("log"); ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)*0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("RMSE (cm)")
    ax.set_xlabel("Gauge density (km² per gauge) (log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(f"{area_name} — Error vs Density — With vs Without Tropospheric Correction (SRTM) — All Pairs")
    ax.legend(ncols=3, fontsize=9)

    area_ref = float(s_idw["area_km2"].median()) if not s_idw.empty else float(sub["area_km2"].median())
    _top_axis_ticks(ax, area_ref, sorted(list(all_n), reverse=True), xmin, xmax)

    out = area_dir / "results" / f"acc_den_area_SRTM_CORRS_{area_name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"📈 Area SRTM-corrs acc_den written: {out}")

# ------------------- All-areas combined acc_den (one plot) -------------------
def plot_acc_den_all_areas(root: Path, df_all: pd.DataFrame, show_mean: bool, show_median: bool,
                           idw_dem_pref: str = IDW_DEM_DENSITY_DEFAULT):
    if df_all.empty:
        print("⏭️  No metrics loaded for ALL-AREAS acc_den."); return
    sub = df_all[df_all["corr"] == CORR_TROPO_IONO].copy()
    if sub.empty:
        print("⏭️  No TROPO_IONO rows for ALL-AREAS acc_den."); return

    idw_dem = _choose_idw_dem(sub[sub["method"] == METHOD_IDW], idw_dem_pref) or "SRTM"

    curves = {}
    colors = {"SRTM": "#377eb8", "3DEP": "#4daf4a", "IDW": "#ff7f0e"}

    for dem in DEMS:
        s = sub[(sub["dem"] == dem) & (sub["method"] == METHOD_LS)]
        if not s.empty:
            curves[f"LS_{dem}"] = _agg_curve(s)

    sidw = sub[(sub["dem"] == idw_dem) & (sub["method"] == METHOD_IDW)]
    if not sidw.empty:
        curves["IDW"] = _agg_curve(sidw)

    if not curves:
        print("⏭️  No curves in ALL-AREAS acc_den."); return

    fig, ax = plt.subplots(figsize=(11.6, 6.3), dpi=140, constrained_layout=True)

    med_max = max((g["med_rmse"].max() for g in curves.values() if not g.empty), default=1.0)
    y_min, y_max = 0.0, float(med_max * 1.15 if np.isfinite(med_max) and med_max > 0 else 1.0)
    ax.set_ylim(y_min, y_max)
    _init_clip_notes(ax)

    all_n = set()
    for key, g in curves.items():
        if key == "IDW":
            if show_median:
                ax.plot(g["med_density"], g["med_rmse"], "--", color=colors["IDW"], lw=1.9, label="IDW median")
            if show_mean:
                ax.plot(g["med_density"], g["mean_rmse"], ":", color=colors["IDW"], lw=1.7, label="IDW mean")
            _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], colors["IDW"],
                                hatched=True, alpha=0.14, z=1, y_min=y_min, y_max=y_max,
                                label_for_note="IDW")
        else:
            dem = key.split("_", 1)[1]
            if show_median:
                ax.plot(g["med_density"], g["med_rmse"], "-", color=colors[dem], lw=1.9, label=f"{dem} median")
            if show_mean:
                ax.plot(g["med_density"], g["mean_rmse"], ":", color=colors[dem], lw=1.7, label=f"{dem} mean")
            _shade_band_clipped(ax, g["med_density"], g["p_low"], g["p_high"], colors[dem],
                                hatched=False, alpha=0.22, z=1, y_min=y_min, y_max=y_max,
                                label_for_note=f"Interferogram • {dem}")
        if "n_cal" in g.columns:
            all_n.update(g["n_cal"].tolist())

    xmin = min(g["med_density"].min() for g in curves.values())
    xmax = max(g["med_density"].max() for g in curves.values())
    ax.set_xscale("log"); ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)*0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("RMSE (cm)")
    ax.set_xlabel("Gauge density (km² per gauge) (log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("All Areas — Accuracy vs Density")
    ax.legend(ncols=3, fontsize=9)

    area_ref = float(sidw["area_km2"].median()) if not sidw.empty else float(sub["area_km2"].median())
    _top_axis_ticks(ax, area_ref, sorted(list(all_n), reverse=True), xmin, xmax)

    outdir = root / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "acc_den_ALL_AREAS.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"📈 ALL-AREAS acc_den written: {out}")

# ------------------------ Time-series boxplots (area) ------------------------
def _pick_rows_for_target_density(df: pd.DataFrame, target_km2: float) -> pd.DataFrame:
    """For each (pair, dem), choose rows for the n_cal closest to 'target_km2' (keep all replicates)."""
    df = df.copy()
    df["density"] = df["area_km2"] / df["n_cal"].astype(float)
    picks = []
    for (pref, psec, dem), g in df.groupby(["pair_ref", "pair_sec", "dem"], as_index=False):
        gagg = (g.groupby("n_cal", as_index=False)
                  .agg(med_density=("density","median")))
        if gagg.empty:
            continue
        gagg["d_abs"] = (gagg["med_density"] - target_km2).abs()
        choose_n = int(gagg.loc[gagg["d_abs"].idxmin(), "n_cal"])
        picks.append(g[g["n_cal"] == choose_n])
    if not picks:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(picks, ignore_index=True)

def plot_period_area(area_dir: Path, df_area: pd.DataFrame, target_km2: float):
    area_name = area_dir.name
    sub = df_area[(df_area["method"] == METHOD_LS) & (df_area["corr"] == CORR_TROPO_IONO)].copy()
    if sub.empty:
        print(f"⏭️  No LS TROPO_IONO rows for {area_name}; skipping period plots.")
        return
    picked = _pick_rows_for_target_density(sub, target_km2)
    if picked.empty:
        print(f"⏭️  No rows near {target_km2:g} km²/g for {area_name}; skipping period plots.")
        return

    def _draw_one_dem(dem: str, out: Path):
        g = picked[picked["dem"] == dem]
        if g.empty:
            print(f"⏭️  No {dem} rows for {area_name} period plot."); return

        to_dt = lambda s: pd.to_datetime(s, format="%Y-%m-%d")
        meta = (g[["pair_ref","pair_sec"]].drop_duplicates()
                   .assign(t_ref=lambda d: to_dt(d["pair_ref"]),
                           t_sec=lambda d: to_dt(d["pair_sec"]),
                           t_mid=lambda d: d["t_ref"] + (d["t_sec"] - d["t_ref"])/2,
                           span_days=lambda d: (d["t_sec"] - d["t_ref"]).dt.days.clip(lower=1).astype(float))
                   .sort_values("t_mid"))
        data, pos, widths = [], [], []
        for _, r in meta.iterrows():
            vals = g[(g["pair_ref"]==r["pair_ref"]) & (g["pair_sec"]==r["pair_sec"])]["rmse_cm"].to_numpy()
            if vals.size == 0: continue
            data.append(vals)
            pos.append(mdates.date2num(r["t_mid"]))
            widths.append(float(r["span_days"]))

        fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=140, constrained_layout=True)
        bp = ax.boxplot(data, positions=pos, widths=widths, whis=1.5, patch_artist=True, showfliers=True)
        c = "#377eb8" if dem == "SRTM" else "#4daf4a"
        for box in bp["boxes"]:    box.set(facecolor=to_rgba(c,0.30), edgecolor=c, linewidth=1.2)
        for whisk in bp["whiskers"]: whisk.set(color=c, linewidth=1.2)
        for cap in bp["caps"]:       cap.set(color=c, linewidth=1.2)
        for med in bp["medians"]:    med.set(color="k", linewidth=1.8)
        for fl in bp["fliers"]:      fl.set(marker="o", ms=3.5, mfc=c, mec="white", alpha=0.85)

        ax.set_ylabel("RMSE (cm) — box: IQR; whiskers: 1.5xIQR; dots: outliers")
        ax.set_xlabel("Time (box width spans the pair dates)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.set_title(f"{area_name} — {dem} — Interferogram at ≈ {target_km2:g} km² per Gauge Calibration")
        fig.autofmt_xdate()
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"📈 Period (per-DEM) written: {out}")

    out_s = area_dir / "results" / f"acc_period_{area_name}_SRTM_{str(target_km2).replace('.','p')}.png"
    out_3 = area_dir / "results" / f"acc_period_{area_name}_3DEP_{str(target_km2).replace('.','p')}.png"
    _draw_one_dem("SRTM", out_s)
    _draw_one_dem("3DEP", out_3)

    g = picked.copy()
    to_dt = lambda s: pd.to_datetime(s, format="%Y-%m-%d")
    meta = (g[["pair_ref","pair_sec"]].drop_duplicates()
              .assign(t_ref=lambda d: to_dt(d["pair_ref"]),
                      t_sec=lambda d: to_dt(d["pair_sec"]),
                      t_mid=lambda d: d["t_ref"] + (d["t_sec"] - d["t_ref"])/2,
                      span_days=lambda d: (d["t_sec"] - d["t_ref"]).dt.days.clip(lower=1).astype(float))
              .sort_values("t_mid"))
    if meta.empty:
        return
    fig, ax = plt.subplots(figsize=(12.0, 5.6), dpi=140, constrained_layout=True)
    colors = {"SRTM": "#377eb8", "3DEP": "#4daf4a"}
    hatches = {"SRTM": "///", "3DEP": "\\\\\\"}

    for _, row in meta.iterrows():
        xmid = mdates.date2num(row["t_mid"]); span = float(row["span_days"])
        gap = span * 0.04
        w_each = (span - gap) / 2.0
        x_srtm = xmid - (gap/2 + w_each/2)
        x_3dep = xmid + (gap/2 + w_each/2)

        for dem, xpos in (("SRTM", x_srtm), ("3DEP", x_3dep)):
            vals = g[(g["pair_ref"]==row["pair_ref"]) & (g["pair_sec"]==row["pair_sec"]) & (g["dem"]==dem)]["rmse_cm"].to_numpy()
            if vals.size == 0: 
                continue
            bp = ax.boxplot([vals], positions=[xpos], widths=[w_each], whis=1.5, patch_artist=True, showfliers=True)
            for box in bp["boxes"]: box.set(facecolor=to_rgba(colors[dem],0.30), edgecolor=colors[dem], linewidth=1.2, hatch=hatches[dem])
            for whisk in bp["whiskers"]: whisk.set(color=colors[dem], linewidth=1.2)
            for cap in bp["caps"]:       cap.set(color=colors[dem], linewidth=1.2)
            for med in bp["medians"]:    med.set(color="k", linewidth=1.8)
            for fl in bp["fliers"]:      fl.set(marker="o", ms=3.5, mfc=colors[dem], mec="white", alpha=0.85)

    ax.set_ylabel("RMSE (cm) — box: IQR; whiskers: 1.5xIQR; dots: outliers")
    ax.set_xlabel("Time (paired boxes span each pair's dates)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.legend(handles=[Patch(facecolor=to_rgba(colors["SRTM"],0.30), edgecolor=colors["SRTM"], hatch=hatches["SRTM"], label="SRTM"),
                       Patch(facecolor=to_rgba(colors["3DEP"],0.30), edgecolor=colors["3DEP"], hatch=hatches["3DEP"], label="3DEP")],
              loc="upper right")
    ax.set_title(f"{area_name} — Interferogram at ≈ {target_km2:g} km² per Gauge Calibration — Combined DEMs")
    fig.autofmt_xdate()
    out = area_dir / "results" / f"acc_period_{area_name}_COMBINED_{str(target_km2).replace('.','p')}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"📈 Period (combined) written: {out}")

# ------------------- All-areas combined time-series boxplot ------------------
def plot_period_all_areas(root: Path, df_all: pd.DataFrame, target_km2: float):
    sub = df_all[(df_all["method"] == METHOD_LS) & (df_all["corr"] == CORR_TROPO_IONO)].copy()
    if sub.empty:
        print("⏭️  No LS TROPO_IONO rows for ALL-AREAS period plot."); return
    picked = _pick_rows_for_target_density(sub, target_km2)
    if picked.empty:
        print("⏭️  No rows near target density for ALL-AREAS period plot."); return

    to_dt = lambda s: pd.to_datetime(s, format="%Y-%m-%d")
    meta = (picked[["area","pair_ref","pair_sec"]].drop_duplicates()
              .assign(t_ref=lambda d: to_dt(d["pair_ref"]),
                      t_sec=lambda d: to_dt(d["pair_sec"]),
                      t_mid=lambda d: d["t_ref"] + (d["t_sec"] - d["t_ref"])/2,
                      span_days=lambda d: (d["t_sec"] - d["t_ref"]).dt.days.clip(lower=1).astype(float))
              .sort_values("t_mid"))
    if meta.empty:
        return

    fig, ax = plt.subplots(figsize=(13.0, 6.2), dpi=140, constrained_layout=True)
    colors = {"SRTM": "#377eb8", "3DEP": "#4daf4a"}
    hatches = {"SRTM": "///", "3DEP": "\\\\\\"}

    for _, row in meta.iterrows():
        xmid = mdates.date2num(row["t_mid"]); span = float(row["span_days"])
        gap = span * 0.04
        w_each = (span - gap) / 2.0
        x_srtm = xmid - (gap/2 + w_each/2)
        x_3dep = xmid + (gap/2 + w_each/2)

        for dem, xpos in (("SRTM", x_srtm), ("3DEP", x_3dep)):
            vals = picked[(picked["pair_ref"]==row["pair_ref"]) &
                          (picked["pair_sec"]==row["pair_sec"]) &
                          (picked["dem"]==dem) &
                          (picked["area"]==row["area"])]["rmse_cm"].to_numpy()
            if vals.size == 0:
                continue
            bp = ax.boxplot([vals], positions=[xpos], widths=[w_each], whis=1.5, patch_artist=True, showfliers=True)
            for box in bp["boxes"]: box.set(facecolor=to_rgba(colors[dem],0.30), edgecolor=colors[dem], linewidth=1.2, hatch=hatches[dem])
            for whisk in bp["whiskers"]: whisk.set(color=colors[dem], linewidth=1.2)
            for cap in bp["caps"]:       cap.set(color=colors[dem], linewidth=1.2)
            for med in bp["medians"]:    med.set(color="k", linewidth=1.8)
            for fl in bp["fliers"]:      fl.set(marker="o", ms=3.5, mfc=colors[dem], mec="white", alpha=0.85)

    ax.set_ylabel("RMSE (cm) — box: IQR; whiskers: 1.5xIQR; dots: outliers")
    ax.set_xlabel("Time (paired boxes span each pair's dates)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.legend(handles=[Patch(facecolor=to_rgba(colors["SRTM"],0.30), edgecolor=colors["SRTM"], hatch=hatches["SRTM"], label="SRTM"),
                       Patch(facecolor=to_rgba(colors["3DEP"],0.30), edgecolor=colors["3DEP"], hatch=hatches["3DEP"], label="3DEP")],
              loc="upper right")
    ax.set_title(f"All Areas — Interferogram at ≈ {target_km2:g} km² per Gauge Calibration — Combined DEMs")
    fig.autofmt_xdate()

    outdir = root / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"acc_period_ALL_AREAS_{str(target_km2).replace('.','p')}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"📈 ALL-AREAS period written: {out}")

# ----------------------------------- CLI -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize accuracy metrics and exported TIFFs (with clipped 5–95% uncertainty bands).")
    ap.add_argument("--areas-root", type=str, default=str(AREAS_ROOT_DEFAULT),
                    help="Root folder containing per-area subfolders (default: %(default)s)")
    ap.add_argument("--area", type=str,
                    help="Only process this AREA (subfolder name under --areas-root).")
    ap.add_argument("--target-density", type=float, default=TARGET_DENSITY_DEFAULT,
                    help="Target density (km²/gauge) for period boxplots (closest n_cal is chosen).")
    ap.add_argument("--idw-dem-density", type=str, default=IDW_DEM_DENSITY_DEFAULT,
                    choices=["SRTM", "3DEP", "AUTO"],
                    help="Which DEM's density to use for IDW line's x-axis (default: %(default)s).")
    ap.add_argument("--no-show-mean", action="store_true", help="Hide mean lines on area/all-areas acc_den.")
    ap.add_argument("--no-show-median", action="store_true", help="Hide median lines on area/all-areas acc_den.")
    args = ap.parse_args()

    root = Path(args.areas_root)
    targets = [root / args.area] if args.area else sorted([d for d in root.iterdir() if d.is_dir()])

    # Load per-area metrics
    area_metrics: Dict[str, pd.DataFrame] = {}
    for area_dir in targets:
        df = _read_area_metrics(area_dir)
        if df is None or df.empty:
            print(f"⏭️  No metrics for area: {area_dir.name}")
            continue
        area_metrics[area_dir.name] = df

    if not area_metrics:
        print("⏭️  Nothing to visualize."); return

    show_mean   = not args.no_show_mean
    show_median = not args.no_show_median

    # Per-area visualizations
    for area_name, df_area in area_metrics.items():
        area_dir = root / area_name

        # 1) Per-pair acc_den (core trio) + maps
        resdir = area_dir / "results"
        pairs = sorted({
            m.group(1)
            for p in resdir.glob("idw60_*.tif")
            for m in [re.match(r"idw60_(\d{8}_\d{8})\.tif", p.name)]
            if m
        }) or sorted(set(
            (df_area["pair_ref"].astype(str).str.replace("-", "", regex=False) + "_" +
             df_area["pair_sec"].astype(str).str.replace("-", "", regex=False)).tolist()
        ))

        for pair_tag in pairs:
            try:
                plot_acc_den_pair(area_dir, df_area, pair_tag, idw_dem_pref=args.idw_dem_density)
            except Exception as e:
                print(f"⚠️  Pair plot failed for {area_name}:{pair_tag} (core): {e}")
            try:
                plot_acc_den_pair_srtm_corrs(area_dir, df_area, pair_tag, idw_dem_pref="SRTM")
            except Exception as e:
                print(f"⚠️  Pair plot failed for {area_name}:{pair_tag} (SRTM corrs): {e}")

        # 2) Per-area acc_den (core)
        try:
            plot_acc_den_area(area_dir, df_area, show_mean=show_mean, show_median=show_median,
                              idw_dem_pref=args.idw_dem_density)
        except Exception as e:
            print(f"⚠️  Area acc_den failed for {area_name} (core): {e}")

        # 3) Per-area acc_den (SRTM corrections)
        try:
            plot_acc_den_area_srtm_corrs(area_dir, df_area, show_mean=show_mean, show_median=show_median,
                                         idw_dem_pref="SRTM")
        except Exception as e:
            print(f"⚠️  Area acc_den failed for {area_name} (SRTM corrs): {e}")

        # 4) Period plots (per-DEM and combined)
        try:
            plot_period_area(area_dir, df_area, target_km2=args.target_density)
        except Exception as e:
            print(f"⚠️  Area period plots failed for {area_name}: {e}")

    # ---------------- All-areas combined figures ----------------
    try:
        df_all = pd.concat(area_metrics.values(), ignore_index=True)
    except Exception:
        df_all = pd.DataFrame()

    if not df_all.empty:
        try:
            plot_acc_den_all_areas(root, df_all, show_mean=show_mean, show_median=show_median,
                                   idw_dem_pref=args.idw_dem_density)
        except Exception as e:
            print(f"⚠️  ALL-AREAS acc_den failed: {e}")

        try:
            plot_period_all_areas(root, df_all, target_km2=args.target_density)
        except Exception as e:
            print(f"⚠️  ALL-AREAS period failed: {e}")

if __name__ == "__main__":
    main()
