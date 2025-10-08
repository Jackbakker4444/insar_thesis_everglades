#!/usr/bin/env python3
"""
8_visualization.py (density-only)
=================================

From:
- 7_accuracy_assessment_density.py   -> <area>/results/accuracy_metrics_density_SRTM_RAW.csv

Creates:
C1) Per-pair density visualization (RAW): RMSE vs density, Bias vs density (no elbows)
    • Bottom row with three big panels: RAW LS (calibrated), Satellite AOI (middle), RAW IDW
    • Gauges overlay on all three bottom panels (calibration=black, validation=red)
    • Tunable vertical distance between curves and bottom maps (--maps-gap-frac)
    • Tunable bottom-map height (--maps-height-frac)
    -> <area>/results/density_raw_idw_pair_<PAIR>.png

C2) All-areas density vs RMSE overview (RAW):
    • Scatter of *all* points from all areas
    • Median line + 5–95% uncertainty band for LS and for IDW
    -> <areas-root>/results/density_all_areas_raw.png
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
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

# Optional deps
try:
    import contextily as cx
except Exception:
    cx = None
try:
    import geopandas as gpd
except Exception:
    gpd = None
try:
    from shapely.geometry import box as shapely_box
except Exception:
    shapely_box = None
try:
    from pyproj import Geod
except Exception:
    Geod = None

# -------------------------- Defaults / CLI -----------------------------------
AREAS_ROOT_DEFAULT = Path("/mnt/DATA2/bakke326l/processing/areas")

METHOD_LS  = "LEAST_SQUARES"
METHOD_IDW = "IDW_DHVIS"

CB = {
    "black":  "#000000",
    "blue":   "#0072B2",
    "green":  "#009E73",
    "orange": "#D55E00",
    "pink":   "#CC79A7",
    "yellow": "#F0E442",
}
COLORS_DEM  = {"SRTM": CB["blue"], "3DEP": CB["green"]}
COLOR_IDW   = CB["pink"]
CMAP_INV    = "viridis_r"

# Uncertainty bands (central 90%)
P_LOW, P_HIGH = 5.0, 95.0

# Water overlay & satellite defaults
DEF_WATER_AREAS = "/home/bakke326l/InSAR/main/data/vector/water_areas.geojson"
DEF_SAT_PROVIDER = "Esri.WorldImagery"
DEF_SAT_URL = ""  # custom XYZ if you want

# Gauge overlay search pattern (per pair)
GAUGE_NAME_CANDIDATES = [
    "gauges_split_60pct_{pair}.geojson",
    "gauges_split_{pair}.geojson",
    "gauges_{pair}.geojson",
    "split_60_40_{pair}.geojson",
    "split_*_{pair}.geojson",
]

# Quiet noisy libs
os.environ.setdefault("CPL_DEBUG", "NO")
for _n in ("rasterio", "rasterio._io", "rasterio.env", "rasterio._base", "matplotlib.font_manager"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# ============================== Small helpers ==============================
def _pair_dates_from_tag(pair_tag: str) -> Tuple[str, str]:
    a, b = pair_tag.split("_")
    return f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"

def _ensure_upper(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("dem", "corr", "method"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()
    return df

def _ensure_density_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "density" in df.columns:
        df["density"] = pd.to_numeric(df["density"], errors="coerce"); return df
    if "density_km2_per_gauge" in df.columns:
        df["density"] = pd.to_numeric(df["density_km2_per_gauge"], errors="coerce"); return df

    def _first(cands):
        for c in cands:
            if c in df.columns: return c
        return None

    area_col = _first(["area_km2", "area_sqkm", "area_sq_km"])
    ncal_col = _first(["n_cal", "n_calibration", "n_cal_gauges", "n_calib", "n_cal_pts"])
    if area_col and ncal_col:
        a = pd.to_numeric(df[area_col], errors="coerce")
        n = pd.to_numeric(df[ncal_col], errors="coerce").clip(lower=1)
        df["density"] = a / n; return df
    if ncal_col:
        n = pd.to_numeric(df[ncal_col], errors="coerce").clip(lower=1)
        df["density"] = 1.0 / n; return df
    df["density"] = 1.0; return df

def _read_area_density_metrics(area_dir: Path) -> Optional[pd.DataFrame]:
    f = area_dir / "results" / "accuracy_metrics_density_SRTM_RAW.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    df = _ensure_upper(df)
    df = _ensure_density_column(df)
    for c in ("pair_ref", "pair_sec"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
    df["pair_tag"] = (
        df["pair_ref"].str.replace("-", "", regex=False) + "_" +
        df["pair_sec"].str.replace("-", "", regex=False)
    )
    return df

def _read_tif_array(path: Optional[Path]) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    if not path or not path.exists(): return None
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(float)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            a = np.where(a == ds.nodata, np.nan, a)
        extent = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top)
        return a, extent

def _find_corr_map(area_dir: Path, dem: str, corr: str, pair_tag: str) -> Optional[Path]:
    res = area_dir / "results"
    for pat in (f"cal_60pct_{dem}_{corr}_{pair_tag}.tif",
                f"cal_ti_60pct_{dem}_{corr}_{pair_tag}.tif",
                f"dens_cal_60pct_{dem}_{corr}_{pair_tag}.tif"):
        p = res / pat
        if p.exists(): return p
    hits = list(res.glob(f"*{dem}_{corr}_{pair_tag}.tif"))
    return hits[0] if hits else None

def _find_idw_map(area_dir: Path, dem: str, corr: str, pair_tag: str) -> Optional[Path]:
    res = area_dir / "results"
    for pat in (f"dens_idw60_{dem}_{corr}_{pair_tag}.tif",
                f"idw60_{pair_tag}.tif"):
        p = res / pat
        if p.exists(): return p
    return None

# ====================== Map helpers: scalebar & north arrow ===================
def _geod() -> Optional[Geod]:
    if Geod is None: return None
    return Geod(ellps="WGS84")

def _map_width_km(extent: Tuple[float, float, float, float]) -> float:
    geod = _geod()
    xmin, xmax, ymin, ymax = extent; lat = (ymin + ymax)/2.0
    if geod:
        _, _, d = geod.inv(xmin, lat, xmax, lat)
        return max(d/1000.0, 1e-6)
    return (xmax-xmin)*111.32*np.cos(np.deg2rad(lat))

def _nice_scale_bar(width_km: float) -> float:
    target = max(width_km*0.22, 0.001)
    steps = [1,2,5]
    pow10 = 10**int(np.floor(np.log10(target))) if target>0 else 1
    for s in steps:
        if s*pow10 >= target: return s*pow10
    return 10*pow10

def _deg_lon_for_km(lat_deg: float, km: float) -> float:
    geod = _geod()
    if geod is None:
        return km/(111.32*np.cos(np.deg2rad(lat_deg)) + 1e-9)
    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid = 0.5*(lo+hi)
        _, _, d = geod.inv(0.0, lat_deg, mid, lat_deg)
        if d/1000.0 < km: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def _draw_scalebar(ax, extent: Tuple[float, float, float, float], *, pad_frac=0.04):
    xmin, xmax, ymin, ymax = extent
    lat_mid = (ymin+ymax)/2.0
    width_km = _map_width_km(extent)
    L_km = _nice_scale_bar(width_km)
    L_deg = _deg_lon_for_km(lat_mid, L_km)
    dx = xmax-xmin; dy = ymax-ymin
    x0 = xmin + pad_frac*dx; y0 = ymin + pad_frac*dy
    x_mid = x0 + L_deg/2.0
    ax.plot([x0, x0+L_deg], [y0, y0], color="k", lw=2.0)
    ax.add_patch(Rectangle((x0, y0-0.006*dy), L_deg/2.0, 0.012*dy, facecolor="k", edgecolor="k", lw=0))
    ax.add_patch(Rectangle((x_mid, y0-0.006*dy), L_deg/2.0, 0.012*dy, facecolor="w", edgecolor="k", lw=0.6))
    ax.text((x0+x0+L_deg)/2, y0+0.015*dy, f"{int(L_km)} km", ha="center", va="bottom", fontsize=8, color="k")

def _draw_north_arrow(ax, extent: Tuple[float, float, float, float], *, size_frac=0.08, pad_frac=0.05):
    xmin, xmax, ymin, ymax = extent
    dx = xmax-xmin; dy = ymax-ymin
    x = xmax - pad_frac*dx; y = ymin + pad_frac*dy
    size = size_frac*dy
    ax.add_patch(FancyArrow(x, y, 0, size, width=size*0.12, head_width=size*0.28, head_length=size*0.28,
                            length_includes_head=True, color="k"))
    ax.text(x, y+size+0.01*dy, "N", ha="center", va="bottom", fontsize=9, color="k")

# =========== Satellite basemap, water & gauges overlays, extents =============
def _sat_source_label(provider_name: str, xyz_url: str) -> str:
    if xyz_url: return "Custom XYZ"
    return {"Esri.WorldImagery": "Esri World Imagery"}.get(provider_name, provider_name)

def _add_basemap(ax, extent: Tuple[float, float, float, float], *, provider_name: str, xyz_url: str):
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ok = False
    if cx is not None:
        try:
            if xyz_url:
                cx.add_basemap(ax, source=xyz_url, crs="EPSG:4326", attribution=False)
            else:
                prov = getattr(cx.providers, provider_name, cx.providers.Esri.WorldImagery)
                cx.add_basemap(ax, source=prov, crs="EPSG:4326", attribution=False)
            ok = True
        except Exception:
            ok = False
    if not ok:
        ax.set_facecolor("#dddddd")
        ax.text(0.5, 0.5, "Satellite basemap unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#444")

def _overlay_water(ax, extent: Tuple[float, float, float, float],
                   water_path: Optional[str], *, highlight_extent: Optional[Tuple[float,float,float,float]]):
    if not water_path or gpd is None or not Path(water_path).exists(): return
    try:
        w = gpd.read_file(water_path)
        w.boundary.plot(ax=ax, color="#00aaff", linewidth=0.6, alpha=0.4, zorder=3)
        if shapely_box is not None and highlight_extent is not None:
            aoi = shapely_box(highlight_extent[0], highlight_extent[2], highlight_extent[1], highlight_extent[3])
            hi = w[w.geometry.intersects(aoi)]
            if not hi.empty:
                hi.boundary.plot(ax=ax, color="#ffff00", linewidth=1.6, alpha=0.9, zorder=4)
                hi.plot(ax=ax, color="#ffff00", alpha=0.10, edgecolor="none", zorder=3.5)
    except Exception:
        pass

def _find_gauges_geojson(area_dir: Path, pair_tag: str, gauges_template: Optional[str]) -> Optional[Path]:
    # 1) explicit template
    if gauges_template:
        try:
            p = Path(gauges_template.format(area=area_dir.name, pair=pair_tag))
            if p.exists(): return p
        except Exception:
            pass
    # 2) common candidates inside area/results or area/water_gauges
    for base in (area_dir / "results", area_dir / "water_gauges"):
        for name in GAUGE_NAME_CANDIDATES:
            if "*" in name:
                hits = sorted((base).glob(name.format(pair=pair_tag)))
                if hits: return hits[0]
            else:
                p = base / name.format(pair=pair_tag)
                if p.exists(): return p
    return None

def _split_roles_from_gdf(gdf: "gpd.GeoDataFrame"):
    role_col = None
    for c in ("role","set","split","type","group","subset"):
        if c in gdf.columns: role_col = c; break
    if role_col is None: return gdf, gdf.iloc[0:0]
    roles = gdf[role_col].astype(str).str.lower()
    is_val = roles.str.contains("val")
    is_cal = roles.str.contains("cal") | roles.str.contains("train")
    return gdf[is_cal], gdf[is_val]

def _overlay_gauges(ax, gdf: "gpd.GeoDataFrame"):
    if gpd is None or gdf is None or gdf.empty: return
    try:
        cal, val = _split_roles_from_gdf(gdf)
        if not cal.empty:
            cal.plot(ax=ax, markersize=12, color="black", marker="o", edgecolor="white", linewidth=0.4, zorder=6)
        if not val.empty:
            val.plot(ax=ax, markersize=14, color="red", marker="o", edgecolor="white", linewidth=0.4, zorder=6)
    except Exception:
        pass

# ============================== Curves / stats ==============================
def _agg_curve_metric(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=["n_cal","med","mean","p_low","p_high","med_density"])
    df = _ensure_density_column(df.copy())
    g_stats = (df.groupby("n_cal", as_index=False)
                 .agg(med=(value_col,"median"),
                      mean=(value_col,"mean"),
                      p_low=(value_col, lambda x: np.nanpercentile(x, P_LOW)),
                      p_high=(value_col, lambda x: np.nanpercentile(x, P_HIGH))))
    if "density" in df.columns:
        g_den = df.groupby("n_cal", as_index=False).agg(med_density=("density","median"))
    elif "area_km2" in df.columns:
        g_den = df.groupby("n_cal", as_index=False).agg(med_density=("area_km2","median"))
        g_den["med_density"] = g_den["med_density"] / g_den["n_cal"].astype(float).clip(lower=1)
    else:
        g_den = g_stats[["n_cal"]].copy()
        g_den["med_density"] = 1.0 / g_den["n_cal"].astype(float).clip(lower=1)
    g = g_stats.merge(g_den, on="n_cal", how="left")
    g = g[np.isfinite(g["med_density"]) & (g["med_density"] > 0)]
    g.sort_values("med_density", inplace=True)
    return g

def _set_log_xlim_safely(ax, x_arrays: List[np.ndarray], *, pad=0.10) -> Tuple[float,float]:
    xs = []
    for x in x_arrays:
        if x is None: continue
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size: xs.append(arr)
    if xs:
        xs = np.concatenate(xs); xmin, xmax = float(xs.min()), float(xs.max())
        if xmax <= xmin * (1 + 1e-9):
            xmin *= (1 - pad); xmax *= (1 + pad)
            if xmin <= 0: xmin = max(xmax / 10.0, 1e-6)
    else:
        xmin, xmax = 0.9, 1.1
    ax.set_xscale("log"); ax.set_xlim(xmin, xmax)
    return xmin, xmax

def _legend_note(ax):
    ax.text(0.995, 0.005, "Line: median   Band: 5–95%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#333333")

def _init_clip_notes(ax): ax._clip_note_top = ax._clip_note_bot = 0

def _shade_band(ax, x, ylo, yhi, color, *, hatched=False, alpha=0.12, z=1, y_min=None, y_max=None, annotate=False, label_for_note=""):
    """
    Fill the uncertainty band. If y_min/y_max are None, no clipping occurs and no annotations are added.
    """
    if x is None: return
    x = np.asarray(x); ylo = np.asarray(ylo, float); yhi = np.asarray(yhi, float)
    if x.size == 0 or ylo.size == 0 or yhi.size == 0: return

    if y_min is None or y_max is None:
        ylo_c, yhi_c = ylo, yhi
    else:
        ylo_c = np.clip(ylo, y_min, y_max)
        yhi_c = np.clip(yhi, y_min, y_max)

    face = to_rgba(color, alpha)
    if hatched:
        ax.fill_between(x, ylo_c, yhi_c, facecolor=face, edgecolor=to_rgba(color,0.8),
                        hatch='//', linewidth=0.6, zorder=z)
    else:
        ax.fill_between(x, ylo_c, yhi_c, facecolor=face, edgecolor='none', zorder=z)

# ============================ DENSITY (RAW vs IDW) =========================
def _water_total_bounds(water_path: str) -> Optional[Tuple[float, float, float, float]]:
    if gpd is None or not water_path or not Path(water_path).exists():
        return None
    try:
        w = gpd.read_file(water_path)
        minx, miny, maxx, maxy = w.total_bounds
        dx = max(maxx - minx, 1e-9)
        dy = max(maxy - miny, 1e-9)
        padx, pady = 0.02 * dx, 0.02 * dy  # small visual padding
        return (minx - padx, maxx + padx, miny - pady, maxy + pady)
    except Exception:
        return None

def plot_density_pair_raw_vs_idw(
    area_dir: Path,
    df7_area: pd.DataFrame,
    pair_tag: str,
    *,
    water_path: str,
    sat_provider: str,
    sat_url: str,
    gauges_template: str = "",
    maps_gap_frac: float = 0.018,     # distance between bias plot and map row (fig fraction)
    maps_height_frac: float = 0.32    # height of the map row (fig fraction)
):
    """Per-pair: RMSE & Bias vs density (median ± 5–95%) — bottom row: RAW LS, Satellite, RAW IDW — with gauges."""

    area = area_dir.name
    corr_label = "RAW"  # this figure is for RAW correction
    ref_iso, sec_iso = _pair_dates_from_tag(pair_tag)
    sub = df7_area[(df7_area["pair_ref"] == ref_iso) & (df7_area["pair_sec"] == sec_iso)].copy()
    if sub.empty:
        print(f"⏭️  No 7_* rows for {area}:{pair_tag}; skip density pair.")
        return

    # Filter for RAW SRTM, LS & IDW
    ls  = sub[(sub["method"] == METHOD_LS)  & (sub["dem"] == "SRTM") & (sub["corr"] == "RAW")]
    idw = sub[(sub["method"] == METHOD_IDW) & (sub["dem"] == "SRTM") & (sub["corr"] == "RAW")]
    if ls.empty and idw.empty:
        print(f"⏭️  No RAW LS/IDW rows for {area}:{pair_tag}.")
        return

    rmse_ls  = _agg_curve_metric(ls,  "rmse_cm")  if not ls.empty  else pd.DataFrame()
    rmse_idw = _agg_curve_metric(idw, "rmse_cm")  if not idw.empty else pd.DataFrame()
    bias_ls  = _agg_curve_metric(ls,  "bias_cm")  if not ls.empty  else pd.DataFrame()
    bias_idw = _agg_curve_metric(idw, "bias_cm")  if not idw.empty else pd.DataFrame()

    fig = plt.figure(figsize=(14.0, 11.0), dpi=150, constrained_layout=False)
    top_margin, bottom_margin = 0.93, 0.06
    fig.subplots_adjust(top=top_margin, bottom=bottom_margin)

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows=4, ncols=3, height_ratios=[2.6, 2.6, 0.10, 3.6], hspace=0.34, wspace=0.08, figure=fig)

    # ----- Title (requested format) -----
    fig.suptitle(f"Error vs Density: {area} -- {ref_iso} to {sec_iso} -- {corr_label}",
                 y=0.965, fontsize=14, fontweight="bold")

    # ---------- RMSE curve ----------
    ax1 = fig.add_subplot(gs[0, :])
    med_max  = max([g["med"].max() for g in (rmse_ls, rmse_idw) if not g.empty] or [1.0])
    band_max = max([np.nanmax(g["p_high"]) if ("p_high" in g) and not g.empty else 0 for g in (rmse_ls, rmse_idw)] + [0])
    y_min, y_max = 0.0, float(max(med_max * 1.15, band_max * 1.05, 1.0))
    for g, lab, col, hat in (
        (rmse_ls,  "Interferogram (RAW) median", COLORS_DEM["SRTM"], False),
        (rmse_idw, "IDW (same gauges) median",   COLOR_IDW,         True),
    ):
        if g.empty: continue
        ax1.plot(g["med_density"], g["med"], "--" if hat else "-", color=col, lw=1.9, label=lab)
        _shade_band(ax1, g["med_density"], g["p_low"], g["p_high"], col,
                    hatched=hat, alpha=0.12 if not hat else 0.10, z=1,
                    y_min=y_min, y_max=y_max, annotate=True, label_for_note=lab)
    _set_log_xlim_safely(ax1, [g["med_density"].values for g in (rmse_ls, rmse_idw) if not g.empty])
    ax1.set_ylim(y_min, y_max)
    ax1.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax1.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1))
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.set_ylabel("RMSE (cm)")
    ax1.set_xlabel("Gauge density (km² per gauge) (log scale)")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(ncols=2, fontsize=9, loc="best")
    _legend_note(ax1)

    # ---------- Bias curve (NO clipping; show all data) ----------
    ax2 = fig.add_subplot(gs[1, :])
    for g, lab, col, hat in (
        (bias_ls,  "Interferogram (RAW) median", COLORS_DEM["SRTM"], False),
        (bias_idw, "IDW (same gauges) median",   COLOR_IDW,         True),
    ):
        if g.empty: continue
        ax2.plot(g["med_density"], g["med"], "--" if hat else "-", color=col, lw=1.9, label=lab)
        # No clipping at all; full band drawn
        _shade_band(ax2, g["med_density"], g["p_low"], g["p_high"], col,
                    hatched=hat, alpha=0.12 if not hat else 0.10, z=1,
                    y_min=None, y_max=None, annotate=False)
    # Autoscale y (keeps all outliers visible)
    _set_log_xlim_safely(ax2, [x["med_density"].values for x in (bias_ls, bias_idw) if not x.empty])
    ax2.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax2.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_ylabel("Bias (cm)")
    ax2.set_xlabel("Gauge density (km² per gauge) (log scale)")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(ncols=2, fontsize=9, loc="best")
    _legend_note(ax2)

    # ---------- Bottom row maps (anchored to bias axis) ----------
    ax_ls  = fig.add_subplot(gs[3, 0])
    ax_sat = fig.add_subplot(gs[3, 1])
    ax_idw = fig.add_subplot(gs[3, 2])

    p_ls, p_sat, p_idw = ax_ls.get_position(), ax_sat.get_position(), ax_idw.get_position()
    p_bias = ax2.get_position()
    y_top_maps = max(p_bias.y0 - maps_gap_frac, bottom_margin + 0.01)
    maps_height = min(maps_height_frac, y_top_maps - bottom_margin)
    y0 = y_top_maps - maps_height

    ax_ls.set_position([p_ls.x0,  y0, p_ls.width,  maps_height])
    ax_sat.set_position([p_sat.x0, y0, p_sat.width, maps_height])
    ax_idw.set_position([p_idw.x0, y0, p_idw.width, maps_height])

    # ---------- Read and draw maps ----------
    m_ls  = _find_corr_map(area_dir, "SRTM", "RAW", pair_tag)
    m_idw = _find_idw_map(area_dir, "SRTM", "RAW", pair_tag)
    a1 = _read_tif_array(m_ls)
    a2 = _read_tif_array(m_idw)

    arrays = [a for a in (a1, a2) if a is not None]
    vmin, vmax = (0.0, 1.0)
    if arrays:
        vv = np.concatenate([arr[0][np.isfinite(arr[0])] for arr in arrays])
        if vv.size:
            vmin, vmax = np.nanpercentile(vv, [2, 98])

    # Load gauges once (if available), reuse for all three panels
    gdf_gauges = None
    try:
        gj_path = _find_gauges_geojson(area_dir, pair_tag, gauges_template)
        if gj_path and gpd is not None:
            gdf_gauges = gpd.read_file(gj_path)
            if getattr(gdf_gauges, "crs", None) and gdf_gauges.crs and gdf_gauges.crs.to_string().upper() not in ("EPSG:4326", "WGS84"):
                gdf_gauges = gdf_gauges.to_crs(epsg=4326)
    except Exception:
        gdf_gauges = None

    # LS map
    if a1:
        im = ax_ls.imshow(a1[0], extent=a1[1], origin="upper", cmap=CMAP_INV, vmin=vmin, vmax=vmax)
        ax_ls.set_title("RAW calibrated", fontsize=10, pad=4)
        _draw_scalebar(ax_ls, a1[1]); _draw_north_arrow(ax_ls, a1[1])
        ax_ls.set_xticks([]); ax_ls.set_yticks([])
        if gdf_gauges is not None: _overlay_gauges(ax_ls, gdf_gauges)
    else:
        ax_ls.text(0.5, 0.5, "Missing RAW calibrated map", ha="center", va="center"); ax_ls.set_axis_off()

    # Satellite center (union extent)
    union = None
    if a1 and a2:
        union = (min(a1[1][0], a2[1][0]), max(a1[1][1], a2[1][1]),
                 min(a1[1][2], a2[1][2]), max(a1[1][3], a2[1][3]))
    elif a1:
        union = a1[1]
    elif a2:
        union = a2[1]

    if union is not None:
        ax_sat.set_xticks([]); ax_sat.set_yticks([]); ax_sat.set_frame_on(False)
        _add_basemap(ax_sat, union, provider_name=sat_provider, xyz_url=sat_url)
        _overlay_water(ax_sat, union, water_path, highlight_extent=union)
        if gdf_gauges is not None: _overlay_gauges(ax_sat, gdf_gauges)
        ax_sat.set_title("Satellite (AOI)", fontsize=10, pad=4)
    else:
        ax_sat.text(0.5, 0.5, "No AOI extent available", ha="center", va="center"); ax_sat.set_axis_off()

    # IDW map
    if a2:
        im2 = ax_idw.imshow(a2[0], extent=a2[1], origin="upper", cmap=CMAP_INV, vmin=vmin, vmax=vmax)
        ax_idw.set_title("IDW baseline", fontsize=10, pad=4)
        _draw_scalebar(ax_idw, a2[1]); _draw_north_arrow(ax_idw, a2[1])
        ax_idw.set_xticks([]); ax_idw.set_yticks([])
        if gdf_gauges is not None: _overlay_gauges(ax_idw, gdf_gauges)
    else:
        ax_idw.text(0.5, 0.5, "Missing RAW IDW map", ha="center", va="center"); ax_idw.set_axis_off()

    # External colorbars
    cbar_w = 0.015; gap = 0.006
    if a1:
        bb_ls = ax_ls.get_position()
        cax_ls = fig.add_axes([bb_ls.x0 - gap - cbar_w, bb_ls.y0, cbar_w, bb_ls.height])
        cb1 = plt.colorbar(im, cax=cax_ls, orientation="vertical", location="left")
        cb1.set_label("cm", fontsize=9); cb1.ax.tick_params(labelsize=8)
    if a2:
        bb_idw = ax_idw.get_position()
        cax_idw = fig.add_axes([bb_idw.x1 + gap, bb_idw.y0, cbar_w, bb_idw.height])
        cb2 = plt.colorbar(im2, cax=cax_idw, orientation="vertical", location="right")
        cb2.set_label("cm", fontsize=9); cb2.ax.tick_params(labelsize=8)

    out = area_dir / "results" / f"density_raw_idw_pair_{pair_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Density RAW per-pair written: {out}")

# --------- All-areas combined density vs RMSE (scatter + uncertainty) ----------
def plot_all_areas_density(area7: Dict[str, pd.DataFrame], *, out_dir: Path):
    if not area7:
        print("⏭️  No areas to combine for all-areas density plot.")
        return

    frames = []
    for area, df in area7.items():
        d = df.copy()
        d["__area__"] = area
        frames.append(d)
    big = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if big.empty:
        print("⏭️  All-areas: empty dataset.")
        return

    big = _ensure_upper(_ensure_density_column(big))
    sel = big[(big["corr"] == "RAW") & (big["dem"] == "SRTM") & (big["method"].isin([METHOD_LS, METHOD_IDW]))].copy()
    if sel.empty:
        print("⏭️  All-areas: no RAW/SRTM rows with LS/IDW.")
        return

    ls  = sel[sel["method"] == METHOD_LS]
    idw = sel[sel["method"] == METHOD_IDW]
    g_ls  = _agg_curve_metric(ls,  "rmse_cm")
    g_idw = _agg_curve_metric(idw, "rmse_cm")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.scatter(sel["density"], sel["rmse_cm"], s=14, alpha=0.20, edgecolors="none", label="All area–pair samples")

    ymax = float(np.nanmax(sel["rmse_cm"])) if np.isfinite(sel["rmse_cm"]).any() else 1.0
    ymax = max(1.0, ymax * 1.10)

    if not g_ls.empty:
        ax.plot(g_ls["med_density"], g_ls["med"], "-", color=COLORS_DEM["SRTM"], lw=2.0, label="LS median")
        _shade_band(ax, g_ls["med_density"], g_ls["p_low"], g_ls["p_high"], COLORS_DEM["SRTM"],
                    hatched=False, alpha=0.12, z=1, y_min=0.0, y_max=ymax)
    if not g_idw.empty:
        ax.plot(g_idw["med_density"], g_idw["med"], "--", color=COLOR_IDW, lw=2.0, label="IDW median")
        _shade_band(ax, g_idw["med_density"], g_idw["p_low"], g_idw["p_high"], COLOR_IDW,
                    hatched=True, alpha=0.10, z=1, y_min=0.0, y_max=ymax)

    _set_log_xlim_safely(ax, [sel["density"].values])
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("Gauge density (km² per gauge) (log scale)")
    ax.set_ylabel("RMSE (cm)")
    ax.grid(True, alpha=0.30, which="both")
    ax.legend()
    fig.suptitle("All areas — RAW: density vs RMSE (all samples + 5–95% uncertainty bands)", y=0.96, fontsize=14, fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "density_all_areas_raw.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🌍 All-areas density figure written: {out}")

# ----------------------------------- CLI -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Density visualization only: per-pair curves + maps, and optional all-areas overview.")
    ap.add_argument("--areas-root", type=str, default=str(AREAS_ROOT_DEFAULT),
                    help="Root folder containing per-area subfolders")
    ap.add_argument("--area", type=str,
                    help="Only process this AREA (subfolder name under --areas-root)")
    ap.add_argument("--water-areas", type=str, default=DEF_WATER_AREAS,
                    help="Path to water_areas.geojson for overview overlay")
    ap.add_argument("--sat-provider", type=str, default=DEF_SAT_PROVIDER,
                    help="contextily provider string (ignored if --sat-url is set)")
    ap.add_argument("--sat-url", type=str, default=DEF_SAT_URL,
                    help="Custom XYZ for satellite (e.g. Google XYZ).")
    ap.add_argument("--gauges-template", type=str, default="",
                    help="Optional template to a per-pair gauges GeoJSON, e.g. '/path/{area}/results/gauges_split_60pct_{pair}.geojson'")
    ap.add_argument("--maps-gap-frac", type=float, default=0.1,
                    help="Vertical gap between bias plot and bottom maps (figure fraction).")
    ap.add_argument("--maps-height-frac", type=float, default=0.32,
                    help="Height of bottom maps (figure fraction).")
    ap.add_argument("--combine-areas", action="store_true",
                    help="Also write an all-areas density vs RMSE overview figure.")

    args = ap.parse_args()

    root = Path(args.areas_root)
    targets = [root / args.area] if args.area else sorted([d for d in root.iterdir() if d.is_dir()])

    # Load RAW density metrics (7_*) per area
    area7: Dict[str, pd.DataFrame] = {}
    for area_dir in targets:
        df7 = _read_area_density_metrics(area_dir)
        if df7 is None or df7.empty:
            print(f"⏭️  No 7_* (RAW density) metrics for area: {area_dir.name}")
        else:
            area7[area_dir.name] = df7

    if not area7:
        print("⏭️  Nothing to visualize."); return

    # Per-pair figures
    for area_name, df7 in area7.items():
        area_dir = root / area_name
        pair_tags7 = sorted(set(df7["pair_tag"].dropna().astype(str).tolist()))
        for pair_tag in pair_tags7:
            try:
                plot_density_pair_raw_vs_idw(
                    area_dir, df7, pair_tag,
                    water_path=args.water_areas,
                    sat_provider=args.sat_provider,
                    sat_url=args.sat_url,
                    gauges_template=args.gauges_template,
                    maps_gap_frac=args.maps_gap_frac,
                    maps_height_frac=args.maps_height_frac
                )
            except Exception as e:
                print(f"⚠️  Density pair failed {area_name}:{pair_tag}: {e}")

    # Combined (all areas) overview
    if args.combine_areas:
        try:
            plot_all_areas_density(area7, out_dir=root / "results")
        except Exception as e:
            print(f"⚠️  All-areas density figure failed: {e}")

if __name__ == "__main__":
    main()
