#!/usr/bin/env python3
"""
8_visualization.py (density-only, always writes all-areas figures)

Inputs (per area):
- results/accuracy_metrics_density_<DEM>_<CORR>.csv   (default: SRTM_RAW)

Outputs:
- <area>/results/density_<corr_lower>_idw_pair_<PAIR>.png
- <areas-root>/results/density_all_areas_by_area_<DEM>_<CORR>_<METHOD>.png
- <areas-root>/results/density_all_areas_combined_<DEM>_<CORR>_<METHOD>.png

What it does:
- Per-pair Error vs Density (RMSE + Bias) with median ± 5–95% bands (no clipping on Bias),
  bottom-row maps (LS, Satellite AOI, IDW) with gauge points (cal=black, val=red).
- All-areas figures (NO DOTS):
  1) ONE figure with TWO panels (RMSE & Bias): each area as its own median line + 5–95% band.
  2) ONE figure with TWO panels (RMSE & Bias): all areas combined into a single median line + 5–95% band.
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

def _read_area_density_metrics(area_dir: Path, dem: str, corr: str) -> Optional[pd.DataFrame]:
    """Read results/accuracy_metrics_density_<DEM>_<CORR>.csv (robust to case); fallback to the first matching file."""
    resdir = area_dir / "results"
    target = resdir / f"accuracy_metrics_density_{dem.upper()}_{corr.upper()}.csv"
    path = None
    if target.exists():
        path = target
    else:
        hits = sorted(resdir.glob("accuracy_metrics_density_*.csv"))
        for h in hits:
            m = re.match(r"accuracy_metrics_density_([A-Z0-9]+)_([A-Z0-9_]+)\.csv", h.name, flags=re.I)
            if m and m.group(1).upper()==dem.upper() and m.group(2).upper()==corr.upper():
                path = h; break
        if path is None and hits:
            path = hits[0]
    if path is None: return None

    df = pd.read_csv(path)
    df = _ensure_upper(df)
    df = _ensure_density_column(df)
    for c in ("pair_ref", "pair_sec"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
    if "pair_tag" not in df.columns and {"pair_ref","pair_sec"}.issubset(df.columns):
        df["pair_tag"] = (df["pair_ref"].str.replace("-", "", regex=False) + "_" +
                          df["pair_sec"].str.replace("-", "", regex=False))
    return df

def _agg_curve_by_density(df: pd.DataFrame, value_col: str, nbins: int = 30, min_count: int = 1) -> pd.DataFrame:
    """
    Aggregate a curve directly as a function of DENSITY (log-binned):
      - bins: logspace between global min/max positive density in df
      - per bin: median, mean, 5–95% of `value_col`, and median density (x)
    Returns columns: ['bin','n','med','mean','p_low','p_high','med_density']
    """
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=["bin","n","med","mean","p_low","p_high","med_density"])

    d = _ensure_density_column(df.copy())
    vals = pd.to_numeric(d[value_col], errors="coerce").to_numpy()
    den  = pd.to_numeric(d["density"], errors="coerce").to_numpy()

    m = np.isfinite(vals) & np.isfinite(den) & (den > 0)
    vals, den = vals[m], den[m]
    if vals.size == 0:
        return pd.DataFrame(columns=["bin","n","med","mean","p_low","p_high","med_density"])

    dmin, dmax = float(np.min(den)), float(np.max(den))
    if dmax <= dmin:
        return pd.DataFrame(columns=["bin","n","med","mean","p_low","p_high","med_density"])

    # Log-spaced bins over the *full* density range in the data
    edges = np.logspace(np.log10(dmin), np.log10(dmax), nbins + 1)
    bin_idx = np.digitize(den, edges) - 1  # 0..nbins-1

    rows = []
    for b in range(nbins):
        mask = (bin_idx == b)
        if not mask.any():
            continue
        v = vals[mask]; x = den[mask]
        n = v.size
        if n < min_count:
            continue
        rows.append({
            "bin": b, "n": n,
            "med": float(np.nanmedian(v)),
            "mean": float(np.nanmean(v)),
            "p_low": float(np.nanpercentile(v, P_LOW)),
            "p_high": float(np.nanpercentile(v, P_HIGH)),
            # x-coordinate at this bin: median of sample densities in the bin
            "med_density": float(np.nanmedian(x))
        })

    g = pd.DataFrame.from_records(rows)
    if g.empty:
        return g
    g.sort_values("med_density", inplace=True)
    return g


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

# =========== Basemap, water & gauges overlays ===========
def _add_basemap(ax, extent, *, provider_name: str, xyz_url: str):
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

def _overlay_water(ax, extent, water_path: Optional[str], *, highlight_extent: Optional[Tuple[float,float,float,float]]):
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
    if gauges_template:
        try:
            p = Path(gauges_template.format(area=area_dir.name, pair=pair_tag))
            if p.exists(): return p
        except Exception:
            pass
    for base in (area_dir / "results", area_dir / "water_gauges"):
        for name in GAUGE_NAME_CANDIDATES:
            if "*" in name:
                hits = sorted(base.glob(name.format(pair=pair_tag)))
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

# ============================ DENSITY (per-pair) =========================
def plot_density_pair(
    area_dir: Path,
    df7_area: pd.DataFrame,
    pair_tag: str,
    *,
    dem: str,
    corr: str,
    water_path: str,
    sat_provider: str,
    sat_url: str,
    gauges_template: str = "",
    maps_gap_frac: float = 0.1,
    maps_height_frac: float = 0.32
):
    """Per-pair: RMSE & Bias vs density (median ± 5–95%) — bottom row: LS, Satellite, IDW — with gauges (no bias clipping)."""

    area = area_dir.name
    ref_iso, sec_iso = _pair_dates_from_tag(pair_tag)
    sub = df7_area[(df7_area["pair_ref"] == ref_iso) & (df7_area["pair_sec"] == sec_iso)].copy()
    if sub.empty:
        print(f"⏭️  No density rows for {area}:{pair_tag}; skip.")
        return

    ls  = sub[(sub["method"] == METHOD_LS)  & (sub["dem"] == dem) & (sub["corr"] == corr)]
    idw = sub[(sub["method"] == METHOD_IDW) & (sub["dem"] == dem) & (sub["corr"] == corr)]
    if ls.empty and idw.empty:
        print(f"⏭️  No {corr} LS/IDW rows for {area}:{pair_tag}.")
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

    # Title (requested format)
    fig.suptitle(f"Error vs Density: {area} -- {ref_iso} to {sec_iso} -- {corr}",
                 y=0.965, fontsize=14, fontweight="bold")

    # ---------- RMSE ----------
    ax1 = fig.add_subplot(gs[0, :])
    med_max  = max([g["med"].max() for g in (rmse_ls, rmse_idw) if not g.empty] or [1.0])
    band_max = max([np.nanmax(g["p_high"]) if ("p_high" in g) and not g.empty else 0 for g in (rmse_ls, rmse_idw)] + [0])
    y_min, y_max = 0.0, float(max(med_max * 1.15, band_max * 1.05, 1.0))
    for g, lab, col, hat in (
        (rmse_ls,  f"Interferogram ({corr}) median", COLORS_DEM.get(dem, CB["blue"]), False),
        (rmse_idw, "IDW (same gauges) median",       COLOR_IDW,                         True),
    ):
        if g.empty: continue
        ax1.plot(g["med_density"], g["med"], "--" if hat else "-", color=col, lw=1.9, label=lab)
        face = to_rgba(col, 0.12 if not hat else 0.10)
        ax1.fill_between(g["med_density"], g["p_low"], g["p_high"], facecolor=face, edgecolor='none', zorder=1)
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

    # ---------- Bias (NO clipping, show all data) ----------
    ax2 = fig.add_subplot(gs[1, :])
    for g, lab, col, hat in (
        (bias_ls,  f"Interferogram ({corr}) median", COLORS_DEM.get(dem, CB["blue"]), False),
        (bias_idw, "IDW (same gauges) median",       COLOR_IDW,                         True),
    ):
        if g.empty: continue
        ax2.plot(g["med_density"], g["med"], "--" if hat else "-", color=col, lw=1.9, label=lab)
        face = to_rgba(col, 0.12 if not hat else 0.10)
        ax2.fill_between(g["med_density"], g["p_low"], g["p_high"], facecolor=face, edgecolor='none', zorder=1)
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

    # ---------- Bottom row maps ----------
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

    # Read rasters
    m_ls  = _find_corr_map(area_dir, dem, corr, pair_tag)
    m_idw = _find_idw_map(area_dir, dem, corr, pair_tag)
    a1 = _read_tif_array(m_ls)
    a2 = _read_tif_array(m_idw)
    arrays = [a for a in (a1, a2) if a is not None]
    vmin, vmax = (0.0, 1.0)
    if arrays:
        vv = np.concatenate([arr[0][np.isfinite(arr[0])] for arr in arrays])
        if vv.size:
            vmin, vmax = np.nanpercentile(vv, [2, 98])

    # Gauges once
    gdf_gauges = None
    try:
        gj_path = _find_gauges_geojson(area_dir, pair_tag, None)
        if gj_path and gpd is not None:
            gdf_gauges = gpd.read_file(gj_path)
            if getattr(gdf_gauges, "crs", None) and gdf_gauges.crs and gdf_gauges.crs.to_string().upper() not in ("EPSG:4326", "WGS84"):
                gdf_gauges = gdf_gauges.to_crs(epsg=4326)
    except Exception:
        gdf_gauges = None

    # LS map
    if a1:
        im = ax_ls.imshow(a1[0], extent=a1[1], origin="upper", cmap=CMAP_INV, vmin=vmin, vmax=vmax)
        ax_ls.set_title(f"{corr} calibrated", fontsize=10, pad=4)
        _draw_scalebar(ax_ls, a1[1]); _draw_north_arrow(ax_ls, a1[1])
        ax_ls.set_xticks([]); ax_ls.set_yticks([])
        if gdf_gauges is not None: _overlay_gauges(ax_ls, gdf_gauges)
    else:
        ax_ls.text(0.5, 0.5, f"Missing {corr} calibrated map", ha="center", va="center"); ax_ls.set_axis_off()

    # Union extent for the satellite panel
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
        ax_idw.text(0.5, 0.5, f"Missing {corr} IDW map", ha="center", va="center"); ax_idw.set_axis_off()

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

    out = area_dir / "results" / f"density_{corr.lower()}_idw_pair_{pair_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Per-pair density figure written: {out}")

# --------- All-areas: each area as a separate line (NO DOTS), RMSE & Bias ----------
# --------- All-areas: each area as a separate line (NO DOTS), RMSE & Bias ----------
def plot_all_areas_density(area7: Dict[str, pd.DataFrame], *, dem: str, corr: str, method: str, out_dir: Path):
    """
    One figure with TWO stacked panels (RMSE on top, Bias below).
    For the selected method (LS or IDW), each AREA is a separate median line with a 5–95% band.
    No scatter/dots are drawn.
    Output: density_all_areas_by_area_<DEM>_<CORR>_<METHOD>.png
    """
    if not area7:
        print("⏭️  No areas to combine for all-areas plot."); return

    # Gather & standardize
    frames = []
    for area, df in area7.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["__area__"] = area
        frames.append(d)
    big = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if big.empty:
        print("⏭️  All-areas: empty dataset."); return

    dem_u, corr_u = dem.upper(), corr.upper()
    big = _ensure_upper(_ensure_density_column(big))
    sel = big[(big["dem"] == dem_u) & (big["corr"] == corr_u) & (big["method"] == method)].copy()
    if sel.empty:
        print(f"⏭️  All-areas: no rows for DEM={dem_u} CORR={corr_u} METHOD={method}."); return

    areas = sorted(sel["__area__"].dropna().unique().tolist())
    palette = mpl.rcParams['axes.prop_cycle'].by_key().get('color', [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ])
    col_for = {a: palette[i % len(palette)] for i, a in enumerate(areas)}

    # Helper: plain number formatter on log axis (no 10^k)
    def _fmt_plain(v, _):
        if not np.isfinite(v) or v <= 0: return ""
        # Show integers when near integer; otherwise compact decimals
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}"
        # keep up to 3 significant digits for decimals like 0.2, 0.3, 0.5, 2.5, 12.5
        s = f"{v:.3g}".rstrip("0").rstrip(".")
        return s

    fig, (ax_r, ax_b) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), dpi=150, sharex=True)

    # ---------- RMSE panel ----------
    for a in areas:
        sa = sel[sel["__area__"] == a]
        g = _agg_curve_metric(sa, "rmse_cm")
        if g.empty:
            continue
        c = col_for[a]
        ax_r.plot(g["med_density"], g["med"], "-", lw=1.8, color=c, label=a)
        ax_r.fill_between(g["med_density"], g["p_low"], g["p_high"],
                          facecolor=to_rgba(c, 0.12), edgecolor="none", zorder=1)

    # X scale + nicer ticks (labels visible on BOTH panels)
    _set_log_xlim_safely(ax_r, [sel["density"].values])
    ax_r.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax_r.xaxis.set_major_formatter(FuncFormatter(_fmt_plain))
    ax_r.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1))
    ax_r.xaxis.set_minor_formatter(NullFormatter())
    ax_r.tick_params(axis="x", which="both", labelbottom=True)  # make top panel show x tick labels too

    # Y limits + ticks
    ax_r.set_ylim(0, 25)
    from matplotlib.ticker import MultipleLocator
    ax_r.yaxis.set_major_locator(MultipleLocator(5))
    ax_r.set_ylabel("RMSE (cm)")
    ax_r.grid(True, alpha=0.30, which="both")
    _legend_note(ax_r)
    ax_r.set_title("RMSE")

    # ---------- BIAS panel ----------
    for a in areas:
        sa = sel[sel["__area__"] == a]
        g = _agg_curve_metric(sa, "bias_cm")
        if g.empty:
            continue
        c = col_for[a]
        ax_b.plot(g["med_density"], g["med"], "-", lw=1.8, color=c, label=a)
        ax_b.fill_between(g["med_density"], g["p_low"], g["p_high"],
                          facecolor=to_rgba(c, 0.12), edgecolor="none", zorder=1)

    # X scale + nicer ticks (mirror top)
    ax_b.set_xscale("log")
    ax_b.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax_b.xaxis.set_major_formatter(FuncFormatter(_fmt_plain))
    ax_b.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1))
    ax_b.xaxis.set_minor_formatter(NullFormatter())

    # Y limits + ticks
    ax_b.set_ylim(-15, 15)
    ax_b.yaxis.set_major_locator(MultipleLocator(5))
    ax_b.set_xlabel("Gauge density (km² per gauge) (log)")
    ax_b.set_ylabel("Bias (cm)")
    ax_b.grid(True, alpha=0.30, which="both")
    _legend_note(ax_b)
    ax_b.set_title("Bias")

    # Shared legend (areas) below
    handles, labels = ax_r.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(f"{corr_u} — {dem_u} — per-area median + 5–95% band",
                 y=0.98, fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.10, hspace=0.28)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"density_all_areas_by_area_{dem_u}_{corr_u}_{method}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🌍 Per-area all-areas figure written: {out}")



# --------- All-areas: single combined line (NO DOTS), RMSE & Bias ----------
def plot_all_areas_combined(area7: Dict[str, pd.DataFrame], *, dem: str, corr: str, method: str, out_dir: Path):
    """
    TWO stacked panels (RMSE on top, Bias below), each showing a SINGLE overall median line
    with a 5–95% band computed from all areas combined, aggregated *by density* (log-bins).
    No dots. X-axis spans the FULL density range in the data (no empty margins, no truncated tails).
    Output: density_all_areas_combined_<DEM>_<CORR>_<METHOD>.png
    """
    if not area7:
        print("⏭️  No areas to combine for combined all-areas plot."); return

    frames = []
    for _, df in area7.items():
        if df is None or df.empty:
            continue
        frames.append(df.copy())
    big = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if big.empty:
        print("⏭️  Combined all-areas: empty dataset."); return

    dem_u, corr_u = dem.upper(), corr.upper()
    big = _ensure_upper(_ensure_density_column(big))
    sel = big[(big["dem"] == dem_u) & (big["corr"] == corr_u) & (big["method"] == method)].copy()
    if sel.empty:
        print(f"⏭️  Combined all-areas: no rows for DEM={dem_u} CORR={corr_u} METHOD={method}."); return

    # Compute curves directly vs density (covers full density range)
    g_rmse = _agg_curve_by_density(sel, "rmse_cm", nbins=30, min_count=1)
    g_bias = _agg_curve_by_density(sel, "bias_cm", nbins=30, min_count=1)

    # Global density range from ALL samples (ensures 65–3000 shows if present)
    dens_all = pd.to_numeric(sel["density"], errors="coerce")
    dens_all = dens_all[np.isfinite(dens_all) & (dens_all > 0)]
    if dens_all.empty:
        print("⏭️  Combined all-areas: no positive density values."); return
    xmin, xmax = float(dens_all.min()), float(dens_all.max())

    # Plain-number formatter for log axis (no 10^k)
    def _fmt_plain(v, _):
        if not np.isfinite(v) or v <= 0: return ""
        if abs(v - round(v)) < 1e-9:  # near-integer → show integer
            return f"{int(round(v))}"
        s = f"{v:.3g}".rstrip("0").rstrip(".")
        return s

    fig, (ax_r, ax_b) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), dpi=150, sharex=False)

    # ---- RMSE panel ----
    if not g_rmse.empty:
        line_color = COLORS_DEM.get(dem_u, "#0072B2")
        ax_r.plot(g_rmse["med_density"], g_rmse["med"], "-", lw=2.8, color=line_color, label="Median")
        ax_r.fill_between(g_rmse["med_density"], g_rmse["p_low"], g_rmse["p_high"],
                          facecolor=to_rgba(line_color, 0.18), edgecolor="none", zorder=1)
    else:
        ax_r.text(0.5, 0.5, "No RMSE curve", ha="center", va="center")

    ax_r.set_xscale("log"); ax_r.set_xlim(xmin, xmax)
    ax_r.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax_r.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_fmt_plain))
    ax_r.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(1,10)*0.1))
    ax_r.xaxis.set_minor_formatter(NullFormatter())
    ax_r.tick_params(axis="x", which="both", labelbottom=True)  # show labels on top panel too

    from matplotlib.ticker import MultipleLocator
    ax_r.set_ylim(0, 25)
    ax_r.yaxis.set_major_locator(MultipleLocator(5))
    ax_r.set_ylabel("RMSE (cm)")
    ax_r.grid(True, alpha=0.30, which="both")
    _legend_note(ax_r); ax_r.set_title("RMSE")

    # ---- Bias panel ----
    if not g_bias.empty:
        line_color_b = COLORS_DEM.get(dem_u, "#0072B2")
        ax_b.plot(g_bias["med_density"], g_bias["med"], "-", lw=2.8, color=line_color_b, label="Median")
        ax_b.fill_between(g_bias["med_density"], g_bias["p_low"], g_bias["p_high"],
                          facecolor=to_rgba(line_color_b, 0.18), edgecolor="none", zorder=1)
    else:
        ax_b.text(0.5, 0.5, "No Bias curve", ha="center", va="center")

    ax_b.set_xscale("log"); ax_b.set_xlim(xmin, xmax)
    ax_b.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax_b.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_fmt_plain))
    ax_b.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(1,10)*0.1))
    ax_b.xaxis.set_minor_formatter(NullFormatter())

    ax_b.set_ylim(-15, 15)
    ax_b.yaxis.set_major_locator(MultipleLocator(5))
    ax_b.set_xlabel("Gauge density (km² per gauge) (log)")
    ax_b.set_ylabel("Bias (cm)")
    ax_b.grid(True, alpha=0.30, which="both")
    _legend_note(ax_b); ax_b.set_title("Bias")

    fig.suptitle(f"All areas combined — {corr_u} — {dem_u} — {method}\nMedian + 5–95% band (no dots)",
                 y=0.98, fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.10, hspace=0.28)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"density_all_areas_combined_{dem_u}_{corr_u}_{method}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🧮 Combined all-areas figure written: {out}")


# ----------------------------------- CLI -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Density visualization: per-pair curves + maps (with gauges) and all-areas figures (no dots).")
    ap.add_argument("--areas-root", type=str, default=str(AREAS_ROOT_DEFAULT),
                    help="Root folder containing per-area subfolders")
    ap.add_argument("--area", type=str,
                    help="Only process this AREA (subfolder name under --areas-root)")
    ap.add_argument("--dem", type=str, default="SRTM", help="DEM to visualize (e.g., SRTM, 3DEP)")
    ap.add_argument("--corr", type=str, default="RAW", help="Correction to visualize (RAW, TROPO, IONO, TROPO_IONO)")
    ap.add_argument("--method", type=str, default="LEAST_SQUARES",
                    help="Method to plot for all-areas figures: LEAST_SQUARES (LS) or IDW_DHVIS (IDW).")
    ap.add_argument("--water-areas", type=str, default=DEF_WATER_AREAS,
                    help="Path to water_areas.geojson for overview overlay")
    ap.add_argument("--sat-provider", type=str, default=DEF_SAT_PROVIDER,
                    help="contextily provider string (ignored if --sat-url is set)")
    ap.add_argument("--sat-url", type=str, default=DEF_SAT_URL,
                    help="Custom XYZ for satellite (e.g. Google XYZ).")
    ap.add_argument("--gauges-template", type=str, default="",
                    help="Optional template to per-pair gauges GeoJSON, e.g. '/path/{area}/results/gauges_split_60pct_{pair}.geojson'")
    ap.add_argument("--maps-gap-frac", type=float, default=0.1,
                    help="Vertical gap between bias plot and bottom maps (figure fraction).")
    ap.add_argument("--maps-height-frac", type=float, default=0.32,
                    help="Height of bottom maps (figure fraction).")

    args = ap.parse_args()

    # normalize method
    mraw = (args.method or "").strip().upper()
    if mraw in ("LS", "LEAST_SQUARES"):
        method_sel = METHOD_LS
    elif mraw in ("IDW", "IDW_DHVIS"):
        method_sel = METHOD_IDW
    else:
        method_sel = METHOD_LS

    root = Path(args.areas_root)
    targets = [root / args.area] if args.area else sorted([d for d in root.iterdir() if d.is_dir()])

    # Load density metrics per area for requested DEM/CORR
    area7: Dict[str, pd.DataFrame] = {}
    for area_dir in targets:
        df7 = _read_area_density_metrics(area_dir, dem=args.dem, corr=args.corr)
        if df7 is None or df7.empty:
            print(f"⏭️  No density metrics for area: {area_dir.name} (DEM={args.dem}, CORR={args.corr})")
        else:
            df7 = _ensure_upper(df7)
            area7[area_dir.name] = df7

    if not area7:
        print("⏭️  Nothing to visualize."); return

    # Per-pair figures
    for area_name, df7 in area7.items():
        area_dir = root / area_name
        pair_tags = sorted(set(df7.get("pair_tag", pd.Series([], dtype=str)).dropna().astype(str).tolist()))
        if not pair_tags and {"pair_ref","pair_sec"}.issubset(df7.columns):
            pair_tags = sorted(set(
                (df7["pair_ref"].str.replace("-", "", regex=False) + "_" +
                 df7["pair_sec"].str.replace("-", "", regex=False)).tolist()
            ))
        for pair_tag in pair_tags:
            try:
                plot_density_pair(
                    area_dir, df7, pair_tag,
                    dem=args.dem, corr=args.corr,
                    water_path=args.water_areas,
                    sat_provider=args.sat_provider,
                    sat_url=args.sat_url,
                    gauges_template=args.gauges_template,
                    maps_gap_frac=args.maps_gap_frac,
                    maps_height_frac=args.maps_height_frac
                )
            except Exception as e:
                print(f"⚠️  Density pair failed {area_name}:{pair_tag}: {e}")

    # All-areas figures (ALWAYS, no dots)
    try:
        out_dir = root / "results"
        plot_all_areas_density(area7, dem=args.dem, corr=args.corr, method=method_sel, out_dir=out_dir)
        plot_all_areas_combined(area7, dem=args.dem, corr=args.corr, method=method_sel, out_dir=out_dir)
    except Exception as e:
        print(f"⚠️  All-areas figures failed: {e}")

if __name__ == "__main__":
    main()
