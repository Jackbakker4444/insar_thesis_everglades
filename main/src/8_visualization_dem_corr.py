#!/usr/bin/env python3
"""
8_visualization_dem_corr.py
===========================

What you can tweak (maps only)
------------------------------
MAP_PANEL_HEIGHT_IN : Height of each map panel (inches). Width is computed from AOI aspect,
                      so the maps never skew.
COL_PAD_IN          : Outer padding (inches) on the **left** side of the two map columns.
TITLE_GAP_IN        : Vertical gap (inches) between the title and the top row of maps.
CBAR_OUTER_PAD_IN   : Extra padding (inches) **to the right of the colorbar** to prevent
                      tick labels from being cut off.

(Internals like the gap between the right column and the colorbar, and the colorbar width,
are handled automatically—but can be edited below if you want finer control.)

Outputs
-------
A) Per-pair corrections maps (3DEP) as a 2×3 grid:
      [ Raw                 |  Tropospheric               ]
      [ Ionospheric         |  Tropospheric + Ionospheric ]
      [ Soil (WATER only)   |  Satellite (WATER only)     ]
   • Two equal-width columns, all panels aligned (same AOI & aspect, equal sizes).
   • Single colorbar outside, to the right of the first two rows.
   • Soil & Satellite clipped to the **area-specific** water polygon from
     /home/bakke326l/InSAR/main/data/vector/water_areas.geojson
     (match on properties.area == <AREA DIR NAME>, case-insensitive).
   • North arrow + scalebar per panel.
   • Sources note in lower-left.

   → <area>/results/corr_maps_pair_<PAIR>_2x3.png

B) Per-area DEM boxplots (LS 60%, TROPO):
   1) Two DEMs (SRTM & 3DEP), equal widths, adjacent per pair.
   2) Two DEMs, variable widths proportional to each pair’s duration.
   • RMSE y-axis fixed to [0, 25] cm; Bias auto.
   • Legends inside (upper-right); outlier note upper-left.
   • Dates in short form on x-axis.

   → <area>/results/dem_boxplots_area_<AREA>_equalwidth_TROPO.png
   → <area>/results/dem_boxplots_area_<AREA>_varwidth_TROPO.png

C) ALL-areas combined:
   • Two DEMs, equal widths:
     → <areas_root>/results/dem_boxplots_ALL_areas_equalwidth_TROPO.png
   • Two DEMs, variable widths:
     → <areas_root>/results/dem_boxplots_ALL_areas_varwidth_TROPO.png

D) ALL-areas scatter: mean RMSE vs temporal baseline (days)
   → <areas_root>/results/scatter_mean_rmse_vs_temporal_baseline_TROPO.png
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
from matplotlib.patches import Rectangle, FancyArrow, Patch, PathPatch
from matplotlib.path import Path as MplPath
import matplotlib.dates as mdates

# Times new roman fonts
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",  # preferred
        "Times",            # generic Times
        "Nimbus Roman",     # common on Linux
        "TeX Gyre Termes",
        "Liberation Serif",
        "DejaVu Serif"
    ],
    "mathtext.fontset": "stix",   # math like $R^2$ matches Times-style
    "axes.unicode_minus": False,  # nicer minus sign with some serif fonts
    # If you ever export PDF/SVG and want real text (not paths):
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

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
    from shapely.geometry import Polygon, MultiPolygon
except Exception:
    Polygon = MultiPolygon = None
try:
    from shapely.ops import unary_union
except Exception:
    unary_union = None
try:
    from pyproj import Geod
except Exception:
    Geod = None

# ========================== TUNE THESE KNOBS ============================
MAP_PANEL_HEIGHT_IN = 3.2   # height of each map panel (inches). Width is computed (no skew).
COL_PAD_IN          = 0.30  # outer left padding (inches) before the two map columns
TITLE_GAP_IN        = 0.75  # gap (inches) between the title and the top row of maps
CBAR_OUTER_PAD_IN   = 0.6  # extra right padding (inches) *after* the colorbar (prevents clipping)
# =======================================================================

# Internals (you can tweak if needed)
_ROW_SPACE_IN    = 0.28     # vertical spacing between map rows (inches)
_COL_SPACE_IN    = 0.32     # horizontal spacing between the two map columns (inches)
_CBAR_GAP_IN     = 0.22     # gap (inches) between right map column and colorbar
_CBAR_WIDTH_IN   = 0.24     # colorbar width (inches)
_TOP_PAD_HEAD_IN = 0.22     # headroom above the title (inches)
_BOTTOM_PAD_IN   = 0.50     # bottom margin for source note etc. (inches)

# DEM figure sizes
DEM_FIG_W = 12.6
DEM_FIG_H = 9.0
DEM_ALL_FIG_W = 14.0
DEM_ALL_FIG_H = 9.8

# Date format
DATE_FMT_SHORT = "'%y-%m-%d"  # e.g. '08-03-17

# Corrections / DEM config
METHOD_LS = "LEAST_SQUARES"
DEMS      = ["SRTM", "3DEP"]
RMSE_YLIMS = (0.0, 25.0)  # fixed

# Colors
CB = {"black":"#000000","blue":"#0072B2","green":"#009E73","orange":"#D55E00"}
COLORS_CORR = {"RAW":CB["black"], "TROPO":CB["blue"], "IONO":CB["green"], "TROPO_IONO":CB["orange"]}
COLORS_DEM  = {"SRTM":CB["blue"],   "3DEP":CB["green"]}
CMAP_INV    = "viridis_r"

# Paths / defaults
AREAS_ROOT_DEFAULT = Path("/mnt/DATA2/bakke326l/processing/areas")
PERP_BASELINES_CSV_DEFAULT = Path("/home/bakke326l/InSAR/main/data/perpendicular_baselines.csv")
DEF_WATER_AREAS    = "/home/bakke326l/InSAR/main/data/vector/water_areas.geojson"
DEF_SAT_PROVIDER   = "Esri.WorldImagery"
DEF_SAT_URL        = ""  # custom XYZ if you want
DEF_SOIL_TIF       = "/home/bakke326l/InSAR/main/data/aux/raster/soil_map_florida.tif"


# Quiet noisy libs
os.environ.setdefault("CPL_DEBUG", "NO")
for _n in ("rasterio","rasterio._io","rasterio.env","rasterio._base","matplotlib.font_manager"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# ============================== Helpers (I/O) ==============================
def _pair_dates_from_tag(pair_tag: str) -> Tuple[str, str]:
    a, b = pair_tag.split("_")
    return f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"

def _ensure_upper(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("dem","corr","method"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()
    return df

def _read_area_metrics(area_dir: Path) -> Optional[pd.DataFrame]:
    f = area_dir / "results" / "accuracy_metrics.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    df = _ensure_upper(df)
    for c in ("pair_ref","pair_sec"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
    if {"pair_ref","pair_sec"}.issubset(df.columns):
        df["pair_tag"] = (
            df["pair_ref"].str.replace("-","",regex=False) + "_" +
            df["pair_sec"].str.replace("-","",regex=False)
        )
    return df

def _read_tif_array(path: Optional[Path]) -> Optional[Tuple[np.ndarray, Tuple[float,float,float,float]]]:
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

def _collect_pair_tags_from_maps(area_dir: Path) -> List[str]:
    resdir = area_dir / "results"
    tags = set()
    for globpat in ("cal_60pct_*_*.tif","cal_ti_60pct_*_*.tif","dens_cal_60pct_*_*.tif","*SRTM_*_*.tif"):
        for p in resdir.glob(globpat):
            m = re.match(r".+_(\d{8}_\d{8})\.tif", p.name)
            if m: tags.add(m.group(1))
    return sorted(tags)

def _read_bperp_csv(perp_csv: Path) -> Optional[pd.DataFrame]:
    """
    Read perpendicular baselines CSV produced by help_perpendicular_baseline.py.
    Expects columns: ref_date (YYYYMMDD), sec_date (YYYYMMDD), bperp_abs_m.
    Returns a dataframe with pair_ref, pair_sec (YYYY-MM-DD), bperp_abs_m (float).
    """
    if not perp_csv.exists():
        print(f"⏭️  Perp-baseline CSV not found: {perp_csv}")
        return None
    df = pd.read_csv(perp_csv, dtype=str)
    need = {"ref_date", "sec_date", "bperp_abs_m"}
    if not need.issubset(df.columns):
        print(f"⏭️  Perp-baseline CSV missing columns {need - set(df.columns)}: {perp_csv}")
        return None
    df["pair_ref"] = pd.to_datetime(df["ref_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    df["pair_sec"] = pd.to_datetime(df["sec_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    df["bperp_abs_m"] = pd.to_numeric(df["bperp_abs_m"], errors="coerce")
    df = df.dropna(subset=["pair_ref","pair_sec","bperp_abs_m"])
    # If duplicates exist (e.g., both SRTM/3DEP runs), average them:
    df = (df.groupby(["pair_ref","pair_sec"], as_index=False)
            .agg(bperp_abs_m=("bperp_abs_m","mean")))
    return df


# ====================== Map helpers: scalebar & north arrow ===================
def _geod() -> Optional[Geod]:
    if Geod is None: return None
    return Geod(ellps="WGS84")

def _map_width_km(ext: Tuple[float,float,float,float]) -> float:
    geod = _geod()
    xmin, xmax, ymin, ymax = ext; lat = (ymin + ymax)/2.0
    if geod:
        _, _, d = geod.inv(xmin, lat, xmax, lat)
        return max(d/1000.0, 1e-6)
    return (xmax-xmin)*111.32*np.cos(np.deg2rad(lat))

def _nice_scale_bar(width_km: float) -> float:
    target = max(width_km*0.22, 0.001)
    for s in (1,2,5):
        pow10 = 10**int(np.floor(np.log10(target))) if target>0 else 1
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

def _draw_scalebar(ax, extent: Tuple[float,float,float,float], *, pad_frac=0.04):
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

def _draw_north_arrow(ax, extent: Tuple[float,float,float,float], *, size_frac=0.08, pad_frac=0.05):
    xmin, xmax, ymin, ymax = extent
    dx = xmax-xmin; dy = ymax-ymin
    x = xmax - pad_frac*dx; y = ymin + pad_frac*dy
    size = size_frac*dy
    ax.add_patch(FancyArrow(x, y, 0, size, width=size*0.12, head_width=size*0.28, head_length=size*0.28,
                            length_includes_head=True, color="k"))
    ax.text(x, y+size+0.01*dy, "N", ha="center", va="bottom", fontsize=9, color="k")

# ========================= Basemap / overlays / soil =========================
def _cx_get_provider(provider_name: str):
    prov = cx.providers
    for token in provider_name.split("."):
        prov = getattr(prov, token, None)
        if prov is None:
            return cx.providers.Esri.WorldImagery
    return prov

def _add_basemap(ax, extent: Tuple[float,float,float,float], *, provider_name: str, xyz_url: str):
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ok = False
    if cx is not None:
        try:
            if xyz_url:
                cx.add_basemap(ax, source=xyz_url, crs="EPSG:4326", attribution=False)
            else:
                prov = _cx_get_provider(provider_name)
                cx.add_basemap(ax, source=prov, crs="EPSG:4326", attribution=False)
            ok = True
        except Exception:
            ok = False
    if not ok:
        ax.set_facecolor("#dddddd")
        ax.text(0.5, 0.5, "Satellite basemap unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#444")

def _soil_png_from_tif(soil_tif: Path) -> Optional[Path]:
    if not soil_tif or not soil_tif.exists():
        return None
    png = soil_tif.with_suffix(".png")
    if png.exists():
        return png
    try:
        with rasterio.open(soil_tif) as ds:
            arr = ds.read()
            if arr.dtype != np.uint8:
                arrf = arr.astype(float)
                lo = np.nanpercentile(arrf, 2); hi = np.nanpercentile(arrf, 98)
                if not np.isfinite(lo): lo = np.nanmin(arrf)
                if not np.isfinite(hi): hi = np.nanmax(arrf)
                if hi <= lo: hi = lo + 1.0
                arr8 = np.clip((arrf - lo)/(hi-lo)*255.0, 0, 255).astype(np.uint8)
            else:
                arr8 = arr
            if arr8.shape[0] == 1:
                rgb = np.repeat(arr8, 3, axis=0)
            elif arr8.shape[0] >= 3:
                rgb = arr8[:3]
            else:
                rgb = np.repeat(arr8, 3, axis=0)
            rgb = np.transpose(rgb, (1,2,0))
        plt.imsave(png, rgb)
        return png
    except Exception:
        return None

# ----------------------- Water geometry for the current AREA ------------------
def _get_area_water_geom(area_name: str, water_path: Optional[str]):
    if not water_path or gpd is None:
        return None
    try:
        w = gpd.read_file(water_path)
        if getattr(w, "crs", None) and w.crs and str(w.crs).upper() not in ("EPSG:4326","WGS84","OGC:CRS84"):
            w = w.to_crs(epsg=4326)
        key = area_name.strip().upper()
        sub = w[w.get("area","").astype(str).str.upper().str.strip() == key] if "area" in w.columns else w.iloc[0:0]
        if sub.empty and "name" in w.columns:
            sub = w[w["name"].astype(str).str.upper().str.contains(key)]
        if sub.empty:
            return None
        return unary_union(sub.geometry) if unary_union is not None else sub.unary_union
    except Exception:
        return None

# ---------- Clip helpers (show ONLY inside geometry, blank outside) ----------
def _poly_to_mpl_path(poly: Polygon) -> MplPath:
    def _ring_to_verts_codes(coords):
        verts = [(x, y) for (x, y) in coords]
        codes = [MplPath.MOVETO] + [MplPath.LINETO]*(len(verts)-2) + [MplPath.CLOSEPOLY]
        return verts, codes
    exterior = poly.exterior
    verts, codes = _ring_to_verts_codes(exterior.coords)
    all_verts = verts[:]; all_codes = codes[:]
    for interior in poly.interiors:
        v, c = _ring_to_verts_codes(interior.coords)
        all_verts.extend(v); all_codes.extend(c)
    return MplPath(all_verts, all_codes)

def _geom_to_clip_patch(ax, geom) -> Optional[PathPatch]:
    if geom is None or Polygon is None:
        return None
    polys = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        try:
            for sub in geom:
                if isinstance(sub, Polygon):
                    polys.append(sub)
        except Exception:
            pass
    if not polys:
        return None
    verts_all, codes_all = [], []
    for p in polys:
        path = _poly_to_mpl_path(p)
        verts_all.extend(path.vertices.tolist())
        codes_all.extend(path.codes.tolist())
    path_combined = MplPath(verts_all, codes_all)
    return PathPatch(path_combined, facecolor="none", edgecolor="none", transform=ax.transData)

def _clip_ax_images_to_geom(ax, geom):
    patch = _geom_to_clip_patch(ax, geom)
    if patch is None: return
    ax.set_facecolor("white")
    for im in list(ax.images):
        im.set_clip_path(patch)

# ============================ CORRECTIONS (2×3) ==============================
def plot_corrections_sixpack(
    area_dir: Path,
    pair_tag: str,
    *,
    water_path: Optional[str],
    sat_provider: str,
    sat_url: str,
    soil_tif: Optional[Path],
):
    """Two aligned columns × three rows of maps, equal sizes; colorbar to the right of top two rows."""
    area = area_dir.name
    ref_iso, sec_iso = _pair_dates_from_tag(pair_tag)

    label_map = {
        "RAW": "Raw",
        "TROPO": "Tropospheric",
        "IONO": "Ionospheric",
        "TROPO_IONO": "Tropospheric + Ionospheric",
    }
    order = [("RAW","RAW"), ("TROPO","TROPO"), ("IONO","IONO"), ("TROPO_IONO","TROPO_IONO")]

    # Load correction rasters
    corr_arrays: List[Optional[Tuple[np.ndarray, Tuple[float,float,float,float]]]] = []
    for _, corr in order:
        p = _find_corr_map(area_dir, "3DEP", corr, pair_tag)
        corr_arrays.append(_read_tif_array(p))
    if all(m is None for m in corr_arrays):
        print(f"⏭️  No 3DEP correction rasters found for {area}:{pair_tag}.")
        return

    # Shared color stretch
    vals = [m[0][np.isfinite(m[0])] for m in corr_arrays if m is not None]
    vmin, vmax = (0.0, 1.0)
    if len(vals):
        vmin, vmax = np.nanpercentile(np.concatenate(vals), [2, 98])

    # AOI extent (union across rasters)
    extents = [m[1] for m in corr_arrays if m is not None]
    aoi_extent = (min(e[0] for e in extents), max(e[1] for e in extents),
                  min(e[2] for e in extents), max(e[3] for e in extents))

    # Soil preview
    soil_png: Optional[Path] = None
    soil_extent: Optional[Tuple[float,float,float,float]] = None
    if soil_tif:
        soil_png = _soil_png_from_tif(soil_tif)
        try:
            with rasterio.open(soil_tif) as ds:
                soil_extent = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top)
        except Exception:
            soil_extent = None

    # Water geom for this area
    area_geom = _get_area_water_geom(area, water_path)

    # ---------- Compute figure size from AOI aspect and the four knobs ----------
    dx = aoi_extent[1] - aoi_extent[0]
    dy = aoi_extent[3] - aoi_extent[2]
    if dy == 0: dy = 1e-6
    panel_h_in = float(MAP_PANEL_HEIGHT_IN)
    panel_w_in = panel_h_in * (dx / dy)  # preserve aspect (no skew)

    fig_h_in = _BOTTOM_PAD_IN + 3*panel_h_in + 2*_ROW_SPACE_IN + (_TOP_PAD_HEAD_IN + TITLE_GAP_IN)
    fig_w_in = (COL_PAD_IN +                      # left outer pad (before columns)
                2*panel_w_in + _COL_SPACE_IN +    # two columns + gap
                _CBAR_GAP_IN + _CBAR_WIDTH_IN +   # colorbar outside right of columns
                CBAR_OUTER_PAD_IN)                # extra right pad AFTER colorbar (prevents clipping)

    # Normalize margins & spaces for subplots_adjust
    left_norm   = COL_PAD_IN / fig_w_in
    # Right edge of the subplot area stops BEFORE the colorbar + its right padding
    right_norm  = (fig_w_in - (_CBAR_GAP_IN + _CBAR_WIDTH_IN + CBAR_OUTER_PAD_IN)) / fig_w_in
    bottom_norm = _BOTTOM_PAD_IN / fig_h_in
    top_norm    = (fig_h_in - (_TOP_PAD_HEAD_IN + TITLE_GAP_IN)) / fig_h_in
    wspace_frac = _COL_SPACE_IN / panel_w_in
    hspace_frac = _ROW_SPACE_IN / panel_h_in

    # Figure + grid
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=150)
    fig.subplots_adjust(left=left_norm, right=right_norm, bottom=bottom_norm, top=top_norm,
                        wspace=wspace_frac, hspace=hspace_frac)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1,1,1], width_ratios=[1,1])

    # Axes
    axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
            fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]),
            fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1])]

    # Helper to style each map
    def _prep_ax(ax, title: str):
        ax.set_xlim(aoi_extent[0], aoi_extent[1])
        ax.set_ylim(aoi_extent[2], aoi_extent[3])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10, pad=3)
        _draw_scalebar(ax, aoi_extent); _draw_north_arrow(ax, aoi_extent)

    # First two rows (corrections)
    ims = []
    for ax, (disp, corr), m in zip(axes[:4], order, corr_arrays):
        if m is None:
            ax.text(0.5, 0.5, f"Missing {label_map[disp]}", ha="center", va="center")
            ax.set_axis_off(); ims.append(None); continue
        arr, extent = m
        im = ax.imshow(arr, extent=extent, origin="upper", cmap=CMAP_INV, vmin=vmin, vmax=vmax)
        ims.append(im)
        _prep_ax(ax, label_map[disp])

    # Soil (WATER-only)
    ax_soil = axes[4]
    if soil_png is not None and soil_extent is not None:
        img = plt.imread(str(soil_png))
        ax_soil.imshow(img, extent=soil_extent, origin="upper")
        _prep_ax(ax_soil, "Soil")
        _clip_ax_images_to_geom(ax_soil, area_geom)
    else:
        ax_soil.text(0.5, 0.5, "Soil PNG/TIF unavailable", ha="center", va="center")
        ax_soil.set_axis_off()

    # Satellite (WATER-only)
    ax_sat = axes[5]
    _add_basemap(ax_sat, aoi_extent, provider_name=sat_provider, xyz_url=sat_url)
    _prep_ax(ax_sat, "Satellite")
    _clip_ax_images_to_geom(ax_sat, area_geom)

    # Title (place at a distance of TITLE_GAP_IN above the top row)
    top_axes_y = max(axes[0].get_position().y1, axes[1].get_position().y1)
    title_y = min(0.99, top_axes_y + (TITLE_GAP_IN / fig_h_in))
    fig.suptitle(
        f"{area} — 3DEP vertical displacement corrections\n{ref_iso} to {sec_iso}",
        y=title_y, fontsize=14, fontweight="bold"
    )

    # Colorbar — outside, to the right of the first two rows
    valid_ims = [im for im in ims if im is not None]
    if valid_ims:
        ax_tr = axes[1]; ax_mr = axes[3]
        pos_tr = ax_tr.get_position(); pos_mr = ax_mr.get_position()
        cbar_x0 = pos_tr.x1 + (_CBAR_GAP_IN / fig_w_in)
        cbar_y0 = pos_mr.y0
        cbar_w  = _CBAR_WIDTH_IN / fig_w_in
        cbar_h  = pos_tr.y1 - pos_mr.y0
        cax = fig.add_axes([cbar_x0, cbar_y0, cbar_w, cbar_h])
        cb = fig.colorbar(valid_ims[0], cax=cax, orientation="vertical")
        cb.set_label("cm", fontsize=10); cb.ax.tick_params(labelsize=8)

    # Sources note (lower-left, under bottom-left axes)
    try:
        pos_ll = axes[4].get_position()
        y = pos_ll.y0 - (0.026 if fig_h_in >= 8 else 0.022)
        src_sat = "Custom XYZ" if sat_url else "Esri World Imagery"
        fig.text(pos_ll.x0, y, f"Sources: ALOS PALSAR, 3DEP, GACOS — Satellite: {src_sat}",
                 ha="left", va="top", fontsize=9, color="#333333")
    except Exception:
        pass

    out = area_dir / "results" / f"corr_maps_pair_{pair_tag}_2x3.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"🖼️  Corrections 2×3 maps written: {out}")

# ========================== DEM boxplots (TROPO) ===============================
def _pick_rows_60pct(df: pd.DataFrame, corr_used: str = "TROPO") -> pd.DataFrame:
    sub = df[(df["method"].str.upper()==METHOD_LS) & (df["corr"].str.upper()==corr_used.upper())].copy()
    if sub.empty: return pd.DataFrame(columns=df.columns)
    out = []
    for (pref,psec), g in sub.groupby(["pair_ref","pair_sec"], as_index=False):
        if "n_cal" not in g.columns: continue
        vals = g["n_cal"].dropna().astype(int).unique()
        if vals.size == 0: continue
        n60 = int(np.max(vals))
        out.append(g[g["n_cal"].astype(int)==n60])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=sub.columns)

def _pair_meta(g: pd.DataFrame) -> pd.DataFrame:
    to_dt = lambda s: pd.to_datetime(s, format="%Y-%m-%d")
    meta = (g[["pair_ref","pair_sec"]].drop_duplicates()
            .assign(t_ref=lambda d: to_dt(d["pair_ref"]),
                    t_sec=lambda d: to_dt(d["pair_sec"]))
            .assign(t_mid=lambda d: d["t_ref"] + (d["t_sec"] - d["t_ref"])/2,
                    span_days=lambda d: (d["t_sec"] - d["t_ref"]).dt.days.clip(lower=1).astype(float))
            .sort_values("t_mid"))
    return meta

def _fmt_pair_label(pref: str, psec: str) -> str:
    pr = pd.to_datetime(pref).strftime(DATE_FMT_SHORT)
    ps = pd.to_datetime(psec).strftime(DATE_FMT_SHORT)
    return f"{pr} – {ps}"

def _annotate_outliers(ax, values: np.ndarray, y_min: float, y_max: float):
    vals = np.asarray(values, dtype=float)
    n_above = int(np.isfinite(vals[vals > y_max]).sum())
    n_below = int(np.isfinite(vals[vals < y_min]).sum())
    if n_above or n_below:
        ax.text(0.015, 0.98,
                f"Outliers beyond axis: +{n_above}" + (f", -{n_below}" if n_below else ""),
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#444")

def _simple_box(ax, values: np.ndarray, center: float, width: float, color: str):
    if values.size == 0: return
    bp = ax.boxplot([values], positions=[center], widths=[width],
                    whis=1.5, patch_artist=True, showfliers=True)
    for box in bp["boxes"]:   box.set(facecolor=to_rgba(color,0.30), edgecolor=color, linewidth=1.2)
    for whisk in bp["whiskers"]: whisk.set(color=color, linewidth=1.2)
    for cap in bp["caps"]:       cap.set(color=color, linewidth=1.2)
    for med in bp["medians"]:    med.set(color="k", linewidth=1.8)
    for fl in bp["fliers"]:      fl.set(marker="o", ms=3.5, mfc=color, mec="white", alpha=0.85)

def _both_dems_equalwidth_adjacent(ax, g: pd.DataFrame, meta: pd.DataFrame,
                                   metric: str, ylab: str, area_label: str,
                                   add_legend: bool = True):
    pairs = list(meta[["pair_ref","pair_sec","t_mid"]].itertuples(index=False, name=None))
    n = len(pairs)
    x_centers = np.arange(n, dtype=float)
    width = 0.35
    all_vals = []
    for i, (pref, psec, _) in enumerate(pairs):
        for dem, dx in (("SRTM",-0.20), ("3DEP",+0.20)):
            vals = g[(g["pair_ref"]==pref) & (g["pair_sec"]==psec) & (g["dem"]==dem)][metric].to_numpy()
            all_vals.append(vals)
            _simple_box(ax, vals, x_centers[i] + dx, width, COLORS_DEM[dem])
    labels = [_fmt_pair_label(pref, psec) for (pref, psec, _) in pairs]
    ax.set_xticks(x_centers); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylab, fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    if metric == "rmse_cm":
        ax.set_ylim(*RMSE_YLIMS)
        if any(v.size for v in all_vals):
            _annotate_outliers(ax, np.concatenate([v for v in all_vals if v.size]), *RMSE_YLIMS)
    ax.set_title(f"{area_label} — Two DEMs (equal width) — {ylab}", fontsize=10)
    if add_legend:
        handles=[Patch(facecolor=to_rgba(COLORS_DEM["SRTM"],0.30), edgecolor=COLORS_DEM["SRTM"], label="SRTM"),
                 Patch(facecolor=to_rgba(COLORS_DEM["3DEP"],0.30), edgecolor=COLORS_DEM["3DEP"], label="3DEP")]
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

def _both_dems_varwidth_time(ax, g: pd.DataFrame, meta: pd.DataFrame,
                             metric: str, ylab: str, area_label: str,
                             add_legend: bool = True):
    for _, row in meta.iterrows():
        t_ref_num = mdates.date2num(row["t_ref"]); t_sec_num = mdates.date2num(row["t_sec"])
        inner = (t_sec_num - t_ref_num) * 0.80
        w_each = inner / 2.0
        c_srtm = t_ref_num + (t_sec_num - t_ref_num) * 0.30
        c_3dep = t_ref_num + (t_sec_num - t_ref_num) * 0.70
        for dem, c in (("SRTM", c_srtm), ("3DEP", c_3dep)):
            vals = g[(g["pair_ref"]==row["pair_ref"]) & (g["pair_sec"]==row["pair_sec"]) & (g["dem"]==dem)][metric].to_numpy()
            _simple_box(ax, vals, c, w_each*0.95, COLORS_DEM[dem])
        ax.axvline(row["t_ref"], color="#dddddd", lw=0.8, zorder=0)
        ax.axvline(row["t_sec"], color="#dddddd", lw=0.8, zorder=0)
    ax.set_ylabel(ylab, fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    if metric == "rmse_cm":
        ax.set_ylim(*RMSE_YLIMS); _annotate_outliers(ax, g[metric].to_numpy(), *RMSE_YLIMS)
    xlims = (meta["t_ref"].min(), meta["t_sec"].max())
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FMT_SHORT))
    ax.tick_params(axis="x", rotation=45, labelsize=8); ax.set_xlim(xlims)
    ax.set_title(f"{area_label} — Two DEMs (variable width) — {ylab}", fontsize=10)
    if add_legend:
        handles=[Patch(facecolor=to_rgba(COLORS_DEM["SRTM"],0.30), edgecolor=COLORS_DEM["SRTM"], label="SRTM"),
                 Patch(facecolor=to_rgba(COLORS_DEM["3DEP"],0.30), edgecolor=COLORS_DEM["3DEP"], label="3DEP")]
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

def _plot_area_boxplots(area_dir: Path, df_area: pd.DataFrame):
    area = area_dir.name
    sub = _pick_rows_60pct(df_area, corr_used="TROPO")
    if sub.empty:
        print(f"⏭️  No LS TROPO 60% rows for {area}; skip DEM boxplots.")
        return
    sub["dem"] = sub["dem"].astype(str).str.upper()
    sub = sub[sub["dem"].isin(["SRTM","3DEP"])]
    if sub.empty:
        print(f"⏭️  No SRTM/3DEP rows for {area} at 60%."); return
    meta = _pair_meta(sub)

    # Equal widths, adjacent
    figA, (axAt, axAb) = plt.subplots(nrows=2, ncols=1, figsize=(DEM_FIG_W, DEM_FIG_H), dpi=150, constrained_layout=True)
    _both_dems_equalwidth_adjacent(axAt, sub, meta, "rmse_cm", "RMSE (cm)", area, add_legend=True)
    _both_dems_equalwidth_adjacent(axAb, sub, meta, "bias_cm", "Bias (cm)", area, add_legend=False)
    axAb.set_xlabel("Pair dates (informational; equal spacing)", fontsize=10)
    outA = area_dir / "results" / f"dem_boxplots_area_{area}_equalwidth_TROPO.png"
    outA.parent.mkdir(parents=True, exist_ok=True)
    figA.savefig(outA, dpi=150); plt.close(figA)
    print(f"📦 DEM boxplots (equal width, both DEMs): {outA}")

    # Variable widths by duration (two DEMs)
    figB, (axBt, axBb) = plt.subplots(nrows=2, ncols=1, figsize=(DEM_FIG_W, DEM_FIG_H), dpi=150, constrained_layout=True)
    _both_dems_varwidth_time(axBt, sub, meta, "rmse_cm", "RMSE (cm)", area, add_legend=True)
    _both_dems_varwidth_time(axBb, sub, meta, "bias_cm", "Bias (cm)", area, add_legend=False)
    axBb.set_xlabel("Time (dates on x-axis; widths reflect pair durations)", fontsize=10)
    outB = area_dir / "results" / f"dem_boxplots_area_{area}_varwidth_TROPO.png"
    figB.savefig(outB, dpi=150); plt.close(figB)
    print(f"📦 DEM boxplots (variable width, both DEMs): {outB}")

# ---------------------------- ALL-AREAS combined -----------------------------
def _collect_all_df(area_df: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    all_subs = []
    for df in area_df.values():
        s = _pick_rows_60pct(df, corr_used="TROPO")
        if s.empty: continue
        s = s[s["dem"].str.upper().isin(["SRTM","3DEP"])]
        if not s.empty: all_subs.append(s)
    if not all_subs:
        return None
    return pd.concat(all_subs, ignore_index=True)

def _plot_all_areas_equalwidth(root: Path, area_df: Dict[str, pd.DataFrame]):
    all_df = _collect_all_df(area_df)
    if all_df is None:
        print("⏭️  ALL-areas combined: no LS TROPO 60% rows.")
        return
    meta = _pair_meta(all_df)

    fig, (ax_t, ax_b) = plt.subplots(nrows=2, ncols=1, figsize=(DEM_ALL_FIG_W, DEM_ALL_FIG_H), dpi=150, constrained_layout=True)
    pairs = list(meta[["pair_ref","pair_sec","t_mid"]].itertuples(index=False, name=None))
    n = len(pairs); x_centers = np.arange(n, dtype=float); width = 0.35

    # RMSE
    rmse_all_vals = []
    for i, (pref, psec, _) in enumerate(pairs):
        for dem, dx in (("SRTM",-0.20), ("3DEP",+0.20)):
            vals = all_df[(all_df["pair_ref"]==pref) & (all_df["pair_sec"]==psec) & (all_df["dem"]==dem)]["rmse_cm"].to_numpy()
            rmse_all_vals.append(vals)
            _simple_box(ax_t, vals, x_centers[i] + dx, width, COLORS_DEM[dem])
    labels = [_fmt_pair_label(pref, psec) for (pref, psec, _) in pairs]
    ax_t.set_xticks(x_centers); ax_t.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_t.set_ylabel("RMSE (cm)", fontsize=10); ax_t.grid(True, alpha=0.3, axis="y")
    ax_t.set_ylim(*RMSE_YLIMS)
    if any(v.size for v in rmse_all_vals):
        _annotate_outliers(ax_t, np.concatenate([v for v in rmse_all_vals if v.size]), *RMSE_YLIMS)
    ax_t.set_title("All areas — Two DEMs (equal width) — RMSE", fontsize=10)
    ax_t.legend(handles=[Patch(facecolor=to_rgba(COLORS_DEM["SRTM"],0.30), edgecolor=COLORS_DEM["SRTM"], label="SRTM"),
                         Patch(facecolor=to_rgba(COLORS_DEM["3DEP"],0.30), edgecolor=COLORS_DEM["3DEP"], label="3DEP")],
                loc="upper right", frameon=True, fontsize=9)

    # Bias
    for i, (pref, psec, _) in enumerate(pairs):
        for dem, dx in (("SRTM",-0.20), ("3DEP",+0.20)):
            vals = all_df[(all_df["pair_ref"]==pref) & (all_df["pair_sec"]==psec) & (all_df["dem"]==dem)]["bias_cm"].to_numpy()
            _simple_box(ax_b, vals, x_centers[i] + dx, width, COLORS_DEM[dem])
    ax_b.set_xticks(x_centers); ax_b.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_b.set_ylabel("Bias (cm)", fontsize=10); ax_b.grid(True, alpha=0.3, axis="y")
    ax_b.set_title("All areas — Two DEMs (equal width) — Bias", fontsize=10)
    ax_b.legend(handles=[Patch(facecolor=to_rgba(COLORS_DEM["SRTM"],0.30), edgecolor=COLORS_DEM["SRTM"], label="SRTM"),
                         Patch(facecolor=to_rgba(COLORS_DEM["3DEP"],0.30), edgecolor=COLORS_DEM["3DEP"], label="3DEP")],
                loc="upper right", frameon=True, fontsize=9)
    ax_b.set_xlabel("Pair dates (informational; equal spacing)", fontsize=10)

    out = root / "results" / "dem_boxplots_ALL_areas_equalwidth_TROPO.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"🌐 ALL-areas (equal width) written: {out}")

def _plot_all_areas_varwidth(root: Path, area_df: Dict[str, pd.DataFrame]):
    all_df = _collect_all_df(area_df)
    if all_df is None:
        print("⏭️  ALL-areas varwidth: no LS TROPO 60% rows.")
        return
    meta = _pair_meta(all_df)

    fig, (ax_t, ax_b) = plt.subplots(nrows=2, ncols=1, figsize=(DEM_ALL_FIG_W, DEM_ALL_FIG_H), dpi=150, constrained_layout=True)

    # RMSE
    for _, row in meta.iterrows():
        t_ref_num = mdates.date2num(row["t_ref"]); t_sec_num = mdates.date2num(row["t_sec"])
        inner = (t_sec_num - t_ref_num) * 0.80
        w_each = inner / 2.0
        c_srtm = t_ref_num + (t_sec_num - t_ref_num) * 0.30
        c_3dep = t_ref_num + (t_sec_num - t_ref_num) * 0.70
        for dem, c in (("SRTM", c_srtm), ("3DEP", c_3dep)):
            vals = all_df[(all_df["pair_ref"]==row["pair_ref"]) & (all_df["pair_sec"]==row["pair_sec"]) & (all_df["dem"]==dem)]["rmse_cm"].to_numpy()
            _simple_box(ax_t, vals, c, w_each*0.95, COLORS_DEM[dem])
        ax_t.axvline(row["t_ref"], color="#dddddd", lw=0.8, zorder=0)
        ax_t.axvline(row["t_sec"], color="#dddddd", lw=0.8, zorder=0)
    ax_t.set_ylabel("RMSE (cm)", fontsize=10); ax_t.grid(True, alpha=0.3, axis="y")
    ax_t.set_ylim(*RMSE_YLIMS); _annotate_outliers(ax_t, all_df["rmse_cm"].to_numpy(), *RMSE_YLIMS)
    xlims = (meta["t_ref"].min(), meta["t_sec"].max())
    ax_t.xaxis_date(); ax_t.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FMT_SHORT))
    ax_t.tick_params(axis="x", rotation=45, labelsize=8); ax_t.set_xlim(xlims)
    ax_t.set_title("All areas — Two DEMs (variable width) — RMSE", fontsize=10)
    ax_t.legend(handles=[Patch(facecolor=to_rgba(COLORS_DEM["SRTM"],0.30), edgecolor=COLORS_DEM["SRTM"], label="SRTM"),
                         Patch(facecolor=to_rgba(COLORS_DEM["3DEP"],0.30), edgecolor=COLORS_DEM["3DEP"], label="3DEP")],
                loc="upper right", frameon=True, fontsize=9)

    # Bias
    for _, row in meta.iterrows():
        t_ref_num = mdates.date2num(row["t_ref"]); t_sec_num = mdates.date2num(row["t_sec"])
        inner = (t_sec_num - t_ref_num) * 0.80
        w_each = inner / 2.0
        c_srtm = t_ref_num + (t_sec_num - t_ref_num) * 0.30
        c_3dep = t_ref_num + (t_sec_num - t_ref_num) * 0.70
        for dem, c in (("SRTM", c_srtm), ("3DEP", c_3dep)):
            vals = all_df[(all_df["pair_ref"]==row["pair_ref"]) & (all_df["pair_sec"]==row["pair_sec"]) & (all_df["dem"]==dem)]["bias_cm"].to_numpy()
            _simple_box(ax_b, vals, c, w_each*0.95, COLORS_DEM[dem])
        ax_b.axvline(row["t_ref"], color="#dddddd", lw=0.8, zorder=0)
        ax_b.axvline(row["t_sec"], color="#dddddd", lw=0.8, zorder=0)
    ax_b.set_ylabel("Bias (cm)", fontsize=10); ax_b.grid(True, alpha=0.3, axis="y")
    ax_b.xaxis_date(); ax_b.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FMT_SHORT))
    ax_b.tick_params(axis="x", rotation=45, labelsize=8); ax_b.set_xlim(xlims)
    ax_b.set_title("All areas — Two DEMs (variable width) — Bias", fontsize=10)
    ax_b.legend(handles=[Patch(facecolor=to_rgba(COLORS_DEM["SRTM"],0.30), edgecolor=COLORS_DEM["SRTM"], label="SRTM"),
                         Patch(facecolor=to_rgba(COLORS_DEM["3DEP"],0.30), edgecolor=COLORS_DEM["3DEP"], label="3DEP")],
                loc="upper right", frameon=True, fontsize=9)
    ax_b.set_xlabel("Time (dates on x-axis; widths reflect pair durations)", fontsize=10)

    out = root / "results" / "dem_boxplots_ALL_areas_varwidth_TROPO.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"🌐 ALL-areas (variable width) written: {out}")

# ------------------- ALL-areas scatter: mean RMSE vs baseline ----------------
def _plot_scatter_rmse_vs_temporal_and_bperp(root: Path,
                                             area_df: Dict[str, pd.DataFrame],
                                             perp_csv: Path):
    """
    Left: mean RMSE vs temporal baseline (days) — as before.
    Right: mean RMSE vs |B⊥| (km), merged from perpendicular_baselines.csv.
    """
    # Build per-pair mean RMSE (across DEMs) and temporal baseline (days)
    rows = []
    for _, df in area_df.items():
        s = _pick_rows_60pct(df, corr_used="RAW")
        if s.empty: 
            continue
        s = s[s["dem"].str.upper().isin(["SRTM","3DEP"])]
        if s.empty:
            continue
        g = (s.groupby(["pair_ref","pair_sec"], as_index=False)
               .agg(mean_rmse_cm=("rmse_cm","mean")))
        g["t_ref"] = pd.to_datetime(g["pair_ref"])
        g["t_sec"] = pd.to_datetime(g["pair_sec"])
        g["baseline_days"] = (g["t_sec"] - g["t_ref"]).dt.days.astype(float)
        rows.append(g[["pair_ref","pair_sec","mean_rmse_cm","baseline_days"]])

    if not rows:
        print("⏭️  Temporal/B⊥ scatter: no data.")
        return

    rmse_df = pd.concat(rows, ignore_index=True).dropna(subset=["baseline_days","mean_rmse_cm"])

    # Read perpendicular baselines and merge
    bperp_df = _read_bperp_csv(perp_csv)
    if bperp_df is None or bperp_df.empty:
        print("⏭️  No perpendicular baseline data -> only temporal scatter will be plotted by existing function.")
        return

    m = rmse_df.merge(bperp_df, on=["pair_ref","pair_sec"], how="inner")
    if m.empty:
        print("⏭️  No overlapping pairs between RMSE table and perpendicular baselines.")
        return

    # Prepare figure: two panels
    fig, (axL, axR) = plt.subplots(nrows=1, ncols=2, figsize=(14.5, 6.2), dpi=150, constrained_layout=True)

    # --- Left: RMSE vs temporal baseline (days) ---
    axL.scatter(m["baseline_days"], m["mean_rmse_cm"], s=22, alpha=0.8, label="Pairs")
    if len(m) >= 2:
        x = m["baseline_days"].to_numpy(float); y = m["mean_rmse_cm"].to_numpy(float)
        m1, b1 = np.polyfit(x, y, 1)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200); y_line = m1*x_line + b1
        axL.plot(x_line, y_line, linewidth=2.0, color="#424141", label="Linear fit")
        y_hat = m1*x + b1; ss_res = np.sum((y-y_hat)**2); ss_tot = np.sum((y-np.mean(y))**2)
        r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
        axL.text(0.02, 0.98, f"Fit: y = {m1:.3f}·x + {b1:.3f}\n$R^2$ = {r2:.3f}",
                 transform=axL.transAxes, ha="left", va="top", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"))
    axL.set_xlabel("Temporal baseline (days)", fontsize=10)
    axL.set_ylabel("Mean RMSE (cm) across DEMs", fontsize=10)
    axL.grid(True, alpha=0.3)
    axL.set_title("Mean RMSE vs Temporal Baseline (RAW, LS 60%)", fontsize=10)
    axL.legend(loc="upper right", fontsize=9, frameon=True)

    # --- Right: RMSE vs |B⊥| (km) ---
    xk = (m["bperp_abs_m"].to_numpy(float) / 1000.0)
    y  = m["mean_rmse_cm"].to_numpy(float)
    axR.scatter(xk, y, s=22, alpha=0.8, label="Pairs")
    if len(m) >= 2:
        m2, b2 = np.polyfit(xk, y, 1)
        x_line = np.linspace(np.nanmin(xk), np.nanmax(xk), 200); y_line = m2*x_line + b2
        axR.plot(x_line, y_line, linewidth=2.0, color="#424141", label="Linear fit")
        y_hat = m2*xk + b2; ss_res = np.sum((y-y_hat)**2); ss_tot = np.sum((y-np.mean(y))**2)
        r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
        axR.text(0.02, 0.98, f"Fit: y = {m2:.3f}·x + {b2:.3f}\n$R^2$ = {r2:.3f}",
                 transform=axR.transAxes, ha="left", va="top", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"))
    axR.set_xlabel("Perpendicular baseline |B⊥| (km)", fontsize=10)
    axR.set_ylabel("Mean RMSE (cm) across DEMs", fontsize=10)
    axR.grid(True, alpha=0.3)
    axR.set_title("Mean RMSE vs Perpendicular Baseline (RAW, LS 60%)", fontsize=10)
    axR.legend(loc="upper right", fontsize=9, frameon=True)

    out = root / "results" / "scatter_mean_rmse_vs_temporal_and_bperp_RAW.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"📈 Temporal & perpendicular baseline scatter written: {out}")


# ----------------------------------- CLI -------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Corrections 2×3 maps (with tight column control) and DEM plots (TROPO)."
    )
    ap.add_argument("--areas-root", type=str, default=str(AREAS_ROOT_DEFAULT),
                    help="Root folder containing per-area subfolders")
    ap.add_argument("--area", type=str,
                    help="Only process this AREA (subfolder name under --areas-root).")
    ap.add_argument("--water-areas", type=str, default=DEF_WATER_AREAS,
                    help="Path to water_areas.geojson (clip Soil & Satellite to this area's water polygon)")
    ap.add_argument("--sat-provider", type=str, default=DEF_SAT_PROVIDER,
                    help="contextily provider string (ignored if --sat-url is set)")
    ap.add_argument("--sat-url", type=str, default=DEF_SAT_URL,
                    help="Custom XYZ for satellite (e.g. 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}')")
    ap.add_argument("--soil-tif", type=str, default=DEF_SOIL_TIF,
                    help="Soil GeoTIFF to convert once to PNG and display (PNG saved next to the TIF).")
    ap.add_argument("--perp-baselines", type=str, default=str(PERP_BASELINES_CSV_DEFAULT),
                help="Path to perpendicular_baselines.csv (from help_perpendicular_baseline.py)")
    args = ap.parse_args()

    root = Path(args.areas_root)
    targets = [root / args.area] if args.area else sorted([d for d in root.iterdir() if d.is_dir()])

    # Load metrics (for DEMs + scatter)
    area_df: Dict[str, pd.DataFrame] = {}
    for area_dir in targets:
        df5 = _read_area_metrics(area_dir)
        if df5 is None or df5.empty:
            print(f"⏭️  No 5_* metrics for area: {area_dir.name}")
        else:
            area_df[area_dir.name] = df5

    if not area_df:
        print("⏭️  Nothing to visualize."); return

    # Corrections — per pair maps
    soil_tif_path = Path(args.soil_tif) if args.soil_tif else None
    for area_name, df5 in area_df.items():
        area_dir = root / area_name
        pair_tags = _collect_pair_tags_from_maps(area_dir)
        if not pair_tags and "pair_tag" in df5.columns:
            pair_tags = sorted(set(df5["pair_tag"].dropna().astype(str).tolist()))
        for pair_tag in pair_tags:
            try:
                plot_corrections_sixpack(area_dir, pair_tag,
                                         water_path=args.water_areas,
                                         sat_provider=args.sat_provider,
                                         sat_url=args.sat_url,
                                         soil_tif=soil_tif_path)
            except Exception as e:
                print(f"⚠️  Corrections maps failed {area_name}:{pair_tag}: {e}")

    # DEM boxplots — per area
    for area_name, df5 in area_df.items():
        try:
            _plot_area_boxplots(root / area_name, df5)
        except Exception as e:
            print(f"⚠️  DEM boxplots failed {area_name}: {e}")

    # ALL-areas combined + scatter
    try:
        _plot_all_areas_equalwidth(root, area_df)
    except Exception as e:
        print(f"⚠️  ALL-areas (equal width) failed: {e}")
    try:
        _plot_all_areas_varwidth(root, area_df)
    except Exception as e:
        print(f"⚠️  ALL-areas (varwidth) failed: {e}")
    try:
        _plot_scatter_rmse_vs_temporal_and_bperp(root, area_df, Path(args.perp_baselines))
    except Exception as e:
        print(f"⚠️  Temporal & B⊥ scatter failed: {e}")

if __name__ == "__main__":
    main()
