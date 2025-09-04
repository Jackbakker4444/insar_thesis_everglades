#!/usr/bin/env python3
"""
Area-wide accuracy vs. gauge-density assessment (with maps & time-series boxplots)
==================================================================================

What this script does
---------------------
For one AREA (or all areas under a root), this script:

1) Discovers all per-area interferograms:
     <areas_root>/<AREA>/interferograms/<AREA>_vertical_cm_<REF>_<SEC>_<DEM>_<CORR>.tif
   where DEM ∈ {SRTM, 3DEP} and CORR ∈ {RAW, TROPO, IONO, TROPO_IONO}.

2) For every available (PAIR, DEM, CORR) it runs a **replicate sweep** of accuracy vs.
   calibration gauge density using:
     • least_squares  (calibrate InSAR→Δh_vis via y = a·x + b; if n_cal ≤ 2 → force a = −1)
     • idw_dhvis      (interpolate gauges’ Δh_vis to validation gauges using IDW)
   The density uses the raster valid-data footprint ONLY:
     density = (valid area in km²) / (# calibration gauges).

3) It writes **a fresh accuracy_metrics.csv** (overwrites on every run):
     <areas_root>/<AREA>/results/accuracy_metrics.csv
   (One file per AREA processed in the run.)

4) For EACH (PAIR, DEM) it produces a **density figure** with exactly **three curves**:
     • LS • RAW            (solid line + translucent band for 95% range)
     • LS • TROPO_IONO     (solid line + translucent band)
     • IDW • TROPO_IONO    (dashed line + **hatched //** translucent band)
   The top x-axis shows number of calibration gauges with **LEFT = many, RIGHT = few**.
   Below the graph, two maps are shown side-by-side:
     • Interferogram (vertical cm; colormap inverted so **dark blue = larger**)
       with a star at the **“center” gauge** (the one used when n_cal=1).
     • IDW map of Δh_vis using a **60% spread-out calibration subset** of gauges,
       masked to the AREA polygon (if available), same inverted colormap, with
       the calibration gauge points drawn.

5) It builds **time-series figures** at a target density (default 500 km²/gauge)
   using **IDW on TROPO_IONO only**:
     • Per-DEM figure: a **boxplot per pair** (median, IQR, whiskers + outliers).
     • Combined figure: paired **SRTM + 3DEP boxplots per pair**,
       visually “stuck together”, whose combined width spans the pair dates.
       Color + hatch make which box belongs to which DEM unmistakable.

Inputs & expectations
---------------------
• Per-area gauge CSV at:
    <areas_root>/<AREA>/water_gauges/eden_gauges.csv
  must contain columns: StationID, Lat, Lon, and wide date columns ‘YYYY-MM-DD’
  for both REF and SEC dates in the pair.

• Per-area interferograms GeoTIFFs named as above, with CRS = EPSG:4326,
  single-band float cm, nodata as NaN or a numeric value.

• OPTIONAL area polygons at:
    /home/bakke326l/InSAR/main/data/vector/water_areas.geojson
  with a string column ‘area’ matching <AREA> folder names.
  (If missing, IDW maps will not be masked, but still render.)

How to run
----------
# Process ALL areas under the default root
python 5_accuracy_assessment_density.py

# Only one area
python 5_accuracy_assessment_density.py --area ENP

# Tweak replicates, random seed, IDW power, target density
python 5_accuracy_assessment_density.py --reps 50 --seed 42 --idw-power 2.0 --target-density 500

Outputs
-------
Per AREA:
  - results/accuracy_metrics.csv                              # rebuilt each run
  - results/acc_den_<PAIR>_<DEM>_LS2_IDW1.png                 # one file per (PAIR, DEM)
  - results/acc_period_<AREA>_<DEM>_<D>.png                   # D = target density tag
  - results/acc_period_<AREA>_COMBINED_<D>.png
"""

from __future__ import annotations
from pathlib import Path
import re, os, logging, argparse
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine, from_bounds as transform_from_bounds
from rasterio.features import shapes, rasterize
from shapely.geometry import mapping
from shapely.ops import unary_union
from pyproj import Geod
import matplotlib as mpl
mpl.set_loglevel("warning")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
import matplotlib.dates as mdates

# -------------------------- CONFIG DEFAULTS -----------------------------------
AREAS_ROOT = Path("/mnt/DATA2/bakke326l/processing/areas")
DEMS       = ["SRTM", "3DEP"]
CORRS_DISCOVER = ["RAW", "TROPO", "IONO", "TROPO_IONO"]  # discovered; plotting filters to RAW + TROPO_IONO
ID_COL, LAT_COL, LON_COL = "StationID", "Lat", "Lon"

REPS_DEFAULT          = 50     # replicates per raster
SEED_DEFAULT          = 42     # RNG seed
IDW_POWER_DEFAULT     = 2.0    # IDW power
TARGET_DENSITY_DEFAULT= 500.0  # km²/gauge for time-series
IDW_GRID_NX, IDW_GRID_NY = 400, 400  # resolution for the IDW preview map

# OPTIONAL polygons (used to mask IDW map to area)
BASE_DIR = Path("/home/bakke326l/InSAR/main")
WATER_AREAS_GEOJSON = BASE_DIR / "data/vector/water_areas.geojson"

# Matplotlib colormap: inverted viridis so dark blue = larger (max)
CMAP_INV = "viridis_r"

# Silence GDAL/Rasterio spam
os.environ.setdefault("CPL_DEBUG", "NO")
for _n in ("rasterio", "rasterio._io", "rasterio.env", "rasterio._base", "matplotlib.font_manager"):
    logging.getLogger(_n).setLevel(logging.ERROR)

GEOD = Geod(ellps="WGS84")

# ----------------------------- SMALL HELPERS ----------------------------------
def _pair_dates_from_tag(pair_tag: str) -> Tuple[str, str]:
    """'YYYYMMDD_YYYYMMDD' → ('YYYY-MM-DD','YYYY-MM-DD')"""
    if not re.fullmatch(r"\d{8}_\d{8}", pair_tag):
        raise ValueError(f"PAIR tag must be YYYYMMDD_YYYYMMDD, got: {pair_tag}")
    a, b = pair_tag.split("_")
    return f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"

def _find_all_pairs(area_dir: Path, area_name: str) -> List[str]:
    """Find unique pair tags available for this area by scanning /interferograms."""
    patt = re.compile(
        rf"^{re.escape(area_name)}_vertical_cm_(\d{{8}}_\d{{8}})_(SRTM|3DEP)_(RAW|TROPO|IONO|TROPO_IONO)\.tif$",
        re.I,
    )
    folder = area_dir / "interferograms"
    if not folder.exists():
        return []
    tags = set()
    for p in folder.glob("*.tif"):
        m = patt.match(p.name)
        if m:
            tags.add(m.group(1))
    return sorted(tags)

def _find_raster(area_dir: Path, area_name: str, pair_tag: str, dem: str, corr: str) -> Optional[Path]:
    """Return path to one per-area interferogram raster if it exists."""
    cand = area_dir / "interferograms" / f"{area_name}_vertical_cm_{pair_tag}_{dem.upper()}_{corr.upper()}.tif"
    return cand if cand.exists() else None

def load_gauges_wide(csv_path: Path) -> pd.DataFrame:
    """Minimal checks on the per-area gauge wide table."""
    df = pd.read_csv(csv_path)
    for c in (ID_COL, LAT_COL, LON_COL):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    return df

def rowcol_from_xy(transform: Affine, x: float, y: float) -> Tuple[float, float]:
    col, row = ~transform * (x, y)
    return float(row), float(col)

def inside_image(h: int, w: int, row: float, col: float) -> bool:
    return (row >= 0) and (col >= 0) and (row < h) and (col < w)

def read_mean_3x3(ds: rasterio.io.DatasetReader, row: int, col: int) -> Optional[float]:
    """Mean over a 3×3 window; returns None if all-NaN / all-nodata."""
    r0 = max(0, row - 1); r1 = min(ds.height - 1, row + 1)
    c0 = max(0, col - 1); c1 = min(ds.width  - 1, col + 1)
    arr = ds.read(1, window=Window.from_slices((r0, r1 + 1), (c0, c1 + 1))).astype("float32")
    if ds.nodata is not None and not np.isnan(ds.nodata):
        arr[arr == ds.nodata] = np.nan
    if not np.isfinite(arr).any(): return None
    return float(np.nanmean(arr))

def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Ordinary least squares y ≈ a·x + b (two-parameter fit)."""
    A = np.c_[x, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])

def _safe_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Robust correlation that avoids NaNs for zero-variance cases."""
    if len(y_true) < 2: return float("nan")
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    vy = np.sum(yt*yt); vp = np.sum(yp*yp)
    if vy <= 1e-12 or vp <= 1e-12:
        return float("nan")
    return float(np.sum(yt*yp) / np.sqrt(vy*vp))

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE/MAE/Bias/Correlation for predictions vs truth."""
    err = y_pred - y_true
    return {
        "rmse_cm": float(np.sqrt(np.mean(err**2))),
        "mae_cm":  float(np.mean(np.abs(err))),
        "bias_cm": float(np.mean(err)),
        "r":       _safe_corrcoef(y_true, y_pred),
    }

def visible_surface_delta(ref_cm: np.ndarray, sec_cm: np.ndarray) -> np.ndarray:
    """Δh_vis = max(sec, 0) − max(ref, 0) in centimeters."""
    return np.maximum(sec_cm.astype(float), 0.0) - np.maximum(ref_cm.astype(float), 0.0)

# ----------------------------- IDW PREDICTION ---------------------------------
def _idw_predict_points(px, py, pz, qx, qy, power: float = 2.0) -> np.ndarray:
    """
    Simple geographic IDW in lon/lat (weights ~ 1 / d^power), with a cosine adjustment
    on Δlon to account for latitude. If a query lands exactly on a known point, we
    snap to that value.
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    pz = np.asarray(pz, dtype=np.float64)
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)

    # Cosine correction to approximate meters in lon dimension
    cx = np.cos(np.deg2rad(np.nanmean(py) if py.size else 0.0))
    dx = (qx[:, None] - px[None, :]) * cx
    dy = (qy[:, None] - py[None, :])
    d2 = dx*dx + dy*dy
    w  = 1.0 / np.maximum(d2, 1e-18) ** (power/2.0)
    pred = (w @ pz) / np.sum(w, axis=1)

    # snap exact matches
    imin = np.argmin(d2, axis=1)
    hits = d2[np.arange(d2.shape[0]), imin] < 1e-18
    if np.any(hits):
        pred[hits] = pz[imin[hits]]
    return pred.astype("float32")

# ------------------------- GAUGE GEOMETRY / SAMPLING --------------------------
def _haversine_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Great-circle distance matrix (meters) for lon/lat arrays."""
    lon = np.deg2rad(lon.astype("float64")); lat = np.deg2rad(lat.astype("float64"))
    sin_lat = np.sin(lat)[:, None]; cos_lat = np.cos(lat)[:, None]
    sin_lat_T = np.sin(lat)[None, :]; cos_lat_T = np.cos(lat)[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.clip(sin_lat * sin_lat_T + cos_lat * cos_lat_T * np.cos(dlon), -1.0, 1.0)
    return 6371000.0 * np.arccos(a)

def _spread_selection(lon: np.ndarray, lat: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Farthest-point sampling for k well-spread indices."""
    n = len(lon)
    if k >= n: return np.arange(n, dtype=int)
    D = _haversine_matrix(lon, lat)
    cur = [int(rng.integers(0, n))]
    remaining = set(range(n)) - set(cur)
    min_d = D[:, cur].min(axis=1)
    while len(cur) < k:
        cand = max(remaining, key=lambda i: float(min_d[i]))
        cur.append(cand); remaining.remove(cand)
        min_d = np.minimum(min_d, D[:, cand])
    return np.array(cur, dtype=int)

def _crowded_candidates(lon: np.ndarray, lat: np.ndarray, idx: np.ndarray, keep_global: int, top_n: int = 4) -> np.ndarray:
    """Indices (within idx) of the most crowded gauges, excluding 'keep_global'."""
    if len(idx) <= 1: return idx
    lon_s, lat_s = lon[idx], lat[idx]
    D = _haversine_matrix(lon_s, lat_s)
    np.fill_diagonal(D, np.inf)
    nnd = np.min(D, axis=1)
    order = np.argsort(nnd)  # smallest NN distance = most crowded
    order = np.array([o for o in order if idx[o] != keep_global], dtype=int)
    if order.size == 0: return np.array([], dtype=int)
    return order[:min(top_n, order.size)]

def _geod_area_of_geojson(geom) -> float:
    """Geodesic area (km²) of a GeoJSON Polygon/MultiPolygon in EPSG:4326."""
    def poly_area(coords):
        area_km2 = 0.0
        if not coords: return 0.0
        lon, lat = zip(*coords[0]);  a_ext, _ = GEOD.polygon_area_perimeter(lon, lat)
        area_km2 += abs(a_ext) / 1e6
        for ring in coords[1:]:
            lon, lat = zip(*ring); a_hole, _ = GEOD.polygon_area_perimeter(lon, lat)
            area_km2 -= abs(a_hole) / 1e6
        return max(area_km2, 0.0)
    typ = geom.get("type")
    if typ == "Polygon":      return poly_area(geom["coordinates"])
    if typ == "MultiPolygon": return sum(poly_area(coords) for coords in geom["coordinates"])
    return 0.0

def _valid_raster_area_km2(ds: rasterio.io.DatasetReader) -> float:
    """Sum area for valid-data polygons from dataset_mask()==255 (expects EPSG:4326)."""
    if ds.crs is None or ds.crs.to_epsg() != 4326:
        raise RuntimeError("Expected EPSG:4326 raster.")
    mask = (ds.dataset_mask() == 255).astype(np.uint8)
    area = 0.0
    for geom, val in shapes(mask, transform=ds.transform):
        if val == 1:
            area += _geod_area_of_geojson(geom)
    return float(area)

# ---------------------------- CORE EVALUATIONS --------------------------------
def _eval_ls_and_idw(pts: pd.DataFrame, cal_idx: np.ndarray, val_idx: np.ndarray, idw_power: float) -> Dict[str, Dict[str, float]]:
    """
    Evaluate both LS (calibrate InSAR→Δh_vis) and IDW (interpolate Δh_vis) on validation gauges.
    Returns a dict of metrics for both methods.
    """
    # Prepare arrays
    x_cal = pts["insar_cm"].values[cal_idx].astype(float)
    y_cal = pts["dh_cm"].values[cal_idx].astype(float)
    x_val = pts["insar_cm"].values[val_idx].astype(float)
    y_val = pts["dh_cm"].values[val_idx].astype(float)

    # LS with a = −1 if small calibration set
    n_unique = len(np.unique(pts[ID_COL].values[cal_idx]))
    if len(cal_idx) <= 2 and n_unique >= 1:
        a = -1.0
        b = float(np.mean(y_cal + x_cal))  # best b for fixed a=-1
    elif n_unique >= 2:
        a, b = fit_affine(x_cal, y_cal)
    else:
        a, b = np.nan, np.nan

    # Predict & score LS
    if np.isfinite(a) and np.isfinite(b):
        y_pred_ls = a * x_val + b
        m_ls = compute_metrics(y_true=y_val, y_pred=y_pred_ls)
        m_ls.update({"a_gain": float(a), "b_offset_cm": float(b)})
    else:
        m_ls = {"rmse_cm": np.nan, "mae_cm": np.nan, "bias_cm": np.nan, "r": np.nan,
                "a_gain": np.nan, "b_offset_cm": np.nan}

    # Predict & score IDW (gauges → Δh_vis at validation gauges)
    px = pts[LON_COL].values[cal_idx].astype(float)
    py = pts[LAT_COL].values[cal_idx].astype(float)
    pz = pts["dh_cm"].values[cal_idx].astype(float)
    qx = pts[LON_COL].values[val_idx].astype(float)
    qy = pts[LAT_COL].values[val_idx].astype(float)
    y_pred_idw = _idw_predict_points(px, py, pz, qx, qy, power=idw_power)
    m_idw = compute_metrics(y_true=y_val, y_pred=y_pred_idw)

    return {"least_squares": m_ls, "idw_dhvis": m_idw}

def evaluate_one_raster(area_name: str,
                        gauge_csv: Path,
                        raster_tif: Path,
                        dem: str,
                        corr: str,
                        ref_iso: str,
                        sec_iso: str,
                        n_repl: int,
                        seed: int,
                        idw_power: float) -> pd.DataFrame:
    """
    Replicate sweep for a single raster. Returns a dataframe of per-step metrics.
    """
    # Load gauges and compute Δh_vis (visible-surface change) for this pair
    gauges = load_gauges_wide(gauge_csv)
    for c in (ref_iso, sec_iso):
        if c not in gauges.columns:
            raise ValueError(f"Gauge CSV missing date column: {c}")

    g = gauges[[ID_COL, LAT_COL, LON_COL, ref_iso, sec_iso]].copy()
    g.rename(columns={ref_iso: "ref_cm", sec_iso: "sec_cm"}, inplace=True)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)
    g.dropna(subset=["ref_cm", "sec_cm", LAT_COL, LON_COL], inplace=True)
    g["dh_cm"] = visible_surface_delta(g["ref_cm"].to_numpy(), g["sec_cm"].to_numpy())

    # Sample InSAR at gauge locations; keep only valid pixels
    with rasterio.open(raster_tif) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise RuntimeError(f"Expected raster in EPSG:4326: {raster_tif}")
        area_km2 = _valid_raster_area_km2(ds)
        if not np.isfinite(area_km2) or area_km2 <= 0:
            return pd.DataFrame()
        rows = []
        for _, r in g.iterrows():
            x, y = float(r[LON_COL]), float(r[LAT_COL])
            rowf, colf = rowcol_from_xy(ds.transform, x, y)
            if not inside_image(ds.height, ds.width, rowf, colf): 
                continue
            ins = read_mean_3x3(ds, int(round(rowf)), int(round(colf)))
            if ins is None or not np.isfinite(ins): 
                continue
            rows.append({ID_COL: r[ID_COL], LON_COL: x, LAT_COL: y,
                         "insar_cm": float(ins), "dh_cm": float(r["dh_cm"])})

    if not rows:
        return pd.DataFrame()

    pts = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["insar_cm", "dh_cm"])
    if pts[ID_COL].nunique() < 3 or len(pts) < 3:
        # Not enough gauges to do holdouts meaningfully
        return pd.DataFrame()

    # Choose "center" gauge (closest to centroid); excluded until n_cal=1
    lon_all = pts[LON_COL].to_numpy(dtype=float)
    lat_all = pts[LAT_COL].to_numpy(dtype=float)
    lon_c, lat_c = float(lon_all.mean()), float(lat_all.mean())
    _, _, d_center = GEOD.inv(np.full_like(lon_all, lon_c), np.full_like(lat_all, lat_c), lon_all, lat_all)
    center_idx_global = int(np.argmin(d_center))

    rng_master = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []
    N = len(pts)
    n_cal0 = max(1, int(round(0.60 * N)))  # ~60% calibration start

    for rep in range(1, n_repl + 1):
        rng = np.random.default_rng(rng_master.integers(0, 2**31-1))

        # All but center gauge are eligible for cal/val split
        all_idx = np.arange(N, dtype=int)
        available_idx = np.setdiff1d(all_idx, np.array([center_idx_global]), assume_unique=False)

        # Initial cal set: spread-out sampling
        n_cal0_eff = min(n_cal0, len(available_idx))
        cal_local = _spread_selection(lon_all[available_idx], lat_all[available_idx], n_cal0_eff, rng)
        cal_idx = available_idx[cal_local]
        val_idx = np.setdiff1d(available_idx, cal_idx, assume_unique=False)

        # If val ended up empty, move one from cal→val (most crowded)
        if len(val_idx) == 0 and len(cal_idx) >= 2:
            crowded = _crowded_candidates(lon_all, lat_all, cal_idx, keep_global=center_idx_global, top_n=4)
            move_pos = crowded[0] if crowded.size else 0
            val_idx = np.r_[val_idx, [cal_idx[move_pos]]]
            cal_idx = np.delete(cal_idx, move_pos)

        # March down to 2 cal gauges
        cur_idx = cal_idx.copy()
        while len(cur_idx) >= 2:
            mm = _eval_ls_and_idw(pts, cur_idx, val_idx, idw_power=idw_power)
            base = {
                "replicate": rep, "n_total": N, "n_val": int(len(val_idx)),
                "n_cal": int(len(cur_idx)), "area_km2": area_km2,
                "area_per_gauge_km2": area_km2 / float(len(cur_idx)),
                "pair_ref": ref_iso, "pair_sec": sec_iso,
                "area": area_name, "dem": dem, "corr": corr,
            }
            for method, metr in mm.items():
                records.append({**base, "method": method, **metr})

            crowded = _crowded_candidates(lon_all, lat_all, cur_idx, keep_global=center_idx_global, top_n=4)
            drop_pos = int(rng.choice(crowded)) if crowded.size else 0
            cur_idx = np.delete(cur_idx, drop_pos)

        # Final single-gauge: ONLY center gauge
        mm = _eval_ls_and_idw(pts, np.array([center_idx_global], dtype=int), val_idx, idw_power=idw_power)
        base = {
            "replicate": rep, "n_total": N, "n_val": int(len(val_idx)),
            "n_cal": 1, "area_km2": area_km2, "area_per_gauge_km2": area_km2 / 1.0,
            "pair_ref": ref_iso, "pair_sec": sec_iso,
            "area": area_name, "dem": dem, "corr": corr,
        }
        for method, metr in mm.items():
            records.append({**base, "method": method, **metr})

    return pd.DataFrame.from_records(records)

# --------------------------- PLOTTING UTILITIES --------------------------------
def _shade_band(ax, x, ylo, yhi, color, *, hatched=False, alpha=0.18, z=1, hatch_density=0.6):
    """
    Draw an uncertainty band. If 'hatched', draw a light transparent fill PLUS a // hatch.
    """
    face = to_rgba(color, alpha)
    if hatched:
        # thin edge so hatch is visible; small alpha to avoid overpowering lines
        coll = ax.fill_between(
            x, ylo, yhi,
            facecolor=face, edgecolor=to_rgba(color, 0.7),
            hatch='//', linewidth=0.5, zorder=z
        )
    else:
        coll = ax.fill_between(x, ylo, yhi, facecolor=face, edgecolor='none', zorder=z)
    return coll

def _plot_ifg_map(ax, tif_path, title="Interferogram vertical (cm)"):
    """
    Quick-look map for the interferogram with inverted colormap.
    Returns (left, right, bottom, top) map bounds.
    """
    with rasterio.open(tif_path) as ds:
        arr = ds.read(1).astype(float)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr[arr == ds.nodata] = np.nan
        m = np.ma.masked_invalid(arr)
        vals = m.compressed()
        vmin, vmax = (0.0, 1.0) if vals.size == 0 else np.nanpercentile(vals, [2, 98])
        im = ax.imshow(
            m, origin='upper',
            extent=(ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top),
            vmin=vmin, vmax=vmax, cmap=CMAP_INV
        )
        bounds = (float(ds.bounds.left), float(ds.bounds.right), float(ds.bounds.bottom), float(ds.bounds.top))
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("cm")
    return bounds

def _make_idw_grid(px, py, pz, bounds, nx=360, ny=360, power=2.0, mask_poly=None):
    """Compute an IDW grid over given bounds, optionally masking outside the area polygon."""
    x0, x1, y0, y1 = map(float, bounds)
    x1 = x1 if x1 > x0 else x0 + 1e-4
    y1 = y1 if y1 > y0 else y0 + 1e-4
    gx = np.linspace(x0, x1, nx, dtype=np.float64)
    gy = np.linspace(y0, y1, ny, dtype=np.float64)
    qx, qy = np.meshgrid(gx, gy)  # (ny, nx)
    pred = _idw_predict_points(
        np.asarray(px, dtype=float), np.asarray(py, dtype=float), np.asarray(pz, dtype=float),
        qx.reshape(-1), qy.reshape(-1), power=power
    ).reshape(ny, nx)

    if mask_poly is not None:
        transform = transform_from_bounds(x0, y0, x1, y1, nx, ny)
        mask = rasterize([(mapping(mask_poly), 1)], out_shape=(ny, nx),
                         transform=transform, fill=0, dtype="uint8")
        pred = np.where(mask == 1, pred, np.nan)
    return gx, gy, pred

def _plot_idw_map(
    ax,
    gauges_df: pd.DataFrame,
    raster_tif: Path,
    area_poly=None,
    power: float = IDW_POWER_DEFAULT,
    title: str = "Gauge IDW 60%/ gauges",
    vmin: float | None = None,
    vmax: float | None = None,
    cal_idx: np.ndarray | None = None,
):
    """
    Render the IDW map on the *raster grid* to match orientation/extent perfectly.

    Parameters
    ----------
    ax : matplotlib axis to draw on (right-hand panel).
    gauges_df : DataFrame with columns [LON_COL, LAT_COL, 'dh_cm'] (and others).
    raster_tif : Path to the same raster shown in the left panel.
    area_poly : optional shapely Polygon/MultiPolygon in EPSG:4326 to clip the IDW.
    power : IDW power.
    title : panel title.
    vmin, vmax : optional color scaling to match interferogram.
    cal_idx : optional integer index array for the 60% *calibration* gauges.
              If provided, we only use those points for the IDW & scatter.
    """
    # --- Pick gauge set: 60% calibration (preferred) or all gauges ---
    if cal_idx is not None and len(cal_idx) > 0:
        gg = gauges_df.iloc[cal_idx]
    else:
        gg = gauges_df

    # Calibration point arrays
    px = gg[LON_COL].to_numpy(dtype="float64")
    py = gg[LAT_COL].to_numpy(dtype="float64")
    pz = gg["dh_cm"].to_numpy(dtype="float64")

    # --- Open raster & create pixel-center lon/lat query points ---
    with rasterio.open(raster_tif) as ds:
        H, W = ds.height, ds.width

        # Pixel-center coords (lon/lat) for every pixel
        rr, cc = np.indices((H, W))
        xs, ys = rasterio.transform.xy(ds.transform, rr, cc, offset="center")
        qx = np.asarray(xs, dtype="float64").ravel()  # longitudes
        qy = np.asarray(ys, dtype="float64").ravel()  # latitudes

        # IDW at raster pixels -> reshape to image
        Q = _idw_predict_points(px, py, pz, qx, qy, power=power).reshape(H, W)

        # Mask to raster valid footprint (dataset_mask == 255)
        valid_mask = (ds.dataset_mask() == 255)

        # Optional: further clip to area polygon
        if area_poly is not None:
            try:
                # Reproject area polygon to raster CRS and rasterize
                area_series = gpd.GeoSeries([area_poly], crs=4326).to_crs(ds.crs)
                area_mask = rasterio.features.rasterize(
                    [(area_series.iloc[0].__geo_interface__, 1)],
                    out_shape=(H, W),
                    transform=ds.transform,
                    fill=0,
                    dtype="uint8",
                ).astype(bool)
                valid_mask = valid_mask & area_mask
            except Exception:
                # If anything goes wrong, fall back to raster mask only
                pass

        Q = np.where(valid_mask, Q, np.nan)

        # Extent for imshow so it lines up with the interferogram
        extent = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top)

    # --- Plot IDW on the right axis (inverted colormap as requested) ---
    im = ax.imshow(Q, extent=extent, origin="upper", cmap=CMAP_INV, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Overlay calibration gauge locations (ring markers)
    ax.scatter(px, py, s=18, facecolors="none", edgecolors="k", linewidths=0.8, zorder=3, label="Calibration gauges")

    # Colorbar
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cm")


# ------------------ DENSITY PLOT (3 lines) + MAPS (per PAIR/DEM) --------------
def plot_density_three_lines_with_maps(
    df_pair: pd.DataFrame,
    area_name: str,
    pair_tag: str,
    dem: str,
    out_png: Path,
    gauge_csv: Path,
    ref_iso: str,
    sec_iso: str,
    raster_for_maps: Path,
    area_poly: Optional[object],
    idw_power: float
):
    """
    Plot (PAIR, DEM) accuracy vs density:
      • LS RAW (solid + translucent band)
      • LS TROPO_IONO (solid + translucent band)
      • IDW TROPO_IONO (dashed + **hatched** band)
    Below: IFG map (with center gauge star) + IDW map from a 60% spread-out calibration subset.
    """
    sub = df_pair[df_pair["dem"] == dem].copy()
    if sub.empty:
        print(f"⚠️  No data to plot for DEM={dem}, {pair_tag}")
        return

    # Aggregate to median + 95% range per (corr, method, n_cal)
    grp = (sub.groupby(["corr", "method", "n_cal"], as_index=False)
              .agg(med_rmse=("rmse_cm", "median"),
                   p2_5=("rmse_cm", lambda x: np.nanpercentile(x, 2.5)),
                   p97_5=("rmse_cm", lambda x: np.nanpercentile(x, 97.5)),
                   area_km2=("area_km2", "median")))
    grp["density"] = grp["area_km2"] / grp["n_cal"].astype(float)

    # Keep only needed curves
    want = pd.concat([
        grp[(grp["method"] == "least_squares") & (grp["corr"].isin(["RAW", "TROPO_IONO"]))],
        grp[(grp["method"] == "idw_dhvis") & (grp["corr"] == "TROPO_IONO")]
    ], ignore_index=True)

    # Figure layout: top plot, bottom 2 maps
    fig = plt.figure(figsize=(12.0, 9.0), dpi=140, constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[3.2, 2.8])

    ax = fig.add_subplot(gs[0, :])

    # Colors
    c_ls_raw        = "#1b9e77"
    c_ls_ti         = "#e7298a"  # TI = TROPO_IONO
    c_idw_ti        = "#1f78b4"

    # Plot LS RAW & LS TROPO_IONO (solid + translucent)
    for corr, col in [("RAW", c_ls_raw), ("TROPO_IONO", c_ls_ti)]:
        gg = want[(want["method"] == "least_squares") & (want["corr"] == corr)].copy()
        if gg.empty: continue
        gg.sort_values("density", inplace=True)
        ax.plot(gg["density"], gg["med_rmse"], "-", color=col, lw=1.9, label=f"LS • {corr}")
        _shade_band(ax, gg["density"], gg["p2_5"], gg["p97_5"], color=col, hatched=False, alpha=0.22, z=1)

    # Plot IDW TROPO_IONO (dashed + hatched)
    gg = want[(want["method"] == "idw_dhvis") & (want["corr"] == "TROPO_IONO")].copy()
    if not gg.empty:
        gg.sort_values("density", inplace=True)
        ax.plot(gg["density"], gg["med_rmse"], "--", color=c_idw_ti, lw=1.9, label="IDW • TROPO_IONO")
        _shade_band(ax, gg["density"], gg["p2_5"], gg["p97_5"], color=c_idw_ti, hatched=True, alpha=0.14, z=1)

    # y-limits (robust)
    ymin = 0.0
    ymax = float(np.nanmax([want["p97_5"].max(), want["med_rmse"].max() * 1.1]))
    ax.set_ylim(ymin, ymax if np.isfinite(ymax) and ymax > 0 else 1.0)

    # x-axis: log density
    xmin = float(want["density"].min()); xmax = float(want["density"].max())
    ax.set_xscale("log"); ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1,10)*0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlabel("Gauge density (km² per gauge) — lower is denser  [log scale]")
    ax.set_ylabel("RMSE (cm) — median ± 95%")
    ax.set_title(f"{area_name} • {pair_tag} • DEM={dem}")
    ax.legend(ncols=3, fontsize=9)

    # top axis: number of gauges (LEFT = many, RIGHT = few)
    ax_top = ax.twiny(); ax_top.set_xscale("log"); ax_top.set_xlim(ax.get_xlim())
    area_ref = float(want["area_km2"].median())
    n_tick = sorted(want["n_cal"].unique().tolist(), reverse=True)  # left = many
    dens_ticks = [area_ref / float(n) for n in n_tick]
    filt = [i for i, d in enumerate(dens_ticks) if xmin <= d <= xmax]
    ax_top.set_xticks([dens_ticks[i] for i in filt]); ax_top.set_xticklabels([str(int(n_tick[i])) for i in filt])
    ax_top.set_xlabel("Number of calibration gauges")

    # --- bottom row maps ---
    # Interferogram map with center gauge star
    ax_ifg = fig.add_subplot(gs[1, 0])
    bounds = _plot_ifg_map(ax_ifg, raster_for_maps, title="Interferogram vertical (cm)")
    # Find & plot center gauge (same logic used in evaluate_one_raster)
    try:
        # Load gauges & sample to get the same usable set as in evaluation
        gauges = load_gauges_wide(gauge_csv)
        g = gauges[[ID_COL, LAT_COL, LON_COL, ref_iso, sec_iso]].copy()
        g.rename(columns={ref_iso: "ref_cm", sec_iso: "sec_cm"}, inplace=True)
        g.replace([np.inf, -np.inf], np.nan, inplace=True)
        g.dropna(subset=["ref_cm", "sec_cm", LAT_COL, LON_COL], inplace=True)
        g["dh_cm"] = visible_surface_delta(g["ref_cm"].to_numpy(), g["sec_cm"].to_numpy())

        with rasterio.open(raster_for_maps) as ds:
            rows = []
            for _, r in g.iterrows():
                x, y = float(r[LON_COL]), float(r[LAT_COL])
                rowf, colf = rowcol_from_xy(ds.transform, x, y)
                if not inside_image(ds.height, ds.width, rowf, colf): 
                    continue
                ins = read_mean_3x3(ds, int(round(rowf)), int(round(colf)))
                if ins is None or not np.isfinite(ins): 
                    continue
                rows.append({ID_COL: r[ID_COL], LON_COL: x, LAT_COL: y, "insar_cm": float(ins)})
        if rows:
            pts = pd.DataFrame(rows)
            lon_all = pts[LON_COL].to_numpy(dtype=float)
            lat_all = pts[LAT_COL].to_numpy(dtype=float)
            lon_c, lat_c = float(lon_all.mean()), float(lat_all.mean())
            _, _, d_center = GEOD.inv(np.full_like(lon_all, lon_c), np.full_like(lat_all, lat_c), lon_all, lat_all)
            i_center = int(np.argmin(d_center))
            ax_ifg.scatter(lon_all[i_center], lat_all[i_center], marker="*", s=120,
                           facecolor="gold", edgecolor="k", linewidths=0.8, zorder=5,
                           label="Center gauge (n_cal=1)")
            ax_ifg.legend(loc="lower right", fontsize=8)
    except Exception:
        pass

    # IDW map from 60% calibration gauges (masked to area)
    ax_idw = fig.add_subplot(gs[1, 1])
    try:
        gauges = load_gauges_wide(gauge_csv)
        if all(c in gauges.columns for c in (ref_iso, sec_iso)):
            g = gauges[[ID_COL, LAT_COL, LON_COL, ref_iso, sec_iso]].copy()
            g.rename(columns={ref_iso: "ref_cm", sec_iso: "sec_cm"}, inplace=True)
            g.replace([np.inf, -np.inf], np.nan, inplace=True)
            g.dropna(subset=["ref_cm", "sec_cm", LAT_COL, LON_COL], inplace=True)
            # optional mask by area polygon
            if area_poly is not None:
                pts_geo = gpd.GeoSeries(gpd.points_from_xy(g[LON_COL].astype(float), g[LAT_COL].astype(float)), crs=4326)
                inside = pts_geo.within(area_poly)
                g = g.loc[inside.values].reset_index(drop=True)
            # Δh_vis + 60% spread selection
            g["dh_cm"] = visible_surface_delta(g["ref_cm"].to_numpy(), g["sec_cm"].to_numpy())
            if len(g) >= 2:
                lon_all = g[LON_COL].to_numpy(dtype=float)
                lat_all = g[LAT_COL].to_numpy(dtype=float)
                rng = np.random.default_rng(SEED_DEFAULT)
                k = max(1, int(round(0.60 * len(g))))
                sel_idx = _spread_selection(lon_all, lat_all, k, rng=rng)
                g_cal = g.iloc[sel_idx].copy()
            else:
                g_cal = g.copy()
            # IDW bounds: area polygon if available, else IFG map bounds
            if area_poly is not None:
                minx, miny, maxx, maxy = area_poly.bounds
                bounds_idw = (minx, maxx, miny, maxy)
            else:
                bounds_idw = bounds
            _plot_idw_map(ax_idw, g_cal, area_poly, bounds=bounds_idw, power=idw_power,
                          title="Gauge IDW (Δh_vis, 60% calibration)")
        else:
            ax_idw.text(0.5, 0.5, "Gauge CSV missing pair-date columns", ha="center", va="center")
    except Exception as e:
        ax_idw.text(0.5, 0.5, f"IDW map failed:\n{e}", ha="center", va="center")

    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"📈 Density figure written: {out_png}")

# --------------------------- TIME-SERIES BOX PLOTS -----------------------------
def _pick_rows_for_target_density(df: pd.DataFrame, target_km2: float) -> pd.DataFrame:
    """
    For a df subset holding a single (AREA, DEM, method='idw_dhvis', corr='TROPO_IONO'),
    choose, for each pair, the n_cal whose density is closest to the target.
    Return the matching rows (all replicates) so we can boxplot their distribution.
    """
    df = df.copy()
    df["density"] = df["area_km2"] / df["n_cal"].astype(float)

    picks = []
    for (pref, psec), gsub in df.groupby(["pair_ref","pair_sec"], as_index=False):
        gagg = (gsub.groupby("n_cal", as_index=False)
                    .agg(density=("density","median")))
        if gagg.empty:
            continue
        gagg["d_abs"] = (gagg["density"] - target_km2).abs()
        choose_n = int(gagg.loc[gagg["d_abs"].idxmin(), "n_cal"])
        picks.append(gsub[gsub["n_cal"] == choose_n])

    if not picks:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(picks, ignore_index=True)

def plot_time_series_per_dem_boxplots(df_all: pd.DataFrame, area_name: str, dem: str, out_png: Path, target_km2: float):
    """
    Per-DEM time-series: one **boxplot per pair** for IDW on TROPO_IONO at the target density.
    Box shows median, IQR; whiskers 1.5*IQR; outliers shown.
    """
    sub = df_all[(df_all["dem"] == dem) &
                 (df_all["method"] == "idw_dhvis") &
                 (df_all["corr"].str.upper() == "TROPO_IONO")].copy()
    if sub.empty:
        print(f"⚠️  No TROPO_IONO IDW data for DEM={dem} — skipping time-series.")
        return

    picked = _pick_rows_for_target_density(sub, target_km2)
    if picked.empty:
        print(f"⚠️  No rows near target density for DEM={dem}."); return

    # Build boxplot inputs per pair (list of arrays), and positions/widths by pair span
    to_dt = lambda s: pd.to_datetime(s, format="%Y-%m-%d")
    meta = (picked[["pair_ref","pair_sec"]].drop_duplicates()
                .assign(t_ref=lambda d: to_dt(d["pair_ref"]),
                        t_sec=lambda d: to_dt(d["pair_sec"]),
                        t_mid=lambda d: d["t_ref"] + (d["t_sec"] - d["t_ref"])/2,
                        span_days=lambda d: (d["t_sec"] - d["t_ref"]).dt.days.clip(lower=1).astype(float))
                .sort_values("t_mid"))
    data = []
    positions = []
    widths = []
    for _, row in meta.iterrows():
        vals = picked[(picked["pair_ref"]==row["pair_ref"]) & (picked["pair_sec"]==row["pair_sec"])]["rmse_cm"].to_numpy()
        if vals.size == 0: 
            continue
        data.append(vals)
        positions.append(mdates.date2num(row["t_mid"]))
        widths.append(float(row["span_days"]))  # total width equals pair span

    if not data:
        print(f"⚠️  No data per pair for DEM={dem}."); return

    fig, ax = plt.subplots(figsize=(11.2, 5.2), dpi=140, constrained_layout=True)

    # Draw boxplots; Matplotlib scales width in x-units (days) on a date axis
    bp = ax.boxplot(
        data, positions=positions, widths=widths,
        whis=1.5, patch_artist=True, showfliers=True
    )
    # Style
    for patch in bp["boxes"]:
        patch.set(facecolor=to_rgba("#377eb8", 0.28), edgecolor="#377eb8", linewidth=1.2)
    for whisk in bp["whiskers"]:
        whisk.set(color="#377eb8", linewidth=1.2)
    for cap in bp["caps"]:
        cap.set(color="#377eb8", linewidth=1.2)
    for median in bp["medians"]:
        median.set(color="#0c3c78", linewidth=2.0)
    for flier in bp["fliers"]:
        flier.set(marker="o", markersize=3.5, markerfacecolor="#377eb8", markeredgecolor="white", alpha=0.8)

    ax.set_ylabel("RMSE (cm) — box: IQR; whiskers: 1.5×IQR; dots: outliers")
    ax.set_xlabel("Time (each box width spans its pair dates)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_title(f"{area_name} • DEM={dem} • IDW(TROPO_IONO) at ≈ {target_km2:g} km²/g")
    fig.autofmt_xdate()

    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"📈 Period (per-DEM) boxplots written: {out_png}")

def plot_time_series_combined_boxplots(df_all: pd.DataFrame, area_name: str, out_png: Path, target_km2: float):
    """
    Combined time-series: **paired boxplots** per pair (SRTM + 3DEP),
    visually stuck together; combined width equals the pair span.
    Color + hatch distinguish DEMs clearly.
    """
    sub = df_all[(df_all["method"] == "idw_dhvis") &
                 (df_all["corr"].str.upper() == "TROPO_IONO")].copy()
    if sub.empty:
        print("⚠️  No TROPO_IONO IDW data — skipping combined time-series.")
        return

    # Pick rows near target density per (pair, DEM)
    picks = []
    for dem in DEMS:
        df_dem = sub[sub["dem"] == dem]
        if df_dem.empty: 
            continue
        picks.append(_pick_rows_for_target_density(df_dem, target_km2))
    if not picks:
        print("⚠️  No rows near target density for combined plot."); return
    picked = pd.concat(picks, ignore_index=True)

    # Build meta per pair (span/center)
    to_dt = lambda s: pd.to_datetime(s, format="%Y-%m-%d")
    meta = (picked[["pair_ref","pair_sec"]].drop_duplicates()
                .assign(t_ref=lambda d: to_dt(d["pair_ref"]),
                        t_sec=lambda d: to_dt(d["pair_sec"]),
                        t_mid=lambda d: d["t_ref"] + (d["t_sec"] - d["t_ref"])/2,
                        span_days=lambda d: (d["t_sec"] - d["t_ref"]).dt.days.clip(lower=1).astype(float))
                .sort_values("t_mid"))
    if meta.empty:
        print("⚠️  No pair metadata for combined plot."); return

    # Styling
    colors = {"SRTM": "#377eb8", "3DEP": "#4daf4a"}
    hatches = {"SRTM": "///", "3DEP": "\\\\\\"}

    fig, ax = plt.subplots(figsize=(12.2, 5.6), dpi=140, constrained_layout=True)

    # For each pair, place two boxes with small offsets; widths add to ~pair span
    for _, row in meta.iterrows():
        x_mid = mdates.date2num(row["t_mid"])
        span = float(row["span_days"])
        # offsets so that together they cover the span, with a tiny gap
        group_gap = span * 0.05
        each_width = (span - group_gap) / 2.0
        x_srtm = x_mid - (group_gap/2 + each_width/2)
        x_3dep = x_mid + (group_gap/2 + each_width/2)

        for dem, xpos in (("SRTM", x_srtm), ("3DEP", x_3dep)):
            g = picked[(picked["pair_ref"]==row["pair_ref"]) &
                       (picked["pair_sec"]==row["pair_sec"]) &
                       (picked["dem"]==dem)]
            if g.empty:
                continue
            vals = g["rmse_cm"].to_numpy()
            bp = ax.boxplot(
                [vals], positions=[xpos], widths=[each_width],
                whis=1.5, patch_artist=True, showfliers=True
            )
            # Style per DEM
            for patch in bp["boxes"]:
                patch.set(facecolor=to_rgba(colors[dem], 0.30), edgecolor=colors[dem], linewidth=1.2, hatch=hatches[dem])
            for whisk in bp["whiskers"]: whisk.set(color=colors[dem], linewidth=1.2)
            for cap in bp["caps"]:       cap.set(color=colors[dem], linewidth=1.2)
            for med in bp["medians"]:    med.set(color="k", linewidth=1.8)
            for flier in bp["fliers"]:
                flier.set(marker="o", markersize=3.5, markerfacecolor=colors[dem], markeredgecolor="white", alpha=0.85)

    ax.set_ylabel("RMSE (cm) — box: IQR; whiskers: 1.5×IQR; dots: outliers")
    ax.set_xlabel("Pairs (SRTM & 3DEP boxes per pair; combined width spans pair dates)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    # Legend: patches with colors+hatches
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=to_rgba(colors["SRTM"],0.30), edgecolor=colors["SRTM"], hatch=hatches["SRTM"], label="SRTM"),
                      Patch(facecolor=to_rgba(colors["3DEP"],0.30), edgecolor=colors["3DEP"], hatch=hatches["3DEP"], label="3DEP")]
    ax.legend(handles=legend_patches, loc="upper right")
    fig.autofmt_xdate()

    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"📈 Period (combined) boxplots written: {out_png}")

# ---------------------------- AREA POLYGON LOADING ----------------------------
def _load_area_polygon(area_name: str) -> Optional[object]:
    """Return unified area polygon for masking (if available), else None."""
    if not WATER_AREAS_GEOJSON.exists():
        return None
    gdf = gpd.read_file(WATER_AREAS_GEOJSON).to_crs(4326)
    cols = {c.lower(): c for c in gdf.columns}
    area_col = cols.get("area")
    if not area_col:
        return None
    sub = gdf[gdf[area_col].astype(str).str.upper() == area_name.upper()]
    if sub.empty:
        return None
    return unary_union(list(sub.geometry))

def _pick_map_raster(area_dir, area_name, pair_tag, dem):
    """Pick a raster file to display in the maps row (prefer TROPO_IONO, else RAW, TROPO, IONO)."""
    for corr_try in ("TROPO_IONO","RAW","TROPO","IONO"):
        p = _find_raster(area_dir, area_name, pair_tag, dem, corr_try)
        if p is not None:
            return p
    return None

# --------------------------------- DRIVER ------------------------------------
def process_area(area_dir: Path, reps: int, seed: int, idw_power: float, target_density: float) -> None:
    """
    Process a single AREA folder:
      • Evaluate all available (PAIR, DEM, CORR)
      • Overwrite results/accuracy_metrics.csv
      • Write per-pair density figures with maps
      • Write time-series boxplots (per-DEM and combined)
    """
    area_name   = area_dir.name
    gauge_csv   = area_dir / "water_gauges" / "eden_gauges.csv"
    results_dir = area_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = results_dir / "accuracy_metrics.csv"

    if not gauge_csv.exists():
        print(f"⏭️  Gauge CSV missing for {area_name}: {gauge_csv} — skipping area.")
        return

    pairs = _find_all_pairs(area_dir, area_name)
    if not pairs:
        print(f"⏭️  No interferograms in {area_dir/'interferograms'} — skipping area.")
        return

    # Load area polygon (for masking the IDW map)
    area_poly = _load_area_polygon(area_name)

    # Collect all evaluation rows for this AREA to write a fresh CSV at the end
    all_chunks = []

    for pair_tag in pairs:
        ref_iso, sec_iso = _pair_dates_from_tag(pair_tag)
        print(f"\n=== {area_name} — Pair {pair_tag} ===")
        pair_chunks = []

        # Evaluate every available raster (DEM×CORR)
        for dem in DEMS:
            for corr in CORRS_DISCOVER:
                tif = _find_raster(area_dir, area_name, pair_tag, dem, corr)
                if tif is None:
                    continue
                print(f"▶ Evaluating: {tif.name}")
                df = evaluate_one_raster(
                    area_name, gauge_csv, tif, dem, corr, ref_iso, sec_iso,
                    n_repl=reps, seed=seed, idw_power=idw_power
                )
                if not df.empty:
                    pair_chunks.append(df)

        if not pair_chunks:
            print(f"⏭️  No rasters evaluated for {pair_tag}.")
            continue

        # Merge this pair’s results & remember for CSV
        df_pair = pd.concat(pair_chunks, ignore_index=True)
        all_chunks.append(df_pair)

        # Per-DEM density plots with 3 lines (LS RAW, LS TI, IDW TI) and maps
        for dem in DEMS:
            if dem not in df_pair["dem"].unique():
                continue
            map_raster = _pick_map_raster(area_dir, area_name, pair_tag, dem)
            if map_raster is None:
                print(f"⚠️  No raster found for maps (DEM={dem}, pair={pair_tag}).")
                continue
            out_png = results_dir / f"acc_den_{pair_tag}_{dem}_LS2_IDW1.png"
            try:
                plot_density_three_lines_with_maps(
                    df_pair, area_name, pair_tag, dem, out_png,
                    gauge_csv=gauge_csv, ref_iso=ref_iso, sec_iso=sec_iso,
                    raster_for_maps=map_raster, area_poly=area_poly, idw_power=idw_power
                )
            except Exception as e:
                print(f"⚠️  Density plot failed for {pair_tag}, DEM={dem}: {e}")

    if not all_chunks:
        print(f"⏭️  No results to write for {area_name}.")
        return

    # Rebuild metrics CSV **fresh** for this AREA
    df_all = pd.concat(all_chunks, ignore_index=True)
    df_all.to_csv(metrics_csv, index=False)
    print(f"\n✅ [{area_name}] Metrics written (fresh): {metrics_csv}  (rows: {len(df_all)})")

    # Time-series per DEM (IDW on TROPO_IONO at target density)
    for dem in DEMS:
        out_png = results_dir / f"acc_period_{area_name}_{dem}_{str(target_density).replace('.', 'p')}.png"
        try:
            plot_time_series_per_dem_boxplots(df_all, area_name, dem, out_png, target_density)
        except Exception as e:
            print(f"⚠️  Period (per DEM) failed for {area_name}, DEM={dem}: {e}")

    # Combined SRTM + 3DEP boxplots
    out_png = results_dir / f"acc_period_{area_name}_COMBINED_{str(target_density).replace('.', 'p')}.png"
    try:
        plot_time_series_combined_boxplots(df_all, area_name, out_png, target_density)
    except Exception as e:
        print(f"⚠️  Period (combined) failed for {area_name}: {e}")

def main():
    ap = argparse.ArgumentParser(
        description="Accuracy vs. gauge-density with maps + time-series boxplots (per AREA)."
    )
    ap.add_argument("--areas-root", type=str, default=str(AREAS_ROOT),
                    help="Root folder containing per-area subfolders (default: %(default)s)")
    ap.add_argument("--area", type=str,
                    help="Run only this AREA (subfolder name under areas-root).")
    ap.add_argument("--reps", type=int, default=REPS_DEFAULT,
                    help="Replicates per raster (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT,
                    help="Random seed (default: %(default)s)")
    ap.add_argument("--idw-power", type=float, default=IDW_POWER_DEFAULT,
                    help="IDW power parameter (default: %(default)s)")
    ap.add_argument("--target-density", type=float, default=TARGET_DENSITY_DEFAULT,
                    help="Target density (km²/gauge) for time-series (default: %(default)s)")
    args = ap.parse_args()

    root = Path(args.areas_root)
    if args.area:
        targets = [root / args.area]
    else:
        targets = sorted([d for d in root.iterdir() if d.is_dir()])

    for area_dir in targets:
        process_area(area_dir, reps=args.reps, seed=args.seed, idw_power=args.idw_power, target_density=args.target_density)

if __name__ == "__main__":
    main()
