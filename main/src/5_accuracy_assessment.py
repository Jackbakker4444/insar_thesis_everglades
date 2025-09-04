#!/usr/bin/env python3
"""
5_accuracy_assessment.py — Shared-split accuracy metrics + 4 per-pair GeoTIFF exports

Purpose
-------
Evaluate per-area InSAR vertical-displacement rasters against EDEN gauges using a
**shared calibration/validation split** across all DEM/CORR variants of each pair.
For every (AREA, PAIR):
- Build a common gauge set (only gauges that sample valid values in **every** raster).
- Generate replicate calibration plans (farthest-point sampling, crowding reduction,
  center-only final step).
- Score two methods on the shared split:
  • least_squares  (calibrate InSAR → Δh_vis via y = a·x + b; fix a = -1 when n_cal ≤ 2)
  • idw_dhvis      (IDW interpolation of gauge Δh_vis at validation gauges)
- Write one fresh metrics CSV per AREA.
- Export four per-pair GeoTIFFs (from replicate #1): IDW(Δh_vis, 60%), calibrated
  TROPO_IONO at 60%, at target density, and center-only.

Needed data (inputs & assumptions)
----------------------------------
- Areas root (default):
  /mnt/DATA2/bakke326l/processing/areas/<AREA>/
    ├─ water_gauges/eden_gauges.csv
    │     Columns: StationID, Lat, Lon, and wide daily 'YYYY-MM-DD' (must include REF & SEC dates)
    └─ interferograms/
          <AREA>_vertical_cm_<REF>_<SEC>_<DEM>_<CORR>.tif
          where <REF>_<SEC> = YYYYMMDD_YYYYMMDD, <DEM> ∈ {SRTM, 3DEP}, <CORR> ∈ {RAW,TROPO,IONO,TROPO_IONO}
- Rasters are single-band float (cm), EPSG:4326, nodata set (or NaN).

Dependencies
------------
- Python: numpy, pandas, rasterio, pyproj (Geod)
- No external command-line tools required.

Outputs & directories
---------------------
Per AREA:
  <areas_root>/<AREA>/results/accuracy_metrics.csv     # rebuilt fresh each run

Per AREA and PAIR (replicate #1 plan):
  <areas_root>/<AREA>/results/
    idw60_<REF>_<SEC>.tif
    cal_ti_60pct_<DEMsel>_<REF>_<SEC>.tif
    cal_ti_d{D}_<DEMsel>_<REF>_<SEC>.tif   (D like 500p0 for 500.0 km²/gauge)
    cal_ti_1g_<DEMsel>_<REF>_<SEC>.tif

How to run
----------
# Process ALL areas under the default root
python 5_accuracy_assessment.py

# Only one area (e.g., ENP)
python 5_accuracy_assessment.py --area ENP

# Change replicates, seed, IDW power, output density target, DEMs/CORRs
python 5_accuracy_assessment.py \
  --reps 50 --seed 42 --idw-power 2.0 --output-density 500 \
  --dems SRTM 3DEP --corrs RAW TROPO IONO TROPO_IONO

Notes
-----
- “Shared split” = same calibration/validation gauges used for **all** rasters
  of the pair, ensuring fair comparisons.
- Chosen TROPO_IONO raster for calibrated exports prefers SRTM, then 3DEP.
- If no TROPO_IONO exists, calibrated exports are skipped; IDW export still written.
"""

from __future__ import annotations
from pathlib import Path
import re, os, logging, argparse
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.enums import Resampling
from pyproj import Geod

# ----------------------------------------------------------------------
# Basic logging hygiene (quiet GDAL/Rasterio chatter)
# ----------------------------------------------------------------------
os.environ.setdefault("CPL_DEBUG", "NO")
for _n in ("rasterio", "rasterio._io", "rasterio.env", "rasterio._base"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# ----------------------------------------------------------------------
# Defaults (overridable from CLI)
# ----------------------------------------------------------------------
AREAS_ROOT_DEFAULT      = Path("/mnt/DATA2/bakke326l/processing/areas")
DEMS_DEFAULT            = ["SRTM", "3DEP"]
CORRS_DEFAULT           = ["RAW", "TROPO", "IONO", "TROPO_IONO"]  # all four as requested
REPS_DEFAULT            = 50
SEED_DEFAULT            = 42
IDW_POWER_DEFAULT       = 2.0
OUTPUT_DENSITY_DEFAULT  = 500.0  # km²/gauge → choose n_cal whose density is closest

# Gauge columns
ID_COL, LAT_COL, LON_COL = "StationID", "Lat", "Lon"

# Geodesy helper for polygon areas
GEOD = Geod(ellps="WGS84")


# ========================= Small utilities =========================
def _pair_dates_from_tag(pair_tag: str) -> Tuple[str, str]:
    """
    Convert a pair tag 'YYYYMMDD_YYYYMMDD' to ISO dates ('YYYY-MM-DD', 'YYYY-MM-DD').

    Parameters
    ----------
    pair_tag : str
        Pair identifier in compact form.

    Returns
    -------
    tuple[str, str]
        (ref_iso, sec_iso).

    Raises
    ------
    ValueError
        If the tag format is invalid.
    """
    if not re.fullmatch(r"\d{8}_\d{8}", pair_tag):
        raise ValueError(f"PAIR tag must be YYYYMMDD_YYYYMMDD, got: {pair_tag}")
    a, b = pair_tag.split("_")
    return f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"

def _find_all_pairs(area_dir: Path, area_name: str, dems: List[str], corrs: List[str]) -> List[str]:
    """
    Scan <AREA>/interferograms for available pair tags, restricted to selected DEM/CORR.

    Parameters
    ----------
    area_dir : Path
    area_name : str
    dems : list[str]
    corrs : list[str]

    Returns
    -------
    list[str]
        Sorted unique pair tags 'YYYYMMDD_YYYYMMDD' present for any of the DEM/CORR combinations.
    """
    dem_pat = "|".join(map(re.escape, dems))
    corr_pat = "|".join(map(re.escape, corrs + ["TROPO", "IONO"]))  # tolerate extras present
    patt = re.compile(
        rf"^{re.escape(area_name)}_vertical_cm_(\d{{8}}_\d{{8}})_({dem_pat})_({corr_pat})\.tif$",
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
    """
    Return the path to one per-area interferogram raster if present.

    Parameters
    ----------
    area_dir : Path
    area_name : str
    pair_tag : str
    dem : str
    corr : str

    Returns
    -------
    Path | None
        The raster path or None if missing.
    """
    cand = area_dir / "interferograms" / f"{area_name}_vertical_cm_{pair_tag}_{dem.upper()}_{corr.upper()}.tif"
    return cand if cand.exists() else None

def _load_gauges_wide(csv_path: Path) -> pd.DataFrame:
    """
    Load an area's wide EDEN gauge CSV and validate required columns.

    Requires
    --------
    Columns: 'StationID', 'Lat', 'Lon' plus date columns 'YYYY-MM-DD'.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(csv_path)
    for c in (ID_COL, LAT_COL, LON_COL):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    return df

def _visible_surface_delta(ref_cm: np.ndarray, sec_cm: np.ndarray) -> np.ndarray:
    """
    Compute Δh_vis (cm) = max(sec, 0) - max(ref, 0) element-wise for gauge levels.
    """
    return np.maximum(sec_cm.astype(float), 0.0) - np.maximum(ref_cm.astype(float), 0.0)

def _rowcol_from_xy(transform: Affine, x: float, y: float) -> Tuple[float, float]:
    col, row = ~transform * (x, y)
    return float(row), float(col)

def _inside_image(h: int, w: int, row: float, col: float) -> bool:
    return (row >= 0) and (col >= 0) and (row < h) and (col < w)

def _read_mean_3x3(ds: rasterio.io.DatasetReader, row: int, col: int) -> Optional[float]:
    """Mean over a 3×3 window; returns None if all-NaN / all nodata."""
    r0 = max(0, row - 1); r1 = min(ds.height - 1, row + 1)
    c0 = max(0, col - 1); c1 = min(ds.width  - 1, col + 1)
    arr = ds.read(1, window=Window.from_slices((r0, r1 + 1), (c0, c1 + 1))).astype("float32")
    if ds.nodata is not None and not np.isnan(ds.nodata):
        arr[arr == ds.nodata] = np.nan
    if not np.isfinite(arr).any():
        return None
    return float(np.nanmean(arr))

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
    """
    Compute valid-data surface area (km²) from dataset_mask()==255 polygons.

    Assumes
    -------
    Raster is EPSG:4326.

    Returns
    -------
    float
        Total valid area in km².

    Raises
    ------
    RuntimeError
        If raster CRS is not EPSG:4326.
    """
    if ds.crs is None or ds.crs.to_epsg() != 4326:
        raise RuntimeError("Expected EPSG:4326 raster.")
    mask = (ds.dataset_mask() == 255).astype(np.uint8)
    area = 0.0
    for geom, val in shapes(mask, transform=ds.transform):
        if val == 1:
            area += _geod_area_of_geojson(geom)
    return float(area)

def _safe_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Correlation coefficient robust to zero-variance cases (returns NaN if undefined).
    """
    if len(y_true) < 2: return float("nan")
    yt = y_true - np.mean(y_true); yp = y_pred - np.mean(y_pred)
    vy = np.sum(yt*yt); vp = np.sum(yp*yp)
    if vy <= 1e-12 or vp <= 1e-12: return float("nan")
    return float(np.sum(yt*yp) / np.sqrt(vy*vp))

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, Bias, and Pearson r between predictions and truth.

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    dict
        {'rmse_cm','mae_cm','bias_cm','r'}
    """
    err = y_pred - y_true
    return {
        "rmse_cm": float(np.sqrt(np.mean(err**2))),
        "mae_cm":  float(np.mean(np.abs(err))),
        "bias_cm": float(np.mean(err)),
        "r":       _safe_corrcoef(y_true, y_pred),
    }

def _idw_predict_points(px, py, pz, qx, qy, power: float = 2.0) -> np.ndarray:
    """
    Inverse-distance weighting in lon/lat with cosine adjustment on Δlon.

    Behavior
    --------
    - Weights ∝ 1 / distance^power.
    - Snap exactly coincident query points to known values.

    Parameters
    ----------
    px, py, pz : arrays
        Known points (lon, lat, value).
    qx, qy : arrays
        Query (lon, lat).
    power : float
        IDW power (default 2.0).

    Returns
    -------
    np.ndarray (float32)
        Predicted values at query points.
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    pz = np.asarray(pz, dtype=np.float64)
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)

    cx = np.cos(np.deg2rad(np.nanmean(py) if py.size else 0.0))
    dx = (qx[:, None] - px[None, :]) * cx
    dy = (qy[:, None] - py[None, :])
    d2 = dx*dx + dy*dy
    w  = 1.0 / np.maximum(d2, 1e-18) ** (power/2.0)
    pred = (w @ pz) / np.sum(w, axis=1)
    imin = np.argmin(d2, axis=1)
    hits = d2[np.arange(d2.shape[0]), imin] < 1e-18
    if np.any(hits):
        pred[hits] = pz[imin[hits]]
    return pred.astype("float32")

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

def _crowded_candidates(lon: np.ndarray, lat: np.ndarray, idx: np.ndarray,
                        keep_global: int, top_n: int = 4) -> np.ndarray:
    """
    Return indices (within `idx`) of most-crowded gauges based on smallest NN distance.

    Parameters
    ----------
    lon, lat : np.ndarray
    idx : np.ndarray[int]
    keep_global : int
        Index of a gauge to keep (excluded from candidates).
    top_n : int
        Max number of candidates to return.
    """
    if len(idx) <= 1: return idx
    lon_s, lat_s = lon[idx], lat[idx]
    D = _haversine_matrix(lon_s, lat_s)
    np.fill_diagonal(D, np.inf)
    nnd = np.min(D, axis=1)
    order = np.argsort(nnd)  # smallest NN distance = most crowded
    order = np.array([o for o in order if idx[o] != keep_global], dtype=int)
    if order.size == 0: return np.array([], dtype=int)
    return order[:min(top_n, order.size)]


# ======================== Core evaluation logic ========================
def _fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Ordinary least squares fit for y ≈ a·x + b.

    Returns
    -------
    (a, b) : tuple[float, float]
    """
    A = np.c_[x, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])

def _eval_ls_and_idw(pts: pd.DataFrame, cal_idx: np.ndarray, val_idx: np.ndarray, idw_power: float) -> Dict[str, Dict[str, float]]:
    """
    Evaluate least-squares calibration and IDW(interpolated Δh_vis) on validation gauges.

    Parameters
    ----------
    pts : DataFrame
        Columns: 'insar_cm', 'dh_cm', 'Lon', 'Lat', and StationID.
    cal_idx, val_idx : np.ndarray[int]
    idw_power : float

    Returns
    -------
    dict[str, dict]
        {'least_squares': metrics_with_a_b, 'idw_dhvis': metrics}
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
        a, b = _fit_affine(x_cal, y_cal)
    else:
        a, b = np.nan, np.nan

    # Predict & score LS
    if np.isfinite(a) and np.isfinite(b):
        y_pred_ls = a * x_val + b
        m_ls = _compute_metrics(y_true=y_val, y_pred=y_pred_ls)
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
    m_idw = _compute_metrics(y_true=y_val, y_pred=y_pred_idw)

    return {"least_squares": m_ls, "idw_dhvis": m_idw}


# ===================== Export helpers (GeoTIFF writing) =====================
def _fit_ls_params(insar_vals: np.ndarray, dh_vals: np.ndarray) -> Tuple[float, float]:
    """
    Fit y = a·x + b with fallback a = -1, b = mean(dh + insar) when n ≤ 2.
    """
    n = insar_vals.size
    if n <= 2:
        a = -1.0
        b = float(np.mean(dh_vals + insar_vals))
        return a, b
    return _fit_affine(insar_vals, dh_vals)

def _write_tif_like(src_tif: Path, out_tif: Path, array2d: np.ndarray, nodata_value: float = -9999.0):
    """
    Write a float32 GeoTIFF using the spatial profile of `src_tif`.

    Parameters
    ----------
    src_tif : Path
        Reference raster (for metadata).
    out_tif : Path
        Output filepath.
    array2d : np.ndarray
        Data to write; NaNs converted to nodata.
    nodata_value : float
        Value to store for nodata (default −9999).
    """
    with rasterio.open(src_tif) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=nodata_value,
            compress="DEFLATE",
            predictor=3,
            tiled=False,
        )
        data = array2d.astype("float32")
        # Replace NaNs with nodata
        data_out = np.where(np.isfinite(data), data, nodata_value).astype("float32")
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(data_out, 1)

def _make_idw_grid_on_raster(px, py, pz, ref_tif: Path, power: float) -> np.ndarray:
    """
    Generate an IDW(Δh_vis) surface on the pixel grid of `ref_tif` (EPSG:4326).

    Parameters
    ----------
    px, py, pz : arrays
        Calibration lon, lat, Δh_vis.
    ref_tif : Path
        Reference raster whose grid/extent/mask define the output.
    power : float
        IDW power.

    Returns
    -------
    np.ndarray (H, W) float32
        IDW predictions with NaN outside the dataset mask.

    Raises
    ------
    RuntimeError
        If ref_tif is not EPSG:4326.
    """
    with rasterio.open(ref_tif) as ds:
        H, W = ds.height, ds.width
        transform = ds.transform
        crs = ds.crs
        if crs is None or crs.to_epsg() != 4326:
            raise RuntimeError(f"Expected EPSG:4326 grid for IDW export: {ref_tif}")

        # Precompute lon/lat for each column and row at pixel centers
        cols = np.arange(W, dtype=np.float64) + 0.5
        rows = np.arange(H, dtype=np.float64) + 0.5
        # Affine: x = a*col + b*row + c ; y = d*col + e*row + f
        a,b,c,d,e,f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        # y depends only on row when b==d==0 (north-up). We still do the general formula.
        xs_cols = a*cols + c  # since b=0 for north-up
        ys_rows = e*rows + f  # since d=0 for north-up

        out = np.full((H, W), np.nan, dtype="float32")

        # cosine factor based on calibration latitudes (same as in _idw_predict_points)
        cx = np.cos(np.deg2rad(np.nanmean(py) if len(py) else 0.0))

        # Row-by-row to limit memory
        for r in range(H):
            qy = np.full(W, ys_rows[r], dtype=np.float64)
            qx = xs_cols.astype(np.float64)
            # vectorized IDW (same as helper)
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
            out[r, :] = pred.astype("float32")

        valid_mask = (ds.dataset_mask() == 255)
        out = np.where(valid_mask, out, np.nan)
        return out

def _apply_calibration_to_raster(src_tif: Path, a: float, b: float) -> np.ndarray:
    """
    Apply affine calibration y = a·x + b to a raster, honoring dataset_mask and nodata.

    Returns
    -------
    np.ndarray
        Calibrated array with NaN outside valid data.
    """
    with rasterio.open(src_tif) as ds:
        arr = ds.read(1).astype("float32")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr = np.where(arr == ds.nodata, np.nan, arr)
        valid_mask = (ds.dataset_mask() == 255) & np.isfinite(arr)
        out = np.full_like(arr, np.nan, dtype="float32")
        out[valid_mask] = a * arr[valid_mask] + b
        return out


# =========================== Main per-pair routine ===========================
def _evaluate_pair_with_shared_split_and_exports(
    area_dir: Path,
    area_name: str,
    pair_tag: str,
    gauge_csv: Path,
    rasters: Dict[tuple, Path],   # key=(dem,corr) -> tif path
    n_repl: int,
    seed: int,
    idw_power: float,
    output_density_target: float,
) -> pd.DataFrame:
    """
    Evaluate one (AREA, PAIR) using a *shared* cal/val split and write 4 per-pair exports.

    Steps
    -----
    1) Build the common gauge set present in **all** rasters for the pair.
    2) Create replicate plans:
    - initial ~60% calibration via farthest-point sampling (exclude center),
    - fixed validation set (remaining gauges),
    - iteratively drop crowded calibration gauges down to n_cal=2,
    - final n_cal=1 is the center-only case.
    3) Score methods at each n_cal:
    - least_squares  (y = a·x + b; fix a =-1 when n_cal ≤ 2),
    - idw_dhvis      (interpolate Δh_vis at validation gauges).
    4) From replicate #1 plan, write:
    - idw60_<PAIR>.tif,
    - cal_ti_60pct_<DEMsel>_<PAIR>.tif,
    - cal_ti_d{D}_<DEMsel>_<PAIR>.tif  (closest density to --output-density),
    - cal_ti_1g_<DEMsel>_<PAIR>.tif.

    Parameters
    ----------
    area_dir : Path
    area_name : str
    pair_tag : str                # 'YYYYMMDD_YYYYMMDD'
    gauge_csv : Path              # <AREA>/water_gauges/eden_gauges.csv
    rasters : dict[(str,str),Path]   # key=(DEM, CORR)
    n_repl : int
    seed : int
    idw_power : float
    output_density_target : float # km² per gauge

    Returns
    -------
    pandas.DataFrame
        Metric rows (one per replicate x n_cal x method x DEM/CORR).

    Writes
    ------
    GeoTIFFs into <AREA>/results/ as described above (may skip calibrated if no TI raster).
    """
    results_dir = area_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load gauges & Δh_vis for the pair -----
    ref_iso, sec_iso = _pair_dates_from_tag(pair_tag)
    gauges = _load_gauges_wide(gauge_csv)
    for c in (ref_iso, sec_iso):
        if c not in gauges.columns:
            raise ValueError(f"[{area_name} {pair_tag}] gauge CSV missing date column: {c}")

    g = gauges[[ID_COL, LAT_COL, LON_COL, ref_iso, sec_iso]].copy()
    g.rename(columns={ref_iso: "ref_cm", sec_iso: "sec_cm"}, inplace=True)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)
    g.dropna(subset=["ref_cm", "sec_cm", LAT_COL, LON_COL], inplace=True)
    g["dh_cm"] = _visible_surface_delta(g["ref_cm"].to_numpy(), g["sec_cm"].to_numpy())

    # ----- Sample EACH raster at gauge points; record area_km2 per raster -----
    pts_by_raster: Dict[tuple, pd.DataFrame] = {}
    area_km2_by_raster: Dict[tuple, float] = {}

    for key, tif in rasters.items():
        with rasterio.open(tif) as ds:
            if ds.crs is None or ds.crs.to_epsg() != 4326:
                raise RuntimeError(f"Expected EPSG:4326: {tif}")
            area_km2_by_raster[key] = _valid_raster_area_km2(ds)

            rows = []
            for _, r in g.iterrows():
                x, y = float(r[LON_COL]), float(r[LAT_COL])
                rowf, colf = _rowcol_from_xy(ds.transform, x, y)
                if not _inside_image(ds.height, ds.width, rowf, colf):
                    continue
                ins = _read_mean_3x3(ds, int(round(rowf)), int(round(colf)))
                if ins is None or not np.isfinite(ins):
                    continue
                rows.append({
                    ID_COL: r[ID_COL],
                    LON_COL: x,
                    LAT_COL: y,
                    "insar_cm": float(ins),
                    "dh_cm": float(r["dh_cm"]),
                })
        if not rows:
            # No usable gauges in this raster → cannot be part of the shared split.
            return pd.DataFrame()
        pts_by_raster[key] = pd.DataFrame(rows)

    # ----- Build the *common* gauge set: present in ALL rasters -----
    sets = [set(df[ID_COL].astype(str)) for df in pts_by_raster.values()]
    common_ids = set.intersection(*sets) if sets else set()
    if len(common_ids) < 3:
        print(f"  ⚠️  Skipping {area_name} {pair_tag}: common usable gauges = {len(common_ids)} (<3).")
        return pd.DataFrame()

    common_ids_sorted = sorted(common_ids)
    any_df = next(iter(pts_by_raster.values()))
    meta = (any_df.set_index(ID_COL)
                 .loc[common_ids_sorted, [LON_COL, LAT_COL]]
                 .reset_index())
    # dh_cm for common set (from g)
    dh_all = g.set_index(ID_COL).loc[common_ids_sorted, "dh_cm"].to_numpy(dtype=float)
    lon_all = meta[LON_COL].to_numpy(dtype=float)
    lat_all = meta[LAT_COL].to_numpy(dtype=float)

    # insar per raster (aligned with common order)
    insar_by_key: Dict[tuple, np.ndarray] = {}
    for key, df in pts_by_raster.items():
        insar_by_key[key] = (df.set_index(ID_COL)
                               .loc[common_ids_sorted, "insar_cm"]
                               .to_numpy(dtype=float))

    # area per DEM (prefer TROPO_IONO if available)
    dem_area_km2: Dict[str, float] = {}
    for dem in sorted({k[0] for k in rasters.keys()}):
        if (dem, "TROPO_IONO") in area_km2_by_raster:
            dem_area_km2[dem] = area_km2_by_raster[(dem, "TROPO_IONO")]
        else:
            for (d, c), a in area_km2_by_raster.items():
                if d == dem:
                    dem_area_km2[dem] = a
                    break

    # Choose TROPO_IONO raster for calibrated exports (prefer SRTM, else 3DEP)
    chosen_ti_key = None
    for dem_try in ("SRTM", "3DEP"):
        if (dem_try, "TROPO_IONO") in rasters:
            chosen_ti_key = (dem_try, "TROPO_IONO")
            break
    # If none, we will still export IDW using some grid; pick *any* raster as fallback grid
    fallback_grid_key = next(iter(rasters.keys()))

    # ----- Shared center gauge -----
    lon_c, lat_c = float(lon_all.mean()), float(lat_all.mean())
    _, _, d_center = GEOD.inv(np.full_like(lon_all, lon_c), np.full_like(lat_all, lat_c), lon_all, lat_all)
    center_idx_global = int(np.argmin(d_center))

    N = len(common_ids_sorted)
    rng_master = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []

    # We’ll capture replicate #1’s plan for the 4 GeoTIFF exports
    export_plan = None  # dict with cal60_idx, val_idx, cal_seq (list of n_cal->idx)

    for rep in range(1, n_repl + 1):
        rng = np.random.default_rng(rng_master.integers(0, 2**31-1))

        # All indices, excluding center gauge for the sweep
        all_idx = np.arange(N, dtype=int)
        available_idx = np.setdiff1d(all_idx, np.array([center_idx_global]), assume_unique=False)

        # Initial ~60% calibration (spread-out), rest = validation
        n_cal0 = max(1, int(round(0.60 * len(available_idx))))
        n_cal0 = min(n_cal0, len(available_idx))
        cal_local = _spread_selection(lon_all[available_idx], lat_all[available_idx], n_cal0, rng=rng)
        cal_idx = available_idx[cal_local]
        val_idx = np.setdiff1d(available_idx, cal_idx, assume_unique=False)

        # Ensure validation not empty (rare, tiny N)
        if len(val_idx) == 0 and len(cal_idx) >= 2:
            crowded = _crowded_candidates(lon_all, lat_all, cal_idx, keep_global=center_idx_global, top_n=4)
            move_pos = crowded[0] if crowded.size else 0
            val_idx = np.r_[val_idx, [cal_idx[move_pos]]]
            cal_idx = np.delete(cal_idx, move_pos)

        # Save export plan from replicate #1
        if rep == 1:
            cal_seq = []
            cur = cal_idx.copy()
            while len(cur) >= 2:
                cal_seq.append(cur.copy())
                crowded = _crowded_candidates(lon_all, lat_all, cur, keep_global=center_idx_global, top_n=4)
                drop_pos = int(rng.choice(crowded)) if crowded.size else 0
                cur = np.delete(cur, drop_pos)
            # Add single-gauge (center)
            cal_seq.append(np.array([center_idx_global], dtype=int))
            export_plan = {
                "cal60_idx": cal_idx.copy(),
                "val_idx": val_idx.copy(),
                "cal_seq": cal_seq,  # ordered largest→2, then [center]
            }

        # March down to 2 cal gauges (metrics)
        cur_idx = cal_idx.copy()
        while len(cur_idx) >= 2:
            # IDW metrics: one line per DEM (density differs by area)
            # Build a pts view for IDW (insar is dummy)
            pts_idw = pd.DataFrame({
                ID_COL: [common_ids_sorted[i] for i in np.r_[cur_idx, val_idx]],
                LON_COL: lon_all[np.r_[cur_idx, val_idx]],
                LAT_COL: lat_all[np.r_[cur_idx, val_idx]],
                "insar_cm": np.zeros(len(cur_idx) + len(val_idx), dtype=float),
                "dh_cm": dh_all[np.r_[cur_idx, val_idx]],
            })
            mm_idw = _eval_ls_and_idw(pts_idw, np.arange(len(cur_idx)), np.arange(len(cur_idx), len(cur_idx) + len(val_idx)), idw_power=idw_power)
            for dem, area_km2 in dem_area_km2.items():
                base = {
                    "area": area_name, "pair_ref": ref_iso, "pair_sec": sec_iso,
                    "dem": dem, "corr": "TROPO_IONO", "method": "idw_dhvis",
                    "replicate": rep, "n_total": N, "n_cal": int(len(cur_idx)), "n_val": int(len(val_idx)),
                    "area_km2": float(area_km2),
                    "area_per_gauge_km2": float(area_km2) / float(len(cur_idx)),
                }
                m = mm_idw["idw_dhvis"]
                records.append({**base, **m, "a_gain": np.nan, "b_offset_cm": np.nan})

            # LS metrics: per raster (dem,corr)
            for (dem, corr), insar_vals in insar_by_key.items():
                vals_stacked = insar_vals[np.r_[cur_idx, val_idx]]
                pts_ls = pd.DataFrame({
                    ID_COL: [common_ids_sorted[i] for i in np.r_[cur_idx, val_idx]],
                    LON_COL: lon_all[np.r_[cur_idx, val_idx]],
                    LAT_COL: lat_all[np.r_[cur_idx, val_idx]],
                    "insar_cm": vals_stacked,
                    "dh_cm": dh_all[np.r_[cur_idx, val_idx]],
                })
                mm_ls = _eval_ls_and_idw(pts_ls, np.arange(len(cur_idx)), np.arange(len(cur_idx), len(cur_idx) + len(val_idx)), idw_power=idw_power)
                m = mm_ls["least_squares"]
                area_km2 = area_km2_by_raster[(dem, corr)]
                base = {
                    "area": area_name, "pair_ref": ref_iso, "pair_sec": sec_iso,
                    "dem": dem, "corr": corr, "method": "least_squares",
                    "replicate": rep, "n_total": N, "n_cal": int(len(cur_idx)), "n_val": int(len(val_idx)),
                    "area_km2": float(area_km2),
                    "area_per_gauge_km2": float(area_km2) / float(len(cur_idx)),
                }
                records.append({**base, **m})

            # drop one crowded and continue
            crowded = _crowded_candidates(lon_all, lat_all, cur_idx, keep_global=center_idx_global, top_n=4)
            drop_pos = int(rng.choice(crowded)) if crowded.size else 0
            cur_idx = np.delete(cur_idx, drop_pos)

        # Final single-gauge (center) metrics
        for (dem, corr), insar_vals in insar_by_key.items():
            vals_stacked = insar_vals[np.r_[ [center_idx_global], export_plan["val_idx"] if export_plan else val_idx ]]
            pts_ls = pd.DataFrame({
                ID_COL: [common_ids_sorted[i] for i in np.r_[ [center_idx_global], export_plan["val_idx"] if export_plan else val_idx ]],
                LON_COL: lon_all[np.r_[ [center_idx_global], export_plan["val_idx"] if export_plan else val_idx ]],
                LAT_COL: lat_all[np.r_[ [center_idx_global], export_plan["val_idx"] if export_plan else val_idx ]],
                "insar_cm": vals_stacked,
                "dh_cm": dh_all[np.r_[ [center_idx_global], export_plan["val_idx"] if export_plan else val_idx ]],
            })
            mm_ls = _eval_ls_and_idw(pts_ls, np.array([0], dtype=int), np.arange(1, pts_ls.shape[0]), idw_power=idw_power)
            m = mm_ls["least_squares"]
            area_km2 = area_km2_by_raster[(dem, corr)]
            base = {
                "area": area_name, "pair_ref": ref_iso, "pair_sec": sec_iso,
                "dem": dem, "corr": corr, "method": "least_squares",
                "replicate": rep, "n_total": N, "n_cal": 1, "n_val": int(len(val_idx)),
                "area_km2": float(area_km2),
                "area_per_gauge_km2": float(area_km2) / 1.0,
            }
            records.append({**base, **m})

        # IDW single-gauge metrics (mirrors density accounting per DEM)
        pts_idw_1 = pd.DataFrame({
            ID_COL: [common_ids_sorted[i] for i in np.r_[ [center_idx_global], val_idx ]],
            LON_COL: lon_all[np.r_[ [center_idx_global], val_idx ]],
            LAT_COL: lat_all[np.r_[ [center_idx_global], val_idx ]],
            "insar_cm": np.zeros(1 + len(val_idx), dtype=float),
            "dh_cm": dh_all[np.r_[ [center_idx_global], val_idx ]],
        })
        mm_idw_1 = _eval_ls_and_idw(pts_idw_1, np.array([0], dtype=int), np.arange(1, pts_idw_1.shape[0]), idw_power=idw_power)
        for dem, area_km2 in dem_area_km2.items():
            base = {
                "area": area_name, "pair_ref": ref_iso, "pair_sec": sec_iso,
                "dem": dem, "corr": "TROPO_IONO", "method": "idw_dhvis",
                "replicate": rep, "n_total": N, "n_cal": 1, "n_val": int(len(val_idx)),
                "area_km2": float(area_km2),
                "area_per_gauge_km2": float(area_km2) / 1.0,
            }
            m = mm_idw_1["idw_dhvis"]
            records.append({**base, **m, "a_gain": np.nan, "b_offset_cm": np.nan})

    # ------------------------ Per-pair 4 GeoTIFF exports ------------------------
    try:
        # 1) IDW(Δh_vis) from first 60% calibration — choose grid
        # Prefer chosen TROPO_IONO raster grid; else fallback to any raster grid.
        grid_key = chosen_ti_key if chosen_ti_key is not None else fallback_grid_key
        grid_tif = rasters[grid_key]
        cal60 = export_plan["cal60_idx"]
        px = lon_all[cal60]; py = lat_all[cal60]; pz = dh_all[cal60]
        idw_grid = _make_idw_grid_on_raster(px, py, pz, ref_tif=grid_tif, power=idw_power)
        out_idw = results_dir / f"idw60_{pair_tag}.tif"
        _write_tif_like(grid_tif, out_idw, idw_grid)
        print(f"  🗺️  IDW Δh_vis (60%) written: {out_idw}")
    except Exception as e:
        print(f"  ⚠️  IDW export failed for {pair_tag}: {e}")

    # 2–4) Calibrated TROPO_IONO rasters (only if a TI raster exists)
    if chosen_ti_key is not None:
        ti_dem = chosen_ti_key[0]
        ti_tif = rasters[chosen_ti_key]
        area_km2_ti = area_km2_by_raster[chosen_ti_key]

        # Build handy access to insar values (aligned with common IDs)
        insar_ti = insar_by_key[chosen_ti_key]

        # helper: compute (a,b) for a calibration index set
        def _ab_for(cal_idx: np.ndarray) -> Tuple[float,float]:
            return _fit_ls_params(insar_ti[cal_idx], dh_all[cal_idx])

        # 2) first 60% calibration
        try:
            a60, b60 = _ab_for(export_plan["cal60_idx"])
            arr60 = _apply_calibration_to_raster(ti_tif, a60, b60)
            out_cal60 = results_dir / f"cal_ti_60pct_{ti_dem}_{pair_tag}.tif"
            _write_tif_like(ti_tif, out_cal60, arr60)
            print(f"  🗺️  Calibrated TI (60%) written: {out_cal60}")
        except Exception as e:
            print(f"  ⚠️  Calibrated TI (60%) failed for {pair_tag}: {e}")

        # 3) closest-to-target density
        try:
            target = float(output_density_target)
            # Pick the n_cal in cal_seq whose area_km2_ti / n_cal is closest to target
            seq = export_plan["cal_seq"]  # list of numpy idx arrays
            n_list = np.array([len(s) for s in seq], dtype=int)
            dens = area_km2_ti / n_list.astype(float)
            i_best = int(np.argmin(np.abs(dens - target)))
            calD = seq[i_best]
            aD, bD = _ab_for(calD)
            arrD = _apply_calibration_to_raster(ti_tif, aD, bD)
            dtag = str(target).replace(".", "p")
            out_calD = results_dir / f"cal_ti_d{dtag}_{ti_dem}_{pair_tag}.tif"
            _write_tif_like(ti_tif, out_calD, arrD)
            print(f"  🗺️  Calibrated TI (≈{target:g} km²/g) written: {out_calD}")
        except Exception as e:
            print(f"  ⚠️  Calibrated TI (target density) failed for {pair_tag}: {e}")

        # 4) single-gauge (center)
        try:
            a1, b1 = _fit_ls_params(insar_ti[[center_idx_global]], dh_all[[center_idx_global]])
            arr1 = _apply_calibration_to_raster(ti_tif, a1, b1)
            out_cal1 = results_dir / f"cal_ti_1g_{ti_dem}_{pair_tag}.tif"
            _write_tif_like(ti_tif, out_cal1, arr1)
            print(f"  🗺️  Calibrated TI (1 gauge) written: {out_cal1}")
        except Exception as e:
            print(f"  ⚠️  Calibrated TI (1 gauge) failed for {pair_tag}: {e}")
    else:
        print(f"  ℹ️  No TROPO_IONO raster for {pair_tag} → calibrated TI exports skipped.")

    # --------------------------------------------------------------------------
    return pd.DataFrame.from_records(records)


# =============================== Driver (AREA) ===============================
def _process_area(area_dir: Path, dems: List[str], corrs: List[str],
                  reps: int, seed: int, idw_power: float, output_density_target: float) -> None:
    """
    Process a single AREA:
    - Discover pairs & rasters constrained by selected DEMS/CORRS.
    - Evaluate each pair with the shared-split pipeline.
    - Overwrite <AREA>/results/accuracy_metrics.csv with fresh results.
    - Write the 4 per-pair GeoTIFF exports.

    Parameters
    ----------
    area_dir : Path
    dems : list[str]
    corrs : list[str]
    reps : int
    seed : int
    idw_power : float
    output_density_target : float
    """
    area_name   = area_dir.name
    gauge_csv   = area_dir / "water_gauges" / "eden_gauges.csv"
    results_dir = area_dir / "results"
    metrics_csv = results_dir / "accuracy_metrics.csv"

    if not gauge_csv.exists():
        print(f"⏭️  Gauge CSV missing for {area_name}: {gauge_csv} — skipping area.")
        return

    pairs = _find_all_pairs(area_dir, area_name, dems, corrs)
    if not pairs:
        print(f"⏭️  No interferograms in {area_dir/'interferograms'} matching selected DEM/CORR — skipping area.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for pair_tag in pairs:
        # Collect rasters present for this pair (restricted to DEMS/CORRS requested)
        rasters: Dict[tuple, Path] = {}
        for dem in dems:
            for corr in corrs:
                p = _find_raster(area_dir, area_name, pair_tag, dem, corr)
                if p is not None:
                    rasters[(dem, corr)] = p
        if not rasters:
            continue

        print(f"\n=== {area_name} — Pair {pair_tag} ===")
        try:
            df_pair = _evaluate_pair_with_shared_split_and_exports(
                area_dir=area_dir,
                area_name=area_name,
                pair_tag=pair_tag,
                gauge_csv=gauge_csv,
                rasters=rasters,
                n_repl=reps,
                seed=seed,
                idw_power=idw_power,
                output_density_target=output_density_target,
            )
        except Exception as e:
            print(f"  ⚠️  Pair {pair_tag} failed: {e}")
            df_pair = pd.DataFrame()

        if not df_pair.empty:
            all_rows.append(df_pair)

    if not all_rows:
        print(f"⏭️  No results to write for {area_name}.")
        return

    # Rebuild metrics CSV **fresh** for this AREA
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(metrics_csv, index=False)
    print(f"\n✅ [{area_name}] metrics written (fresh): {metrics_csv}  (rows: {len(df_all)})")


# =================================== CLI ===================================
def main():
    """
    CLI entry point.

    Arguments
    ---------
    --areas-root : str (default: /mnt/DATA2/bakke326l/processing/areas)
    --area : str            # process only this AREA
    --reps : int            # replicates per pair (default 50)
    --seed : int            # RNG seed (default 42)
    --idw-power : float     # IDW power (default 2.0)
    --output-density : float  # km²/gauge target for 'cal_ti_d*.tif' (default 500.0)
    --dems : list[str]      # DEMs to include (default: SRTM 3DEP)
    --corrs : list[str]     # CORRs to include (default: RAW TROPO IONO TROPO_IONO)

    Behavior
    --------
    Iterates areas, runs evaluation & exports, and writes a fresh metrics CSV per area.
    """
    ap = argparse.ArgumentParser(
        description="Compute accuracy metrics with a SHARED calibration/validation split per (AREA,PAIR) and write 4 per-pair GeoTIFFs."
    )
    ap.add_argument("--areas-root", type=str, default=str(AREAS_ROOT_DEFAULT),
                    help="Root folder containing per-area subfolders (default: %(default)s)")
    ap.add_argument("--area", type=str,
                    help="Process only this AREA (subfolder name under --areas-root).")
    ap.add_argument("--reps", type=int, default=REPS_DEFAULT,
                    help="Replicates per pair (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT,
                    help="Random seed (default: %(default)s)")
    ap.add_argument("--idw-power", type=float, default=IDW_POWER_DEFAULT,
                    help="IDW power parameter (default: %(default)s)")
    ap.add_argument("--output-density", type=float, default=OUTPUT_DENSITY_DEFAULT,
                    help="Target density (km²/gauge) used to choose the calibration size for the 'cal_ti_d*.tif' export (default: %(default)s)")
    ap.add_argument("--dems", nargs="+", default=DEMS_DEFAULT,
                    help="DEMs to consider (default: %(default)s)")
    ap.add_argument("--corrs", nargs="+", default=CORRS_DEFAULT,
                    help="Corrections to consider (default: %(default)s)")
    args = ap.parse_args()

    root = Path(args.areas_root)
    targets = [root / args.area] if args.area else sorted([d for d in root.iterdir() if d.is_dir()])

    for area_dir in targets:
        _process_area(
            area_dir=area_dir,
            dems=[d.upper() for d in args.dems],
            corrs=[c.upper() for c in args.corrs],
            reps=args.reps,
            seed=args.seed,
            idw_power=args.idw_power,
            output_density_target=args.output_density,
        )

if __name__ == "__main__":
    main()
