#!/usr/bin/env python3
# =============================================================================
# Dependencies (Python libraries)
# =============================================================================
# • Standard library: pathlib, re, os, shutil, logging, argparse, json, typing
# • Third-party: numpy, pandas, rasterio, pyproj
#
# Recommended install (conda-forge):
#   conda install -c conda-forge numpy pandas rasterio pyproj

"""
5_accuracy_assessment_dem_corr.py — Spread-out (stochastic farthest-point) 60/45/30/15% split metrics
+ 60% calibrated GeoTIFFs + 60/40 split GeoJSON (from replicate #1)

What changed vs. previous version
---------------------------------
• Metrics are computed for FOUR calibration densities per replicate:
  **60%**, **45%**, **30%**, **15%** (validation is the remainder).
  Each density uses the same *stochastic farthest-point* selection (pick-furthest with randomness).
• Replicates: default N=200 (override with --reps).
• Exports from **replicate #1** remain:
    - IDW(Δh_vis) grid at 60%:     idw60_<PAIR>.tif
    - 60% calibrated rasters:      cal_60pct_<DEM>_<CORR>_<PAIR>.tif
    - 60/40 split GeoJSON:         split_60_40_<PAIR>.geojson

Results CSV
-----------
<areas_root>/<AREA>/results/accuracy_metrics.csv
  • One row per (PAIR, DEM, CORR, replicate, density) + one IDW row per (replicate, density).
  • Includes n_total, n_cal, n_val.
  • Adds density columns: cal_frac (float), cal_pct (int), density_tag (e.g., '60pct').

Assumptions
-----------
- Interferograms: single-band float (cm), EPSG:4326, nodata/NaN honored.
- Gauges CSV: StationID, Lat, Lon, and daily 'YYYY-MM-DD' columns including REF & SEC.

How to run
----------
All areas (defaults):
    python 5_accuracy_assessment_dem_corr.py

Single area:
    python 5_accuracy_assessment_dem_corr.py --area ENP

Choose DEMs/CORRs, reps, etc.:
    python 5_accuracy_assessment_dem_corr.py \
      --reps 200 --seed 42 --idw-power 2.0 \
      --dems SRTM 3DEP --corrs RAW TROPO IONO TROPO_IONO

Overview
--------
This script evaluates InSAR vertical-displacement rasters (cm) against EDEN water-level gauges
using shared calibration/validation splits across multiple densities per replicate. For each
(AREA, PAIR) and for every available (DEM, CORR) variant, it computes accuracy metrics
(RMSE, MAE, bias, sigma_e, r) for Least-Squares (calibrating InSAR → Δh_vis) alongside an IDW baseline
that interpolates Δh_vis from calibration gauges. From replicate #1 at 60% calibration density,
it exports (i) an IDW(Δh_vis) raster, (ii) calibrated GeoTIFFs for each (DEM, CORR), and (iii) a
GeoJSON of the 60/40 split for visualization/QA.
"""

from __future__ import annotations
from pathlib import Path
import re, os, shutil, logging, argparse, json
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.transform import Affine
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
CORRS_DEFAULT           = ["RAW", "TROPO", "IONO", "TROPO_IONO"]
REPS_DEFAULT            = 200
SEED_DEFAULT            = 42
IDW_POWER_DEFAULT       = 2.0

# Calibration subset selection randomness: pick among top-M farthest each step
SPREAD_TOP_M_DEFAULT    = 5

# Densities (fractions of usable gauges for CALIBRATION)
CAL_FRACS_DEFAULT       = [0.60, 0.45, 0.30, 0.15]

# Gauge columns
ID_COL, LAT_COL, LON_COL = "StationID", "Lat", "Lon"

# Geodesy helper
GEOD = Geod(ellps="WGS84")

# ===================== Upfront cleanup helpers (results folders) =====================
def _clean_all_results_dirs(areas_root: Path) -> None:
    """Remove all contents inside every ``<AREA>/results`` under ``areas_root``.

    Parameters
    ----------
    areas_root : pathlib.Path
        Root directory containing per-area subfolders. Each area is expected to
        have a ``results`` subfolder that will be cleared or created if absent.

    Notes
    -----
    • Only files and directories **inside** ``results`` are removed; the area
      folder and its ``results`` directory remain.
    • Any failure to clean a specific area is reported but does not abort the run.
    """
    if not areas_root.exists():
        return
    for d in sorted(p for p in areas_root.iterdir() if p.is_dir()):
        res = d / "results"
        try:
            if res.exists():
                for item in res.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink(missing_ok=True)
            else:
                res.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️  Could not clean results in {res}: {e}")

# ========================= Small utilities =========================
def _pair_dates_from_tag(pair_tag: str) -> Tuple[str, str]:
    """Convert a pair tag ``YYYYMMDD_YYYYMMDD`` → (``YYYY-MM-DD``, ``YYYY-MM-DD``).

    Parameters
    ----------
    pair_tag : str
        Pair identifier of the form ``YYYYMMDD_YYYYMMDD``.

    Returns
    -------
    tuple[str, str]
        ISO-8601 formatted reference and secondary dates.

    Raises
    ------
    ValueError
        If the input does not match the expected ``YYYYMMDD_YYYYMMDD`` pattern.
    """
    if not re.fullmatch(r"\d{8}_\d{8}", pair_tag):
        raise ValueError(f"PAIR tag must be YYYYMMDD_YYYYMMDD, got: {pair_tag}")
    a, b = pair_tag.split("_")
    return f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"

def _find_all_pairs(area_dir: Path, area_name: str, dems: List[str], corrs: List[str]) -> List[str]:
    """Scan ``<AREA>/interferograms`` for available pair tags matching DEM/CORR filters.

    Parameters
    ----------
    area_dir : pathlib.Path
        Path to the area folder.
    area_name : str
        Area short name (prefix in raster filenames).
    dems : list[str]
        Allowed DEM names.
    corrs : list[str]
        Allowed correction names.

    Returns
    -------
    list[str]
        Sorted unique pair tags present for the selected DEM/CORR combinations.
    """
    dem_pat = "|".join(map(re.escape, dems))
    corr_pat = "|".join(map(re.escape, corrs))
    patt = re.compile(
        rf"^{re.escape(area_name)}_vertical_cm_(\d{{8}}_\d{{8}})_({dem_pat})_({corr_pat})\\.tif$",
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
    """Return the path to a specific interferogram raster if present, else ``None``.

    This follows the naming convention ``<AREA>_vertical_cm_<PAIR>_<DEM>_<CORR>.tif``.
    """
    cand = area_dir / "interferograms" / f"{area_name}_vertical_cm_{pair_tag}_{dem.upper()}_{corr.upper()}.tif"
    return cand if cand.exists() else None

def _load_gauges_wide(csv_path: Path) -> pd.DataFrame:
    """Load an area-level EDEN gauge CSV and validate required columns.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to ``eden_gauges.csv`` in *wide* format (StationID, Lat, Lon, dates...).

    Returns
    -------
    pandas.DataFrame
        DataFrame as loaded; raises on missing required columns.
    """
    df = pd.read_csv(csv_path)
    for c in (ID_COL, LAT_COL, LON_COL):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    return df

def _visible_surface_delta(ref_cm: np.ndarray, sec_cm: np.ndarray) -> np.ndarray:
    """Compute visible-surface water-level change Δh_vis in centimeters.

    Definition
    ----------
    ``Δh_vis = max(sec, 0) - max(ref, 0)`` applied element-wise. Negative water
    levels (dry pixels) are clamped to 0 to mimic “visible water surface”.
    """
    return np.maximum(sec_cm.astype(float), 0.0) - np.maximum(ref_cm.astype(float), 0.0)

def _rowcol_from_xy(transform: Affine, x: float, y: float) -> Tuple[float, float]:
    """Map world coordinates (lon, lat) to **fractional** raster (row, col).

    Notes
    -----
    The result is not rounded. Rounding/nearest-neighbor selection is deferred to
    the caller because some sampling windows (e.g., 3x3 mean) benefit from
    explicit integer casting at the last moment.
    """
    col, row = ~transform * (x, y)
    return float(row), float(col)

def _inside_image(h: int, w: int, row: float, col: float) -> bool:
    """Return ``True`` if (row, col) lies inside image bounds ``[0,h)x[0,w)``."""
    return (row >= 0) and (col >= 0) and (row < h) and (col < w)

def _read_mean_3x3(ds: rasterio.io.DatasetReader, row: int, col: int) -> Optional[float]:
    """Read a 3x3 window centered on (row, col) and return the mean, ignoring NaNs.

    Parameters
    ----------
    ds : rasterio.io.DatasetReader
        Open raster dataset (single band assumed).
    row, col : int
        Integer pixel indices for the window center.

    Returns
    -------
    float or None
        NaN-aware mean over the window; ``None`` if all are nodata.
    """
    r0 = max(0, row - 1); r1 = min(ds.height - 1, row + 1)
    c0 = max(0, col - 1); c1 = min(ds.width  - 1, col + 1)
    arr = ds.read(1, window=Window.from_slices((r0, r1 + 1), (c0, c1 + 1))).astype("float32")
    # Convert explicit nodata codes to NaN so statistics can ignore them
    if ds.nodata is not None and not np.isnan(ds.nodata):
        arr[arr == ds.nodata] = np.nan
    if not np.isfinite(arr).any():
        return None
    return float(np.nanmean(arr))

def _safe_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation ``r`` safely; return NaN if undefined.

    Handles edge cases where either vector has near-zero variance or fewer than
    two samples.
    """
    if len(y_true) < 2: return float("nan")
    yt = y_true - np.mean(y_true); yp = y_pred - np.mean(y_pred)
    vy = np.sum(yt*yt); vp = np.sum(yp*yp)
    if vy <= 1e-12 or vp <= 1e-12: return float("nan")
    return float(np.sum(yt*yp) / np.sqrt(vy*vp))

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute base accuracy metrics between truth and predictions.

    Metrics
    -------
    • ``rmse_cm``: Root Mean Square Error (cm)
    • ``mae_cm``:  Mean Absolute Error (cm)
    • ``bias_cm``: Mean signed error (cm)
    • ``sigma_e_cm``: Error STD about the **bias** (cm)
    • ``r``: Pearson correlation coefficient
    """
    err = y_pred - y_true
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    sigma_e = float(np.sqrt(np.mean((err - bias)**2)))
    return {"rmse_cm": rmse, "mae_cm": mae, "bias_cm": bias, "sigma_e_cm": sigma_e, "r": _safe_corrcoef(y_true, y_pred)}

def _val_spread(y_val: np.ndarray) -> Tuple[float, float]:
    """Return validation spread measures: (IQR_cm, SD_cm).

    Returns
    -------
    tuple[float, float]
        Interquartile range (Q75-Q25) and standard deviation of ``y_val``.
    """
    yv = np.asarray(y_val, dtype=float)
    if yv.size == 0 or not np.isfinite(yv).any():
        return float("nan"), float("nan")
    q25, q75 = np.nanpercentile(yv, [25, 75])
    iqr = float(q75 - q25)
    sd  = float(np.nanstd(yv))
    return iqr, sd

def _augment_metrics(m: Dict[str, float], y_val: np.ndarray) -> Dict[str, float]:
    """Augment base metrics with normalized variants and transforms.

    Adds
    ----
    • ``val_dh_iqr_cm``, ``val_dh_sd_cm``
    • ``nrmse_iqr``: RMSE normalized by IQR
    • ``nrmse_sd``:  RMSE normalized by SD
    • ``log_rmse_cm``: natural log of RMSE
    • ``fisher_z``: Fisher z-transform of correlation
    """
    iqr, sd = _val_spread(y_val)
    rmse = m.get("rmse_cm", np.nan); r = m.get("r", np.nan)
    m.update({
        "val_dh_iqr_cm": iqr, "val_dh_sd_cm": sd,
        "nrmse_iqr": (rmse / iqr) if (np.isfinite(rmse) and iqr and iqr > 0) else float("nan"),
        "nrmse_sd":  (rmse / sd)  if (np.isfinite(rmse) and sd  and sd  > 0) else float("nan"),
        "log_rmse_cm": np.log(rmse) if (np.isfinite(rmse) and rmse > 0) else float("nan"),
        "fisher_z": float(np.arctanh(np.clip(r, -0.999999, 0.999999))) if np.isfinite(r) else float("nan"),
    })
    return m

# ======================= IDW & helpers =======================
def _idw_predict_points(px, py, pz, qx, qy, power: float = 2.0) -> np.ndarray:
    """Inverse-distance weighting (IDW) interpolation on lon/lat.

    Parameters
    ----------
    px, py : array-like
        Longitudes and latitudes of **known** points (calibration gauges).
    pz : array-like
        Known values (Δh_vis at calibration gauges).
    qx, qy : array-like
        Longitudes and latitudes of **query** points (validation gauges).
    power : float, default 2.0
        IDW power exponent. Larger values emphasize nearby points.

    Returns
    -------
    numpy.ndarray
        Predicted values for each query point. Exact hits (zero distance) get
        the corresponding observed value.

    Notes
    -----
    A cosine factor is applied to longitudinal differences to approximate
    distance distortion with latitude in geographic coordinates (EPSG:4326).
    """
    px = np.asarray(px, dtype=np.float64); py = np.asarray(py, dtype=np.float64)
    pz = np.asarray(pz, dtype=np.float64); qx = np.asarray(qx, dtype=np.float64)
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
    """Compute a symmetric great-circle distance matrix (meters) for lon/lat arrays.

    Returns
    -------
    numpy.ndarray
        ``NxN`` matrix where entry ``(i,j)`` is the arc distance between points
        ``i`` and ``j`` on a sphere (R≈6371 km).
    """
    lon = np.deg2rad(lon.astype("float64")); lat = np.deg2rad(lat.astype("float64"))
    dlon = lon[:, None] - lon[None, :]
    a = np.clip(np.sin(lat)[:,None]*np.sin(lat)[None,:] + np.cos(lat)[:,None]*np.cos(lat)[None,:]*np.cos(dlon), -1.0, 1.0)
    return 6371000.0 * np.arccos(a)

def _spread_selection_stochastic(lon: np.ndarray, lat: np.ndarray, k: int,
                                 rng: np.random.Generator, top_m: int = SPREAD_TOP_M_DEFAULT) -> np.ndarray:
    """Stochastic farthest-point sampling to select ``k`` well-spread indices.

    Algorithm
    ---------
    1) Seed with a random point.
    2) At each step, compute nearest-neighbor distance to current set and pick
       **one** index *uniformly at random* among the ``top_m`` farthest points.

    Parameters
    ----------
    lon, lat : numpy.ndarray
        Candidate longitudes/latitudes.
    k : int
        Desired number of selected points (clamped to available size).
    rng : numpy.random.Generator
        RNG used for stochastic choices (ensures replicability from seed).
    top_m : int, default ``SPREAD_TOP_M_DEFAULT``
        How many of the farthest candidates to randomize over per step.

    Returns
    -------
    numpy.ndarray
        Sorted array of selected indices into the input arrays.
    """
    n = len(lon)
    if k >= n: return np.arange(n, dtype=int)
    D = _haversine_matrix(lon, lat)
    cur = [int(rng.integers(0, n))]
    remaining = set(range(n)) - set(cur)
    # Track each point's distance to the current set (nearest member)
    min_d = D[:, cur].min(axis=1)
    while len(cur) < k:
        order = np.argsort(-min_d)  # descending by min-distance
        order = [o for o in order if o in remaining]
        if not order:
            break
        take = order[:min(top_m, len(order))]
        cand = int(rng.choice(take))
        cur.append(cand)
        remaining.remove(cand)
        min_d = np.minimum(min_d, D[:, cand])
    return np.array(cur, dtype=int)

# ===================== Export helpers (GeoTIFF & GeoJSON) =====================
def _fit_ls_params(insar_vals: np.ndarray, dh_vals: np.ndarray) -> Tuple[float, float]:
    """Fit parameters ``(a, b)`` for ``y = a·x + b`` with a small-n fallback.

    For ``n≤2``, it uses ``a=-1`` and ``b=mean(dh + insar)`` to stabilize the
    calibration in extremely sparse cases (empirically reasonable in wetlands).
    """
    n = insar_vals.size
    if n <= 2:
        a = -1.0
        b = float(np.mean(dh_vals + insar_vals))
        return a, b
    A = np.c_[insar_vals, np.ones_like(insar_vals)]
    sol, *_ = np.linalg.lstsq(A, dh_vals, rcond=None)
    return float(sol[0]), float(sol[1])

def _write_tif_like(src_tif: Path, out_tif: Path, array2d: np.ndarray, nodata_value: float = -9999.0):
    """Write a float32 GeoTIFF using the spatial profile of ``src_tif``.

    Parameters
    ----------
    src_tif : pathlib.Path
        Reference raster whose georeferencing/profile is copied.
    out_tif : pathlib.Path
        Destination path for the new GeoTIFF.
    array2d : numpy.ndarray
        Array to write (NaNs are converted to ``nodata_value``).
    nodata_value : float, default -9999.0
        Stored nodata sentinel in the output raster.
    """
    with rasterio.open(src_tif) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", dtype="float32", count=1, nodata=nodata_value,
                       compress="DEFLATE", predictor=3, tiled=False)
        data = array2d.astype("float32")
        data_out = np.where(np.isfinite(data), data, nodata_value).astype("float32")
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(data_out, 1)

def _make_idw_grid_on_raster(px, py, pz, ref_tif: Path, power: float) -> np.ndarray:
    """Generate an IDW(Δh_vis) surface on the pixel grid of ``ref_tif`` (EPSG:4326).

    Parameters
    ----------
    px, py, pz : array-like
        Calibration longitudes, latitudes, and Δh_vis values.
    ref_tif : pathlib.Path
        Raster defining target grid, resolution, and spatial extent.
    power : float
        IDW power exponent (passed to :func:`_idw_predict_points`).

    Returns
    -------
    numpy.ndarray
        Float32 raster array aligned to ``ref_tif``; NaN outside valid mask.

    Raises
    ------
    RuntimeError
        If ``ref_tif`` is not EPSG:4326 (geographic).
    """
    with rasterio.open(ref_tif) as ds:
        H, W = ds.height, ds.width
        transform = ds.transform
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise RuntimeError(f"Expected EPSG:4326 grid for IDW export: {ref_tif}")
        base = ds.read(1).astype("float32")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            base[base == ds.nodata] = np.nan

        # Precompute center coordinates of columns/rows
        cols = np.arange(W, dtype=np.float64) + 0.5
        rows = np.arange(H, dtype=np.float64) + 0.5
        a,b,c,d,e,f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        xs_cols = a*cols + c
        ys_rows = e*rows + f

        out = np.full((H, W), np.nan, dtype="float32")
        cx = np.cos(np.deg2rad(np.nanmean(py) if len(py) else 0.0))

        # Row-wise evaluation to bound memory usage
        for r in range(H):
            qy = np.full(W, ys_rows[r], dtype=np.float64)
            qx = xs_cols.astype(np.float64)
            dx = (qx[:, None] - px[None, :]) * cx
            dy = (qy[:, None] - py[None, :])
            d2 = dx*dx + dy*dy
            w  = 1.0 / np.maximum(d2, 1e-18) ** (power/2.0)
            pred = (w @ pz) / np.sum(w, axis=1)
            imin = np.argmin(d2, axis=1)
            hits = d2[np.arange(d2.shape[0]), imin] < 1e-18
            if np.any(hits):
                pred[hits] = pz[imin[hits]]
            out[r, :] = pred.astype("float32")

        # Respect valid mask of reference raster (e.g., coastline)
        valid_mask = np.isfinite(base)
        out = np.where(valid_mask, out, np.nan)
        return out

def _apply_calibration_to_raster(src_tif: Path, a: float, b: float) -> np.ndarray:
    """Apply linear calibration ``y = a·x + b`` to a raster, honoring nodata/NaN.

    Parameters
    ----------
    src_tif : pathlib.Path
        Source raster to transform.
    a, b : float
        Gain and offset estimated from calibration gauges.

    Returns
    -------
    numpy.ndarray
        Calibrated array (float32) with NaNs carried through.
    """
    with rasterio.open(src_tif) as ds:
        arr = ds.read(1).astype("float32")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr = np.where(arr == ds.nodata, np.nan, arr)
        valid_mask = np.isfinite(arr)
        out = np.full_like(arr, np.nan, dtype="float32")
        out[valid_mask] = a * arr[valid_mask] + b
        return out

def _write_split_geojson(out_path: Path,
                         ids: List[str], lons: np.ndarray, lats: np.ndarray,
                         cal_idx: np.ndarray, val_idx: np.ndarray,
                         area: str, pair_tag: str, ref_iso: str, sec_iso: str,
                         replicate: int = 1) -> None:
    """Write a single GeoJSON with gauge features labeled ``cal``/``val``.

    Intended for quick QA/visualization of the 60/40 split from replicate #1.
    Excluded gauges are omitted to reduce clutter in map viewers.
    """
    cal_set = set(int(i) for i in cal_idx.tolist())
    val_set = set(int(i) for i in val_idx.tolist())
    feats = []
    for i in range(len(ids)):
        split = "cal" if i in cal_set else ("val" if i in val_set else "excluded")
        if split == "excluded":
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lons[i]), float(lats[i])]},
            "properties": {
                "area": area,
                "pair_tag": pair_tag,
                "pair_ref": ref_iso,
                "pair_sec": sec_iso,
                "replicate": int(replicate),
                "split": split,
                ID_COL: str(ids[i]),
            }
        })
    fc = {"type": "FeatureCollection", "features": feats}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)

# =========================== Core evaluation logic (multi-density) ===========================
def _eval_ls_and_idw_for_split(pts: pd.DataFrame, cal_idx: np.ndarray, val_idx: np.ndarray, idw_power: float):
    """Evaluate LS calibration and IDW baseline on a given cal/val split.

    Parameters
    ----------
    pts : pandas.DataFrame
        Must contain columns ``['insar_cm','dh_cm', LON_COL, LAT_COL]`` for the
        union of calibration and validation gauges *in that order* (cal then val)
        when indices are provided as ``np.arange`` slices below.
    cal_idx, val_idx : numpy.ndarray
        Integer indices selecting calibration and validation subsets.
    idw_power : float
        Power exponent for IDW baseline evaluation.

    Returns
    -------
    tuple[dict, dict]
        (metrics_ls, metrics_idw), both augmented with normalized metrics.

    Notes
    -----
    For LS:
        If the calibration set has at most two points, a robust fallback
        ``a=-1`` and ``b=mean(dh + insar)`` is used to avoid unstable fits.
    """
    x_cal = pts["insar_cm"].values[cal_idx].astype(float)
    y_cal = pts["dh_cm"].values[cal_idx].astype(float)
    x_val = pts["insar_cm"].values[val_idx].astype(float)
    y_val = pts["dh_cm"].values[val_idx].astype(float)

    # LS params (fallback when tiny cal set)
    if len(cal_idx) <= 2:
        a, b = -1.0, float(np.mean(y_cal + x_cal))
    else:
        A = np.c_[x_cal, np.ones_like(x_cal)]
        sol, *_ = np.linalg.lstsq(A, y_cal, rcond=None)
        a, b = float(sol[0]), float(sol[1])

    y_pred_ls = a * x_val + b
    m_ls = _augment_metrics(_compute_metrics(y_true=y_val, y_pred=y_pred_ls), y_val)
    m_ls.update({"a_gain": float(a), "b_offset_cm": float(b)})

    # IDW baseline (on Δh_vis only)
    px = pts[LON_COL].values[cal_idx].astype(float)
    py = pts[LAT_COL].values[cal_idx].astype(float)
    pz = pts["dh_cm"].values[cal_idx].astype(float)
    qx = pts[LON_COL].values[val_idx].astype(float)
    qy = pts[LAT_COL].values[val_idx].astype(float)
    y_pred_idw = _idw_predict_points(px, py, pz, qx, qy, power=idw_power)
    m_idw = _augment_metrics(_compute_metrics(y_true=y_val, y_pred=y_pred_idw), y_val)

    return m_ls, m_idw

def _evaluate_pair_spread_multidensity_and_exports(
    area_dir: Path,
    area_name: str,
    pair_tag: str,
    gauge_csv: Path,
    rasters: Dict[tuple, Path],   # key=(dem,corr) -> tif path
    n_repl: int,
    seed: int,
    idw_power: float,
    top_m: int = SPREAD_TOP_M_DEFAULT,
    cal_fracs: List[float] = CAL_FRACS_DEFAULT,
) -> pd.DataFrame:
    """Evaluate one (AREA, PAIR) with shared multi-density cal/val splits and do exports.

    Workflow per pair
    -----------------
    1) Load EDEN gauges and compute Δh_vis for (REF, SEC).
    2) Sample every available (DEM, CORR) raster at the same **usable** gauges.
    3) Build shared calibration/validation plans for multiple densities per replicate
       using stochastic farthest-point selection (spread-out but randomized).
    4) Compute IDW baseline metrics and LS metrics per (DEM, CORR) for each density.
    5) From replicate #1 at 60%: export IDW grid, calibrated rasters, and split GeoJSON.

    Returns
    -------
    pandas.DataFrame
        Long table of metrics for all densities x replicates and methods.
        Empty DataFrame if not enough common gauges (<3) or sampling fails.
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

    # ----- Sample EACH raster at gauge points; build shared mask -----
    pts_by_raster: Dict[tuple, pd.DataFrame] = {}

    for key, tif in rasters.items():
        with rasterio.open(tif) as ds:
            if ds.crs is None or ds.crs.to_epsg() != 4326:
                raise RuntimeError(f"Expected EPSG:4326: {tif}")

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
            return pd.DataFrame()
        pts_by_raster[key] = pd.DataFrame(rows)

    # ----- Common set of usable gauges across ALL rasters -----
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

    dh_all  = g.set_index(ID_COL).loc[common_ids_sorted, "dh_cm"].to_numpy(dtype=float)
    lon_all = meta[LON_COL].to_numpy(dtype=float)
    lat_all = meta[LAT_COL].to_numpy(dtype=float)
    ids_all = [str(i) for i in common_ids_sorted]

    # Stack per-raster InSAR samples in a dict keyed by (DEM, CORR)
    insar_by_key: Dict[tuple, np.ndarray] = {}
    for key, df in pts_by_raster.items():
        insar_by_key[key] = (df.set_index(ID_COL)
                               .loc[common_ids_sorted, "insar_cm"]
                               .to_numpy(dtype=float))

    # ----- Center gauge (closest to centroid); exclude from pool to avoid trivial anchor -----
    lon_c, lat_c = float(lon_all.mean()), float(lat_all.mean())
    _, _, d_center = GEOD.inv(np.full_like(lon_all, lon_c), np.full_like(lat_all, lat_c), lon_all, lat_all)
    center_idx_global = int(np.argmin(d_center))

    N = len(common_ids_sorted)
    rng_master = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []

    export_plan_60 = None  # store replicate #1 plan for the 60% export

    for rep in range(1, n_repl + 1):
        # Derive a fresh RNG per replicate for reproducibility without correlation
        rng = np.random.default_rng(rng_master.integers(0, 2**31-1))
        all_idx = np.arange(N, dtype=int)
        # Exclude center point from selection set to avoid degenerate splits
        available_idx = np.setdiff1d(all_idx, np.array([center_idx_global]), assume_unique=False)
        n_avail = len(available_idx)

        for frac in cal_fracs:
            # number of calibration gauges (clamped to [1, n_avail-1])
            k = int(round(frac * n_avail))
            k = max(1, min(k, n_avail - 1)) if n_avail >= 2 else 1

            # spread-out (stochastic farthest-point) selection
            cal_local = _spread_selection_stochastic(
                lon_all[available_idx], lat_all[available_idx], k, rng=rng, top_m=top_m
            )
            cal_idx = available_idx[cal_local]
            val_idx = np.setdiff1d(available_idx, cal_idx, assume_unique=False)

            # Save export plan for replicate #1 @ 60%
            if rep == 1 and abs(frac - 0.60) < 1e-6 and export_plan_60 is None:
                export_plan_60 = {"cal60_idx": cal_idx.copy(), "val_idx": val_idx.copy()}

            # ---------- Metrics for this replicate & density ----------
            # IDW baseline (Δh_vis only)
            pts_idw = pd.DataFrame({
                ID_COL: [ids_all[i] for i in np.r_[cal_idx, val_idx]],
                LON_COL: lon_all[np.r_[cal_idx, val_idx]],
                LAT_COL: lat_all[np.r_[cal_idx, val_idx]],
                "insar_cm": np.zeros(len(cal_idx) + len(val_idx), dtype=float),
                "dh_cm": dh_all[np.r_[cal_idx, val_idx]],
            })
            _, m_idw = _eval_ls_and_idw_for_split(
                pts_idw,
                np.arange(len(cal_idx)),
                np.arange(len(cal_idx), len(cal_idx)+len(val_idx)),
                idw_power=idw_power
            )
            cal_pct = int(round(frac * 100))
            density_tag = f"{cal_pct}pct"

            records.append({
                "area": area_name, "pair_ref": ref_iso, "pair_sec": sec_iso,
                "dem": "N/A", "corr": "N/A", "method": "idw_dhvis",
                "replicate": rep, "n_total": N, "n_cal": int(len(cal_idx)), "n_val": int(len(val_idx)),
                "cal_frac": float(frac), "cal_pct": cal_pct, "density_tag": density_tag,
                **m_idw, "a_gain": np.nan, "b_offset_cm": np.nan
            })

            # LS metrics per raster (dem,corr)
            for (dem, corr), insar_vals in insar_by_key.items():
                vals_stacked = insar_vals[np.r_[cal_idx, val_idx]]
                pts_ls = pd.DataFrame({
                    ID_COL: [ids_all[i] for i in np.r_[cal_idx, val_idx]],
                    LON_COL: lon_all[np.r_[cal_idx, val_idx]],
                    LAT_COL: lat_all[np.r_[cal_idx, val_idx]],
                    "insar_cm": vals_stacked,
                    "dh_cm": dh_all[np.r_[cal_idx, val_idx]],
                })
                m_ls, _ = _eval_ls_and_idw_for_split(
                    pts_ls,
                    np.arange(len(cal_idx)),
                    np.arange(len(cal_idx), len(cal_idx) + len(val_idx)),
                    idw_power=idw_power
                )
                records.append({
                    "area": area_name, "pair_ref": ref_iso, "pair_sec": sec_iso,
                    "dem": dem, "corr": corr, "method": "least_squares",
                    "replicate": rep, "n_total": N, "n_cal": int(len(cal_idx)), "n_val": int(len(val_idx)),
                    "cal_frac": float(frac), "cal_pct": cal_pct, "density_tag": density_tag,
                    **m_ls
                })

    # ------------------------ Exports from replicate #1 @ 60% ------------------------
    if export_plan_60 is None:
        print(f"  ⚠️  No export plan available for {pair_tag} @ 60%; skipping exports.")
        return pd.DataFrame.from_records(records) if records else pd.DataFrame()

    cal60 = export_plan_60["cal60_idx"]; val60 = export_plan_60["val_idx"]

    # 1) IDW(Δh_vis) 60% on a reference grid
    try:
        grid_tif = next(iter(rasters.values()))
        px = lon_all[cal60]; py = lat_all[cal60]; pz = dh_all[cal60]
        idw_grid = _make_idw_grid_on_raster(px, py, pz, ref_tif=grid_tif, power=idw_power)
        out_idw = results_dir / f"idw60_{pair_tag}.tif"
        _write_tif_like(grid_tif, out_idw, idw_grid)
        print(f"  🗺️  IDW Δh_vis (60%) written: {out_idw}")
    except Exception as e:
        print(f"  ⚠️  IDW export failed for {pair_tag}: {e}")

    # 2) 60% calibrated rasters for EVERY available (DEM, CORR)
    try:
        for (dem, corr), tif in rasters.items():
            insar_vals = insar_by_key[(dem, corr)]
            a60, b60 = _fit_ls_params(insar_vals[cal60], dh_all[cal60])
            arr60 = _apply_calibration_to_raster(tif, a60, b60)
            out_cal60 = results_dir / f"cal_60pct_{dem}_{corr}_{pair_tag}.tif"
            _write_tif_like(tif, out_cal60, arr60)
        print(f"  🗺️  Calibrated 60% rasters written for {len(rasters)} variants.")
    except Exception as e:
        print(f"  ⚠️  60% calibrated exports failed for {pair_tag}: {e}")

    # 3) Split GeoJSON: 60% cal vs 40% val (replicate #1)
    try:
        out_geojson = results_dir / f"split_60_40_{pair_tag}.geojson"
        _write_split_geojson(out_geojson, ids_all, lon_all, lat_all, cal60, val60,
                             area=area_name, pair_tag=pair_tag, ref_iso=ref_iso, sec_iso=sec_iso, replicate=1)
        print(f"  🧭  Split GeoJSON written: {out_geojson}")
    except Exception as e:
        print(f"  ⚠️  Split GeoJSON export failed for {pair_tag}: {e}")

    return pd.DataFrame.from_records(records)

# =============================== Driver (AREA) ===============================
def _process_area(area_dir: Path, dems: List[str], corrs: List[str],
                  reps: int, seed: int, idw_power: float) -> None:
    """Process a single area end-to-end and write a fresh metrics CSV.

    Parameters
    ----------
    area_dir : pathlib.Path
        Path to the area folder under ``--areas-root``.
    dems, corrs : list[str]
        DEM and correction names to consider.
    reps : int
        Number of replicates per pair.
    seed : int
        RNG seed for reproducibility of split plans.
    idw_power : float
        Power exponent for IDW baseline.

    Side Effects
    ------------
    Writes ``accuracy_metrics.csv`` under ``<AREA>/results`` and exports rasters
    and GeoJSON for replicate #1 @ 60%.
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
            df_pair = _evaluate_pair_spread_multidensity_and_exports(
                area_dir=area_dir,
                area_name=area_name,
                pair_tag=pair_tag,
                gauge_csv=gauge_csv,
                rasters=rasters,
                n_repl=reps,
                seed=seed,
                idw_power=idw_power,
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
    print(f"\n✅ [{area_name}] metrics written (fresh, multi-density): {metrics_csv}  (rows: {len(df_all)})")

# =================================== CLI ===================================
def main():
    """CLI entry point to run accuracy assessment and exports.

    Arguments
    ---------
    --areas-root : str (default: /mnt/DATA2/bakke326l/processing/areas)
    --area : str            # process only this AREA
    --reps : int            # replicates per pair (default 200)
    --seed : int            # RNG seed (default 42)
    --idw-power : float     # IDW power (default 2.0)
    --dems : list[str]      # DEMs to include (default: SRTM 3DEP)
    --corrs : list[str]     # Corrections to include (default: RAW TROPO IONO TROPO_IONO)

    Behavior
    --------
    1) Cleans all <AREA>/results folders under --areas-root.
    2) Iterates areas, runs multi-density (60/45/30/15%) spread-out evaluation & exports,
       then writes a fresh metrics CSV per area.
    """
    ap = argparse.ArgumentParser(
        description="Compute accuracy metrics with SHARED spread-out 60/45/30/15% calibration splits per (AREA,PAIR); export 60% IDW + calibrated GeoTIFFs and a 60/40 split GeoJSON from replicate #1."
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
    ap.add_argument("--dems", nargs="+", default=DEMS_DEFAULT,
                    help="DEMs to consider (default: %(default)s)")
    ap.add_argument("--corrs", nargs="+", default=CORRS_DEFAULT,
                    help="Corrections to consider (default: %(default)s)")
    args = ap.parse_args()

    root = Path(args.areas_root)

    # ---------- Upfront cleanup of ALL <AREA>/results folders ----------
    print(f"🧹 Cleaning all '<AREA>/results' folders under: {root}")
    _clean_all_results_dirs(root)
    # -------------------------------------------------------------------

    targets = [root / args.area] if args.area else sorted([d for d in root.iterdir() if d.is_dir()])

    for area_dir in targets:
        _process_area(
            area_dir=area_dir,
            dems=[d.upper() for d in args.dems],
            corrs=[c.upper() for c in args.corrs],
            reps=args.reps,
            seed=args.seed,
            idw_power=args.idw_power,
        )

if __name__ == "__main__":
    main()
