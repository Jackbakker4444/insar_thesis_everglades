#!/usr/bin/env python3
"""
5_accuracy_assessment_dem_corr.py — Shared-split accuracy metrics + 60% calibrated GeoTIFFs
(With an upfront cleanup of all <AREA>/results/ folders)

Overview
--------
This script evaluates InSAR vertical-displacement rasters (cm) against EDEN water-level
gauges using a **shared calibration/validation split** across all DEM/CORR variants for
each (AREA, PAIR). It produces per-gauge accuracy curves for:

  • `least_squares`: affine calibration of InSAR → visible water-level change (Δh_vis),
  • `idw_dhvis`:     gauge-only baseline via inverse-distance weighting of Δh_vis.

From **replicate #1**, it also exports:
  • an IDW(Δh_vis) grid from the 60% calibration set:  `idw60_<REF>_<SEC>.tif`
  • **60% calibrated** rasters for every available (DEM, CORR) with
    CORR ∈ {RAW, TROPO, IONO, TROPO_IONO}:
      `cal_60pct_<DEM>_<CORR>_<REF>_<SEC>.tif`

⚠️ Intentionally omitted: target-density and center-only calibrated exports, and any
DEM-only calibrated export that you previously called “calibrated DEM”.

Upfront cleanup
---------------
At startup the script **empties every `<AREA>/results/` folder** under `--areas-root`
(creates it if missing). This ensures a clean slate for metrics and GeoTIFF outputs.

Core definitions & equations
----------------------------
**Visible water-surface change** at gauges:
\\[
\\Delta h_{\\mathrm{vis}}[\\mathrm{cm}] = \\max(\\mathrm{sec}, 0) - \\max(\\mathrm{ref}, 0)
\\]

**Affine calibration** (least squares) fits:
\\[
y \\approx a\\,x + b
\\]
where \\(x\\) is InSAR (cm) sampled at gauges and \\(y = \\Delta h_{\\mathrm{vis}}\\) (cm).
When the calibration set has ≤ 2 gauges, we **fix** \\(a=-1\\) and choose
\\[
b = \\mathrm{mean}(y + x)
\\]
which minimizes \\(\\sum(y_i - (-1)x_i - b)^2\\).

**IDW baseline** predicts Δh_vis at a query location \\(q\\) from calibration gauges
\\((p_{x,i},p_{y,i}, z_i)\\) with weights \\(w_i \\propto d_i^{-p}\\) (power \\(p>0\\)):
\\[
\\hat{z}(q) = \\frac{\\sum_i w_i z_i}{\\sum_i w_i}, \\quad
w_i = d_i^{-p}, \\quad
d_i^2 \\approx \\big((q_x - p_{x,i})\\cos\\bar{\\varphi}\\big)^2 + (q_y - p_{y,i})^2
\\]
where \\(\\bar{\\varphi}\\) is the mean latitude of the calibration gauges (degrees).

**Metrics** (on validation gauges):
\\[
\\begin{aligned}
\\mathrm{bias} &= \\frac{1}{n}\\sum ( \\hat{y}-y ) \\\\
\\mathrm{RMSE} &= \\sqrt{\\frac{1}{n}\\sum ( \\hat{y}-y )^2} \\\\
\\mathrm{MAE}  &= \\frac{1}{n}\\sum | \\hat{y}-y | \\\\
\\sigma_e      &= \\sqrt{\\frac{1}{n}\\sum \\big((\\hat{y}-y)-\\mathrm{bias}\\big)^2} \\\\
r             &= \\frac{\\sum (\\hat{y}-\\bar{\\hat{y}})(y-\\bar{y})}
                      {\\sqrt{\\sum (\\hat{y}-\\bar{\\hat{y}})^2 \\sum (y-\\bar{y})^2}}
\\end{aligned}
\\]

Shared-split design
-------------------
For each pair:
1) Keep only gauges that have **valid samples in every raster** (shared mask).
2) Replicate plan per pair:
   - ~60% calibration via farthest-point sampling (excluding the center gauge),
   - fixed validation = remaining gauges,
   - iterative “crowding” drop to trace the density curve (down to 2 gauges),
   - (metrics also reported for center-only; **exports** use only the 60% plan).

Outputs
-------
Per AREA:
  <areas_root>/<AREA>/results/accuracy_metrics.csv

Per AREA & PAIR (from replicate #1 plan):
  <areas_root>/<AREA>/results/
    idw60_<REF>_<SEC>.tif
    cal_60pct_<DEM>_<CORR>_<REF>_<SEC>.tif

How to run
----------
Process all areas:
    python 5_accuracy_assessment_dem_corr.py
One area (e.g., ENP):
    python 5_accuracy_assessment_dem_corr.py --area ENP
Choose DEMs/CORRs and params:
    python 5_accuracy_assessment_dem_corr.py \
      --reps 50 --seed 42 --idw-power 2.0 \
      --dems SRTM 3DEP --corrs RAW TROPO IONO TROPO_IONO

Assumptions
-----------
- Interferograms are single-band float (cm), EPSG:4326, with nodata/NaN.
- Gauges file: StationID, Lat, Lon, plus daily 'YYYY-MM-DD' columns including REF & SEC.
"""

from __future__ import annotations
from pathlib import Path
import re, os, shutil, logging, argparse
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
REPS_DEFAULT            = 50
SEED_DEFAULT            = 42
IDW_POWER_DEFAULT       = 2.0
OUTPUT_DENSITY_DEFAULT  = 500.0  # retained for CLI compatibility (not used by 60% exports)

# Gauge columns
ID_COL, LAT_COL, LON_COL = "StationID", "Lat", "Lon"

# Geodesy helper for polygon areas
GEOD = Geod(ellps="WGS84")


# ===================== Upfront cleanup helpers (results folders) =====================
def _clean_all_results_dirs(areas_root: Path) -> None:
    """
    Remove all contents inside every '<AREA>/results' directory under `areas_root`,
    then recreate an empty 'results' directory if missing.

    Notes
    -----
    - This is intentionally conservative: it only touches '<AREA>/results', not other folders.
    - Use with caution: previous runs' outputs in 'results' will be deleted.
    """
    if not areas_root.exists():
        return
    for d in sorted(p for p in areas_root.iterdir() if p.is_dir()):
        res = d / "results"
        try:
            if res.exists():
                # Delete *everything* under results (files + subfolders)
                for item in res.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        # missing_ok avoids raising if file disappears mid-loop
                        item.unlink(missing_ok=True)
            else:
                # Create 'results' if the area is new or missing the folder
                res.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️  Could not clean results in {res}: {e}")


# ========================= Small utilities =========================
def _pair_dates_from_tag(pair_tag: str) -> Tuple[str, str]:
    """Convert 'YYYYMMDD_YYYYMMDD' → ('YYYY-MM-DD','YYYY-MM-DD')."""
    if not re.fullmatch(r"\d{8}_\d{8}", pair_tag):
        raise ValueError(f"PAIR tag must be YYYYMMDD_YYYYMMDD, got: {pair_tag}")
    a, b = pair_tag.split("_")
    return f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"


def _find_all_pairs(area_dir: Path, area_name: str, dems: List[str], corrs: List[str]) -> List[str]:
    """
    Scan <AREA>/interferograms for available pair tags restricted to selected DEM/CORR.

    Returns
    -------
    list[str]
        Unique pair tags 'YYYYMMDD_YYYYMMDD' present for any selected DEM/CORR.
    """
    dem_pat = "|".join(map(re.escape, dems))
    corr_pat = "|".join(map(re.escape, corrs))
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
    """Return the path to an interferogram raster if present, else None."""
    cand = area_dir / "interferograms" / f"{area_name}_vertical_cm_{pair_tag}_{dem.upper()}_{corr.upper()}.tif"
    return cand if cand.exists() else None


def _load_gauges_wide(csv_path: Path) -> pd.DataFrame:
    """
    Load an area's wide EDEN gauge CSV and validate required columns.

    Requires
    --------
    Columns: 'StationID', 'Lat', 'Lon' + date columns 'YYYY-MM-DD'.
    """
    df = pd.read_csv(csv_path)
    for c in (ID_COL, LAT_COL, LON_COL):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    return df


def _visible_surface_delta(ref_cm: np.ndarray, sec_cm: np.ndarray) -> np.ndarray:
    r"""
    Compute visible water-surface change Δh_vis (cm):
        Δh_vis = max(sec, 0) − max(ref, 0)
    """
    return np.maximum(sec_cm.astype(float), 0.0) - np.maximum(ref_cm.astype(float), 0.0)


def _rowcol_from_xy(transform: Affine, x: float, y: float) -> Tuple[float, float]:
    """Map world coordinates (lon,lat) to fractional raster (row,col)."""
    col, row = ~transform * (x, y)
    return float(row), float(col)


def _inside_image(h: int, w: int, row: float, col: float) -> bool:
    """Check if a raster (row,col) is inside the image bounds."""
    return (row >= 0) and (col >= 0) and (row < h) and (col < w)


def _read_mean_3x3(ds: rasterio.io.DatasetReader, row: int, col: int) -> Optional[float]:
    """
    Read a 3×3 window around (row,col) and return the NaN-mean.
    Returns None if the window is entirely nodata/NaN.
    """
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
    Compute valid-data surface area (km²) from **finite (non-NaN) pixel** polygons.

    Assumes
    -------
    Raster is EPSG:4326.
    """
    if ds.crs is None or ds.crs.to_epsg() != 4326:
        raise RuntimeError("Expected EPSG:4326 raster.")
    arr = ds.read(1).astype("float32")
    if ds.nodata is not None and not np.isnan(ds.nodata):
        arr[arr == ds.nodata] = np.nan
    mask = np.isfinite(arr).astype(np.uint8)  # 1 where finite, 0 elsewhere
    area = 0.0
    for geom, val in shapes(mask, transform=ds.transform):
        if val == 1:
            area += _geod_area_of_geojson(geom)
    return float(area)


def _safe_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson correlation coefficient (r) that returns NaN when undefined.

    Undefined cases
    ---------------
    - n < 2 samples
    - zero variance in y_true or y_pred
    """
    if len(y_true) < 2: return float("nan")
    yt = y_true - np.mean(y_true); yp = y_pred - np.mean(y_pred)
    vy = np.sum(yt*yt); vp = np.sum(yp*yp)
    if vy <= 1e-12 or vp <= 1e-12: return float("nan")
    return float(np.sum(yt*yp) / np.sqrt(vy*vp))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    r"""
    Compute accuracy metrics on the validation set.

    Equations
    ---------
        bias  = mean( e )
        RMSE  = sqrt( mean( e^2 ) )
        MAE   = mean( |e| )
        σ_e   = sqrt( mean( (e-bias)^2 ) )
        r     = Pearson correlation (NaN if undefined)

    where e = y_pred − y_true.
    """
    err = y_pred - y_true
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    sigma_e = float(np.sqrt(np.mean((err - bias)**2)))
    return {"rmse_cm": rmse, "mae_cm": mae, "bias_cm": bias, "sigma_e_cm": sigma_e, "r": _safe_corrcoef(y_true, y_pred)}


# --------------- validation spread & metric augmentation helpers ---------------
def _val_spread(y_val: np.ndarray) -> Tuple[float, float]:
    """Return (IQR_cm, SD_cm) of validation Δh_vis for normalization & diagnostics."""
    yv = np.asarray(y_val, dtype=float)
    if yv.size == 0 or not np.isfinite(yv).any():
        return float("nan"), float("nan")
    q25, q75 = np.nanpercentile(yv, [25, 75])
    iqr = float(q75 - q25)
    sd  = float(np.nanstd(yv))
    return iqr, sd


def _augment_metrics(m: Dict[str, float], y_val: np.ndarray) -> Dict[str, float]:
    """
    Append derived fields to a metrics dict:
      - val_dh_iqr_cm, val_dh_sd_cm
      - nrmse_iqr, nrmse_sd
      - log_rmse_cm
      - fisher_z (atanh(r)) for variance stabilization
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
# ------------------------------------------------------------------------------


def _idw_predict_points(px, py, pz, qx, qy, power: float = 2.0) -> np.ndarray:
    r"""
    Inverse-distance weighting in lon/lat with cosine adjustment on Δlon.

    Weights
    -------
        w_i = d_i^{-p},  p>0

    Distance (approx, degrees scaled)
    ---------------------------------
        d_i^2 ≈ ((Δlon)·cos(φ̄))^2 + (Δlat)^2

    Notes
    -----
    - Uses a small floor on distance to avoid division-by-zero.
    - Snaps exactly coincident query points to the known value.
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
    """Great-circle distance matrix (meters) for lon/lat arrays (symmetric NxN)."""
    lon = np.deg2rad(lon.astype("float64")); lat = np.deg2rad(lat.astype("float64"))
    dlon = lon[:, None] - lon[None, :]
    a = np.clip(np.sin(lat)[:,None]*np.sin(lat)[None,:] + np.cos(lat)[:,None]*np.cos(lat)[None,:]*np.cos(dlon), -1.0, 1.0)
    return 6371000.0 * np.arccos(a)


def _spread_selection(lon: np.ndarray, lat: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Farthest-point sampling to select k well-spread indices.

    Heuristic
    ---------
    - Pick a random seed point.
    - Repeatedly add the point farthest from the current set (max of min-distance).
    """
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
    keep_global : int
        The global index that should *not* be removed (e.g., the center gauge).
    """
    if len(idx) <= 1: return idx
    lon_s, lat_s = lon[idx], lat[idx]
    D = _haversine_matrix(lon_s, lat_s)
    np.fill_diagonal(D, np.inf)
    nnd = np.min(D, axis=1)
    order = np.argsort(nnd)  # smallest NN distance -> most crowded
    order = np.array([o for o in order if idx[o] != keep_global], dtype=int)
    if order.size == 0: return np.array([], dtype=int)
    return order[:min(top_n, order.size)]


# ======================== Core evaluation logic ========================
def _fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    r"""
    Ordinary least squares fit for y ≈ a·x + b.

    Closed form
    -----------
        [a, b]^T = argmin_{a,b} ||y - a x - b||²
    """
    A = np.c_[x, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def _eval_ls_and_idw(pts: pd.DataFrame, cal_idx: np.ndarray, val_idx: np.ndarray, idw_power: float) -> Dict[str, Dict[str, float]]:
    """
    Evaluate least-squares calibration and IDW(interpolated Δh_vis) on validation gauges.

    LS
    --
    Fits y ≈ a·x + b on calibration gauges (fix a=-1, b=mean(y+x) when n_cal ≤ 2),
    then evaluates metrics on validation gauges.

    IDW
    ---
    Ignores InSAR and interpolates Δh_vis from calibration gauges to validation gauges.
    """
    # Slice calibration/validation arrays
    x_cal = pts["insar_cm"].values[cal_idx].astype(float)
    y_cal = pts["dh_cm"].values[cal_idx].astype(float)
    x_val = pts["insar_cm"].values[val_idx].astype(float)
    y_val = pts["dh_cm"].values[val_idx].astype(float)

    # Robust small-sample fallback for LS
    n_unique = len(np.unique(pts[ID_COL].values[cal_idx]))
    if len(cal_idx) <= 2 and n_unique >= 1:
        a = -1.0
        b = float(np.mean(y_cal + x_cal))
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
        m_ls = {"rmse_cm": np.nan, "mae_cm": np.nan, "bias_cm": np.nan, "sigma_e_cm": np.nan,
                "r": np.nan, "a_gain": np.nan, "b_offset_cm": np.nan}

    # Predict & score IDW baseline on validation gauges
    px = pts[LON_COL].values[cal_idx].astype(float)
    py = pts[LAT_COL].values[cal_idx].astype(float)
    pz = pts["dh_cm"].values[cal_idx].astype(float)
    qx = pts[LON_COL].values[val_idx].astype(float)
    qy = pts[LAT_COL].values[val_idx].astype(float)
    y_pred_idw = _idw_predict_points(px, py, pz, qx, qy, power=idw_power)
    m_idw = _compute_metrics(y_true=y_val, y_pred=y_pred_idw)

    # Add normalized & transformed metrics
    return {"least_squares": _augment_metrics(m_ls, y_val),
            "idw_dhvis"    : _augment_metrics(m_idw, y_val)}


# ===================== Export helpers (GeoTIFF writing) =====================
def _fit_ls_params(insar_vals: np.ndarray, dh_vals: np.ndarray) -> Tuple[float, float]:
    r"""
    Fit y = a·x + b with fallback when n ≤ 2:
        a = -1,  b = mean(dh + insar)
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

    Notes
    -----
    - NaNs in `array2d` are replaced by `nodata_value`.
    - Compression uses DEFLATE with Predictor=3 (float-friendly).
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
    """
    Generate an IDW(Δh_vis) surface on the pixel grid of `ref_tif` (EPSG:4326),
    masked by the finite-pixel mask of the reference raster.

    Implementation details
    ----------------------
    - Computes lon/lat per column/row via the affine transform (north-up assumed).
    - Does the IDW row-by-row to keep memory usage modest on large rasters.
    """
    with rasterio.open(ref_tif) as ds:
        H, W = ds.height, ds.width
        transform = ds.transform
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise RuntimeError(f"Expected EPSG:4326 grid for IDW export: {ref_tif}")
        base = ds.read(1).astype("float32")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            base[base == ds.nodata] = np.nan

        # Pixel-center coordinates for each row/col
        cols = np.arange(W, dtype=np.float64) + 0.5
        rows = np.arange(H, dtype=np.float64) + 0.5
        a,b,c,d,e,f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        xs_cols = a*cols + c   # north-up: x depends on column only
        ys_rows = e*rows + f   # north-up: y depends on row only

        out = np.full((H, W), np.nan, dtype="float32")
        cx = np.cos(np.deg2rad(np.nanmean(py) if len(py) else 0.0))

        for r in range(H):
            # Vectorized distances for all columns in this row
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

        # Respect the reference raster's valid-data mask
        valid_mask = np.isfinite(base)
        out = np.where(valid_mask, out, np.nan)
        return out


def _apply_calibration_to_raster(src_tif: Path, a: float, b: float) -> np.ndarray:
    """
    Apply affine calibration y = a·x + b to a raster, honoring NaN/nodata.

    Returns
    -------
    np.ndarray
        Calibrated array with NaN outside valid data.
    """
    with rasterio.open(src_tif) as ds:
        arr = ds.read(1).astype("float32")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr = np.where(arr == ds.nodata, np.nan, arr)
        valid_mask = np.isfinite(arr)
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
) -> pd.DataFrame:
    """
    Evaluate one (AREA, PAIR) using a *shared* cal/val split and write:
      • idw60_<PAIR>.tif
      • cal_60pct_<DEM>_<CORR>_<PAIR>.tif for all present DEM/CORR rasters.

    Notes
    -----
    - The **shared split** guarantees each method sees the same gauges across DEM/CORR.
    - Exports are taken from replicate #1's 60% calibration plan.
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

    # Consistent ordering of common gauges and metadata frame
    common_ids_sorted = sorted(common_ids)
    any_df = next(iter(pts_by_raster.values()))
    meta = (any_df.set_index(ID_COL)
                 .loc[common_ids_sorted, [LON_COL, LAT_COL]]
                 .reset_index())

    # Aligned arrays for all rasters
    dh_all  = g.set_index(ID_COL).loc[common_ids_sorted, "dh_cm"].to_numpy(dtype=float)
    lon_all = meta[LON_COL].to_numpy(dtype=float)
    lat_all = meta[LAT_COL].to_numpy(dtype=float)

    insar_by_key: Dict[tuple, np.ndarray] = {}
    for key, df in pts_by_raster.items():
        insar_by_key[key] = (df.set_index(ID_COL)
                               .loc[common_ids_sorted, "insar_cm"]
                               .to_numpy(dtype=float))

    # ----- Shared center gauge (closest to centroid) -----
    lon_c, lat_c = float(lon_all.mean()), float(lat_all.mean())
    _, _, d_center = GEOD.inv(np.full_like(lon_all, lon_c), np.full_like(lat_all, lat_c), lon_all, lat_all)
    center_idx_global = int(np.argmin(d_center))

    N = len(common_ids_sorted)
    rng_master = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []

    # Capture replicate #1’s plan for the exports (60% plan)
    export_plan = None  # dict with cal60_idx, val_idx, cal_seq (list of n_cal->idx)

    for rep in range(1, n_repl + 1):
        rng = np.random.default_rng(rng_master.integers(0, 2**31-1))

        # All indices, excluding center gauge for the sweep
        all_idx = np.arange(N, dtype=int)
        available_idx = np.setdiff1d(all_idx, np.array([center_idx_global]), assume_unique=False)

        # Initial ~60% calibration (well spread), remainder = validation
        n_cal0 = max(1, int(round(0.60 * len(available_idx))))
        n_cal0 = min(n_cal0, len(available_idx))
        cal_local = _spread_selection(lon_all[available_idx], lat_all[available_idx], n_cal0, rng=rng)
        cal_idx = available_idx[cal_local]
        val_idx = np.setdiff1d(available_idx, cal_idx, assume_unique=False)

        # Ensure validation not empty (rare edge-case for tiny N)
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
            # Add final center-only step for completeness (not used for exports here)
            cal_seq.append(np.array([center_idx_global], dtype=int))
            export_plan = {"cal60_idx": cal_idx.copy(), "val_idx": val_idx.copy(), "cal_seq": cal_seq}

        # March down to 2 cal gauges (metrics for each raster + IDW baseline)
        cur_idx = cal_idx.copy()
        while len(cur_idx) >= 2:
            # IDW baseline metrics (single method; uses only gauge lon/lat + Δh_vis)
            pts_idw = pd.DataFrame({
                ID_COL: [common_ids_sorted[i] for i in np.r_[cur_idx, val_idx]],
                LON_COL: lon_all[np.r_[cur_idx, val_idx]],
                LAT_COL: lat_all[np.r_[cur_idx, val_idx]],
                "insar_cm": np.zeros(len(cur_idx) + len(val_idx), dtype=float),
                "dh_cm": dh_all[np.r_[cur_idx, val_idx]],
            })
            mm_idw = _eval_ls_and_idw(
                pts_idw,
                np.arange(len(cur_idx)),
                np.arange(len(cur_idx), len(cur_idx) + len(val_idx)),
                idw_power=idw_power
            )
            records.append({
                "area": area_name, "pair_ref": ref_iso, "pair_sec": sec_iso,
                "dem": "N/A", "corr": "N/A", "method": "idw_dhvis",
                "replicate": rep, "n_total": N, "n_cal": int(len(cur_idx)), "n_val": int(len(val_idx)),
                **mm_idw["idw_dhvis"], "a_gain": np.nan, "b_offset_cm": np.nan
            })

            # LS metrics: per raster (dem,corr) using the same shared split
            for (dem, corr), insar_vals in insar_by_key.items():
                vals_stacked = insar_vals[np.r_[cur_idx, val_idx]]
                pts_ls = pd.DataFrame({
                    ID_COL: [common_ids_sorted[i] for i in np.r_[cur_idx, val_idx]],
                    LON_COL: lon_all[np.r_[cur_idx, val_idx]],
                    LAT_COL: lat_all[np.r_[cur_idx, val_idx]],
                    "insar_cm": vals_stacked,
                    "dh_cm": dh_all[np.r_[cur_idx, val_idx]],
                })
                mm_ls = _eval_ls_and_idw(
                    pts_ls,
                    np.arange(len(cur_idx)),
                    np.arange(len(cur_idx), len(cur_idx) + len(val_idx)),
                    idw_power=idw_power
                )
                records.append({
                    "area": area_name, "pair_ref": ref_iso, "pair_sec": sec_iso,
                    "dem": dem, "corr": corr, "method": "least_squares",
                    "replicate": rep, "n_total": N, "n_cal": int(len(cur_idx)), "n_val": int(len(val_idx)),
                    **mm_ls["least_squares"]
                })

            # Drop one crowded calibration gauge and continue the density sweep
            crowded = _crowded_candidates(lon_all, lat_all, cur_idx, keep_global=center_idx_global, top_n=4)
            drop_pos = int(rng.choice(crowded)) if crowded.size else 0
            cur_idx = np.delete(cur_idx, drop_pos)

        # Final single-gauge (center) metrics
        for (dem, corr), insar_vals in insar_by_key.items():
            vals_stacked = insar_vals[np.r_[ [center_idx_global], val_idx ]]
            pts_ls = pd.DataFrame({
                ID_COL: [common_ids_sorted[i] for i in np.r_[ [center_idx_global], val_idx ]],
                LON_COL: lon_all[np.r_[ [center_idx_global], val_idx ]],
                LAT_COL: lat_all[np.r_[ [center_idx_global], val_idx ]],
                "insar_cm": vals_stacked,
                "dh_cm": dh_all[np.r_[ [center_idx_global], val_idx ]],
            })
            mm_ls = _eval_ls_and_idw(pts_ls, np.array([0], dtype=int), np.arange(1, pts_ls.shape[0]), idw_power=idw_power)
            records.append({
                "area": area_name, "pair_ref": ref_iso, "pair_sec": sec_iso,
                "dem": dem, "corr": corr, "method": "least_squares",
                "replicate": rep, "n_total": N, "n_cal": 1, "n_val": int(len(val_idx)),
                **mm_ls["least_squares"]
            })

    # ------------------------ Per-pair 60% GeoTIFF exports ------------------------
    # 1) IDW(Δh_vis) from first 60% calibration — choose any raster grid as reference
    try:
        grid_tif = next(iter(rasters.values()))  # consistent grid; first available raster
        cal60 = export_plan["cal60_idx"]
        px = lon_all[cal60]; py = lat_all[cal60]; pz = dh_all[cal60]
        idw_grid = _make_idw_grid_on_raster(px, py, pz, ref_tif=grid_tif, power=idw_power)
        out_idw = results_dir / f"idw60_{pair_tag}.tif"
        _write_tif_like(grid_tif, out_idw, idw_grid)
        print(f"  🗺️  IDW Δh_vis (60%) written: {out_idw}")
    except Exception as e:
        print(f"  ⚠️  IDW export failed for {pair_tag}: {e}")

    # 2) 60% calibrated rasters for EVERY available (DEM, CORR)
    try:
        cal60_idx = export_plan["cal60_idx"]
        for (dem, corr), tif in rasters.items():
            insar_vals = insar_by_key[(dem, corr)]
            a60, b60 = _fit_ls_params(insar_vals[cal60_idx], dh_all[cal60_idx])
            arr60 = _apply_calibration_to_raster(tif, a60, b60)
            out_cal60 = results_dir / f"cal_60pct_{dem}_{corr}_{pair_tag}.tif"
            _write_tif_like(tif, out_cal60, arr60)
        print(f"  🗺️  Calibrated 60% rasters written for {len(rasters)} variants.")
    except Exception as e:
        print(f"  ⚠️  60% calibrated exports failed for {pair_tag}: {e}")

    # --------------------------------------------------------------------------
    return pd.DataFrame.from_records(records)


# =============================== Driver (AREA) ===============================
def _process_area(area_dir: Path, dems: List[str], corrs: List[str],
                  reps: int, seed: int, idw_power: float) -> None:
    """
    Process a single AREA:
    - Discover pairs & rasters constrained by selected DEMS/CORRS.
    - Evaluate each pair with the shared-split pipeline.
    - Overwrite <AREA>/results/accuracy_metrics.csv with fresh results.
    - Write: idw60_<PAIR>.tif and cal_60pct_<DEM>_<CORR>_<PAIR>.tif (for all present rasters).
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
    --dems : list[str]      # DEMs to include (default: SRTM 3DEP)
    --corrs : list[str]     # CORRs to include (default: RAW TROPO IONO TROPO_IONO)

    Behavior
    --------
    1) Cleans all <AREA>/results folders under --areas-root.
    2) Iterates areas, runs evaluation & exports, and writes a fresh metrics CSV per area.
    """
    ap = argparse.ArgumentParser(
        description="Compute accuracy metrics with a SHARED calibration/validation split per (AREA,PAIR) and write IDW+60% calibrated GeoTIFFs."
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

    # If --area is provided, process only that subfolder; else all directories under root
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
