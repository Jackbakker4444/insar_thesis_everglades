#!/usr/bin/env python3
"""
7_accuracy_assessment_density.py — Single-raster density & accuracy assessment (SRTM+RAW only)

Overview
--------
This script evaluates, per area and per interferometric pair, a **single fixed raster
choice** — DEM = **SRTM** and correction = **RAW** — against EDEN water gauges to
produce accuracy-versus-density curves. Two accuracy baselines are computed **using
identical gauge sets and splits at every step**:

  • **RAW (least_squares)** — affine calibration of InSAR → Δh_vis on calibration gauges
  • **IDW (gauge-only baseline)** — inverse-distance interpolation of Δh_vis (no raster)

The workflow is:
  1) Randomly split usable gauges into **60% calibration** and **40% validation**.
     The validation set remains **fixed and independent** throughout.
  2) Perform a sweep in which the calibration set is reduced **one gauge at a time** by
     iteratively removing the **closest (most crowded) calibration gauge** based on
     nearest-neighbor distance within the current calibration set, down to **1 gauge**.
  3) At each step we log accuracy metrics and the implied **density** (km²/gauge), where
     the km² refers to the valid-data footprint of the SRTM+RAW raster.

Inputs & outputs
----------------
Per AREA directory (under --areas-root):
  Input
    • water_gauges/eden_gauges.csv (wide daily; must include the pair dates)
    • interferograms/<AREA>_vertical_cm_<REF>_<SEC>_SRTM_RAW.tif
  Output (in <AREA>/results)
    • accuracy_metrics_density_SRTM_RAW.csv
    • GeoTIFFs per pair:
        - dens_idw60_SRTM_RAW_<PAIR>.tif        (IDW Δh_vis from 60% cal gauges)
        - dens_cal_60pct_SRTM_RAW_<PAIR>.tif    (LS-calibrated raster using 60% gauges)
        - dens_cal_1g_SRTM_RAW_<PAIR>.tif       (LS-calibrated raster using 1 gauge)

Run examples
------------
All areas:
  python 7_accuracy_assessment_density.py
One area (e.g., ENP):
  python 7_accuracy_assessment_density.py --area ENP
Tuning:
  python 7_accuracy_assessment_density.py --reps 50 --seed 42 --idw-power 2.0 \
    --spread-top-m 5   # (deprecated; no effect)

Design guarantees
-----------------
• Pair discovery is **strict**: only files named
  <AREA>_vertical_cm_<REF>_<SEC>_SRTM_RAW.tif are used.
• The IDW baseline *always* uses the **same gauges** and **same splits** as LS:
  identical valid mask, identical cal/val indices at each step.
• IDW grid exports are masked by the SRTM+RAW raster's NaN mask.

Assumptions
-----------
• Raster is single-band float (cm), EPSG:4326, nodata set or NaN.
• Gauge CSV has columns StationID, Lat, Lon, and wide daily date headers ('YYYY-MM-DD').

Dependencies
------------
• Standard library: argparse, logging, os, re, pathlib
• Typing: typing (Tuple, Dict, Optional, List)
• Third-party: numpy, pandas, rasterio, pyproj (Geod)

Notes for reviewers
-------------------
• Initial split is **random 60/40 (cal/val)** to ensure independence of validation.
• The calibration sweep removes the **closest** (most crowded) gauge at each step.
• Randomness is controlled via --seed.
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
from pyproj import Geod

# ------------------------------ Logging hygiene ------------------------------
os.environ.setdefault("CPL_DEBUG", "NO")
for _n in ("rasterio", "rasterio._io", "rasterio.env", "rasterio._base"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# ------------------------------ Fixed setup ----------------------------------
AREAS_ROOT_DEFAULT      = Path("/mnt/DATA2/bakke326l/processing/areas")
DEM_FIXED               = "SRTM"
CORR_FIXED              = "RAW"

# Other defaults
REPS_DEFAULT            = 100
SEED_DEFAULT            = 42
IDW_POWER_DEFAULT       = 2.0

# Gauge columns
ID_COL, LAT_COL, LON_COL = "StationID", "Lat", "Lon"

# Geodesy
GEOD = Geod(ellps="WGS84")

# ============================== Small utilities ==============================
def _pair_dates_from_tag(pair_tag: str) -> Tuple[str, str]:
    """Parse a compact PAIR tag into ISO dates.

    Parameters
    ----------
    pair_tag : str
        String of the form 'YYYYMMDD_YYYYMMDD'.

    Returns
    -------
    (str, str)
        Tuple of ISO-8601 date strings ('YYYY-MM-DD', 'YYYY-MM-DD').

    Raises
    ------
    ValueError
        If the tag is not in the expected 8+8 digit format.
    """
    if not re.fullmatch(r"\d{8}_\d{8}", pair_tag):
        raise ValueError(f"PAIR tag must be YYYYMMDD_YYYYMMDD, got: {pair_tag}")
    a, b = pair_tag.split("_")
    return f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"

def _find_pairs_for_dem_corr(area_dir: Path, area_name: str, dem: str, corr: str) -> List[str]:
    """Discover all pair tags available for a fixed (DEM, CORR) within an AREA.

    Files must match the strict name pattern used by this project.
    """
    patt = re.compile(
        rf"^{re.escape(area_name)}_vertical_cm_(\d{{8}}_\d{{8}})_{re.escape(dem)}_{re.escape(corr)}\.tif$",
        re.I,
    )
    folder = area_dir / "interferograms"
    if not folder.exists():
        return []
    tags = []
    for p in folder.glob("*.tif"):
        m = patt.match(p.name)
        if m:
            tags.append(m.group(1))
    return sorted(set(tags))

def _raster_path(area_dir: Path, area_name: str, pair_tag: str, dem: str, corr: str) -> Optional[Path]:
    """Build the expected SRTM+RAW raster path for a given AREA and PAIR.

    Returns the path if it exists, else None.
    """
    cand = area_dir / "interferograms" / f"{area_name}_vertical_cm_{pair_tag}_{dem}_{corr}.tif"
    return cand if cand.exists() else None

def _load_gauges_wide(csv_path: Path) -> pd.DataFrame:
    """Load the per-area wide EDEN gauge table and validate required columns."""
    df = pd.read_csv(csv_path)
    for c in (ID_COL, LAT_COL, LON_COL):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    return df

def _visible_surface_delta(ref_cm: np.ndarray, sec_cm: np.ndarray) -> np.ndarray:
    """Compute Δh_vis (cm) = max(sec, 0) - max(ref, 0)."""
    return np.maximum(sec_cm.astype(float), 0.0) - np.maximum(ref_cm.astype(float), 0.0)

def _rowcol_from_xy(transform: Affine, x: float, y: float) -> Tuple[float, float]:
    """Map world coordinates (lon, lat) to fractional (row, col) in a raster grid."""
    col, row = ~transform * (x, y)
    return float(row), float(col)

def _inside_image(h: int, w: int, row: float, col: float) -> bool:
    """True if the given (row, col) falls inside the raster bounds."""
    return (row >= 0) and (col >= 0) and (row < h) and (col < w)

def _read_mean_3x3(ds: rasterio.io.DatasetReader, row: int, col: int) -> Optional[float]:
    """Read a 3×3 window around (row, col) and return the NaN-mean.

    Returns None if all pixels in the window are nodata/NaN.
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
    """Compute geodesic polygon area (km²) from a GeoJSON-like mapping.

    Handles Polygons and MultiPolygons; returns 0.0 for other types.
    """
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
    """Compute the valid-data footprint area (km²) from a raster’s mask (EPSG:4326)."""
    if ds.crs is None or ds.crs.to_epsg() != 4326:
        raise RuntimeError("Expected EPSG:4326 raster.")
    arr = ds.read(1).astype("float32")
    if ds.nodata is not None and not np.isnan(ds.nodata):
        arr[arr == ds.nodata] = np.nan
    mask = np.isfinite(arr).astype(np.uint8)
    area = 0.0
    for geom, val in shapes(mask, transform=ds.transform):
        if val == 1:
            area += _geod_area_of_geojson(geom)
    return float(area)

def _safe_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Numerically stable Pearson correlation (returns NaN when undefined)."""
    if len(y_true) < 2: return float("nan")
    yt = y_true - np.mean(y_true); yp = y_pred - np.mean(y_pred)
    vy = np.sum(yt*yt); vp = np.sum(yp*yp)
    if vy <= 1e-12 or vp <= 1e-12: return float("nan")
    return float(np.sum(yt*yp) / np.sqrt(vy*vp))

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute core accuracy metrics: RMSE, MAE, bias, σ_e, and r."""
    err = y_pred - y_true
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    sigma_e = float(np.sqrt(np.mean((err - bias)**2)))
    return {"rmse_cm": rmse, "mae_cm": mae, "bias_cm": bias, "sigma_e_cm": sigma_e, "r": _safe_corrcoef(y_true, y_pred)}

def _val_spread(y_val: np.ndarray) -> Tuple[float, float]:
    """Compute IQR and standard deviation of the validation Δh_vis vector."""
    yv = np.asarray(y_val, dtype=float)
    if yv.size == 0 or not np.isfinite(yv).any():
        return float("nan"), float("nan")
    q25, q75 = np.nanpercentile(yv, [25, 75])
    iqr = float(q75 - q25)
    sd  = float(np.nanstd(yv))
    return iqr, sd

def _augment_metrics(m: Dict[str, float], y_val: np.ndarray) -> Dict[str, float]:
    """Append derived metrics (NRMSE wrt IQR/SD, log RMSE, Fisher z)."""
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

# ----------------------------- IDW + selection ------------------------------
def _idw_predict_points(px, py, pz, qx, qy, power: float = 2.0) -> np.ndarray:
    """Inverse-distance weighting in lon/lat with cosine adjustment on Δlon."""
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
    """Pairwise great-circle distance matrix (meters) for lon/lat arrays."""
    lon = np.deg2rad(lon.astype("float64")); lat = np.deg2rad(lat.astype("float64"))
    dlon = lon[:, None] - lon[None, :]
    a = np.clip(np.sin(lat)[:,None]*np.sin(lat)[None,:] + np.cos(lat)[:,None]*np.cos(lat)[None,:]*np.cos(dlon), -1.0, 1.0)
    return 6371000.0 * np.arccos(a)

def _crowded_candidates(lon: np.ndarray, lat: np.ndarray, idx: np.ndarray,
                        top_n: int = 4) -> np.ndarray:
    """Return the indices (within idx) of the most spatially crowded calibration points.

    Used when marching down calibration size: we prefer to drop a point that is
    redundant (small nearest-neighbor distance).
    """
    if len(idx) <= 1: return idx
    lon_s, lat_s = lon[idx], lat[idx]
    D = _haversine_matrix(lon_s, lat_s)
    np.fill_diagonal(D, np.inf)
    nnd = np.min(D, axis=1)
    order = np.argsort(nnd)  # smallest NND -> most crowded
    if len(order) == 0: return np.array([], dtype=int)
    return order[:min(top_n, order.size)]

# ----------------------------- Model fitting --------------------------------
def _fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least-squares fit of y = a·x + b (returns a, b)."""
    A = np.c_[x, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])

def _fit_ls_params(insar_vals: np.ndarray, dh_vals: np.ndarray) -> Tuple[float, float]:
    """Fit affine parameters with a robust fallback for tiny calibration sets."""
    n = insar_vals.size
    if n <= 2:
        a = -1.0
        b = float(np.mean(dh_vals + insar_vals))
        return a, b
    return _fit_affine(insar_vals, dh_vals)

def _eval_ls_and_idw(pts: pd.DataFrame, cal_idx: np.ndarray, val_idx: np.ndarray, idw_power: float) -> Dict[str, Dict[str, float]]:
    """
    Run LS (on InSAR→Δh_vis) and IDW (on Δh_vis only) **using the same indices**.

    This guarantees that LS and IDW are evaluated on identical calibration and
    validation gauge sets for a fair comparison.
    Returns a dict keyed by method name → metrics dict.
    """
    x_cal = pts["insar_cm"].values[cal_idx].astype(float)
    y_cal = pts["dh_cm"].values[cal_idx].astype(float)
    x_val = pts["insar_cm"].values[val_idx].astype(float)
    y_val = pts["dh_cm"].values[val_idx].astype(float)

    n_unique = len(np.unique(pts.index.values[cal_idx]))
    if len(cal_idx) <= 2 and n_unique >= 1:
        a = -1.0
        b = float(np.mean(y_cal + x_cal))
    elif n_unique >= 2:
        a, b = _fit_affine(x_cal, y_cal)
    else:
        a, b = np.nan, np.nan

    if np.isfinite(a) and np.isfinite(b):
        y_pred_ls = a * x_val + b
        m_ls = _compute_metrics(y_true=y_val, y_pred=y_pred_ls)
        m_ls.update({"a_gain": float(a), "b_offset_cm": float(b)})
    else:
        m_ls = {"rmse_cm": np.nan, "mae_cm": np.nan, "bias_cm": np.nan, "sigma_e_cm": np.nan,
                "r": np.nan, "a_gain": np.nan, "b_offset_cm": np.nan}

    # IDW (same gauges)
    px = pts["Lon"].values[cal_idx].astype(float)
    py = pts["Lat"].values[cal_idx].astype(float)
    pz = pts["dh_cm"].values[cal_idx].astype(float)
    qx = pts["Lon"].values[val_idx].astype(float)
    qy = pts["Lat"].values[val_idx].astype(float)
    y_pred_idw = _idw_predict_points(px, py, pz, qx, qy, power=idw_power)
    m_idw = _compute_metrics(y_true=y_val, y_pred=y_pred_idw)

    return {"least_squares": _augment_metrics(m_ls, y_val),
            "idw_dhvis"    : _augment_metrics(m_idw, y_val)}

# ---------------------------- GeoTIFF writers -------------------------------
def _write_tif_like(src_tif: Path, out_tif: Path, array2d: np.ndarray, nodata_value: float = -9999.0):
    """Write a float32 GeoTIFF mirroring the spatial profile of an existing raster."""
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
    Build an IDW Δh_vis grid masked by the SAME valid pixels as the SRTM+RAW raster.

    px/py/pz must come from the RAW run's calibration indices to preserve consistency.
    """
    with rasterio.open(ref_tif) as ds:
        H, W = ds.height, ds.width
        transform = ds.transform
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise RuntimeError(f"Expected EPSG:4326 grid for IDW export: {ref_tif}")
        base = ds.read(1).astype("float32")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            base[base == ds.nodata] = np.nan

        cols = np.arange(W, dtype=np.float64) + 0.5
        rows = np.arange(H, dtype=np.float64) + 0.5
        a,b,c,d,e,f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        xs_cols = a*cols + c
        ys_rows = e*rows + f

        out = np.full((H, W), np.nan, dtype="float32")
        cx = np.cos(np.deg2rad(np.nanmean(py) if len(py) else 0.0))
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

        valid_mask = np.isfinite(base)
        out = np.where(valid_mask, out, np.nan)
        return out

def _apply_calibration_to_raster(src_tif: Path, a: float, b: float) -> np.ndarray:
    """Apply the affine LS calibration y = a·x + b to an entire raster (NaNs honored)."""
    with rasterio.open(src_tif) as ds:
        arr = ds.read(1).astype("float32")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr = np.where(arr == ds.nodata, np.nan, arr)
        valid_mask = np.isfinite(arr)
        out = np.full_like(arr, np.nan, dtype="float32")
        out[valid_mask] = a * arr[valid_mask] + b
        return out

# ============================= Core per-pair run =============================
def _evaluate_pair_single_raster_and_exports(
    area_dir: Path,
    area_name: str,
    pair_tag: str,
    gauge_csv: Path,        # wide EDEN CSV
    raster_tif: Path,        # the single chosen raster (DEM+CORR) => SRTM+RAW
    dem: str,
    corr: str,
    n_repl: int,
    seed: int,
    idw_power: float,
) -> pd.DataFrame:
    """Evaluate one (AREA, PAIR) for the fixed SRTM+RAW raster and export artifacts.

    The initial calibration set is a **random 60%** split of usable gauges; the
    remaining **40%** are held out for validation and remain fixed. We then
    iteratively **remove** the **closest (most crowded)** calibration gauge to
    generate a full accuracy-density trajectory down to **1 gauge**. At each step,
    **LS and IDW** are computed on the same split.
    """

    results_dir = area_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Gauges and Δh_vis ----
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

    # ---- Sample raster at gauge points; compute valid area ----
    with rasterio.open(raster_tif) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise RuntimeError(f"Expected EPSG:4326 raster: {raster_tif}")
        area_km2 = _valid_raster_area_km2(ds)

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
        print(f"  ⚠️  Skipping {area_name} {pair_tag}: no usable gauges in selected raster.")
        return pd.DataFrame()

    pts = pd.DataFrame(rows).set_index(ID_COL)
    common_ids_sorted = list(pts.index.astype(str))
    N = len(common_ids_sorted)
    if N < 3:
        print(f"  ⚠️  Skipping {area_name} {pair_tag}: usable gauges = {N} (<3).")
        return pd.DataFrame()

    # Aligned arrays
    lon_all = pts[LON_COL].to_numpy(float)
    lat_all = pts[LAT_COL].to_numpy(float)
    insar_all = pts["insar_cm"].to_numpy(float)
    dh_all = pts["dh_cm"].to_numpy(float)

    rng_master = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []
    export_plan = {}

    for rep in range(1, n_repl + 1):
        rng = np.random.default_rng(rng_master.integers(0, 2**31-1))
        all_idx = np.arange(N, dtype=int)

        # ---- Random 60/40 split (calibration/validation) ----
        n_cal0 = max(1, int(round(0.60 * N)))
        n_cal0 = min(n_cal0, N - 1)  # ensure validation non-empty
        perm = rng.permutation(all_idx)
        cal_idx = np.sort(perm[:n_cal0])
        val_idx = np.sort(perm[n_cal0:])

        # Save export plan (replicate #1)
        if rep == 1:
            export_plan["cal60_idx"] = cal_idx.copy()
            export_plan["val_idx"] = val_idx.copy()
            export_plan["single_idx"] = None  # to be filled when len(cur_idx)==1

        # ---- March down to 1 gauge: LS + IDW (same cal/val each step) ----
        cur_idx = cal_idx.copy()
        while len(cur_idx) >= 1:
            stack_idx = np.r_[cur_idx, val_idx]
            pts_split = pd.DataFrame({
                ID_COL: [common_ids_sorted[i] for i in stack_idx],
                LON_COL: lon_all[stack_idx],
                LAT_COL: lat_all[stack_idx],
                "insar_cm": insar_all[stack_idx],
                "dh_cm": dh_all[stack_idx],
            }).set_index(ID_COL)

            mm = _eval_ls_and_idw(
                pts_split,
                np.arange(len(cur_idx)),
                np.arange(len(cur_idx), len(stack_idx)),
                idw_power=idw_power
            )

            base_common = {
                "area": area_name, "pair_ref": _pair_dates_from_tag(pair_tag)[0], "pair_sec": _pair_dates_from_tag(pair_tag)[1],
                "dem": dem, "corr": corr, "replicate": rep,
                "n_total": N, "n_cal": int(len(cur_idx)), "n_val": int(len(val_idx)),
                "area_km2": float(area_km2),
                "density_km2_per_gauge": float(area_km2) / float(len(cur_idx)),
            }
            records.append({**base_common, "method": "least_squares", **mm["least_squares"]})
            records.append({**base_common, "method": "idw_dhvis",    **mm["idw_dhvis"], "a_gain": np.nan, "b_offset_cm": np.nan})

            if len(cur_idx) == 1:
                if rep == 1:
                    export_plan["single_idx"] = int(cur_idx[0])
                break

            crowded = _crowded_candidates(lon_all, lat_all, cur_idx, top_n=1)
            drop_pos = int(crowded[0]) if crowded.size else 0
            cur_idx = np.delete(cur_idx, drop_pos)

    # -------------------------- Per-pair GeoTIFFs ---------------------------
    try:
        cal60 = export_plan["cal60_idx"]
        px = lon_all[cal60]; py = lat_all[cal60]; pz = dh_all[cal60]
        idw_grid = _make_idw_grid_on_raster(px, py, pz, ref_tif=raster_tif, power=idw_power)
        out_idw = results_dir / f"dens_idw60_{dem}_{corr}_{pair_tag}.tif"
        _write_tif_like(raster_tif, out_idw, idw_grid)
        print(f"  🗺️  IDW Δh_vis (60%) written: {out_idw}")
    except Exception as e:
        print(f"  ⚠️  IDW export failed for {pair_tag}: {e}")

    try:
        a60, b60 = _fit_ls_params(insar_all[export_plan["cal60_idx"]], dh_all[export_plan["cal60_idx"]])
        arr60 = _apply_calibration_to_raster(raster_tif, a60, b60)
        out_cal60 = results_dir / f"dens_cal_60pct_{dem}_{corr}_{pair_tag}.tif"
        _write_tif_like(raster_tif, out_cal60, arr60)
        print(f"  🗺️  Calibrated (60%) written: {out_cal60}")
    except Exception as e:
        print(f"  ⚠️  Calibrated (60%) failed for {pair_tag}: {e}")

    try:
        single_idx = export_plan.get("single_idx", None)
        if single_idx is None:
            raise RuntimeError("Single-gauge index not recorded.")
        a1, b1 = _fit_ls_params(insar_all[[single_idx]], dh_all[[single_idx]])
        arr1 = _apply_calibration_to_raster(raster_tif, a1, b1)
        out_cal1 = results_dir / f"dens_cal_1g_{dem}_{corr}_{pair_tag}.tif"
        _write_tif_like(raster_tif, out_cal1, arr1)
        print(f"  🗺️  Calibrated (1 gauge) written: {out_cal1}")
    except Exception as e:
        print(f"  ⚠️  Calibrated (1 gauge) failed for {pair_tag}: {e}")

    return pd.DataFrame.from_records(records)

# ================================ Area driver ================================
def _process_area(area_dir: Path,
                  reps: int, seed: int, idw_power: float) -> None:
    """Process a single AREA: evaluate all SRTM+RAW pairs and write fresh outputs."""
    area_name   = area_dir.name
    dem         = DEM_FIXED
    corr        = CORR_FIXED

    gauge_csv   = area_dir / "water_gauges" / "eden_gauges.csv"
    results_dir = area_dir / "results"
    metrics_csv = results_dir / f"accuracy_metrics_density_{dem}_{corr}.csv"

    if not gauge_csv.exists():
        print(f"⏭️  Gauge CSV missing for {area_name}: {gauge_csv} — skipping area.")
        return

    pairs = _find_pairs_for_dem_corr(area_dir, area_name, dem, corr)
    if not pairs:
        print(f"⏭️  No interferograms for {area_name} with DEM={dem} CORR={corr} — skipping area.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for pair_tag in pairs:
        tif = _raster_path(area_dir, area_name, pair_tag, dem, corr)
        if tif is None:
            continue
        print(f"\n=== {area_name} — Pair {pair_tag} — DEM={dem} CORR={corr} ===")
        try:
            df_pair = _evaluate_pair_single_raster_and_exports(
                area_dir=area_dir,
                area_name=area_name,
                pair_tag=pair_tag,
                gauge_csv=gauge_csv,
                raster_tif=tif,
                dem=dem, corr=corr,
                n_repl=reps, seed=seed, idw_power=idw_power,
            )
        except Exception as e:
            print(f"  ⚠️  Pair {pair_tag} failed: {e}")
            df_pair = pd.DataFrame()

        if not df_pair.empty:
            all_rows.append(df_pair)

    if not all_rows:
        print(f"⏭️  No results to write for {area_name}.")
        return

    # Fresh metrics
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(metrics_csv, index=False)
    print(f"\n✅ [{area_name}] metrics written (fresh): {metrics_csv}  (rows: {len(df_all)})")

# =================================== CLI ====================================
def main():
    """CLI entry point: iterate areas, run assessment, and write results.

    Key flags
    ---------
    --reps / --seed :
        Control replicate count and randomness.
    --idw-power :
        Controls the inverse-distance weighting power for the IDW baseline.
    """
    ap = argparse.ArgumentParser(
        description="SRTM+RAW accuracy & density assessment with random 60/40 cal/val split and closest-gauge sweep."
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
                    help="IDW power (default: %(default)s)")
    args = ap.parse_args()

    root = Path(args.areas_root)
    targets = [root / args.area] if args.area else sorted([d for d in root.iterdir() if d.is_dir()])

    for area_dir in targets:
        _process_area(
            area_dir=area_dir,
            reps=args.reps,
            seed=args.seed,
            idw_power=args.idw_power,
        )

if __name__ == "__main__":
    main()
