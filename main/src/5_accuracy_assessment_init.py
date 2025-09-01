#!/usr/bin/env python3
"""
Calibrate a vertical-displacement interferogram (cm) with EDEN gauges via
least-squares (affine), create a fast IDW interpolation of gauge Δh_vis,
and write accuracy metrics for both methods across multiple calibration fractions.

Fractions run (calibration share): 60%, 40%, 30%, 20%, 10%, 5%
Each fraction uses a reproducible by-station split (val = 1 - cal).
"""

from __future__ import annotations
from pathlib import Path
import re
import os
import logging
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from typing import Tuple, Dict, Optional, List

# ---------------------------- LOGGING: silence GDAL/Rasterio debug -----------

os.environ.setdefault("CPL_DEBUG", "NO")  # silence GDAL debug
for name in ("rasterio", "rasterio._io", "rasterio.env", "rasterio._base"):
    logging.getLogger(name).setLevel(logging.ERROR)

# ---------------------------- CONFIG -----------------------------------------

GAUGE_CSV  = Path("/mnt/DATA2/bakke326l/processing/areas/ENP/water_gauges/eden_gauges.csv")
RASTER_TIF = Path("/mnt/DATA2/bakke326l/processing/areas/ENP/ENP_vertical_cm_20071216_20080131.tif")
METRICS_CSV = Path("/mnt/DATA2/bakke326l/processing/areas/_reports/accuracy_metrics.csv")

# CSV column names (wide table)
ID_COL  = "StationID"
LAT_COL = "Lat"
LON_COL = "Lon"

# Random seed for reproducible splits
RANDOM_SEED = 42

# Fractions of stations used for calibration (the rest is validation)
CALIBRATION_FRACTIONS: List[float] = [0.60, 0.40, 0.30, 0.20, 0.10, 0.05]

# IDW params (keep simple & fast)
IDW_POWER = 2.0
IDW_BLOCK = 512  # block size for grid interpolation (rows/cols per tile)

# ---------------------------- HELPERS ----------------------------------------

def parse_pair_dates_from_path(tif_path: Path) -> Tuple[str, str]:
    """Extract YYYYMMDD_YYYYMMDD from filename and return 'YYYY-MM-DD' strings."""
    m = re.search(r"(\d{8})_(\d{8})", tif_path.name)
    if not m:
        raise ValueError(f"Could not parse dates from filename: {tif_path.name}")
    ref = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
    sec = f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:]}"
    return ref, sec

def load_gauges_wide(csv_path: Path) -> pd.DataFrame:
    """Load the wide EDEN CSV; ensure essential columns exist."""
    df = pd.read_csv(csv_path)
    for col in [ID_COL, LAT_COL, LON_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")
    return df

def rowcol_from_xy(transform: Affine, x: float, y: float) -> Tuple[float, float]:
    """Map coords (x,y) → (row, col) in float (not rounded)."""
    col, row = ~transform * (x, y)
    return float(row), float(col)

def inside_image(h: int, w: int, row: float, col: float) -> bool:
    """Check if floating (row, col) lands inside the image bounds."""
    return (row >= 0) and (col >= 0) and (row < h) and (col < w)

def read_mean_3x3(ds: rasterio.io.DatasetReader, row: int, col: int) -> Optional[float]:
    """Read a 3×3 window centered on (row, col) and return nan-mean; skip if all nan."""
    r0 = max(0, row - 1); r1 = min(ds.height - 1, row + 1)
    c0 = max(0, col - 1); c1 = min(ds.width  - 1, col + 1)
    window = Window.from_slices((r0, r1 + 1), (c0, c1 + 1))
    arr = ds.read(1, window=window).astype("float32")
    nodata = ds.nodata
    if nodata is not None and not np.isnan(nodata):
        arr[arr == nodata] = np.nan
    if not np.isfinite(arr).any():
        return None
    return float(np.nanmean(arr))

def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least squares fit y = a*x + b."""
    A = np.c_[x, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])

def apply_affine_to_array(arr: np.ndarray, a: float, b: float, nodata: Optional[float]) -> np.ndarray:
    """Apply y = a*x + b to full array; preserve NaNs."""
    out = arr.astype("float32").copy()
    if nodata is not None and not np.isnan(nodata):
        out = np.where(out == nodata, np.nan, out)
    out = a * out + b
    return out

def write_geotiff_like(dst_path: Path, src_ds: rasterio.io.DatasetReader, data: np.ndarray):
    profile = src_ds.profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", predictor=3, tiled=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)

def regression_on_validation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Diagnostic fit y_pred ≈ s*y_true + i."""
    if len(y_true) < 2:
        return np.nan, np.nan
    A = np.c_[y_true, np.ones_like(y_true)]
    s, i = np.linalg.lstsq(A, y_pred, rcond=None)[0]
    return float(s), float(i)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    bias = float(np.mean(err))
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    r    = float(np.corrcoef(y_true, y_pred)[0,1]) if len(y_true) >= 2 else np.nan
    slope, intercept = regression_on_validation(y_true, y_pred)
    pct5  = float(np.mean(np.abs(err) <= 5.0) * 100.0)
    pct10 = float(np.mean(np.abs(err) <= 10.0) * 100.0)
    return {
        "rmse_cm": rmse,
        "mae_cm": mae,
        "bias_cm": bias,
        "r": r,
        "slope_val": slope,
        "intercept_cm": intercept,
        "pct_within_5cm": pct5,
        "pct_within_10cm": pct10,
    }

def cal_val_split_by_station(df: pd.DataFrame, frac_cal: float, seed: int, id_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stations = df[id_col].unique()
    rng.shuffle(stations)
    n_cal = max(1, int(round(frac_cal * len(stations))))
    cal_ids = set(stations[:n_cal])
    cal = df[df[id_col].isin(cal_ids)].copy()
    val = df[~df[id_col].isin(cal_ids)].copy()
    return cal, val

# --- Visible-surface change rule ---------------------------------------------

def visible_surface_delta(ref_cm: np.ndarray, sec_cm: np.ndarray) -> np.ndarray:
    """
    Compute Δh using ONLY the part visible to InSAR (water above ground).
    Δh_vis = max(sec_cm, 0) - max(ref_cm, 0)
    """
    ref_clip = np.maximum(ref_cm.astype(float), 0.0)
    sec_clip = np.maximum(sec_cm.astype(float), 0.0)
    return sec_clip - ref_clip

# --- Fast IDW interpolation (lon/lat with cos(lat) scaling) ------------------

def _idw_predict_points(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                        qx: np.ndarray, qy: np.ndarray,
                        power: float = 2.0) -> np.ndarray:
    """
    Predict values at query points (qx,qy) from scattered points (px,py,pz) using IDW.
    Uses simple lon-degree scaling by cos(mean lat) so distances are quasi-metric.
    """
    px = px.astype("float64"); py = py.astype("float64"); pz = pz.astype("float64")
    qx = qx.astype("float64"); qy = qy.astype("float64")
    lat0 = np.nanmean(py)
    cx = np.cos(np.deg2rad(lat0))
    dx = (qx[:, None] - px[None, :]) * cx
    dy = (qy[:, None] - py[None, :])
    d2 = dx*dx + dy*dy
    w = 1.0 / np.maximum(d2, 1e-18) ** (power / 2.0)
    pred = (w @ pz) / np.sum(w, axis=1)
    imin = np.argmin(d2, axis=1)
    dmin = d2[np.arange(d2.shape[0]), imin]
    hit = dmin < 1e-18
    if np.any(hit):
        pred[hit] = pz[imin[hit]]
    return pred.astype("float32")

def _idw_grid_from_points(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                          transform: Affine, width: int, height: int,
                          power: float = 2.0, block: int = 512) -> np.ndarray:
    """Interpolate onto a full raster grid using tiled IDW (fast & memory-safe)."""
    out = np.full((height, width), np.nan, dtype="float32")
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    for r0 in range(0, height, block):
        r1 = min(height, r0 + block)
        rows = np.arange(r0, r1, dtype="float64") + 0.5
        for c0 in range(0, width, block):
            c1 = min(width, c0 + block)
            cols = np.arange(c0, c1, dtype="float64") + 0.5
            XX = c + a * cols[None, :] + b * rows[:, None]
            YY = f + d * cols[None, :] + e * rows[:, None]
            qx = XX.ravel(); qy = YY.ravel()
            z = _idw_predict_points(px, py, pz, qx, qy, power=power)
            out[r0:r1, c0:c1] = z.reshape((r1 - r0, c1 - c0))
    return out

# ---------------------------- CORE RUN ---------------------------------------

def run_for_fraction(ds, pts: pd.DataFrame, ref_str: str, sec_str: str, cal_frac: float, seed: int):
    """
    Do one cal/val split at 'cal_frac' and compute:
      - InSAR least-squares metrics (+ save calibrated raster)
      - IDW interpolation metrics (+ save IDW raster)
    Returns two metrics dicts and output raster paths.
    """
    frac_tag = f"cal{int(round(cal_frac*100))}p"

    cal, val = cal_val_split_by_station(pts, cal_frac, seed, ID_COL)
    if cal[ID_COL].nunique() < 2 or len(cal) < 2:
        raise RuntimeError(f"Too few calibration stations for fraction {cal_frac:.2f}.")
    if val[ID_COL].nunique() < 1 or len(val) < 1:
        raise RuntimeError(f"Too few validation stations for fraction {cal_frac:.2f}.")

    # --- InSAR least-squares calibration
    x_cal = cal["insar_cm"].values.astype(float)
    y_cal = cal["dh_cm"].values.astype(float)
    a, b = fit_affine(x_cal, y_cal)

    x_val = val["insar_cm"].values.astype(float)
    y_val = val["dh_cm"].values.astype(float)
    y_pred_val_insar = a * x_val + b
    metrics_insar = compute_metrics(y_true=y_val, y_pred=y_pred_val_insar)

    # Save calibrated raster (tagged by fraction)
    arr = ds.read(1)
    arr_cal = apply_affine_to_array(arr, a, b, ds.nodata)
    out_tif_cal = RASTER_TIF.with_name(f"{RASTER_TIF.stem}_calib_least_squares_{frac_tag}.tif")
    write_geotiff_like(out_tif_cal, ds, arr_cal)

    # --- Gauge-only IDW interpolation
    px = cal["x"].values.astype("float64")
    py = cal["y"].values.astype("float64")
    pz = cal["dh_cm"].values.astype("float64")

    y_pred_val_idw = _idw_predict_points(px, py, pz, val["x"].values, val["y"].values, power=IDW_POWER)
    metrics_idw = compute_metrics(y_true=y_val, y_pred=y_pred_val_idw)

    interp_grid = _idw_grid_from_points(px, py, pz, ds.transform, ds.width, ds.height,
                                        power=IDW_POWER, block=IDW_BLOCK)
    out_tif_idw = RASTER_TIF.with_name(f"{RASTER_TIF.stem}_interp_idw_dhvis_{frac_tag}.tif")
    write_geotiff_like(out_tif_idw, ds, interp_grid)

    # Package rows
    r1 = dict(metrics_insar)
    r1.update({
        "pair_ref": ref_str,
        "pair_sec": sec_str,
        "method": f"least_squares_{frac_tag}",
        "n_cal": int(cal[ID_COL].nunique()),
        "n_val": int(val[ID_COL].nunique()),
        "fraction_calibration": cal_frac,
        "a_gain": float(a),
        "b_offset_cm": float(b),
        "note": f"InSAR calibrated with {int(round(cal_frac*100))}% gauges; visible-surface Δh.",
    })

    r2 = dict(metrics_idw)
    r2.update({
        "pair_ref": ref_str,
        "pair_sec": sec_str,
        "method": f"interp_idw_{frac_tag}",
        "n_cal": int(cal[ID_COL].nunique()),
        "n_val": int(val[ID_COL].nunique()),
        "fraction_calibration": cal_frac,
        "a_gain": np.nan,
        "b_offset_cm": np.nan,
        "note": f"Gauge-only IDW from {int(round(cal_frac*100))}% gauges; visible-surface Δh.",
    })

    return r1, r2, out_tif_cal, out_tif_idw

# ---------------------------- PIPELINE ---------------------------------------

def main():
    # Pair dates
    ref_str, sec_str = parse_pair_dates_from_path(RASTER_TIF)

    # Load gauges and build Δh_vis from wide table
    gauges = load_gauges_wide(GAUGE_CSV)
    for col in (ref_str, sec_str):
        if col not in gauges.columns:
            raise ValueError(f"Gauge CSV is missing date column: {col}")

    g = gauges[[ID_COL, LAT_COL, LON_COL, ref_str, sec_str]].copy()
    g = g.rename(columns={ref_str: "ref_cm", sec_str: "sec_cm"})
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["ref_cm", "sec_cm", LAT_COL, LON_COL])
    if g.empty:
        raise RuntimeError("No gauges with both ref and sec values.")
    g["dh_cm"] = visible_surface_delta(g["ref_cm"].to_numpy(), g["sec_cm"].to_numpy())

    # Sample raster at gauges once
    with rasterio.open(RASTER_TIF) as ds:
        if ds.crs is None or (ds.crs.to_epsg() != 4326):
            raise RuntimeError("Expected raster in EPSG:4326.")
        rows = []
        for _, r in g.iterrows():
            x = float(r[LON_COL]); y = float(r[LAT_COL])
            rowf, colf = rowcol_from_xy(ds.transform, x, y)
            if not inside_image(ds.height, ds.width, rowf, colf):
                continue
            row = int(round(rowf)); col = int(round(colf))
            insar = read_mean_3x3(ds, row, col)
            if insar is None or not np.isfinite(insar):
                continue
            rows.append({
                ID_COL: r[ID_COL],
                "x": x, "y": y,
                "insar_cm": float(insar),
                "dh_cm": float(r["dh_cm"]),
            })
        pts = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["insar_cm", "dh_cm"])
        if pts[ID_COL].nunique() < 3:
            raise RuntimeError(f"Too few usable gauges inside raster with valid pixels: {pts[ID_COL].nunique()}")

        # Run all requested calibration fractions
        rows_out = []
        last_cal_path = None
        last_idw_path = None
        for cal_frac in CALIBRATION_FRACTIONS:
            try:
                r1, r2, out_tif_cal, out_tif_idw = run_for_fraction(ds, pts, ref_str, sec_str, cal_frac, RANDOM_SEED)
                rows_out.extend([r1, r2])
                last_cal_path = out_tif_cal
                last_idw_path = out_tif_idw
                print(f"✓ {int(round(cal_frac*100))}% cal: InSAR RMSE={r1['rmse_cm']:.2f} cm, IDW RMSE={r2['rmse_cm']:.2f} cm")
            except RuntimeError as e:
                print(f"⚠️  Skipping fraction {int(round(cal_frac*100))}%: {e}")

    # Write/append metrics
    if rows_out:
        METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(rows_out)
        if METRICS_CSV.exists():
            df_out.to_csv(METRICS_CSV, mode="a", header=False, index=False)
        else:
            df_out.to_csv(METRICS_CSV, index=False)

        # Friendly summary
        if last_cal_path and last_idw_path:
            print(f"✅ Latest calibrated raster:  {last_cal_path}")
            print(f"🗺️  Latest IDW Δh_vis raster: {last_idw_path}")
        print(f"📈 Metrics appended to: {METRICS_CSV}")
    else:
        print("No results written (all fractions skipped).")

if __name__ == "__main__":
    main()
