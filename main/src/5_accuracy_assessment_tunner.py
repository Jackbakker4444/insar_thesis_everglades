#!/usr/bin/env python3
"""
Calibrate a vertical-displacement interferogram (cm) with EDEN gauges via
least-squares (affine) and Huber (robust) regression; also evaluate a flipped
InSAR variant and a fast IDW interpolation of gauge Δh_vis.

Per calibration fraction we compute **five** methods:
  1) least_squares           : y = a*x + b
  2) huber                   : robust affine (Huber loss) y = a*x + b
  3) least_squares_flipped   : fit on (-x) i.e., y = a*(-x) + b
  4) huber_flipped           : robust fit on (-x)
  5) interp_idw              : Δh_vis surface from calibration gauges only

Fractions (calibration share): 60%, 40%, 30%, 20%, 10%, 5%.
Each fraction uses a reproducible by-station split (val = 1 - cal).
All gauge changes use the **visible-surface rule**: Δh_vis = max(sec,0) - max(ref,0).
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
logging.getLogger().setLevel(logging.WARNING)
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

# ---------- Huber regression (IRLS) ------------------------------------------

def fit_huber_affine(x: np.ndarray, y: np.ndarray, delta: float = 1.5,
                     max_iter: int = 50, tol: float = 1e-6) -> Tuple[float, float]:
    """
    Robust affine fit y = a*x + b using Huber loss via IRLS.
    delta ~1.5 (in residual std units) is a reasonable default for small N.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    A = np.c_[x, np.ones_like(x)]
    # initial OLS
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    for _ in range(max_iter):
        r = y - (a * x + b)
        # robust scale
        mad = np.median(np.abs(r - np.median(r)))
        s = 1.4826 * mad if mad > 0 else (np.std(r) if np.std(r) > 0 else 1.0)
        u = r / (s + 1e-12)
        # Huber weights
        w = np.ones_like(u)
        mask = np.abs(u) > delta
        w[mask] = (delta / (np.abs(u[mask]) + 1e-12))
        # weighted least squares
        Aw = A * w[:, None]
        yw = y * w
        beta_new, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        da = abs(beta_new[0] - a); db = abs(beta_new[1] - b)
        a, b = float(beta_new[0]), float(beta_new[1])
        if max(da, db) < tol:
            break
    return a, b

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
    Do one cal/val split at 'cal_frac' and compute FIVE methods:
      - least squares (affine)
      - Huber (robust affine)
      - least squares on flipped InSAR (x->-x)
      - Huber on flipped InSAR (x->-x)
      - IDW interpolation (gauge-only)

    Returns a list of metric dicts and a dict of output raster paths.
    """
    frac_tag = f"cal{int(round(cal_frac*100))}p"

    cal, val = cal_val_split_by_station(pts, cal_frac, seed, ID_COL)
    if cal[ID_COL].nunique() < 2 or len(cal) < 2:
        raise RuntimeError(f"Too few calibration stations for fraction {cal_frac:.2f}.")
    if val[ID_COL].nunique() < 1 or len(val) < 1:
        raise RuntimeError(f"Too few validation stations for fraction {cal_frac:.2f}.")

    x_cal = cal["insar_cm"].values.astype(float)
    y_cal = cal["dh_cm"].values.astype(float)
    x_val = val["insar_cm"].values.astype(float)
    y_val = val["dh_cm"].values.astype(float)

    metrics_rows = []
    out_paths = {}

    # Read array once
    arr = ds.read(1)

    # --- (1) InSAR least-squares calibration
    a_ls, b_ls = fit_affine(x_cal, y_cal)
    y_pred_val_ls = a_ls * x_val + b_ls
    m_ls = compute_metrics(y_true=y_val, y_pred=y_pred_val_ls)
    m_ls.update({
        "pair_ref": ref_str, "pair_sec": sec_str,
        "method": f"least_squares_{frac_tag}",
        "n_cal": int(cal[ID_COL].nunique()), "n_val": int(val[ID_COL].nunique()),
        "fraction_calibration": cal_frac,
        "a_gain": float(a_ls), "b_offset_cm": float(b_ls),
        "note": f"InSAR OLS with {int(round(cal_frac*100))}% gauges; visible-surface Δh.",
    })
    metrics_rows.append(m_ls)

    out_tif_ls = RASTER_TIF.with_name(f"{RASTER_TIF.stem}_calib_least_squares_{frac_tag}.tif")
    write_geotiff_like(out_tif_ls, ds, apply_affine_to_array(arr, a_ls, b_ls, ds.nodata))
    out_paths["least_squares"] = out_tif_ls

    # --- (2) InSAR Huber robust calibration
    a_hu, b_hu = fit_huber_affine(x_cal, y_cal, delta=1.5)
    y_pred_val_hu = a_hu * x_val + b_hu
    m_hu = compute_metrics(y_true=y_val, y_pred=y_pred_val_hu)
    m_hu.update({
        "pair_ref": ref_str, "pair_sec": sec_str,
        "method": f"huber_{frac_tag}",
        "n_cal": int(cal[ID_COL].nunique()), "n_val": int(val[ID_COL].nunique()),
        "fraction_calibration": cal_frac,
        "a_gain": float(a_hu), "b_offset_cm": float(b_hu),
        "note": f"InSAR Huber (robust) with {int(round(cal_frac*100))}% gauges; visible-surface Δh.",
    })
    metrics_rows.append(m_hu)

    out_tif_hu = RASTER_TIF.with_name(f"{RASTER_TIF.stem}_calib_huber_{frac_tag}.tif")
    write_geotiff_like(out_tif_hu, ds, apply_affine_to_array(arr, a_hu, b_hu, ds.nodata))
    out_paths["huber"] = out_tif_hu

    # --- (3) least squares on flipped InSAR (x -> -x)
    x_cal_f = -x_cal
    x_val_f = -x_val
    a_lsf, b_lsf = fit_affine(x_cal_f, y_cal)
    y_pred_val_lsf = a_lsf * x_val_f + b_lsf
    m_lsf = compute_metrics(y_true=y_val, y_pred=y_pred_val_lsf)
    m_lsf.update({
        "pair_ref": ref_str, "pair_sec": sec_str,
        "method": f"least_squares_flipped_{frac_tag}",
        "n_cal": int(cal[ID_COL].nunique()), "n_val": int(val[ID_COL].nunique()),
        "fraction_calibration": cal_frac,
        "a_gain": float(a_lsf), "b_offset_cm": float(b_lsf),
        "note": "Fitted on flipped InSAR (x→-x). Mathematically equivalent to a sign change in slope.",
    })
    metrics_rows.append(m_lsf)

    out_tif_lsf = RASTER_TIF.with_name(f"{RASTER_TIF.stem}_calib_least_squares_flipped_{frac_tag}.tif")
    write_geotiff_like(out_tif_lsf, ds, apply_affine_to_array(-arr, a_lsf, b_lsf, ds.nodata))
    out_paths["least_squares_flipped"] = out_tif_lsf

    # --- (4) Huber on flipped InSAR (x -> -x)
    a_huf, b_huf = fit_huber_affine(x_cal_f, y_cal, delta=1.5)
    y_pred_val_huf = a_huf * x_val_f + b_huf
    m_huf = compute_metrics(y_true=y_val, y_pred=y_pred_val_huf)
    m_huf.update({
        "pair_ref": ref_str, "pair_sec": sec_str,
        "method": f"huber_flipped_{frac_tag}",
        "n_cal": int(cal[ID_COL].nunique()), "n_val": int(val[ID_COL].nunique()),
        "fraction_calibration": cal_frac,
        "a_gain": float(a_huf), "b_offset_cm": float(b_huf),
        "note": "Robust fit on flipped InSAR (x→-x). May match non-flipped metrics.",
    })
    metrics_rows.append(m_huf)

    out_tif_huf = RASTER_TIF.with_name(f"{RASTER_TIF.stem}_calib_huber_flipped_{frac_tag}.tif")
    write_geotiff_like(out_tif_huf, ds, apply_affine_to_array(-arr, a_huf, b_huf, ds.nodata))
    out_paths["huber_flipped"] = out_tif_huf

    # --- (5) Gauge-only IDW interpolation
    px = cal["x"].values.astype("float64")
    py = cal["y"].values.astype("float64")
    pz = cal["dh_cm"].values.astype("float64")

    y_pred_val_idw = _idw_predict_points(px, py, pz, val["x"].values, val["y"].values, power=IDW_POWER)
    m_idw = compute_metrics(y_true=y_val, y_pred=y_pred_val_idw)
    m_idw.update({
        "pair_ref": ref_str, "pair_sec": sec_str,
        "method": f"interp_idw_{frac_tag}",
        "n_cal": int(cal[ID_COL].nunique()), "n_val": int(val[ID_COL].nunique()),
        "fraction_calibration": cal_frac,
        "a_gain": np.nan, "b_offset_cm": np.nan,
        "note": f"Gauge-only IDW from {int(round(cal_frac*100))}% gauges; visible-surface Δh.",
    })
    metrics_rows.append(m_idw)

    out_tif_idw = RASTER_TIF.with_name(f"{RASTER_TIF.stem}_interp_idw_dhvis_{frac_tag}.tif")
    interp_grid = _idw_grid_from_points(px, py, pz, ds.transform, ds.width, ds.height,
                                        power=IDW_POWER, block=IDW_BLOCK)
    write_geotiff_like(out_tif_idw, ds, interp_grid)
    out_paths["interp_idw"] = out_tif_idw

    return metrics_rows, out_paths

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
        all_rows = []
        last_paths = {}
        for cal_frac in CALIBRATION_FRACTIONS:
            try:
                rows_out, paths = run_for_fraction(ds, pts, ref_str, sec_str, cal_frac, RANDOM_SEED)
                all_rows.extend(rows_out)
                last_paths = paths
                # quick console summary
                r_ls  = next(r for r in rows_out if r["method"].startswith("least_squares_"))
                r_hu  = next(r for r in rows_out if r["method"].startswith("huber_") and "flipped" not in r["method"])
                r_lsf = next(r for r in rows_out if r["method"].startswith("least_squares_flipped_"))
                r_huf = next(r for r in rows_out if r["method"].startswith("huber_flipped_"))
                r_idw = next(r for r in rows_out if r["method"].startswith("interp_idw_"))
                p = int(round(cal_frac*100))
                print(f"✓ {p}% cal | OLS RMSE={r_ls['rmse_cm']:.2f} | Huber RMSE={r_hu['rmse_cm']:.2f} "
                      f"| OLS flip RMSE={r_lsf['rmse_cm']:.2f} | Huber flip RMSE={r_huf['rmse_cm']:.2f} "
                      f"| IDW RMSE={r_idw['rmse_cm']:.2f}  (cm)")
            except RuntimeError as e:
                print(f"⚠️  Skipping fraction {int(round(cal_frac*100))}%: {e}")

    # Write/append metrics
    if all_rows:
        METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(all_rows)
        if METRICS_CSV.exists():
            df_out.to_csv(METRICS_CSV, mode="a", header=False, index=False)
        else:
            df_out.to_csv(METRICS_CSV, index=False)

        # Friendly summary
        if last_paths:
            for k, v in last_paths.items():
                print(f"🗺️  {k}: {v}")
        print(f"📈 Metrics appended to: {METRICS_CSV}")
    else:
        print("No results written (all fractions skipped).")

if __name__ == "__main__":
    main()
