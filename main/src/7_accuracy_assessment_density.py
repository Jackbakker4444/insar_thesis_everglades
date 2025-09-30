#!/usr/bin/env python3
'''
7_accuracy_assessment_density.py — Single-raster density & accuracy assessment (SRTM+RAW only)

Summary
-------
Evaluate per-area InSAR vertical-displacement rasters against EDEN gauges for ONE
fixed raster choice: DEM=SRTM and CORR=RAW. Two accuracy curves are produced:
  • RAW  (least-squares calibration of InSAR → Δh_vis)
  • IDW  (gauge-only interpolation baseline, using EXACTLY the same gauges/splits as RAW)

Knee points
-----------
- Computed **only for least_squares (RAW)**, not for IDW.
- Computed for **RMSE** and **bias** (|bias| by default; use --bias-signed to switch).
- Uses **mixed density bins** (default: 0–600 by 25; 600–3000 by 200).

Selection randomness
--------------------
- Initial ~60% calibration subset uses **stochastic farthest-point sampling**:
  at each step, pick **randomly among the top-M farthest** candidates (default M=5).
  Control with --spread-top-m.

Design guarantees
-----------------
- Pair discovery strictly looks for: <AREA>_vertical_cm_<REF>_<SEC>_SRTM_RAW.tif
- Gauges used by IDW are the SAME gauges used by RAW at every step:
    * Same valid-mask (from the RAW raster)
    * Same calibration/validation splits
    * Same march-down sequence
- IDW grid export is masked by THIS raster’s NaN mask (SRTM+RAW).

Outputs (per area)
------------------
results/accuracy_metrics_density_SRTM_RAW.csv
results/critical_density_pairs_SRTM_RAW.csv      # includes RMSE & bias knees for LS only
results/critical_density_area_SRTM_RAW.csv       # area-level summaries for RMSE & bias
Plus per-pair GeoTIFFs:
  dens_idw60_SRTM_RAW_<PAIR>.tif
  dens_cal_60pct_SRTM_RAW_<PAIR>.tif
  dens_cal_1g_SRTM_RAW_<PAIR>.tif

Run examples
------------
All areas:
  python 7_accuracy_assessment_density.py
One area (e.g., ENP):
  python 7_accuracy_assessment_density.py --area ENP
Tuning:
  python 7_accuracy_assessment_density.py --reps 50 --seed 42 --idw-power 2.0 \
    --knee-bins "0:600:25,600:3000:200" --knee-bootstrap 300 --knee-slope-floor 0.002 \
    --spread-top-m 5

Assumptions
-----------
/mnt/DATA2/bakke326l/processing/areas/<AREA>/
  ├─ water_gauges/eden_gauges.csv   (wide daily, includes REF & SEC dates)
  └─ interferograms/<AREA>_vertical_cm_<REF>_<SEC>_SRTM_RAW.tif
Raster is single-band float (cm), EPSG:4326, nodata set or NaN.
'''
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

# Stochastic farthest-point selection
SPREAD_TOP_M_DEFAULT    = 5       # pick randomly among top-M farthest at each add step

# “Elbow” (critical density) detector defaults
KNEE_BOOTSTRAP_DEFAULT   = 300     # resamples (replicate-bootstrap)
KNEE_SLOPE_FLOOR_DEFAULT = 0.002   # cm per (km²/gauge): require slope_right >= slope_left + floor
KNEE_MIN_SEG_POINTS      = 3       # at least 3 points per segment
KNEE_MIN_TOTAL_POINTS    = 6       # need >=6 median points along the curve
KNEE_BINS_DEFAULT        = "0:600:25,600:3000:200"  # mixed bins: fine then coarse

# Gauge columns
ID_COL, LAT_COL, LON_COL = "StationID", "Lat", "Lon"

# Geodesy
GEOD = Geod(ellps="WGS84")

# ============================== Small utilities ==============================
def _pair_dates_from_tag(pair_tag: str) -> Tuple[str, str]:
    if not re.fullmatch(r"\d{8}_\d{8}", pair_tag):
        raise ValueError(f"PAIR tag must be YYYYMMDD_YYYYMMDD, got: {pair_tag}")
    a, b = pair_tag.split("_")
    return f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"

def _find_pairs_for_dem_corr(area_dir: Path, area_name: str, dem: str, corr: str) -> List[str]:
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
    cand = area_dir / "interferograms" / f"{area_name}_vertical_cm_{pair_tag}_{dem}_{corr}.tif"
    return cand if cand.exists() else None

def _load_gauges_wide(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in (ID_COL, LAT_COL, LON_COL):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    return df

def _visible_surface_delta(ref_cm: np.ndarray, sec_cm: np.ndarray) -> np.ndarray:
    return np.maximum(sec_cm.astype(float), 0.0) - np.maximum(ref_cm.astype(float), 0.0)

def _rowcol_from_xy(transform: Affine, x: float, y: float) -> Tuple[float, float]:
    col, row = ~transform * (x, y)
    return float(row), float(col)

def _inside_image(h: int, w: int, row: float, col: float) -> bool:
    return (row >= 0) and (col >= 0) and (row < h) and (col < w)

def _read_mean_3x3(ds: rasterio.io.DatasetReader, row: int, col: int) -> Optional[float]:
    r0 = max(0, row - 1); r1 = min(ds.height - 1, row + 1)
    c0 = max(0, col - 1); c1 = min(ds.width  - 1, col + 1)
    arr = ds.read(1, window=Window.from_slices((r0, r1 + 1), (c0, c1 + 1))).astype("float32")
    if ds.nodata is not None and not np.isnan(ds.nodata):
        arr[arr == ds.nodata] = np.nan
    if not np.isfinite(arr).any():
        return None
    return float(np.nanmean(arr))

def _geod_area_of_geojson(geom) -> float:
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
    if len(y_true) < 2: return float("nan")
    yt = y_true - np.mean(y_true); yp = y_pred - np.mean(y_pred)
    vy = np.sum(yt*yt); vp = np.sum(yp*yp)
    if vy <= 1e-12 or vp <= 1e-12: return float("nan")
    return float(np.sum(yt*yp) / np.sqrt(vy*vp))

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    sigma_e = float(np.sqrt(np.mean((err - bias)**2)))
    return {"rmse_cm": rmse, "mae_cm": mae, "bias_cm": bias, "sigma_e_cm": sigma_e, "r": _safe_corrcoef(y_true, y_pred)}

def _val_spread(y_val: np.ndarray) -> Tuple[float, float]:
    yv = np.asarray(y_val, dtype=float)
    if yv.size == 0 or not np.isfinite(yv).any():
        return float("nan"), float("nan")
    q25, q75 = np.nanpercentile(yv, [25, 75])
    iqr = float(q75 - q25)
    sd  = float(np.nanstd(yv))
    return iqr, sd

def _augment_metrics(m: Dict[str, float], y_val: np.ndarray) -> Dict[str, float]:
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
    lon = np.deg2rad(lon.astype("float64")); lat = np.deg2rad(lat.astype("float64"))
    dlon = lon[:, None] - lon[None, :]
    a = np.clip(np.sin(lat)[:,None]*np.sin(lat)[None,:] + np.cos(lat)[:,None]*np.cos(lat)[None,:]*np.cos(dlon), -1.0, 1.0)
    return 6371000.0 * np.arccos(a)

def _spread_selection_stochastic(lon: np.ndarray, lat: np.ndarray, k: int,
                                 rng: np.random.Generator, top_m: int = SPREAD_TOP_M_DEFAULT) -> np.ndarray:
    """
    Stochastic farthest-point sampling to select k well-spread indices.
    - Start at a random seed point.
    - Iteratively add: compute each candidate's distance to its nearest selected point,
      sort by that min-distance (desc), then **pick randomly among the top-M farthest**.
    """
    n = len(lon)
    if k >= n: return np.arange(n, dtype=int)
    D = _haversine_matrix(lon, lat)
    cur = [int(rng.integers(0, n))]
    remaining = set(range(n)) - set(cur)
    min_d = D[:, cur].min(axis=1)
    while len(cur) < k:
        order = np.argsort(-min_d)  # farthest first
        order = [o for o in order if o in remaining]
        if not order:
            break
        take = order[:min(top_m, len(order))]
        cand = int(rng.choice(take))
        cur.append(cand)
        remaining.remove(cand)
        min_d = np.minimum(min_d, D[:, cand])
    return np.array(cur, dtype=int)

def _crowded_candidates(lon: np.ndarray, lat: np.ndarray, idx: np.ndarray,
                        keep_global: int, top_n: int = 4) -> np.ndarray:
    if len(idx) <= 1: return idx
    lon_s, lat_s = lon[idx], lat[idx]
    D = _haversine_matrix(lon_s, lat_s)
    np.fill_diagonal(D, np.inf)
    nnd = np.min(D, axis=1)
    order = np.argsort(nnd)  # smallest NND -> most crowded
    order = np.array([o for o in order if idx[o] != keep_global], dtype=int)
    if order.size == 0: return np.array([], dtype=int)
    return order[:min(top_n, order.size)]

# ----------------------------- Model fitting --------------------------------
def _fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    A = np.c_[x, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])

def _fit_ls_params(insar_vals: np.ndarray, dh_vals: np.ndarray) -> Tuple[float, float]:
    n = insar_vals.size
    if n <= 2:
        a = -1.0
        b = float(np.mean(dh_vals + insar_vals))
        return a, b
    return _fit_affine(insar_vals, dh_vals)

def _eval_ls_and_idw(pts: pd.DataFrame, cal_idx: np.ndarray, val_idx: np.ndarray, idw_power: float) -> Dict[str, Dict[str, float]]:
    """
    IMPORTANT: This uses the SAME cal/val indices (built from the RAW raster samples)
    for both LS and IDW — guaranteeing identical gauge sets & masks.
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
    Build an IDW Δh_vis grid that is masked by the SAME raster valid-mask (SRTM+RAW)
    used to pick gauges. px/py/pz MUST come from the calibration indices of the RAW run.
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
    with rasterio.open(src_tif) as ds:
        arr = ds.read(1).astype("float32")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr = np.where(arr == ds.nodata, np.nan, arr)
        valid_mask = np.isfinite(arr)
        out = np.full_like(arr, np.nan, dtype="float32")
        out[valid_mask] = a * arr[valid_mask] + b
        return out

# ============================= Knee (with variable bins) =====================
def _parse_knee_bins(spec: str) -> np.ndarray:
    """
    spec like "150:600:25,600:2000:200" -> strictly increasing bin edges array.
    """
    edges: List[float] = []
    for i, token in enumerate(spec.split(",")):
        if not token.strip(): continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad knee-bins token '{token}'. Expected start:end:step.")
        start, end, step = map(float, parts)
        if step <= 0 or end <= start:
            raise ValueError(f"Invalid knee-bins range: {token}")
        arr = np.arange(start, end + 0.5 * step, step, dtype=float)
        if i > 0 and len(edges) and arr[0] == edges[-1]:
            arr = arr[1:]  # avoid duplicate
        edges.extend(arr.tolist())
    edges = np.unique(np.array(edges))
    if edges.size < 3:
        raise ValueError("Bins need at least 3 edges.")
    return edges

def _binned_curve(df: pd.DataFrame, bins: np.ndarray, metric_col: str, min_per_bin: int = 3) -> pd.DataFrame:
    d = df.copy()
    d = d[np.isfinite(d["density_km2_per_gauge"]) & np.isfinite(d[metric_col])]
    d["dens_bin"] = pd.cut(d["density_km2_per_gauge"], bins=bins, include_lowest=True)
    g = (d.groupby("dens_bin", observed=True)
           .agg(x=("density_km2_per_gauge", "mean"),
                y=(metric_col, "median"),
                n=(metric_col, "size"))
           .dropna()
           .reset_index(drop=True))
    g = g[g["n"] >= min_per_bin].sort_values("x").reset_index(drop=True)
    return g

def _two_segment_break(x, y, min_seg=KNEE_MIN_SEG_POINTS, slope_floor=KNEE_SLOPE_FLOOR_DEFAULT):
    """
    Single breakpoint two-segment linear fit; returns dict:
      {i, x0, sse, sl, sr}
    Enforces: slope_right >= slope_left + slope_floor.
    Picks earliest breakpoint within 5% of min-SSE.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    cands = []
    for i in range(min_seg, len(x) - min_seg + 1):
        sl, il = np.polyfit(x[:i], y[:i], 1);  yhat_l = sl * x[:i] + il;  sse_l = np.sum((y[:i] - yhat_l) ** 2)
        sr, ir = np.polyfit(x[i-1:], y[i-1:], 1); yhat_r = sr * x[i-1:] + ir; sse_r = np.sum((y[i-1:] - yhat_r) ** 2)
        sse = sse_l + sse_r
        cands.append({"i": i-1, "x0": x[i-1], "sse": sse, "sl": sl, "sr": sr})
    cands = [c for c in cands if c["sr"] >= c["sl"] + slope_floor]
    if not cands:
        return None
    sse_min = min(c["sse"] for c in cands)
    near = [c for c in cands if c["sse"] <= 1.05 * sse_min]
    best = sorted(near, key=lambda c: c["x0"])[0]
    return best

def _bootstrap_break_bins(df_pair: pd.DataFrame, bins: np.ndarray, metric_col: str, B=KNEE_BOOTSTRAP_DEFAULT,
                          min_total_pts=KNEE_MIN_TOTAL_POINTS, min_seg=KNEE_MIN_SEG_POINTS,
                          slope_floor=KNEE_SLOPE_FLOOR_DEFAULT) -> Tuple[float,float,float]:
    boot = []
    has_rep = "replicate" in df_pair.columns
    if has_rep:
        reps = df_pair["replicate"].unique()
    rng = np.random.default_rng(42)
    for _ in range(B):
        if has_rep:
            samp = rng.choice(reps, size=len(reps), replace=True)
            d = pd.concat([df_pair[df_pair["replicate"] == r] for r in samp], ignore_index=True)
        else:
            d = df_pair.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 2**31-1)))
        cur = _binned_curve(d, bins=bins, metric_col=metric_col, min_per_bin=3)
        if len(cur) < min_total_pts:
            continue
        br = _two_segment_break(cur["x"].to_numpy(), cur["y"].to_numpy(),
                                min_seg=min_seg, slope_floor=slope_floor)
        if br:
            boot.append(br["x0"])
    if not boot:
        return (np.nan, np.nan, np.nan)
    lo, med, hi = np.nanpercentile(boot, [2.5, 50, 97.5])
    return (med, lo, hi)

def compute_breakpoints_for_area_LS(df_area: pd.DataFrame, bins: np.ndarray, slope_floor: float, B: int, use_abs_bias: bool) -> pd.DataFrame:
    """
    Compute knees ONLY for method == 'least_squares', for both RMSE and bias (|bias| if use_abs_bias).
    Returns one row per (area, pair_ref, pair_sec, method=least_squares) with both sets of columns.
    """
    rows = []
    df_ls = df_area[df_area["method"] == "least_squares"].copy()
    if df_ls.empty:
        return pd.DataFrame(columns=[
            "area","pair_ref","pair_sec","method",
            "critical_density_rmse_km2_per_g","critical_rmse_cm","critical_slope_rmse_left","critical_slope_rmse_right","critical_ci95_rmse_lo","critical_ci95_rmse_hi",
            "critical_density_bias_km2_per_g","critical_bias_cm","critical_slope_bias_left","critical_slope_bias_right","critical_ci95_bias_lo","critical_ci95_bias_hi",
        ])
    keys = df_ls[["area","pair_ref","pair_sec"]].drop_duplicates()
    bias_col = "bias_cm"
    for ar, ref, sec in keys.itertuples(index=False, name=None):
        dfp = df_ls[(df_ls["area"]==ar) & (df_ls["pair_ref"]==ref) & (df_ls["pair_sec"]==sec)]

        # ---------- RMSE ----------
        cur_r = _binned_curve(dfp, bins=bins, metric_col="rmse_cm", min_per_bin=3)
        rm = {"critical_density_rmse_km2_per_g": np.nan, "critical_rmse_cm": np.nan,
              "critical_slope_rmse_left": np.nan, "critical_slope_rmse_right": np.nan,
              "critical_ci95_rmse_lo": np.nan, "critical_ci95_rmse_hi": np.nan}
        if len(cur_r) >= KNEE_MIN_TOTAL_POINTS:
            br_r = _two_segment_break(cur_r["x"].to_numpy(), cur_r["y"].to_numpy(),
                                      min_seg=KNEE_MIN_SEG_POINTS, slope_floor=slope_floor)
            if br_r:
                ix = int(np.argmin(np.abs(cur_r["x"].to_numpy() - br_r["x0"])))
                rm.update({
                    "critical_density_rmse_km2_per_g": float(br_r["x0"]),
                    "critical_rmse_cm": float(cur_r["y"].iloc[ix]),
                    "critical_slope_rmse_left": float(br_r["sl"]),
                    "critical_slope_rmse_right": float(br_r["sr"]),
                })
                med, lo, hi = _bootstrap_break_bins(dfp, bins=bins, metric_col="rmse_cm", B=B, slope_floor=slope_floor)
                rm["critical_ci95_rmse_lo"] = lo
                rm["critical_ci95_rmse_hi"] = hi

        # ---------- BIAS ----------
        dfp_bias = dfp.copy()
        if use_abs_bias:
            dfp_bias["bias_use_cm"] = np.abs(dfp_bias[bias_col].astype(float))
        else:
            dfp_bias["bias_use_cm"] = dfp_bias[bias_col].astype(float)

        cur_b = _binned_curve(dfp_bias, bins=bins, metric_col="bias_use_cm", min_per_bin=3)
        bm = {"critical_density_bias_km2_per_g": np.nan, "critical_bias_cm": np.nan,
              "critical_slope_bias_left": np.nan, "critical_slope_bias_right": np.nan,
              "critical_ci95_bias_lo": np.nan, "critical_ci95_bias_hi": np.nan}
        if len(cur_b) >= KNEE_MIN_TOTAL_POINTS:
            br_b = _two_segment_break(cur_b["x"].to_numpy(), cur_b["y"].to_numpy(),
                                      min_seg=KNEE_MIN_SEG_POINTS, slope_floor=slope_floor)
            if br_b:
                ix = int(np.argmin(np.abs(cur_b["x"].to_numpy() - br_b["x0"])))
                bm.update({
                    "critical_density_bias_km2_per_g": float(br_b["x0"]),
                    "critical_bias_cm": float(cur_b["y"].iloc[ix]),
                    "critical_slope_bias_left": float(br_b["sl"]),
                    "critical_slope_bias_right": float(br_b["sr"]),
                })
                med, lo, hi = _bootstrap_break_bins(dfp_bias, bins=bins, metric_col="bias_use_cm", B=B, slope_floor=slope_floor)
                bm["critical_ci95_bias_lo"] = lo
                bm["critical_ci95_bias_hi"] = hi

        rows.append({
            "area": ar, "pair_ref": ref, "pair_sec": sec, "method": "least_squares",
            **rm, **bm
        })
    return pd.DataFrame(rows)

def summarize_critical_by_area_dual(df_bp: pd.DataFrame) -> pd.DataFrame:
    if df_bp.empty:
        return pd.DataFrame(columns=[
            "area","method","pairs",
            "rmse_critical_density_median","rmse_critical_density_IQR_lo","rmse_critical_density_IQR_hi",
            "bias_critical_density_median","bias_critical_density_IQR_lo","bias_critical_density_IQR_hi",
        ])
    def q25(s): return np.nanpercentile(s, 25)
    def q75(s): return np.nanpercentile(s, 75)
    g = (df_bp.groupby(["area","method"], as_index=False)
             .agg(
                 pairs=("critical_density_rmse_km2_per_g","size"),
                 rmse_critical_density_median=("critical_density_rmse_km2_per_g","median"),
                 rmse_critical_density_IQR_lo=("critical_density_rmse_km2_per_g", q25),
                 rmse_critical_density_IQR_hi=("critical_density_rmse_km2_per_g", q75),
                 bias_critical_density_median=("critical_density_bias_km2_per_g","median"),
                 bias_critical_density_IQR_lo=("critical_density_bias_km2_per_g", q25),
                 bias_critical_density_IQR_hi=("critical_density_bias_km2_per_g", q75),
             ))
    return g

# ============================= Core per-pair run =============================
def _evaluate_pair_single_raster_and_exports(
    area_dir: Path,
    area_name: str,
    pair_tag: str,
    gauge_csv: Path,
    raster_tif: Path,        # the single chosen raster (DEM+CORR) => SRTM+RAW
    dem: str,
    corr: str,
    n_repl: int,
    seed: int,
    idw_power: float,
    spread_top_m: int,
) -> pd.DataFrame:

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
    dh_all = g.set_index(ID_COL).loc[common_ids_sorted, "dh_cm"].to_numpy(dtype=float)

    # Center (closest to centroid)
    lon_c, lat_c = float(lon_all.mean()), float(lat_all.mean())
    _, _, d_center = GEOD.inv(np.full_like(lon_all, lon_c), np.full_like(lat_all, lat_c), lon_all, lat_all)
    center_idx_global = int(np.argmin(d_center))

    rng_master = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []
    export_plan = None

    for rep in range(1, n_repl + 1):
        rng = np.random.default_rng(rng_master.integers(0, 2**31-1))
        all_idx = np.arange(N, dtype=int)
        available_idx = np.setdiff1d(all_idx, np.array([center_idx_global]), assume_unique=False)

        # Initial ~60% calibration (stochastic farthest-point)
        n_cal0 = max(1, int(round(0.60 * len(available_idx))))
        n_cal0 = min(n_cal0, len(available_idx))
        cal_local = _spread_selection_stochastic(
            lon_all[available_idx], lat_all[available_idx], n_cal0, rng=rng, top_m=spread_top_m
        )
        cal_idx = available_idx[cal_local]
        val_idx = np.setdiff1d(available_idx, cal_idx, assume_unique=False)

        # Ensure validation not empty for tiny N
        if len(val_idx) == 0 and len(cal_idx) >= 2:
            crowded = _crowded_candidates(lon_all, lat_all, cal_idx, keep_global=center_idx_global, top_n=4)
            move_pos = crowded[0] if crowded.size else 0
            val_idx = np.r_[val_idx, [cal_idx[move_pos]]]
            cal_idx = np.delete(cal_idx, move_pos)

        # Save export plan (replicate #1)
        if rep == 1:
            export_plan = {"cal60_idx": cal_idx.copy(), "val_idx": val_idx.copy()}

        # March down to 2 gauges: LS + IDW (same cal/val each step)
        cur_idx = cal_idx.copy()
        while len(cur_idx) >= 2:
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

            crowded = _crowded_candidates(lon_all, lat_all, cur_idx, keep_global=center_idx_global, top_n=4)
            drop_pos = int(rng.choice(crowded)) if crowded.size else 0
            cur_idx = np.delete(cur_idx, drop_pos)

        # Single-gauge (center) — LS + IDW (same val set)
        stack_idx_1 = np.r_[ [center_idx_global], val_idx ]
        pts_1 = pd.DataFrame({
            ID_COL: [common_ids_sorted[i] for i in stack_idx_1],
            LON_COL: lon_all[stack_idx_1],
            LAT_COL: lat_all[stack_idx_1],
            "insar_cm": insar_all[stack_idx_1],
            "dh_cm": dh_all[stack_idx_1],
        }).set_index(ID_COL)

        mm1 = _eval_ls_and_idw(pts_1, np.array([0], dtype=int), np.arange(1, pts_1.shape[0]), idw_power=idw_power)

        base_1 = {
            "area": area_name, "pair_ref": _pair_dates_from_tag(pair_tag)[0], "pair_sec": _pair_dates_from_tag(pair_tag)[1],
            "dem": dem, "corr": corr, "replicate": rep,
            "n_total": N, "n_cal": 1, "n_val": int(len(val_idx)),
            "area_km2": float(area_km2),
            "density_km2_per_gauge": float(area_km2) / 1.0,
        }
        records.append({**base_1, "method": "least_squares", **mm1["least_squares"]})
        records.append({**base_1, "method": "idw_dhvis",    **mm1["idw_dhvis"], "a_gain": np.nan, "b_offset_cm": np.nan})

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
        a1, b1 = _fit_ls_params(insar_all[[center_idx_global]], dh_all[[center_idx_global]])
        arr1 = _apply_calibration_to_raster(raster_tif, a1, b1)
        out_cal1 = results_dir / f"dens_cal_1g_{dem}_{corr}_{pair_tag}.tif"
        _write_tif_like(raster_tif, out_cal1, arr1)
        print(f"  🗺️  Calibrated (1 gauge) written: {out_cal1}")
    except Exception as e:
        print(f"  ⚠️  Calibrated (1 gauge) failed for {pair_tag}: {e}")

    return pd.DataFrame.from_records(records)

# ================================ Area driver ================================
def _process_area(area_dir: Path,
                  reps: int, seed: int, idw_power: float,
                  knee_bootstrap: int, knee_bins_spec: str, knee_slope_floor: float,
                  bias_signed: bool, spread_top_m: int) -> None:
    area_name   = area_dir.name
    dem         = DEM_FIXED
    corr        = CORR_FIXED

    gauge_csv   = area_dir / "water_gauges" / "eden_gauges.csv"
    results_dir = area_dir / "results"
    metrics_csv = results_dir / f"accuracy_metrics_density_{dem}_{corr}.csv"
    pairs_bp_csv= results_dir / f"critical_density_pairs_{dem}_{corr}.csv"
    area_bp_csv = results_dir / f"critical_density_area_{dem}_{corr}.csv"

    if not gauge_csv.exists():
        print(f"⏭️  Gauge CSV missing for {area_name}: {gauge_csv} — skipping area.")
        return

    pairs = _find_pairs_for_dem_corr(area_dir, area_name, dem, corr)
    if not pairs:
        print(f"⏭️  No interferograms for {area_name} with DEM={dem} CORR={corr} — skipping area.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    bins = _parse_knee_bins(knee_bins_spec)
    use_abs_bias = not bias_signed

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
                spread_top_m=spread_top_m,
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

    # ---- Compute & inject critical density (LS only) for RMSE and bias ----
    try:
        bp_df = compute_breakpoints_for_area_LS(
            df_area=df_all,
            bins=bins,
            slope_floor=knee_slope_floor,
            B=knee_bootstrap,
            use_abs_bias=use_abs_bias
        )
        if not bp_df.empty:
            # Merge only into LS rows; IDW rows will retain NaNs in knee columns
            df_all = df_all.merge(bp_df, on=["area","pair_ref","pair_sec","method"], how="left")
            bp_df.to_csv(pairs_bp_csv, index=False)
            print(f"📄 Critical density (LS only; RMSE & bias) → {pairs_bp_csv}  (rows: {len(bp_df)})")
            area_sum = summarize_critical_by_area_dual(bp_df)
            area_sum.to_csv(area_bp_csv, index=False)
            print(f"📄 Critical density (area summary; RMSE & bias) → {area_bp_csv}")
        else:
            # Ensure columns exist (as NaN) for schema stability
            for col in [
                "critical_density_rmse_km2_per_g","critical_rmse_cm","critical_slope_rmse_left","critical_slope_rmse_right","critical_ci95_rmse_lo","critical_ci95_rmse_hi",
                "critical_density_bias_km2_per_g","critical_bias_cm","critical_slope_bias_left","critical_slope_bias_right","critical_ci95_bias_lo","critical_ci95_bias_hi",
            ]:
                df_all[col] = np.nan
            print("ℹ️  No critical-density breakpoints found for LS in this area.")
    except Exception as e:
        print(f"⚠️  Critical-density computation failed: {e}")

    df_all.to_csv(metrics_csv, index=False)
    print(f"\n✅ [{area_name}] metrics written (fresh): {metrics_csv}  (rows: {len(df_all)})")

# =================================== CLI ====================================
def main():
    ap = argparse.ArgumentParser(
        description="SRTM+RAW accuracy & density assessment. IDW uses SAME gauges/splits as RAW. Knees computed ONLY for least_squares (RMSE & bias)."
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
    ap.add_argument("--spread-top-m", type=int, default=SPREAD_TOP_M_DEFAULT,
                    help="Farthest-point selection: pick randomly among top-M farthest at each add step (default: %(default)s)")
    # Critical-density detector
    ap.add_argument("--knee-bootstrap", type=int, default=KNEE_BOOTSTRAP_DEFAULT,
                    help="Bootstrap resamples for knee CI (default: %(default)s)")
    ap.add_argument("--knee-bins", type=str, default=KNEE_BINS_DEFAULT,
                    help="Density bins as 'start:end:step,...' in km²/gauge (default: %(default)s)")
    ap.add_argument("--knee-slope-floor", type=float, default=KNEE_SLOPE_FLOOR_DEFAULT,
                    help="Minimum slope increase across breakpoint (cm per km²/gauge) (default: %(default)s)")
    ap.add_argument("--bias-signed", action="store_true",
                    help="Use SIGNED bias for knee (default: absolute bias).")
    args = ap.parse_args()

    root = Path(args.areas_root)
    targets = [root / args.area] if args.area else sorted([d for d in root.iterdir() if d.is_dir()])

    for area_dir in targets:
        _process_area(
            area_dir=area_dir,
            reps=args.reps,
            seed=args.seed,
            idw_power=args.idw_power,
            knee_bootstrap=args.knee_bootstrap,
            knee_bins_spec=args.knee_bins,
            knee_slope_floor=args.knee_slope_floor,
            bias_signed=args.bias_signed,
            spread_top_m=args.spread_top_m,
        )

if __name__ == "__main__":
    main()
