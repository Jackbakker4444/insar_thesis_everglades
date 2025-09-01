#!/usr/bin/env python3
"""
Gauge-density accuracy sweep for a single vertical-displacement interferogram.

Overview
--------
We evaluate accuracy vs. calibration gauge density by:
  • Fixing ~40% of gauges as validation (held out) and ~60% as calibration,
    using a spread-out selection (farthest-point sampling).
  • Iteratively removing one “crowded” calibration gauge at a time,
    recomputing accuracy each step,
    while the center gauge is EXCLUDED from calibration and validation.
  • Final step: use ONLY the center gauge as the single calibration point (n_cal=1).
  • If only one calibration gauge is available, we force a = -1 and fit to b
    so the calibration point is matched exactly.
  • Repeating the above 20–100 times (replicates) with different random seeds.

Methods:
  • Least-squares: calibrate InSAR to visible-surface Δh (Δh_vis).
  • IDW: interpolate gauge Δh_vis to the validation gauges.

Density axis:
  • Uses the **valid interferogram area only** (derived from its data mask).
  • Density = (valid area in km²) / (# calibration gauges), lower = denser.
  • Bottom x-axis: log density; top x-axis: number of calibration gauges.

Outputs (per AREA):
  • areas/<AREA>/results/accuracy_metrics.csv     # all rows, includes 'area' and 'dem'
  • areas/<AREA>/results/acc_den_<REF>_<SEC>_<DEMTYPE>.png
  • areas/<AREA>/results/acc_period_<AREA>_<DEMTYPE>_<DENSITY>.png
"""

from __future__ import annotations
from pathlib import Path
import re, os, logging, math, datetime as dt
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.features import shapes
from pyproj import Geod
import matplotlib as mpl
mpl.set_loglevel("warning")  # keep Matplotlib quiet
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
import matplotlib.dates as mdates


# Silence GDAL/Rasterio & font manager chatter
os.environ.setdefault("CPL_DEBUG", "NO")
for name in ("rasterio", "rasterio._io", "rasterio.env", "rasterio._base", "matplotlib.font_manager"):
    logging.getLogger(name).setLevel(logging.ERROR)

# ---------------------------- CONFIG -----------------------------------------
# Inputs for THIS area/pair
GAUGE_CSV   = Path("/mnt/DATA2/bakke326l/processing/areas/ENP/water_gauges/eden_gauges.csv")
RASTER_TIF  = Path("/mnt/DATA2/bakke326l/processing/areas/ENP/ENP_vertical_cm_20071216_20080131.tif")

# Tag the DEM used for this interferogram (appears in filenames & CSV)
DEM_TYPE    = "SRTM"      # <- set "SRTM" or "3DEP" (or another label you prefer)

# Density target (km² per gauge) for the time-series figure
TARGET_DENSITY_KM2 = 150.0

# Replicates / algorithm knobs
ID_COL, LAT_COL, LON_COL = "StationID", "Lat", "Lon"
N_REPLICATES, RANDOM_SEED = 100, 42
IDW_POWER = 2.0

# ---------------------------- HELPERS ----------------------------------------
GEOD = Geod(ellps="WGS84")

def parse_pair_dates_from_path(tif_path: Path) -> Tuple[str, str, str]:
    """
    Returns: (ref_iso, sec_iso, pair_raw) where pair_raw is 'YYYYMMDD_YYYYMMDD'.
    """
    m = re.search(r"(\d{8})_(\d{8})", tif_path.name)
    if not m:
        raise ValueError(f"Could not parse dates from filename: {tif_path.name}")
    ref_raw, sec_raw = m.group(1), m.group(2)
    ref_iso = f"{ref_raw[:4]}-{ref_raw[4:6]}-{ref_raw[6:]}"
    sec_iso = f"{sec_raw[:4]}-{sec_raw[4:6]}-{sec_raw[6:]}"
    return ref_iso, sec_iso, f"{ref_raw}_{sec_raw}"

def load_gauges_wide(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in (ID_COL, LAT_COL, LON_COL):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")
    return df

def rowcol_from_xy(transform: Affine, x: float, y: float) -> Tuple[float, float]:
    col, row = ~transform * (x, y)
    return float(row), float(col)

def inside_image(h: int, w: int, row: float, col: float) -> bool:
    return (row >= 0) and (col >= 0) and (row < h) and (col < w)

def read_mean_3x3(ds: rasterio.io.DatasetReader, row: int, col: int) -> Optional[float]:
    r0 = max(0, row - 1); r1 = min(ds.height - 1, row + 1)
    c0 = max(0, col - 1); c1 = min(ds.width  - 1, col + 1)
    arr = ds.read(1, window=Window.from_slices((r0, r1 + 1), (c0, c1 + 1))).astype("float32")
    if ds.nodata is not None and not np.isnan(ds.nodata):
        arr[arr == ds.nodata] = np.nan
    if not np.isfinite(arr).any(): return None
    return float(np.nanmean(arr))

def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    A = np.c_[x, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    return {
        "rmse_cm": float(np.sqrt(np.mean(err**2))),
        "mae_cm":  float(np.mean(np.abs(err))),
        "bias_cm": float(np.mean(err)),
        "r":       float(np.corrcoef(y_true, y_pred)[0,1]) if len(y_true) >= 2 else np.nan,
    }

def visible_surface_delta(ref_cm: np.ndarray, sec_cm: np.ndarray) -> np.ndarray:
    return np.maximum(sec_cm.astype(float), 0.0) - np.maximum(ref_cm.astype(float), 0.0)

# ---- IDW ---------------------------------------------------------------------
def _idw_predict_points(px, py, pz, qx, qy, power: float = 2.0) -> np.ndarray:
    px = px.astype("float64"); py = py.astype("float64"); pz = pz.astype("float64")
    qx = qx.astype("float64"); qy = qy.astype("float64")
    cx = np.cos(np.deg2rad(np.nanmean(py)))
    dx = (qx[:, None] - px[None, :]) * cx
    dy = (qy[:, None] - py[None, :])
    d2 = dx*dx + dy*dy
    w  = 1.0 / np.maximum(d2, 1e-18) ** (power/2.0)
    pred = (w @ pz) / np.sum(w, axis=1)
    imin = np.argmin(d2, axis=1)
    pred[d2[np.arange(d2.shape[0]), imin] < 1e-18] = pz[imin[d2[np.arange(d2.shape[0]), imin] < 1e-18]]
    return pred.astype("float32")

# ---- geometry / selection ----------------------------------------------------
def _haversine_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(lon.astype("float64")); lat = np.deg2rad(lat.astype("float64"))
    sin_lat = np.sin(lat)[:, None]; cos_lat = np.cos(lat)[:, None]
    sin_lat_T = np.sin(lat)[None, :]; cos_lat_T = np.cos(lat)[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.clip(sin_lat * sin_lat_T + cos_lat * cos_lat_T * np.cos(dlon), -1.0, 1.0)
    return 6371000.0 * np.arccos(a)

def _spread_selection(lon: np.ndarray, lat: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
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
    """Return subset positions (within idx) of the top-N most crowded gauges, excluding the keep_global index."""
    if len(idx) <= 1: return idx
    lon_s, lat_s = lon[idx], lat[idx]
    D = _haversine_matrix(lon_s, lat_s)
    np.fill_diagonal(D, np.inf)
    nnd = np.min(D, axis=1)
    order = np.argsort(nnd)  # smallest first
    order = np.array([o for o in order if idx[o] != keep_global], dtype=int)
    if order.size == 0: return np.array([], dtype=int)
    return order[:min(top_n, order.size)]

def _geod_area_of_geojson(geom) -> float:
    """Geodesic area (km²) of a GeoJSON Polygon/MultiPolygon in EPSG:4326."""
    def poly_area(coords):
        area_km2 = 0.0
        if not coords: return 0.0
        lon, lat = zip(*coords[0])                       # exterior
        a_ext, _ = GEOD.polygon_area_perimeter(lon, lat)
        area_km2 += abs(a_ext) / 1e6
        for ring in coords[1:]:                          # holes
            lon, lat = zip(*ring)
            a_hole, _ = GEOD.polygon_area_perimeter(lon, lat)
            area_km2 -= abs(a_hole) / 1e6
        return max(area_km2, 0.0)
    typ = geom.get("type")
    if typ == "Polygon":
        return poly_area(geom["coordinates"])
    elif typ == "MultiPolygon":
        return sum(poly_area(coords) for coords in geom["coordinates"])
    return 0.0

def _valid_raster_area_km2(ds: rasterio.io.DatasetReader) -> float:
    """Sum geodesic area of all valid-data polygons from dataset_mask()==255."""
    if ds.crs is None or ds.crs.to_epsg() != 4326:
        raise RuntimeError("Expected raster in EPSG:4326 for geodesic area.")
    mask = (ds.dataset_mask() == 255).astype(np.uint8)
    area = 0.0
    for geom, val in shapes(mask, transform=ds.transform):
        if val == 1:
            area += _geod_area_of_geojson(geom)
    return float(area)

# ---------------------------- CORE EVALS -------------------------------------
def _eval_ls_and_idw(pts: pd.DataFrame, cal_idx: np.ndarray, val_idx: np.ndarray) -> Dict[str, float]:
    # --- LS (InSAR -> Δh_vis)
    x_cal = pts["insar_cm"].values[cal_idx].astype(float)
    y_cal = pts["dh_cm"].values[cal_idx].astype(float)
    x_val = pts["insar_cm"].values[val_idx].astype(float)
    y_val = pts["dh_cm"].values[val_idx].astype(float)

    n_unique = len(np.unique(pts[ID_COL].values[cal_idx]))
    if n_unique >= 2:
        a, b = fit_affine(x_cal, y_cal)
    elif n_unique == 1:
        a = -1.0
        b = float(y_cal[0] - a * x_cal[0])   # = y1 + x1
    else:
        a, b = np.nan, np.nan

    if np.isfinite(a) and np.isfinite(b):
        y_pred_ls = a * x_val + b
        m_ls = compute_metrics(y_true=y_val, y_pred=y_pred_ls)
        m_ls.update({"a_gain": float(a), "b_offset_cm": float(b)})
    else:
        m_ls = {"rmse_cm": np.nan, "mae_cm": np.nan, "bias_cm": np.nan, "r": np.nan,
                "a_gain": np.nan, "b_offset_cm": np.nan}

    # --- IDW (gauges -> Δh_vis)
    px = pts[LON_COL].values[cal_idx].astype(float)
    py = pts[LAT_COL].values[cal_idx].astype(float)
    pz = pts["dh_cm"].values[cal_idx].astype(float)
    qx = pts[LON_COL].values[val_idx].astype(float)
    qy = pts[LAT_COL].values[val_idx].astype(float)
    y_pred_idw = _idw_predict_points(px, py, pz, qx, qy, power=IDW_POWER)
    m_idw = compute_metrics(y_true=y_val, y_pred=y_pred_idw)

    return m_ls, m_idw

# ---------------------------- PIPELINE ---------------------------------------
def main():
    # Resolve AREA + pair tags + results dir
    area_name = RASTER_TIF.parent.name
    ref_str, sec_str, pair_raw = parse_pair_dates_from_path(RASTER_TIF)
    results_dir = RASTER_TIF.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # File outputs (per-pair figs + area-wide CSV + time-series fig)
    FIG_DEN_PATH = results_dir / f"acc_den_{pair_raw}_{DEM_TYPE}.png"
    METRICS_CSV  = results_dir / "accuracy_metrics.csv"   # area-wide growing file
    # time-series figure (across pairs in this AREA)
    dens_tag = f"{TARGET_DENSITY_KM2:g}".replace(".", "p")
    FIG_PERIOD_PATH = results_dir / f"acc_period_{area_name}_{DEM_TYPE}_{dens_tag}.png"

    # Gauges → Δh_vis
    gauges = load_gauges_wide(GAUGE_CSV)
    for col in (ref_str, sec_str):
        if col not in gauges.columns:
            raise ValueError(f"Gauge CSV is missing date column: {col}")
    g = gauges[[ID_COL, LAT_COL, LON_COL, ref_str, sec_str]].copy()
    g = g.rename(columns={ref_str: "ref_cm", sec_str: "sec_cm"})
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["ref_cm", "sec_cm", LAT_COL, LON_COL])
    g["dh_cm"] = visible_surface_delta(g["ref_cm"].to_numpy(), g["sec_cm"].to_numpy())

    # Sample InSAR at gauges **and keep only gauges with valid pixels**
    with rasterio.open(RASTER_TIF) as ds:
        if ds.crs is None or (ds.crs.to_epsg() != 4326):
            raise RuntimeError("Expected raster in EPSG:4326.")
        # valid-area (km²) from the mask (255 = data)
        area_km2 = _valid_raster_area_km2(ds)
        if not np.isfinite(area_km2) or area_km2 <= 0:
            raise RuntimeError("Could not derive valid interferogram area.")
        rows = []
        for _, r in g.iterrows():
            x, y = float(r[LON_COL]), float(r[LAT_COL])
            rowf, colf = rowcol_from_xy(ds.transform, x, y)
            if not inside_image(ds.height, ds.width, rowf, colf): continue
            ins = read_mean_3x3(ds, int(round(rowf)), int(round(colf)))
            if ins is None or not np.isfinite(ins): continue
            rows.append({ID_COL: r[ID_COL], LON_COL: x, LAT_COL: y,
                         "insar_cm": float(ins), "dh_cm": float(r["dh_cm"])})
    pts = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["insar_cm", "dh_cm"])
    if pts[ID_COL].nunique() < 3 or len(pts) < 3:
        raise RuntimeError(f"Too few usable gauges inside valid raster area: {pts[ID_COL].nunique()} stations")

    lon_all, lat_all = pts[LON_COL].values, pts[LAT_COL].values

    # Choose center gauge (closest to centroid) — EXCLUDED until final n_cal=1 step
    lon_c, lat_c = lon_all.mean(), lat_all.mean()
    _, _, d_center = GEOD.inv(np.full_like(lon_all, lon_c), np.full_like(lat_all, lat_c), lon_all, lat_all)
    center_idx_global = int(np.argmin(d_center))

    # Replicates
    rng_master = np.random.default_rng(RANDOM_SEED)
    records: List[Dict[str, float]] = []
    N = len(pts)
    n_cal0 = max(1, int(round(0.60 * N)))  # ~60% calibration target

    for rep in range(1, N_REPLICATES + 1):
        rng = np.random.default_rng(rng_master.integers(0, 2**31-1))

        # --- Build initial sets EXCLUDING the center gauge entirely
        all_idx = np.arange(N, dtype=int)
        available_idx = np.setdiff1d(all_idx, np.array([center_idx_global]), assume_unique=False)

        n_cal0_eff = min(n_cal0, len(available_idx))
        cal_local = _spread_selection(lon_all[available_idx], lat_all[available_idx], n_cal0_eff, rng)
        cal_idx = available_idx[cal_local]
        val_idx = np.setdiff1d(available_idx, cal_idx, assume_unique=False)

        # If validation accidentally ends up empty (tiny N), move one from cal → val
        if len(val_idx) == 0 and len(cal_idx) >= 2:
            crowded = _crowded_candidates(lon_all, lat_all, cal_idx, keep_global=center_idx_global, top_n=4)
            move_pos = crowded[0] if crowded.size else 0
            val_idx = np.r_[val_idx, [cal_idx[move_pos]]]
            cal_idx = np.delete(cal_idx, move_pos)

        # --- March down from current cal set to size 2 (center still excluded)
        cur_idx = cal_idx.copy()
        while len(cur_idx) >= 2:
            m_ls, m_idw = _eval_ls_and_idw(pts, cur_idx, val_idx)
            base = {
                "replicate": rep, "n_total": N, "n_val": int(len(val_idx)),
                "n_cal": int(len(cur_idx)), "area_km2": area_km2,
                "area_per_gauge_km2": area_km2 / float(len(cur_idx)),
                "pair_ref": ref_str, "pair_sec": sec_str,
                "area": area_name, "dem": DEM_TYPE,
            }
            records.extend([
                {**base, "method": "least_squares", **m_ls},
                {**base, "method": "idw_dhvis", **m_idw},
            ])

            # drop one crowded non-center gauge
            crowded = _crowded_candidates(lon_all, lat_all, cur_idx, keep_global=center_idx_global, top_n=4)
            drop_pos = int(rng.choice(crowded)) if crowded.size else 0
            cur_idx = np.delete(cur_idx, drop_pos)

        # --- Final single-gauge step: use ONLY the center gauge
        m_ls, m_idw = _eval_ls_and_idw(pts, np.array([center_idx_global], dtype=int), val_idx)
        base = {
            "replicate": rep, "n_total": N, "n_val": int(len(val_idx)),
            "n_cal": 1, "area_km2": area_km2, "area_per_gauge_km2": area_km2 / 1.0,
            "pair_ref": ref_str, "pair_sec": sec_str,
            "area": area_name, "dem": DEM_TYPE,
        }
        records.extend([
            {**base, "method": "least_squares", **m_ls},
            {**base, "method": "idw_dhvis", **m_idw},
        ])

    # Save CSV (area-wide file)
    df_out = pd.DataFrame.from_records(records)
    if METRICS_CSV.exists():
        df_out.to_csv(METRICS_CSV, mode="a", header=False, index=False)
    else:
        df_out.to_csv(METRICS_CSV, index=False)
    print(f"✅ Metrics written: {METRICS_CSV}  (rows appended: {len(df_out)})")

    # ----- Terminal summary: mean RMSE vs density (per method) -----
    for method in ("least_squares", "idw_dhvis"):
        sub = df_out[(df_out["method"] == method) & np.isfinite(df_out["rmse_cm"])].copy()
        grp = sub.groupby("n_cal", as_index=False).agg(
            mean_rmse=("rmse_cm", "mean"),
            mean_density=("area_per_gauge_km2", "mean"),
            nrepl=("rmse_cm", "count"),
        ).sort_values("mean_density")
        print(f"\n▶ Mean RMSE vs density — {method} [{DEM_TYPE}] ({area_name}):")
        for _, r in grp.iterrows():
            print(f"  n_cal={int(r['n_cal']):3d}  density={r['mean_density']:.3f} km²/g  "
                  f"RMSE={r['mean_rmse']:.2f} cm  (n={int(r['nrepl'])})")

    # ----- Plot: double x-axes (bottom = density [log], top = #gauges) -----
    # Aggregate BY ACTUAL n_cal for each method
    agg = {}
    for method in ("least_squares", "idw_dhvis"):
        sub = df_out[(df_out["method"] == method) & np.isfinite(df_out["rmse_cm"])].copy()
        grp = sub.groupby("n_cal", as_index=False).agg(
            mean_rmse=("rmse_cm", "mean"),
            p2_5=("rmse_cm", lambda x: np.nanpercentile(x, 2.5)),
            p97_5=("rmse_cm", lambda x: np.nanpercentile(x, 97.5)),
        )
        grp["density"] = area_km2 / grp["n_cal"].astype(float)  # km² per gauge
        agg[method] = grp

    # Use ONLY the n_cal values that exist for BOTH methods (so curves end together)
    common_ncal = sorted(
        set(agg["least_squares"]["n_cal"]).intersection(set(agg["idw_dhvis"]["n_cal"])),
        key=lambda n: area_km2 / float(n)  # sort by increasing density (left=denser)
    )
    if not common_ncal:
        raise RuntimeError("No overlapping n_cal between methods to plot.")

    plt.figure(figsize=(8.8, 5.6), dpi=140)
    ax = plt.gca()

    # robust y-clip: cap axis but annotate clipped values
    ls_mask = agg["least_squares"]["n_cal"].isin(common_ncal)
    idw_mask = agg["idw_dhvis"]["n_cal"].isin(common_ncal)
    all_p97 = np.concatenate([
        agg["least_squares"].loc[ls_mask, "p97_5"].to_numpy(),
        agg["idw_dhvis"].loc[idw_mask, "p97_5"].to_numpy(),
    ])
    all_means = np.concatenate([
        agg["least_squares"].loc[ls_mask, "mean_rmse"].to_numpy(),
        agg["idw_dhvis"].loc[idw_mask, "mean_rmse"].to_numpy(),
    ])
    y_top = float(np.nanpercentile(all_p97, 98))
    y_top = max(y_top, float(np.nanmax(all_means)) * 1.10)
    y_bottom = 0.0

    colors = {"least_squares": "#1f77b4", "idw_dhvis": "#ff7f0e"}
    for method, label in [
        ("least_squares", f"Least-squares (Calibrated InSAR) — {DEM_TYPE}"),
        ("idw_dhvis",     f"Interpolation (IDW gauges) — {DEM_TYPE}"),
    ]:
        g = agg[method]
        g = g[g["n_cal"].isin(common_ncal)].copy().sort_values("density")
        ax.plot(g["density"], g["mean_rmse"], label=label,
                color=colors[method], marker="o", ms=3, lw=1.6, zorder=3)
        ax.fill_between(g["density"], g["p2_5"], g["p97_5"], alpha=0.2, color=colors[method], zorder=1)

        exceed = g["p97_5"] > y_top
        if np.any(exceed):
            xd = g.loc[exceed, "density"].to_numpy()
            yt = np.full(int(exceed.sum()), y_top, dtype=float)
            true_vals = g.loc[exceed, "p97_5"].to_numpy()
            ax.scatter(xd, yt*0.995, marker="^", s=16, color=colors[method], edgecolor="none", zorder=4)
            for x0, vtrue in zip(xd, true_vals):
                ax.annotate(f"↑ {vtrue:.1f} cm", xy=(x0, y_top),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    # Bottom axis (density)
    ax.set_xscale("log")
    dens_ticks_all = np.array([area_km2 / float(n) for n in common_ncal], dtype=float)
    xmin, xmax = float(dens_ticks_all.min()), float(dens_ticks_all.max())
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y_bottom, y_top)

    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", labelsize=9)

    ax.set_xlabel("Gauge density (km² per gauge) — lower is denser  [log scale]")
    ax.set_ylabel("Accuracy (RMSE, cm)")
    ax.set_title(f"{area_name}: Accuracy vs Gauge Density (mean ± 95% range; {N_REPLICATES} replicates)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # Top x-axis with ACTUAL number of gauges, aligned to density ticks
    ax_top = ax.twiny()
    ax_top.set_xscale("log")
    ax_top.set_xlim(ax.get_xlim())

    # Thin out top ticks to avoid overlap (keep ≤ 10; include endpoints)
    max_ticks = 10
    if len(dens_ticks_all) > max_ticks:
        idx = np.unique(np.linspace(0, len(dens_ticks_all) - 1, max_ticks, dtype=int))
        top_ticks = dens_ticks_all[idx]
        top_labels = [str(int(common_ncal[i])) for i in idx]
    else:
        top_ticks = dens_ticks_all
        top_labels = [str(int(n)) for n in common_ncal]

    ax_top.set_xticks(top_ticks)
    ax_top.set_xticklabels(top_labels)
    ax_top.set_xlabel("Number of calibration gauges")
    ax_top.tick_params(axis="x", labelsize=8)  # smaller font to avoid overlap

    ax.text(0.99, 0.02,
            f"Note: values above {y_top:.1f} cm are clipped; carets show true p97.5.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")

    plt.tight_layout()
    plt.savefig(FIG_DEN_PATH, bbox_inches="tight")
    plt.close()
    print(f"📈 Density figure written: {FIG_DEN_PATH}")

    # -------------------- Time-series figure at TARGET_DENSITY_KM2 (one axis, overlaid bars) --------------------
    # Build AREA/DEM info + output path
    # def _guess_dem_label(p: Path) -> str:
    #     stem = p.stem.lower()
    #     names = [q.name.lower() for q in p.parents] + [stem]
    #     for lab in ("3dep", "srtm"):
    #         if any(lab in n for n in names):
    #             return lab.upper()
    #     return "UNKNOWN"

    area_name  = GAUGE_CSV.parent.parent.name  # .../areas/<AREA>/water_gauges/eden_gauges.csv -> <AREA>
    DEM_TYPE   = "SRTM" #_guess_dem_label(RASTER_TIF)
    results_dir = (GAUGE_CSV.parent.parent / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    fig_period_name = f"acc_period_{area_name}_{DEM_TYPE}_{str(TARGET_DENSITY_KM2).replace('.', 'p')}.png"
    FIG_PERIOD_PATH = results_dir / fig_period_name

    if METRICS_CSV.exists():
        hist = pd.read_csv(METRICS_CSV)
        # keep correct AREA/DEM rows; your file already has these columns from earlier steps
        hist = hist[(hist["area"] == area_name) & (hist["dem"] == DEM_TYPE)]
        if hist.empty:
            print("⚠️ No matching rows in accuracy_metrics.csv for this AREA/DEM; skipping time-series.")
        else:
            # density per row
            hist["density"] = hist["area_km2"] / hist["n_cal"].astype(float)

            # aggregate to mean ± 95% for each (pair, method, n_cal)
            grp = (hist.groupby(["pair_ref", "pair_sec", "method", "n_cal"], as_index=False)
                        .agg(mean_rmse=("rmse_cm", "mean"),
                             p2_5=("rmse_cm", lambda x: np.nanpercentile(x, 2.5)),
                             p97_5=("rmse_cm", lambda x: np.nanpercentile(x, 97.5)),
                             density=("density", "mean")))

            # choose the n_cal whose density is closest to target, per (pair, method)
            rows_ts = []
            for (pref, psec, method), gsub in grp.groupby(["pair_ref", "pair_sec", "method"]):
                gsub = gsub.copy()
                gsub["d_abs"] = (gsub["density"] - TARGET_DENSITY_KM2).abs()
                best = gsub.loc[gsub["d_abs"].idxmin()]
                rows_ts.append({
                    "pair_ref": pref, "pair_sec": psec, "method": method,
                    "density": float(best["density"]),
                    "mean_rmse": float(best["mean_rmse"]),
                    "p2_5": float(best["p2_5"]), "p97_5": float(best["p97_5"]),
                })
            ts = pd.DataFrame(rows_ts)
            if ts.empty:
                print("⚠️ Could not assemble time-series rows; skipping time-series.")
            else:
                # dates & widths
                to_dt = lambda s: pd.to_datetime(s, format="%Y-%m-%d")
                ts["t_ref"] = to_dt(ts["pair_ref"])
                ts["t_sec"] = to_dt(ts["pair_sec"])
                ts["t_mid"] = ts["t_ref"] + (ts["t_sec"] - ts["t_ref"]) / 2
                ts["span_days"] = (ts["t_sec"] - ts["t_ref"]).dt.days.clip(lower=1).astype(float)

                # split by method
                ls = ts[ts["method"] == "least_squares"].copy().sort_values("t_mid")
                idw = ts[ts["method"] == "idw_dhvis"].copy().sort_values("t_mid")

                # one axis, overlay bars
                fig, ax = plt.subplots(figsize=(10, 4.6), dpi=140)

                # colors
                c_ls, c_idw = "#1f77b4", "#ff7f0e"

                # convert dates to numbers for width/offset arithmetic
                x_ls  = mdates.date2num(ls["t_mid"])
                w_ls  = ls["span_days"].to_numpy()          # days
                x_idw = mdates.date2num(idw["t_mid"])
                w_idw = (0.70 * idw["span_days"]).to_numpy()  # slightly narrower so both bars are visible

                # bars (draw LS first, then IDW on top)
                ax.bar(x_ls,  ls["mean_rmse"],  width=w_ls,  align="center",
                       color=c_ls,  alpha=0.28, edgecolor=c_ls,  linewidth=1.0, label="Least-squares (mean)")
                ax.bar(x_idw, idw["mean_rmse"], width=w_idw, align="center",
                       color=c_idw, alpha=0.35, edgecolor=c_idw, linewidth=1.0, label="IDW (mean)")

                # uncertainty lines — nudge horizontally so they don't collide
                # LS to the right, IDW to the left (proportional to bar width; min offset = 0.4 day)
                off_ls_days  = np.maximum(0.15 * w_ls,  0.4)
                off_idw_days = np.maximum(0.15 * w_idw, 0.4)

                ax.vlines(x_ls  + off_ls_days,  ls["p2_5"],  ls["p97_5"],  color=c_ls,  linewidth=1.6, zorder=3)
                ax.vlines(x_idw - off_idw_days, idw["p2_5"], idw["p97_5"], color=c_idw, linewidth=1.6, zorder=3)

                # small mean markers on top of bars
                ax.plot(x_ls,  ls["mean_rmse"],  "o", ms=3.0, color=c_ls,  zorder=4)
                ax.plot(x_idw, idw["mean_rmse"], "o", ms=3.0, color=c_idw, zorder=4)

                # axes cosmetics
                ax.set_ylabel("RMSE (cm)")
                ax.set_xlabel("Time (bar width spans pair dates)")
                ax.grid(True, alpha=0.3)
                ax.legend()

                # calendar axis formatting + limits
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                xmin = min(ts["t_ref"].min(), ts["t_mid"].min()) - pd.Timedelta(days=5)
                xmax = max(ts["t_sec"].max(), ts["t_mid"].max()) + pd.Timedelta(days=5)
                ax.set_xlim((xmin, xmax))

                # title with AREA • DEM • target density
                ax.set_title(f"{area_name} • DEM={DEM_TYPE} • target density ≈ {TARGET_DENSITY_KM2:g} km²/g")

                fig.autofmt_xdate()
                plt.tight_layout()
                plt.savefig(FIG_PERIOD_PATH, bbox_inches="tight")
                plt.close()
                print(f"📈 Time-series figure written: {FIG_PERIOD_PATH}")



if __name__ == "__main__":
    main()
