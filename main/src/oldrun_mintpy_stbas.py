#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MintPy Thesis Runner — simple, commented, and thesis-friendly
=============================================================

What this script does (end-to-end):
  1) Scans your ISCE2 pair folders and writes a MintPy template (smallbaselineApp.cfg)
  2) Runs MintPy's smallbaselineApp pipeline (with unwrap-error + tropospheric options)
  3) Loads MintPy outputs and makes publication-ready figures:
       • Velocity map (m/yr)
       • Temporal coherence map and histogram
       • Residual RMS map (if available)
       • Example displacement maps at first/last dates
       • Time series at your water-gauge locations (from CSV), with optional LOS→vertical conversion
  4) Exports gauge time series as CSV (InSAR extracted at nearest pixel) for analysis

Notes / Assumptions
-------------------
• Input stack: geocoded unwrapped interferograms from ISCE2 (e.g., *.unw.geo)
• Tropospheric correction: uses MintPy built-ins (PyAPS/ERA5 or GACOS) if enabled below
• Ionosphere: MintPy does not subtract ionosphere automatically. If you prepared iono-corrected
  unwrapped rasters (non-dispersive phase), point UNW_GLOB to them. Otherwise, this script
  proceeds without iono subtraction but still produces all visuals.

Keep it simple: everything is in one file, minimal functions, lots of comments.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1) USER SETTINGS — EDIT THESE
# ─────────────────────────────────────────────────────────────────────────────
# Where your pair folders live (those "path150_YYYYMMDD_YYYYMMDD_*" dirs)
BASE_DIR = Path("/mnt/DATA2/bakke326l/processing/tuning")

# Where to write MintPy outputs and figures
PROJECT_DIR = BASE_DIR / "mintpy_path150_project"
FIG_DIR = PROJECT_DIR / "figs"

# Which files to load from each pair directory
# If you have iono-corrected, non-dispersive unwrapped rasters, point UNW_GLOB to those instead
UNW_GLOB = "interferogram/filt_topophase.unw.geo"          # full-band unwrapped (geocoded)
COR_GLOB_PHSIG = "interferogram/phsig.cor.geo"             # preferred weight
COR_GLOB_TOPOCOR = "interferogram/topophase.cor.geo"       # fallback weight

# Geometry directory (MintPy needs latitude/longitude/etc). We'll auto-pick one if present
GEOM_DIR_NAME = "geometry"

# Reference point: pick a stable pixel (levee/platform) near your gauge (NOT open water)
REF_LAT, REF_LON = 25.6000, -80.6000   # set to None,None to let MintPy auto-pick

# Tropospheric correction inside MintPy: "pyaps" (ERA5), "gacos", or "none"
TROPO_METHOD = "gacos"
TROPO_MODEL = "ERA5"                   # for PyAPS

# Unwrap-error correction and ramp removal (recommended)
ENABLE_UNWRAP_ERROR_FIX = True
ENABLE_DERAMP = True

# Parallelism (0/1 disables MintPy parallel; >1 enables Dask workers)
DASK_WORKERS = 8

# Gauges CSV and plotting
GAUGE_CSV = Path("/home/bakke326l/InSAR/main/data/aux/gauges/eden_water_levels.csv")
PLOT_GAUGE_TS = True          # plot INSAR vs gauge time series
LOS_TO_VERTICAL = True        # convert LOS displacement to approximate vertical (divide by cos(inc))

# Figure styling (simple defaults; adjust as you like)
CMAP_MAP = "viridis"          # for maps
CMAP_COHS = "plasma"          # for coherence
CMAP_RMS = "magma"            # for residual RMS

# ─────────────────────────────────────────────────────────────────────────────
# 2) DISCOVER INPUTS & WRITE MINTPY TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def discover_pairs(base: Path) -> List[Path]:
    """Return a sorted list of pair directories under BASE_DIR matching path*_YYYYMMDD_YYYYMMDD_*"""
    pair_dirs: List[Path] = []
    pat = re.compile(r"^path\d+_\d{8}_\d{8}_")
    for p in sorted(base.iterdir()):
        if p.is_dir() and pat.match(p.name):
            # ensure it contains an unwrapped file
            if (p / UNW_GLOB).exists():
                pair_dirs.append(p)
    return pair_dirs


def pick_geometry_dir(pair_dirs: List[Path]) -> Optional[Path]:
    """Pick the first pair that has a geometry/ directory."""
    for p in pair_dirs:
        g = p / GEOM_DIR_NAME
        if g.exists():
            return g
    return None


def write_smallbaseline_cfg(cfg_path: Path, geometry_dir: Optional[Path]) -> None:
    """Create a simple MintPy template (smallbaselineApp.cfg)."""
    cfg_lines = []

    # Load block
    cfg_lines += [
        "mintpy.load.processor = isce",
        "mintpy.load.autoPath = y",
        f"mintpy.load.unwFile = {BASE_DIR}/path*_*_*/{UNW_GLOB}",
        f"mintpy.load.corFile = {BASE_DIR}/path*_*_*/{COR_GLOB_PHSIG}",
        f"# fallback coherence (if phsig is missing in some pairs):",
        f"# mintpy.load.corFile = {BASE_DIR}/path*_*_*/{COR_GLOB_TOPOCOR}",
    ]
    if geometry_dir is not None:
        cfg_lines.append(f"mintpy.load.geometryDir = {geometry_dir}")
    else:
        cfg_lines.append("# mintpy.load.geometryDir = ")

    # Subset (optional; keep empty to use all)
    cfg_lines += [
        "# mintpy.subset.lalo = ",
        "# mintpy.subset.poly = ",
    ]

    # Reference
    if REF_LAT is not None and REF_LON is not None:
        cfg_lines.append(f"mintpy.reference.lalo = {REF_LAT:.6f}, {REF_LON:.6f}")
    else:
        cfg_lines.append("# mintpy.reference.lalo = 25.600000, -80.600000  # or leave blank to auto-pick")
    cfg_lines.append("mintpy.reference.maskDataset = coherence")

    # Inversion
    cfg_lines += [
        "mintpy.network.inversionMethod = SVD",
        "mintpy.network.keepMinSpanTree = y",
    ]

    # QC & Corrections
    cfg_lines.append(f"mintpy.unwrapError.flag = {'y' if ENABLE_UNWRAP_ERROR_FIX else 'n'}")
    cfg_lines.append(f"mintpy.deramp = {'y' if ENABLE_DERAMP else 'n'}")
    if TROPO_METHOD.lower() == "pyaps":
        cfg_lines += [
            "mintpy.troposphericDelay.method = pyaps",
            f"mintpy.troposphericDelay.weatherModel = {TROPO_MODEL}",
            "mintpy.troposphericDelay.weatherDir = auto",
        ]
    elif TROPO_METHOD.lower() == "gacos":
        cfg_lines += [
            "mintpy.troposphericDelay.method = gacos",
            "mintpy.troposphericDelay.weatherDir = /home/bakke326l/InSAR/main/data/aux/tropo",
        ]
    else:
        cfg_lines.append("mintpy.troposphericDelay.method = no")

    # Outputs & performance
    cfg_lines += [
        "mintpy.save.hdfEos5 = y",
        "mintpy.save.kmz = n",
        "mintpy.save.maskByCoherence = 0.2",
        f"mintpy.compute.parallel = {DASK_WORKERS if DASK_WORKERS and DASK_WORKERS > 1 else 0}",
    ]

    cfg_path.write_text("\n".join(cfg_lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3) RUN MINTPY PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_smallbaseline(cfg_path: Path, start: str = "", end: str = "") -> None:
    """Call MintPy smallbaselineApp.py. If start/end empty, run full pipeline.
    We run with cwd set to the PROJECT_DIR so MintPy uses our template in that directory.
    """
    cmd = ["smallbaselineApp.py", str(cfg_path.name)]
    if start:
        cmd += ["--start", start]
    if end:
        cmd += ["--end", end]
    workdir = cfg_path.parent
    print("📁 Working dir:", workdir)
    print("🧩 Using template:", cfg_path)
    print("🚀 Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=workdir)


# ─────────────────────────────────────────────────────────────────────────────
# 4) POST-PROCESS: LOAD H5 AND MAKE FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def read_h5_dates(h5_path: Path) -> List[str]:
    """Return list of date strings in timeseries.h5 (YYYY-MM-DD)."""
    with h5py.File(h5_path, "r") as f:
        if "date" in f:
            # standard MintPy
            dset = f["date"][:]
            dates = [d.decode("utf-8") if isinstance(d, (bytes, np.bytes_)) else str(d) for d in dset]
        else:
            dates = [d.decode("utf-8") for d in f["timeseries"].attrs["DATE_LIST"]]
    return dates


def load_map(h5_path: Path, dset_name: str) -> np.ndarray:
    """Read a 2D map from an H5 file (e.g., velocity.h5 -> velocity)."""
    with h5py.File(h5_path, "r") as f:
        # Try common nesting first, then flat
        if dset_name in f:
            arr = f[dset_name][:]
        elif f"HDFEOS/GRIDS/{dset_name}/observation" in f:
            arr = f[f"HDFEOS/GRIDS/{dset_name}/observation/data"][:]
        else:
            # fall back by scanning keys
            key = next((k for k in f.keys() if dset_name in k.lower()), None)
            arr = f[key][:] if key else None
    return np.array(arr)


def percentile_limits(arr: np.ndarray, pmin=2, pmax=98) -> Tuple[float, float]:
    m = np.isfinite(arr)
    if not np.any(m):
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(arr[m], [pmin, pmax])
    if vmin == vmax:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def plot_map(arr: np.ndarray, title: str, out_png: Path, cmap: str = CMAP_MAP, units: str = "") -> None:
    vmin, vmax = percentile_limits(arr)
    plt.figure(figsize=(9, 7))
    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label(units)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_hist(arr: np.ndarray, title: str, out_png: Path, bins: int = 60) -> None:
    m = np.isfinite(arr)
    data = arr[m]
    plt.figure(figsize=(8, 5))
    plt.hist(data.ravel(), bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def load_geo(geom_h5: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return latitude, longitude, and incidence angle arrays from geometryGeo.h5."""
    with h5py.File(geom_h5, "r") as g:
        lat = g["latitude"][:]
        lon = g["longitude"][:]
        inc = g["incidenceAngle"][:] if "incidenceAngle" in g else None
    return lat, lon, inc


def nearest_pixel(lat_map: np.ndarray, lon_map: np.ndarray, lat: float, lon: float) -> Tuple[int, int]:
    dlat = (lat_map - lat) ** 2
    dlon = (lon_map - lon) ** 2
    iy, ix = np.unravel_index(np.argmin(dlat + dlon), lat_map.shape)
    return int(iy), int(ix)


def find_timeseries_file(project_dir: Path) -> Path:
    """Find the MintPy time-series HDF5 produced by the run.
    Different MintPy versions/methods may name it timeseries.h5, timeseries_GACOS.h5,
    timeseries_ERA5.h5, etc. This helper returns the first one found.
    """
    candidates = [
        "timeseries.h5",
        "timeseries_GACOS.h5",
        "timeseries_ERA5.h5",
        "timeseries_ECMWF.h5",
        "timeseries_NO.h5",
    ]
    for name in candidates:
        p = project_dir / name
        if p.exists():
            return p
    # Fallback: any timeseries*.h5
    for p in project_dir.glob("timeseries*.h5"):
        return p
    # Default guess
    return project_dir / "timeseries.h5"


def plot_timeseries_vs_gauge(timeseries_h5: Path,
                             geom_h5: Path,
                             gauge_csv: Path,
                             out_dir: Path,
                             los_to_vertical: bool = True) -> None:
    """Extract InSAR time series at each gauge and plot against the gauge record.

    Gauge CSV format (wide): StationID,Lat,Lon,2007-01-01,2007-01-02,... (daily columns)
    We'll sample the InSAR time series at the nearest pixel and align dates.
    """
    if not gauge_csv.exists():
        print(f"⚠️  Gauge CSV not found: {gauge_csv}")
        return

    # Load MintPy time series and dates
    with h5py.File(timeseries_h5, "r") as f:
        ts = f["timeseries"][:]   # shape: (n_dates, ny, nx), meters (LOS)
        dates_insar = read_h5_dates(timeseries_h5)
    # Load geometry
    lat_map, lon_map, inc_map = load_geo(geom_h5)
    cos_inc = None
    if los_to_vertical and inc_map is not None:
        cos_inc = np.cos(np.deg2rad(inc_map))

    # Read gauges
    with open(gauge_csv, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        stations = list(reader)

    # Loop gauges
    for row in stations:
        name = row.get("StationID", "Gauge")
        try:
            glat = float(row["Lat"]) ; glon = float(row["Lon"]) 
        except Exception:
            print(f"⚠️  Skipping row without Lat/Lon: {name}")
            continue
        iy, ix = nearest_pixel(lat_map, lon_map, glat, glon)

        # Use the 9 closest pixels as a 3×3 neighborhood around the nearest pixel
        # (simple, robust, and fast). We average their time series (NaN-safe).
        ys = np.clip(np.arange(iy - 1, iy + 2), 0, ts.shape[1] - 1)
        xs = np.clip(np.arange(ix - 1, ix + 2), 0, ts.shape[2] - 1)
        YY, XX = np.meshgrid(ys, xs, indexing="ij")
        YY = YY.ravel(); XX = XX.ravel()  # 9 indices

        # Stack LOS time series for the 9 pixels → shape (n_dates, 9)
        pix_ts_los_stack = np.stack([ts[:, y, x] for y, x in zip(YY, XX)], axis=1)

        # Convert each pixel to vertical (approx.) before averaging, if requested
        if los_to_vertical and cos_inc is not None:
            ci_stack = np.array([cos_inc[y, x] for y, x in zip(YY, XX)], dtype=float)
            # Guard against invalid/small cos(inc)
            ci_stack[~np.isfinite(ci_stack)] = np.nan
            ci_stack[ci_stack < 0.1] = np.nan
            # Broadcast to (n_dates, 9)
            ci2d = np.tile(ci_stack, (pix_ts_los_stack.shape[0], 1))
            pix_ts_stack = pix_ts_los_stack / ci2d
            y_label_insar = "InSAR displacement (approx. vertical, m, 3×3 mean)"
        else:
            pix_ts_stack = pix_ts_los_stack
            y_label_insar = "InSAR displacement (LOS, m, 3×3 mean)"

        # Average across the 9 pixels, ignoring NaNs
        pix_ts = np.nanmean(pix_ts_stack, axis=1)

        # Parse gauge time series from the wide row
        dates_gauge: List[str] = []
        vals_gauge: List[float] = []
        for k, v in row.items():
            if k in ("StationID", "Lat", "Lon"):
                continue
            if k and re.match(r"^\d{4}-\d{2}-\d{2}$", k):
                try:
                    vv = float(v) if v not in ("", "NA", "NaN", None) else np.nan
                except Exception:
                    vv = np.nan
                dates_gauge.append(k)
                vals_gauge.append(vv)

        # Align dates: pick values at MintPy dates when available
        # (For days without gauge data, we skip; simple and robust)
        gauge_on_insar = []
        for d in dates_insar:
            try:
                idx = dates_gauge.index(d)
                gauge_on_insar.append(vals_gauge[idx])
            except ValueError:
                gauge_on_insar.append(np.nan)
        gauge_on_insar = np.array(gauge_on_insar, dtype=float)

        # Align offsets for visual comparison (subtract median of overlapping finite samples)
        both_ok = np.isfinite(pix_ts) & np.isfinite(gauge_on_insar)
        offset = np.median(pix_ts[both_ok] - gauge_on_insar[both_ok]) if np.any(both_ok) else 0.0
        gauge_aligned = gauge_on_insar + offset

        # Plot
        plt.figure(figsize=(11, 6))
        plt.plot(dates_insar, pix_ts, label=y_label_insar)
        plt.plot(dates_insar, gauge_aligned, label=f"Gauge {name} (aligned)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f"Time series (3×3 mean) near gauge {name}\n(lat={glat:.5f}, lon={glon:.5f})")
        plt.tight_layout()
        out_png = out_dir / f"ts_vs_gauge_{name}.png"
        plt.savefig(out_png, dpi=200)
        plt.close()

        # Save CSV of the extracted InSAR time series (and the aligned gauge for convenience)
        out_csv = out_dir / f"ts_vs_gauge_{name}.csv"
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "insar_m", "gauge_aligned_m", "gauge_raw_m"])
            for d, ins, galn, gr in zip(dates_insar, pix_ts, gauge_aligned, gauge_on_insar):
                w.writerow([d, f"{ins:.6f}", f"{galn if np.isfinite(galn) else np.nan}", f"{gr if np.isfinite(gr) else np.nan}"])
        print(f"✅ Saved: {out_png} and {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# 5) MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _count_matches(pattern: str) -> int:
    """Return how many files match a glob pattern (debug helper)."""
    from glob import glob
    return len(glob(pattern))

if __name__ == "__main__":
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Debug: check that our globs actually see files
    unw_glob_abs = f"{BASE_DIR}/path*_*_*/{UNW_GLOB}"
    cor_glob_abs = f"{BASE_DIR}/path*_*_*/{COR_GLOB_PHSIG}"
    print("🔎 Preflight:")
    print("   UNW glob:", unw_glob_abs, "→", _count_matches(unw_glob_abs), "files")
    print("   COR glob:", cor_glob_abs, "→", _count_matches(cor_glob_abs), "files")

    # Discover pair directories
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Discover pair directories
    pair_dirs = discover_pairs(BASE_DIR)
    if not pair_dirs:
        raise SystemExit(f"No pair folders found under {BASE_DIR} with {UNW_GLOB}")
    print(f"Found {len(pair_dirs)} pair directories")

    # Pick geometry dir
    geom_dir = pick_geometry_dir(pair_dirs)
    if geom_dir is None:
        print("⚠️  Could not locate a geometry/ folder — MintPy may still proceed if geometry is embedded.")

    # Write MintPy template
    cfg_path = PROJECT_DIR / "smallbaselineApp.cfg"
    write_smallbaseline_cfg(cfg_path, geom_dir)
    print(f"📝 Wrote template: {cfg_path}")
    if geom_dir:
        print("   geometryDir:", geom_dir)
    else:
        print("   geometryDir: (none set — will try auto)")

    # Run MintPy (full pipeline)
    try:
        run_smallbaseline(cfg_path)
    except subprocess.CalledProcessError as e:
        print("❌ MintPy run failed:", e)
        print("   Tip: you can restart with partial steps, e.g. --start load_data --end time_series")
        raise

    # Paths to standard outputs
    ts_h5   = find_timeseries_file(PROJECT_DIR)
    vel_h5  = PROJECT_DIR / "velocity.h5"
    coh_h5  = PROJECT_DIR / "temporalCoherence.h5"
    rms_h5  = PROJECT_DIR / "rmse.h5"  # may be named residualRMS.h5 in some versions
    geom_h5 = PROJECT_DIR / "inputs" / "geometryGeo.h5"

    # Make visuals — velocity
    if vel_h5.exists():
        vel = load_map(vel_h5, "velocity")  # m/yr
        plot_map(vel, "Velocity (m/yr)", FIG_DIR / "velocity.png", cmap=CMAP_MAP, units="m/yr")
        print("✅ Wrote:", FIG_DIR / "velocity.png")

    # Temporal coherence map + histogram
    if coh_h5.exists():
        coh = load_map(coh_h5, "temporalCoherence")
        plot_map(coh, "Temporal Coherence", FIG_DIR / "temporal_coherence.png", cmap=CMAP_COHS, units="")
        plot_hist(coh, "Temporal Coherence Histogram", FIG_DIR / "temporal_coherence_hist.png")
        print("✅ Wrote:", FIG_DIR / "temporal_coherence.png")

    # Residual RMS (if present)
    if rms_h5.exists():
        try:
            rms = load_map(rms_h5, "rmse")
        except Exception:
            # try alternative dataset name
            rms = load_map(rms_h5, "residualRMS")
        if rms is not None:
            plot_map(rms, "Residual RMS (m)", FIG_DIR / "residual_rms.png", cmap=CMAP_RMS, units="m")
            print("✅ Wrote:", FIG_DIR / "residual_rms.png")

    # Displacement maps at first/last dates
    if ts_h5.exists():
        dates = read_h5_dates(ts_h5)
        with h5py.File(ts_h5, "r") as f:
            ts = f["timeseries"][:]  # (n, y, x) in meters
        if len(dates) >= 1:
            first = ts[0]
            plot_map(first, f"Displacement @ {dates[0]} (m)", FIG_DIR / f"disp_{dates[0]}.png", cmap=CMAP_MAP, units="m")
        if len(dates) >= 2:
            last = ts[-1]
            plot_map(last, f"Displacement @ {dates[-1]} (m)", FIG_DIR / f"disp_{dates[-1]}.png", cmap=CMAP_MAP, units="m")
            # Also cumulative change (last - first)
            cum = last - ts[0]
            plot_map(cum, f"Cumulative Change {dates[0]} → {dates[-1]} (m)", FIG_DIR / "disp_cumulative.png", cmap=CMAP_MAP, units="m")
        print("✅ Wrote displacement maps.")

    # Plot time series vs gauges (optional)
    if PLOT_GAUGE_TS and ts_h5.exists() and geom_h5.exists():
        try:
            plot_timeseries_vs_gauge(ts_h5, geom_h5, GAUGE_CSV, FIG_DIR, los_to_vertical=LOS_TO_VERTICAL)
        except Exception as e:
            print("⚠️  Gauge plotting failed:", e)

    print("\n🎉 Done. Figures are in:", FIG_DIR)
    print("   Key products:")
    print("   • velocity.png  • temporal_coherence.png  • residual_rms.png (if available)")
    print("   • disp_<date>.png  • disp_cumulative.png  • ts_vs_gauge_*.png / .csv")