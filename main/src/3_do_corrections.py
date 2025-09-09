#!/usr/bin/env python3
"""
3_do_corrections.py — Apply iono/tropo corrections and export vertical displacement maps

Purpose
-------
Post-process finished interferograms per pair directory by:
1) Generating a **RAW** vertical-displacement map from the base unwrapped phase.
2) Creating **IONO-only** corrected unwrapped, quicklooks, and vertical map.
3) Creating **TROPO-only** corrected unwrapped, quicklooks, and vertical map.
4) Creating **TROPO+IONO** corrected unwrapped, quicklooks, and vertical map.
5) NEW: Export quicklook PNGs for:
   • Tropospheric **differential delay** (ΔAPS = ref - sec).
   • Ionospheric **dispersive phase** used for correction.
All vertical products are masked to the SAR swath and written as GeoTIFFs
(with PNG quicklooks saved both locally and to a global inspect folder).

Needed data (inputs & assumptions)
----------------------------------
- Pair directories laid out as:
    /mnt/DATA2/bakke326l/processing/interferograms/
      path<PATH>_<REF>_<SEC>_<DEM>/
        interferogram/
          filt_topophase.unw.geo or filt_topophase.unw.geo.vrt   # base unwrapped
        geometry/
          los.rdr.geo or los.rdr.geo.vrt                         # line-of-sight (optional but needed for vertical)
- Atmospheric aux data directory for tropospheric correction:
    ~/InSAR/main/data/aux/tropo/  (GACOS tiles etc.)
- Helper module providing correction routines:
    help_atm_correction.do_iono_correction(...)
    help_atm_correction.do_tropo_correction(...)
- Assumes ALOS L-band wavelength λ = 0.2362 m for phase→LOS conversion.
- Expects pair directory names like: path150_YYYYMMDD_YYYYMMDD_{SRTM|3DEP}

Dependencies
------------
- Python: numpy, rasterio, matplotlib
- Raster reprojection: rasterio.warp (reproject, Resampling)
- System: gdal_translate on PATH
- Local: help_atm_correction (iono/tropo), and standard scientific stack

Outputs & directories
---------------------
Inside each pair directory:
  interferogram/
    filt_topophase_iono.unw.geo           # IONO-only
    filt_topophase_tropo.unw.geo          # TROPO-only
    filt_topophase_tropo_iono.unw.geo     # TROPO+IONO
  inspect/
    <above>.tif (translated quick GeoTIFF copies, where applicable)
    <above>.png quicklooks
    vertical_displacement_cm_<REF>_<SEC>_{RAW|IONO|TROPO|TROPO_IONO}.geo.tif
    vertical_displacement_cm_<REF>_<SEC>_{...}.png
    tropo_differential_delay_<REF>_<SEC>.png
    iono_dispersive_phase_<REF>_<SEC>.png
Global PNG copies are also written to:
  ~/InSAR/main/processing/inspect/

How it works
------------
- Reads unwrapped phase and applies a **swath mask**:
  uses GDAL band mask (preferred) or amplitude!=0 if available; off-swath→NaN.
- Converts phase→LOS displacement: d_LOS = (λ / 4π) * phase (meters).
- Removes **LOS median** prior to vertical conversion to avoid 1/cos ramp.
- Converts to **vertical**: d_vert = - d_LOS_zeroed / cos(incidence) in **cm**.
- Regrids incidence to IFG grid if shape/CRS/transform differ.
- Runs IONO/TROPO corrections via helpers; builds quicklooks throughout.

How to run
----------
# One specific pair directory
python 3_do_corrections.py /mnt/DATA2/bakke326l/processing/interferograms/path150_20071216_20080131_SRTM

# Batch under a root (will scan for subdirs starting with 'path')
python 3_do_corrections.py --batch /mnt/DATA2/bakke326l/processing/interferograms

Notes
-----
- If LOS is missing, vertical exports are skipped (corrections still produced).
- Quicklooks use a robust 1-99% stretch and downsampled reads where possible.
- All outputs use NaN off-swath; compression is DEFLATE for GeoTIFFs.
"""


from __future__ import annotations
import argparse
import sys
import subprocess
from pathlib import Path

# --- silence DEBUG spam ---
import os, logging
import shutil  # <-- added
os.environ.setdefault("CPL_DEBUG", "OFF")
os.environ.setdefault("CPL_LOG", "/dev/null")
os.environ.setdefault("RIO_LOG_LEVEL", "CRITICAL")
logging.basicConfig(level=logging.INFO, force=True)
for name in ("rasterio", "matplotlib", "fiona", "shapely"):
    logging.getLogger(name).setLevel(logging.WARNING)

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib import colormaps
from rasterio.warp import reproject, Resampling

# Helpers 
from help_atm_correction import do_iono_correction, do_tropo_correction

# ---------------------------------------------------------------------- paths
BASE       = Path(__file__).resolve().parents[1]           # ~/InSAR/main
TROPO_DIR  = BASE / "data" / "aux" / "tropo"               # GACOS, etc.
HOME_PROC  = BASE / "processing"                           # for global PNG copies

# ------------------------------------------------------------ small utilities
def quicklook_png(src: Path, dst: Path, band: int = 1) -> None:
    """
    Write a compact PNG quicklook of a raster band with robust contrast.

    Behavior
    --------
    - Reads the requested band as a masked array.
    - If fully masked or empty, saves a black image of the same shape.
    - Uses 1-99% percentiles for contrast stretching.
    - Applies matplotlib's 'turbo' colormap and drops alpha channel.

    Parameters
    ----------
    src : Path
        Input raster (GeoTIFF/VRT) readable by rasterio.
    dst : Path
        Output PNG path.
    band : int, default=1
        1-based band index to visualize.

    Outputs
    -------
    dst : PNG image saved to disk.
    """
    turbo = colormaps.get_cmap("turbo")
    with rasterio.open(src) as ds:
        arr = ds.read(band, masked=True)  # uses GDAL mask/nodata if present
    if hasattr(arr, "mask") and np.all(getattr(arr, "mask")):
        plt.imsave(dst, np.zeros((*arr.shape, 3), dtype=np.uint8))
        return
    a = arr.compressed()
    if a.size == 0:
        plt.imsave(dst, np.zeros((*arr.shape, 3), dtype=np.uint8))
        return
    vmin, vmax = np.percentile(a, [1, 99])
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = (turbo(norm.filled(0)) * 255).astype(np.uint8)
    plt.imsave(dst, rgba[..., :3])  # drop alpha

def translate_to_tif(src: Path, dst: Path) -> None:
    """
    Create a GeoTIFF copy of a source raster using gdal_translate.

    Parameters
    ----------
    src : Path
        Source raster (e.g., *.unw.geo, *.vrt).
    dst : Path
        Destination GeoTIFF.

    Raises
    ------
    subprocess.CalledProcessError
        If gdal_translate returns a non-zero exit code.
    """
    subprocess.check_call(["gdal_translate", "-of", "GTiff", str(src), str(dst)])

def parse_pair_id(pairdir: Path) -> tuple[int, str, str, str]:
    """
    Parse path, reference, secondary, and DEM label from a pair directory name.

    Expected pattern
    ----------------
    path<PATH>_<YYYYMMDD>_<YYYYMMDD>_<DEM>

    Returns
    -------
    (int path, str ref, str sec, str dem)

    Raises
    ------
    ValueError
        If the directory name does not match the expected pattern.
    """
    name = pairdir.name
    toks = name.split("_")
    if len(toks) < 4 or not toks[0].startswith("path"):
        raise ValueError(f"Cannot parse pair id from folder name: {name}")
    path = int(toks[0][4:])
    ref  = toks[1]
    sec  = toks[2]
    dem  = toks[3]  # SRTM or 3DEP
    return path, ref, sec, dem

def find_base_unw(igram_dir: Path) -> Path:
    """
    Locate the base unwrapped interferogram to serve as the RAW input.

    Search order
    ------------
    1) interferogram/filt_topophase.unw.geo
    2) interferogram/filt_topophase.unw.geo.vrt

    Returns
    -------
    Path
        Path to the chosen base unwrapped product.

    Raises
    ------
    FileNotFoundError
        If neither candidate exists.
    """
    cand = igram_dir / "filt_topophase.unw.geo"
    if cand.exists():
        return cand
    cand = igram_dir / "filt_topophase.unw.geo.vrt"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Base unwrapped not found in {igram_dir}")

def _pstats(arr: np.ndarray, name: str) -> None:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        print(f"   • {name}: all NaN")
        return
    p2, p50, p98 = np.percentile(a, [2, 50, 98])
    print(f"   • {name}: P2={p2:.6f}, P50={p50:.6f}, P98={p98:.6f}")

def _regrid_to_match(src_arr, src_transform, src_crs, dst_shape, dst_transform, dst_crs):
    """Bilinear reproject of src_arr to match dst grid."""
    dst = np.empty(dst_shape, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        num_threads=2
    )
    return dst

# ---------- NEW: build a "swath mask" from the unwrapped dataset ----------
def _swath_mask(unw_path: Path, count: int, phase_band: int) -> np.ndarray:
    """
    Build a boolean in-swath mask for the unwrapped dataset.

    Logic
    -----
    1) Try GDAL's per-band mask for the phase band (preferred).
    2) If unavailable and dataset has ≥2 bands, use amplitude (band 1) != 0.
    3) Fallback to all-True if neither yields a usable mask.

    Parameters
    ----------
    unw_path : Path
        Path to the unwrapped product.
    count : int
        Number of bands in the dataset.
    phase_band : int
        1-based index of the phase band.

    Returns
    -------
    np.ndarray of bool
        True for valid in-swath pixels; False elsewhere.
    """
    with rasterio.open(unw_path) as ds:
        # (1) phase band mask if available
        try:
            m = ds.read_masks(phase_band)
        except Exception:
            m = None
        if m is not None and not np.all(m == 255):
            return (m > 0)

        # (2) amplitude fallback for classic ISCE unw (band 1 = amp)
        if count >= 2:
            amp = ds.read(1)
            valid = np.isfinite(amp) & (amp != 0)
            if valid.any():
                return valid

        # Fallback: all valid (will not change values)
        h, w = ds.height, ds.width
        return np.ones((h, w), dtype=bool)

def _read_phase(unw_path: Path) -> tuple[np.ndarray, dict, any, any, tuple[int,int]]:
    """
    Read the phase band from an unwrapped product and mask off-swath pixels.

    Rules
    -----
    - If dataset has ≥2 bands (amp+phase), take band 2 as phase.
    - Otherwise take band 1.
    - Apply swath mask so off-swath is set to NaN.

    Returns
    -------
    phase : np.ndarray (float32)
        Phase array with NaN off-swath.
    profile : dict
        Rasterio profile of the input dataset.
    crs : any
        Coordinate reference system.
    transform : affine.Affine
        Geo-transform of the input dataset.
    shape : (int, int)
        (height, width) of the dataset.
    """
    with rasterio.open(unw_path) as ds:
        phase_band = 2 if ds.count >= 2 else 1
        phase = ds.read(phase_band).astype(np.float32)
        prof  = ds.profile
        crs   = ds.crs
        tr    = ds.transform
        shape = (ds.height, ds.width)

    # Apply swath mask: set outside-swath to NaN
    valid = _swath_mask(unw_path, count=prof.get("count", 1), phase_band=phase_band)
    if valid.shape == phase.shape:
        off = (~valid).sum()
        if off > 0:
            phase = phase.copy()
            phase[~valid] = np.float32(np.nan)
            print(f"   • Swath mask applied: set {int(off):,} pixel(s) to NaN (off-swath).")
    else:
        print("   • Swath mask shape mismatch — skipping mask application.")

    return phase, prof, crs, tr, shape

def write_vertical_cm(unw_path: Path, los_path: Path, out_path: Path, label: str = "") -> None:
    """
    Convert unwrapped phase (radians) to vertical displacement (centimeters).

    Computation
    -----------
    - d_LOS = (λ / 4π) * phase, with λ = 0.2362 m (ALOS L-band).
    - Subtract the median of d_LOS over valid pixels (removes constant offset and
    mitigates a 1/cos(inc) ramp).
    - vertical = - d_LOS_zeroed / cos(incidence).

    Grid handling
    -------------
    - Reads incidence (degrees) from `los_path`.
    - If CRS/transform/shape differ from the unwrapped grid, reprojects incidence
    to match (bilinear).
    - Off-swath pixels (from swath mask) are kept as NaN.
    """
    WAVELENGTH = 0.2362  # m

    # --- Read corrected phase (radians), already masked off-swath ---
    phase, prof, u_crs, u_tr, u_shape = _read_phase(unw_path)

    # --- Read incidence (degrees), regrid if needed ---
    with rasterio.open(los_path) as ds_inc:
        inc_deg = ds_inc.read(1).astype(np.float32)
        i_crs   = ds_inc.crs
        i_tr    = ds_inc.transform
        i_shape = (ds_inc.height, ds_inc.width)

    if (i_shape != u_shape) or (i_crs != u_crs) or (i_tr != u_tr):
        print("   • Regridding incidence to match IFG grid …")
        inc_deg = _regrid_to_match(
            inc_deg, i_tr, i_crs,
            u_shape, u_tr, u_crs
        )

    # --- Diagnostics before conversion ---
    crs_txt = u_crs.to_string() if u_crs else "None"
    print(f"   • Grid: {u_shape[1]}x{u_shape[0]}, CRS={crs_txt}  [{label}]")
    _pstats(phase,   "phase [rad]")
    _pstats(inc_deg, "incidence [deg]")

    # --- Convert to LOS and remove median (reference) ---
    inc  = np.deg2rad(inc_deg)
    cosi = np.cos(inc).astype(np.float32)

    d_los_m = (WAVELENGTH / (4.0 * np.pi)) * phase  # meters along LOS
    valid = np.isfinite(d_los_m) & np.isfinite(cosi) & (np.abs(cosi) > 1e-6)
    los_med = np.nanmedian(d_los_m[valid]) if np.any(valid) else 0.0
    d_los_m_zeroed = d_los_m - los_med  # <-- removes artificial 1/cos ramp
    print(f"   • Removed LOS median offset: {los_med:.6f} m ({los_med*100:.3f} cm)  [{label}]")

    _pstats(d_los_m_zeroed, "LOS (zeroed) [m]")
    _pstats(cosi,           "cos(inc) [-]")

    # --- Vertical (cm) ---
    d_vert_cm = np.full_like(d_los_m_zeroed, np.nan, dtype=np.float32)
    d_vert_cm[valid] = - (d_los_m_zeroed[valid] / cosi[valid]) * 100.0

    _pstats(d_vert_cm, "vertical [cm]")

    # --- Write GeoTIFF ---
    prof = prof.copy()
    prof.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "nodata": np.float32(np.nan),
        "compress": "deflate",
        "predictor": 3,
        "zlevel": 6,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(d_vert_cm, 1)

    # quicklook (per-pair inspect only)
    try:
        png = out_path.with_suffix(".png")
        quicklook_png(out_path, png, band=1)
        # removed global HOME_PROC/inspect copy
    except Exception as e:
        print(f"⚠️  vertical PNG failed: {e}")

# --------------------------- readiness check (unchanged) ---------------------------
def readiness(pairdir: Path) -> tuple[bool, str | None, Path | None]:
    """
    Check whether a pair directory has the required inputs to proceed.

    Validates
    ---------
    - Presence of base unwrapped product:
    interferogram/filt_topophase.unw.geo(.vrt)
    - Line-of-sight file los.rdr.geo(.vrt) is optional; if missing, vertical
    exports will be skipped but corrections may still run.
    """
    igram_dir = pairdir / "interferogram"
    geom_dir  = pairdir / "geometry"

    # Base unwrapped must exist to proceed at all
    try:
        _ = find_base_unw(igram_dir)
    except FileNotFoundError:
        return (False, "missing base unwrapped interferogram (filt_topophase.unw.geo[.vrt])", None)

    # LOS is optional; if missing we'll skip vertical exports
    los_vrt = geom_dir / "los.rdr.geo.vrt"
    los_geo = geom_dir / "los.rdr.geo"
    los = los_vrt if los_vrt.exists() else (los_geo if los_geo.exists() else None)

    return (True, None, los)

# ------------------------------------------------------------ main per-pair
def process_pair(pairdir: Path) -> bool:
    """
    Run ionospheric and tropospheric corrections and export vertical maps for one pair.

    Pipeline
    --------
    1) **RAW vertical** from base unwrapped (if LOS present).
    2) **IONO-only** correction → quicklooks → vertical (if LOS present).
    3) **TROPO-only** correction → quicklooks → vertical (if LOS present).
    4) **TROPO+IONO** (iono applied to tropo product) → quicklooks → vertical.
    5) Quicklooks for ΔAPS (tropo) and dispersive phase (iono).
    All vertical GeoTIFFs are masked to the swath and compressed; PNG quicklooks
    are written locally and to the global inspect dir.
    """
    try:
        path_no, ref, sec, dem = parse_pair_id(pairdir)
    except Exception as e:
        print(f"⚠️  Skipping {pairdir}: {e}")
        return False

    ok, reason, los_path = readiness(pairdir)
    if not ok:
        print(f"⏭️  Skipping {pairdir.name}: {reason}")
        return False

    igram_dir   = pairdir / "interferogram"
    tropo_dir   = pairdir / "troposphere"
    inspect_dir = pairdir / "inspect"
    geom_dir    = pairdir / "geometry"
    tropo_dir.mkdir(exist_ok=True)
    inspect_dir.mkdir(exist_ok=True)

    # NEW: central quicklook dir under interferograms/
    quickroot = pairdir.parent / "quicklook"
    quickroot.mkdir(exist_ok=True)

    # Base unwrapped (guaranteed by readiness)
    base_unw = find_base_unw(igram_dir)

    # ---------------- RAW vertical ----------------
    if los_path is not None:
        print(f"🧭  RAW vertical (no atmos correction): {pairdir.name}")
        try:
            out_raw = inspect_dir / f"vertical_displacement_cm_{ref}_{sec}_RAW.geo.tif"
            write_vertical_cm(base_unw, los_path, out_raw, label="RAW")
            # copy RAW PNG to central quicklook
            raw_png = out_raw.with_suffix(".png")
            if raw_png.exists():
                shutil.copy2(raw_png, quickroot / f"{pairdir.name}__{raw_png.name}")
        except Exception as e:
            print(f"⚠️  RAW vertical export failed: {e}")
    else:
        print("ℹ️  LOS not found → skipping RAW vertical export.")

    # ---------------- IONO-only ----------------
    print(f"🌌  Ionospheric correction (IONO-only): {pairdir.name}")
    try:
        do_iono_correction(work_dir=pairdir, out_dir=igram_dir,
                           input_unw=base_unw, output_suffix="_iono")
    except Exception as e:
        print(f"⚠️  iono-only correction failed: {e}")

    # quicklook for the ionospheric **dispersive phase** used for correction
    try:
        iono_phase = pairdir / "ionosphere" / "dispersive.bil.unwCor.filt.geo"
        if iono_phase.exists():
            local_png  = inspect_dir / f"iono_dispersive_phase_{ref}_{sec}.png"
            quicklook_png(iono_phase, local_png)
            # removed global HOME_PROC/inspect copy
            print(f"✓ iono dispersive quicklook → {local_png.name}")
        else:
            print("ℹ️  ionosphere/dispersive.bil.unwCor.filt.geo not found — skipping iono quicklook.")
    except Exception as e:
        print(f"⚠️  iono dispersive quicklook failed: {e}")

    iono_unw = igram_dir / "filt_topophase_iono.unw.geo"
    if iono_unw.exists():
        # quicklooks for corrected unwrapped (per-pair inspect only)
        try:
            tif = inspect_dir / f"{iono_unw.stem}_{ref}_{sec}.tif"
            translate_to_tif(iono_unw, tif)
            quicklook_png(iono_unw, inspect_dir / f"{iono_unw.stem}_{ref}_{sec}.png")
        except Exception as e:
            print(f"⚠️  iono export failed: {e}")

        # vertical
        if los_path is not None:
            try:
                out_iono = inspect_dir / f"vertical_displacement_cm_{ref}_{sec}_IONO.geo.tif"
                write_vertical_cm(iono_unw, los_path, out_iono, label="IONO")
            except Exception as e:
                print(f"⚠️  IONO vertical export failed: {e}")
        else:
            print("ℹ️  LOS not found → skipping IONO vertical displacement export.")
    else:
        print("⚠️  iono-only output not found; check helper implementation.")

    # ---------------- TROPO ----------------
    print(f"😶‍🌫️  Tropospheric correction (GACOS): {pairdir.name}")
    try:
        do_tropo_correction(wdir=pairdir, ref=ref, sec=sec,
                            gacos_dir=TROPO_DIR, tropo_dir=tropo_dir)
    except Exception as e:
        print(f"⚠️  tropo correction failed for {pairdir.name}: {e}")

    # quicklook for **ΔAPS (ref−sec)** difference map (per-pair inspect only)
    try:
        aps_diff = tropo_dir / f"{ref}_{sec}.aps.geo"
        if aps_diff.exists():
            local_png  = inspect_dir / f"tropo_differential_delay_{ref}_{sec}.png"
            quicklook_png(aps_diff, local_png)
            # removed global HOME_PROC/inspect copy
            print(f"✓ tropo ΔAPS quicklook → {local_png.name}")
        else:
            print("ℹ️  ΔAPS file not found (tropo/<ref>_<sec>.aps.geo) — skipping TROPO diff quicklook.")
    except Exception as e:
        print(f"⚠️  TROPO ΔAPS quicklook failed: {e}")

    tropo_unw = igram_dir / "filt_topophase_tropo.unw.geo"
    if tropo_unw.exists():
        # quicklooks for tropo-only unwrapped (per-pair inspect only)
        try:
            tif_tropo = inspect_dir / f"{tropo_unw.stem}_{ref}_{sec}.tif"
            translate_to_tif(tropo_unw, tif_tropo)
            quicklook_png(tropo_unw, inspect_dir / f"{tropo_unw.stem}_{ref}_{sec}.png")
        except Exception as e:
            print(f"⚠️  tropo export failed: {e}")

        # vertical (TROPO)
        if los_path is not None:
            try:
                out_tropo = inspect_dir / f"vertical_displacement_cm_{ref}_{sec}_TROPO.geo.tif"
                write_vertical_cm(tropo_unw, los_path, out_tropo, label="TROPO")
            except Exception as e:
                print(f"⚠️  TROPO vertical export failed: {e}")
        else:
            print("ℹ️  LOS not found → skipping TROPO vertical displacement export.")

        # ------------- TROPO + IONO -------------
        print("🌌  Ionospheric correction on TROPO product (TROPO+IONO)")
        try:
            do_iono_correction(work_dir=pairdir, out_dir=igram_dir,
                               input_unw=tropo_unw, output_suffix="_tropo_iono")
        except Exception as e:
            print(f"⚠️  iono-on-tropo failed: {e}")

        tropo_iono_unw = igram_dir / "filt_topophase_tropo_iono.unw.geo"
        if tropo_iono_unw.exists():
            # quicklooks (per-pair inspect only)
            try:
                tif_ti = inspect_dir / f"{tropo_iono_unw.stem}_{ref}_{sec}.tif"
                translate_to_tif(tropo_iono_unw, tif_ti)
                quicklook_png(tropo_iono_unw, inspect_dir / f"{tropo_iono_unw.stem}_{ref}_{sec}.png")
            except Exception as e:
                print(f"⚠️  tropo+iono export failed: {e}")

            # vertical (TROPO_IONO)
            if los_path is not None:
                try:
                    out_ti = inspect_dir / f"vertical_displacement_cm_{ref}_{sec}_TROPO_IONO.geo.tif"
                    write_vertical_cm(tropo_iono_unw, los_path, out_ti, label="TROPO_IONO")
                    # copy TROPO_IONO PNG to central quicklook
                    ti_png = out_ti.with_suffix(".png")
                    if ti_png.exists():
                        shutil.copy2(ti_png, quickroot / f"{pairdir.name}__{ti_png.name}")
                except Exception as e:
                    print(f"⚠️  TROPO+IONO vertical export failed: {e}")
            else:
                print("ℹ️  LOS not found → skipping TROPO+IONO vertical displacement export.")
        else:
            print("⚠️  tropo+iono output not found; check helper implementation.")
    else:
        print("⚠️  tropo output not found; skipping TROPO and TROPO+IONO branch.")

    print(f"✅  Corrections complete for {pairdir}\n")
    return True

# -------------------------------------------------------------- batch helpers
def collect_pairs(root: Path) -> list[Path]:
    """
    List immediate subdirectories under a root that look like pair directories.

    Parameters
    ----------
    root : Path
        Root folder expected to contain many path* subdirectories.

    Returns
    -------
    list[Path]
        Sorted list of subdirectories whose names start with 'path'.
    """
    return sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("path")])

# ---------------------------------------------------------------------- CLI
def main() -> None:
    """
    CLI entry point.

    Usage
    -----
    # One or more explicit pair directories
    python 3_do_corrections.py /.../path150_20071216_20080131_SRTM /.../path150_20080131_20080317_3DEP

    # Batch over a root (auto-detects path* subdirectories)
    python 3_do_corrections.py --batch /mnt/DATA2/bakke326l/processing/interferograms

    Arguments
    ---------
    pairdir : Path, optional, repeated
        One or more specific pair directories to process.
    --batch : Path, optional
        Root directory from which to collect path* subdirectories.

    Behavior
    --------
    - If neither `pairdir` nor `--batch` is provided, exits with a message.
    - Continues across pairs, reporting per-pair success/skip counts.
    """
    ap = argparse.ArgumentParser(
        description="Apply ionosphere-only and tropospheric corrections to pair directories; export vertical displacement for RAW, TROPO, IONO, TROPO_IONO, and quicklooks for ΔAPS + iono dispersive. All outputs are masked to the SAR swath."
    )
    ap.add_argument("pairdir", nargs="*", type=Path,
                    help="One or more individual pair directories (…/pathXXX_REF_SEC_SRTM or …_3DEP)")
    ap.add_argument("--batch", type=Path,
                    help="Root folder containing many pair directories (scans for 'path*')")
    args = ap.parse_args()

    targets: list[Path] = []
    if args.batch:
        targets.extend(collect_pairs(args.batch))
    targets.extend(args.pairdir)

    if not targets:
        sys.exit("Nothing to do. Supply one or more pair dirs or --batch ROOT.")

    n_ok = 0
    n_skip = 0
    for p in targets:
        processed = process_pair(p.resolve())
        if processed:
            n_ok += 1
        else:
            n_skip += 1

    print(f"🏁 Done. Processed: {n_ok} | Skipped: {n_skip}")

if __name__ == "__main__":
    main()
