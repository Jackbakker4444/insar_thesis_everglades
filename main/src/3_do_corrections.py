#!/usr/bin/env python3
"""
3_do_corrections.py
==========================

Run ionospheric and tropospheric corrections *after* interferograms are built.
Works with your directory naming:

  /mnt/DATA2/bakke326l/processing/interferograms/
      path150_<REF>_<SEC>_SRTM/
      path150_<REF>_<SEC>_3DEP/

For each pair dir it produces:
  interferogram/
    filt_topophase_iono.unw.geo            # IONO-only (single-band phase Geo)
    filt_topophase_tropo.unw.geo           # TROPO-only intermediate (2-band ENVI)
    filt_topophase_tropo_iono.unw.geo      # TROPO+IONO (single-band phase Geo)
  inspect/
    <above>.tif + PNG quicklooks
    vertical_displacement_cm_<REF>_<SEC>_{RAW|TROPO|IONO|TROPO_IONO}.geo.tif
    (plus PNG quicklooks)

Usage
-----
# one specific pair directory
python 3_do_corrections.py /mnt/DATA2/bakke326l/processing/interferograms/path150_20071216_20080131_SRTM

# batch under a root (will scan for dirs starting with 'path')
python 3_do_corrections.py --batch /mnt/DATA2/bakke326l/processing/interferograms
"""

from __future__ import annotations
import argparse
import sys
import subprocess
from pathlib import Path

# --- silence DEBUG spam ---
import os, logging
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
    """Save BAND as PNG."""
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
    """gdal_translate wrapper."""
    subprocess.check_call(["gdal_translate", "-of", "GTiff", str(src), str(dst)])

def parse_pair_id(pairdir: Path) -> tuple[int, str, str, str]:
    """
    Parse path/ref/sec/DEMlabel from: path150_YYYYMMDD_YYYYMMDD_SRTM (or _3DEP)
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
    Return the best available base unwrapped file to correct against.
    Prefers the .geo (ENVI) file, falls back to .vrt if needed.
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
    True where pixels are in-swath (valid), False outside.
    Preference order:
      1) Use GDAL mask of the phase band.
      2) If no useful mask and dataset has 2 bands (amp+phase), use amp != 0.
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
    Read phase from an unwrapped product and apply swath mask:
      - For 2-band ISCE unw (amp+phase), phase is band 2.
      - For single-band corrected (iono/tropo_iono), phase is band 1.
    Returns (phase [float32 with NaNs off-swath], profile, crs, transform, shape).
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
    Convert unwrapped phase (radians) to vertical displacement (cm):
      d_LOS = (λ / 4π) * phase, λ=0.2362 m (ALOS L-band)
      vertical = - (d_LOS_zeroed / cos(inc))
    We remove the **LOS median** before division to prevent a 1/cos ramp.
    Pixels outside the SAR swath are forced to NaN (via _read_phase()).
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

    # quicklook(s)
    try:
        png = out_path.with_suffix(".png")
        quicklook_png(out_path, png, band=1)
        (HOME_PROC / "inspect").mkdir(parents=True, exist_ok=True)
        quicklook_png(out_path, HOME_PROC / "inspect" / out_path.name.replace(".tif", ".png"), band=1)
    except Exception as e:
        print(f"⚠️  vertical PNG failed: {e}")

# --------------------------- readiness check (unchanged) ---------------------------
def readiness(pairdir: Path) -> tuple[bool, str | None, Path | None]:
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
    For a given pair directory:
      1) Vertical for RAW (swath-masked).
      2) IONO-only correction → vertical (swath-masked).
      3) TROPO correction → vertical (swath-masked) → IONO on that (TROPO+IONO) → vertical (swath-masked).
      4) Export TIF quicklooks along the way.
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

    # Base unwrapped (guaranteed by readiness)
    base_unw = find_base_unw(igram_dir)

    # ---------------- RAW vertical ----------------
    if los_path is not None:
        print(f"🧭  RAW vertical (no atmos correction): {pairdir.name}")
        try:
            out_raw = inspect_dir / f"vertical_displacement_cm_{ref}_{sec}_RAW.geo.tif"
            write_vertical_cm(base_unw, los_path, out_raw, label="RAW")
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

    iono_unw = igram_dir / "filt_topophase_iono.unw.geo"
    if iono_unw.exists():
        # quicklooks for corrected unwrapped
        try:
            tif = inspect_dir / f"{iono_unw.stem}_{ref}_{sec}.tif"
            translate_to_tif(iono_unw, tif)
            quicklook_png(iono_unw, HOME_PROC / "inspect" / f"{iono_unw.stem}_{ref}_{sec}.png")
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

    tropo_unw = igram_dir / "filt_topophase_tropo.unw.geo"
    if tropo_unw.exists():
        # quicklooks for tropo-only unwrapped
        try:
            tif_tropo = inspect_dir / f"{tropo_unw.stem}_{ref}_{sec}.tif"
            translate_to_tif(tropo_unw, tif_tropo)
            quicklook_png(tropo_unw, HOME_PROC / "inspect" / f"{tropo_unw.stem}_{ref}_{sec}.png")
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
            # quicklooks
            try:
                tif_ti = inspect_dir / f"{tropo_iono_unw.stem}_{ref}_{sec}.tif"
                translate_to_tif(tropo_iono_unw, tif_ti)
                quicklook_png(tropo_iono_unw, HOME_PROC / "inspect" / f"{tropo_iono_unw.stem}_{ref}_{sec}.png")
            except Exception as e:
                print(f"⚠️  tropo+iono export failed: {e}")

            # vertical (TROPO_IONO)
            if los_path is not None:
                try:
                    out_ti = inspect_dir / f"vertical_displacement_cm_{ref}_{sec}_TROPO_IONO.geo.tif"
                    write_vertical_cm(tropo_iono_unw, los_path, out_ti, label="TROPO_IONO")
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
    """Return immediate subdirectories that look like pair dirs."""
    return sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("path")])

# ---------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply ionosphere-only and tropospheric corrections to pair directories; export vertical displacement for RAW, TROPO, IONO, TROPO_IONO. All outputs are masked to the SAR swath."
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