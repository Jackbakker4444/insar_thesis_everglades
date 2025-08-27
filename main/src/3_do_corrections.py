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
    filt_topophase_iono.unw.geo            # IONO-only
    filt_topophase_tropo.unw.geo           # TROPO-only intermediate
    filt_topophase_tropo_iono.unw.geo      # TROPO+IONO
  inspect/
    <above>.tif + PNG quicklooks
    vertical_displacement_cm_<REF>_<SEC>_IONO.geo.tif
    vertical_displacement_cm_<REF>_<SEC>_TROPO_IONO.geo.tif
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

# Helpers you already have
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
        arr = ds.read(band, masked=True)
    if hasattr(arr, "mask") and np.all(getattr(arr, "mask")):
        plt.imsave(dst, np.zeros((*arr.shape, 3), dtype=np.uint8))
        return
    vmin, vmax = np.percentile(arr.compressed(), [1, 99])
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

def write_vertical_cm(unw_path: Path, los_path: Path, out_path: Path) -> None:
    """
    Convert unwrapped phase (radians) to vertical displacement (cm):
      d_LOS = (λ / 4π) * phase, λ=0.2362 m (ALOS L-band)
      vertical = - d_LOS / cos(inc)
    """
    WAVELENGTH = 0.2362  # m

    with rasterio.open(unw_path) as ds:
        phase = ds.read(1).astype(np.float32)   # radians
        prof  = ds.profile

    with rasterio.open(los_path) as ds_inc:
        inc_deg = ds_inc.read(1).astype(np.float32)

    inc  = np.deg2rad(inc_deg)
    cosi = np.cos(inc).astype(np.float32)

    d_los_m  = (WAVELENGTH / (4.0 * np.pi)) * phase
    d_vert_m = np.full_like(d_los_m, np.nan, dtype=np.float32)
    ok = np.abs(cosi) > 1e-6
    d_vert_m[ok] = - d_los_m[ok] / cosi[ok]
    d_vert_cm = d_vert_m * 100.0

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

# --------------------------- readiness check (NEW) ---------------------------
def readiness(pairdir: Path) -> tuple[bool, str | None, Path | None]:
    """
    Check whether minimum inputs exist to run corrections.

    Returns (ok, reason_if_not_ok, los_path_or_None)
    - ok=False only when the *base unwrapped* is missing (pair still building)
    - los_path can be None → we will run corrections but skip vertical export
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
    For a given pair directory:
      1) Make IONO-only correction from base unwrapped.
      2) Make TROPO correction (GACOS) → then IONO on that (TROPO+IONO).
      3) Export TIF quicklooks.
      4) Make vertical displacement (cm) for both cases (if LOS available).

    Returns True if processed, False if skipped (e.g., still building).
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

    # ---------------- IONO-only ----------------
    print(f"🌌  Ionospheric correction (IONO-only): {pairdir.name}")
    do_iono_correction(work_dir=pairdir, out_dir=igram_dir,
                       input_unw=base_unw, output_suffix="_iono")

    iono_unw = igram_dir / "filt_topophase_iono.unw.geo"
    if iono_unw.exists():
        try:
            tif = inspect_dir / f"{iono_unw.stem}_{ref}_{sec}.tif"
            translate_to_tif(iono_unw, tif)
            quicklook_png(iono_unw, HOME_PROC / "inspect" / f"{iono_unw.stem}_{ref}_{sec}.png")
        except Exception as e:
            print(f"⚠️  iono export failed: {e}")

        if los_path is not None:
            try:
                write_vertical_cm(
                    iono_unw,
                    los_path,
                    inspect_dir / f"vertical_displacement_cm_{ref}_{sec}_IONO.geo.tif"
                )
            except Exception as e:
                print(f"⚠️  iono vertical export failed: {e}")
        else:
            print("ℹ️  LOS not found → skipping IONO vertical displacement export.")
    else:
        print("⚠️  iono-only output not found; check helper implementation.")

    # ---------------- TROPO + IONO ----------------
    print(f"😶‍🌫️  Tropospheric correction (GACOS): {pairdir.name}")
    try:
        do_tropo_correction(wdir=pairdir, ref=ref, sec=sec,
                            gacos_dir=TROPO_DIR, tropo_dir=tropo_dir)
    except Exception as e:
        print(f"⚠️  tropo correction failed for {pairdir.name}: {e}")

    tropo_unw = igram_dir / "filt_topophase_tropo.unw.geo"

    if tropo_unw.exists():
        try:
            tif_tropo = inspect_dir / f"{tropo_unw.stem}_{ref}_{sec}.tif"
            translate_to_tif(tropo_unw, tif_tropo)
            quicklook_png(tropo_unw, HOME_PROC / "inspect" / f"{tropo_unw.stem}_{ref}_{sec}.png")
        except Exception as e:
            print(f"⚠️  tropo export failed: {e}")

        print("🌌  Ionospheric correction on TROPO product (TROPO+IONO)")
        try:
            do_iono_correction(work_dir=pairdir, out_dir=igram_dir,
                               input_unw=tropo_unw, output_suffix="_tropo_iono")
        except Exception as e:
            print(f"⚠️  iono-on-tropo failed: {e}")

        tropo_iono_unw = igram_dir / "filt_topophase_tropo_iono.unw.geo"
        if tropo_iono_unw.exists():
            try:
                tif_ti = inspect_dir / f"{tropo_iono_unw.stem}_{ref}_{sec}.tif"
                translate_to_tif(tropo_iono_unw, tif_ti)
                quicklook_png(tropo_iono_unw, HOME_PROC / "inspect" / f"{tropo_iono_unw.stem}_{ref}_{sec}.png")
            except Exception as e:
                print(f"⚠️  tropo+iono export failed: {e}")

            if los_path is not None:
                try:
                    write_vertical_cm(
                        tropo_iono_unw,
                        los_path,
                        inspect_dir / f"vertical_displacement_cm_{ref}_{sec}_TROPO_IONO.geo.tif"
                    )
                except Exception as e:
                    print(f"⚠️  tropo+iono vertical export failed: {e}")
            else:
                print("ℹ️  LOS not found → skipping TROPO+IONO vertical displacement export.")
        else:
            print("⚠️  tropo+iono output not found; check helper implementation.")
    else:
        print("⚠️  tropo output not found; skipping TROPO+IONO branch.")

    print(f"✅  Corrections complete for {pairdir}\n")
    return True

# -------------------------------------------------------------- batch helpers
def collect_pairs(root: Path) -> list[Path]:
    """Return immediate subdirectories that look like pair dirs."""
    return sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("path")])

# ---------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply ionosphere-only and tropo+ionosphere corrections to pair directories."
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
