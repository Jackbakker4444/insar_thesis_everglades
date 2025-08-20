#!/usr/bin/env python3
"""
3_do_corrections.py
==========================

Run ionospheric and tropospheric corrections *after* interferograms are built.
Works with your new directory naming:

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
python apply_atmos_corrections.py /mnt/DATA2/bakke326l/processing/interferograms/path150_20071216_20080131_SRTM

# batch under a root (will scan for dirs starting with 'path')
python apply_atmos_corrections.py --batch /mnt/DATA2/bakke326l/processing/interferograms
"""

from __future__ import annotations
import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Helpers you already have; we’ll adapt them next to match the simple API below.
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
    if arr.mask.all():
        plt.imsave(dst, np.zeros((*arr.shape, 3), dtype=np.uint8))
        return
    vmin, vmax = np.percentile(arr.compressed(), [1, 99])
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
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

    # quicklook
    try:
        png = out_path.with_suffix(".png")
        quicklook_png(out_path, png, band=1)
        # also copy to global inspect for convenience
        (HOME_PROC / "inspect").mkdir(parents=True, exist_ok=True)
        quicklook_png(out_path, HOME_PROC / "inspect" / out_path.name.replace(".tif", ".png"), band=1)
    except Exception as e:
        print(f"⚠️  vertical PNG failed: {e}")

# ------------------------------------------------------------ main per-pair
def process_pair(pairdir: Path) -> None:
    """
    For a given pair directory:
      1) Make IONO-only correction from base unwrapped.
      2) Make TROPO correction (GACOS) → then IONO on that (TROPO+IONO).
      3) Export TIF quicklooks.
      4) Make vertical displacement (cm) for both cases.
    """
    try:
        path_no, ref, sec, dem = parse_pair_id(pairdir)
    except Exception as e:
        print(f"⚠️  Skipping {pairdir}: {e}")
        return

    igram_dir   = pairdir / "interferogram"
    tropo_dir   = pairdir / "troposphere"
    inspect_dir = pairdir / "inspect"
    geom_dir    = pairdir / "geometry"
    tropo_dir.mkdir(exist_ok=True)
    inspect_dir.mkdir(exist_ok=True)

    # Geometry LOS (incidence) source
    los_vrt = geom_dir / "los.rdr.geo.vrt"
    los_geo = geom_dir / "los.rdr.geo"
    los_path = los_vrt if los_vrt.exists() else los_geo

    # Base unwrapped
    base_unw = find_base_unw(igram_dir)

    # ---------------- IONO-only ----------------
    print(f"🌌  Ionospheric correction (IONO-only): {pairdir.name}")

    do_iono_correction(work_dir=pairdir, out_dir=igram_dir,
                       input_unw=base_unw, output_suffix="_iono")

    iono_unw = igram_dir / "filt_topophase_iono.unw.geo"
    if iono_unw.exists():
        # export TIF + PNG (no parameter names, dates are enough; DEM is implied by folder)
        tif = inspect_dir / f"{iono_unw.stem}_{ref}_{sec}.tif"
        translate_to_tif(iono_unw, tif)
        quicklook_png(iono_unw, HOME_PROC / "inspect" / f"{iono_unw.stem}_{ref}_{sec}.png")
        # vertical (cm)
        write_vertical_cm(
            iono_unw,
            los_path,
            inspect_dir / f"vertical_displacement_cm_{ref}_{sec}_IONO.geo.tif"
        )
    else:
        print("⚠️  iono-only output not found; check helper implementation.")

    # ---------------- TROPO + IONO ----------------
    print(f"😶‍🌫️  Tropospheric correction (GACOS): {pairdir.name}")
    # Helper API:
    #   do_tropo_correction(wdir, ref, sec, gacos_dir, tropo_dir)
    do_tropo_correction(wdir=pairdir, ref=ref, sec=sec,
                        gacos_dir=TROPO_DIR, tropo_dir=tropo_dir)

    tropo_unw = igram_dir / "filt_topophase_tropo.unw.geo"

    if tropo_unw.exists():
        # Now iono on top of tropo:
        print("🌌  Ionospheric correction on TROPO product (TROPO+IONO)")
        do_iono_correction(work_dir=pairdir, out_dir=igram_dir,
                           input_unw=tropo_unw, output_suffix="_tropo_iono")

        tropo_iono_unw = igram_dir / "filt_topophase_tropo_iono.unw.geo"

        # Exports for TROPO (intermediate) and TROPO+IONO (final)
        try:
            tif_tropo = inspect_dir / f"{tropo_unw.stem}_{ref}_{sec}.tif"
            translate_to_tif(tropo_unw, tif_tropo)
            quicklook_png(tropo_unw, HOME_PROC / "inspect" / f"{tropo_unw.stem}_{ref}_{sec}.png")
        except Exception as e:
            print(f"⚠️  tropo export failed: {e}")

        if tropo_iono_unw.exists():
            tif_ti = inspect_dir / f"{tropo_iono_unw.stem}_{ref}_{sec}.tif"
            translate_to_tif(tropo_iono_unw, tif_ti)
            quicklook_png(tropo_iono_unw, HOME_PROC / "inspect" / f"{tropo_iono_unw.stem}_{ref}_{sec}.png")
            # vertical (cm)
            write_vertical_cm(
                tropo_iono_unw,
                los_path,
                inspect_dir / f"vertical_displacement_cm_{ref}_{sec}_TROPO_IONO.geo.tif"
            )
        else:
            print("⚠️  tropo+iono output not found; check helper implementation.")
    else:
        print("⚠️  tropo output not found; skipping TROPO+IONO branch.")

    print(f"✅  Corrections complete for {pairdir}\n")

# -------------------------------------------------------------- batch helpers
def collect_pairs(root: Path) -> list[Path]:
    """Return immediate subdirectories that look like pair dirs."""
    return [d for d in root.iterdir() if d.is_dir() and d.name.startswith("path")]

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

    for p in targets:
        process_pair(p.resolve())

if __name__ == "__main__":
    main()
