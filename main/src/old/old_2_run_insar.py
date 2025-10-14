#!/usr/bin/env python3
"""
run_path150_allpairs.py
=======================

Purpose
-------
Run **all ALOS pairs for path 150** listed in
/home/bakke326l/InSAR/main/data/pairs.csv with fixed settings
(range=10, az=16, alpha=0.6). For **each pair** the script runs **twice**:
once using the **SRTM DEM** and once using the **3DEP DTM**, then moves on
to the next pair.

Where results go
----------------
/mnt/DATA2/bakke326l/processing/interferograms/
    path150_<REF>_<SEC>_SRTM/   # ISCE workdir + products + inspect quicklooks
    path150_<REF>_<SEC>_3DEP/

Quicklook PNG copies are also written to:
~/InSAR/main/processing/inspect/

Inputs & assumptions
--------------------
- CSV: /home/bakke326l/InSAR/main/data/pairs.csv
  columns: path,reference,secondary   (header is skipped)
  Only rows with path == 150 are processed.
- DEMs:
  - SRTM:  /home/bakke326l/InSAR/main/data/aux/dem/srtm_30m.tif
  - 3DEP:  /home/bakke326l/InSAR/main/data/aux/dem/3dep_10m.tif
  (The script auto-builds the *.dem.wgs84 and *.wgs84.xml if missing.)
- ISCE2 is installed and ISCE_HOME is set.
- gdal_translate is available on PATH.

How to run
----------
# 1) Batch mode (default): run ALL path-150 pairs from the CSV
python 2_run_pair_conda.py

# 2) Batch but limit number of rows (useful for smoke tests)
python 2_run_pair_conda.py --n 10

# 3) Single pair (also runs both SRTM and 3DEP)
python 2_run_pair_conda.py 20071216 20080131

# 4) (Optional) override fixed processing settings
python 2_run_pair_conda.py --range 10 --az 16 --alpha 0.6

Performance tip (optional):
  export OMP_NUM_THREADS=$(nproc)
  export MKL_NUM_THREADS=$OMP_NUM_THREADS

Changing the path
-----------------
This script is hard-wired to path 150. To run a different path from the CSV,
edit the filter in batch_from_csv() (the line `if path != 150: continue`)
or adapt the script to accept a --path flag.

Naming
------
All output folders/files drop parameter names and append the DEM label:
    path150_<REF>_<SEC>_SRTM/
    path150_<REF>_<SEC>_3DEP/
"""

from __future__ import annotations
import argparse, csv, os, shutil, subprocess, sys
from pathlib import Path
from subprocess import check_call
import numpy as np, rasterio
import matplotlib.pyplot as plt
from matplotlib import colormaps

# your helpers
from help_xml_isce import write_stripmap_xml
from help_xml_dem  import write_dem_report
from help_show_fringes import create_fringe_tif

# ───────────────────────── paths ─────────────────────────
BASE        = Path("/home/bakke326l/InSAR/main")
PAIRS_CSV   = BASE / "data" / "pairs.csv"

DATA_BASE   = Path("/mnt/DATA2/bakke326l")
PROC_DIR    = DATA_BASE / "processing" / "interferograms"
RAW_DIR     = DATA_BASE / "raw"

# DEM sources
SRTM_TIF    = BASE / "data" / "aux" / "dem" / "srtm_30m.tif"
THREEDEP_TIF= BASE / "data" / "aux" / "dem" / "3dep_10m.tif"

HOME_PROC   = BASE / "processing"

# Fixed processing knobs (no sweep)
DEF_RANGE = 10
DEF_AZ    = 16
DEF_ALPHA = 0.6

# ─────────────────────── utilities ───────────────────────
def _quicklook_png(src: Path, dst: Path, band: int = 1) -> None:
    """Save tiff as PNG."""
    turbo = colormaps.get_cmap("turbo")
    with rasterio.open(src) as ds:
        arr = ds.read(band, masked=True)
    if arr.mask.all():
        plt.imsave(dst, np.zeros((*arr.shape, 3), dtype=np.uint8))
        return
    vmin, vmax = np.percentile(arr.compressed(), [1, 99])
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    rgba = (turbo(norm.filled(0)) * 255).astype(np.uint8)
    rgb  = rgba[..., :3]
    plt.imsave(dst, rgb)


def _ensure_dem(tif_path: Path) -> Path:
    """
    Make sure <tif>.dem.wgs84 (+ .wgs84.xml) exists for ISCE.
    Returns the path to the .dem.wgs84 binary.
    """
    dem_bin = tif_path.with_suffix(".dem.wgs84")
    dem_xml = dem_bin.with_suffix(".wgs84.xml")
    if not dem_xml.exists():
        print(f"⚙️  Building DEM in WGS84 for {tif_path.name} …")
        check_call([
            sys.executable,
            str(Path(__file__).parent / "help_xml_dem.py"),
            "--input",  str(tif_path),
            "--output", str(dem_bin),
            "--overwrite",
        ])
        write_dem_report(out_bin=dem_bin, keep_egm=False)
    return dem_bin


def _pair_dir_name(path: int, ref: str, sec: str, dem_label: str) -> str:
    """Directory/file stem without parameter names, with DEM tag."""
    return f"path{path}_{ref}_{sec}_{dem_label}"


def run_pair_with_dem(
    path: int,
    ref: str,
    sec: str,
    dem_label: str,     # "SRTM" or "3DEP"
    dem_tif: Path,      # source TIF to convert for ISCE
    range_looks: int = DEF_RANGE,
    az_looks: int = DEF_AZ,
    alpha: float = DEF_ALPHA,
) -> None:
    """
    Run ONE pair for ONE DEM/DTM. Naming: pathXXX_REF_SEC_<DEM>.
    """
    dem_wgs84 = _ensure_dem(dem_tif)
    pair_id   = _pair_dir_name(path, ref, sec, dem_label)

    wdir        = PROC_DIR / pair_id
    inspect_dir = wdir / "inspect"

    # # Create work dirs
    # if wdir.exists():
    #     print(f"🗑  Removing previous directory {wdir}")
    #     shutil.rmtree(wdir)
    # inspect_dir.mkdir(parents=True, exist_ok=True)
    
    # Create/reuse work dirs (do NOT delete so ISCE2 can resume via PICKLE/)
    if not wdir.exists():
        wdir.mkdir(parents=True, exist_ok=True)
        print(f"🆕 New work dir: {wdir}")
    else:
        if (wdir / "PICKLE").exists():
            print(f"🔁 Resuming existing run in {wdir} (PICKLE/ found)")
        else:
            print(f"🔁 Reusing existing dir {wdir}")
    inspect_dir.mkdir(parents=True, exist_ok=True)

    # Write stripmap XML
    xml_file = wdir / "stripmapApp.xml"
    write_stripmap_xml(
        xml_file         = xml_file,
        path             = str(path),
        ref_date         = ref,
        sec_date         = sec,
        raw_dir          = RAW_DIR,
        work_dir         = wdir,
        dem_wgs84        = dem_wgs84,          
        range_looks      = range_looks,
        az_looks         = az_looks,
        filter_strength  = alpha,
    )
    print(f"📝  XML written: {xml_file}")

    # Run ISCE
    isce_home   = Path(os.environ["ISCE_HOME"])
    stripmap_py = isce_home / "applications" / "stripmapApp.py"
    if not stripmap_py.exists():
        sys.exit(f"❌  stripmapApp.py not found in {isce_home}")
    cmd = [sys.executable, str(stripmap_py), str(xml_file), "--steps"]
    print("🚀", " ".join(cmd))
    subprocess.check_call(cmd, cwd=wdir)

    # Export quicklooks
    igram_dir = wdir / "interferogram"
    translate = lambda src, dst: subprocess.check_call(["gdal_translate", "-of", "GTiff", str(src), str(dst)])

    products = [
        igram_dir / "phsig.cor.geo.vrt",
        igram_dir / "topophase.cor.geo.vrt",
        igram_dir / "filt_topophase.unw.geo.vrt",
    ]
    for src in products:
        if not src.exists():
            print(f"⚠️  Missing product (skipping): {src.name}")
            continue
        # filenames 
        tif = inspect_dir / f"{src.stem}_{ref}_{sec}_{dem_label}.tif"
        translate(src, tif)

        png = HOME_PROC / "inspect" / f"{src.stem}_{ref}_{sec}_{dem_label}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        _quicklook_png(src, png)

    # Fringes -> rename to DEM-tagged name (helper creates name with params)
    create_fringe_tif(work_dir=wdir, out_dir=inspect_dir,
                      ref=ref, sec=sec, range_looks=range_looks,
                      azi_looks=az_looks, alpha=alpha)
    old_fringe = inspect_dir / f"FRINGES_{ref}_{sec}_ra{range_looks}_az{az_looks}_{alpha}.tif"
    new_fringe = inspect_dir / f"FRINGES_{ref}_{sec}_{dem_label}.tif"
    if old_fringe.exists():
        shutil.move(old_fringe, new_fringe)
        _quicklook_png(new_fringe, HOME_PROC / "inspect" / f"FRINGES_{ref}_{sec}_{dem_label}.png")

    print(f"📑 results: {wdir}")
    print(f"✅ done: {pair_id}")


# ─────────────────────────── batch (shape preserved) ───────────────────────────
def batch_from_csv(n_pairs: int = 999999) -> None:
    """
    Keep the simple shape of your original:
      - open CSV
      - skip header
      - enumerate rows with an n_pairs ceiling
      - parse (path, ref, sec)
    Here we only run **path 150**, and for **each pair** we run **SRTM then 3DEP**.
    """
    csv_path = Path("/home/bakke326l/InSAR/main/data/pairs.csv")
    with csv_path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)            # skip header if present
        for i, row in enumerate(rdr):
            if i >= n_pairs:
                break
            path, ref, sec = int(row[0]), row[1], row[2]
            if path != 150:
                continue
            # run DEM then DTM before moving to next pair
            run_pair_with_dem(path, ref, sec, "3DEP", THREEDEP_TIF)
            run_pair_with_dem(path, ref, sec, "SRTM", SRTM_TIF)
            


# ─────────────────────────── CLI ───────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run ALL pairs for path 150 (both SRTM and 3DEP) or a single pair."
    )
    p.add_argument("reference", nargs="?", help="Reference date YYYYMMDD (single-pair mode)")
    p.add_argument("secondary", nargs="?", help="Secondary date YYYYMMDD (single-pair mode)")
    p.add_argument("--range",  type=int,   default=DEF_RANGE, help="Range looks (fixed)")
    p.add_argument("--az",     type=int,   default=DEF_AZ,    help="Azimuth looks (fixed)")
    p.add_argument("--alpha",  type=float, default=DEF_ALPHA, help="Goldstein alpha (fixed)")
    p.add_argument("--n",      type=int,   default=999999,    help="Limit number of CSV rows (batch mode)")
    args = p.parse_args()

    if args.reference and args.secondary:
        # single pair (also run both DEMs)
        run_pair_with_dem(150, args.reference, args.secondary, "3DEP", THREEDEP_TIF,
                    range_looks=args.range, az_looks=args.az, alpha=args.alpha)
        run_pair_with_dem(150, args.reference, args.secondary, "SRTM", SRTM_TIF,
                          range_looks=args.range, az_looks=args.az, alpha=args.alpha)
    else:
        # batch, path 150 only, both DEMs per pair
        batch_from_csv(args.n)
