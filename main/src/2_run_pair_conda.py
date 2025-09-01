#!/usr/bin/env python3
"""
run_allpairs.py
================

Purpose
-------
Run ALOS pairs listed in /home/bakke326l/InSAR/main/data/pairs.csv with fixed
settings (range=10, az=16, alpha=0.6). For **each pair** the script runs **twice**:
once using the **SRTM DEM** and once using the **3DEP DTM**, then moves on
to the next pair.

Where results go
----------------
/mnt/DATA2/bakke326l/processing/interferograms/
    path<PATH>_<REF>_<SEC>_SRTM/   # ISCE workdir + products + inspect quicklooks
    path<PATH>_<REF>_<SEC>_3DEP/

Quicklook PNG copies are also written to:
~/InSAR/main/processing/inspect/

Inputs & assumptions
--------------------
- CSV: /home/bakke326l/InSAR/main/data/pairs.csv
  columns: path,reference,secondary   (header is skipped)
  All rows are processed as-is (no path filter).
- DEMs:
  - SRTM:  /home/bakke326l/InSAR/main/data/aux/dem/srtm_30m.tif
  - 3DEP:  /home/bakke326l/InSAR/main/data/aux/dem/3dep_10m.tif
  (The script auto-builds the *.dem.wgs84 and *.wgs84.xml if missing.)
- ISCE2 is installed and ISCE_HOME is set.
- gdal_translate is available on PATH.

How to run
----------
# 1) Batch mode (default): run ALL pairs from the CSV
python 2_run_pair_conda.py

# 2) Batch but limit number of rows (useful for smoke tests)
python 2_run_pair_conda.py --n 10

# 3) Single pair (also runs both SRTM and 3DEP)
python 2_run_pair_conda.py --path 150 20071216 20080131
# (omit --path to default to 150)

# 4) (Optional) override fixed processing settings
python 2_run_pair_conda.py --range 10 --az 16 --alpha 0.6

# 5) Resume mode (skip ISCE if interferogram already exists, rebuild inspect)
python 2_run_pair_conda.py --resume
"""

from __future__ import annotations
import argparse, csv, os, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
from subprocess import check_call, CalledProcessError
import numpy as np, rasterio
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Make GDAL more memory-friendly for large translates
os.environ.setdefault("GDAL_CACHEMAX", "512")  # MB

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
SRTM_TIF     = BASE / "data" / "aux" / "dem" / "srtm_30m.tif"
THREEDEP_TIF = BASE / "data" / "aux" / "dem" / "3dep_10m.tif"

HOME_PROC   = BASE / "processing"

# Status log
REPORT_DIR  = PROC_DIR / "_reports"
STATUS_CSV  = REPORT_DIR / "path_status.csv"

# Fixed processing knobs (no sweep)
DEF_RANGE = 10
DEF_AZ    = 16
DEF_ALPHA = 0.6

# ─────────────────────── status logging ───────────────────────
def _log_status(path: int, ref: str, sec: str, dem: str,
                action: str, result: str, notes: str, workdir: Path) -> None:
    """
    Append one row to the status CSV.
    action: 'run_isce' | 'resume' | 'outer'
    result: 'ok' | 'failed' | 'skipped'
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    header_needed = not STATUS_CSV.exists()
    with STATUS_CSV.open("a", newline="") as f:
        if header_needed:
            f.write("timestamp,path,ref,sec,dem,action,result,notes,workdir\n")
        safe_notes = (notes or "").replace("\n", " ").replace(",", ";")
        row = f"{datetime.utcnow().isoformat()}Z,{path},{ref},{sec},{dem},{action},{result},{safe_notes},{workdir}\n"
        f.write(row)

# ─────────────────────── utilities ───────────────────────
def _quicklook_png(src: Path, dst: Path, band: int = 1, max_wh: int = 4096) -> None:
    """
    Fast, low-RAM PNG quicklook from any raster (VRT/TIF).
    Downsamples on read so we never load a full huge array.
    """
    turbo = colormaps.get_cmap("turbo")
    from rasterio.enums import Resampling
    with rasterio.open(src) as ds:
        scale = max(1, int(np.ceil(max(ds.width, ds.height) / max_wh)))
        out_h, out_w = max(1, ds.height // scale), max(1, ds.width // scale)
        arr = ds.read(
            band,
            out_shape=(1, out_h, out_w),
            resampling=Resampling.bilinear,
            masked=True,
        ).squeeze()
    if hasattr(arr, "mask") and np.asarray(arr.mask).all():
        plt.imsave(dst, np.zeros((out_h, out_w, 3), dtype=np.uint8))
        return
    vmin, vmax = np.percentile(arr.compressed(), [1, 99])
    if not np.isfinite([vmin, vmax]).all() or vmin == vmax:
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = (turbo(np.ma.filled(norm, 0)) * 255).astype(np.uint8)
    plt.imsave(dst, rgba[..., :3])

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

def _core_ifg_exists(wdir: Path) -> bool:
    """Detect whether the heavy ISCE stage finished for the pair."""
    ig = wdir / "interferogram"
    return (ig / "filt_topophase.unw.geo.vrt").exists() or (ig / "filt_topophase.unw.geo").exists()

def _safe_translate(src: Path, dst: Path) -> bool:
    """gdal_translate wrapper that won't crash the batch."""
    try:
        subprocess.check_call([
            "gdal_translate",
            "-of", "GTiff",
            "-co", "TILED=YES",
            "-co", "COMPRESS=DEFLATE",
            "-co", "BIGTIFF=IF_SAFER",
            str(src), str(dst)
        ])
        return True
    except CalledProcessError as e:
        print(f"⚠️  gdal_translate failed: {src.name} → {e}")
        return False
    except FileNotFoundError:
        print("❌  gdal_translate not found on PATH.")
        return False

# ───────────────────── quicklooks builder ─────────────────────
def _make_quicklooks(wdir: Path, ref: str, sec: str, dem_label: str) -> tuple[int, int]:
    """
    Build inspect TIFs/PNGs; robust to missing pieces.
    Returns (num_ok, num_missing).
    """
    ok = 0
    missing = 0
    inspect_dir = wdir / "inspect"
    inspect_dir.mkdir(exist_ok=True, parents=True)
    igram_dir = wdir / "interferogram"

    # For phsig.cor: skip TIF copy (PNG directly from VRT to avoid heavy I/O)
    product_specs = [
        (igram_dir / "phsig.cor.geo.vrt",          False),  # no TIF; PNG from VRT
        (igram_dir / "topophase.cor.geo.vrt",      True),   # make TIF + PN
        (igram_dir / "filt_topophase.unw.geo.vrt", True),   # make TIF + PNG
    ]
    for src, make_tif in product_specs:
        if not src.exists():
            print(f"⚠️  Missing product (skipping): {src}")
            missing += 1
            continue
        png = inspect_dir / f"{src.stem}_{ref}_{sec}_{dem_label}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        if make_tif:
            tif = inspect_dir / f"{src.stem}_{ref}_{sec}_{dem_label}.tif"
            if _safe_translate(src, tif):
                ok += 1
                try:
                    _quicklook_png(tif, png)
                except Exception as e:
                    print(f"⚠️  quicklook failed for {tif.name}: {e}")
            else:
                missing += 1
        else:
            # phsig.cor: create PNG directly from VRT to reduce memory/IO
            try:
                _quicklook_png(src, png)
                ok += 1
            except Exception as e:
                print(f"⚠️  quicklook (direct VRT) failed for {src.name}: {e}")
                missing += 1

    # Fringes (helper names with params → we rename if found)
    try:
        create_fringe_tif(work_dir=wdir, out_dir=inspect_dir,
                          ref=ref, sec=sec, range_looks=DEF_RANGE,
                          azi_looks=DEF_AZ, alpha=DEF_ALPHA)
        old = inspect_dir / f"FRINGES_{ref}_{sec}_ra{DEF_RANGE}_az{DEF_AZ}_{DEF_ALPHA}.tif"
        new = inspect_dir / f"FRINGES_{ref}_{sec}_{dem_label}.tif"
        if old.exists():
            try:
                shutil.move(old, new)
                _quicklook_png(new, inspect_dir / f"FRINGES_{ref}_{sec}_{dem_label}.png")
                ok += 1
            except Exception as e:
                print(f"⚠️  fringe rename/quicklook failed: {e}")
                missing += 1
        else:
            missing += 1
    except Exception as e:
        print(f"⚠️  create_fringe_tif failed: {e}")
        missing += 1

    return ok, missing

# ─────────────────────────── main runner ───────────────────────────
def run_pair_with_dem(
    path: int,
    ref: str,
    sec: str,
    dem_label: str,     # "SRTM" or "3DEP"
    dem_tif: Path,      # source TIF to convert for ISCE
    range_looks: int = DEF_RANGE,
    az_looks: int = DEF_AZ,
    alpha: float = DEF_ALPHA,
    resume: bool = False,
) -> None:
    """
    Run ONE pair for ONE DEM/DTM. Naming: path<path>_<ref>_<sec>_<DEM>.
    Adds status logging and resume support.
    """
    dem_wgs84 = _ensure_dem(dem_tif)
    pair_id   = _pair_dir_name(path, ref, sec, dem_label)

    wdir        = PROC_DIR / pair_id
    inspect_dir = wdir / "inspect"

    if not wdir.exists():
        wdir.mkdir(parents=True, exist_ok=True)
        print(f"🆕 New work dir: {wdir}")
    else:
        if (wdir / "PICKLE").exists():
            print(f"🔁 Resuming existing run in {wdir} (PICKLE/ found)")
        else:
            print(f"🔁 Reusing existing dir {wdir}")
    inspect_dir.mkdir(parents=True, exist_ok=True)

    # Resume path: skip ISCE if core IFG exists
    if resume and _core_ifg_exists(wdir):
        ok, miss = _make_quicklooks(wdir, ref, sec, dem_label)
        note = f"resume_only: quicklooks ok={ok}, missing={miss}"
        print(f"✅ done (resume): {pair_id} — {note}")
        _log_status(path, ref, sec, dem_label, action="resume",
                    result="ok", notes=note, workdir=wdir)
        return

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
    try:
        isce_home   = Path(os.environ["ISCE_HOME"])
        stripmap_py = isce_home / "applications" / "stripmapApp.py"
        if not stripmap_py.exists():
            raise FileNotFoundError(f"stripmapApp.py not found in {isce_home}")
        cmd = [sys.executable, str(stripmap_py), str(xml_file), "--steps"]
        print("🚀", " ".join(cmd))
        subprocess.check_call(cmd, cwd=wdir)
    except Exception as e:
        msg = f"isce_failed: {type(e).__name__}: {e}"
        print(f"❌ {msg}")
        _log_status(path, ref, sec, dem_label, action="run_isce",
                    result="failed", notes=msg, workdir=wdir)
        return

    # Export quicklooks
    ok, miss = _make_quicklooks(wdir, ref, sec, dem_label)
    note = f"run_isce: quicklooks ok={ok}, missing={miss}"
    print(f"📑 results: {wdir}")
    print(f"✅ done: {pair_id} — {note}")
    _log_status(path, ref, sec, dem_label, action="run_isce",
                result="ok", notes=note, workdir=wdir)

# ─────────────────────────── batch (no path filter) ───────────────────────────
def batch_from_csv(n_pairs: int = 999999, resume: bool = False) -> None:
    """
    Read pairs.csv and run *every* row (no path filter).
    For each (path, ref, sec) run SRTM then 3DEP before moving to the next row.
    """
    csv_path = PAIRS_CSV
    with csv_path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)  # skip header if present
        for i, row in enumerate(rdr):
            if i >= n_pairs:
                break
            path, ref, sec = int(row[0]), row[1], row[2]
            for dem_label, dem_tif in (("3DEP", THREEDEP_TIF), ("SRTM", SRTM_TIF)):
                try:
                    run_pair_with_dem(path, ref, sec, dem_label, dem_tif, resume=resume)
                except Exception as e:
                    msg = f"outer_failed: {type(e).__name__}: {e}"
                    print(f"❌ Pair {path}_{ref}_{sec}_{dem_label} failed in outer loop: {e}")
                    _log_status(path, ref, sec, dem_label,
                                action="outer", result="failed",
                                notes=msg,
                                workdir=PROC_DIR / _pair_dir_name(path, ref, sec, dem_label))
                    continue

# ─────────────────────────── CLI ───────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run pairs from CSV (both SRTM and 3DEP) or a single pair. Supports --resume."
    )
    p.add_argument("reference", nargs="?", help="Reference date YYYYMMDD (single-pair mode)")
    p.add_argument("secondary", nargs="?", help="Secondary date YYYYMMDD (single-pair mode)")
    p.add_argument("--path",   type=int, default=150, help="Path/track for single-pair mode (default 150)")
    p.add_argument("--range",  type=int,   default=DEF_RANGE, help="Range looks (fixed)")
    p.add_argument("--az",     type=int,   default=DEF_AZ,    help="Azimuth looks (fixed)")
    p.add_argument("--alpha",  type=float, default=DEF_ALPHA, help="Goldstein alpha (fixed)")
    p.add_argument("--n",      type=int,   default=999999,    help="Limit number of CSV rows (batch mode)")
    p.add_argument("--resume", action="store_true",
                   help="Skip ISCE if interferogram exists and only (re)build inspect outputs.")
    args = p.parse_args()

    if args.reference and args.secondary:
        # single pair (also run both SRTM and 3DEP)
        for dem_label, dem_tif in (("3DEP", THREEDEP_TIF), ("SRTM", SRTM_TIF)):
            run_pair_with_dem(args.path, args.reference, args.secondary, dem_label, dem_tif,
                              range_looks=args.range, az_looks=args.az, alpha=args.alpha,
                              resume=args.resume)
    else:
        # batch from CSV, both DEMs per row, NO path filter
        batch_from_csv(args.n, resume=args.resume)
