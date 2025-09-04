#!/usr/bin/env python3
"""
2_run_insar.py — Batch-run ALOS PALSAR interferograms with fixed settings (SRTM & 3DEP)

Purpose
-------
Batch-process ALOS pairs listed in a CSV using ISCE2 with fixed parameters
(range=10, az=16, alpha=0.6). For each pair the script runs **twice**:
first with the **3DEP DTM**, then with the **SRTM DEM**. It supports
resume-safe execution and writes a simple status log per attempt.

Needed data (inputs & assumptions)
----------------------------------
- Pairs table (CSV): /home/bakke326l/InSAR/main/data/pairs.csv
  Columns: path,reference,secondary   (header is skipped automatically)
  • All rows are processed as-is (no path filtering in batch mode).
- DEM sources (GeoTIFFs):
  • SRTM : /home/bakke326l/InSAR/main/data/aux/dem/srtm_30m.tif
  • 3DEP : /home/bakke326l/InSAR/main/data/aux/dem/3dep_10m.tif
  The script will auto-generate ISCE-compatible binaries alongside each TIF:
  <tif>.dem.wgs84 and <tif>.wgs84.xml
- Raw ALOS archives organized under:
  /mnt/DATA2/bakke326l/raw/
- ISCE2 installed and usable via $ISCE_HOME/applications/stripmapApp.py
- System GDAL: gdal_translate must be on PATH.

Dependencies
------------
- Python: numpy, rasterio, matplotlib
- System tools: gdal_translate
- Local helpers (same repo/dir):
  • help_xml_isce.write_stripmap_xml
  • help_xml_dem.write_dem_report  (+ CLI in help_xml_dem.py)
  • help_show_fringes.create_fringe_tif

Outputs & directories
---------------------
- Work dirs per pair & DEM:
  /mnt/DATA2/bakke326l/processing/interferograms/
      path<PATH>_<REF>_<SEC>_3DEP/
      path<PATH>_<REF>_<SEC>_SRTM/
  Containing:
  • stripmapApp.xml
  • interferogram/  (e.g., filt_topophase.unw.geo(.vrt), topophase.cor.geo.vrt, phsig.cor.geo.vrt)
  • inspect/        (GeoTIFF + PNG quicklooks for key products, plus FRINGES_*.tif/.png)
- DEM binaries and XML sidecars are created **next to** the source TIFs.
- Status log (append-only):
  /mnt/DATA2/bakke326l/processing/interferograms/_reports/path_status.csv

How to run
----------
# 1) Batch mode: run ALL pairs from the CSV (default parameters)
python 2_run_insar.py

# 2) Batch but limit number of rows (smoke test)
python 2_run_insar.py --n 10

# 3) Single pair (runs both 3DEP and SRTM)
python 2_run_insar.py --path 150 20071216 20080131

# 4) Override fixed processing knobs
python 2_run_insar.py --range 10 --az 16 --alpha 0.6

# 5) Resume mode (skip ISCE if IFG exists; rebuild inspect only)
python 2_run_insar.py --resume

Notes
-----
- Resume mode detects completion via the presence of filt_topophase.unw.geo(.vrt)
  inside the pair's interferogram/ directory.
- Quicklooks are generated with downsampled reads to reduce RAM and file size.
- Failures are recorded in _reports/path_status.csv with action and notes.
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
    Run ONE pair for ONE DEM/DTM with ISCE2 and build inspect quicklooks.

    Parameters
    ----------
    path : int
        ALOS path/track number (e.g., 150).
    ref, sec : str
        Reference and secondary acquisition dates in YYYYMMDD.
    dem_label : str
        Label for the elevation model ("SRTM" or "3DEP") used in naming.
    dem_tif : pathlib.Path
        Source DEM/DTM GeoTIFF to convert into ISCE WGS84 binary.
    range_looks : int, default=DEF_RANGE
    az_looks : int, default=DEF_AZ
    alpha : float, default=DEF_ALPHA
        Goldstein filter strength.
    resume : bool, default=False
        If True and a core IFG already exists, skip ISCE run and only (re)build inspect outputs.

    Side effects
    ------------
    - Ensures <tif>.dem.wgs84 and .wgs84.xml exist next to the input DEM TIF.
    - Creates the work dir:
    /mnt/DATA2/bakke326l/processing/interferograms/path<path>_<ref>_<sec>_<DEM>/
    - Writes stripmapApp.xml, runs stripmapApp.py --steps, and generates:
    • interferogram/ products (VRT/TIF as produced by ISCE2)
    • inspect/ quicklooks: PNGs (and lightweight GeoTIFF copies where configured)
    • FRINGES_<ref>_<sec>_<DEM>.tif/.png (renamed from helper’s parameterized name)
    - Appends a status row to:
    /mnt/DATA2/bakke326l/processing/interferograms/_reports/path_status.csv
    with action ∈ {"run_isce","resume"} and result ∈ {"ok","failed"}.

    Raises
    ------
    FileNotFoundError
        If $ISCE_HOME/applications/stripmapApp.py cannot be found.
    subprocess.CalledProcessError
        If external commands fail (help_xml_dem.py or stripmapApp.py).
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
    Read pairs.csv and execute every row (no path filter). For each (path, ref, sec),
    run 3DEP first then SRTM, capturing failures and continuing.

    Parameters
    ----------
    n_pairs : int, default=999999
        Upper limit on the number of CSV rows to process (for smoke tests).
    resume : bool, default=False
        If True, skip ISCE runs for pairs whose core IFG already exists
        and only (re)build inspect outputs.

    Inputs
    ------
    CSV at /home/bakke326l/InSAR/main/data/pairs.csv with columns:
    path,reference,secondary  (a single header row is skipped automatically)

    Outputs
    -------
    Work dirs under:
    /mnt/DATA2/bakke326l/processing/interferograms/path<PATH>_<REF>_<SEC>_{3DEP,SRTM}/
    Status CSV appended at:
    /mnt/DATA2/bakke326l/processing/interferograms/_reports/path_status.csv

    Behavior
    --------
    - Continues past failures, logging each with action="outer" and result="failed".
    - Processes rows in file order up to n_pairs.
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
