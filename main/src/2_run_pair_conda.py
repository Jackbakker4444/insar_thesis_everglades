#!/usr/bin/env python3
"""
run_pair_conda.py – batch-process multiple ALOS pairs with ISCE2
and sweep multilook / filter parameters.

If called **without CLI arguments** the script will:
  • read the first 4 rows of /home/bakke326l/InSAR/main/data/pairs.csv
  • run them for every combo in COMBOS (see below)

You can still run a *single* pair manually:

    python src/run_pair_conda.py 150 20071216 20080131 --range 4 --az 1 --alpha 0.8
"""

from __future__ import annotations
import argparse, csv, os, shutil, subprocess, sys
from pathlib import Path
from subprocess import check_call

from help_xml_isce import write_stripmap_xml
from help_xml_dem  import write_dem_report
from help_show_fringes import create_fringe_tif
from help_atm_correction import do_iono_correction, do_tropo_correction

# ───────────────────────── constants ──────────────────────────
ABS_BASE   = Path(__file__).resolve().parents[5]
DATA_BASE  = ABS_BASE / "mnt" / "DATA2" / "bakke326l"
PROC_DIR   = DATA_BASE / "processing" / "tuning"          # <— new
RAW_DIR    = DATA_BASE / "raw"

BASE       = Path(__file__).resolve().parents[1]           # ~/InSAR/main
DEM_TIF    = BASE / "data" / "aux" / "dem" / "srtm_30m.tif"
TROPO_DIR  = BASE / "data" / "aux" / "tropo"
DEM_WGS84  = BASE / "data" / "aux" / "dem" / "srtm_30m.dem.wgs84"

# sweep these six combos
COMBOS: list[tuple[int, int, float]] = [
    (4, 1, 0.8),
    (4, 4, 0.7),
    (8, 3, 0.8),
    (10, 2, 0.5),
    (2, 1, 0.8),
    (4, 4, 0.9),
]

# ─────────────────────── helper: one pair, one combo ───────────────────────
def run_pair(
    path: int,
    ref: str,
    sec: str,
    range_looks: int,
    az_looks: int,
    alpha: float,
) -> None:
    """
    Prepare XML → run stripmapApp → export quick-look tiffs.
    All outputs live in
        mnt/DATA2/bakke326l/processing/tuning/
        path{path}_{ref}_{sec}_{range}x_{az}x_{alpha}/
    """
    pair_id = f"path{path}_{ref}_{sec}_ra{range_looks}_az{az_looks}_filter{alpha}"
    wdir    = PROC_DIR / pair_id
    inspect_dir = wdir / "inspect"

    # (re)create work dirs
    if wdir.exists():
        print(f"🗑  Removing previous directory {wdir}")
        shutil.rmtree(wdir)
    inspect_dir.mkdir(parents=True, exist_ok=True)

    # make sure DEM (wgs84) exists --------------------------------------------------
    dem_bin = DEM_TIF.with_suffix(".dem.wgs84")
    dem_xml = dem_bin.with_suffix(".wgs84.xml")
    if not dem_xml.exists():
        print("⚙️  Making DEM wgs84 …")
        check_call([
            sys.executable,
            str(Path(__file__).parent / "dem_xml.py"),
            "--input",  str(DEM_TIF),
            "--output", str(dem_bin),
            "--overwrite",
        ])
        write_dem_report(out_bin=dem_bin, keep_egm=False)

    # write per-pair XML -------------------------------------------------------------
    xml_file = wdir / "stripmapApp.xml"
    write_stripmap_xml(
        xml_file     = xml_file,
        path         = str(path),
        ref_date     = ref,
        sec_date     = sec,
        raw_dir      = RAW_DIR,
        work_dir     = wdir,
        dem_wgs84    = DEM_WGS84,
        range_looks  = range_looks,
        az_looks     = az_looks,
        filter_strength = alpha,
    )
    print(f"📝  XML written: {xml_file}")

    # locate ISCE runner ------------------------------------------------------------
    isce_home   = Path(os.environ["ISCE_HOME"])
    stripmap_py = isce_home / "applications" / "stripmapApp.py"
    if not stripmap_py.exists():
        sys.exit(f"❌  stripmapApp.py not found in {isce_home}")

    # run the full stack
    cmd = [sys.executable, str(stripmap_py), str(xml_file)]
    print("🚀", " ".join(cmd))
    subprocess.check_call(cmd, cwd=wdir)

    # atmos corrections -------------------------------------------------------------
    igram_dir = wdir / "interferogram"
    tropo_dir = wdir / "troposphere"; tropo_dir.mkdir(exist_ok=True)
    
    print(f"😶‍🌫️  Tropospheric correction will now be done for {ref} & {sec}")
    do_tropo_correction(wdir=wdir, ref=ref, sec=sec,
                        gacos_dir=TROPO_DIR, tropo_dir=tropo_dir)
    tropo_corrected = igram_dir / "filt_topophase_tropo.unw.geo"
    print(f"☀️  Tropospheric correction complete, corrected file: {tropo_corrected}")
    
    print(f"🌌  Ionospheric correction will now be done for {ref} & {sec}")
    do_iono_correction(work_dir=wdir, out_dir=igram_dir)
    iono_tropo_corrected = igram_dir / "filt_topophase_tropo_iono.unw.geo"
    print(f"🌍  Ionospheric correction complete, corrected file: {iono_tropo_corrected}")

    # export quick-look GeoTIFFs ----------------------------------------------------
    translate = lambda src, dst: subprocess.check_call(
        ["gdal_translate", "-of", "GTiff", str(src), str(dst)]
    )
    for src in [
        igram_dir/"phsig.cor.geo.vrt",
        igram_dir/"topophase.cor.geo.vrt",
        igram_dir/"filt_topophase.unw.geo.vrt",
        igram_dir/"filt_topophase_tropo.unw.geo",
        igram_dir/"filt_topophase_tropo_iono.unw.geo",
        igram_dir/"filt_topophase_tropo_iono_wrapped.unw.geo",
        tropo_dir / f"{ref}_{sec}.aps.geo",
    ]:
        tif = inspect_dir / (src.name + ".tif")
        translate(src, tif)

    create_fringe_tif(work_dir=wdir, out_dir=inspect_dir)
    
    # End of interferogram build ----------------------------------------------------
    print(f"📑 results can be found at: {wdir}")
    print(f"✅ pair: {pair_id} ran successfully!")

# ─────────────────────────── batch driver ────────────────────────────────
def batch_from_csv(n_pairs: int = 4) -> None:
    csv_path = Path("/home/bakke326l/InSAR/main/data/pairs.csv")
    with csv_path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)            # skip header if present
        for i, row in enumerate(rdr):
            if i >= n_pairs:
                break
            path, ref, sec = int(row[0]), row[1], row[2]
            for r, a, alp in COMBOS:
                run_pair(path, ref, sec, r, a, alp)

# ─────────────────────────── CLI fallback ────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run one ALOS pair (manual) or nothing to batch the first 4 pairs."
    )
    p.add_argument("path", nargs="?", type=int, help="ALOS path/track number")
    p.add_argument("reference", nargs="?", help="Reference date  YYYYMMDD")
    p.add_argument("secondary", nargs="?", help="Secondary date YYYYMMDD")
    p.add_argument("--range",  type=int,   default=4,   help="Range looks")
    p.add_argument("--az",     type=int,   default=1,   help="Azimuth looks")
    p.add_argument("--alpha",  type=float, default=0.8, help="Goldstein alpha")

    args = p.parse_args()

    if args.path is None:
        # no positional args → full batch mode
        batch_from_csv(4)
    else:
        run_pair(args.path, args.reference, args.secondary,
                 args.range, args.az, args.alpha)