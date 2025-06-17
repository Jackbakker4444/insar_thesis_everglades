#!/usr/bin/env python3
"""
run_pair_conda.py – run one ALOS (raw or SLC) pair with Conda-forge ISCE2.

Usage examples
──────────────
# default folder: processing/path464_20081104_20081220
python src/run_pair_conda.py 464 20081104 20081220

# custom folder (will be deleted if it exists)
python src/run_pair_conda.py 464 20081104 20081220 -w scratch/test1
"""

from __future__ import annotations
import argparse, os, shutil, subprocess, sys
import datetime as dt
from pathlib import Path
from subprocess import check_call

from xml_isce import write_stripmap_xml                            
from xml_dem import write_dem_report
from show_fringes import create_fringe_tif
from atm_correction import do_iono_correction, do_tropo_correction

# ─────────────────────────────── constants ───────────────────────────────────
BASE            = Path(__file__).resolve().parents[1]             # ~/InSAR/main
PROC_DIR        = BASE / "processing"
DEM_TIF         = BASE / "data" / "aux" / "dem" / "srtm_30m.tif"
TROPO_DIR       = BASE / "data" / "aux" / "tropo"


# ──────────────────────────── core functions ────────────────────────────────
def run_pair(path: int, ref: str, sec: str, workdir: Path, step: str, steps:bool, start: str | None = None) -> None:
    """Prepare XML, optionally wipe/prepare workdir, then launch stripmapApp.py."""
    
    # ----------- Get the DEM in the right format for the ISCE to perform -------------
    print("DEM_TIF is in this path:", DEM_TIF)
    dem_bin = DEM_TIF.with_suffix(".dem.wgs84")     # output produced by dsm_to_isce
    dem_xml = dem_bin.with_suffix(".wgs84.xml")
    
    if not dem_xml.exists():
        print("DEM xml:", dem_xml," not found")
        print("⚙️  Creating XML file ...... ")
        keep_egm = False
        # call the CLI once – simplest
        check_call([
            sys.executable,                         # same Python that runs ISCE
            str(Path(__file__).parent / "dem_xml.py"),
            "--input",  str(DEM_TIF),
            "--output", str(dem_bin),
            "--overwrite",
            *(["--keep-egm"] if keep_egm else [])
        ])
        write_dem_report(out_bin = dem_bin, keep_egm = keep_egm)
        
    else:
        print(f"🔍 XML found at:{dem_xml} --> Skipping dem xml creation")
        

    
    # ---------- decide processing directory ----------
    if workdir is None:
        pair_id = f"path{path}_{ref}_{sec}"
        wdir    = PROC_DIR / pair_id
    else:
        wdir = workdir.expanduser().resolve()
        pair_id = workdir.name
        
    if step is not None:
        step_wdir = wdir / "steps" / step
        step_wdir.mkdir(parents = True, exist_ok= True)
    
    # ---------- clean existing directory, then (re)create ----------
    if step is start is None:
        if wdir.exists():
            print(f"🗑️  Removing existing work directory: {wdir}")
            shutil.rmtree(wdir)
        wdir.mkdir(parents=True, exist_ok=True)

    # ---------- build XML if not present ----------
    xml_file = wdir / "stripmapApp.xml"
    write_stripmap_xml(xml_file = xml_file, path = str(path),
                        ref_date = ref, sec_date = sec)
    print(f"📝  Wrote {xml_file}")

    # ---------- locate & call stripmapApp.py ----------
    isce_home   = Path(os.environ["ISCE_HOME"])
    print("[DIR] ISCE_HOME:", isce_home)
    
    stripmap_py = isce_home / "applications" / "stripmapApp.py"
    if not stripmap_py.exists():
        sys.exit(f"❌  Cannot find {stripmap_py}")
    print("[DIR] stripmapApp.py: ", stripmap_py)
    
    # Command for getting al options for steps
    # help_cmd = [sys.executable, str(stripmap_py), str(xml_file), "--steps", "--help"] 
    # subprocess.check_call(help_cmd)
  
    # Running the ISCE stripmappApp
    cmd: list[str]=[sys.executable, str(stripmap_py), str(xml_file)]
    
    if step:
        cmd.append(f"--dostep={step}")
        run_dir = wdir / "steps" / step
    elif start:
        cmd.append(f"--start={start}")
        run_dir = wdir
    elif steps:
        cmd.append("--steps")
        run_dir = wdir
    else:
        run_dir = wdir
    
    run_dir.mkdir(parents=True, exist_ok=True)  
    
    print("🚀 ", " ".join(cmd))  
    subprocess.check_call(cmd, cwd=run_dir)
    
    igram_dir = str(wdir/"inteferogram")
    
    # Apply tropospheric correction
    tropo_dir = wdir / "troposphere"
    tropo_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"😶‍🌫️  Tropospheric correction will now be done for {ref} & {sec}")
    do_tropo_correction(wdir=wdir, ref=ref, sec=sec, gacos_dir=TROPO_DIR, tropo_dir=tropo_dir)
    tropo_corrected = igram_dir / "filt_topophase_tropo.unw.geo"
    print(f"☀️  Tropospheric correction complete, corrected file: {tropo_corrected}")
    
    # Apply ionospheric correction
    print(f"🌌  Ionospheric correction will now be done for {ref} & {sec}")
    do_iono_correction(work_dir = wdir, out_dir = igram_dir)
    iono_tropo_corrected = igram_dir / "filt_topophase_tropo_iono.unw.geo"
    print(f"🌍  Ionospheric correction complete, corrected file: {iono_tropo_corrected}")
    
    print(f"📑 results can be found at: {run_dir}")
    print(f"✅ pair: {pair_id} ran successfully!")
    
    
    # Make fiels suitable for inspecting in QGIS etc. 
    inspect_dir = wdir / "inspect"
    inspect_dir.mkdir(parents = True, exist_ok= True)
    
    
    tif_cmd_phis_cor = ["gdal_translate","-of","GTiff", f"{wdir}/interferogram/phsig.cor.geo.vrt", f"{inspect_dir}/phsig.cor.geo.vrt"+".tif"]
    tif_cmd_cor = ["gdal_translate","-of","GTiff", f"{wdir}/interferogram/topophase.cor.geo.vrt", f"{inspect_dir}/topophase.cor.geo.vrt"+".tif"]
    tif_cmd_unw = ["gdal_translate","-of","GTiff", f"{wdir}/interferogram/filt_topophase.unw.geo.vrt", f"{inspect_dir}/filt_topophase.unw.geo.vrt"+".tif"]
    tif_cmd_tropo = ["gdal_translate","-of","GTiff", f"{wdir}/interferogram/filt_topophase_tropo.unw.geo", f"{inspect_dir}/filt_topophase_tropo.unw.geo"+".tif"]
    tif_cmd_tropo_iono = ["gdal_translate","-of","GTiff", f"{wdir}/interferogram/filt_topophase_tropo_iono.unw.geo", f"{inspect_dir}/filt_topophase_tropo_iono.unw.geo"+".tif"]
    tif_cmd_tropo_iono_wrapped = ["gdal_translate","-of","GTiff", f"{wdir}/interferogram/filt_topophase_tropo_iono_wrapped.unw.geo", f"{inspect_dir}/filt_topophase_tropo_iono_wrapped.unw.geo"+".tif"]
    tif_cmd_troposphere = ["gdal_translate","-of","GTiff", f"{tropo_dir}/{ref}_{sec}.aps.geo", f"{inspect_dir}/LOS_troposphere.geo"+".tif"]
  
    subprocess.check_call(tif_cmd_phis_cor)
    subprocess.check_call(tif_cmd_cor)
    subprocess.check_call(tif_cmd_unw)
    subprocess.check_call(tif_cmd_tropo)
    subprocess.check_call(tif_cmd_tropo_iono)
    subprocess.check_call(tif_cmd_tropo_iono_wrapped)
    subprocess.check_call(tif_cmd_troposphere)

    
    # Creates a map showing 
    create_fringe_tif(work_dir = wdir, out_dir = inspect_dir)
    
    print("🖨️   images converted to tiff")

# ───────────────────────────── CLI ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an ALOS interferometric pair")
    parser.add_argument("path",      type=int, help="ALOS path/track number")
    parser.add_argument("reference",            help="Reference date  YYYYMMDD")
    parser.add_argument("secondary",            help="Secondary date YYYYMMDD")
    parser.add_argument("-w", "--workdir",      help="Custom processing directory (will be removed if it exists)",
                            default=None, type=Path)
    parser.add_argument("--step", type=str,  help=
                        "Step name which you want to be done choose from list[ \
                        verifyDEM, \
                        topo, \
                        geo2rdr, \
                        coarse_resample, \
                        misregistration, \
                        refined_resample, \
                        dense_offsets, \
                        rubber_sheet_range, \
                        rubber_sheet_azimuth, \
                        fine_resample, \
                        split_range_spectrum, \
                        sub_band_resample, \
                        interferogram, \
                        'sub_band_interferogram', \
                        filter, \
                        unwrap, \
                        geocode \
                        ]", 
                        default=None)
    parser.add_argument("--steps", type=bool,  help=
                        "When turned on, all step outputs are saved in pickle", 
                        default=None)
    parser.add_argument("--start", type=str,  help=
                        "Step name which you want to start from, choose from list[ \
                        verifyDEM, \
                        topo, \
                        geo2rdr, \
                        coarse_resample, \
                        misregistration, \
                        refined_resample, \
                        dense_offsets, \
                        rubber_sheet_range, \
                        rubber_sheet_azimuth, \
                        fine_resample, \
                        split_range_spectrum, \
                        sub_band_resample, \
                        interferogram, \
                        'sub_band_interferogram', \
                        filter, \
                        unwrap, \
                        geocode \
                        ]", 
                        default=None)
    args = parser.parse_args()
    
    # Call function and put in Arguments from terminal
    run_pair(args.path, args.reference, args.secondary, args.workdir, args.step, args.steps, args.start)

# -------------- Steps information -----------------------------------------------------------------------------------------
"""
The step names are chosen from the following list:

['startup', 'preprocess', 'cropraw', 'formslc', 'cropslc']                                              ## NOT NEEDED SINCE SLC IMAGE FORMAT
['verifyDEM', 'topo', 'geo2rdr', 'coarse_resample', 'misregistration']
['refined_resample', 'dense_offsets', 'rubber_sheet_range', 'rubber_sheet_azimuth', 'fine_resample']
['split_range_spectrum', 'sub_band_resample', 'interferogram', 'sub_band_interferogram', 'filter']
['filter_low_band', 'filter_high_band', 'unwrap', 'unwrap_low_band', 'unwrap_high_band']
['ionosphere', 'geocode', 'geocodeoffsets', 'endup']

If --start is missing, then processing starts at the first step.
If --end is missing, then processing ends at the final step.
If --dostep is used, then only the named step is processed.

In order to use either --start or --dostep, it is necessary that a
previous run was done using one of the steps options to process at least
through the step immediately preceding the starting step of the current run.

When NOT specified, stripmapApp.py will go through these steps:
['startup', 'preprocess', 'formslc'],                                                                   ## SKIPPED SINCE SLC FORMAT
['verifyDEM', 'topo', 'geo2rdr', 'coarse_resample', 'misregistration'], 
['refined_resample'], 
['interferogram', 'filter'], 
['unwrap'], 
['geocode']
"""
# --------------------------------------------------------------------------------------------------------------------------