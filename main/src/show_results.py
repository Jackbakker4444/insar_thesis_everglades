#!/usr/bin/env python3

from __future__ import annotations
import argparse, os, shutil, subprocess, sys
import datetime as dt
from pathlib import Path
from subprocess import check_call


from InSAR.main.src.show_fringes import create_fringe_tif
from atm_correction  import do_iono_correct

# _______________________________________ paths and variable settings ___________________________________
BASE      = Path(__file__).resolve().parents[1]             # ~/InSAR/main
PROC_DIR  = BASE / "processing"

path = 464
ref = 20081104
sec = 20081220

pair_id = f"path{path}_{ref}_{sec}"
wdir    = PROC_DIR / pair_id
igram_dir = wdir / "interferogram"
inspect_dir = wdir / "inspect"


# Creates a map showing 
do_iono_correct(work_dir=wdir, out_dir = igram_dir)
tif1_cmd = ["gdal_translate","-of","GTiff", f"{wdir}/interferogram/filt_topophase_ionoCorr.unw.geo", f"{inspect_dir}/filt_topophase_ionoCorr.unw.geo"+".tif"]
tif2_cmd = ["gdal_translate","-of","GTiff", f"{wdir}/interferogram/filt_topophase_ionoCorr_wrapped.unw.geo", f"{inspect_dir}/filt_topophase_ionoCorr_wrapped.unw.geo"+".tif"]

subprocess.check_call(tif1_cmd)
subprocess.check_call(tif2_cmd)


create_fringe_tif(work_dir = wdir, out_dir = inspect_dir)

