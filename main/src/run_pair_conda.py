#!/usr/bin/env python3
"""
run_pair_conda.py – Run one ALOS interferometric pair using the Conda-forge
ISCE2 install.  No external ISCE tree needed.
"""

from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

BASE      = Path(__file__).resolve().parents[1]        # ~/InSAR/main
DATA_DIR  = BASE / "data"
PROC_DIR  = BASE / "processing"

from isce_xml import write_stripmap_xml                 # your existing helper

def run_pair(track: int, ref: str, sec: str) -> None:
    pair_id  = f"path{track}_{ref}_{sec}"
    wdir     = PROC_DIR / pair_id
    wdir.mkdir(parents=True, exist_ok=True)

    xml_file = wdir / "stripmapApp.xml"
    if not xml_file.exists():
        write_stripmap_xml(track, ref, sec,
                           data_dir=DATA_DIR, out_xml=xml_file)
        print(f"📝  Wrote {xml_file}")

    # — locate Conda’s stripmapApp.py via ISCE_HOME —
    isce_home = Path(os.environ["ISCE_HOME"])
    stripmap_app = isce_home / "applications" / "stripmapApp.py"
    if not stripmap_app.exists():
        sys.exit(f"❌  Cannot find {stripmap_app}")

    cmd = [sys.executable, str(stripmap_app), str(xml_file)]
    print("🚀 ", " ".join(cmd))
    subprocess.check_call(cmd, cwd=wdir)

# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("track",     type=int)
    p.add_argument("reference")
    p.add_argument("secondary")
    a = p.parse_args()
    run_pair(a.track, a.reference, a.secondary)
