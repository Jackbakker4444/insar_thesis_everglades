# ──────────────────────────────────────────────────────────────────────────────
# src/run_pair_xml.py
#
# High-level “fire-and-forget” driver that
#   1) builds  stripmapApp.xml  **per interferometric pair**
#   2) spawns ISCE2’s  stripmapApp.py  with that XML
#
# Call examples
#   $ python src/run_pair_xml.py --pair 465 20081006 20081121   # one pair
#   $ python src/run_pair_xml.py --all                         # every pair in CSV
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import logging
import shutil
import subprocess
import faulthandler
import subprocess
import logging
import shutil
import os
import sys
import signal

# function from isce_xml.py that writes the XML file
from isce_xml import write_stripmap_xml
os.environ["OMP_NUM_THREADS"] = "1" # FFTW crashes with >1 thread on big ALOS patches
os.environ["FFTW_NUM_THREADS"] = "1"

# ─────────────────────────────── constants ───────────────────────────────────
BASE      = Path(__file__).resolve().parents[1]
PAIR_CSV  = BASE / "data" / "pairs.csv"          # csv of all pairs to process
OUT_ROOT  = BASE / "processing"                  # each pair gets its own folder

# ─────────────────────────────── logging settings ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ──────────────────────────── core functions ────────────────────────────────
def run_pair(path: str, ref: str, sec: str) -> None:                        
    """
    🔧 *Run one reference/secondary pair.*

    Steps
    -----
    1.  Create a fresh working directory
    2.  Write `stripmapApp.xml` via *write_stripmap_xml()*
    3.  Create output subfolders
    4.  Invoke   stripmapApp.py --xml <file>   inside that directory
    """
    pair_name = f"path{path}_{ref}_{sec}"    # path465_20081006_20081121
    work_dir  = OUT_ROOT / pair_name          # --> processing/path465_20081006_20081121
    xml_file  = work_dir / "stripmapApp.xml" # --> processing/path465_.../stripmapApp.xml

    logging.info("🛠️   Processing %s", pair_name)

    # 1 fresh work-dir
    if work_dir.exists():
        logging.info("🗑️   Removing existing directory")
        shutil.rmtree(work_dir)              # Clean up from previous runs
    work_dir.mkdir(parents=True)
    logging.info("🗂️   Created working directory: %s", work_dir)

    # 2 write XML
    logging.info("📝   Writing stripmapApp.xml")
    write_stripmap_xml(xml_file, path, ref, sec)

    # 3 create expected OUTPUT folders for sensor.extractImage()
    for sub_dir in (ref, sec):
        out_dir = work_dir / sub_dir
        out_dir.mkdir(parents=True, exist_ok=True)  # ensure no error if already exists
        logging.debug(f"📂 Ensured output folder: {out_dir}")
        
    # 4 launch ISCE
    cmd = ["stripmapApp.py", "--xml", str(xml_file)]
    logging.info("🚀   Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=out_dir)
    
    result = subprocess.run(
        cmd, cwd=out_dir, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    (out_dir / "stripmap.log").write_text(result.stdout)

    if result.returncode != 0: 
        logging.error("Pair %s crashed ( exit=%d ) – log saved",
                    pair_name, result.returncode)
    
    logging.info("✅   Finished %s\n", pair_name)
    
    faulthandler.enable()                           # prints a Python‑side back‑trace if C code segfaults
    logfile = open(os.path.join(out_dir, "stripmap_stdout.log"), "w")
    try:
        result = subprocess.run(
            ["stripmapApp.py", "--xml", xml_file],
            cwd=out_dir,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            check=False,              # <- we want to keep going to inspect return‑code
        )
    finally:
        logfile.close()

    if result.returncode < 0:          # negative = killed by signal
        sig = signal.Signals(-result.returncode).name
        logging.error(f"stripmapApp died by {sig}")
        sys.exit(1)
    elif result.returncode > 0:
        logging.error(f"stripmapApp exited {result.returncode}")
        sys.exit(result.returncode)


# ────────────────────────────── CLI facade ───────────────────────────────────
def main() -> None:
    """Parse CLI args and process either *one* pair or *all* pairs in the CSV."""
    ap   = argparse.ArgumentParser(description="Run ISCE2 on ALOS pairs via XML")
    ag   = ap.add_mutually_exclusive_group(required=True)
    ag.add_argument("--pair", nargs=3, metavar=("PATH", "REF", "SEC"),
                    help="single pair: path# refDate secDate")
    ag.add_argument("--all",  action="store_true",
                    help="process every line in data/pairs.csv")
    args = ap.parse_args()

    # read CSV exactly once
    with PAIR_CSV.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if not r["path"].startswith("#")]

    if args.all:
        logging.info("♾️   Running ALL %d pairs from CSV", len(rows))
        for r in rows:
            run_pair(r["path"], r["ref_date"], r["sec_date"])
    else:
        logging.info("🔂   Running SINGLE pair")
        run_pair(*args.pair)


# ───────────────────────────────── entry point ───────────────────────────────
if __name__ == "__main__":
    main()
