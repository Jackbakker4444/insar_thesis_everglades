#!/usr/bin/env python3
"""
1_obtaining_sar_data.py — Download & organize ASF SAR archives for the InSAR thesis

Purpose
-------
Automate two steps:
1) Invoke your ASF/Vertex bulk-downloader so newly fetched ZIPs land in the raw
   staging area, and
2) Unpack and file each scene into its final location while saving the metadata
   and deleting the original ZIPs.

Needed data (inputs & assumptions)
----------------------------------
- Temporary download folder with new ASF ZIP archives and a Vertex CSV:
  <RAW_DIR>/tmp_downloads/
  * The CSV must include the columns: Granule Name, Path Number, Acquisition Date (YYYY-MM-DD...).
  * Each ZIP filename must start with the Granule Name (e.g., ALPSRP268560510-...zip).
- A Python downloader script placed next to this file:
  help_download_all_path_150.py  (rename here if your downloader name changes).

Dependencies
------------
- Python standard library: csv, subprocess, zipfile, pathlib.
- External but called by this script: your ASF/Vertex downloader that writes into
  <RAW_DIR>/tmp_downloads/.

Outputs & directories
---------------------
- Unpacked scenes organized by path and date:
  <RAW_DIR>/path<PATH_NUM>/<YYYYMMDD>/
  e.g., /mnt/DATA2/bakke326l/raw/path150/20071231/
- One sidecar metadata text per scene:
  <RAW_DIR>/path<PATH_NUM>/<YYYYMMDD>/<GRANULE_NAME>.txt  (full CSV row).
- Original ZIPs in <RAW_DIR>/tmp_downloads/ are deleted after successful extraction.

How it works
------------
1) Run the downloader; expect ZIPs + a CSV in tmp_downloads/.
2) Read the first *.csv in tmp_downloads/ and index rows by Granule Name.
3) For each *.zip:
   - infer GRANULE_NAME,
   - look up its CSV row,
   - build path<PATH_NUM>/<YYYYMMDD>,
   - unzip there,
   - write <GRANULE_NAME>.txt with the metadata row,
   - delete the ZIP.

How to run
----------
From a shell (Python 3.x):
    python 1_obtaining_sar_data.py

Notes
-----
- If the downloader script is missing, a FileNotFoundError is raised.
- If the downloader exits non-zero, a subprocess.CalledProcessError is raised.
- ZIPs without matching metadata rows are skipped and left in tmp_downloads/.
"""
from __future__ import annotations

import csv
import subprocess
import zipfile
from pathlib import Path

def run() -> None:
    """
    Execute the end-to-end fetch -> organize workflow.

    Steps
    -----
    1) Call the local downloader (help_download_all_path_150.py) so new ASF ZIPs and a
    Vertex CSV appear in <RAW_DIR>/tmp_downloads/.
    2) Load the first *.csv in tmp_downloads/ and map rows by Granule Name.
    3) For each ZIP:
    - determine target folder <RAW_DIR>/path<PATH_NUM>/<YYYYMMDD>/,
    - extract the ZIP there,
    - write <GRANULE_NAME>.txt with the full CSV row,
    - delete the ZIP on success.

    Side effects
    ------------
    - Reads:  <RAW_DIR>/tmp_downloads/  (ZIPs, CSV)
    - Writes: <RAW_DIR>/path<PATH_NUM>/<YYYYMMDD>/  (unpacked scene, metadata)
    - Deletes: original ZIPs after successful extraction

    Raises
    ------
    FileNotFoundError
        If the downloader script cannot be found next to this file.
    subprocess.CalledProcessError
        If the downloader script exits with a non-zero status.

    Returns
    -------
    None
    """
    print("🚀 script started, download of data an unzipping will be done")
    # ------------------------------------------------------------------ paths
    # Script paths
    script_dir = Path(__file__).resolve().parent           # where *this* script lives (…/src/)
    download_script = script_dir / "help_download_all_path_150.py"  # adjust if name changes
    
    # Data paths
    ABS_BASE        = Path(__file__).resolve().parents[5]
    DATA_BASE       = ABS_BASE / "mnt" / "DATA2" / "bakke326l"
    RAW_DIR         = DATA_BASE / "raw"
    TMP_DIR         = RAW_DIR / "tmp_downloads"

    if not download_script.exists():
        raise FileNotFoundError(f"Cannot locate {download_script}")

    # ------------------------------------------------------ 1⃣ run downloader
    print("\n ⬇️  Starting ASF bulk download…\n")
    
    # Call entire python script 
    subprocess.check_call(["python", str(download_script)])

    # ------------------------------------------------------ 2⃣ load metadata
    csv_files = sorted(TMP_DIR.glob("*.csv"))
    if not csv_files:
        print("⚠️  No CSV metadata file found in tmp_downloads; nothing to organise.")
        return
    meta_csv = csv_files[0]  # assume the first CSV is the right one
    print(f"\n📑 Using metadata file: {meta_csv.name}\n")

    with meta_csv.open(newline="") as f:
        rows_by_granule = {row["Granule Name"]: row for row in csv.DictReader(f)}

    # ------------------------------------------------------ 3⃣ process zips
    for zip_path in TMP_DIR.glob("*.zip"):
        granule = zip_path.stem.split("-", 1)[0]  # e.g. ALPSRP268560510
        info = rows_by_granule.get(granule)
        if info is None:
            print(f"🚫  No metadata for {granule}, skipping …")
            continue

        # build target folder path<PATH_NUM>/YYYYMMDD
        path_num = info["Path Number"].zfill(3)  # keep leading zeros
        acq_date = info["Acquisition Date"][:10].replace("-", "")  # YYYYMMDD

        target_dir = RAW_DIR / f"path{path_num}" / acq_date
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"📦  Unzipping {zip_path.name} ➜ {target_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)

        # write simple txt with metadata
        info_txt = target_dir / f"{granule}.txt"
        with info_txt.open("w", encoding="utf-8") as fh:
            for k, v in info.items():
                fh.write(f"{k}: {v}\n")
                
        # delete the original ZIP after successful extraction & metadata write
        zip_path.unlink()

    print("\n✅  All done.")


if __name__ == "__main__":
    run()
