#!/usr/bin/env python3
"""obtaining_sar_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ultra-lean helper for the InSAR thesis workflow.

* Launches your ASF bulk-download script via *subprocess* so that all ZIPs land
  in `main/data/raw/tmp_downloads/` (per the edits you already made).
* For every freshly downloaded ZIP it
    1. finds the matching metadata row in the Vertex CSV that lives in the
       same **tmp_downloads** directory,
    2. builds the final target path
       `main/data/raw/path<PATH_NUM>/<YYYYMMDD>/`,
    3. unzips the archive there,
    4. drops a `<GRANULE_NAME>.txt` file containing the full metadata row, and
    5. **deletes** the original ZIP once everything succeeded to save space.


Everything happens inside a single `run()` function to keep the script minimal.
"""
from __future__ import annotations

import csv
import subprocess
import zipfile
from pathlib import Path
import shutil


def run() -> None:
    """Kick off the download and tidy the results in one go."""
    print("🚀 script started, download of data an unzipping will be done")
    # ------------------------------------------------------------------ paths
    script_dir = Path(__file__).resolve().parent           # where *this* script lives (…/src/)
    project_root = script_dir.parent                       # step up to …/main/
    tmp_dir = project_root / "data" / "raw" / "tmp_downloads"

    download_script = script_dir / "help_download_two_path_150.py"  # adjust if name changes

    if not download_script.exists():
        raise FileNotFoundError(f"Cannot locate {download_script}")

    # ------------------------------------------------------ 1⃣ run downloader
    print("\n ⬇️  Starting ASF bulk download…\n")
    subprocess.check_call(["python", str(download_script)])

    # ------------------------------------------------------ 2⃣ load metadata
    csv_files = sorted(tmp_dir.glob("*.csv"))
    if not csv_files:
        print("⚠️  No CSV metadata file found in tmp_downloads; nothing to organise.")
        return
    meta_csv = csv_files[0]  # assume the first CSV is the right one
    print(f"\n📑 Using metadata file: {meta_csv.name}\n")

    with meta_csv.open(newline="") as f:
        rows_by_granule = {row["Granule Name"]: row for row in csv.DictReader(f)}

    # ------------------------------------------------------ 3⃣ process zips
    for zip_path in tmp_dir.glob("*.zip"):
        granule = zip_path.stem.split("-", 1)[0]  # e.g. ALPSRP268560510
        info = rows_by_granule.get(granule)
        if info is None:
            print(f"🚫  No metadata for {granule}, skipping …")
            continue

        # build target folder path<PATH_NUM>/YYYYMMDD
        path_num = info["Path Number"].zfill(3)  # keep leading zeros
        acq_date = info["Acquisition Date"][:10].replace("-", "")  # YYYYMMDD

        target_dir = project_root / "data" / "raw" / f"path{path_num}" / acq_date
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"📦  Unzipping {zip_path.name} ➜ {target_dir.relative_to(project_root)}")
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
