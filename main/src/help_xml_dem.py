#!/usr/bin/env python3
"""
=============================================================
dsm_to_isce.py  –  Convert a GeoTIFF DSM/DEM into ISCE2 format
=============================================================

• Reprojects (if needed) to geographic WGS‑84 (EPSG:4326)
• Writes binary DEM + GDAL VRT using the GDAL *ISCE* driver
• Generates ISCE‑compatible XML metadata via gdal2isce_xml

Requires:
    - GDAL >= 3.4 (conda‑forge gdal)
    - isce2 installed in the active environment

Usage
-----
```bash
python dsm_to_isce.py  \
       --input   /path/to/my_lidar_utm.tif \
       --output  /path/to/dem/my_area.dem.wgs84
```

Optional flags:
    --keep_egm      keep geoid heights (skip EGM→WGS84 conversion)
    --overwrite     overwrite existing output files

After running, you will have:
    my_area.dem.wgs84          (binary float32, little‑endian)
    my_area.dem.wgs84.vrt      (GDAL VRT)
    my_area.dem.wgs84.xml      (ISCE metadata)

You can drop the *.wgs84* basename into stripmapApp / stackStripMap.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from osgeo import gdal
from applications.gdal2isce_xml import gdal2isce_xml

gdal.UseExceptions()


##################### SOURCE #################################################
# Link to source: https://github.com/isce-framework/isce2/discussions/347    #
# Inspected: 26/05/2025                                                      #
##############################################################################

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def reproject_if_needed(src_tif: Path, tmp_dir: Path, keep_egm: bool) -> Path:
    """Ensure input is geographic WGS-84 (EPSG:4326) + ellipsoidal heights."""

    info = gdal.Info(str(src_tif), format="json")
    srs  = info.get("coordinateSystem")

    if srs and "WGS 84" in srs.get("wkt", "") and srs.get("geographic", False):
        print("✔  Input already in geographic WGS-84 - skipping reprojection") 
        return src_tif

    dst = tmp_dir / (src_tif.stem + "_wgs84.tif")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gdalwarp",
        "-t_srs", "EPSG:4326",
        "-r", "cubic",
        "-overwrite",
        str(src_tif),
        str(dst),
    ]
    if not keep_egm:
        cmd[1:1] = ["-s_srs", "EPSG:4326"]  # triggers geoid→ellipsoid shift via PROJ

    print("⚙️  Reprojecting to geographic WGS-84 …")
    subprocess.check_call(cmd)
    return dst


def gdal_translate_isce(src_tif: Path, out_bin: Path, overwrite: bool) -> None:
    if out_bin.exists() and not overwrite:
        print(f"✔  {out_bin.name} exists - skipping gdal_translate")
        return
    print(f"⚙️  Converting to ISCE binary: {out_bin.name}")
    drv_opts = ["-of", "ISCE", 
                "-ot", "Float32", 
                "-co", "FORMAT=FLOAT",
                "-co", f"ABSOLUTE_PATH=YES"]
    subprocess.check_call(["gdal_translate", *drv_opts, str(src_tif), str(out_bin)])


def build_dem_xml(out_bin: Path, overwrite: bool) -> None:
    xml_file = out_bin.with_suffix(".xml")
    if xml_file.exists() and not overwrite:
        print(f"✔  {xml_file.name} exists - skipping gdal2isce_xml")
        return
    
    print("⚙️  Generating ISCE XML …")
    gdal2isce_xml(str(out_bin))


def write_dem_report(out_bin: Path, keep_egm: bool) -> None:
    """
    Create <out_bin>.txt with CRS, datum, and basic stats.
    """
    report = out_bin.with_suffix(".txt")
    ds = gdal.Open(str(out_bin.with_suffix(".wgs84.vrt")))          # read via VRT
    band = ds.GetRasterBand(1)
    stats = band.GetStatistics(True, True)   # min, max, mean, std
    nodata = band.GetNoDataValue()
    gt     = ds.GetGeoTransform()

    with open(report, "w") as fh:
        fh.write(f"DEM report for {out_bin.name}\n")
        fh.write("-" * 60 + "\n")
        fh.write(f"Size (pixels):  {ds.RasterXSize}  x  {ds.RasterYSize}\n")
        fh.write(f"Pixel posting:  {gt[1]:.9f}  deg  x  {abs(gt[5]):.9f} deg\n")
        fh.write(f"BBox (Lon/Lat): {gt[0]:.6f}, {gt[3] + gt[5]*ds.RasterYSize:.6f} "
                 f"  to  {gt[0] + gt[1]*ds.RasterXSize:.6f}, {gt[3]:.6f}\n\n")
        fh.write("Coordinate system  :  EPSG:4326  (geographic WGS-84)\n")
        fh.write("Vertical reference :  " +
                 ("EGM96/2008 geoid heights" if keep_egm 
                  else "WGS-84 ellipsoid heights")
                 + "\n\n")
        fh.write(f"NoData value      :  {nodata}\n")
        fh.write(f"Minimum           :  {stats[0]:.3f} m\n")
        fh.write(f"Maximum           :  {stats[1]:.3f} m\n")
        fh.write(f"Mean              :  {stats[2]:.3f} m\n")
        fh.write(f"Std-dev           :  {stats[3]:.3f} m\n")
    print(f"📝  Wrote DEM report  →  {report.name}")


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert GeoTIFF DSM/DEM to ISCE2 format")
    ap.add_argument("--input", required=True, type=Path, help="Input GeoTIFF DSM/DEM")
    ap.add_argument("--output", required=True, type=Path, help="Output basename (*.dem.wgs84)")
    ap.add_argument("--tmp-dir", default="./tmp_reproj", type=Path, help="Temp directory for reprojection")
    ap.add_argument("--keep-egm", action="store_true", help="Keep geoid heights (skip ellipsoid conversion)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    src_tif  = args.input.expanduser().resolve()
    out_bin  = args.output.expanduser().resolve()

    if not src_tif.exists():
        sys.exit(f"Input DEM not found: {src_tif}")

    # 1. Reproject if needed (writes tmp file)
    tmp_dir = args.tmp_dir.expanduser().resolve()
    reproj_tif = reproject_if_needed(src_tif, tmp_dir, args.keep_egm)

    # 2. gdal_translate to ISCE binary (+ .vrt)
    gdal_translate_isce(reproj_tif, out_bin, args.overwrite)

    # 3. Generate ISCE XML side‑car
    build_dem_xml(out_bin, args.overwrite)

    # 4. Clean temp dir (optional)
    if reproj_tif.parent == tmp_dir:
        shutil.rmtree(tmp_dir)

    print("🎉  All done ->", out_bin)


if __name__ == "__main__":
    main()
