#!/usr/bin/env python3
"""
convert_sar_lu_to_db.py
=======================

Convert a SAR GeoTIFF from linear units (LU) to decibels (dB).

By default, it assumes the raster values represent **power** and uses:

    dB = 10 * log10(LU)

If your file stores **amplitude** instead of power, you may want:

    dB = 20 * log10(LU)

Usage
-----
# Minimal (uses default input path and auto output name):
python convert_sar_lu_to_db.py

# Explicit input and output:
python convert_sar_lu_to_db.py \
    /home/bakke326l/InSAR/main/data/aux/raster/SAR_baselayer.tif \
    /home/bakke326l/InSAR/main/data/aux/raster/SAR_baselayer_db.tif
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio


def convert_lu_to_db(in_path: Path, out_path: Path) -> None:
    """Convert all bands in a SAR GeoTIFF from linear units to dB."""
    in_path = Path(in_path)
    out_path = Path(out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {in_path}")

    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        nodata = src.nodata

        # We write float32 dB values
        profile.update(
            dtype="float32",
            nodata=-9999.0 if nodata is None else float(nodata),
        )

        # Prepare output directory
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                # Read band as a masked array (respects nodata)
                band = src.read(b, masked=True)

                # Mask non-positive values, since log10 is undefined there
                # (they’ll become nodata in the output)
                band = np.ma.masked_where((band <= 0) | band.mask, band)

                # Convert LU -> dB assuming power units:
                # If your data is amplitude: change 10 to 20 below.
                band_db = 20.0 * np.ma.log10(band)

                # Fill masked cells with nodata
                band_db_filled = band_db.filled(profile["nodata"]).astype("float32")

                dst.write(band_db_filled, b)

    print(f"Written dB-converted raster to: {out_path}")


def main():
    default_in = "/home/bakke326l/InSAR/main/data/aux/raster/SAR_baselayer.tif"
    default_out = "/home/bakke326l/InSAR/main/data/aux/raster/SAR_baselayer_db.tif"

    parser = argparse.ArgumentParser(
        description="Convert a SAR GeoTIFF from linear units (LU) to decibels (dB)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=default_in,
        help=f"Input SAR GeoTIFF in linear units (default: {default_in})",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=default_out,
        help=f"Output SAR GeoTIFF in dB (default: {default_out})",
    )

    args = parser.parse_args()
    convert_lu_to_db(args.input, args.output)


if __name__ == "__main__":
    main()
