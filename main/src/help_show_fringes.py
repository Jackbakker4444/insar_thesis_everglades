#!/usr/bin/env python
"""
help_show_fringes.py
====================

Create a wrapped-phase “fringe” GeoTIFF from an ISCE2 complex interferogram.

What it does
------------
Reads the *geocoded* complex interferogram:

    <work_dir>/interferogram/filt_topophase.flat.geo

computes its wrapped phase (radians in [−π, π)) via ``np.angle`` and writes a
single-band GeoTIFF named:

    FRINGES_<REF>_<SEC>_ra<RANGE>_az<AZ>_<ALPHA>.tif

to the requested ``out_dir``. The filename mirrors the rest of the pipeline so
downstream steps can pick it up for quicklook PNGs.

Inputs & assumptions
--------------------
- The complex interferogram is stored as ENVI ``*.geo`` (ISCE2 default) with
  a complex numeric type in band 1.
- The dataset carries a valid GeoTransform and projection; both are copied to
  the output.
- If you keep a ``.geo.vrt`` next to the ENVI pair, this helper expects the
  real ENVI binary (``*.geo``) path. (If you prefer, you can point ``work_dir``
  at a pair whose ``*.geo.vrt`` is the only available file; GDAL will still
  open it transparently.)

Outputs
-------
- A single-band GeoTIFF with wrapped phase (float64), nodata set to NaN, and
  the same spatial metadata as the source:
    <out_dir>/FRINGES_<REF>_<SEC>_ra<RANGE>_az<AZ>_<ALPHA>.tif

Notes
-----
Source inspiration/discussion:
https://github.com/isce-framework/isce2/discussions/550  (viewed 2025-11-06)
"""

from osgeo import gdal
import numpy as np
from pathlib import Path


def create_fringe_tif(work_dir : Path, out_dir : Path, ref : int, sec : int, range_looks : int, azi_looks :int, alpha : float) -> None:
    """
    Build a wrapped-phase “FRINGES” GeoTIFF from the complex interferogram.

    Parameters
    ----------
    work_dir : Path
        Pair directory (…/pathXXX_YYYYMMDD_YYYYMMDD_{SRTM|3DEP}), which must
        contain ``interferogram/filt_topophase.flat.geo``.
    out_dir : Path
        Destination folder; will be created if missing.
    ref, sec : int
        Reference and secondary acquisition dates (YYYYMMDD). Used in the
        output filename only.
    range_looks : int
        Range multilooking factor (for filename provenance).
    azi_looks : int
        Azimuth multilooking factor (for filename provenance).
    alpha : float
        Goldstein filter alpha (for filename provenance).

    Behavior
    --------
    - Opens ``filt_topophase.flat.geo`` (band 1, complex), computes
      ``np.angle`` to obtain wrapped phase in radians.
    - Writes a single-band GeoTIFF with nodata set to NaN, copying the source
      geotransform and CRS.

    Returns
    -------
    None
        The function writes the output TIFF to disk and does not return a value.

    Raises
    ------
    RuntimeError
        If the source file cannot be opened or the output cannot be created.

    Example
    -------
    >>> create_fringe_tif(
    ...     work_dir=Path("/mnt/DATA2/.../path150_20071216_20080131_SRTM"),
    ...     out_dir=Path("/home/user/InSAR/main/processing/inspect"),
    ...     ref=20071216, sec=20080131, range_looks=10, azi_looks=16, alpha=0.6
    ... )
    """
    src_path = work_dir / "interferogram" / "filt_topophase.flat.geo"
    dst_path  = out_dir   / f"FRINGES_{ref}_{sec}_ra{range_looks}_az{azi_looks}_{alpha}.tif"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    transform = ds.GetGeoTransform()
    proj = ds.GetProjection()

    # Convert to numpy and extract the complex (interferogram) part
    combined_prod = ds.GetRasterBand(1).ReadAsArray()
    ifg = np.angle(combined_prod)

    # Export the interferogram in GTiff
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    output = driver.Create(str(dst_path), xsize = ifg.shape[1], ysize = ifg.shape[0], bands=1, eType = gdal.GDT_Float64)
    output.SetGeoTransform(transform)
    output.SetProjection(proj)
    outband = output.GetRasterBand(1)
    outband.WriteArray(ifg)
    outband.SetNoDataValue(np.nan)
    outband.FlushCache()
    outband = output = None