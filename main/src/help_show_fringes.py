#!/usr/bin/env python
from osgeo import gdal
import numpy as np
from pathlib import Path


# SOURCE: https://github.com/isce-framework/isce2/discussions/550 (2022)
# Viewed: 11/06/2025 

# BASE      = Path(__file__).resolve().parents[1]             # ~/InSAR/main
# PROC_DIR  = BASE / "processing"
# #PAIR_DIR  = PROC_DIR / "path150_20080131_20080317"
# PAIR_DIR  = PROC_DIR / "path464_20081104_20081220"
# INTER_DIR = PAIR_DIR / "interferogram"


# # Read the file and extract attributes 
# input_path = str(INTER_DIR / "filt_topophase.flat.geo")

# ds = gdal.Open(input_path, gdal.GA_ReadOnly)
# transform = ds.GetGeoTransform()
# proj = ds.GetProjection()

# # Convert to numpy and extract the complex (interferogram) part
# combined_prod = ds.GetRasterBand(1).ReadAsArray()
# ifg = np.angle(combined_prod)

# # Export the interferogram in GTiff
# driver = gdal.GetDriverByName("GTiff")
# driver.Register()
# output = driver.Create(str(INTER_DIR / "FRINGES.tif"), xsize = ifg.shape[1], ysize = ifg.shape[0], bands=1, eType = gdal.GDT_Float64)
# output.SetGeoTransform(transform)
# output.SetProjection(proj)
# outband = output.GetRasterBand(1)
# outband.WriteArray(ifg)
# outband.SetNoDataValue(np.nan)
# outband.FlushCache()
# outband = output = None


def create_fringe_tif(work_dir : Path, out_dir : Path, ref : int, sec : int, range_looks : int, azi_looks :int, alpha : float) -> None:
    """
    Read filt_topophase.flat.geo (complex), convert to wrapped phase,
    and save a single–band GTiff called FRINGES.tif.
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