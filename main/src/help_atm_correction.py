#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio import float32
from osgeo import gdal, gdalconst



def do_iono_correction(work_dir: Path, out_dir: Path, input_unw: Path, output_suffix: str = "_iono" ) -> Path:
    """
    Perform ionospheric phase correction on a full-band unwrapped interferogram.

    This function subtracts the estimated dispersive phase (caused by ionospheric 
    TEC variations) from the unwrapped interferometric phase. It also applies a 
    mask and writes both the corrected and rewrapped (wrapped back into [-π, π]) 
    interferograms as GeoTIFFs.

    Parameters
    ----------
    work_dir : Path
        Pair directory (…/pathXXX_REF_SEC_SRTM or …_3DEP).
    out_dir : Path
        Output directory (usually work_dir / "interferogram").
    input_unw : Path
        Unwrapped input to correct (e.g., filt_topophase.unw.geo or
        filt_topophase_tropo.unw.geo).
    output_suffix : str
        Suffix in output filename: "_iono" (IONO-only) or "_tropo_iono" (TROPO+IONO).

    Returns
    -------
    Path to ionosphere-corrected unwrapped interferogram (.geo).

    References
    ----------
    - ISCE2 Ionosphere Tutorial (UNAVCO 2020):
      https://github.com/isce-framework/isce2-docs/blob/master/Notebooks/UNAVCO_2020/Atmosphere/Ionosphere/stripmapApp_ionosphere.ipynb
      Inspectable script with comments can be found at InSAR/main/src/show_ionospheric_correction.
    """
    iono_dir  = work_dir / "ionosphere"
    igram_dir = work_dir / "interferogram"
    out_dir = out_dir or igram_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs produced by your iono workflow
    iono_phase = iono_dir / "dispersive.bil.unwCor.filt.geo"
    iono_mask  = iono_dir / "mask.bil.geo"

    if not input_unw.exists():
        raise FileNotFoundError(f"input_unw not found: {input_unw}")
    if not iono_phase.exists():
        raise FileNotFoundError(f"iono phase not found: {iono_phase}")
    if not iono_mask.exists():
        raise FileNotFoundError(f"iono mask not found:  {iono_mask}")

    # Read input unwrapped (phase band is commonly band 2; fallback to band 1 if single-band)
    with rasterio.open(input_unw) as src_igram:
        profile = src_igram.profile
        profile.update(dtype=float32, count=1, nodata=np.nan, compress="deflate", predictor=3, zlevel=6)
        if src_igram.count == 1:
            phase_in = src_igram.read(1)
        else:
            phase_in = src_igram.read(2)  # band 2 = phase (ISCE unw)

    with rasterio.open(iono_phase) as src_iono:
        iono_band = src_iono.read(1)

    with rasterio.open(iono_mask) as src_mask:
        mask_band = src_mask.read(1)

    # Apply correction and rewrap
    corrected = (phase_in - iono_band) * mask_band
    wrapped   = corrected - np.round(corrected / (2.0 * np.pi)) * 2.0 * np.pi

    # Outputs (names matched to apply_atmos_corrections.py)
    dst_corr_path = out_dir / f"filt_topophase{output_suffix}.unw.geo"
    dst_wrap_path = out_dir / f"filt_topophase{output_suffix}_wrapped.unw.geo"

    with rasterio.open(dst_corr_path, "w", **profile) as dst:
        dst.write(corrected.astype(float32), 1)
    with rasterio.open(dst_wrap_path, "w", **profile) as dst:
        dst.write(wrapped.astype(float32), 1)

    print("✓ ionosphere corrected:", dst_corr_path)
    print("✓ wrapped version:",     dst_wrap_path)
    return dst_corr_path


# ─────────────────────────────────────────────────────────────────────────────
# Tropo GACOS Correction Script for ISCE2 Interferograms
#
# This script implements the workflow described in the ISCE2 tutorial by UNAVCO,
# available at:
# https://github.com/isce-framework/isce2-docs/blob/master/Notebooks/UNAVCO_2020/Atmosphere/Troposphere/Tropo.ipynb
#
# It performs:
#   1. GACOS .rsc to ENVI .hdr conversion
#   2. Interpolation of zenith delay to match interferogram grid
#   3. Projection to slant delay using incidence angle
#   4. Computation of differential atmospheric delay
#   5. Tropospheric correction of the unwrapped interferogram
#
# All output files are saved in the ISCE2 processing directory.
# ─────────────────────────────────────────────────────────────────────────────

def do_tropo_correction(wdir: Path, ref: int, sec: int, gacos_dir: Path, tropo_dir: Path):
    """
    Runs the full GACOS tropospheric correction workflow for a given interferometric pair.

    This includes:
      1. Converting GACOS .rsc metadata files into ENVI-compatible .hdr files
      2. Reprojecting GACOS zenith delay maps to match the ISCE2 interferogram grid
      3. Converting zenith delays to slant phase delays using LOS information
      4. Computing the differential phase delay between reference and secondary acquisitions
      5. Correcting the original unwrapped interferogram using this differential delay

    Parameters:
        wdir (Path): Path to the ISCE2 pair processing directory (e.g. /processing/pathXXX_YYYYMMDD_YYYYMMDD)
        ref (int): Reference acquisition date (e.g. 20081104)
        sec (int): Secondary acquisition date (e.g. 20081220)
        gacos_dir (Path): Path to folder containing the .ztd and .rsc files from GACOS
        tropo_dir (Path): Path to folder meant for intermediate troposheric files

    All corrected files and intermediate products will be saved inside `tropo_dir`.
    Inspectable scripts with comments can be found at InSAR/main/src/show_tropospheric_correction.
    """
    def loadrsc(infile):
        with open(infile + '.rsc') as f:
            headers = {}
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    headers[parts[0]] = parts[1]
        headers['FILENAME'] = infile
        headers['Y_STEP'] = str(abs(float(headers['Y_STEP'])))
        return headers

    def writehdr(filename, headers):
        with open(filename + '.hdr', 'w') as fo:
            fo.write(
                'ENVI\n'
                f'description = {{GACOS: {headers["FILENAME"]} }}\n'
                f'samples = {headers["WIDTH"]}\n'
                f'lines = {headers["FILE_LENGTH"]}\n'
                'bands = 1\n'
                'header offset = 0\n'
                'file type = ENVI Standard\n'
                'data type = 4\n'
                'interleave = bsq\n'
                'sensor type = Unknown\n'
                'byte order = 0\n'
                f'map info = {{Geographic Lat/Lon, 1, 1, {headers["X_FIRST"]}, {headers["Y_FIRST"]}, {headers["X_STEP"]}, {headers["Y_STEP"]}, WGS-84, units=Degrees}}\n'
                'coordinate system string = {GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.017453292519943295]]}\n'
            )

    def GACOS_rsc2hdr(inputfile):
        if os.path.splitext(inputfile)[1] in ('.hdr', '.rsc'):
            raise ValueError("Pass the ENVI binary (no .hdr/.rsc extension).")
        headers = loadrsc(inputfile)
        writehdr(inputfile, headers)

    def file_transform(match_raster, apsfile, apsfile_out):
        """Reproject apsfile to match match_raster grid/CRS."""
        apsfile     = os.path.abspath(apsfile)
        apsfile_out = os.path.abspath(apsfile_out)

        src = gdal.Open(apsfile, gdalconst.GA_ReadOnly)
        src_proj = src.GetProjection()
        src_geotrans = src.GetGeoTransform()

        match_ds = gdal.Open(match_raster, gdalconst.GA_ReadOnly)
        match_proj = match_ds.GetProjection()
        match_geotrans = match_ds.GetGeoTransform()
        wide = match_ds.RasterXSize
        high = match_ds.RasterYSize

        dst = gdal.GetDriverByName('ENVI').Create(apsfile_out, wide, high, 1, gdalconst.GDT_Float32)
        dst.SetGeoTransform(match_geotrans)
        dst.SetProjection(match_proj)
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)

        dst = None; src = None; match_ds = None

    def zenith2slant(losfile, aps_zenith, aps_slant):
        """Project zenith delay to slant phase delay using incidence angle."""
        WAVELENGTH = 0.2362  # meters (ALOS L-band)
        ds = gdal.Open(aps_zenith, gdal.GA_ReadOnly)
        zenith = ds.GetRasterBand(1).ReadAsArray()
        proj   = ds.GetProjection()
        geotrans = ds.GetGeoTransform()
        ds = None

        ds = gdal.Open(losfile, gdal.GA_ReadOnly)
        inc = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        inc = inc * np.pi / 180.0
        # Convert zenith delay (m) to phase (radians) in slant:
        # phase = -(4π / λ) * (zenith / cos(inc))
        scaling = -4.0 * np.pi / WAVELENGTH
        slant = scaling * zenith / np.cos(inc)
        slant[(zenith == 0) | (inc == 0)] = 0

        drv = gdal.GetDriverByName('ENVI').Create(aps_slant, slant.shape[1], slant.shape[0], 1, gdal.GDT_Float32)
        drv.SetGeoTransform(geotrans)
        drv.SetProjection(proj)
        drv.GetRasterBand(1).WriteArray(slant)
        drv = None

    def differential_delay(ref_aps, sec_aps, outname):
        ds = gdal.Open(ref_aps, gdal.GA_ReadOnly)
        ref = ds.GetRasterBand(1).ReadAsArray()
        proj = ds.GetProjection()
        geotrans = ds.GetGeoTransform()
        ds = None
        ds = gdal.Open(sec_aps, gdal.GA_ReadOnly)
        sec = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        diffAPS = ref - sec
        drv = gdal.GetDriverByName('ENVI').Create(outname, diffAPS.shape[1], diffAPS.shape[0], 1, gdal.GDT_Float32)
        drv.SetGeoTransform(geotrans)
        drv.SetProjection(proj)
        drv.GetRasterBand(1).WriteArray(diffAPS)
        drv = None

    def IFG_correction(unw_path, aps_path, outname):
        """
        Subtract differential APS (phase) from unwrapped IFG (band 2) and
        write 2-band ENVI (band1=amplitude, band2=phase).
        """
        ds = gdal.Open(unw_path, gdal.GA_ReadOnly)
        # ISCE unw: band1 amplitude, band2 phase
        amp  = ds.GetRasterBand(1).ReadAsArray()
        phase = ds.GetRasterBand(2).ReadAsArray()
        proj  = ds.GetProjection()
        geotrans = ds.GetGeoTransform()
        ds = None

        ds = gdal.Open(aps_path, gdal.GA_ReadOnly)
        aps = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        phase = phase - aps
        phase[(phase == 0) | (aps == 0)] = 0

        drv = gdal.GetDriverByName('ENVI').Create(outname, phase.shape[1], phase.shape[0], 2, gdal.GDT_Float32)
        drv.SetGeoTransform(geotrans)
        drv.SetProjection(proj)
        drv.GetRasterBand(1).WriteArray(amp)
        drv.GetRasterBand(2).WriteArray(phase)
        drv = None

    GACOS_rsc2hdr(str(gacos_dir / f"{ref}.ztd"))
    GACOS_rsc2hdr(str(gacos_dir / f"{sec}.ztd"))
    base_unw = wdir / "interferogram" / "filt_topophase.unw.geo"
    file_transform(str(base_unw), str(gacos_dir / f"{ref}.ztd"), str(tropo_dir / f"{ref}.ztd.geo"))
    file_transform(str(base_unw), str(gacos_dir / f"{sec}.ztd"), str(tropo_dir / f"{sec}.ztd.geo"))
    los_file = wdir / "geometry" / "los.rdr.geo.vrt"
    zenith2slant(str(los_file), str(tropo_dir / f"{ref}.ztd.geo"), str(tropo_dir / f"{ref}.aps.geo"))
    zenith2slant(str(los_file), str(tropo_dir / f"{sec}.ztd.geo"), str(tropo_dir / f"{sec}.aps.geo"))
    differential_delay(str(tropo_dir / f"{ref}.aps.geo"), str(tropo_dir / f"{sec}.aps.geo"), str(tropo_dir / f"{ref}_{sec}.aps.geo"))
    IFG_correction(str(base_unw) + ".vrt", str(tropo_dir / f"{ref}_{sec}.aps.geo"), str(wdir / "interferogram" / "filt_topophase_tropo.unw.geo"))