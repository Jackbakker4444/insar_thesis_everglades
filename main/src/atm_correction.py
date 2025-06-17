#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio import float32
from osgeo import gdal, gdalconst



def do_iono_correction(work_dir: Path, out_dir: Path = None) -> Path:
    """
    Perform ionospheric phase correction on a full-band unwrapped interferogram.

    This function subtracts the estimated dispersive phase (caused by ionospheric 
    TEC variations) from the unwrapped interferometric phase. It also applies a 
    mask and writes both the corrected and rewrapped (wrapped back into [-π, π]) 
    interferograms as GeoTIFFs.

    Parameters
    ----------
    work_dir : Path
        The main processing directory containing ISCE2 output folders.
    out_dir : Path, optional
        Directory where the corrected interferograms will be saved.
        If None, defaults to `work_dir/interferogram`.

    Returns
    -------
    Path
        Path to the ionosphere-corrected interferogram GeoTIFF.

    References
    ----------
    - ISCE2 Ionosphere Tutorial (UNAVCO 2020):
      https://github.com/isce-framework/isce2-docs/blob/master/Notebooks/UNAVCO_2020/Atmosphere/Ionosphere/stripmapApp_ionosphere.ipynb
      Inspectable script with comments can be found at InSAR/main/src/show_ionospheric_correction.
    """
    igram_path = work_dir / "interferogram" / "filt_topophase_tropo.unw.geo"
    iono_path  = work_dir / "ionosphere"    / "dispersive.bil.unwCor.filt.geo"
    mask_path  = work_dir / "ionosphere"    / "mask.bil.geo"

    if out_dir is None:
        out_dir = work_dir / "interferogram"
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_corr_path  = out_dir / "filt_topophase_tropo_iono.unw.geo"
    dst_wrap_path  = out_dir / "filt_topophase_tropo_iono_wrapped.unw.geo"

    # Load full-band unwrapped interferogram (band 2)
    with rasterio.open(igram_path) as src_igram:
        profile = src_igram.profile
        profile.update(dtype=float32, count=1, nodata=np.nan, compress="deflate")
        igram_band = src_igram.read(2)

    # Load dispersive phase (band 1)
    with rasterio.open(iono_path) as src_iono:
        iono_band = src_iono.read(1)

    # Load mask
    with rasterio.open(mask_path) as src_mask:
        mask_band = src_mask.read(1)

    # Apply correction and wrapping
    igram_iono_corrected = (igram_band - iono_band) * mask_band
    igram_iono_corrected_wrap = igram_iono_corrected - np.round(igram_iono_corrected / (2.0 * np.pi)) * 2 * np.pi

    # Save corrected interferogram
    with rasterio.open(dst_corr_path, "w", **profile) as dst:
        dst.write(igram_iono_corrected.astype(float32), 1)

    # Save wrapped corrected interferogram
    with rasterio.open(dst_wrap_path, "w", **profile) as dst:
        dst.write(igram_iono_corrected_wrap.astype(float32), 1)

    print("✓ interferogram corrected for ionosphere:", dst_corr_path)
    print("✓ wrapped version also saved as:", dst_wrap_path)

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
                elif len(parts) > 2:
                    continue
        headers['FILENAME'] = infile
        headers['Y_STEP'] = str(np.abs(float(headers['Y_STEP'])))
        return headers

    def writehdr(filename, headers):
        print('Writing output HDR file...')
        enviHDRFile = open(filename + '.hdr', 'w')
        enviHDR = '''ENVI
                description = {{GACOS: {FILENAME} }}
                samples = {WIDTH}
                lines = {FILE_LENGTH}
                bands = 1
                header offset = 0
                file type = ENVI Standard
                data type = 4
                interleave = bsq
                sensor type = Unknown
                byte order = 0
                map info = {{Geographic Lat/Lon, 1, 1, {X_FIRST}, {Y_FIRST}, {X_STEP}, {Y_STEP}, WGS-84, units=Degrees}}
                coordinate system string = {{GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.017453292519943295]]}}'''.format(**headers)
        enviHDRFile.write(enviHDR)
        enviHDRFile.close()
        print('Output HDR file =', filename)

    def GACOS_rsc2hdr(inputfile):
        print('Generating hdr file for: ' + inputfile + '...')
        filename, file_extension = os.path.splitext(inputfile)
        if file_extension in ['.hdr', '.rsc']:
            raise Exception("Give path to the ENVI file not the .hdr or .rsc file")
        headers = loadrsc(inputfile)
        writehdr(inputfile, headers)
        print('hdr for ' + inputfile + ' generated\n')

    def file_transform(unwfile, apsfile, apsfile_out):
        apsfile = os.path.abspath(apsfile)
        apsfile_out = os.path.abspath(apsfile_out)
        src = gdal.Open(apsfile, gdalconst.GA_ReadOnly)
        src_proj = src.GetProjection()
        src_geotrans = src.GetGeoTransform()
        match_ds = gdal.Open(unwfile + '.vrt', gdalconst.GA_ReadOnly)
        match_proj = match_ds.GetProjection()
        match_geotrans = match_ds.GetGeoTransform()
        wide = match_ds.RasterXSize
        high = match_ds.RasterYSize
        dst = gdal.GetDriverByName('ENVI').Create(apsfile_out, wide, high, 1, gdalconst.GDT_Float32)
        dst.SetGeoTransform(match_geotrans)
        dst.SetProjection(match_proj)
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
        print("Reprojected:", apsfile_out)
        dst = None
        src = None

    def zenith2slant(losfile, aps_zenith, aps_slant):
        aps_zenith = os.path.abspath(aps_zenith)
        aps_slant = os.path.abspath(aps_slant)
        losfile = os.path.abspath(losfile)
        ds = gdal.Open(aps_zenith, gdal.GA_ReadOnly)
        zenith = ds.GetRasterBand(1).ReadAsArray()
        proj = ds.GetProjection()
        geotrans = ds.GetGeoTransform()
        ds = None
        ds = gdal.Open(losfile, gdal.GA_ReadOnly)
        inc = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        inc = inc * np.pi / 180
        scaling = -4 * np.pi / (23.62 / 100)
        slant = scaling * zenith / np.cos(inc)
        slant[zenith == 0] = 0
        slant[inc == 0] = 0
        drv = gdal.GetDriverByName('ENVI').Create(aps_slant, slant.shape[1], slant.shape[0], 1, gdal.GDT_Float32)
        drv.SetGeoTransform(geotrans)
        drv.SetProjection(proj)
        drv.GetRasterBand(1).WriteArray(slant)
        drv = None

    def differential_delay(ref_aps, sec_aps, outname):
        ref_aps = os.path.abspath(ref_aps)
        sec_aps = os.path.abspath(sec_aps)
        outname = os.path.abspath(outname)
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

    def IFG_correction(unw, aps, outname):
        unw = os.path.abspath(unw)
        aps = os.path.abspath(aps)
        outname = os.path.abspath(outname)
        ds = gdal.Open(unw, gdal.GA_ReadOnly)
        unwdata_phase = ds.GetRasterBand(2).ReadAsArray()
        unwdata_amplitude = ds.GetRasterBand(1).ReadAsArray()
        proj = ds.GetProjection()
        geotrans = ds.GetGeoTransform()
        ds = None
        ds = gdal.Open(aps, gdal.GA_ReadOnly)
        apsdata = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        unwdata_phase = unwdata_phase - apsdata
        unwdata_phase[unwdata_phase == 0] = 0
        unwdata_phase[apsdata == 0] = 0
        drv = gdal.GetDriverByName('ENVI').Create(outname, unwdata_phase.shape[1], unwdata_phase.shape[0], 2, gdal.GDT_Float32)
        drv.SetGeoTransform(geotrans)
        drv.SetProjection(proj)
        drv.GetRasterBand(1).WriteArray(unwdata_amplitude)
        drv.GetRasterBand(2).WriteArray(unwdata_phase)
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