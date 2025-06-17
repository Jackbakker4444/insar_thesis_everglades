#!/usr/bin/env python3

from __future__ import annotations

import os
from osgeo import gdal, gdalconst
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

# _______________________________________ paths and variable settings ___________________________________
BASE      = Path(__file__).resolve().parents[1]             # ~/InSAR/main
PROC_DIR  = BASE / "processing"
GACOS_DIR = BASE / "data" / "aux" / "tropo"

path = 464
ref = 20081104
sec = 20081220

pair_id = f"path{path}_{ref}_{sec}"
wdir    = PROC_DIR / pair_id
os.chdir(wdir)
#________________________________________________________________________________________________________

def loadrsc(infile):
    '''A function to load the content of .rsc file and pass it back as a dictionary'''
    with open(infile + '.rsc') as f:
        headers = {}
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                headers[parts[0]] = parts[1]
            elif len(parts) > 2:
                # Skip lines like "DEM COP- DEM_GLO-90"
                continue
    
    # add the filename such it can be called when making envi header
    headers['FILENAME'] = infile
    # take the abs of the y-spacing as upper left corner is to be specified
    headers['Y_STEP'] = str(np.abs(float(headers['Y_STEP'])))
    return headers

def writehdr(filename, headers):
    '''A function that writes a .hdr file from a template and a dictionary describing the fields'''
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
    '''Wrapper to generate .hdr file from GACOS .rsc''' 
    print('Generating hdr file for: ' + inputfile + '...')
    filename, file_extension = os.path.splitext(inputfile)
    if file_extension in ['.hdr', '.rsc']:
        raise Exception("Give path to the ENVI file not the .hdr or .rsc file")
    headers = loadrsc(inputfile)
    writehdr(inputfile, headers)
    print('hdr for ' + inputfile + ' generated\n')

def file_transform(unwfile, apsfile, apsfile_out):
    '''convert the aps file into the same geo frame as the unw file'''
    # convert all to absolute paths
    apsfile = os.path.abspath(apsfile)
    apsfile_out = os.path.abspath(apsfile_out)

    # Source
    src = gdal.Open(apsfile, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    print("Working on " + apsfile )
    

    match_ds = gdal.Open(unwfile + '.vrt', gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    print("Getting target reference information")
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    dst = gdal.GetDriverByName('ENVI').Create(apsfile_out, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
    print("Reprojected:", apsfile_out)
    dst = None
    src = None

# 3.1 Create ENVI headers for both GACOS delay files
GACOS_rsc2hdr(str(GACOS_DIR / f"{ref}.ztd"))
GACOS_rsc2hdr(str(GACOS_DIR / f"{sec}.ztd"))

# 3.2 Interpolate delays to ISCE interferogram grid
unw_base = str(wdir / "interferogram" / "filt_topophase.unw.geo")
file_transform(unw_base, str(GACOS_DIR / f"{ref}.ztd"), f"{ref}.ztd.geo")
file_transform(unw_base, str(GACOS_DIR / f"{sec}.ztd"), f"{sec}.ztd.geo")

# 3.3 Convert zenith delay to slant delay

def zenith2slant(losfile, aps_zenith, aps_slant):
    # convert all to absolute paths
    aps_zenith = os.path.abspath(aps_zenith)
    aps_slant = os.path.abspath(aps_slant)
    losfile = os.path.abspath(losfile)

    # loading the zenith APS file
    ds = gdal.Open(aps_zenith, gdal.GA_ReadOnly)
    zenith = ds.GetRasterBand(1).ReadAsArray()
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    ds = None

    # loading the incidence angle file
    ds = gdal.Open(losfile, gdal.GA_ReadOnly)
    inc = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    inc = inc * np.pi / 180

    # scaling factor to convert pseudo-range [m] increase to phase delay [rad]
    scaling = -4 * np.pi / (23.62 / 100)
    
    # projecting the zenith into the slant
    slant = scaling * zenith / np.cos(inc)
    
    # making sure the no-date is propagated
    slant[zenith == 0] = 0
    slant[inc == 0] = 0

    # writing out the file   
    drv = gdal.GetDriverByName('ENVI').Create(aps_slant, slant.shape[1], slant.shape[0], 1, gdal.GDT_Float32)
    drv.SetGeoTransform(geotrans)
    drv.SetProjection(proj)
    drv.GetRasterBand(1).WriteArray(slant)
    drv = None

los_dir = wdir / "geometry" / "los.rdr.geo.vrt"
zenith2slant(str(los_dir), f"{ref}.ztd.geo", f"{ref}.aps.geo")
zenith2slant(str(los_dir), f"{sec}.ztd.geo", f"{sec}.aps.geo")

# 4. Differential tropospheric delay

def differential_delay(ref_aps, sec_aps, outname):
    # convert all to absolute paths
    ref_aps = os.path.abspath(ref_aps)
    sec_aps = os.path.abspath(sec_aps)
    outname = os.path.abspath(outname)

    # loading the master APS file
    ds = gdal.Open(ref_aps, gdal.GA_ReadOnly)
    ref = ds.GetRasterBand(1).ReadAsArray()
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    ds = None

    # loading the slave APS file
    ds = gdal.Open(sec_aps, gdal.GA_ReadOnly)
    sec = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    # computing the differential APS
    diffAPS = ref - sec

    # writing out the file 
    drv = gdal.GetDriverByName('ENVI').Create(outname, diffAPS.shape[1], diffAPS.shape[0], 1, gdal.GDT_Float32)
    drv.SetGeoTransform(geotrans)
    drv.SetProjection(proj)
    drv.GetRasterBand(1).WriteArray(diffAPS)
    drv = None

differential_delay(f"{ref}.aps.geo", f"{sec}.aps.geo", f"{ref}_{sec}.aps.geo")

# 5. Correct original interferogram

def IFG_correction(unw, aps, outname):
    # convert all to absolute paths
    unw = os.path.abspath(unw)
    aps = os.path.abspath(aps)
    outname = os.path.abspath(outname)

    # loading the UNW file
    ds = gdal.Open(unw, gdal.GA_ReadOnly)
    unwdata_phase = ds.GetRasterBand(2).ReadAsArray()
    unwdata_amplitude = ds.GetRasterBand(1).ReadAsArray()
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    ds = None

    # loading the APS file
    ds = gdal.Open(aps, gdal.GA_ReadOnly)
    apsdata = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    # Correcting the IFG
    unwdata_phase = unwdata_phase - apsdata
    # making sure the no-date is propagated
    unwdata_phase[unwdata_phase == 0] = 0
    unwdata_phase[apsdata == 0] = 0

    # writing out the file
    drv = gdal.GetDriverByName('ENVI').Create(outname, unwdata_phase.shape[1], unwdata_phase.shape[0], 2, gdal.GDT_Float32)
    drv.SetGeoTransform(geotrans)
    drv.SetProjection(proj)
    drv.GetRasterBand(1).WriteArray(unwdata_amplitude)
    drv.GetRasterBand(2).WriteArray(unwdata_phase)
    drv = None

igram_dir = wdir / "interferogram"
IFG_correction(str(igram_dir / "filt_topophase.unw.geo.vrt"), f"{ref}_{sec}.aps.geo", str(igram_dir / "filt_topophase_aps.unw.geo"))

# Plot functions

def plotdata2(file1, file2, band=1, title=("file1", "file2"), colormap='jet', datamin=None, datamax=None):
    ds1 = gdal.Open(file1, gdal.GA_ReadOnly)
    data1 = ds1.GetRasterBand(band).ReadAsArray()
    ds1 = None

    ds2 = gdal.Open(file2, gdal.GA_ReadOnly)
    data2 = ds2.GetRasterBand(band).ReadAsArray()
    ds2 = None

    fig = plt.figure(figsize=(14, 6))

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(data1, cmap=colormap, vmin=datamin, vmax=datamax)
    ax.set_title(title[0])
    ax.set_axis_off()

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(data2, cmap=colormap, vmin=datamin, vmax=datamax)
    ax.set_title(title[1])
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()

plotdata2(f"{sec}.aps.geo", f"{ref}.aps.geo", 1,
          title=[f"Secondary slant phase delay [rad]: {sec}", f"Reference slant phase delay [rad]: {ref}"],
          colormap='hsv')#, datamin=-725, datamax=-525)

plotdata2(str(igram_dir / "filt_topophase.unw.geo"), str(igram_dir / "filt_topophase_aps.unw.geo"), 2,
          title=["UNW before [rad]", "UNW after APS correction [rad]"],
          colormap='hsv')#, datamin=-50, datamax=50)

