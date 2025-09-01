#!/usr/bin/env python3

from __future__ import annotations

from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

# _______________________________________ paths and variable settings ___________________________________
ABS_BASE        = Path(__file__).resolve().parents[5]
DATA_BASE       = ABS_BASE / "mnt" / "DATA2" / "bakke326l"
PROC_DIR        = DATA_BASE / "processing" / "interferograms"

path = 150
ref = 20071216
sec = 20080131
dem = "SRTM"
rangelooks = 10
azilooks = 16
alpha = 0.6

pair_id = f"path{path}_{ref}_{sec}_{dem}"
wdir    = PROC_DIR / pair_id
#________________________________________________________________________________________________________


ds = gdal.Open(f"{wdir}/interferogram/filt_topophase.flat", gdal.GA_ReadOnly)
igram_full_band = ds.GetRasterBand(1).ReadAsArray()
ds = None

ds = gdal.Open(f"{wdir}/interferogram/lowBand/filt_topophase.flat", gdal.GA_ReadOnly)
igram_low_band = ds.GetRasterBand(1).ReadAsArray()
ds = None

ds = gdal.Open(f"{wdir}/interferogram/highBand/filt_topophase.flat", gdal.GA_ReadOnly)
igram_high_band = ds.GetRasterBand(1).ReadAsArray()
ds = None

fig = plt.figure(figsize=(14, 12))

ax = fig.add_subplot(1,3,1)
ax.imshow(cv2.flip(np.angle(igram_full_band),0), cmap='turbo')
ax.set_title("full-band")
ax.set_axis_off()

ax = fig.add_subplot(1,3,2)
ax.imshow(cv2.flip(np.angle(igram_low_band), 0), cmap='turbo')
ax.set_title("low-band")
ax.set_axis_off()

ax = fig.add_subplot(1,3,3)
ax.imshow(cv2.flip(np.angle(igram_high_band), 0), cmap='turbo')
ax.set_title("high-band")
ax.set_axis_off()

difference_full_low = igram_full_band*np.conjugate(igram_low_band)
difference_high_low = igram_high_band*np.conjugate(igram_low_band)
fig = plt.figure(figsize=(18, 16))

ax = fig.add_subplot(1,2,1)
cax = ax.imshow(cv2.flip(np.angle(difference_full_low),0), cmap='jet', vmin = -1, vmax = 2)
ax.set_title("difference of full-band and low band interferograms")
ax.set_axis_off()

ax = fig.add_subplot(1,2,2)
cax = ax.imshow(cv2.flip(np.angle(difference_high_low), 0), cmap='jet', vmin = -1, vmax =2)
ax.set_title("difference of high-band and low band interferograms")
ax.set_axis_off()


igram_low_band = None
igram_full_band = None
igram_high_band = None
difference_full_low = None

# -------------------------------------------------------------------------------------------------------------------------------------------
## This step, uses the low-band and high-bamd unwrapped interferograms to estimate the dispersive and non-dispersive phase components. 
## The disperive phase is related to the ionosphere's TEC variation.
# -------------------------------------------------------------------------------------------------------------------------------------------
def rewrap(data):
    return data-np.round(data/2./np.pi)*2*np.pi
    
    
ds = gdal.Open(f"{wdir}/ionosphere/dispersive.bil.filt", gdal.GA_ReadOnly)
iono = ds.GetRasterBand(1).ReadAsArray()
ds = None

ds = gdal.Open(f"{wdir}/ionosphere/nondispersive.bil.filt", gdal.GA_ReadOnly)
non_dispersive = ds.GetRasterBand(1).ReadAsArray()
ds = None

ds = gdal.Open(f"{wdir}/ionosphere/mask.bil", gdal.GA_ReadOnly)
mask = ds.GetRasterBand(1).ReadAsArray()
ds = None

ds = gdal.Open(f"{wdir}/ionosphere/dispersive.bil.unwCor.filt", gdal.GA_ReadOnly)
iono_con = ds.GetRasterBand(1).ReadAsArray()
ds = None

fig = plt.figure(figsize=(14, 12))

ax = fig.add_subplot(1,3,1)
ax.imshow(cv2.flip(rewrap((iono)*mask), 0), cmap='turbo')
ax.set_title(f"dispersive (ionospheric phase) \n r{rangelooks} a{azilooks} f{alpha}" )
ax.set_axis_off()

ax = fig.add_subplot(1,3,2)
ax.imshow(cv2.flip(rewrap((non_dispersive)*mask), 0), cmap='turbo')
ax.set_title(f"non-dispersive \n r{rangelooks} a{azilooks} f{alpha}")
ax.set_axis_off()

ax = fig.add_subplot(1,3,3)
ax.imshow(cv2.flip(iono_con,0), cmap='turbo')
ax.set_title(f"iono consistency \n r{rangelooks} a{azilooks} f{alpha}")
ax.set_axis_off()

iono = None
non_dispersive = None
iono_con = None

fig = plt.figure(figsize=(14, 12))

ax = fig.add_subplot(1,1,1)
ax.imshow(cv2.flip(mask, 0), cmap='turbo', interpolation="nearest")
ax.set_title(f"mask \n r{rangelooks} a{azilooks} f{alpha}" )
ax.set_axis_off()

## stripmapApp does not correct the interferogram for ionospheric phase. 
## However, the corrected interferogram can be easily obtained by removing the estimated dispersive phase from the full-band interferogram.

ds = gdal.Open(f"{wdir}/interferogram/filt_topophase.unw", gdal.GA_ReadOnly)
igram = ds.GetRasterBand(2).ReadAsArray()
ds = None

ds = gdal.Open(f"{wdir}/ionosphere/dispersive.bil.unwCor.filt", gdal.GA_ReadOnly)
iono = ds.GetRasterBand(1).ReadAsArray()
ds = None

igram_iono_corrected = igram - iono

fig = plt.figure(figsize=(14, 12))

ax = fig.add_subplot(1,2,1)
ax.imshow(cv2.flip(rewrap(igram), 0), cmap='turbo')
ax.set_title(f"before ionospheric phase correction \n r{rangelooks} a{azilooks} f{alpha}")
ax.set_axis_off()

ax = fig.add_subplot(1,2,2)
ax.imshow(cv2.flip((rewrap(igram_iono_corrected)*mask), 0), cmap="turbo")
ax.set_title(f"after ionospheric phase correction \n r{rangelooks} a{azilooks} f{alpha}")
ax.set_axis_off()

iono = None
igram = None
igram_iono_corrected = None

# reading the multi-looked wrapped interferogram
# path to study area
cutline = "/home/bakke326l/InSAR/main/data/vector/study_area.geojson"

# reading the multi-looked wrapped interferogram (CLIPPED)
ds = gdal.Open(f"{wdir}/interferogram/filt_topophase.unw.geo", gdal.GA_ReadOnly)
ds = gdal.Warp("", ds, format="VRT", cutlineDSName=cutline, cropToCutline=True, dstNodata=float("nan"))
igram = ds.GetRasterBand(2).ReadAsArray()
ds = None

# dispersive map (CLIPPED)
ds = gdal.Open(f"{wdir}/ionosphere/dispersive.bil.unwCor.filt.geo", gdal.GA_ReadOnly)
ds = gdal.Warp("", ds, format="VRT", cutlineDSName=cutline, cropToCutline=True, dstNodata=float("nan"))
iono = ds.GetRasterBand(1).ReadAsArray()
ds = None

# mask (CLIPPED)
ds = gdal.Open(f"{wdir}/ionosphere/mask.bil.geo", gdal.GA_ReadOnly)
ds = gdal.Warp("", ds, format="VRT", cutlineDSName=cutline, cropToCutline=True, dstNodata=float("nan"))
mask = ds.GetRasterBand(1).ReadAsArray()
ds = None

igram_iono_corrected = (igram - iono) * mask

fig = plt.figure(figsize=(14,12))

ax = fig.add_subplot(1,2,1)
cax = ax.imshow(rewrap(igram), cmap='turbo')
ax.set_title(f"geocoded wrapped (before ionospheric phase correction) \n r{rangelooks} a{azilooks} f{alpha}")
ax.set_axis_off()

ax = fig.add_subplot(1,2,2)
cax = ax.imshow(rewrap(igram_iono_corrected), cmap='turbo')
ax.set_title(f"geocoded wrapped (after ionospheric phase correction)\n r{rangelooks} a{azilooks} f{alpha}")
ax.set_axis_off()

fig = plt.figure(figsize=(14,12))

ax = fig.add_subplot(1,2,1)
cax = ax.imshow((igram) * mask, cmap='hsv')
ax.set_title(f"geocoded unwrapped (before ionospheric phase correction)\n r{rangelooks} a{azilooks} f{alpha}")
ax.set_axis_off()

ax = fig.add_subplot(1,2,2)
cax = ax.imshow(igram_iono_corrected, cmap='turbo')
ax.set_title(f"geocoded unwrapped (after ionospheric phase correction)\n r{rangelooks} a{azilooks} f{alpha}")
ax.set_axis_off()

igram = None
iono = None
igram_iono_corrected = None