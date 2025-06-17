#!/usr/bin/env python3

from __future__ import annotations

from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# _______________________________________ paths and variable settings ___________________________________
BASE      = Path(__file__).resolve().parents[1]             # ~/InSAR/main
PROC_DIR  = BASE / "processing"

path = 464
ref = 20081104
sec = 20081220

pair_id = f"path{path}_{ref}_{sec}"
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

fig = plt.figure(figsize=(18, 16))

ax = fig.add_subplot(1,3,1)
ax.imshow(np.angle(igram_full_band), cmap='hsv')
ax.set_title("full-band")
ax.set_axis_off()

ax = fig.add_subplot(1,3,2)
ax.imshow(np.angle(igram_low_band), cmap='hsv')
ax.set_title("low-band")
ax.set_axis_off()

ax = fig.add_subplot(1,3,3)
ax.imshow(np.angle(igram_high_band), cmap='hsv')
ax.set_title("high-band")
ax.set_axis_off()

difference_full_low = igram_full_band*np.conjugate(igram_low_band)
difference_high_low = igram_high_band*np.conjugate(igram_low_band)
fig = plt.figure(figsize=(18, 16))

ax = fig.add_subplot(1,3,1)
cax = ax.imshow(np.angle(difference_full_low), cmap='jet', vmin = -1, vmax = 2)
ax.set_title("difference of full-band and low band interferograms")
ax.set_axis_off()

ax = fig.add_subplot(1,3,2)
cax = ax.imshow(np.angle(difference_high_low), cmap='jet', vmin = -1, vmax =2)
ax.set_title("difference of high-band and low band interferograms")
ax.set_axis_off()
cbar = fig.colorbar(cax, ticks=[-2.5,-2], orientation='horizontal')



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

fig = plt.figure(figsize=(18, 16))

ax = fig.add_subplot(1,3,1)
ax.imshow(rewrap(iono)*mask, cmap='hsv')
ax.set_title("dispersive (ionospheric phase)")
ax.set_axis_off()

ax = fig.add_subplot(1,3,2)
ax.imshow(rewrap(non_dispersive)*mask, cmap='hsv')
ax.set_title("non-dispersive")
ax.set_axis_off()

iono = None
non_dispersive = None

## stripmapApp does not correct the interferogram for ionospheric phase. 
## However, the corrected interferogram can be easily obtained by removing the estimated dispersive phase from the full-band interferogram.

ds = gdal.Open(f"{wdir}/interferogram/filt_topophase.unw", gdal.GA_ReadOnly)
igram = ds.GetRasterBand(2).ReadAsArray()
ds = None

ds = gdal.Open(f"{wdir}/ionosphere/dispersive.bil.filt", gdal.GA_ReadOnly)
iono = ds.GetRasterBand(1).ReadAsArray()
ds = None

igram_iono_corrected = igram - iono

fig = plt.figure(figsize=(18, 16))

ax = fig.add_subplot(1,3,1)
ax.imshow(rewrap(igram), cmap='hsv')
ax.set_title("before ionospheric phase correction")
ax.set_axis_off()

ax = fig.add_subplot(1,3,2)
ax.imshow(rewrap(igram_iono_corrected)*mask, cmap='hsv')
ax.set_title("after ionospheric phase correction")
ax.set_axis_off()

iono = None
igram = None
igram_iono_corrected = None

# reading the multi-looked wrapped interferogram
ds = gdal.Open(f"{wdir}/interferogram/filt_topophase.unw.geo", gdal.GA_ReadOnly)
igram = ds.GetRasterBand(2).ReadAsArray()
ds = None

ds = gdal.Open(f"{wdir}/ionosphere/dispersive.bil.unwCor.filt.geo", gdal.GA_ReadOnly)
iono = ds.GetRasterBand(1).ReadAsArray()
ds = None

ds = gdal.Open(f"{wdir}/ionosphere/mask.bil.geo", gdal.GA_ReadOnly)
mask = ds.GetRasterBand(1).ReadAsArray()
ds = None

igram_iono_corrected = (igram - iono)*mask

fig = plt.figure(figsize=(14,12))

ax = fig.add_subplot(1,2,1)

cax = ax.imshow(rewrap(igram), cmap = 'hsv')
ax.set_title("geocoded wrapped (before ionospheric phase correction)")
ax.set_axis_off()

ax = fig.add_subplot(1,2,2)
cax = ax.imshow(rewrap(igram_iono_corrected), cmap = 'hsv')
ax.set_title("geocoded wrapped (after ionospheric phase correction)")
ax.set_axis_off()

fig = plt.figure(figsize=(14,12))

ax = fig.add_subplot(1,2,1)

cax = ax.imshow((igram)*mask, cmap = 'hsv')
ax.set_title("geocoded unwrapped (before ionospheric phase correction)")
ax.set_axis_off()

ax = fig.add_subplot(1,2,2)
cax = ax.imshow(igram_iono_corrected, cmap = 'hsv')
ax.set_title("geocoded unwrapped (after ionospheric phase correction)")
ax.set_axis_off()

igram = None
iono = None
igram_iono_corrected = None
mask = None