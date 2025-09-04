InSAR ALOS Stripmap Processing Pipeline — README

This repository contains a fully scripted pipeline for building ALOS stripmap interferograms with ISCE2, applying ionospheric & tropospheric corrections, clipping per-area products, assessing accuracy against EDEN water‐level gauges, and generating publication-ready figures.

It’s organized as a set of Python entrypoints (2_* through 6_*) with a few helper modules. Everything is file-system driven and designed to be robust for large batch runs.

Table of Contents

Overview

Prerequisites

Directory Layout & Key Paths

Data & Inputs

Quick Start

Pipeline Stages

2_run_pair_conda.py — Build interferograms

3_do_corrections.py — Apply IONO/TROPO & export vertical displacement

4_organize_areas.py — Per-area clipping & gauge tables

5_accuracy_assessment.py — Shared-split metrics + per-pair TIFFs

6_visualization.py — Figures (accuracy vs density, period plots)

Helper Modules

Outputs & Naming Conventions

Configuration Notes & Tips

Troubleshooting

Reproducibility

License / Acknowledgements

Overview

Goal: From raw ALOS CEOS data and DEMs, produce corrected interferograms, vertical displacement (cm) rasters, per-area subsets, metrics vs. gauge records, and figures.

Highlights:

Dual DEM support (SRTM 30 m, 3DEP 10 m).

Two DEM runs per pair (SRTM and 3DEP).

Ionospheric correction (split-spectrum) and GACOS troposphere correction.

Vertical displacement exports in centimeters, swath-masked, LOS-median zeroed before vertical projection.

Per-area clipping with coverage accounting + gauge tables.

Accuracy analysis with shared calibration/validation split across all DEM/CORR variants and replicates.

Figures with clipped 5–95% uncertainty bands and per-pair map panels.

Prerequisites

Python 3.9+

ISCE2 installed and on PATH (ISCE_HOME set). The pipeline calls stripmapApp.py.

snaphu available (used by ISCE2 for unwrapping).

GDAL (>= 3.4, with command-line tools in PATH: gdal_translate, gdalwarp).

conda-forge stack recommended.

Python packages

Install via conda/mamba (recommended):

mamba install -c conda-forge gdal rasterio numpy scipy pandas matplotlib pyproj shapely fiona geopandas

Directory Layout & Key Paths

These defaults are hard-coded in the scripts; adjust if your system differs.

~/InSAR/main/                      # BASE
├─ data/
│  ├─ aux/
│  │  ├─ dem/
│  │  │  ├─ srtm_30m.tif
│  │  │  └─ 3dep_10m.tif
│  │  └─ tropo/                    # GACOS .ztd + .rsc files
│  ├─ vector/
│  │  ├─ gauge_locations.geojson
│  │  └─ water_areas.geojson
│  └─ aux/gauges/
│     ├─ eden_water_levels.csv
│     ├─ eden_ground_elevation.txt
│     ├─ eden_water_elevation.csv  # (auto-made: above-ground cm, same schema)
│     └─ acquisition_dates.csv     # date column 'YYYYMMDD'
├─ processing/
│  ├─ inspect/                     # global quicklook PNG copies
│  └─ areas/
│     ├─ <AREA>/
│     │  ├─ interferograms/        # per-area clipped products
│     │  ├─ water_gauges/          # per-area EDEN tables
│     │  └─ results/               # metrics + figures
│     └─ _reports/coverage_report.csv
└─ src/ (your scripts)


Raw ALOS and heavy outputs live on a fast disk (adjust to taste):

/mnt/DATA2/bakke326l/
├─ raw/                               # CEOS files: path<PATH>/<YYYYMMDD>/
└─ processing/
   └─ interferograms/
      └─ path<PATH>_<REF>_<SEC>_<DEM>/
         ├─ stripmapApp.xml
         ├─ interferogram/
         ├─ geometry/
         ├─ ionosphere/
         ├─ troposphere/
         └─ inspect/

Data & Inputs

pairs.csv (~/InSAR/main/data/pairs.csv) with columns:

path,reference,secondary
150,20071216,20080131
...


DEMs:

~/InSAR/main/data/aux/dem/srtm_30m.tif

~/InSAR/main/data/aux/dem/3dep_10m.tif

Auto-converted to ISCE (*.dem.wgs84 + .vrt + .xml) on first use.

Gauges / areas:

data/vector/gauge_locations.geojson (WGS84 points).

data/vector/water_areas.geojson (WGS84 polygons with area field).

data/aux/gauges/eden_water_levels.csv (wide: StationID, Lat, Lon, YYYY-MM-DD...).

data/aux/gauges/eden_ground_elevation.txt (for above-ground correction).

data/aux/gauges/acquisition_dates.csv (column: date in YYYYMMDD).

Tropo (GACOS): .ztd + .rsc files in data/aux/tropo/ (one per REF/SEC).

Quick Start
# 1) Build interferograms for ALL pairs (SRTM and 3DEP per pair)
python 2_run_pair_conda.py

# (Optional) Resume quicklooks only, skip ISCE if IFG exists
python 2_run_pair_conda.py --resume

# 2) Apply IONO & TROPO, export vertical displacement (RAW, IONO, TROPO, TROPO_IONO)
python 3_do_corrections.py --batch /mnt/DATA2/bakke326l/processing/interferograms

# 3) Build per-area gauge tables + clip vertical products; write coverage report
python 4_organize_areas.py

# 4) Accuracy assessment (shared split) + 4 per-pair exports into <AREA>/results/
python 5_accuracy_assessment.py  --reps 50 --seed 42

# 5) Visualizations (per-pair, per-area, all-areas; plus time-series)
python 6_visualization.py --target-density 500

Pipeline Stages
2_run_pair_conda.py — Build interferograms

Runs each pair twice (SRTM & 3DEP) using ISCE2 stripmapApp. Creates quicklooks & logs status.

Key features

Auto-convert DEM GeoTIFF to ISCE format (*.dem.wgs84 + .xml) on first use.

Fixed multilooks & filter defaults (override via CLI).

Resume mode: if IFG exists, rebuilds inspect/ quicklooks and skips ISCE.

Per-pair workdir: path<PATH>_<REF>_<SEC>_<DEM>.

Usage

# Batch all pairs (from pairs.csv); both DEMs
python 2_run_pair_conda.py

# Limit to first N rows (smoke test)
python 2_run_pair_conda.py --n 10

# Single pair (both DEMs)
python 2_run_pair_conda.py --path 150 20071216 20080131

# Override multilooks & Goldstein alpha
python 2_run_pair_conda.py --range 10 --az 16 --alpha 0.6

# Resume (skip ISCE if IFG present; rebuild quicklooks)
python 2_run_pair_conda.py --resume


Outputs

interferogram/ (VRT/ENVI products).

inspect/ quicklook TIFF/PNG (phsig.cor, topophase.cor, filt_topophase.unw).

inspect/FRINGES_<REF>_<SEC>_<DEM>.{tif,png} (wrapped phase preview).

Status CSV: /mnt/DATA2/bakke326l/processing/interferograms/_reports/path_status.csv.

3_do_corrections.py — Apply IONO/TROPO & export vertical displacement

Processes each pair directory:

RAW vertical displacement (vertical_displacement_cm_*_RAW.geo.tif).

IONO-only correction → vertical (*_IONO.geo.tif).

TROPO correction (GACOS) → vertical (*_TROPO.geo.tif) → IONO on TROPO → vertical (*_TROPO_IONO.geo.tif).

Vertical cm conversion

Uses ALOS L-band wavelength (0.2362 m).

Removes LOS median prior to dividing by cos(incidence) to avoid 1/cos ramp.

Swath mask applied (GDAL mask or amplitude fallback) — off-swath → NaN.

Usage

# One pair dir
python 3_do_corrections.py /mnt/DATA2/.../path150_20071216_20080131_SRTM

# Batch process root
python 3_do_corrections.py --batch /mnt/DATA2/bakke326l/processing/interferograms


Outputs (per pair)

interferogram/filt_topophase_iono.unw.geo

interferogram/filt_topophase_tropo.unw.geo (2-band ENVI: amp, phase)

interferogram/filt_topophase_tropo_iono.unw.geo

inspect/vertical_displacement_cm_<REF>_<SEC>_{RAW|IONO|TROPO|TROPO_IONO}.geo.tif (+ PNGs)

4_organize_areas.py — Per-area clipping & gauge tables

Computes water above ground (cm) for all dates using EDEN levels and ground elevations — writes a corrected CSV with the same columns as the source.

Builds per-area gauge tables:

<AREA>/water_gauges/eden_gauges.csv

<AREA>/water_gauges/eden_metadata.csv

Clips each vertical displacement raster by each water area polygon, requiring coverage (default ≥ 65%) and not all-NaN. DEM/CORR tags are embedded to avoid overwrites.

Writes a coverage report CSV.

Usage

# Global search for */inspect/vertical_displacement_cm_*.geo.tif under VERT_ROOT
python 4_organize_areas.py

# One file
python 4_organize_areas.py --vertical-file /path/to/vertical_displacement_cm_..._RAW.geo.tif

# One pair directory (searches its inspect/)
python 4_organize_areas.py --pair-dir /mnt/DATA2/.../path150_20071216_20080131_SRTM

# Change coverage threshold
python 4_organize_areas.py --min-coverage-pct 50


Outputs

/mnt/DATA2/.../processing/areas/<AREA>/
├─ water_gauges/
│  ├─ eden_gauges.csv
│  └─ eden_metadata.csv
├─ interferograms/
│  ├─ <AREA>_vertical_cm_<REF_SEC>_<DEM>_<CORR>.tif
│  └─ <AREA>_vertical_cm_<REF_SEC>_<DEM>_<CORR>.png
└─ _reports/coverage_report.csv

5_accuracy_assessment.py — Shared-split metrics + per-pair TIFFs

For each AREA (or a single area):

Discovers per-area interferograms:

<AREA>/interferograms/<AREA>_vertical_cm_<REF>_<SEC>_<DEM>_<CORR>.tif


Loads per-area gauge table; computes Δh_vis = max(sec,0) − max(ref,0) (cm).

Samples every raster at gauges (3×3 mean).

Builds the common gauge set valid across all DEM/CORR rasters for the pair.

Creates replicate plans (default 50) with:

~60% farthest-point calibration (spread), rest validation.

Iteratively remove crowded calibration points down to n_cal=2; then center-only n_cal=1.

Evaluate:

least_squares (y = a·x + b; force a=−1 for n_cal≤2)

idw_dhvis (IDW on Δh_vis at validation gauges)

Per pair, writes 4 GeoTIFFs (using replicate #1 plan):

idw60_<PAIR>.tif (Δh_vis, 60% cal)

cal_ti_60pct_<DEMsel>_<PAIR>.tif (TROPO_IONO, 60% cal)

cal_ti_d{D}_<DEMsel>_<PAIR>.tif (TROPO_IONO, n_cal closest to target density)

cal_ti_1g_<DEMsel>_<PAIR>.tif (TROPO_IONO, center-only)

Usage

# All areas (defaults: DEMS=SRTM 3DEP; CORRS=RAW TROPO IONO TROPO_IONO)
python 5_accuracy_assessment.py

# One area
python 5_accuracy_assessment.py --area ENP

# Tuning
python 5_accuracy_assessment.py \
  --reps 50 --seed 42 --idw-power 2.0 --output-density 500 \
  --dems SRTM 3DEP --corrs RAW TROPO IONO TROPO_IONO


Output CSV (fresh each run)

<AREA>/results/accuracy_metrics.csv


One row per (replicate, n_cal, DEM/CORR, method) with:
area, pair_ref, pair_sec, dem, corr, method, replicate, n_total, n_cal, n_val, area_km2, area_per_gauge_km2, rmse_cm, mae_cm, bias_cm, r, a_gain, b_offset_cm.

6_visualization.py — Figures (accuracy vs density, period plots)

Builds publication-ready figures from accuracy_metrics.csv and per-pair TIFFs.

What it makes

Per-PAIR acc-vs-density with clipped 5–95% bands, SRTM & 3DEP LS + IDW; bottom row shows:

60% calibrated TROPO_IONO map

60% IDW(Δh_vis) map
→ <AREA>/results/acc_den_pair_<PAIR>.png

Per-AREA acc-vs-density (all pairs), optional median/mean lines
→ <AREA>/results/acc_den_area_<AREA>.png

ALL-AREAS acc-vs-density
→ <areas_root>/results/acc_den_ALL_AREAS.png

Per-AREA time-series boxplots (IDW • TROPO_IONO at target density), per-DEM & combined
→ <AREA>/results/acc_period_*.png

ALL-AREAS time-series (IDW • TROPO_IONO at target density)
→ <areas_root>/results/acc_period_ALL_AREAS_<D>.png

Usage

# Everything
python 6_visualization.py

# One area; change IDW density DEM, hide mean
python 6_visualization.py --area ENP --idw-dem-density SRTM --no-show-mean

# Set target density for period plots (km²/gauge)
python 6_visualization.py --target-density 500

Helper Modules

help_xml_isce.py
Build a minimal stripmapApp.xml for ALOS with sensible defaults (multilooks, split-spectrum, two-stage unwrap, dense offsets, etc.). Detects FBD/FBS and sets RESAMPLE_FLAG=dual2single for FBD.

help_xml_dem.py
Convert DEM GeoTIFF → ISCE: reproject to WGS-84 (EPSG:4326) if needed, run gdal_translate -of ISCE, and generate ISCE XML (gdal2isce_xml). Writes a small DEM report.

help_atm_correction.py
do_iono_correction(...) subtracts dispersive (ionospheric) phase; writes corrected and rewrapped products.
do_tropo_correction(...) runs the GACOS troposphere workflow: .rsc→.hdr, reproject to IFG grid, zenith→slant using incidence, reference−secondary differential, then subtract from unwrapped IFG.

help_show_fringes.py
create_fringe_tif(...) reads filt_topophase.flat.geo (complex), computes wrapped phase, writes FRINGES_*.tif for quick inspection.

Outputs & Naming Conventions

Pair directory: path<PATH>_<REF>_<SEC>_<DEM> where:

<PATH>: track (e.g., 150)

<REF>/<SEC>: YYYYMMDD

<DEM>: SRTM or 3DEP

Vertical displacement (per pair):

interferogram/
inspect/vertical_displacement_cm_<REF>_<SEC>_{RAW|IONO|TROPO|TROPO_IONO}.geo.tif


Per-area clipped rasters:

<AREA>/interferograms/<AREA>_vertical_cm_<REF_SEC>_<DEM>_<CORR>.{tif,png}


Per-pair figure:

<AREA>/results/acc_den_pair_<REF_SEC>.png


Global reports:

Status: /mnt/DATA2/.../interferograms/_reports/path_status.csv

Coverage: <areas_root>/_reports/coverage_report.csv

Configuration Notes & Tips

Environment variables

ISCE_HOME must point to ISCE2 install (script calls stripmapApp.py).

GDAL_CACHEMAX: 2_run_pair_conda.py sets 512 MB by default; bump if RAM allows.

Performance

Place /mnt/DATA2 on fast storage (NVMe/SSD).

Use --n to smoke-test a subset before full batch.

Resume mode

2_run_pair_conda.py --resume detects core IFG presence and skips ISCE, rebuilding inspect/ only.

Masks

Swath masks are honored for vertical conversion; off-swath → NaN.

Units

Vertical displacement in centimeters, LOS median removed per pair before vertical projection.

Troubleshooting

ISCE not found: ensure ISCE_HOME is set and stripmapApp.py exists at $ISCE_HOME/applications/.

gdal_translate / gdalwarp not found: install GDAL via conda-forge and ensure tools are on PATH.

No CEOS files: check raw layout: <raw>/path<PATH>/<YYYYMMDD>/ and presence of LED-* & IMG-HH-*.

Ionosphere/Tropo failures:

IONO: ensure ionosphere/dispersive.bil.unwCor.filt.geo and mask.bil.geo exist (produced earlier by your iono workflow).

TROPO (GACOS): ensure <TROPO_DIR>/<YYYYMMDD>.ztd and .rsc are present for ref & sec.

No per-area outputs:

Verify water_areas.geojson has an area field; geometries valid in WGS84.

Increase --min-coverage-pct or check raster footprint overlap.

Accuracy CSV empty:

The common gauge set may be too small (<3). Check per-area clipped coverage & sampling.

Reproducibility

Set --seed (default 42) in 5_accuracy_assessment.py to keep replicate plans stable.

6_visualization.py reads the fresh accuracy_metrics.csv per area.

DEM conversion is deterministic; reports capture basic stats and metadata.

License / Acknowledgements

This pipeline depends on ISCE2, GDAL, snaphu, and the GACOS product.

EDEN water‐level data acknowledged for gauge validation.

Portions of iono/tropo workflows mirror ISCE2 tutorial patterns.

Happy interferogramming! If you want this README rendered with your actual custom paths, just say the word and share any deviations from the defaults above.