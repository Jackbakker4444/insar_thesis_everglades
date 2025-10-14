# InSAR Thesis — Everglades Water-Level Monitoring (ALOS PALSAR)

> End-to-end pipeline for downloading SAR data, generating interferograms with ISCE2, applying iono/tropo corrections, clipping to water areas, benchmarking against EDEN gauges, and producing figures & stats for the thesis.

---

## TL;DR (What this repo does)

1. **Acquire & stage ALOS SAR** via ASF/Vertex (cookie-auth bulk download) into `raw/path<PATH>/<YYYYMMDD>/…`.
2. **Prepare inputs**: convert **DEM/DTM → ISCE WGS-84 bundle** and auto-write **`stripmapApp.xml`** per pair (handles FBD→single-pol, looks, filtering, split-spectrum).
3. **Batch-run ISCE2** interferograms **twice per pair** (**SRTM** and **3DEP**) at consistent posting (30 m).
4. **Atmospheric corrections**: build **RAW**, **IONO**, **TROPO (GACOS)**, and **TROPO+IONO** products; optional wrapped-phase **FRINGES** quicklooks.
5. **Per-area organization** of gauges, AOI clips, coverage, and pair metadata.
6. **Accuracy evaluation**: RMSE/Bias across **DEM × correction** and vs **gauge density** (sparsity); exports metrics CSVs.
7. **Inferential tests**: repeated-measures ANOVA & paired t-tests (with/without **IDW** baseline); area-level and all-areas summaries + rankings.
8. **Visualize**: corrections **2×3** maps, **DEM** boxplots (equal/variable widths), **density** curves with per-pair LS/AOI/IDW maps, and all-areas overlays; plus scatter of mean RMSE vs temporal baseline.

---

## Table of Contents

* [Repository Layout](#repository-layout)
* [Environment & Dependencies](#environment--dependencies)
* [Data Prerequisites](#data-prerequisites)
* [Pipeline Overview (Scripts 1–8)](#pipeline-overview-scripts-1–8)

  * [1) Obtain SAR data](#1-obtain-sar-data)
  * [2) Batch-run ISCE2 (3DEP & SRTM)](#2-batch-run-isce2-3dep--srtm)
  * [3) Corrections + Vertical Exports](#3-corrections--vertical-exports)
  * [4) Per-area Gauges & Clips](#4-per-area-gauges--clips)
  * [5) Accuracy (DEM × Corr) at Multiple Densities](#5-accuracy-dem--corr-at-multiple-densities)
  * [6) Inferential Tests (Corrections & DEMs)](#6-inferential-tests-corrections--dems)
  * [7) Density Study (SRTM+RAW)](#7-density-study-srtmraw)
  * [8) Visualizations](#8-visualizations)
  * [Helper scripts](#helper-scripts)
* [Naming Conventions](#naming-conventions)
* [Troubleshooting](#troubleshooting)
* [Quickstart](#quickstart)
* [Acknowledgements](#acknowledgements)
* [License & Citation](#license--citation)

---

## Repository Layout

```
~/InSAR/main/                      # Code & helpers
├─ data/
│  ├─ aux/
│  │  ├─ dem/                      # SRTM & 3DEP TIFFs (+ ISCE .wgs84 binaries written alongside)
│  │  └─ gauges/                   # EDEN tables, ground elevations, acquisition dates
│  └─ vector/                      # gauge_locations.geojson, water_areas.geojson
└─ processing/
   ├─ interferograms/              # path<PATH>_<REF>_<SEC>_{SRTM|3DEP}/…
   ├─ areas/                       # <AREA>/water_gauges, <AREA>/interferograms, results/
   └─ inspect/                     # global quicklook PNGs
/mnt/DATA2/bakke326l/raw/          # ASF archives organized per path/date
```

> A fuller filesystem snapshot exists in `project_structure.txt`.

---

## Environment & Dependencies

* **Core:** Python 3.10+, GDAL utils (`gdal_translate` on `PATH`), ISCE2 (SCons install), GCC/GFortran.
* **Python:** `numpy pandas geopandas shapely rasterio pyproj matplotlib scipy statsmodels`

Create a minimal environment (conda-forge):

```bash
conda create -n insar python=3.10 numpy pandas geopandas shapely rasterio pyproj matplotlib scipy statsmodels -c conda-forge
conda activate insar
# Ensure: gdal_translate is on PATH (install GDAL via conda-forge if needed).
# Ensure: ISCE2 apps (e.g., stripmapApp.py) reachable via $ISCE_HOME/applications/
```

---

## Data Prerequisites

* **ALOS PALSAR** Vertex ZIPs + CSV → `<RAW_DIR>/tmp_downloads/`
* **DEM TIFFs**

  * SRTM: `~/InSAR/main/data/aux/dem/srtm_30m.tif`
  * 3DEP: `~/InSAR/main/data/aux/dem/3dep_10m.tif`
    (ISCE `.dem.wgs84` and `.wgs84.xml` are created next to each TIF.)
* **Tropo/Iono** auxiliaries (e.g., GACOS) → `~/InSAR/main/data/aux/tropo/`
* **Gauges & areas**

  * EDEN wide daily table + ground elevations + acquisition dates
  * `data/vector/gauge_locations.geojson`, `data/vector/water_areas.geojson`

---

## Pipeline Overview (Scripts 1–8)

### 1) Obtain SAR data

**File:** `1_obtaining_sar_data.py`
**Does:**

* Runs your ASF downloader (`help_download_all_path_150.py`) → ZIPs + Vertex CSV into `<RAW_DIR>/tmp_downloads/`.
* Unpacks each ZIP to `path<PATH>/<YYYYMMDD>/`, writes `<GRANULE>.txt` with the full CSV row, and deletes the ZIP on success.

**Run**

```bash
python 1_obtaining_sar_data.py
```

---

### 2) Batch-run ISCE2 (3DEP & SRTM)

**File:** `2_run_insar.py`
**Inputs:** `data/pairs.csv` (`path,reference,secondary`), DEM TIFFs, `raw/` layout.
**Does:**

* For each pair: runs **twice** (3DEP, then SRTM) with `range=10, az=16, alpha=0.6`.
* Creates ISCE XML, product VRTs, **quicklook** GeoTIFF/PNG under `inspect/`.
* Appends to `_reports/path_status.csv`.

**Output root:**
`/mnt/DATA2/bakke326l/processing/interferograms/path<PATH>_<REF>_<SEC>_{3DEP|SRTM}/…`

**Run**

```bash
python 2_run_insar.py
python 2_run_insar.py --n 10                 # smoke test
python 2_run_insar.py --path 150 20071216 20080131
python 2_run_insar.py --range 10 --az 16 --alpha 0.6
python 2_run_insar.py --resume               # skip if IFG already exists
```

---

### 3) Corrections + Vertical Exports

**File:** `3_do_corrections.py`
**Does:**

* Converts unwrapped phase → **LOS** → **vertical (cm)** with off-swath masking.
* Builds **RAW**, **IONO**, **TROPO**, **TROPO+IONO** variants of unwrapped & vertical.
* Exports quicklooks for **ΔAPS (tropo differential)** and **ionospheric dispersive** phase.

**Notes:** Needs `geometry/los.rdr.geo(.vrt)` for vertical; if missing, vertical exports are skipped.

**Run**

```bash
python 3_do_corrections.py /mnt/DATA2/.../path150_20071216_20080131_SRTM
python 3_do_corrections.py --batch /mnt/DATA2/bakke326l/processing/interferograms
```

---

### 4) Per-area Gauges & Clips

**File:** `4_organize_areas.py`
**Does:**

1. Builds **per-area EDEN** “water above ground (cm)” tables + metadata under `<AREA>/water_gauges/`.
2. For each vertical raster under `…/inspect/vertical_displacement_cm_<REF>_<SEC>_{RAW|IONO|TROPO|TROPO_IONO}.geo.tif`, clips by *(valid-data footprint ∩ area polygon)* and writes per-area **TIFF + PNG** if coverage ≥ threshold.
3. Writes global **coverage report** CSV.

**Outputs:**

* `<AREA>/water_gauges/eden_gauges.csv`, `eden_metadata.csv`
* `<AREA>/interferograms/<AREA>_vertical_cm_<REF>_<SEC>_<DEM>_<CORR>.{tif,png}`
* `processing/areas/_reports/coverage_report.csv`

**Run**

```bash
python 4_organize_areas.py                      # global search
python 4_organize_areas.py --min-coverage-pct 50.0
python 4_organize_areas.py --pair-dir /mnt/DATA2/.../path150_20071216_20080131_SRTM
```

---

### 5) Accuracy (DEM × Corr) at Multiple Densities

**File:** `5_accuracy_assessment_dem_corr.py`
**Does:**

* For each `(AREA, PAIR, DEM, CORR)` and **four calibration densities (60/45/30/15%)**, computes LS metrics (RMSE, MAE, bias, σₑ, r) and an **IDW(Δh_vis)** baseline on identical splits.
* From replicate #1 at **60%**: exports **IDW Δh_vis** grid, **calibrated** rasters, and **60/40 split** GeoJSON.

**Output:** `<AREA>/results/accuracy_metrics.csv` (+ GeoTIFFs/GeoJSON from replicate #1)

**Run**

```bash
python 5_accuracy_assessment_dem_corr.py --reps 200 --seed 42 \
  --dems SRTM 3DEP --corrs RAW TROPO IONO TROPO_IONO
```

---

### 6) Inferential Tests (Corrections & DEMs)

**File:** `6_inferential_tests.py`
**Corrections (two tracks):**

1. *WITH IDW*: RAW, IONO, TROPO, TROPO_IONO, **IDW**
2. *NO IDW*: RAW, IONO, TROPO, TROPO_IONO

* Per-pair repeated-measures ANOVA; per-pair paired t-tests (Holm adjusted).
* Area-level & all-areas ANOVA; **ranking** tables (with-IDW).

**DEMs:** SRTM vs 3DEP for a chosen correction (default RAW) with paired t-tests at pair/area/all-areas.

**Outputs (per area + all areas):**
A set of CSVs including `corrections_*anova__{with_idw|no_idw}.csv`, `*_ranking__with_idw.csv`, `dem_*ttest__<CORR>.csv`, and summaries.

**Run**

```bash
python 6_inferential_tests.py
python 6_inferential_tests.py --area ENP
python 6_inferential_tests.py --metric log_rmse_cm
python 6_inferential_tests.py --dem-corr IONO
```

---

### 7) Density Study (SRTM+RAW)

**File:** `7_accuracy_assessment_density.py`

**Does:**

* Locks to each pair’s `…_SRTM_RAW.tif` (shared valid-data mask across steps).
* Evaluates **least_squares (LS)** and **IDW** on **identical** gauge sets and **identical** cal/val splits.
* Starts near **60%** calibration using **stochastic farthest-point**, then steps down one by one, with the **validation set fixed**.
* Records per-step metrics (`rmse_cm`, `mae_cm`, `bias_cm`, `r`, `n_cal`, `n_val`) and **density** = km²(valid swath) / `n_cal`.
* Exports three sanity-check rasters per pair: `dens_idw60_…`, `dens_cal_60pct_…`, `dens_cal_1g_…` (float32, NaN nodata).

**Outputs:**

* `<AREA>/results/accuracy_metrics_density_SRTM_RAW.csv`
* Per-pair GeoTIFFs:

  * `dens_idw60_SRTM_RAW_<PAIR>.tif`
  * `dens_cal_60pct_SRTM_RAW_<PAIR>.tif`
  * `dens_cal_1g_SRTM_RAW_<PAIR>.tif`

**Run**

```bash
python 7_accuracy_assessment_density.py --reps 50 --seed 42 --spread-top-m 5
```

**Notes:**

* Increase `--reps` for smoother curves; keep `--seed` for reproducibility.
* IDW grids and LS exports inherit the SRTM+RAW mask.


---

### 8) Visualizations

**File:** `8_visualization_dem_corr.py`
**What it covers (Corrections & DEMs):**

* **Corrections maps (SRTM)** as a **2×3** grid per pair:
  `Raw | Tropospheric / Ionospheric | Tropospheric+Ionospheric / Soil (WATER-only) | Satellite (WATER-only)`
  Single shared colorbar; per-panel north arrow & scalebar; Soil & Satellite clipped to the area’s WATER polygon.
* **DEM comparisons (RAW, LS 60%)** per area: SRTM vs 3DEP boxplots shown **(i)** with equal widths and **(ii)** with widths ∝ pair duration.
  RMSE panel fixed to 0–25 cm; Bias auto; dates compact on x-axis.
* **All-areas summaries (RAW, LS 60%)**: the same DEM comparison logic applied across all areas.
* **Baseline sensitivity** (RAW, LS 60%): mean RMSE vs temporal baseline (days) across pairs.

**Outputs:**

* `corr_maps_pair_<PAIR>_2x3.png` (per pair, under `<area>/results/`)
* `dem_boxplots_area_<AREA>_equalwidth_RAW.png` and `dem_boxplots_area_<AREA>_varwidth_RAW.png`
* `dem_boxplots_ALL_areas_equalwidth_RAW.png` and `dem_boxplots_ALL_areas_varwidth_RAW.png`
* `scatter_mean_rmse_vs_temporal_baseline_RAW.png`

**Run**

```bash
# Everything with defaults (all areas)
python 8_visualization_dem_corr.py

# Single area + custom satellite XYZ + custom water polygons
python 8_visualization_dem_corr.py --area ENP \
  --sat-url 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' \
  --water-areas /home/bakke326l/InSAR/main/data/vector/water_areas.geojson
```

---

**File:** `8_visualization_density.py`
**What it covers (Density):**

* **Per-pair density curves** for **RMSE** and **Bias** vs **gauge density** (km² per gauge, log-x):
  median line + **5–95%** band, for **Interferogram (LS)** and **IDW (same gauges)**.
* **Bottom-row AOI maps** per pair: **LS calibrated**, **Satellite AOI** (with WATER overlay), and **IDW baseline**; gauge overlay (cal = black, val = red).
  Map spacing/height are tunable via `--maps-gap-frac` and `--maps-height-frac`.
* **All-areas (by area, no dots)**: each area plotted as its own median line + 5–95% band for the selected method (`LEAST_SQUARES` or `IDW_DHVIS`).
* **All-areas (combined, no dots)**: a **single** median + 5–95% band aggregated directly by density, with the x-axis spanning the **full density range** present in the data.

**Outputs (per pair + all areas):**

* `density_<corr_lower>_idw_pair_<PAIR>.png` (per pair, under `<area>/results/`)
* `density_all_areas_by_area_<DEM>_<CORR>_<METHOD>.png`
* `density_all_areas_combined_<DEM>_<CORR>_<METHOD>.png`

**Run**

```bash
# Defaults: DEM=SRTM, CORR=RAW, METHOD=LEAST_SQUARES (writes per-pair figures + both all-areas figures)
python 8_visualization_density.py

# Single area, TROPO_IONO, all-areas with IDW method, custom WATER overlay and provider
python 8_visualization_density.py --area ENP --dem SRTM --corr TROPO_IONO --method IDW_DHVIS \
  --water-areas /home/bakke326l/InSAR/main/data/vector/water_areas.geojson \
  --sat-provider Esri.WorldImagery

# With explicit gauges template + tighter map gap below the Bias panel
python 8_visualization_density.py --gauges-template "/mnt/DATA2/.../{area}/results/gauges_split_60pct_{pair}.geojson" \
  --maps-gap-frac 0.08 --maps-height-frac 0.30
```

---

---

### Helper Scripts

**File:** `help_xml_isce.py`
**Purpose:** build a minimal, robust **`stripmapApp.xml`** for ISCE2 (ALOS stripmap).

* **What it provides:**
  `ceos_files()` (find LED/IMG-HH files), `detect_beam_mode()` (FBD/FBS),
  `sensor_component()` (adds `RESAMPLE_FLAG=dual2single` for **FBD**),
  `write_stripmap_xml()` (writes a full `stripmapApp.xml` for one REF–SEC pair).
* **Assumptions:** raw ALOS CEOS under `<raw_dir>/path<PATH>/<YYYYMMDD>/…`, HH channel only.
* **XML defaults (tuned for this project):** multilooks (user-set), Goldstein filter `filter strength`, **dense offsets ON**, **split spectrum + dispersive ON**, **rubbersheeting in range & azimuth ON**, posting **30 m**, two-stage unwrapping (`snaphu`/`MCF`).
* **Output:** a ready-to-run `stripmapApp.xml` in your chosen location.

**Run**

```python
from pathlib import Path
from help_xml_isce import write_stripmap_xml

write_stripmap_xml(
    xml_file=Path("/out/stripmapApp.xml"),
    path="150", ref_date="20071216", sec_date="20080131",
    raw_dir=Path("/mnt/DATA2/bakke326l/raw"),
    work_dir=Path("/mnt/DATA2/bakke326l/processing/interferograms/path150_20071216_20080131_SRTM"),
    dem_wgs84=Path("/home/bakke326l/InSAR/main/data/aux/dem/srtm_30m.dem.wgs84"),
    range_looks=10, az_looks=16, filter_strength=0.6
)
```

---

**File:** `help_xml_dem.py`
**Purpose:** convert a GeoTIFF DEM/DTM to an **ISCE2-ready WGS-84 DEM bundle**.

* **What it does (pipeline):**

  1. Reproject to **EPSG:4326** (ellipsoidal heights; `--keep-egm` to skip),
  2. `gdal_translate` → **ISCE** binary `*.dem.wgs84` (+ `.vrt`),
  3. `gdal2isce_xml` → sidecar `*.xml`,
  4. optional text report `*.txt` (size, posting, bbox, stats).
* **Requires:** GDAL ≥ 3.4 (tools on `PATH`), ISCE2 Python (`applications.gdal2isce_xml`).
* **Outputs (same basename):** `.dem.wgs84`, `.dem.wgs84.vrt`, `.dem.wgs84.xml`, `.dem.wgs84.txt`.

**Run**

```bash
# Basic
python help_xml_dem.py --input /path/lidar_utm.tif \
                       --output /path/dem/my_area.dem.wgs84

# Keep geoid heights, overwrite if exists, custom temp dir
python help_xml_dem.py --input /path/dtm_egm.tif \
                       --output /path/dem/my_area.dem.wgs84 \
                       --keep-egm --overwrite --tmp-dir ./tmp_reproj
```

---

**File:** `help_show_fringes.py`
**Purpose:** generate a wrapped-phase **“FRINGES”** GeoTIFF from the **geocoded complex** interferogram.

* **Reads:** `<pair>/interferogram/filt_topophase.flat.geo` (band 1 complex).
* **Writes:** `FRINGES_<REF>_<SEC>_ra<RANGE>_az<AZ>_<ALPHA>.tif` (float64, wrapped phase in [−π, π), NaN nodata) to your chosen `out_dir`.
* **Spatial metadata:** GeoTransform + CRS copied from source.

**Run**

```python
from pathlib import Path
from help_show_fringes import create_fringe_tif

create_fringe_tif(
    work_dir=Path("/mnt/DATA2/.../path150_20071216_20080131_SRTM"),
    out_dir=Path("/mnt/DATA2/.../inspect"),
    ref=20071216, sec=20080131, range_looks=10, azi_looks=16, alpha=0.6
)
```

---

**File:** `help_atm_correction.py`
**Purpose:** high-level helpers for **ionospheric** and **tropospheric (GACOS)** corrections on ISCE2 products.

* **`do_iono_correction(work_dir, out_dir, input_unw, output_suffix="_iono")`**
  – Subtracts **dispersive** iono phase (`ionosphere/dispersive.bil.unwCor.filt.geo`),
  – Applies `ionosphere/mask.bil.geo` (off-mask → NaN),
  – Writes:
  `interferogram/filt_topophase{suffix}.unw.geo` (single-band phase, rad) and
  `interferogram/filt_topophase{suffix}_wrapped.unw.geo` (wrapped to [−π, π]).
  – Band handling: reads phase from band 2 if IFG is 2-band, else band 1.

* **`do_tropo_correction(wdir, ref, sec, gacos_dir, tropo_dir)`**
  – Converts GACOS `*.rsc` → ENVI `*.hdr`,
  – Reprojects `<ref>.ztd` / `<sec>.ztd` to IFG grid,
  – Converts **zenith → slant phase** using incidence (deg) and λ(ALOS)=**0.2362 m**:
  `phase = -(4π/λ) * (ZTD / cos(inc))`,
  – Builds ΔAPS = APS_ref − APS_sec,
  – Subtracts ΔAPS from unwrapped IFG (keeps amplitude in band 1),
  – **Outputs:**
  `<tropo_dir>/{ref}.ztd.geo`, `{sec}.ztd.geo`, `{ref}.aps.geo`, `{sec}.aps.geo`, `{ref}_{sec}.aps.geo`,
  `interferogram/filt_topophase_tropo.unw.geo` (2-band: amp, corrected phase).

**Run**

```python
from pathlib import Path
from help_atm_correction import do_iono_correction, do_tropo_correction

pair = Path("/mnt/DATA2/.../path150_20071216_20080131_SRTM")

# 1) Tropospheric correction (GACOS)
do_tropo_correction(
    wdir=pair, ref=20071216, sec=20080131,
    gacos_dir=Path("/mnt/DATA2/.../gacos"),
    tropo_dir=pair / "troposphere"
)

# 2) Ionospheric correction (on raw or on tropo-corrected)
iono_out = do_iono_correction(
    work_dir=pair, out_dir=pair / "interferogram",
    input_unw=pair / "interferogram" / "filt_topophase_tropo.unw.geo",
    output_suffix="_tropo_iono"
)
```

---

**File:** `download_all_path_150.py`
**Purpose:** **ASF/Vertex bulk downloader** with Earthdata Login (UR S) auth and a project-local download path.

* **Defaults:** downloads to `…/raw/tmp_downloads` (created if missing).
* **Sources:** embedded URL list **or** a provided **`.metalink`**/**`.csv`** from Vertex.
* **Auth:** saves cookie to `~/.bulk_download_cookiejar.txt` and reuses it.
* **Flags:** `--insecure` to relax SSL checks (use only for trusted hosts).
* **Outputs:** ZIP archives placed in the destination directory; prints a transfer summary.

**Run**

```bash
# Use the embedded URL list
python download_all_path_150.py

# Use a metalink/CSV you saved from Vertex
python download_all_path_150.py /path/to/downloads.metalink local.metalink local.csv

# (Optional) Disable strict SSL cert checks for trusted endpoints
python download_all_path_150.py --insecure
```

---

**File:** `isce_cleanup_all_pairs.sh`
**Purpose:** safe, batch **cleanup** of ISCE pair folders (dry-run by default).

* **Defaults:** `--root=/mnt/DATA2/bakke326l/processing/interferograms`, `--pattern='path150_*'`.
* **Keeps:** `interferogram/`, `ionosphere/`, `troposphere/`, `geometry/`, `PICKLE/`, `inspect/`, `coregisteredSlc/`, and `stripmapApp.xml`, `stripmapProc.xml`, `isce.log`.
* **Deletes:** `*_raw*`, `*_slc*`, `offsets/`, `denseOffsets/`, `misreg/`, `SplitSpectrum/`, `resampinfo.bin`, zero-byte `*.log`, and DEM scratch (`3dep_*.dem.*`, `dem.crop*`).
* **Safety:** refuses to run on `/`; non-matching dirs are skipped.

**Run**

```bash
# Dry-run (default): shows what would be removed
bash isce_cleanup_all_pairs.sh

# Apply deletions
bash isce_cleanup_all_pairs.sh --apply

# Custom root/pattern
bash isce_cleanup_all_pairs.sh --root=/path/to/interferograms --pattern='path150_*' --apply
```

---

## Naming Conventions

* **Pair folders:** `path<PATH>_<REF>_<SEC>_{SRTM|3DEP}` (dates `YYYYMMDD`).
* **Vertical GeoTIFFs:** `vertical_displacement_cm_<REF>_<SEC>_{RAW|IONO|TROPO|TROPO_IONO}.geo.tif` (under `…/interferogram/inspect/`).
* **Per-area clips:** `<AREA>_vertical_cm_<REF>_<SEC>_<DEM>_<CORR>.tif`.
* **Density rasters:**

  * `dens_idw60_SRTM_RAW_<PAIR>.tif`
  * `dens_cal_60pct_SRTM_RAW_<PAIR>.tif`
  * `dens_cal_1g_SRTM_RAW_<PAIR>.tif`

---

## Troubleshooting

* **Missing LOS grid** → vertical export is skipped in step 3 (corrections still produced).
* **DEM binaries (.dem.wgs84/.wgs84.xml)** → ensure GDAL present; re-run step 2 (it auto-creates them).
* **Path expectations** → scripts assume docstring paths; override via CLI options where available.
* **Memory** → quicklooks use downsampled reads; further reduce read windows in helpers if needed.

---

## Quickstart

```bash
# 1) Place Vertex ZIPs + CSV under <RAW_DIR>/tmp_downloads/
python 1_obtaining_sar_data.py

# 2) Run interferograms (3DEP + SRTM)
python 2_run_insar.py --resume

# 3) Apply corrections + export vertical maps
python 3_do_corrections.py --batch /mnt/DATA2/bakke326l/processing/interferograms

# 4) Build per-area gauges & clips
python 4_organize_areas.py

# 5) Accuracy across DEM × Corrections
python 5_accuracy_assessment_dem_corr.py

# 6) Inferential tests (ANOVA/t-tests)
python 6_inferential_tests.py

# 7) Density study (SRTM+RAW)
python 7_accuracy_assessment_density.py

# 8) Figures
python 8_visualization.py
```

---

## Acknowledgements

* ALOS PALSAR via ASF/Vertex.
* ISCE2 (JPL/Caltech) for interferometric processing.
* EDEN gauge network for water-level validation.
* GACOS for tropospheric delay products.

---

## License & Citation

* **License:** not specified (default: all rights reserved by the author unless indicated otherwise).
* **Cite as:**
  Jack Bakker (2025), *Assessing Data Availability Impact for InSAR Monitoring of Wetland Water Levels: Everglades, ALOS PALSAR*, MSc Thesis, WUR.
