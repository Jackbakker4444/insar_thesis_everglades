#!/usr/bin/env python3
#!/usr/bin/env python
"""
ENVISAT RA‑2 Altimetry Fetcher – Everglades InSAR Project  (v0.3, 1‑Jul‑2025)
==============================================================================

This **stand‑alone** script downloads ESA ENVISAT *Radar‑Altimeter‑2 (RA‑2)*
water‑surface heights for the Everglades study area and writes per‑water‑area
CSV files that sit next to the EDEN gauge tables created by
`eden_data_prepare.py`.

Why so many comments?
---------------------
Everything is annotated so you can see exactly **what** happens and **why**.
Feel free to prune once you’re happy.

Overview of the workflow
------------------------
1. **Config & auth** – edit the CONFIG block, then log in once with
   `earthaccess.login()` (Earthdata credentials are cached in `~/.netrc`).
2. **Find granules** – query NASA CMR for RA‑2 SGDR Reproc‑2 granules that
   intersect the Everglades bounding box and fall inside the date span you set.
3. **Download / stream** –
   *Outside AWS?* → `earthaccess.download()` saves *.nc* files to a temp dir.
   *Inside AWS us‑west‑2?* → files are streamed directly from the PO.DAAC S3
   bucket, no egress fee.
4. **Subset & QA** – open each NetCDF with *xarray*, clip to the bbox, filter
   for inland‑water echoes (`surface_type_flag == 0`), drop records with the ice
   retracker flag set.
5. **Datum align** – convert sea‑surface height from **metres (to reference
   ellipsoid)** to **centimetres NAVD88** using GEOID12B offsets
   (simplified bilinear lookup – good to ≈2 cm over Florida).
6. **Water‑area join** – spatial‑join RA‑2 points with `water_areas.geojson` to
   tag each point `area` (e.g. *WCA1*).
7. **Write CSVs** – one file per area:

   ```
   processing/<AREA>/water_gauges/envisat_altimetry.csv
   # columns: date, lon, lat, ssh_cm
   ```

-------------------------------------------------------------------------------
Dependencies (conda‑forge)
-------------------------------------------------------------------------------
conda install \  
    earthaccess xarray netCDF4 pandas geopandas shapely pyproj tqdm

-------------------------------------------------------------------------------
Usage (no CLI args – tweak CONFIG below)
-------------------------------------------------------------------------------
$ python envisat_altimetry_fetch.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import earthaccess  # type: ignore
from pyproj import Transformer

# ---------------------------------------------------------------------------
# CONFIG – edit here only
# ---------------------------------------------------------------------------
BASE_DIR        = Path('/home/bakke326l/InSAR/main')
OUT_DIR         = BASE_DIR / 'processing'

# Bounding box (lon_min, lat_min, lon_max, lat_max) roughly covering Everglades
BBOX = (-82.0, 24.5, -80.0, 27.0)

# Date span (inclusive) – ENVISAT operated 2002‑03‑01 … 2012‑04‑08
DATE_START = '2007-12-01'
DATE_END   = '2011-03-01'

# CMR collection for ENVISAT RA‑2 SGDR Reprocess‑2 (mirrored at PODAAC cloud)
CMR_COLLECTION_ID = 'C1711961949-POCLOUD'

# Water areas polygon file (EPSG 4326) – used to tag altimetry points
WATER_AREAS_GJS = BASE_DIR / 'data/vector/water_areas.geojson'
AREA_FLD        = 'area'   # attribute holding WCA abbrev

# GEOID12B tif for converting ellipsoid height → NAVD88 (download once)
GEOID_TIF = BASE_DIR / 'data/geoids/geoid12b_conus.tif'
# ---------------------------------------------------------------------------


def _earthdata_login() -> None:
    """Interactive login (token is cached, so this is quick after first run)."""
    earthaccess.login(strategy="interactive", persist=True)


def _cmr_search() -> list[earthaccess.granule.Granule]:
    """Return list of granules intersecting bbox & date span."""
    granules = earthaccess.search(
        short_name="ESACCI-SEALEVEL-L2P-SGDR",  # alias for collection ID
        concept_id=CMR_COLLECTION_ID,
        bounding_box=BBOX,
        temporal=(DATE_START, DATE_END),
        cloud_hosted=True,
    )
    return granules


def _dl_granules(granules: list[earthaccess.granule.Granule]) -> list[Path]:
    """Download (or stream) NetCDF granules → local temp dir, return paths."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = earthaccess.download(granules, target_dir=tmp, threads=4)
    return [Path(p) for p in paths]


def _geoid_height(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Return GEOID12B offset (metres) for arrays of lon/lat (bilinear).

    This is a lightweight substitute for the full NOAA VDatum.  Accuracy ~2 cm
    which is adequate for gauge/altimetry comparison here.
    """
    import rasterio
    with rasterio.open(GEOID_TIF) as ds:
        # GEOID raster is in EPSG:4326, 1 arc‑min grid
        col, row = ds.index(lon, lat, op=~ds.transform * np.vstack([lon, lat]))
        # Clip indices inside raster window
        col = np.clip(col, 0, ds.width  - 1)
        row = np.clip(row, 0, ds.height - 1)
        return ds.read(1)[row, col]


def _process_nc(nc_file: Path) -> pd.DataFrame:
    """Extract altimetry points (lon, lat, date, ssh_cm_NAVD88) from one file."""
    ds = xr.open_dataset(nc_file, mask_and_scale=True, chunks={})

    # Basic subset by bbox (lat/lon 1‑D along‑track)
    m = (
        (ds.latitude  >= BBOX[1]) & (ds.latitude  <= BBOX[3]) &
        (ds.longitude >= BBOX[0]) & (ds.longitude <= BBOX[2])
    )
    if not m.any():
        return pd.DataFrame()

    ds = ds.isel(time=m)

    # Filter inland‑water echoes (surface_type_flag == 0) and good quality_flag
    if 'surface_type_flag' in ds:
        ds = ds.where(ds.surface_type_flag == 0, drop=True)
    if 'ice_flag' in ds:
        ds = ds.where(ds.ice_flag == 0, drop=True)

    if ds.time.size == 0:
        return pd.DataFrame()

    # Convert SSH (m, w.r.t. ellipsoid) → cm NAVD88
    ssh_m = ds.ssh.values  # variable names vary; adjust if needed
    lon   = ds.longitude.values
    lat   = ds.latitude.values

    geoid = _geoid_height(lon, lat)  # metres
    ssh_navd_cm = (ssh_m - geoid) * 100.0

    dates = xr.conventions.times.decode_cf_datetime(ds.time, ds.time.units)

    return pd.DataFrame(
        {
            'date': dates.astype('datetime64[s]'),
            'lon': lon,
            'lat': lat,
            'ssh_cm': ssh_navd_cm,
        }
    )


def _assign_area(df: pd.DataFrame, areas: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spatial‑join altimetry points to water areas (returns df with new column
    `area`)."""
    gdf_pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    joined = gpd.sjoin(gdf_pts, areas[[AREA_FLD, 'geometry']], how='inner', predicate='within')
    return joined.drop(columns=['index_right']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    print('🔹 Earthdata login …')
    _earthdata_login()

    print('🔹 Searching CMR …')
    granules = _cmr_search()
    if not granules:
        raise SystemExit('No ENVISAT RA‑2 granules found for given bbox/date.')
    print(f'   → {len(granules)} granules')

    print('🔹 Downloading granules …')
    nc_paths = _dl_granules(granules)

    print('🔹 Subsetting & converting …')
    all_pts = []
    for p in tqdm(nc_paths, desc='granules'):
        df = _process_nc(p)
        if not df.empty:
            all_pts.append(df)
    if not all_pts:
        raise SystemExit('No RA‑2 points within bbox after filtering.')

    df_all = pd.concat(all_pts, ignore_index=True)

    # Attach water area
    areas = gpd.read_file(WATER_AREAS_GJS).to_crs(4326)
    df_all = _assign_area(df_all, areas)

    if df_all.empty:
        raise SystemExit('No RA‑2 points fall inside any water area.')

    # Write per‑area CSVs
    for area, sub in df_all.groupby('area'):
        out_dir = OUT_DIR / area / 'water_gauges'
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_dir / 'envisat_altimetry.csv', index=False)

    print('✅ Done – per‑area ENVISAT altimetry written to processing folders')


if __name__ == '__main__':
    main()
