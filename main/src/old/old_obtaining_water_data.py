#!/usr/bin/env python
"""
EDEN Water-Level PreProcessor
==============================================================================

Purpose
-------
Prepare EDEN water-level time-series for **stratified calibration / validation**
experiments (strata = water-area polygons such as WCA1, WCA3AS, …).

For every area we:
1. Merge the wide EDEN CSV (water level in cm, NAVD88) with gauge attributes /
   geometry from `gauge_locations.geojson`.
2. Spatial-join gauges into `water_areas.geojson` (EPSG 4326) **and** keep only
   physical types *Marsh*, *Forest*, or *River*.
3. Retain ALOS PALSAR acquisitiondate columns listed in
   `acquisition_dates.csv`.
4. Produce **nested calibration sets** (60% ⊇ 30% ⊇ 15% ⊇ 1-gauge) while
   holding back a common 40% validation set.
5. Write CSVs into `processing/<AREA>/water_gauges/` (see layout below).

```
processing/<AREA>/
    water_gauges/
        eden_val_40.csv   #     40% validation (fixed)
        eden_calib_60.csv #     60% calibration
        eden_calib_30.csv #     30% within  60%
        eden_calib_15.csv #     15% within  30%
        eden_calib_1pa.csv#     1   within  15%  (centroid-closest)
        eden_metadata.csv # attributes per gauge
    interferogram/        # <- clips live here later
```

Usage
-----
```
$ python eden_data_prepare.py
```

"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

# ---------------------------------------------------------------------------
# Configuration – tweak these paths if your repo layout changes
# ---------------------------------------------------------------------------
BASE_DIR        = Path('/home/bakke326l/InSAR/main')
EDEN_CSV        = BASE_DIR / 'data/aux/gauges/eden_water_levels.csv'
ACQ_CSV         = BASE_DIR / 'data/aux/gauges/acquisition_dates.csv'
GAUGE_GEOJSON   = BASE_DIR / 'data/vector/gauge_locations.geojson'
WATER_AREAS_GJS = BASE_DIR / 'data/vector/water_areas.geojson'
OUT_DIR         = BASE_DIR / 'processing' 

# Split design
VALID_FRAC      = 0.40                 # 40 % validation (fixed across runs)
CALIB_FRACS     = (0.60, 0.30, 0.15)   # nested calibration fracs (of total)
RANDOM_SEED     = 42                   # reproducible splits for >1 gauge sets

# Acceptable physical station types (column in gauge GeoJSON)
TYPE_FLD    = 'Type of Station (Physical Location)'
EDEN_FLD    = 'EDEN Station Name'
VALID_TYPES = {'marsh', 'forest', 'river'}  # compare in lower‑case

# ---------------------------------------------------------------------------
# Helper routines
# ---------------------------------------------------------------------------

def _load_eden_csv(csv_path: Path) -> pd.DataFrame:
    """Read the wide EDEN CSV and return a tidy DataFrame."""
    df = pd.read_csv(csv_path, dtype={'StationID': str})
    df['StationID'] = df['StationID'].str.strip()
    return df


def _enrich_and_filter_gauges(eden_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Merge EDEN with GeoJSON attributes.

    Returns a GeoDataFrame indexed by **StationID** with columns:
    StationID, area, geometry, Lat, Lon, … (plus all date cols).
    """
    # 1) Read gauge attributes & build geometry if missing
    gauges = gpd.read_file(GAUGE_GEOJSON).to_crs(4326)
    gauges['StationID'] = gauges[EDEN_FLD].str.strip()

    merged = pd.merge(eden_df, gauges.drop(columns='geometry'), on='StationID', how='left')

    geom = []
    for _, row in merged.iterrows():
        g = row.get('geometry')
        if isinstance(g, Point):
            geom.append(g)
        else:
            try:
                lat, lon = float(row['Lat']), float(row['Lon'])
                geom.append(Point(lon, lat))
            except Exception:
                geom.append(None)
    merged['geometry'] = geom
    gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs=4326)

    # 2) Spatial join into water‑areas + drop non‑Marsh/Forest/River
    areas = gpd.read_file(WATER_AREAS_GJS).to_crs(4326)[['area', 'geometry']]
    gdf = gpd.sjoin(gdf, areas, how='inner', predicate='within').drop(columns='index_right')

    mask = gdf[TYPE_FLD].str.lower().isin(VALID_TYPES)
    gdf = gdf[mask].reset_index(drop=True)
    return gdf


def _pick_centroid_gauge(area_poly, gauges: gpd.GeoDataFrame) -> str:
    """Return StationID of gauge whose point is closest to the polygon centroid."""
    centroid = area_poly.centroid
    # Find nearest gauge point to centroid
    distances = gauges.geometry.distance(centroid)
    return gauges.loc[distances.idxmin(), 'StationID']


def _nested_splits(gdf: gpd.GeoDataFrame, areas_gdf: gpd.GeoDataFrame) -> dict[str, list[str]]:
    """Generate validation + nested calibration splits (stratified).

    * Validation (40 %) is random (seeded) per area.
    * calib_60 / 30 / 15 are nested random subsets of the training pool.
    * calib_1pa picks the centroid-closest gauge **from the training pool**  if
      the centroid gauge landed in validation, we swap it with the nearest
      training gauge to keep uniqueness.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    splits = {k: [] for k in ('val_40', 'calib_60', 'calib_30', 'calib_15', 'calib_1pa')}

    # Pre‑index areas for centroid lookup
    poly_by_area = areas_gdf.set_index('area')['geometry']

    for area, group in gdf.groupby('area'):
        ids = group['StationID'].tolist()
        rng.shuffle(ids)
        n = len(ids)

        n_val = max(1, math.ceil(n * VALID_FRAC))
        val_ids   = ids[:n_val]
        train_ids = ids[n_val:]

        # Add val & calib60 first
        splits['val_40'].extend(val_ids)
        splits['calib_60'].extend(train_ids)

        # Nested random subsets from training pool
        n30 = max(1, math.ceil(n * CALIB_FRACS[1]))
        n15 = max(1, math.ceil(n * CALIB_FRACS[2]))
        splits['calib_30'].extend(train_ids[:n30])
        splits['calib_15'].extend(train_ids[:n15])

        # Determine centroid gauge
        centroid_sid = _pick_centroid_gauge(poly_by_area[area], group)
        if centroid_sid in val_ids:
            # Swap: move centroid into training, replace with first training id
            swap_id = train_ids[0] if train_ids else centroid_sid
            val_ids[val_ids.index(centroid_sid)] = swap_id
            train_ids[train_ids.index(swap_id)] = centroid_sid
            # Update previously stored lists
            splits['val_40'][-1] = swap_id
            splits['calib_60'][splits['calib_60'].index(swap_id)] = centroid_sid
            if swap_id in splits['calib_30']:
                splits['calib_30'][splits['calib_30'].index(swap_id)] = centroid_sid
            if swap_id in splits['calib_15']:
                splits['calib_15'][splits['calib_15'].index(swap_id)] = centroid_sid

        splits['calib_1pa'].append(centroid_sid)

    # Deduplicate while preserving order
    for key, lst in splits.items():
        seen = set()
        splits[key] = [x for x in lst if not (x in seen or seen.add(x))]
    return splits


def _export_by_area(df_wl: pd.DataFrame, gdf: gpd.GeoDataFrame, splits: dict[str, list[str]]):
    """Write CSVs under processing/<AREA>/water_gauges/."""
    subset_files = {
        'val_40':   'eden_val_40.csv',
        'calib_60': 'eden_calib_60.csv',
        'calib_30': 'eden_calib_30.csv',
        'calib_15': 'eden_calib_15.csv',
        'calib_1pa':'eden_calib_1pa.csv',
    }

    for area, area_ids in gdf.groupby('area')['StationID']:
        area_dir = OUT_DIR / 'areas' / area / 'water_gauges'
        area_dir.mkdir(parents=True, exist_ok=True)

        # Save each subset with columns belonging to this area only
        for split_key, filename in subset_files.items():
            col_ids = [sid for sid in splits[split_key] if sid in area_ids.values]
            if not col_ids:
                continue
            df_wl[col_ids].to_csv(area_dir / filename, index_label='date')

        # Metadata for gauges in this area
        meta_cols = ['StationID', 'area', TYPE_FLD, 'Lat', 'Lon']
        gdf[gdf['area'] == area][meta_cols].to_csv(area_dir / 'eden_metadata.csv', index=False)

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    print('🔹 Reading EDEN CSV …')
    eden_df = _load_eden_csv(EDEN_CSV)

    print('🔹 Reading acquisition dates …')
    acq_dates = pd.read_csv(ACQ_CSV)['date'].astype(str).tolist()
    date_cols = [pd.to_datetime(d, format='%Y%m%d').strftime('%Y-%m-%d') for d in acq_dates]

    # Merge EDEN table with GIS, attach areas, type filter
    print('🔹 Merging & filtering gauges …')
    gdf = _enrich_and_filter_gauges(eden_df)
    print(f'   → {len(gdf)} gauges after area/type filter')
    if gdf.empty:
        raise SystemExit('No gauges available - aborting.')

    # Wide water‑level DataFrame (index=date , columns=StationID)
    df_wl = gdf[['StationID'] + date_cols].set_index('StationID').T

    # Build splits (needs areas_gdf for centroid)
    areas_gdf = gpd.read_file(WATER_AREAS_GJS).to_crs(4326)[['area', 'geometry']]
    print('🔹 Generating stratified splits …')
    splits = _nested_splits(gdf, areas_gdf)

    print('🔹 Writing per-area CSV files …')
    _export_by_area(df_wl, gdf, splits)
    print('✅ Done! Outputs in processing/<AREA>/water_gauges/')


if __name__ == '__main__':
    main()
