#!/usr/bin/env python
"""
Fill area folders with gauge tables + per-area vertical displacement clips (TIFF + PNG)
======================================================================================

What it does
------------
1) Builds per-area EDEN gauge tables:
   /mnt/DATA2/bakke326l/processing/areas/<AREA>/water_gauges/
       eden_gauges.csv
       eden_metadata.csv

2) Finds vertical displacement rasters produced by your pipeline:
   <pair_dir>/inspect/vertical_displacement_cm_<REF>_<SEC>_{RAW|TROPO|IONO|TROPO_IONO}.geo.tif

   For each raster, clips it by each water area and writes (if coverage passes):
   /mnt/DATA2/bakke326l/processing/areas/<AREA>/
       <AREA>_vertical_cm_<REF_SEC>_<DEM>_<CORR>.tif
       <AREA>_vertical_cm_<REF_SEC>_<DEM>_<CORR>.png
   where <DEM> is parsed from the pair directory name suffix: _SRTM or _3DEP
   and <CORR> is one of RAW, TROPO, IONO, TROPO_IONO.

3) Writes a coverage report for everything processed:
   /mnt/DATA2/bakke326l/processing/areas/_reports/coverage_report.csv

Run examples
------------
# Global search (default VERT_ROOT) for */inspect/vertical_displacement_cm_*.geo.tif
python 4_organize_areas.py

# One file
python 4_organize_areas.py \
  --vertical-file /mnt/DATA2/bakke326l/processing/interferograms/path150_20071216_20080131_SRTM/inspect/vertical_displacement_cm_20071216_20080131_RAW.geo.tif

# One pair directory (searches pair_dir/inspect/)
python 4_organize_areas.py \
  --pair-dir /mnt/DATA2/bakke326l/processing/interferograms/path150_20071216_20080131_SRTM

# Tweak coverage threshold
python 4_organize_areas.py --min-coverage-pct 50.0
"""

from __future__ import annotations

# --- silence DEBUG spam ---
import os, logging
os.environ.setdefault("CPL_DEBUG", "OFF")
os.environ.setdefault("CPL_LOG", "/dev/null")
os.environ.setdefault("RIO_LOG_LEVEL", "CRITICAL")
logging.basicConfig(level=logging.INFO, force=True)
for name in ("rasterio", "matplotlib", "fiona", "shapely", "geopandas"):
    logging.getLogger(name).setLevel(logging.WARNING)


import argparse
import re
from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import mask as rio_mask
from rasterio import features as rio_features
import matplotlib.pyplot as plt
from matplotlib import colormaps
from shapely.geometry import Point, shape as shp_shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.errors import GEOSException


# -----------------------------------------------------------------------------
# Configuration — tweak these paths if your repo layout changes
# -----------------------------------------------------------------------------
BASE_DIR        = Path('/home/bakke326l/InSAR/main')

# Input tables / vectors
EDEN_CSV        = BASE_DIR / 'data/aux/gauges/eden_water_levels.csv'     # wide daily time series
ACQ_CSV         = BASE_DIR / 'data/aux/gauges/acquisition_dates.csv'     # yyyymmdd in 'date' column
GAUGE_GEOJSON   = BASE_DIR / 'data/vector/gauge_locations.geojson'       # point features (WGS84)
WATER_AREAS_GJS = BASE_DIR / 'data/vector/water_areas.geojson'           # polygons with 'area' field (WGS84)

# Output directory (per-area clips + gauge tables)
OUT_DIR         = Path('/mnt/DATA2/bakke326l/processing/areas')

# Where to SEARCH for vertical displacement rasters to clip
# (matches the new pipeline: .../processing/interferograms/pathXXX_..._{SRTM|3DEP}/inspect/)
VERT_ROOT       = Path('/mnt/DATA2/bakke326l/processing/interferograms')
VERT_PATTERN    = 'vertical_displacement_cm_*.geo.tif'   # inside */inspect/

# Optional type filter (if field exists in gauge attributes)
TYPE_FLD    = 'Type of Station (Physical Location)'
EDEN_FLD    = 'EDEN Station Name'   # used as StationID fallback if needed
VALID_TYPES = {'marsh', 'forest', 'river'}

# Varibales for correction the water level to water elevation 
EDEN_GROUND_TXT      = BASE_DIR / 'data/aux/gauges/eden_ground_elevation.txt'
EDEN_WATER_ELEV_CSV  = BASE_DIR / 'data/aux/gauges/eden_water_elevation.csv'
FT_TO_CM             = 30.48

# -----------------------------------------------------------------------------
# Small helpers (simple & explicit)
# -----------------------------------------------------------------------------
def _save_png(png_path: Path, arr_cm: np.ndarray, title: str) -> None:
    """Quick-look PNG"""
    m = np.ma.masked_invalid(arr_cm)  # mask NaNs
    plt.figure(figsize=(7, 6), dpi=150)
    finite = m.compressed()
    vmin, vmax = (0.0, 1.0) if finite.size == 0 else (float(np.nanpercentile(finite, 2)),
                                                      float(np.nanpercentile(finite, 98)))
    cmap = colormaps.get_cmap('viridis').copy()
    cmap.set_bad('white', alpha=0)  # NaNs transparent
    im = plt.imshow(m, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title(title, fontsize=10)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Vertical displacement [cm]')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()


def _write_geotiff(path: Path, data: np.ndarray, ref_profile, transform) -> None:
    """Write a single-band float32 GeoTIFF with nodata=NaN, DEFLATE compression."""
    if data is None:
        return
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError(f"need 1 band, got {data.shape}")
        data = data[0]
    if data.ndim != 2 or data.size == 0:
        print(f"   • Skipping {path.name}: empty clip {getattr(data, 'shape', None)}")
        return
    prof = ref_profile.copy()
    prof.update(
        driver='GTiff',
        dtype='float32',
        count=1,
        compress='deflate',
        predictor=3,
        transform=transform,
        height=int(data.shape[0]),
        width=int(data.shape[1]),
        nodata=np.nan,
        BIGTIFF='IF_SAFER',
    )
    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(data.astype('float32'), 1)


def _raster_footprint(src: rasterio.DatasetReader):
    """
    Polygonize the raster's valid-data mask (255 = valid). This ensures we only
    claim coverage where the dataset actually has data (not just bbox overlap).
    """
    mask = src.dataset_mask()  # uint8, 255=valid, 0=nodata
    geoms = []
    for geom, val in rio_features.shapes(mask, transform=src.transform):
        if val == 255:
            geoms.append(shp_shape(geom))
    return unary_union(geoms) if geoms else None


def _poly_only(g):
    """Return a valid Polygon/MultiPolygon (or None) from any input geometry."""
    if g is None or g.is_empty:
        return None
    try:
        g = g if g.is_valid else make_valid(g)
    except Exception:
        g = g.buffer(0)
    if g.is_empty:
        return None
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    if g.geom_type == "GeometryCollection":
        polys = [gg for gg in g.geoms if gg.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        return unary_union(polys) if len(polys) > 1 else polys[0]
    return None


def _safe_intersection(a, b):
    """Intersection that tries to repair inputs on topology errors."""
    try:
        return a.intersection(b)
    except GEOSException:
        a2 = _poly_only(a); b2 = _poly_only(b)
        if a2 is None or b2 is None:
            return None
        try:
            return a2.intersection(b2)
        except GEOSException:
            return a2.buffer(0).intersection(b2.buffer(0))


# -----------------------------------------------------------------------------
# Gauge processing and division
# -----------------------------------------------------------------------------
def _load_eden_csv(csv_path: Path) -> pd.DataFrame:
    """Read the wide EDEN CSV and ensure 'StationID' is present and clean."""
    df = pd.read_csv(csv_path, dtype={'StationID': str})
    if 'StationID' not in df.columns:
        print("ERROR: 'StationID' column is required in EDEN CSV.", file=sys.stderr)
        sys.exit(1)
    df['StationID'] = df['StationID'].str.strip()
    return df

def _make_above_ground_csv_same_columns(
    eden_csv: Path,
    ground_txt: Path,
    out_csv: Path,
    ground_col_ft: str = "Average Ground Elevation (ft NAVD88)",
    name_col: str = "EDEN Station Name",
) -> Path:
    """
    Build a corrected CSV with the *same columns* as the source EDEN CSV,
    where every YYYY-MM-DD column is water level *above ground* (cm),
    and keep **only** stations that appear in the ground-elevation file.

    Join key: StationID == 'EDEN Station Name'.
    """
    
    # 1) Read source EDEN levels (wide; cm NAVD88)
    df = pd.read_csv(eden_csv, dtype={"StationID": str})
    if "StationID" not in df.columns:
        raise SystemExit("EDEN CSV must contain 'StationID'.")
    df["StationID"] = df["StationID"].astype(str).str.strip()
    original_cols = list(df.columns)
    total_src = len(df)

    # 2) Read ground elevations (ft NAVD88) and normalize columns
    try:
        elev = pd.read_csv(ground_txt, sep="\t")
    except Exception:
        elev = pd.read_csv(ground_txt)

    canon = {c.strip(): c for c in elev.columns}
    if name_col not in canon or ground_col_ft not in canon:
        raise SystemExit(
            f"Ground-elevation file must contain '{name_col}' and '{ground_col_ft}'."
        )

    elev = elev.rename(columns={
        canon[name_col]: "EDEN_Name",
        canon[ground_col_ft]: "ground_ft",
    })
    elev["EDEN_Name"] = elev["EDEN_Name"].astype(str).str.strip()
    elev["ground_cm"] = round(pd.to_numeric(elev["ground_ft"], errors="coerce") * FT_TO_CM, 3)

    # Ensure unique EDEN_Name to avoid duplicating rows on merge
    if elev["EDEN_Name"].duplicated().any():
        print("⚠️  Duplicate EDEN names in ground table — keeping first occurrence.")
        elev = elev.drop_duplicates(subset=["EDEN_Name"])

    # 3) **INNER JOIN** → drop any gauges not present in ground-elevation table
    df = df.merge(
        elev[["EDEN_Name", "ground_cm"]],
        left_on="StationID",
        right_on="EDEN_Name",
        how="inner",
    )
    matched = len(df)
    dropped = total_src - matched
    print(f"   → Ground elevation matched for {matched}/{total_src} stations "
          f"(dropped {dropped}).")

    if matched == 0:
        raise SystemExit("No stations matched between EDEN CSV and ground-elevation file.")

    # 4) Subtract ground from every date column (YYYY-MM-DD)
    date_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", c)]
    if not date_cols:
        raise SystemExit("No date columns detected (expected YYYY-MM-DD headers).")

    for c in date_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") - df["ground_cm"]

    # 5) Restore original schema: same headers/order as input; drop helper columns
    for hc in ("EDEN_Name", "ground_cm"):
        if hc in df.columns and hc not in original_cols:
            df.drop(columns=hc, inplace=True)

    df_out = df[[c for c in original_cols if c in df.columns]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Wrote water-above-ground (cm) CSV with same headers: {out_csv}")
    return out_csv

def _enrich_and_filter_gauges(eden_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Merge EDEN rows with gauge attributes and geometry, spatial-join into water areas,
    and (if available) filter by physical station type (Marsh/Forest/River).
    """
    gauges = gpd.read_file(GAUGE_GEOJSON).to_crs(4326)
    if 'StationID' not in gauges.columns:
        if EDEN_FLD in gauges.columns:
            gauges['StationID'] = gauges[EDEN_FLD].astype(str).str.strip()
        else:
            print("ERROR: 'gauge_locations.geojson' needs 'StationID' or 'EDEN Station Name'.",
                  file=sys.stderr)
            sys.exit(1)
    gauges['StationID'] = gauges['StationID'].astype(str).str.strip()
    gauges = gauges.drop_duplicates(subset=['StationID'])

    merged = pd.merge(eden_df, gauges.drop(columns='geometry'), on='StationID', how='left')

    # Attach geometry: prefer GeoJSON geometry; fallback to Lat/Lon
    lookup = dict(zip(gauges['StationID'], gauges['geometry']))
    geom = []
    for _, row in merged.iterrows():
        g = lookup.get(row['StationID'])
        if g is not None:
            geom.append(g)
        else:
            try:
                lat = float(row['Lat']); lon = float(row['Lon'])
                geom.append(Point(lon, lat))
            except Exception:
                geom.append(None)

    gdf = gpd.GeoDataFrame(merged, geometry=geom, crs=4326).dropna(subset=['geometry'])

    # Spatial join gauges → water areas (repair areas first)
    areas = gpd.read_file(WATER_AREAS_GJS).to_crs(4326)
    cols_lower = {c.lower(): c for c in areas.columns}
    if 'area' not in cols_lower:
        print(f"ERROR: 'area' column not found in {WATER_AREAS_GJS}.", file=sys.stderr)
        sys.exit(1)
    area_col = cols_lower['area']
    areas = areas[[area_col, 'geometry']].rename(columns={area_col: 'area'})
    areas['geometry'] = areas['geometry'].apply(_poly_only)
    areas = areas.dropna(subset=['geometry']).explode(index_parts=False).reset_index(drop=True)

    gdf = gpd.sjoin(gdf, areas, how='inner', predicate='within').drop(columns='index_right')

    if TYPE_FLD in gdf.columns:
        mask = gdf[TYPE_FLD].astype(str).str.lower().isin(VALID_TYPES)
        before, after = len(gdf), int(mask.sum())
        gdf = gdf[mask].reset_index(drop=True)
        print(f"   → Type filter '{TYPE_FLD}': {before} → {after} gauges kept")
    else:
        print(f"   ⚠️  '{TYPE_FLD}' not found — skipping type filter")

    return gdf


def _export_by_area_simple(gdf: gpd.GeoDataFrame, keep_dates_iso: list[str]) -> None:
    """
    Write per-area CSVs (no subsets):
      /mnt/DATA2/bakke326l/processing/areas/<AREA>/water_gauges/eden_gauges.csv
      /mnt/DATA2/bakke326l/processing/areas/<AREA>/water_gauges/eden_metadata.csv
    """
    present_dates = [d for d in keep_dates_iso if d in gdf.columns]
    if not present_dates:
        print("ERROR: None of the acquisition dates match EDEN CSV column names.", file=sys.stderr)
        sys.exit(1)

    ts_cols   = ['StationID', 'Lat', 'Lon'] + present_dates
    date_set  = set(keep_dates_iso)
    meta_cols = [c for c in gdf.columns if c not in date_set and c != 'geometry']

    gdf = gdf.sort_values(['area', 'StationID']).drop_duplicates(['area', 'StationID'])

    written = 0
    for area, sub in gdf.groupby('area'):
        area_dir = OUT_DIR / area / 'water_gauges'
        area_dir.mkdir(parents=True, exist_ok=True)

        sub[ts_cols].to_csv(area_dir / 'eden_gauges.csv', index=False)

        meta = sub[meta_cols].copy()
        if 'area' not in meta.columns:
            meta['area'] = area
        meta.to_csv(area_dir / 'eden_metadata.csv', index=False)
        written += 1

    print(f"✅ Wrote gauge tables for {written} area(s) under:\n    {OUT_DIR}")


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDEN per-area exports + coverage-aware clipping of vertical displacement rasters (TIFF + PNG)."
    )
    parser.add_argument('--vertical-file', type=str,
                        help='Path to ONE vertical displacement GeoTIFF (centimeters).')
    parser.add_argument('--pair-dir', type=str,
                        help='Path to ONE pair directory; searches pair_dir/inspect/ for rasters.')
    parser.add_argument('--vertical-root', type=str, default=str(VERT_ROOT),
                        help='Root to search recursively for */inspect/vertical_displacement_cm_*.geo.tif (default: global search).')
    parser.add_argument('--min-coverage-pct', type=float, default=65.0,
                        help='Minimum polygon coverage (percent) to write outputs. Default: 50.0')
    args = parser.parse_args()

    # ---- Gauges: read, join, export per area ----
    print('🔹 Computing EDEN water *above ground* (cm) for ALL dates …')
    
    eden_corrected_csv = _make_above_ground_csv_same_columns(
        eden_csv=EDEN_CSV,
        ground_txt=EDEN_GROUND_TXT,
        out_csv=EDEN_WATER_ELEV_CSV
    )

    
    print('🔹 Reading EDEN CSV …')
    eden_df = _load_eden_csv(eden_corrected_csv)

    print('🔹 Reading acquisition dates …')
    acq_df = pd.read_csv(ACQ_CSV)
    if acq_df.empty or acq_df.columns[0].lower() != 'date':
        print("ERROR: acquisition_dates.csv must contain a 'date' column (yyyymmdd).", file=sys.stderr)
        sys.exit(1)
    keep_dates_iso = []
    for raw in acq_df['date'].astype(str):
        raw = raw.strip()
        if len(raw) == 8 and raw.isdigit():
            keep_dates_iso.append(f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}")

    print('🔹 Merging & filtering gauges …')
    gdf = _enrich_and_filter_gauges(eden_df)
    print(f'   → {len(gdf)} gauges after area/type handling')
    if gdf.empty:
        raise SystemExit('No gauges available — aborting.')

    print('🔹 Writing per-area CSV files …')
    _export_by_area_simple(gdf, keep_dates_iso)

    # ---- Collect rasters to clip (single file / one pair / global search) ----
    print('🔹 Collecting vertical displacement rasters …')
    vpaths: list[Path] = []
    if args.vertical_file:
        p = Path(args.vertical_file)
        if p.is_file():
            vpaths = [p]
        else:
            print(f"ERROR: --vertical-file not found: {p}", file=sys.stderr)
            sys.exit(1)
    elif args.pair_dir:
        insp = Path(args.pair_dir) / 'inspect'
        vpaths = sorted(insp.glob(VERT_PATTERN))
    else:
        vroot = Path(args.vertical_root)
        # search all pair dirs for inspect/vertical_displacement_cm_*.geo.tif
        vpaths = sorted(vroot.rglob(f'*/inspect/{VERT_PATTERN}'))

    if not vpaths:
        print("⚠️  No vertical displacement rasters found to clip. Done with gauge CSVs.")
        return

    # Load + repair area polygons once (WGS84)
    areas_wgs84 = gpd.read_file(WATER_AREAS_GJS).to_crs(4326)
    area_col = {c.lower(): c for c in areas_wgs84.columns}.get('area')
    if not area_col:
        print(f"ERROR: 'area' column not found in {WATER_AREAS_GJS}.", file=sys.stderr)
        sys.exit(1)
    areas_wgs84 = areas_wgs84[[area_col, 'geometry']].rename(columns={area_col: 'area'})
    areas_wgs84['geometry'] = areas_wgs84['geometry'].apply(_poly_only)
    areas_wgs84 = areas_wgs84.dropna(subset=['geometry']).explode(index_parts=False).reset_index(drop=True)

    # Coverage report rows will be stored here and written once at the end
    report_rows = []

    # ---- Process each raster with coverage-aware clipping ----
    for vpath in vpaths:
        # Extract dates from filename
        m_dates = re.search(r'(\d{8}_\d{8})', vpath.name)
        dates = m_dates.group(1) if m_dates else 'unknownpair'

        # Extract DEM tag from the pair dir name (parent of 'inspect'): ..._SRTM or ..._3DEP
        pair_dir = vpath.parent.parent  # .../<pair>/inspect/<file>
        m_dem = re.search(r'_(SRTM|3DEP)\b', pair_dir.name, flags=re.IGNORECASE)
        dem_tag = m_dem.group(1).upper() if m_dem else 'DEM'
        
        # Identify which correction this raster came from
        stem = vpath.stem.upper()
        if "TROPO_IONO" in stem:
            corr_tag = "TROPO_IONO"
        elif stem.endswith("_IONO") or "_IONO" in stem:
            corr_tag = "IONO"
        elif stem.endswith("_TROPO") or "_TROPO" in stem:
            corr_tag = "TROPO"
        elif stem.endswith("_RAW") or "_RAW" in stem:
            corr_tag = "RAW"
        else:
            corr_tag = "RAW"  # fallback for legacy files without a suffix

        with rasterio.open(vpath) as src:
            raster_crs = src.crs

            # Build + repair the true data footprint once per raster
            footprint = _poly_only(_raster_footprint(src))
            if footprint is None:
                for _, arow in areas_wgs84.iterrows():
                    report_rows.append({
                        'pair': dates, 'dem': dem_tag, 'corr': corr_tag,'area': arow['area'],
                        'coverage_pct': 0.0, 'valid_pixels': 0,
                        'wrote': False, 'reason': 'no_valid_data_in_raster'
                    })
                print(f"   • {vpath.name}: no valid data in raster mask — skipping.")
                continue

            for _, arow in areas_wgs84.iterrows():
                area = arow['area']
                area_dir = OUT_DIR / area / "interferograms"
                area_dir.mkdir(parents=True, exist_ok=True)

                # Reproject area polygon to raster CRS and repair
                area_geom = gpd.GeoSeries([arow.geometry], crs=4326).to_crs(raster_crs).iloc[0]
                area_geom = _poly_only(area_geom)
                if area_geom is None:
                    report_rows.append({
                        'pair': dates, 'dem': dem_tag, 'corr': corr_tag,'area': area,
                        'coverage_pct': 0.0, 'valid_pixels': 0,
                        'wrote': False, 'reason': 'invalid_area_geom'
                    })
                    continue

                # Robust intersection with raster footprint
                inter = _safe_intersection(area_geom, footprint)
                if inter is None or inter.is_empty or inter.area == 0:
                    report_rows.append({
                        'pair': dates, 'dem': dem_tag, 'corr': corr_tag, 'area': area,
                        'coverage_pct': 0.0, 'valid_pixels': 0,
                        'wrote': False, 'reason': 'no_overlap'
                    })
                    continue

                # % overlap relative to full polygon area (same CRS)
                poly_area = float(area_geom.area) if area_geom.area else 0.0
                coverage_pct = (float(inter.area) / poly_area * 100.0) if poly_area > 0 else 0.0

                # Clip **by the intersection geometry**; out-of-footprint → NaN via nodata
                try:
                    out_image, out_transform = rio_mask.mask(
                        src,
                        shapes=[inter.__geo_interface__],
                        crop=True,
                        filled=True,
                        nodata=np.nan,
                        indexes=[1],          # <<— make it 3-D: (1, rows, cols)
                    )
                except Exception as e:
                    ...
                    continue

                if out_image.size == 0:
                    ...
                    continue

                # Handle both 2-D (rows, cols) and 3-D (1, rows, cols) returns safely
                if out_image.ndim == 3:
                    clipped = out_image[0].astype('float32')
                else:  # 2-D already
                    clipped = out_image.astype('float32')

                # Extra guard: if something still came back 1-D, expand it to a single row
                if clipped.ndim == 1:
                    clipped = clipped.reshape(1, -1)

                # Count finite pixels only (for reporting; not used for threshold)
                valid_pixels = int(np.count_nonzero(np.isfinite(clipped)))

                # Threshold test: ONLY coverage % (still skip if all-NaN)
                if valid_pixels == 0:
                    wrote = False; reason = 'all_nan'
                elif coverage_pct < float(args.min_coverage_pct):
                    wrote = False; reason = 'below_threshold'
                else:
                    # Include DEM and CORR tag so variants don't overwrite one another
                    tif_out = area_dir / f"{area}_vertical_cm_{dates}_{dem_tag}_{corr_tag}.tif"
                    _write_geotiff(tif_out, clipped, src.profile, out_transform)

                    png_out = area_dir / f"{area}_vertical_cm_{dates}_{dem_tag}_{corr_tag}.png"
                    _save_png(png_out, clipped, title=f"{area} • vertical (cm) • {dates} • {dem_tag} • {corr_tag}")

                    wrote = True; reason = 'ok'

                report_rows.append({
                    'pair': dates, 'dem': dem_tag, 'corr': corr_tag, 'area': area,
                    'coverage_pct': round(coverage_pct, 3),
                    'valid_pixels': valid_pixels,
                    'wrote': wrote,
                    'reason': reason
                })

        print(f"   ✓ Processed raster: {vpath}")

    # ---- Write coverage report ----
    reports_dir = OUT_DIR / "_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_csv = reports_dir / "coverage_report.csv"
    pd.DataFrame(report_rows).to_csv(report_csv, index=False)
    print(f"✅ Coverage report written: {report_csv}")

    print('✅ All done.')

if __name__ == '__main__':
    main()
