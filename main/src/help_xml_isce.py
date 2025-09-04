#!/usr/bin/env python3
"""
help_xml_isce.py
================

Create a minimal but robust **stripmapApp.xml** for ISCE2 (ALOS stripmap).

What this module provides
-------------------------
- `ceos_files(...)` — locate CEOS **LED-*** (leader) and **IMG-HH-*** (image) files
  under a conventional raw directory layout: `<raw_dir>/path<PATH>/<DATE>/…`.
- `detect_beam_mode(...)` — best-effort detection of **FBD** vs **FBS** beam mode
  from a small metadata text file found in the acquisition folder (returns `"FBD"`,
  `"FBS"`, or `None`).
- `sensor_component(...)` — build the `<component name="Reference|Secondary">`
  block, optionally adding `RESAMPLE_FLAG=dual2single` when beam mode is **FBD**
  (dual-pol to single-pol resampling for Level-1.0 data).
- `write_stripmap_xml(...)` — the main entry point: writes a complete
  `stripmapApp.xml` for one (REF, SEC) pair, wiring all inputs (CEOS files,
  DEM, multilooks, filtering, split-spectrum, etc.).

Assumptions
-----------
- **ALOS** stripmap CEOS data laid out as: `<raw_dir>/path<PATH>/<YYYYMMDD>/…`
- Only **HH** channel is processed (IMG-HH-* files).
- ISCE2 will run `stripmapApp.py` against the XML this module writes.

Typical usage
-------------
```python
from pathlib import Path
from help_xml_isce import write_stripmap_xml

write_stripmap_xml(
    xml_file=Path("/out/stripmapApp.xml"),
    path="150",
    ref_date="20071216",
    sec_date="20080131",
    raw_dir=Path("/mnt/DATA2/bakke326l/raw"),
    work_dir=Path("/mnt/DATA2/bakke326l/processing/interferograms/path150_20071216_20080131_SRTM"),
    dem_wgs84=Path("/home/bakke326l/InSAR/main/data/aux/dem/srtm_30m.dem.wgs84"),
    range_looks=10, az_looks=16, filter_strength=0.6,
)
Notes
---------
The function writes a concise XML with sensible defaults that match the rest of
this pipeline (split-spectrum enabled for ionosphere handling, 30 m posting,
dense offsets on, two-stage unwrap enabled, Goldstein filter strength provided
by caller, etc.). The regionOfInterest is kept as a fixed WGS-84 bbox in
this helper—adjust as needed for other scenes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import math

# ──────────────────── CEOS helper dataclass ─────────────────────────────────
@dataclass(frozen=True)
class CeosSet:
    leader: list[Path]
    image : list[Path]          # only HH image is passed

# ─────────────────────── helper functions ───────────────────────────────────
def ceos_files(path: str, date: str, raw_dir: Path) -> CeosSet:
    folder = raw_dir / f"path{path}" / date
    leaders = sorted(folder.rglob("LED-*"))  # recurse into subdirs
    images  = sorted(folder.rglob("IMG-HH-*"))
    
    if not leaders or not images:
        raise FileNotFoundError(f"No CEOS files found in {folder}")
    if len(leaders) != len(images):
        raise RuntimeError(f"Unequal LED/IMG count in {folder}\nLEDs: {len(leaders)}, IMGs: {len(images)}")
    
    return CeosSet(leaders, images)

def detect_beam_mode(date_folder: Path) -> str:
    """
    Function detecting the beam mode: FBD or FBS. 
    """
    meta_txt = next(date_folder.glob("*.txt"), None)
    if meta_txt is None:
        print(" NO meta text file found") 
        
    txt = meta_txt.read_text(errors="ignore")
    if "FBD" in txt:
        return "FBD"
    if "FBS" in txt:
        return "FBS"
    else:
        print(f" ⚠️ NO beam mode found in txt file: {meta_txt}")
            

# ───────────────────── XML building helpers ────────────────────────────────
def sensor_component(tag: str, files: CeosSet, out_dir: Path, mode: str) -> ET.Element:
    """
    Build either the <component name="Reference"> … </component> or <Secondary> block.
    If beam mode is FBD, add RESAMPLE_FLAG=dual2single so ISCE2 converts dual-pol SLC
    to single-pol before further processing.
    """
    comp = ET.Element("component", name=tag)
    
    img_list = [str(p) for p in files.image]
    led_list = [str(p) for p in files.leader]
    
    if len(img_list) == 1:
        ET.SubElement(comp, "property", name="IMAGEFILE").text  = img_list[0]
        ET.SubElement(comp, "property", name="LEADERFILE").text = led_list[0]
    else:
        ET.SubElement(comp, "property", name="IMAGEFILE").text  = str(img_list)
        ET.SubElement(comp, "property", name="LEADERFILE").text = str(led_list)
        
    if mode == "FBD":
        ET.SubElement(comp, "property", name="RESAMPLE_FLAG").text = "dual2single" ## Only possible for 1.0 level data    
            
    ET.SubElement(comp, "property", name="OUTPUT").text      = str(out_dir)
    
    return comp

# ───────────────────────── main API ─────────────────────────────────────────
def write_stripmap_xml(
    *,                 # force keyword args for clarity
    xml_file : Path,
    path     : str,
    ref_date : str,
    sec_date : str,
    raw_dir  : Path,
    work_dir : Path,
    dem_wgs84: Path,
    range_looks:      int,
    az_looks:         int,
    filter_strength: float
) -> None:
    """Create stripmapApp.xml for one interferometric pair."""
    ref_out, sec_out = work_dir / ref_date, work_dir / sec_date
    ref_out.mkdir(parents=True, exist_ok=True)
    sec_out.mkdir(parents=True, exist_ok=True)
    
    ref_date_dir = raw_dir / f"path{path}" / ref_date
    sec_date_dir = raw_dir / f"path{path}" / sec_date

    # input CEOS sets & beam modes
    ref_files = ceos_files(path, ref_date, raw_dir)
    sec_files = ceos_files(path, sec_date, raw_dir)
    
    ref_mode  = detect_beam_mode(ref_date_dir)
    sec_mode  = detect_beam_mode(sec_date_dir)

    # assemble XML
    root = ET.Element("stripmapApp")
    app  = ET.SubElement(root, "component", name="stripmapApp")
    ET.SubElement(app, "property", name="SENSORNAME").text  = "ALOS"
    ET.SubElement(app, "property", name="DEMFILENAME").text = str(dem_wgs84)
    ET.SubElement(app, "property", name="regionOfInterest").text = str([25.1, 26.7, -81.4, -80.2])     # be sure to give EPSG 4326 (WGS 84) coordinates
    ET.SubElement(app, "property", name="unwrapper name").text = "snaphu"
    
    
    ##------------------------------== Tuning of parameters for the model to run --------------------------------------------------------------------
    
    # Different algorithms handle low-coherence gaps differently.
    ET.SubElement(app, "property", name="do unwrap 2 stage").text = str(True)
    ET.SubElement(app, "property", name="unwrapper 2stage name").text = "MCF"        	                # Available: MCF, REDARC0, REDARC1, REDARC2
    
    # Controls ENL (noise) ↔ spatial resolution trade-off before filtering / unwrapping.
    # Start: range = 4, az = 9 for L-band stripmap → ~30 m ground pix. Raise both if coherence is still < 0.2 in flats (e.g. 6×13).
    # Lower if details (urban, coastlines) are getting smeared and mean γ > 0.5.
    ET.SubElement(app, "property", name="Range Looks").text = str(range_looks)
    ET.SubElement(app, "property", name="Azimuth looks").text = str(az_looks)
    
    # Required for rubber-sheeting & iono tropospheric split.
    ET.SubElement(app, "property", name="do DenseOffsets").text = str(True)                             # Needed for dense offsets and split spectrum
    
    # L-band needs it to remove long-wavelength iono phase.
    ET.SubElement(app, "property", name="do split spectrum").text = str(True)                           # Does splitting needed for iono correction
    ET.SubElement(app, "property", name="do dispersive").text = str(True)
    
    
    # ET.SubElement(app, "property", name="patch size").text = 2000
    # ET.SubElement(app, "property", name="overlap").text = 200
    
    # Defines pixel spacing after geocode / multilook.
    ET.SubElement(app, "property", name="posting").text = str(30)                                                # 30 m, matches DEM
    
    # Removes residual geometric ramps before making γ; critical when topo mis-modelled.
    # Enable range sheeting when steep relief OR burst-edge ramps appear. Usually keep azimuth = False unless vessel wake-like streaks persist.
    ET.SubElement(app, "property", name="do rubbersheetingAzimuth").text = str(True)
    ET.SubElement(app, "property", name="do rubbersheetingRange").text = str(True)
    
    # Suppresses phase noise prior to unwrapping; too strong ⇒ loss of details.
    ET.SubElement(app, "property", name="filter strength").text = str(filter_strength)                                       # 0.5 is default
    
    ## Dense-offset grid
    # Size of the FFT cross-correlation match window in azimuth × range (SLC pixels). Larger window → higher SNR, lower spatial detail.
    ET.SubElement(app, "property", name="dense window width").text = str(64)                                    # 64 is default
    ET.SubElement(app, "property", name="dense window height").text = str(64) 
    
    # Search “pull-in” around the window centre, i.e. the largest offset (in pixels) the matcher will look for.
    ET.SubElement(app, "property", name="dense search width").text = str(40)
    ET.SubElement(app, "property", name="dense search height").text = str(40)
    
    # Grid spacing (skip) between windows. 32-pixel skip on an L–band stripmap (≈ 4.7 m slant-range pixel) ⇒ ~150 m between tie-points.
    ET.SubElement(app, "property", name="dense skip width").text = str(32)                                       # 32 is default
    ET.SubElement(app, "property", name="dense skip height").text = str(32)
    
    
    ## ------------------------------------ End of tuning -----------------------------------------------------------------------------
    # reference / secondary
    app.append(sensor_component("Reference", ref_files, ref_out, ref_mode))
    app.append(sensor_component("Secondary", sec_files, sec_out, sec_mode))

    # pretty-print & write
    ET.indent(root)
    xml_file.write_text(ET.tostring(root, encoding="unicode"))