# ──────────────────────────────────────────────────────────────────────────────
# Build a minimal stripmapApp.xml that ISCE2 understands (ALOS / ALOS_SLC)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

# ───────────────────────────── paths ─────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[1]
RAW_DIR  = BASE / "data" / "raw"
DEM_WGS84 = BASE / "data" / "aux" / "dem" / "srtm_30m.dem.wgs84"

# ──────────────────── CEOS helper dataclass ─────────────────────────────────
@dataclass(frozen=True)
class CeosSet:
    leader: Path
    image : Path          # only HH image is passed

# ─────────────────────── helper functions ───────────────────────────────────
def ceos_files(path: str, date: str) -> CeosSet:
    folder   = RAW_DIR / f"path{path}" / date
    led_file = next(folder.glob("LED-*"))
    img_file = next(folder.glob("IMG-HH-*"))
    return CeosSet(led_file, img_file)

def detect_beam_mode(summary_txt: Path) -> str:
    """
    Function detecting the beam mode: FBD or FBS. 
    """
    try:
        txt = summary_txt.read_text()
    except FileNotFoundError:
        return "FBS"
    return "FBD" if "IMG-HV" in txt else "FBS"

# ───────────────────── XML building helpers ────────────────────────────────
def sensor_component(tag: str, files: CeosSet, out_dir: Path, mode: str) -> ET.Element:
    """
    Build either the <component name="Reference"> … </component> or <Secondary> block.
    If beam mode is FBD, add RESAMPLE_FLAG=dual2single so ISCE2 converts dual-pol SLC
    to single-pol before further processing.
    """
    comp = ET.Element("component", name=tag)
    ET.SubElement(comp, "property", name="IMAGEFILE").text   = str(files.image)
    ET.SubElement(comp, "property", name="LEADERFILE").text  = str(files.leader)
    ET.SubElement(comp, "property", name="OUTPUT").text      = str(out_dir)
    
    if mode == "FBD":
        ET.SubElement(comp, "property", name="RESAMPLE_FLAG").text = "dual2single" ## Only possible for 1.0 level data
    
    return comp

# ───────────────────────── main API ─────────────────────────────────────────
def write_stripmap_xml(
    *,                 # force keyword args for clarity
    xml_file : Path,
    path     : str,
    ref_date : str,
    sec_date : str,
) -> None:
    """Create stripmapApp.xml for one interferometric pair."""
    work_dir = xml_file.parent
    ref_out, sec_out = work_dir / ref_date, work_dir / sec_date
    ref_out.mkdir(parents=True, exist_ok=True)
    sec_out.mkdir(parents=True, exist_ok=True)

    # input CEOS sets & beam modes
    ref_files = ceos_files(path, ref_date)
    sec_files = ceos_files(path, sec_date)
    ref_mode  = detect_beam_mode(ref_files.image.parent / "summary.txt")
    sec_mode  = detect_beam_mode(sec_files.image.parent / "summary.txt")

    # assemble XML
    root = ET.Element("stripmapApp")
    app  = ET.SubElement(root, "component", name="stripmapApp")
    ET.SubElement(app, "property", name="SENSORNAME").text  = "ALOS_SLC"
    ET.SubElement(app, "property", name="DEMFILENAME").text = str(DEM_WGS84)
    ET.SubElement(app, "property", name="regionOfInterest").text = str([25.00, 26.70, -81.80, -80.15])      # be sure to give EPSG 4326 (WGS 84) coordinates
    ET.SubElement(app, "property", name="unwrapper name").text = "snaphu"
    
    
    ##------------------------------== Tuning of parameters for the model to run --------------------------------------------------------------------
    
    # Different algorithms handle low-coherence gaps differently.
    ET.SubElement(app, "property", name="do unwrap 2 stage").text = str(True)
    ET.SubElement(app, "property", name="unwrapper 2stage name").text = "MCF"        	                # Available: MCF, REDARC0, REDARC1, REDARC2
    
    # Controls ENL (noise) ↔ spatial resolution trade-off before filtering / unwrapping.
    # Start: range = 4, az = 9 for L-band stripmap → ~30 m ground pix. Raise both if coherence is still < 0.2 in flats (e.g. 6×13).
    # Lower if details (urban, coastlines) are getting smeared and mean γ > 0.5.
    ET.SubElement(app, "property", name="Range Looks").text = str(2)
    ET.SubElement(app, "property", name="Azimuth looks").text = str(5)
    
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
    ET.SubElement(app, "property", name="filter strength").text = str(0.6)                                       # 0.5 is default
    
    ## Dense-offset grid
    # Size of the FFT cross-correlation match window in azimuth × range (SLC pixels). Larger window → higher SNR, lower spatial detail.
    ET.SubElement(app, "property", name="dense window width").text = str(128)                                    # 64 is default
    ET.SubElement(app, "property", name="dense window height").text = str(128) 
    
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