# ──────────────────────────────────────────────────────────────────────────────
# src/isce_xml.py
#
# Build a minimal stripmapApp.xml that ISCE2 understands (ALOS edition)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

# ────────────────────────────── paths ─────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[1]
RAW_DIR  = BASE / "data" / "raw"
DEM_TIF  = BASE / "data" / "aux" / "dem" / "srtm_30m.tif"

# ────────────────────────── tiny helper dataclass ────────────────────────────
@dataclass(frozen=True)
class CeosSet:
    """Container holding the CEOS leader/image files for one acquisition."""
    leader: Path
    image:  Path

# ───────────────────────── helper functions ──────────────────────────────────
def ceos_files(path: str, date: str) -> CeosSet:
    """
    Return the LED- and IMG-file paths for a given ALOS scene, e.g.
      data/raw/path465/20081006/LED-… , IMG-HH-…
    """
    folder   = RAW_DIR / f"path{path}" / date
    led_file = next(folder.glob("LED-*"))
    hh_file = next(folder.glob("IMG-HH-*"))

    return CeosSet(led_file, hh_file)   # always pass HH **only**

def detect_beam_mode(summary_txt: Path) -> str:
    """
    Read *summary.txt* and decide whether the acquisition is
    FBD (dual-pol) or FBS (single-pol).
    """
    try:
        content = summary_txt.read_text()
    except FileNotFoundError:
        return "FBS"          # fall-back if there is no summary
    return "FBD" if "IMG-HV" in content else "FBS"


# ───────────────────────── XML building helpers ──────────────────────────────
def sensor_component(
    tag: str, 
    files: CeosSet,
    out_dir: Path, 
    mode: str
    ) -> ET.Element:
    """
    Build either the <component name="Reference"> … </component> or the
    <component name="Secondary"> … </component> block.
    """
    print(f"📄 Writing component for {tag}")
    print(f"    IMAGE:  {files.image}")
    print(f"    LEADER: {files.leader}")
    print(f"    OUTPUT: {out_dir}")
    
    comp = ET.Element("component", name=tag)
    ET.SubElement(comp, "property", name="IMAGEFILE").text  = str(files.image)
    ET.SubElement(comp, "property", name="LEADERFILE").text = str(files.leader)
    ET.SubElement(comp, "property", name="OUTPUT").text     = str(out_dir)
    

    # FBD → FBS resampling switch
    if mode == "FBD":
        ET.SubElement(comp, "property", name="RESAMPLE_FLAG").text = "dual2single"

    return comp


# ───────────────────────── main public function ──────────────────────────────
def write_stripmap_xml(
    xml_file: Path,
    path: str,
    ref_date: str,
    sec_date: str,
) -> None:
    """
    Create a *stripmapApp.xml* suitable for `stripmapApp.py --xml …`.
    Outputs are created under xml_file.parent/ref_date and /sec_date.
    """
    work_dir = xml_file.parent
    ref_out  = work_dir / ref_date
    sec_out  = work_dir / sec_date
    ref_out.mkdir(parents=True, exist_ok=True)
    sec_out.mkdir(parents=True, exist_ok=True)

    # gather input file paths & modes
    ref_folder = RAW_DIR / f"path{path}" / ref_date
    sec_folder = RAW_DIR / f"path{path}" / sec_date

    ref_files = ceos_files(path, ref_date)
    sec_files = ceos_files(path, sec_date)

    ref_mode = detect_beam_mode(ref_folder / "summary.txt")
    sec_mode = detect_beam_mode(sec_folder / "summary.txt")

    # assemble XML tree
    root = ET.Element("stripmapApp")
    app  = ET.SubElement(root, "component", name="stripmapApp")

    # global properties
    ET.SubElement(app, "property", name="SENSORNAME").text  = "ALOS"
    ET.SubElement(app, "property", name="DEMFILENAME").text  = str(DEM_TIF)

    # reference & secondary blocks
    app.append(sensor_component("Reference", ref_files, ref_out, ref_mode))
    app.append(sensor_component("Secondary", sec_files, sec_out, sec_mode))
    
    # formslc = ET.SubElement(app, "component", name="formslc")
    # ET.SubElement(formslc, "property", name="AZIMUTH_PATCH_SIZE").text = "4096"
    # ET.SubElement(formslc, "property", name="NUMBER_VALID_PULSES").text = "2048"

    # pretty-print & write
    ET.indent(root)
    xml_file.write_text(ET.tostring(root, encoding="unicode"))
