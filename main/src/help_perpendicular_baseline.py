#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
help_perpendicular_baseline.py
==============================

Compute signed perpendicular baselines (meters) for all ALOS pairs listed in a CSV.

Input
-----
- Pairs CSV (default: /home/bakke326l/InSAR/main/data/pairs.csv)
  Required columns: path, ref_date, sec_date  (dates as YYYYMMDD)
- Interferograms root (default: /mnt/DATA2/bakke326l/processing/interferograms)
  Pair directories look like: path{PATH}_{REF}_{SEC}_{DEM}
  Example: /mnt/DATA2/.../path150_20071216_20080131_3DEP

Inside each pair directory we expect:
  {REF}_raw.xml  and  {SEC}_raw.xml

Output
------
CSV with columns:
  path, ref_date, sec_date, bperp_signed_m, bperp_abs_m
(default: /home/bakke326l/InSAR/main/data/perpendicular_baselines.csv)

Definition (literature-standard)
--------------------------------
Perpendicular baseline is the component of the baseline vector that is **perpendicular
to the line-of-sight (LOS)** at scene center:

  B     = r_sec - r_ref
  û_LOS = unit( -cos(inc)*r̂_ref + sin(inc)*ê_right ),  with ê_right from orbit geometry & squint
  B⊥    = B - (B·û_LOS) û_LOS
  |B⊥|  = perpendicular baseline magnitude
  sign  = + if B⊥ points toward right-looking direction, else -

Notes
-----
- Standard library only.
- If `sensing_mid` is missing, fallback = midpoint of `sensing_start` / `sensing_stop`.
- Linear interpolation of (r,v) between state vectors; clamp to nearest if outside span.
"""

from __future__ import annotations
import argparse, csv, glob, math
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


# ------------------------ small helpers: time & vectors ------------------------

def _parse_time_any(s: str) -> datetime:
    """Parse 'YYYY-MM-DDTHH:MM:SS(.f*)' or 'YYYY-MM-DD HH:MM:SS(.f*)' as UTC."""
    s = s.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Unrecognized datetime: {s}")

def _epoch_s(t: datetime) -> float:
    return (t - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()

def _norm(a): return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
def _unit(a):
    n = _norm(a)
    return (a[0]/n, a[1]/n, a[2]/n) if n > 0 else (0.0, 0.0, 0.0)
def _sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _add(a,b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def _mul(a,s): return (a[0]*s, a[1]*s, a[2]*s)
def _dot(a,b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def _cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def _parse_triplet(txt: str):
    """Parse '[x, y, z]' or 'x, y, z' → (x,y,z) floats."""
    s = txt.strip().strip("[]()")
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 numbers, got: {txt}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


# -------------------- XML reading: state vectors & sensing_mid --------------------

def _xml_prop_text(root: ET.Element, name: str) -> str | None:
    """Return value of <property name='name'><value>...</value></property> anywhere in XML."""
    for p in root.findall(".//property"):
        if p.get("name") == name:
            v = p.findtext("value")
            return v.strip() if v else None
    return None

def _read_state_vectors_and_mid(xml_path: Path):
    """Return (state_vectors, t_mid). state_vectors = list of (t, r, v), sorted."""
    root = ET.parse(xml_path).getroot()

    # sensing_mid or midpoint fallback
    t_mid_txt = _xml_prop_text(root, "sensing_mid")
    if t_mid_txt:
        t_mid = _parse_time_any(t_mid_txt)
    else:
        t1_txt = _xml_prop_text(root, "sensing_start")
        t2_txt = _xml_prop_text(root, "sensing_stop")
        if not (t1_txt and t2_txt):
            raise RuntimeError(f"{xml_path}: missing sensing times.")
        t1 = _parse_time_any(t1_txt)
        t2 = _parse_time_any(t2_txt)
        t_mid = t1 + (t2 - t1)/2

    # orbit → nested state vectors
    orbit = None
    for comp in root.findall(".//component"):
        if comp.get("name") == "orbit":
            orbit = comp; break
    if orbit is None:
        raise RuntimeError(f"{xml_path}: <component name='orbit'> not found.")

    sv = []
    for comp in orbit.findall(".//component"):
        nm = (comp.get("name") or "").lower()
        if "statevector" not in nm:
            continue
        t_txt = _xml_prop_text(comp, "time")
        r_txt = _xml_prop_text(comp, "position")
        v_txt = _xml_prop_text(comp, "velocity")
        if not (t_txt and r_txt and v_txt):
            continue
        sv.append((_parse_time_any(t_txt), _parse_triplet(r_txt), _parse_triplet(v_txt)))

    if not sv:
        raise RuntimeError(f"{xml_path}: no state vectors found.")
    sv.sort(key=lambda x: x[0])
    return sv, t_mid

def _interp_kinematics(sv, t_query: datetime):
    """Linear interpolation of (r,v) at t_query; nearest if outside span."""
    tq = _epoch_s(t_query)
    ts = [_epoch_s(t) for (t,_,_) in sv]
    if tq <= ts[0]:  return sv[0][1], sv[0][2]
    if tq >= ts[-1]: return sv[-1][1], sv[-1][2]
    lo, hi = 0, len(ts)-1
    while hi - lo > 1:
        mid = (lo + hi)//2
        if tq < ts[mid]: hi = mid
        else:            lo = mid
    t0, t1 = ts[lo], ts[hi]
    a = (tq - t0)/(t1 - t0) if t1 != t0 else 0.0
    r0, v0 = sv[lo][1], sv[lo][2]
    r1, v1 = sv[hi][1], sv[hi][2]
    r = (r0[0]*(1-a)+r1[0]*a, r0[1]*(1-a)+r1[1]*a, r0[2]*(1-a)+r1[2]*a)
    v = (v0[0]*(1-a)+v1[0]*a, v0[1]*(1-a)+v1[1]*a, v0[2]*(1-a)+v1[2]*a)
    return r, v


# ------------------- angles & LOS/right-looking construction -------------------

def _read_angles(xml_path: Path):
    """
    Read incidence and squint angles from XML.
    - incidence_angle: degrees in many ISCE ALOS XMLs (e.g., '38.7'); we convert to radians if > 1°
    - squint_angle: often radians near 0; if value > π, treat as degrees and convert.
    Fallbacks: inc=0, squint=0 (shouldn't happen with ALOS frames).
    """
    root = ET.parse(xml_path).getroot()
    # try a few variants for safety
    inc_txt = (_xml_prop_text(root, "incidence_angle")
               or _xml_prop_text(root, "incidenceAngle")
               or "0")
    sq_txt  = (_xml_prop_text(root, "squint_angle")
               or _xml_prop_text(root, "squintAngle")
               or "0")
    inc = float(inc_txt)
    squint = float(sq_txt)
    # degrees → radians (incidence usually in degrees)
    inc = math.radians(inc) if abs(inc) > 1.0 else inc
    # squint: if looks like degrees, convert
    if abs(squint) > math.pi:
        squint = math.radians(squint)
    return inc, squint


# ------------------ baseline for one pair (LOS-perpendicular) ------------------

def baseline_perpendicular(ref_xml: Path, sec_xml: Path):
    """
    Compute (signed, absolute) LOS-perpendicular baseline at sensing_mid.
    Sign is positive if B⊥ points to the right-looking direction.
    """
    ref_sv, ref_mid = _read_state_vectors_and_mid(ref_xml)
    sec_sv, sec_mid = _read_state_vectors_and_mid(sec_xml)

    r_ref, v_ref = _interp_kinematics(ref_sv, ref_mid)
    r_sec, _     = _interp_kinematics(sec_sv, sec_mid)

    # Satellite frame at reference mid-time
    rhat = _unit(r_ref)                  # radial (to satellite)
    vhat = _unit(v_ref)                  # along-track (approx)
    nhat = _unit(_cross(rhat, vhat))     # orbit normal
    that = _unit(_cross(nhat, rhat))     # tangential (along-track, consistent with nhat × rhat)

    # Scene-center angles
    inc, squint = _read_angles(ref_xml)

    # Right-looking unit: rotate nhat toward that by squint
    e_right = _unit(_add(_mul(nhat, math.cos(squint)), _mul(that, math.sin(squint))))

    # LOS (toward Earth) in the plane of -rhat and e_right; tilt from vertical by 'inc'
    u_los = _unit(_add(_mul(rhat, -math.cos(inc)), _mul(e_right, math.sin(inc))))

    # Baseline & projection
    B = _sub(r_sec, r_ref)
    B_par = _mul(u_los, _dot(B, u_los))       # LOS-parallel component
    B_perp = _sub(B, B_par)                   # component ⟂ LOS
    b_abs = _norm(B_perp)
    sign = 1.0 if _dot(B_perp, e_right) >= 0 else -1.0
    return sign * b_abs, b_abs


# ------------------------------ pair discovery ------------------------------

def find_pair_dir(root: Path, path_num: int, ref: str, sec: str) -> Path | None:
    """Return first matching pair directory. Prefer *_3DEP when both exist."""
    pat = f"path{path_num}_{ref}_{sec}_*"
    cands = sorted(root.glob(pat))
    if not cands:
        return None
    _3dep = [p for p in cands if str(p).upper().endswith("_3DEP")]
    return _3dep[0] if _3dep else cands[0]

def find_xml(pair_dir: Path, date_tag: str) -> Path | None:
    """Return {date_tag}_raw.xml if present, else first '*{date_tag}*raw*.xml' match."""
    p = pair_dir / f"{date_tag}_raw.xml"
    if p.exists():
        return p
    hits = glob.glob(str(pair_dir / f"*{date_tag}*raw*.xml"))
    return Path(hits[0]) if hits else None


# ----------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute LOS-perpendicular baselines for all pairs.")
    ap.add_argument("--pairs", default="/home/bakke326l/InSAR/main/data/pairs.csv",
                    help="CSV with columns: path, ref_date, sec_date (YYYYMMDD).")
    ap.add_argument("--root", default="/mnt/DATA2/bakke326l/processing/interferograms",
                    help="Interferograms root folder.")
    ap.add_argument("--out",  default="/home/bakke326l/InSAR/main/data/perpendicular_baselines.csv",
                    help="Output CSV path.")
    args = ap.parse_args()

    pairs_csv = Path(args.pairs)
    root = Path(args.root)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with pairs_csv.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                path_num = int(str(row["path"]).strip())
                ref = str(row["ref_date"]).strip()
                sec = str(row["sec_date"]).strip()
            except Exception:
                continue

            pair_dir = find_pair_dir(root, path_num, ref, sec)
            if pair_dir is None:
                print(f"skip: pair dir not found for path{path_num} {ref}_{sec}")
                continue

            ref_xml = find_xml(pair_dir, ref)
            sec_xml = find_xml(pair_dir, sec)
            if not (ref_xml and ref_xml.exists() and sec_xml and sec_xml.exists()):
                print(f"skip: XMLs missing in {pair_dir.name}")
                continue

            try:
                b_signed, b_abs = baseline_perpendicular(ref_xml, sec_xml)
            except Exception as e:
                print(f"skip: failed {pair_dir.name}: {e}")
                continue

            rows.append({
                "path": path_num,
                "ref_date": ref,
                "sec_date": sec,
                "bperp_signed_m": f"{b_signed:.3f}",
                "bperp_abs_m": f"{b_abs:.3f}",
            })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "path","ref_date","sec_date","bperp_signed_m","bperp_abs_m"
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"✅ {len(rows)} baselines → {out_csv.resolve()}")


if __name__ == "__main__":
    main()
