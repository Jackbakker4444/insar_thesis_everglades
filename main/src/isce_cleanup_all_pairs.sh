#!/usr/bin/env bash
# isce_cleanup_all_pairs.sh
# Batch-clean ISCE pair folders safely from ANY location.
#
# Default target root: /mnt/DATA2/bakke326l/processing/interferograms
# Default pair pattern: path150_*
#
# Keeps (no pruning inside these):
#   interferogram/ ionosphere/ troposphere/ geometry/ PICKLE/ inspect/ coregisteredSlc/
#   stripmapApp.xml stripmapProc.xml isce.log
#
# Deletes:
#   *_raw/ *_raw_crop/ *_slc/ *_slc_crop/
#   offsets/ denseOffsets/ misreg/ SplitSpectrum/
#   resampinfo.bin
#   zero-byte *.log (anywhere below the pair)
#   DEMs: 3dep_*.dem.*  dem.crop*
#
# Usage:
#   bash isce_cleanup_all_pairs.sh [--apply] [--root=/path/to/interferograms] [--pattern='path150_*']
# Examples:
#   bash /home/bakke326l/InSAR/main/src/isce_cleanup_all_pairs.sh
#   bash /home/bakke326l/InSAR/main/src/isce_cleanup_all_pairs.sh --apply
#   bash /home/bakke326l/InSAR/main/src/isce_cleanup_all_pairs.sh --root=/mnt/DATA2/bakke326l/processing/interferograms --pattern='path150_*' --apply

set -euo pipefail

# ----------- defaults -----------
ROOT="/mnt/DATA2/bakke326l/processing/interferograms"
PAIR_PATTERN="path150_*"
APPLY=0

# ----------- parse args ----------
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --root=*) ROOT="${arg#*=}" ;;
    --pattern=*) PAIR_PATTERN="${arg#*=}" ;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# ----------- helpers -------------
say() { printf "%b\n" "$*"; }
hr()  { printf "%s\n" "------------------------------------------------------------"; }

del_dirs_matching() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob
  ((${#matches[@]}==0)) && return 0
  for d in "${matches[@]}"; do
    [[ -d "$d" ]] || continue
    if ((APPLY)); then
      rm -rf -- "$d"
      say "🗑️  deleted dir: $d"
    else
      say "would delete dir: $d"
    fi
  done
}

del_files_matching() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob
  ((${#matches[@]}==0)) && return 0
  if ((APPLY)); then
    rm -f -- "${matches[@]}"
    say "🗑️  deleted files: $pattern"
  else
    for f in "${matches[@]}"; do say "would delete file: $f"; done
  fi
}

# ----------- safety checks -------
if [[ ! -d "$ROOT" ]]; then
  echo "Target root does not exist: $ROOT" >&2; exit 1
fi
if [[ "$ROOT" == "/" ]]; then
  echo "Refusing to run on /" >&2; exit 1
fi

hr
say "ISCE cleanup"
say "Root:    $ROOT"
say "Pattern: $PAIR_PATTERN"
say "Mode:    $([[ $APPLY -eq 1 ]] && echo 'APPLY (deleting)' || echo 'dry-run')"
hr

# Enumerate pairs
shopt -s nullglob
pairs=( "$ROOT"/$PAIR_PATTERN )
shopt -u nullglob

if ((${#pairs[@]}==0)); then
  say "No pair folders found under $ROOT matching '$PAIR_PATTERN'."
  exit 0
fi

for pair in "${pairs[@]}"; do
  [[ -d "$pair" ]] || continue
  # Guard against hitting non-pair directories like quicklook/_reports
  base="$(basename "$pair")"
  [[ "$base" =~ ^path[0-9]+_ ]] || { say "↩️  skip (not a pair): $base"; continue; }

  say "📁 Pair: $base"
  du -sh "$pair" 2>/dev/null || true

  pushd "$pair" >/dev/null

  # ---- MUST KEEP (announce only) ----
  for k in interferogram ionosphere troposphere geometry PICKLE inspect coregisteredSlc; do
    [[ -e "$k" ]] && say "✅ keeping: $k"
  done
  [[ -f stripmapApp.xml  ]] && say "✅ keeping: stripmapApp.xml"
  [[ -f stripmapProc.xml ]] && say "✅ keeping: stripmapProc.xml"
  [[ -f isce.log         ]] && say "✅ keeping: isce.log"

  # ---- DEMs: delete (easy to regenerate) ----
  del_files_matching "3dep_*.dem.*"
  del_files_matching "dem.crop*"

  # ---- RAW/SLC build artifacts ----
  del_dirs_matching "*_raw"
  del_dirs_matching "*_raw_crop"
  del_dirs_matching "*_slc"
  del_dirs_matching "*_slc_crop"

  # ---- Co-registration / offsets / dense matching scratch ----
  # (KEEP coregisteredSlc/ as requested)
  del_dirs_matching "offsets"
  del_dirs_matching "denseOffsets"
  del_dirs_matching "misreg"

  # ---- Split-spectrum intermediates (finals live in ionosphere/) ----
  del_dirs_matching "SplitSpectrum"

  # ---- Misc scratch & empty logs ----
  del_files_matching "resampinfo.bin"

  # Zero-byte *.log anywhere under this pair
  zlogs=$(find . -type f -name "*.log" -size 0 -print 2>/dev/null || true)
  if [[ -n "$zlogs" ]]; then
    while IFS= read -r z; do
      if ((APPLY)); then
        rm -f -- "$z"; say "🗑️  deleted empty log: $z"
      else
        say "would delete empty log: $z"
      fi
    done <<< "$zlogs"
  fi

  popd >/dev/null
  du -sh "$pair" 2>/dev/null || true
  hr
done

say "Done. This was $([[ $APPLY -eq 1 ]] && echo 'APPLY' || echo 'dry-run')."
say "Re-run with --apply to perform deletions."
