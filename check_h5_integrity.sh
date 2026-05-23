#!/bin/bash

# Script to check integrity of all .h5 trajectory files for a parameter sweep.
# Walks /home/treado/data/GPUParticles/<psweep_name>/<psweep_name>_<instant_id>/
# and reports valid vs corrupted files plus per-instant_id summary stats.

# Variables
BASE_DIR="${BASE_DIR:-/home/treado/data/GPUParticles}"

# Check if the psweep name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <psweep_name>"
  exit 1
fi

PSWEEP_NAME="$1"
PSWEEP_DIR="${BASE_DIR}/${PSWEEP_NAME}"

if [ ! -d "$PSWEEP_DIR" ]; then
  echo "Error: psweep directory not found: ${PSWEEP_DIR}"
  exit 2
fi

# Require h5ls (from hdf5-tools). Validity = h5ls succeeds AND at least one
# frame_ group is listed at the root.
if ! command -v h5ls >/dev/null 2>&1; then
  echo "Error: h5ls not found. Install hdf5-tools (apt) or hdf5 (brew)."
  exit 3
fi

# Per-instant_id counters, totals across the sweep
declare -A N_TOTAL N_VALID N_CORRUPT N_EMPTY
sweep_total=0
sweep_valid=0
sweep_corrupt=0
sweep_empty=0

shopt -s nullglob

# Loop over instant_id subdirectories
for instant_dir in "${PSWEEP_DIR}/${PSWEEP_NAME}_"*/; do
  instant_id="$(basename "$instant_dir")"
  instant_id="${instant_id#${PSWEEP_NAME}_}"

  N_TOTAL[$instant_id]=0
  N_VALID[$instant_id]=0
  N_CORRUPT[$instant_id]=0
  N_EMPTY[$instant_id]=0

  for h5file in "${instant_dir}"${PSWEEP_NAME}_${instant_id}_seed*.h5; do
    N_TOTAL[$instant_id]=$((N_TOTAL[$instant_id] + 1))
    sweep_total=$((sweep_total + 1))

    # h5ls returns nonzero on a corrupt/unreadable file. Suppress its output
    # and inspect ourselves.
    listing="$(h5ls "$h5file" 2>/dev/null)"
    rc=$?

    if [ $rc -ne 0 ]; then
      N_CORRUPT[$instant_id]=$((N_CORRUPT[$instant_id] + 1))
      sweep_corrupt=$((sweep_corrupt + 1))
      echo "CORRUPT: $h5file"
      continue
    fi

    # Count frame_ groups at the root.
    n_frames=$(printf '%s\n' "$listing" | grep -c '^frame_')

    if [ "$n_frames" -eq 0 ]; then
      N_EMPTY[$instant_id]=$((N_EMPTY[$instant_id] + 1))
      sweep_empty=$((sweep_empty + 1))
      echo "EMPTY:   $h5file"
    else
      N_VALID[$instant_id]=$((N_VALID[$instant_id] + 1))
      sweep_valid=$((sweep_valid + 1))
    fi
  done
done

# Summary
echo
echo "=========================================================="
echo "Summary for psweep: ${PSWEEP_NAME}"
echo "Base: ${PSWEEP_DIR}"
echo "=========================================================="
printf "%-30s %8s %8s %8s %8s\n" "instant_id" "total" "valid" "empty" "corrupt"
echo "----------------------------------------------------------"

# Sort instant_ids for stable output
for instant_id in $(printf '%s\n' "${!N_TOTAL[@]}" | sort); do
  printf "%-30s %8d %8d %8d %8d\n" \
    "$instant_id" \
    "${N_TOTAL[$instant_id]}" \
    "${N_VALID[$instant_id]}" \
    "${N_EMPTY[$instant_id]}" \
    "${N_CORRUPT[$instant_id]}"
done

echo "----------------------------------------------------------"
printf "%-30s %8d %8d %8d %8d\n" \
  "TOTAL" "$sweep_total" "$sweep_valid" "$sweep_empty" "$sweep_corrupt"

if [ "$sweep_total" -eq 0 ]; then
  echo
  echo "No .h5 files found under ${PSWEEP_DIR}."
  exit 4
fi

exit 0
