#!/bin/bash

# Script to fetch log.json (and optionally psweep_map.json, psweep_info.json)
# from a remote cluster sweep directory so that PsweepCollector can be used locally.
#
# Usage:
#   ./fetch_log_json.sh <simulation_name>
#
# The simulation_name should match the psweep directory name on the remote host,
# e.g.:  psweep_2026-01-15_mips_run1
#
# After fetching, the files land in:
#   ./output/<simulation_name>/log.json
#   ./output/<simulation_name>/psweep_map.json
#   ./output/<simulation_name>/psweep_info.json
#
# which is the same root that fetch_h5_files.sh uses, so PsweepCollector will
# find everything it needs in one place.

# --- Connection / path config (match fetch_h5_files.sh) ---
REMOTE_USER="treado"
REMOTE_HOST="vesta.pks.mpg.de"
REMOTE_BASE_DIR="~/data/GPUParticles"
LOCAL_OUTPUT_DIR="./output"

# --- Manifest files needed by PsweepCollector ---
MANIFEST_FILES=("log.json" "psweep_map.json" "psweep_info.json")

# --- Argument check ---
if [ -z "$1" ]; then
  echo "Usage: $0 <simulation_name>"
  echo "  e.g. $0 psweep_2026-01-15_mips_run1"
  exit 1
fi

SIMULATION_NAME="$1"
REMOTE_DIR="${REMOTE_BASE_DIR}/${SIMULATION_NAME}"
LOCAL_DIR="${LOCAL_OUTPUT_DIR}/${SIMULATION_NAME}"

mkdir -p "$LOCAL_DIR"

echo "Fetching manifest files from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} → ${LOCAL_DIR}/"
echo ""

ERRORS=0
for FILE in "${MANIFEST_FILES[@]}"; do
  REMOTE_PATH="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${FILE}"
  echo "  Fetching ${FILE}..."
  scp "$REMOTE_PATH" "${LOCAL_DIR}/${FILE}"
  if [ $? -ne 0 ]; then
    echo "  Warning: could not fetch ${FILE} (may not exist — skipping)"
    ERRORS=$((ERRORS + 1))
  fi
done

echo ""
if [ "$ERRORS" -eq 0 ]; then
  echo "All manifest files fetched successfully to ${LOCAL_DIR}."
elif [ "$ERRORS" -lt "${#MANIFEST_FILES[@]}" ]; then
  echo "Done (${ERRORS} file(s) missing on remote — PsweepCollector requires log.json and psweep_map.json)."
else
  echo "Error: no manifest files could be fetched. Check your connection and simulation name."
  exit 2
fi

echo ""
echo "You can now use PsweepCollector locally:"
echo "  from analysis.collector import PsweepCollector"
echo "  coll = PsweepCollector('${LOCAL_DIR}')"
echo "  print(coll.log)"
