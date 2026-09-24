#!/bin/bash

# Script to fetch all .h5 files from a remote cluster to the local output directory.

# Variables
REMOTE_USER="treado"
REMOTE_HOST="vesta"
REMOTE_BASE_DIR="~/data/GPUParticles"
LOCAL_OUTPUT_DIR="./output"

# Help function
show_help() {
  cat << EOF
Usage: $0 <simulation_name> <instance_id> <seed>

Fetch .h5 simulation files from remote cluster to local output directory.

Required Arguments:
  simulation_name     Name of the simulation/sweep
  instance_id         Instance identifier (e.g., 001, 002)
  seed                Random seed number

Optional Flags:
  -h, --help          Show this help message

Examples:
  $0 psweep_my_sim 001 0
  $0 my_simulation run_001 42

Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}
Local:  ${LOCAL_OUTPUT_DIR}

EOF
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" || -z "$1" ]]; then
  show_help
  [[ "$1" == "-h" || "$1" == "--help" ]] && exit 0 || exit 1
fi

# Check if all required arguments are provided
if [ $# -lt 3 ]; then
  echo "Error: Missing required arguments."
  show_help
  exit 1
fi

# Simulation name (subdirectory)
SIMULATION_NAME="$1"
INSTANCE_ID="$2"
SEED="$3"

# Construct remote and local paths
REMOTE_DIR="${REMOTE_BASE_DIR}/${SIMULATION_NAME}/${SIMULATION_NAME}_${INSTANCE_ID}"
LOCAL_DIR="${LOCAL_OUTPUT_DIR}/${SIMULATION_NAME}/${SIMULATION_NAME}_${INSTANCE_ID}"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Perform the SCP operation
echo "Fetching .h5 files from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} to ${LOCAL_DIR}..."
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${SIMULATION_NAME}_${INSTANCE_ID}_seed${SEED}.h5" "$LOCAL_DIR"

# Check if the SCP command was successful
if [ $? -eq 0 ]; then
  echo "Files successfully fetched to ${LOCAL_DIR}."
else
  echo "Error: Failed to fetch files. Please check your connection and paths."
  echo ""
  echo "Diagnostic info:"
  echo "================================================"
  
  # Check if remote directory exists
  if ssh "${REMOTE_USER}@${REMOTE_HOST}" "[ -d ${REMOTE_DIR} ]"; then
    echo "✓ Remote directory exists: ${REMOTE_DIR}"
    echo ""
    echo "Files in remote directory:"
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "ls -lh ${REMOTE_DIR}/*.h5 2>/dev/null || echo '  (no .h5 files found)'"
  else
    echo "✗ Remote directory does NOT exist: ${REMOTE_DIR}"
  fi
  
  echo "================================================"
  exit 2
fi
