#!/bin/bash

# Script to fetch all .h5 files from a remote cluster to the local output directory.

# Variables
REMOTE_USER="treado"          # Replace with your remote username
REMOTE_HOST="vesta.pks.mpg.de"          # Replace with your remote hostname
REMOTE_BASE_DIR="~/data/GPUParticles"  # Base directory on the remote cluster
LOCAL_OUTPUT_DIR="./output"       # Local directory relative to the repository

# Check if the simulation name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <simulation_name>"
  exit 1
fi

# Simulation name (subdirectory)
SIMULATION_NAME="$1"

# Construct remote and local paths
REMOTE_DIR="${REMOTE_BASE_DIR}/${SIMULATION_NAME}"
LOCAL_DIR="${LOCAL_OUTPUT_DIR}/${SIMULATION_NAME}"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Perform the SCP operation
echo "Fetching .h5 files from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} to ${LOCAL_DIR}..."
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/*.h5" "$LOCAL_DIR"

# Check if the SCP command was successful
if [ $? -eq 0 ]; then
  echo "Files successfully fetched to ${LOCAL_DIR}."
else
  echo "Error: Failed to fetch files. Please check your connection and paths."
  exit 2
fi
