#!/bin/bash
#SBATCH --partition=None
#SBATCH --time=06:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --job-name=psweep_2026-05-07_phi_dense_0000
#SBATCH --array=1-10
#SBATCH --output=/Users/jacktreado/Dresden/Projects/GPUParticles/psweeps/psweep_2026-05-07_phi_dense/slurm/psweep_2026-05-07_phi_dense_0000/psweep_2026-05-07_phi_dense_0000-%a.out

set -eo pipefail
echo "Job started on $(date) on $(hostname)"
cmd=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "/Users/jacktreado/Dresden/Projects/GPUParticles/psweeps/psweep_2026-05-07_phi_dense/slurm/psweep_2026-05-07_phi_dense_0000/psweep_2026-05-07_phi_dense_0000.task")
echo "+ $cmd"
eval "$cmd"
echo "Job finished on $(date)"
