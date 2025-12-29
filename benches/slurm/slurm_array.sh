#!/bin/bash
#
# QUEKO Benchmark Array Job (Phase 2 of 3)
#
# Runs individual benchmark tasks from the task list.
# Each array task processes one (circuit, backend, method) combination.
#
# Usage:
#   sbatch --dependency=afterok:<setup_job_id> slurm_array.sh
#
# References:
#   - SLURM job arrays: https://slurm.schedmd.com/job_array.html
#   - Google Shell Style Guide: https://google.github.io/styleguide/shellguide.html

#SBATCH --job-name=quariadne-array
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-27179%64
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --uenv=prgenv-gnu/25.11:v1

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

if [[ -n "${SCRATCH:-}" ]]; then
  WORK_DIR="${SCRATCH}/quariadne-benchmark"
else
  WORK_DIR="$(pwd)/quariadne-benchmark"
fi

readonly WORK_DIR
readonly TASK_LIST="${WORK_DIR}/task_list.txt"
readonly RESULTS_DIR="${WORK_DIR}/results"

# For local testing, allow override
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ============================================================================
# Read Task
# ============================================================================

# Task list is 0-indexed, sed line numbers are 1-indexed
line_num=$((TASK_ID + 1))
task_line=$(sed -n "${line_num}p" "${TASK_LIST}")

if [[ -z "${task_line}" ]]; then
  echo "ERROR: No task at line ${line_num}" >&2
  exit 1
fi

# Parse pipe-delimited task: circuit|backend|method
IFS='|' read -r circuit_path backend method <<< "${task_line}"

echo "=========================================="
echo "QUEKO Benchmark Task ${TASK_ID}"
echo "=========================================="
echo "Circuit: ${circuit_path}"
echo "Backend: ${backend}"
echo "Method: ${method}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  echo "Array Job: ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
fi
echo ""

# ============================================================================
# Find Project Root
# ============================================================================

# On cluster: repo is cloned to WORK_DIR/quariadne
# Locally: we're already inside the repo
if [[ -d "${WORK_DIR}/quariadne/.git" ]]; then
  cd "${WORK_DIR}/quariadne"
fi

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
readonly PROJECT_ROOT

echo "Project root: ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"

# ============================================================================
# Run Benchmark
# ============================================================================

echo ""
echo "Running transpilation..."

uv run python benches/run_benchmarks.py \
  --output-dir "${RESULTS_DIR}" \
  --log-level INFO \
  single \
  --circuit "${circuit_path}" \
  --backend "${backend}" \
  --method "${method}"

echo ""
echo "Task ${TASK_ID} completed"
