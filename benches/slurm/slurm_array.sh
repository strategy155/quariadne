#!/bin/bash
#
# QUEKO Benchmark Array Job (Phase 2 of 3)
#
# Each array task processes a CHUNK of benchmarks from the task list.
# This keeps array size within SLURM limits (MaxArraySize=20000).
#
# With CHUNK_SIZE=30 and 27180 total tasks:
#   - Array size: 906 tasks (well under 20000 limit)
#   - Each task processes up to 30 benchmarks sequentially
#
# Usage:
#   sbatch --dependency=afterok:<setup_job_id> slurm_array.sh
#
# References:
#   - SLURM job arrays: https://slurm.schedmd.com/job_array.html
#   - Google Shell Style Guide: https://google.github.io/styleguide/shellguide.html

#SBATCH --job-name=quariadne-array
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-905%64
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --uenv=prgenv-gnu/25.11:v1

set -euo pipefail

# Number of benchmarks each array task processes
readonly CHUNK_SIZE=30

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

# For local testing, allow override of array task ID
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ============================================================================
# Calculate Chunk Range
# ============================================================================

# Each array task processes CHUNK_SIZE benchmarks
# Task 0 processes lines 1-30, Task 1 processes lines 31-60, etc.
chunk_start=$((ARRAY_TASK_ID * CHUNK_SIZE))
chunk_end=$((chunk_start + CHUNK_SIZE - 1))

# Get total number of tasks in task list
total_tasks=$(wc -l < "${TASK_LIST}")

# Clamp chunk_end to total_tasks - 1 (task indices are 0-based)
if [[ ${chunk_end} -ge ${total_tasks} ]]; then
  chunk_end=$((total_tasks - 1))
fi

echo "=========================================="
echo "QUEKO Benchmark Chunk ${ARRAY_TASK_ID}"
echo "=========================================="
echo "Processing tasks ${chunk_start} to ${chunk_end}"
echo "Total tasks in chunk: $((chunk_end - chunk_start + 1))"
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
# Process All Tasks in Chunk
# ============================================================================

completed=0
failed=0

for task_idx in $(seq "${chunk_start}" "${chunk_end}"); do
  # Task list is 0-indexed, sed line numbers are 1-indexed
  line_num=$((task_idx + 1))
  task_line=$(sed -n "${line_num}p" "${TASK_LIST}")

  if [[ -z "${task_line}" ]]; then
    echo "WARNING: No task at line ${line_num}, skipping"
    continue
  fi

  # Parse pipe-delimited task: circuit|backend|method
  IFS='|' read -r circuit_path backend method <<< "${task_line}"

  echo ""
  echo "--- Task ${task_idx}: ${method} on $(basename "${circuit_path}" .qasm) ---"

  # Run benchmark, continue on failure
  if uv run python benches/run_benchmarks.py \
       --output-dir "${RESULTS_DIR}" \
       --log-level INFO \
       single \
       --circuit "${circuit_path}" \
       --backend "${backend}" \
       --method "${method}"; then
    completed=$((completed + 1))
  else
    echo "FAILED: ${circuit_path} with ${method}"
    failed=$((failed + 1))
  fi
done

echo ""
echo "=========================================="
echo "Chunk ${ARRAY_TASK_ID} Complete"
echo "=========================================="
echo "Completed: ${completed}"
echo "Failed: ${failed}"
