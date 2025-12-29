#!/bin/bash
#
# QUEKO Benchmark Pipeline Orchestrator
#
# Submits all three phases of the benchmark pipeline with SLURM dependencies:
#   Phase 1: Setup      - Clone repo, install dependencies, generate task list
#   Phase 2: Array Job  - Run benchmarks in parallel (one task per array element)
#   Phase 3: Aggregate  - Collect results and generate summary CSV
#
# The dependency chain ensures proper ordering:
#   - Array job starts only after setup succeeds (afterok)
#   - Aggregate runs after array completes, even with failures (afterany)
#
# Usage:
#   ./submit_benchmark_pipeline.sh [--test] [--skip-setup] [--dry-run]
#
# Options:
#   --test        Run 6 tasks only (one circuit, all methods) for validation
#   --skip-setup  Skip Phase 1 if already completed (reuse existing task list)
#   --dry-run     Print sbatch commands without executing them
#
# Reference:
#   - SLURM Dependencies: https://slurm.schedmd.com/sbatch.html#OPT_dependency

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Locate sibling scripts relative to this orchestrator
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
readonly SCRIPT_DIR

readonly SETUP_SCRIPT="${SCRIPT_DIR}/slurm_setup.sh"
readonly ARRAY_SCRIPT="${SCRIPT_DIR}/slurm_array.sh"
readonly AGGREGATE_SCRIPT="${SCRIPT_DIR}/slurm_aggregate.sh"

# Command-line flags
DRY_RUN=false
TEST_MODE=false
SKIP_SETUP=false

# Test mode overrides: smaller array range for quick validation
# Full mode uses the array range defined in slurm_array.sh (0-27179)
# Test mode: single chunk (chunk 0 processes up to 30 tasks)
readonly TEST_ARRAY_RANGE="0-0"
readonly TEST_ARRAY_TIME="00:30:00"

# ============================================================================
# Parse Arguments
# ============================================================================

for arg in "$@"; do
  case "${arg}" in
    --test)
      TEST_MODE=true
      ;;
    --skip-setup)
      SKIP_SETUP=true
      ;;
    --dry-run)
      DRY_RUN=true
      ;;
    --help|-h)
      echo "Usage: $0 [--test] [--skip-setup] [--dry-run]"
      echo ""
      echo "Options:"
      echo "  --test        Run 6 tasks only (one circuit, all methods)"
      echo "  --skip-setup  Skip Phase 1 if already completed"
      echo "  --dry-run     Print commands without executing"
      echo "  --help        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

# ============================================================================
# Verify Scripts Exist
# ============================================================================

echo "=========================================="
if [[ "${TEST_MODE}" == true ]]; then
  echo "QUEKO Benchmark Pipeline (TEST MODE)"
else
  echo "QUEKO Benchmark Pipeline"
fi
echo "=========================================="
echo ""

if [[ "${TEST_MODE}" == true ]]; then
  echo "Test configuration:"
  echo "  Array range: ${TEST_ARRAY_RANGE} (1 chunk)"
  echo "  Array time:  ${TEST_ARRAY_TIME}"
  echo ""
fi

# Verify all required scripts are present before submitting anything
for script in "${SETUP_SCRIPT}" "${ARRAY_SCRIPT}" "${AGGREGATE_SCRIPT}"; do
  if [[ ! -f "${script}" ]]; then
    echo "ERROR: Missing script: ${script}" >&2
    exit 1
  fi
done

echo "Scripts verified:"
echo "  Phase 1: ${SETUP_SCRIPT}"
echo "  Phase 2: ${ARRAY_SCRIPT}"
echo "  Phase 3: ${AGGREGATE_SCRIPT}"
echo ""

# ============================================================================
# Build sbatch Options
# ============================================================================

# Array job options differ between test and full mode
ARRAY_OPTS=()
if [[ "${TEST_MODE}" == true ]]; then
  # Override array range and time limit for quick validation
  ARRAY_OPTS+=(--array="${TEST_ARRAY_RANGE}" --time="${TEST_ARRAY_TIME}")
fi

# Setup script options
SETUP_OPTS=()
if [[ "${TEST_MODE}" == true ]]; then
  # Pass --test flag to setup script for minimal task list
  SETUP_OPTS+=("--test")
fi

# ============================================================================
# Dry Run Mode
# ============================================================================

if [[ "${DRY_RUN}" == true ]]; then
  echo "[DRY RUN] Would execute:"
  echo ""
  if [[ "${SKIP_SETUP}" == false ]]; then
    echo "  sbatch --parsable ${SETUP_SCRIPT} ${SETUP_OPTS[*]:-}"
    echo "  sbatch --parsable --dependency=afterok:<setup_id> ${ARRAY_OPTS[*]:-} ${ARRAY_SCRIPT}"
  else
    echo "  (skipping setup)"
    echo "  sbatch --parsable ${ARRAY_OPTS[*]:-} ${ARRAY_SCRIPT}"
  fi
  echo "  sbatch --parsable --dependency=afterany:<array_id> ${AGGREGATE_SCRIPT}"
  exit 0
fi

# ============================================================================
# Submit Pipeline
# ============================================================================

setup_id=""

# Phase 1: Setup (unless skipped)
if [[ "${SKIP_SETUP}" == false ]]; then
  echo "Submitting Phase 1 (Setup)..."
  setup_id=$(sbatch --parsable "${SETUP_SCRIPT}" "${SETUP_OPTS[@]}")
  echo "  Job ID: ${setup_id}"
  echo ""

  # Phase 2: Array Job (depends on setup success)
  echo "Submitting Phase 2 (Array Job)..."
  echo "  Dependency: afterok:${setup_id}"
  array_id=$(sbatch --parsable --dependency="afterok:${setup_id}" "${ARRAY_OPTS[@]}" "${ARRAY_SCRIPT}")
  echo "  Job ID: ${array_id}"
else
  echo "Skipping Phase 1 (--skip-setup)"
  echo ""

  # Phase 2: Array Job (no dependency)
  echo "Submitting Phase 2 (Array Job)..."
  array_id=$(sbatch --parsable "${ARRAY_OPTS[@]}" "${ARRAY_SCRIPT}")
  echo "  Job ID: ${array_id}"
fi
echo ""

# Phase 3: Aggregate (runs after array completes, regardless of success/failure)
echo "Submitting Phase 3 (Aggregate)..."
echo "  Dependency: afterany:${array_id}"
aggregate_id=$(sbatch --parsable --dependency="afterany:${array_id}" "${AGGREGATE_SCRIPT}")
echo "  Job ID: ${aggregate_id}"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo "Pipeline Submitted"
echo "=========================================="
echo ""

if [[ "${TEST_MODE}" == true ]]; then
  echo "This is a TEST run with 6 tasks."
  echo "After validation, run full pipeline without --test"
  echo ""
fi

echo "Job IDs:"
if [[ -n "${setup_id}" ]]; then
  echo "  Setup:     ${setup_id}"
fi
echo "  Array:     ${array_id}"
echo "  Aggregate: ${aggregate_id}"
echo ""
echo "Monitor with: squeue -u \$USER"
if [[ -n "${setup_id}" ]]; then
  echo "Cancel all:  scancel ${setup_id} ${array_id} ${aggregate_id}"
else
  echo "Cancel all:  scancel ${array_id} ${aggregate_id}"
fi
