#!/bin/bash
#
# QUEKO Benchmark Setup Script (Phase 1 of 3)
#
# Generates task list and prepares environment for array job execution.
#
# Usage:
#   ./slurm_setup.sh [--test]
#
# Options:
#   --test  Generate only 6 tasks (one circuit, all methods) for validation
#
# References:
#   - Google Shell Style Guide: https://google.github.io/styleguide/shellguide.html

#SBATCH --job-name=quariadne-setup
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=setup_%j.out
#SBATCH --error=setup_%j.err
#SBATCH --uenv=prgenv-gnu/25.11:v1

set -euo pipefail

# ============================================================================
# Script Location
# ============================================================================

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
SCRIPT_GRANDPARENT="$(realpath "${SCRIPT_DIR}/../..")"

# ============================================================================
# Command Line Arguments
# ============================================================================

TEST_MODE=false
if [[ "${1:-}" == "--test" ]]; then
  TEST_MODE=true
  echo "Running in TEST MODE"
fi

# ============================================================================
# Configuration
# ============================================================================

if [[ -n "${SCRATCH:-}" ]]; then
  WORK_DIR="${SCRATCH}/quariadne-benchmark"
else
  WORK_DIR="$(pwd)/quariadne-benchmark"
fi

readonly WORK_DIR
readonly RESULTS_DIR="${WORK_DIR}/results"
readonly TASK_LIST="${WORK_DIR}/task_list.txt"

# Quariadne repository
readonly QUARIADNE_REPO="https://github.com/strategy155/quariadne.git"
readonly QUARIADNE_BRANCH="dev"

# Benchmark parameters
readonly BACKENDS=("Ourense" "Sycamore" "Rochester" "Tokyo" "Aspen-4")
readonly METHODS=("sabre" "quariadne_ilp" "quariadne_lpm" "quariadne_lpe" "quariadne_bipartite")

# ============================================================================
# Setup
# ============================================================================

echo "=========================================="
echo "QUEKO Benchmark Setup (Phase 1)"
echo "=========================================="
echo "Work directory: ${WORK_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "SLURM Job ID: ${SLURM_JOB_ID}"
fi

mkdir -p "${WORK_DIR}"
mkdir -p "${RESULTS_DIR}"

# ============================================================================
# Clone or Use Local Repository
# ============================================================================

echo ""
echo "Setting up repository..."

if [[ -f "${SCRIPT_GRANDPARENT}/pyproject.toml" ]]; then
  echo "Using local repo: ${SCRIPT_GRANDPARENT}"
  PROJECT_ROOT="${SCRIPT_GRANDPARENT}"
else
  PROJECT_ROOT="${WORK_DIR}/quariadne"

  if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "Cloning from ${QUARIADNE_REPO}..."
    git clone --recurse-submodules "${QUARIADNE_REPO}" "${PROJECT_ROOT}"
  fi

  echo "Updating to ${QUARIADNE_BRANCH}..."
  git -C "${PROJECT_ROOT}" fetch origin
  git -C "${PROJECT_ROOT}" checkout "${QUARIADNE_BRANCH}"
  git -C "${PROJECT_ROOT}" pull origin "${QUARIADNE_BRANCH}"
  git -C "${PROJECT_ROOT}" submodule update --init --recursive
fi

readonly PROJECT_ROOT
readonly BENCHMARK_DIR="${PROJECT_ROOT}/QUEKO-benchmark"

echo "Project root: ${PROJECT_ROOT}"

# ============================================================================
# Environment Setup
# ============================================================================

echo ""
echo "Setting up Python environment..."

uv_path=$(command -v uv)
if [[ -z "${uv_path}" ]]; then
  echo "Installing uv..."
  curl -L https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "uv version: $(uv --version)"

cd "${PROJECT_ROOT}" || { echo "ERROR: cd failed" >&2; exit 1; }

readonly VENV_DIR="${PROJECT_ROOT}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Creating venv..."
  uv venv "${VENV_DIR}"
fi

echo "Syncing dependencies..."
uv sync

# ============================================================================
# Generate Task List
# ============================================================================

echo ""
echo "Generating task list..."

# Remove existing task list if present
rm -f "${TASK_LIST}"

task_count=0

if [[ "${TEST_MODE}" == true ]]; then
  # Test mode: one circuit, all methods, one backend
  # Use -quit to stop after first match (avoids SIGPIPE with pipefail)
  test_circuit=$(find "${BENCHMARK_DIR}" -name "*.qasm" -type f -print -quit)
  test_circuit_relative="${test_circuit#"${PROJECT_ROOT}/"}"

  for method in "${METHODS[@]}"; do
    echo "${test_circuit_relative}|Aspen-4|${method}" >> "${TASK_LIST}"
    task_count=$((task_count + 1))
  done
else
  # Full mode: all circuits × all backends × all methods
  while IFS= read -r circuit_path; do
    circuit_relative="${circuit_path#"${PROJECT_ROOT}/"}"

    for backend in "${BACKENDS[@]}"; do
      for method in "${METHODS[@]}"; do
        echo "${circuit_relative}|${backend}|${method}" >> "${TASK_LIST}"
        task_count=$((task_count + 1))
      done
    done
  done < <(find "${BENCHMARK_DIR}" -name "*.qasm" -type f | sort)
fi

echo "Generated ${task_count} tasks"
echo "Task list: ${TASK_LIST}"
echo ""
echo "Sample tasks:"
head -3 "${TASK_LIST}"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo "Setup Complete"
echo "=========================================="
echo "Total tasks: ${task_count}"
