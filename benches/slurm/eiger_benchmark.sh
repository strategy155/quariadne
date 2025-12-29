#!/bin/bash
#
# QUEKO Benchmarking Script for CSCS Eiger
#
# This script works both:
# - Locally: bash eiger_benchmark.sh (ignores #SBATCH directives)
# - Cluster: sbatch eiger_benchmark.sh (uses SLURM)
#
# Runs the Quariadne router benchmarks on QUEKO circuits with persistence
# for resuming after job restarts.
#
# Usage:
#   ./eiger_benchmark.sh [--test]
#
# Options:
#   --test  Run in test mode: single circuit with sabre algorithm
#
# First-time setup on CSCS (run once before sbatch):
#   uenv repo create
#   uenv image pull prgenv-gnu/25.11:v1
#
# References:
#   - CSCS Eiger docs: https://docs.cscs.ch/clusters/eiger/
#   - CSCS uenv docs: https://docs.cscs.ch/software/uenv/
#   - Google Shell Style Guide: https://google.github.io/styleguide/shellguide.html

#SBATCH --job-name=quariadne-bench
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

# Load uenv environment (provides Python and build tools)
# Note: Before first use, run: uenv repo create && uenv image pull prgenv-gnu/25.11:v1
#SBATCH --uenv=prgenv-gnu/25.11:v1

# ============================================================================
# Script Location (must be first, before any operations)
# ============================================================================

# Get absolute path of script's parent directory (potential quariadne repo)
SCRIPT_RELATIVE_DIR="$(dirname "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(realpath "${SCRIPT_RELATIVE_DIR}")"
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

# Work directory for cloning and results:
# - On CSCS cluster: use $SCRATCH (fast storage with 30-day retention)
# - Locally: use current directory
# See: https://docs.cscs.ch/platforms/hpcp/#scratch
if [[ -n "${SCRATCH:-}" ]]; then
  WORK_DIR="${SCRATCH}/quariadne-benchmark"
else
  WORK_DIR="$(pwd)/quariadne-benchmark"
fi

readonly RESULTS_DIR="${WORK_DIR}/results"

# Quariadne repository configuration
readonly QUARIADNE_REPO="https://github.com/strategy155/quariadne.git"
readonly QUARIADNE_BRANCH="dev"

# File patterns for circuit discovery
readonly QASM_FILE_PATTERN="*.qasm"

# ============================================================================
# Setup
# ============================================================================

start_time=$(date)

echo "=========================================="
echo "QUEKO Benchmark Script"
echo "=========================================="
echo "Start time: ${start_time}"
echo "Work directory: ${WORK_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "SLURM Job ID: ${SLURM_JOB_ID}"
  echo "Node: ${SLURM_NODELIST:-N/A}"
fi
echo ""

mkdir -p "${WORK_DIR}"
mkdir -p "${RESULTS_DIR}"

# ============================================================================
# Clone or Use Local Repository
# ============================================================================

echo ""
echo "Setting up Quariadne repository..."
echo "=========================================="

# Detect if script is running from within a quariadne repository
# by checking for pyproject.toml in the script's grandparent directory
if [[ -f "${SCRIPT_GRANDPARENT}/pyproject.toml" ]]; then
  echo "Using local quariadne repo: ${SCRIPT_GRANDPARENT}"
  PROJECT_ROOT="${SCRIPT_GRANDPARENT}"
else
  # Clone repository to work directory if not present
  PROJECT_ROOT="${WORK_DIR}/quariadne"

  if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "Cloning quariadne from ${QUARIADNE_REPO}..."
    git clone --recurse-submodules "${QUARIADNE_REPO}" "${PROJECT_ROOT}"
  fi

  # Update to latest version of target branch
  echo "Updating to latest ${QUARIADNE_BRANCH} branch..."
  git -C "${PROJECT_ROOT}" fetch origin
  git -C "${PROJECT_ROOT}" checkout "${QUARIADNE_BRANCH}"
  git -C "${PROJECT_ROOT}" pull origin "${QUARIADNE_BRANCH}"

  # Ensure QUEKO-benchmark submodule is initialised
  git -C "${PROJECT_ROOT}" submodule update --init --recursive
fi

readonly PROJECT_ROOT
readonly BENCHMARK_DIR="${PROJECT_ROOT}/QUEKO-benchmark"
readonly VENV_DIR="${PROJECT_ROOT}/.venv"

echo "Project root: ${PROJECT_ROOT}"
echo "Benchmark directory: ${BENCHMARK_DIR}"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

echo "Setting up Python environment..."

# Check if uv package manager is installed
# command -v returns path if command exists, empty string otherwise
# Reference: https://docs.astral.sh/uv/getting-started/installation/
uv_path=$(command -v uv)
if [[ -z "${uv_path}" ]]; then
  echo "Installing uv package manager..."
  # -L: follow HTTP redirects (the URL redirects to latest release)
  curl -L https://astral.sh/uv/install.sh | sh
  # uv installs to ~/.local/bin by default
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "uv version: $(uv --version)"

cd "${PROJECT_ROOT}" || { echo "ERROR: Failed to cd to ${PROJECT_ROOT}" >&2; exit 1; }

# Check if virtual environment exists by looking for activate script
venv_activate_script="${VENV_DIR}/bin/activate"
if [[ ! -f "${venv_activate_script}" ]]; then
  echo "Creating virtual environment with uv..."
  uv venv "${VENV_DIR}"
fi

# Install/update project dependencies from pyproject.toml
echo "Syncing dependencies with uv..."
uv sync

# Activate the virtual environment
# shellcheck source=/dev/null
source "${venv_activate_script}"

echo "Python: $(which python)"
echo ""

# ============================================================================
# Run Benchmarks
# ============================================================================

echo "Running benchmarks..."
echo ""

if [[ "${TEST_MODE}" == true ]]; then
  echo "TEST MODE: Running single circuit benchmark"

  # Find all QASM circuit files in the benchmark directory
  all_qasm_files=$(find "${BENCHMARK_DIR}" -name "${QASM_FILE_PATTERN}" -type f)

  # Take only the first circuit for testing
  first_circuit=$(echo "${all_qasm_files}" | head -1)

  if [[ -z "${first_circuit}" ]]; then
    echo "ERROR: No ${QASM_FILE_PATTERN} files found in ${BENCHMARK_DIR}" >&2
    exit 1
  fi

  echo "Circuit: ${first_circuit}"

  # Common arguments must come before the subcommand
  python benches/run_benchmarks.py \
    --output-dir "${RESULTS_DIR}" \
    --log-level INFO \
    single \
    --circuit "${first_circuit}" \
    --backend Aspen-4 \
    --method sabre
else
  echo "FULL MODE: Running all QUEKO benchmarks"

  python benches/run_benchmarks.py \
    --output-dir "${RESULTS_DIR}" \
    --log-level INFO \
    full \
    --benchmark-dir "${BENCHMARK_DIR}"
fi

# ============================================================================
# Summary
# ============================================================================

end_time=$(date)

echo ""
echo "=========================================="
echo "Benchmarking Complete"
echo "=========================================="
echo "End time: ${end_time}"
echo "Results directory: ${RESULTS_DIR}"