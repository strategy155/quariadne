"""Shared constants for benchmark configuration and CLI help strings.

This module centralises all benchmark-related constants for consistency
and future internationalisation (i18n) support.

References:
    - Python 3.13 pathlib: https://docs.python.org/3/library/pathlib.html
    - Google Python Style Guide: https://google.github.io/styleguide/pyguide.html#317-constants
"""

from enum import StrEnum
from pathlib import Path


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

BENCHMARK_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
BENCHMARK_DEFAULT_LOG_LEVEL = "INFO"
BENCHMARK_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


# -----------------------------------------------------------------------------
# File Extensions
# -----------------------------------------------------------------------------

QASM_EXTENSION = ".qasm"
QPY_EXTENSION = ".qpy"
JSON_EXTENSION = ".json"
JSON_INDENT = 2


# -----------------------------------------------------------------------------
# Default Values
# -----------------------------------------------------------------------------

# Qiskit optimisation level for benchmarking (0 = no optimisation)
DEFAULT_OPTIMISATION_LEVEL = 0

# Timeout in seconds (1 hour for cluster jobs)
DEFAULT_TIMEOUT_SECONDS = 3600

# Default backend for testing (Aspen-4 has 16 qubits, fits QUEKO 16QBT circuits)
DEFAULT_BACKEND = "Aspen-4"

# Relative paths from project root
DEFAULT_BENCHMARK_DIR = Path("QUEKO-benchmark")
DEFAULT_OUTPUT_DIR = Path("benches/results")


# -----------------------------------------------------------------------------
# QUEKO Benchmark Categories
# -----------------------------------------------------------------------------


class QuekoCategory(StrEnum):
    """QUEKO benchmark circuit categories.

    Attributes:
        BIGD: Big D circuits
        BNTF: Boolean satisfiability circuits
        BSS: BigSoft circuits
    """

    BIGD = "BIGD"
    BNTF = "BNTF"
    BSS = "BSS"


# -----------------------------------------------------------------------------
# Routing Methods
# -----------------------------------------------------------------------------


class RoutingMethod(StrEnum):
    """Routing methods for benchmarking (6 algorithms total).

    Quariadne methods match entry points registered in pyproject.toml.
    Qiskit methods use built-in routing algorithms.

    Attributes:
        QUARIADNE_ILP: Integer Linear Programming (exact solver)
        QUARIADNE_LPM: LP with Birkhoff decomposition (mapping recovery)
        QUARIADNE_LPE: LP with edge-based bipartite matching
        SABRE: Qiskit SABRE routing algorithm
        BASIC: Qiskit basic routing algorithm
        LOOKAHEAD: Qiskit lookahead routing algorithm
    """

    # Quariadne methods (match pyproject.toml entry points)
    QUARIADNE_ILP = "quariadne_ilp"
    QUARIADNE_LPM = "quariadne_lpm"
    QUARIADNE_LPE = "quariadne_lpe"

    # Qiskit baseline methods
    SABRE = "sabre"
    BASIC = "basic"
    LOOKAHEAD = "lookahead"


# -----------------------------------------------------------------------------
# CLI Help Strings (for future i18n support)
# -----------------------------------------------------------------------------

HELP_DESCRIPTION = "Run QUEKO benchmarks for Quariadne router"
HELP_LOG_LEVEL = "Logging level"
HELP_OUTPUT_DIR = "Output directory for results"
HELP_SUBCOMMAND = "Benchmark mode"

# Full benchmark mode
HELP_FULL_MODE = "Run benchmarks on all QUEKO circuits"
HELP_BENCHMARK_DIR = "Path to QUEKO-benchmark directory"

# Single circuit mode
HELP_SINGLE_MODE = "Run benchmark on a single circuit (for local testing)"
HELP_CIRCUIT = "Path to QASM circuit file"
HELP_BACKEND = "Target backend"
HELP_METHOD = "Specific routing method (default: runs all methods)"
HELP_TIMEOUT = "Timeout per algorithm in seconds"


# -----------------------------------------------------------------------------
# Hardware Backend Definitions
# -----------------------------------------------------------------------------

# Coupling map edge lists for QUEKO benchmark backends
# These define the physical qubit connectivity for each backend
BACKEND_EDGES = {
    "Ourense": [(0, 1), (1, 2), (1, 3), (3, 4)],
    "Sycamore": [
        (0, 6),
        (1, 6),
        (1, 7),
        (2, 7),
        (2, 8),
        (3, 8),
        (3, 9),
        (4, 9),
        (4, 10),
        (5, 10),
        (5, 11),
        (6, 12),
        (6, 13),
        (7, 13),
        (7, 14),
        (8, 14),
        (8, 15),
        (9, 15),
        (9, 16),
        (10, 16),
        (10, 17),
        (11, 17),
        (12, 18),
        (13, 18),
        (13, 19),
        (14, 19),
        (14, 20),
        (15, 20),
        (15, 21),
        (16, 21),
        (16, 22),
        (17, 22),
        (17, 23),
        (18, 24),
        (18, 25),
        (19, 25),
        (19, 26),
        (20, 26),
        (20, 27),
        (21, 27),
        (21, 28),
        (22, 28),
        (22, 29),
        (23, 29),
        (24, 30),
        (25, 30),
        (25, 31),
        (26, 31),
        (26, 32),
        (27, 32),
        (27, 33),
        (28, 33),
        (28, 34),
        (29, 34),
        (29, 35),
        (30, 36),
        (30, 37),
        (31, 37),
        (31, 38),
        (32, 38),
        (32, 39),
        (33, 39),
        (33, 40),
        (34, 40),
        (34, 41),
        (35, 41),
        (36, 42),
        (37, 42),
        (37, 43),
        (38, 43),
        (38, 44),
        (39, 44),
        (39, 45),
        (40, 45),
        (40, 46),
        (41, 46),
        (41, 47),
        (42, 48),
        (42, 49),
        (43, 49),
        (43, 50),
        (44, 50),
        (44, 51),
        (45, 51),
        (45, 52),
        (46, 52),
        (46, 53),
        (47, 53),
    ],
    "Rochester": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (4, 6),
        (5, 9),
        (6, 13),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (7, 16),
        (11, 17),
        (15, 18),
        (16, 19),
        (17, 23),
        (18, 27),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
        (24, 25),
        (25, 26),
        (26, 27),
        (21, 28),
        (25, 29),
        (28, 32),
        (29, 36),
        (30, 31),
        (31, 32),
        (32, 33),
        (33, 34),
        (34, 35),
        (35, 36),
        (36, 37),
        (37, 38),
        (30, 39),
        (34, 40),
        (38, 41),
        (39, 42),
        (40, 46),
        (41, 50),
        (42, 43),
        (43, 44),
        (44, 45),
        (45, 46),
        (46, 47),
        (47, 48),
        (48, 49),
        (49, 50),
        (44, 51),
        (48, 52),
    ],
    "Tokyo": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (1, 6),
        (1, 7),
        (2, 6),
        (2, 7),
        (3, 8),
        (3, 9),
        (4, 8),
        (4, 9),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (5, 10),
        (5, 11),
        (6, 10),
        (6, 11),
        (7, 12),
        (7, 13),
        (8, 12),
        (8, 13),
        (9, 14),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (10, 15),
        (11, 16),
        (11, 17),
        (12, 16),
        (12, 17),
        (13, 18),
        (13, 19),
        (14, 18),
        (14, 19),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
    ],
    "Aspen-4": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (0, 8),
        (3, 11),
        (4, 12),
        (7, 15),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
    ],
}

# List of available backend names for CLI validation
AVAILABLE_BACKENDS = list(BACKEND_EDGES.keys())
