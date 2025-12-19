#!/usr/bin/env python3
"""QUEKO benchmarking script for Quariadne router.

This script runs benchmarks on all QUEKO circuits with different backends,
measuring transpilation time and saving results for analysis.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import StrEnum
from pathlib import Path


import networkx as nx
import qiskit.qasm2
import qiskit.qpy
import qiskit.transpiler
from qiskit import QuantumCircuit

# Logging constants
QUEKO_BENCHMARK_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# QUEKO circuit file extension
QUEKO_QASM_EXTENSION = ".qasm"

# Transpiled circuit file extension (QPY format)
QUEKO_QPY_EXTENSION = ".qpy"

# Default Qiskit optimisation level for benchmarking
QUEKO_DEFAULT_OPTIMISATION_LEVEL = 0


class QuekoCategory(StrEnum):
    """QUEKO benchmark circuit categories."""

    BIGD = "BIGD"
    BNTF = "BNTF"
    BSS = "BSS"


class QuekoRoutingMethod(StrEnum):
    """Quariadne routing methods for benchmarking."""

    QUARIADNE_ILP = "quariadne_ilp"
    QUARIADNE_LP = "quariadne_lp"


# Type aliases
type QuekoCircuitsByCategory = dict[QuekoCategory, list[Path]]


@dataclass
class BenchmarkTimingResult:
    """Container for individual circuit benchmark timing data.

    Stores the timing information for a single transpilation run of a QUEKO circuit
    using a specific routing method on a specific backend.

    Attributes:
        category: QUEKO circuit category (BIGD, BNTF, or BSS)
        circuit_name: Name of the circuit file without extension
        routing_method: Routing method used (quariadne_ilp or quariadne_lp)
        transpilation_time: Time taken for transpilation in seconds, or None if failed
    """

    category: QuekoCategory
    circuit_name: str
    routing_method: QuekoRoutingMethod
    transpilation_time: float | None


# Copied backend definitions
QUEKO_BACKEND_EDGES = {
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


def create_backend_coupling_map(backend_name: str) -> qiskit.transpiler.CouplingMap:
    """Create a Qiskit CouplingMap from backend edge list.

    Args:
        backend_name: Name of the backend (e.g., "Sycamore", "Rochester")

    Returns:
        Qiskit CouplingMap object with directed edges
    """
    edgelist_undirected = QUEKO_BACKEND_EDGES[backend_name]

    # Qiskit coupling graph needed to convert undirected edges to bidirectional directed edges
    graph_undirected = nx.from_edgelist(edgelist_undirected)
    edgelist_directed = graph_undirected.to_directed()
    # Qiskit requires list format, not NetworkX edge view
    directed_edgelist = list(edgelist_directed.edges())
    coupling_map = qiskit.transpiler.CouplingMap(directed_edgelist)
    return coupling_map


def get_queko_filepaths_by_category(
    queko_benchmark_dir: Path,
) -> QuekoCircuitsByCategory:
    """Get all QASM circuit filepaths from QUEKO-benchmark organized by category.

    Args:
        queko_benchmark_dir: Path to QUEKO-benchmark directory

    Returns:
        Dictionary mapping category enums to lists of circuit filepaths
    """
    queko_circuits_by_category: QuekoCircuitsByCategory = {}

    for queko_category in QuekoCategory:
        queko_category_path = queko_benchmark_dir / queko_category

        # QUEKO benchmark circuits are .qasm files; suffix filtering avoids non-circuit files
        # Sorting ensures reproducible benchmark order across different runs
        queko_category_circuits = sorted(
            path
            for path in queko_category_path.iterdir()
            if path.is_file() and path.suffix == QUEKO_QASM_EXTENSION
        )
        queko_circuits_by_category[queko_category] = queko_category_circuits
        logging.info(
            f"Found {len(queko_category_circuits)} circuits in {queko_category}"
        )

    queko_total_circuits = sum(
        len(circuits) for circuits in queko_circuits_by_category.values()
    )
    logging.info(f"Total circuits discovered: {queko_total_circuits}")

    return queko_circuits_by_category


def transpile_queko_circuit(
    queko_circuit_filepath: Path,
    queko_backend_coupling_map: qiskit.transpiler.CouplingMap,
    queko_routing_method: QuekoRoutingMethod,
) -> QuantumCircuit:
    """Transpile QUEKO circuit using specified routing method.

    Args:
        queko_circuit_filepath: Path to QASM circuit file
        queko_backend_coupling_map: Coupling map for target backend
        queko_routing_method: Routing method for transpilation (enum value)

    Returns:
        Transpiled QuantumCircuit
    """
    # Load circuit from QASM
    queko_qiskit_circuit = qiskit.qasm2.load(queko_circuit_filepath)

    # Create pass manager with routing method used for both routing and layout
    queko_pass_manager = qiskit.transpiler.generate_preset_pass_manager(
        coupling_map=queko_backend_coupling_map,
        optimization_level=QUEKO_DEFAULT_OPTIMISATION_LEVEL,
        routing_method=queko_routing_method,
        layout_method=queko_routing_method,
    )

    queko_transpiled_circuit = queko_pass_manager.run(queko_qiskit_circuit)
    return queko_transpiled_circuit


def run_queko_benchmarks(
    queko_circuits_by_category: QuekoCircuitsByCategory,
    queko_output_dir: Path,
) -> None:
    """Run benchmarks on all QUEKO circuits across all backends and methods.

    Args:
        queko_circuits_by_category: Dictionary of circuit filepaths by category enum
        queko_output_dir: Base output directory for results
    """
    # Benchmark each backend
    for queko_backend_name in QUEKO_BACKEND_EDGES.keys():
        logging.info(f"Starting benchmarks for backend: {queko_backend_name}")

        # Create coupling map for backend
        queko_backend_coupling_map = create_backend_coupling_map(queko_backend_name)

        # Create output directory for this backend
        queko_backend_output_dir = queko_output_dir / queko_backend_name
        queko_backend_output_dir.mkdir(parents=True, exist_ok=True)

        # Collect timing results for this backend
        queko_backend_timing_results: list[BenchmarkTimingResult] = []

        # Benchmark each category separately to preserve structure
        for queko_category, queko_circuit_paths in queko_circuits_by_category.items():
            # Create category subdirectory within backend directory
            queko_category_output_dir = queko_backend_output_dir / queko_category
            queko_category_output_dir.mkdir(parents=True, exist_ok=True)

            # Benchmark each circuit with each routing method
            for queko_circuit_path in queko_circuit_paths:
                queko_circuit_name = queko_circuit_path.stem

                for queko_routing_method in QuekoRoutingMethod:
                    # Time the transpilation using perf_counter for precision
                    queko_transpilation_start_time = time.perf_counter()
                    try:
                        queko_transpiled_circuit = transpile_queko_circuit(
                            queko_circuit_path,
                            queko_backend_coupling_map,
                            queko_routing_method,
                        )
                    except (
                        qiskit.transpiler.TranspilerError,
                        TimeoutError,
                        ValueError,
                    ) as error:
                        queko_transpiled_circuit = None
                        logging.error(
                            f"Failed {queko_category}/{queko_circuit_name} with {queko_routing_method}: "
                            f"{type(error).__name__}: {error}"
                        )
                    queko_transpilation_end_time = time.perf_counter()

                    if queko_transpiled_circuit is None:
                        queko_transpilation_time = None
                    else:
                        queko_transpilation_time = (
                            queko_transpilation_end_time
                            - queko_transpilation_start_time
                        )

                        # Save transpiled circuit in QPY format within category directory
                        queko_output_filepath = (
                            queko_category_output_dir
                            / f"{queko_routing_method}_{queko_circuit_name}{QUEKO_QPY_EXTENSION}"
                        )
                        with open(queko_output_filepath, "wb") as queko_qpy_file:
                            qiskit.qpy.dump(queko_transpiled_circuit, queko_qpy_file)

                        logging.info(
                            f"Completed {queko_category}/{queko_circuit_name} with {queko_routing_method} "
                            f"in {queko_transpilation_time:.4f}s"
                        )

                    # Record timing result with category information
                    queko_backend_timing_results.append(
                        BenchmarkTimingResult(
                            category=queko_category,
                            circuit_name=queko_circuit_name,
                            routing_method=queko_routing_method,
                            transpilation_time=queko_transpilation_time,
                        )
                    )

        # Save timing results for this backend at the backend level
        queko_timing_filepath = queko_backend_output_dir / "timing_results.json"
        with open(queko_timing_filepath, "w") as queko_timing_file:
            json.dump(
                [asdict(result) for result in queko_backend_timing_results],
                queko_timing_file,
                indent=2,
            )

        logging.info(f"Completed all benchmarks for {queko_backend_name}")


def main() -> None:
    """Main benchmarking function."""
    queko_benchmark_parser = argparse.ArgumentParser(
        description="Run QUEKO benchmarks for Quariadne router"
    )
    queko_benchmark_parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("QUEKO-benchmark"),
        help="Path to QUEKO-benchmark directory",
    )
    queko_benchmark_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benches/results"),
        help="Output directory for results",
    )
    queko_benchmark_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    queko_benchmark_args = queko_benchmark_parser.parse_args()

    # Configure logging with level from logging module
    queko_log_level_mapping = logging.getLevelNamesMapping()
    queko_benchmark_log_level = queko_log_level_mapping[queko_benchmark_args.log_level]
    logging.basicConfig(
        level=queko_benchmark_log_level,
        format=QUEKO_BENCHMARK_LOG_FORMAT,
    )

    logging.info("Starting QUEKO benchmarks")
    logging.info(f"Benchmark directory: {queko_benchmark_args.benchmark_dir}")
    logging.info(f"Output directory: {queko_benchmark_args.output_dir}")

    # Create output directory
    queko_benchmark_args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all QUEKO circuits
    queko_circuits_by_category = get_queko_filepaths_by_category(
        queko_benchmark_args.benchmark_dir
    )

    # Run benchmarks on all discovered circuits
    run_queko_benchmarks(queko_circuits_by_category, queko_benchmark_args.output_dir)

    logging.info("All QUEKO benchmarks completed successfully")


if __name__ == "__main__":
    main()
