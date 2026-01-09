#!/usr/bin/env python3
"""QUEKO benchmarking script for Quariadne router.

This script runs benchmarks on all QUEKO circuits with different backends,
measuring transpilation time and saving results for analysis.

References:
    - Python 3.13 argparse: https://docs.python.org/3/library/argparse.html
    - Python 3.13 logging: https://docs.python.org/3/library/logging.html
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import networkx as nx
import qiskit.qasm2
import qiskit.qpy
import qiskit.transpiler
from qiskit import QuantumCircuit

import quariadne.benchmarks.constants as bench_const


# Type aliases
type QuekoCircuitsByCategory = dict[bench_const.QuekoCategory, list[Path]]


@dataclass
class BenchmarkTimingResult:
    """Container for individual circuit benchmark timing data.

    Stores the timing information for a single transpilation run of a QUEKO circuit
    using a specific routing method on a specific backend.

    Attributes:
        category: QUEKO circuit category (BIGD, BNTF, or BSS)
        circuit_name: Name of the circuit file without extension
        routing_method: Routing method used
        transpilation_time: Time taken for transpilation in seconds, or None if failed
    """

    category: bench_const.QuekoCategory
    circuit_name: str
    routing_method: bench_const.RoutingMethod
    transpilation_time: float | None


QUEKO_BACKEND_EDGES = bench_const.BACKEND_EDGES


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

    for queko_category in bench_const.QuekoCategory:
        queko_category_path = queko_benchmark_dir / queko_category

        # QUEKO benchmark circuits are .qasm files; suffix filtering avoids non-circuit files
        # Sorting ensures reproducible benchmark order across different runs
        queko_category_circuits = sorted(
            path
            for path in queko_category_path.iterdir()
            if path.is_file() and path.suffix == bench_const.QASM_EXTENSION
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
    queko_routing_method: bench_const.RoutingMethod,
    queko_timeout_seconds: int | None = None,
) -> QuantumCircuit:
    """Transpile QUEKO circuit using specified routing method.

    Args:
        queko_circuit_filepath: Path to QASM circuit file
        queko_backend_coupling_map: Coupling map for target backend
        queko_routing_method: Routing method for transpilation (enum value)
        queko_timeout_seconds: Timeout in seconds for routing algorithm.
            Only used by Quariadne methods; Qiskit methods ignore this.

    Returns:
        Transpiled QuantumCircuit
    """
    # Load circuit from QASM
    queko_qiskit_circuit = qiskit.qasm2.load(queko_circuit_filepath)

    # Create pass manager with routing and matching layout method
    # Reference: https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.generate_preset_pass_manager
    queko_layout_method = queko_routing_method.layout_method
    queko_pass_manager = qiskit.transpiler.generate_preset_pass_manager(
        coupling_map=queko_backend_coupling_map,
        optimization_level=bench_const.DEFAULT_OPTIMISATION_LEVEL,
        routing_method=queko_routing_method,
        layout_method=queko_layout_method,
    )

    queko_transpiled_circuit = queko_pass_manager.run(queko_qiskit_circuit)
    return queko_transpiled_circuit


EMPTY_TIMING_RESULTS: list[BenchmarkTimingResult] = []


def load_existing_timing_results(
    queko_timing_filepath: Path,
) -> list[BenchmarkTimingResult]:
    """Load existing timing results from JSON file for resume capability.

    Args:
        queko_timing_filepath: Path to timing_results.json file.

    Returns:
        List of BenchmarkTimingResult objects, empty list if file doesn't exist.

    References:
        - Python 3.13 json: https://docs.python.org/3/library/json.html
    """
    if queko_timing_filepath.exists():
        with open(queko_timing_filepath) as queko_timing_file:
            queko_existing_results = json.load(queko_timing_file)
        logging.info(f"Loaded {len(queko_existing_results)} existing results")
        queko_timing_results = [
            BenchmarkTimingResult(**result) for result in queko_existing_results
        ]
    else:
        queko_timing_results = EMPTY_TIMING_RESULTS

    return queko_timing_results


def save_timing_results(
    queko_timing_filepath: Path,
    queko_timing_results: list[BenchmarkTimingResult],
) -> None:
    """Save timing results to JSON file for persistence.

    Args:
        queko_timing_filepath: Path to timing_results.json file.
        queko_timing_results: List of BenchmarkTimingResult objects to save.

    References:
        - Python 3.13 json: https://docs.python.org/3/library/json.html
    """
    queko_results_as_dicts = [asdict(result) for result in queko_timing_results]

    with open(queko_timing_filepath, "w") as queko_timing_file:
        json.dump(
            queko_results_as_dicts,
            queko_timing_file,
            indent=bench_const.JSON_INDENT,
        )


def run_queko_benchmarks(
    queko_circuits_by_category: QuekoCircuitsByCategory,
    queko_output_dir: Path,
    methods: list[bench_const.RoutingMethod] | None = None,
) -> None:
    """Run benchmarks on all QUEKO circuits across all backends and methods.

    Args:
        queko_circuits_by_category: Dictionary of circuit filepaths by category enum
        queko_output_dir: Base output directory for results
        methods: List of routing methods to benchmark. If None, runs all methods.
    """
    methods_to_run = methods or list(bench_const.RoutingMethod)
    # Benchmark each backend
    for queko_backend_name in QUEKO_BACKEND_EDGES.keys():
        logging.info(f"Starting benchmarks for backend: {queko_backend_name}")

        # Create coupling map for backend
        queko_backend_coupling_map = create_backend_coupling_map(queko_backend_name)

        # Create output directory for this backend
        queko_backend_output_dir = queko_output_dir / queko_backend_name
        queko_backend_output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing timing results for resume capability
        queko_timing_filepath = queko_backend_output_dir / "timing_results.json"
        queko_backend_timing_results = load_existing_timing_results(
            queko_timing_filepath
        )

        # Benchmark each category separately to preserve structure
        for queko_category, queko_circuit_paths in queko_circuits_by_category.items():
            # Create category subdirectory within backend directory
            queko_category_output_dir = queko_backend_output_dir / queko_category
            queko_category_output_dir.mkdir(parents=True, exist_ok=True)

            # Benchmark each circuit with each routing method
            for queko_circuit_path in queko_circuit_paths:
                queko_circuit_name = queko_circuit_path.stem

                for queko_routing_method in methods_to_run:
                    # Check if already completed (QPY file exists)
                    queko_output_filepath = (
                        queko_category_output_dir
                        / f"{queko_routing_method}_{queko_circuit_name}{bench_const.QPY_EXTENSION}"
                    )
                    if queko_output_filepath.exists():
                        logging.info(
                            f"SKIP: {queko_category}/{queko_circuit_name} with {queko_routing_method} "
                            f"(already completed)"
                        )
                        continue

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

                        with open(queko_output_filepath, "wb") as queko_qpy_file:
                            qiskit.qpy.dump(queko_transpiled_circuit, queko_qpy_file)

                        logging.info(
                            f"Completed {queko_category}/{queko_circuit_name} with {queko_routing_method} "
                            f"in {queko_transpilation_time:.4f}s"
                        )

                    # Record timing result and save incrementally for persistence
                    queko_backend_timing_results.append(
                        BenchmarkTimingResult(
                            category=queko_category,
                            circuit_name=queko_circuit_name,
                            routing_method=queko_routing_method,
                            transpilation_time=queko_transpilation_time,
                        )
                    )
                    save_timing_results(
                        queko_timing_filepath, queko_backend_timing_results
                    )

        # Final save for this backend
        save_timing_results(queko_timing_filepath, queko_backend_timing_results)

        logging.info(f"Completed all benchmarks for {queko_backend_name}")


def run_full_benchmark(
    queko_benchmark_dir: Path,
    queko_output_dir: Path,
    methods: list[bench_const.RoutingMethod] | None = None,
) -> None:
    """Run benchmarks on all QUEKO circuits.

    Args:
        queko_benchmark_dir: Path to QUEKO-benchmark directory.
        queko_output_dir: Output directory for results.
        methods: List of routing methods to benchmark. If None, runs all methods.

    References:
        - Python 3.13 pathlib: https://docs.python.org/3/library/pathlib.html
    """
    methods_to_run = methods or list(bench_const.RoutingMethod)

    logging.info("Starting QUEKO benchmarks (full mode)")
    logging.info(f"Benchmark directory: {queko_benchmark_dir}")
    logging.info(f"Output directory: {queko_output_dir}")
    logging.info(f"Methods: {[m.value for m in methods_to_run]}")

    queko_output_dir.mkdir(parents=True, exist_ok=True)

    queko_circuits_by_category = get_queko_filepaths_by_category(queko_benchmark_dir)
    run_queko_benchmarks(queko_circuits_by_category, queko_output_dir, methods_to_run)

    logging.info("All QUEKO benchmarks completed successfully")


def run_single_benchmark(
    queko_circuit_path: Path,
    queko_backend_name: str,
    queko_output_dir: Path,
    queko_routing_method: bench_const.RoutingMethod,
    queko_timeout_seconds: int,
) -> None:
    """Run benchmark on a single circuit with one routing method.

    Args:
        queko_circuit_path: Path to QASM circuit file.
        queko_backend_name: Target backend name.
        queko_output_dir: Output directory for results.
        queko_routing_method: Routing method to use.
        queko_timeout_seconds: Timeout in seconds for routing algorithm.

    References:
        - Python 3.13 pathlib: https://docs.python.org/3/library/pathlib.html
    """
    logging.info("Starting single circuit benchmark")
    logging.info(f"Circuit: {queko_circuit_path}")
    logging.info(f"Backend: {queko_backend_name}")
    logging.info(f"Method: {queko_routing_method}")

    queko_output_dir.mkdir(parents=True, exist_ok=True)

    queko_backend_coupling_map = create_backend_coupling_map(queko_backend_name)
    queko_circuit_name = queko_circuit_path.stem

    queko_transpilation_start_time = time.perf_counter()
    queko_transpiled_circuit = None

    try:
        queko_transpiled_circuit = transpile_queko_circuit(
            queko_circuit_path,
            queko_backend_coupling_map,
            queko_routing_method,
            queko_timeout_seconds,
        )
    except TimeoutError:
        logging.warning(
            f"TIMEOUT: {queko_circuit_name} with {queko_routing_method} "
            f"exceeded time limit"
        )
    except qiskit.transpiler.TranspilerError as error:
        logging.error(
            f"TRANSPILER: {queko_circuit_name} with {queko_routing_method}: {error}"
        )
    except ValueError as error:
        logging.error(
            f"VALUE: {queko_circuit_name} with {queko_routing_method}: {error}"
        )

    queko_transpilation_end_time = time.perf_counter()

    if queko_transpiled_circuit is not None:
        queko_transpilation_time = (
            queko_transpilation_end_time - queko_transpilation_start_time
        )

        queko_output_filepath = (
            queko_output_dir
            / f"{queko_routing_method}_{queko_circuit_name}{bench_const.QPY_EXTENSION}"
        )
        with open(queko_output_filepath, "wb") as queko_qpy_file:
            qiskit.qpy.dump(queko_transpiled_circuit, queko_qpy_file)

        logging.info(
            f"Completed {queko_circuit_name} with {queko_routing_method} "
            f"in {queko_transpilation_time:.4f}s"
        )

    logging.info("Single circuit benchmark completed")


def main() -> None:
    """Main entry point with subcommand dispatch.

    References:
        - Python 3.13 argparse subparsers: https://docs.python.org/3/library/argparse.html#sub-commands
    """
    queko_benchmark_parser = argparse.ArgumentParser(
        description=bench_const.HELP_DESCRIPTION
    )

    # Common arguments for all subcommands
    queko_benchmark_parser.add_argument(
        "--output-dir",
        type=Path,
        default=bench_const.DEFAULT_OUTPUT_DIR,
        help=bench_const.HELP_OUTPUT_DIR,
    )
    queko_benchmark_parser.add_argument(
        "--log-level",
        default=bench_const.BENCHMARK_DEFAULT_LOG_LEVEL,
        choices=bench_const.BENCHMARK_LOG_LEVELS,
        help=bench_const.HELP_LOG_LEVEL,
    )

    # Subcommand parsers
    queko_subparsers = queko_benchmark_parser.add_subparsers(
        dest="subcommand",
        help=bench_const.HELP_SUBCOMMAND,
    )

    # Full benchmark mode
    queko_full_parser = queko_subparsers.add_parser(
        "full",
        help=bench_const.HELP_FULL_MODE,
    )
    queko_full_parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=bench_const.DEFAULT_BENCHMARK_DIR,
        help=bench_const.HELP_BENCHMARK_DIR,
    )
    queko_full_parser.add_argument(
        "--methods",
        type=bench_const.RoutingMethod,
        nargs="+",
        default=list(bench_const.RoutingMethod),
        choices=list(bench_const.RoutingMethod),
        help="Routing methods to benchmark (default: all)",
    )

    # Single circuit mode (handler to be implemented)
    queko_single_parser = queko_subparsers.add_parser(
        "single",
        help=bench_const.HELP_SINGLE_MODE,
    )
    queko_single_parser.add_argument(
        "--circuit",
        type=Path,
        required=True,
        help=bench_const.HELP_CIRCUIT,
    )
    queko_single_parser.add_argument(
        "--backend",
        type=str,
        default=bench_const.DEFAULT_BACKEND,
        choices=bench_const.AVAILABLE_BACKENDS,
        help=bench_const.HELP_BACKEND,
    )
    queko_single_parser.add_argument(
        "--method",
        type=bench_const.RoutingMethod,
        required=True,
        choices=list(bench_const.RoutingMethod),
        help=bench_const.HELP_METHOD,
    )
    queko_single_parser.add_argument(
        "--timeout",
        type=int,
        default=bench_const.DEFAULT_TIMEOUT_SECONDS,
        help=bench_const.HELP_TIMEOUT,
    )

    queko_benchmark_args = queko_benchmark_parser.parse_args()

    # Configure logging
    queko_log_level_mapping = logging.getLevelNamesMapping()
    queko_benchmark_log_level = queko_log_level_mapping[queko_benchmark_args.log_level]
    logging.basicConfig(
        level=queko_benchmark_log_level,
        format=bench_const.BENCHMARK_LOG_FORMAT,
    )

    # Dispatch to appropriate handler
    if queko_benchmark_args.subcommand == "full":
        run_full_benchmark(
            queko_benchmark_args.benchmark_dir,
            queko_benchmark_args.output_dir,
            methods=queko_benchmark_args.methods,
        )
    elif queko_benchmark_args.subcommand == "single":
        run_single_benchmark(
            queko_benchmark_args.circuit,
            queko_benchmark_args.backend,
            queko_benchmark_args.output_dir,
            queko_benchmark_args.method,
            queko_benchmark_args.timeout,
        )
    else:
        queko_benchmark_parser.print_help()


if __name__ == "__main__":
    main()
