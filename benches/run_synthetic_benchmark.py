#!/usr/bin/env python3
"""Synthetic benchmark runner for thesis scaling study.

This script generates circuits of 7 types on Watts-Strogatz topologies
for qubits 5-12, runs 5 routers, and outputs pgfplots-compatible .dat files.

Usage:
    python benches/run_synthetic_benchmark.py --help
    python benches/run_synthetic_benchmark.py --output-dir results/synthetic

References:
    - Python 3.13 argparse: https://docs.python.org/3/library/argparse.html
    - Thesis Section 4.3: Experimental Setup
"""

import argparse
import dataclasses
import logging
import time
from pathlib import Path

import pandas
import qiskit.transpiler

import quariadne.benchmarks.constants as bench_const
import quariadne.benchmarks.circuit_generators as circuit_gen
import quariadne.benchmarks.topology_generators as topo_gen


# Metric name constants
METRIC_TIME = "time"
METRIC_GATES = "gates"
METRIC_RATE = "rate"
METRIC_SWAPS = "swaps"

# Column name for .dat files
COLUMN_QUBIT_COUNT = "qubit_count"

# File naming (matches thesis LaTeX: {metric}_{circuit_type}_WS.dat)
DAT_FILE_SUFFIX = "WS"
DAT_SEPARATOR = "\t"

# Gate operation names
GATE_OP_SWAP = "swap"

# Default values
NAN_VALUE = float("nan")
ZERO_SWAPS = 0


@dataclasses.dataclass
class BenchmarkMetrics:
    """Metrics collected from a single benchmark run.

    Attributes:
        time: Execution time in seconds.
        gates: Final gate count after routing.
        rate: Gate increase ratio (final/original).
        swaps: Number of SWAP gates inserted.
    """

    time: float
    gates: float
    rate: float
    swaps: float

    @classmethod
    def failed(cls) -> "BenchmarkMetrics":
        """Create metrics instance for a failed run."""
        return cls(
            time=NAN_VALUE,
            gates=NAN_VALUE,
            rate=NAN_VALUE,
            swaps=NAN_VALUE,
        )


# Default output directory for synthetic benchmark results
DEFAULT_SYNTHETIC_OUTPUT_DIR = Path("benches/results/synthetic")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run synthetic scaling benchmarks for thesis",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SYNTHETIC_OUTPUT_DIR,
        help="Output directory for .dat files",
    )
    parser.add_argument(
        "--log-level",
        default=bench_const.BENCHMARK_DEFAULT_LOG_LEVEL,
        choices=bench_const.BENCHMARK_LOG_LEVELS,
        help="Logging level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=bench_const.SCALING_SEED_BASE,
        help="Base random seed for reproducibility",
    )

    return parser.parse_args()


def transpile_circuit(
    circuit: qiskit.circuit.QuantumCircuit,
    coupling_map: qiskit.transpiler.CouplingMap,
    routing_method: bench_const.RoutingMethod,
) -> tuple[qiskit.circuit.QuantumCircuit | None, float]:
    """Transpile circuit with specified routing method.

    Args:
        circuit: Input quantum circuit.
        coupling_map: Target hardware coupling map.
        routing_method: Routing method to use.

    Returns:
        Tuple of (transpiled circuit or None if failed, execution time in seconds).
    """
    layout_method = routing_method.layout_method

    pass_manager = qiskit.transpiler.generate_preset_pass_manager(
        coupling_map=coupling_map,
        optimization_level=bench_const.DEFAULT_OPTIMISATION_LEVEL,
        routing_method=routing_method.value,
        layout_method=layout_method,
    )

    start_time = time.perf_counter()
    try:
        transpiled = pass_manager.run(circuit)
    except (qiskit.transpiler.TranspilerError, TimeoutError, ValueError) as error:
        logging.warning(f"Transpilation failed: {error}")
        transpiled = None
    elapsed_time = time.perf_counter() - start_time

    return transpiled, elapsed_time


def generate_circuit_for_type(
    circuit_type: circuit_gen.CircuitType,
    num_qubits: int,
    seed: int,
) -> qiskit.circuit.QuantumCircuit:
    """Generate a circuit of the specified type.

    Args:
        circuit_type: Type of circuit to generate.
        num_qubits: Number of qubits.
        seed: Random seed for reproducible generation.

    Returns:
        Generated QuantumCircuit.
    """
    num_gates = num_qubits * bench_const.DEFAULT_GATES_PER_QUBIT
    num_layers = bench_const.DEFAULT_GATES_PER_QUBIT

    match circuit_type:
        case circuit_gen.CircuitType.RANDOM_CLIFFORD:
            return circuit_gen.generate_random_clifford(num_qubits, num_gates, seed)

        case circuit_gen.CircuitType.DEPTH_CONTROLLED:
            depth = num_qubits * bench_const.DEPTH_MULTIPLIER_DEFAULT
            return circuit_gen.generate_depth_controlled(num_qubits, depth, seed)

        case circuit_gen.CircuitType.FULL_LAYER:
            return circuit_gen.generate_full_layer(
                num_qubits, bench_const.FULL_LAYER_DEFAULT_LAYERS
            )

        case circuit_gen.CircuitType.LINEAR_CHAIN:
            return circuit_gen.generate_linear_chain(num_qubits, num_layers)

        case circuit_gen.CircuitType.RING:
            return circuit_gen.generate_ring(num_qubits, num_layers)

        case circuit_gen.CircuitType.STAR:
            return circuit_gen.generate_star(num_qubits, num_layers)

        case circuit_gen.CircuitType.QFT_LIKE:
            return circuit_gen.generate_qft_like(num_qubits, num_layers)


def extract_metrics(
    transpiled: qiskit.circuit.QuantumCircuit | None,
    original_gate_count: int,
    elapsed_time: float,
) -> BenchmarkMetrics:
    """Extract benchmark metrics from transpiled circuit.

    Args:
        transpiled: Transpiled circuit, or None if failed.
        original_gate_count: Gate count before transpilation.
        elapsed_time: Transpilation time in seconds.

    Returns:
        BenchmarkMetrics instance with extracted values.
    """
    if transpiled is None:
        return BenchmarkMetrics.failed()

    final_gate_count = transpiled.size()
    gate_rate = final_gate_count / original_gate_count
    swap_count = transpiled.count_ops().get(GATE_OP_SWAP, ZERO_SWAPS)

    return BenchmarkMetrics(
        time=elapsed_time,
        gates=final_gate_count,
        rate=gate_rate,
        swaps=swap_count,
    )


def build_dat_filename(
    metric: str,
    circuit_type: circuit_gen.CircuitType,
) -> str:
    """Build .dat filename matching thesis LaTeX format.

    Args:
        metric: Metric name (time, gates, rate, swaps).
        circuit_type: Circuit type enum.

    Returns:
        Filename like "gates_random_clifford_WS.dat".
    """
    return f"{metric}_{circuit_type.value}_{DAT_FILE_SUFFIX}.dat"


def write_results_dataframe(
    dataframe: pandas.DataFrame,
    output_dir: Path,
    metric: str,
    circuit_type: circuit_gen.CircuitType,
) -> Path:
    """Write benchmark results DataFrame to .dat file.

    Args:
        dataframe: Results with qubit_count index and router columns.
        output_dir: Directory for output files.
        metric: Metric name for filename.
        circuit_type: Circuit type for filename.

    Returns:
        Path to written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = build_dat_filename(metric, circuit_type)
    filepath = output_dir / filename

    dataframe.to_csv(filepath, sep=DAT_SEPARATOR, index=True)

    return filepath


def create_results_dataframes(
    qubit_counts: list[int],
) -> dict[str, pandas.DataFrame]:
    """Create empty DataFrames for each metric.

    Args:
        qubit_counts: List of qubit counts for index.

    Returns:
        Dictionary mapping metric name to DataFrame.
    """
    router_columns = [method.column_name for method in bench_const.ENABLED_ROUTERS]

    dataframes = {}
    for metric in [METRIC_TIME, METRIC_GATES, METRIC_RATE, METRIC_SWAPS]:
        df = pandas.DataFrame(index=qubit_counts, columns=router_columns)
        df.index.name = COLUMN_QUBIT_COUNT
        dataframes[metric] = df

    return dataframes


def benchmark_single_qubit_count(
    circuit_type: circuit_gen.CircuitType,
    num_qubits: int,
    base_seed: int,
    dataframes: dict[str, pandas.DataFrame],
) -> None:
    """Run benchmark for one qubit count, updating dataframes in place.

    Args:
        circuit_type: Type of circuit to benchmark.
        num_qubits: Number of qubits for this run.
        base_seed: Base random seed.
        dataframes: Dictionary of DataFrames to update.
    """
    # Generate topology
    seed_topology = base_seed + num_qubits
    coupling_map = topo_gen.generate_watts_strogatz_coupling_map(
        num_qubits,
        seed=seed_topology,
    )

    # Generate circuit
    seed_circuit = base_seed + num_qubits * 100
    circuit = generate_circuit_for_type(circuit_type, num_qubits, seed_circuit)
    original_gate_count = circuit.size()

    # Run each enabled router
    for routing_method in bench_const.ENABLED_ROUTERS:
        column_name = routing_method.column_name

        transpiled, elapsed = transpile_circuit(circuit, coupling_map, routing_method)
        metrics = extract_metrics(transpiled, original_gate_count, elapsed)

        dataframes[METRIC_TIME].loc[num_qubits, column_name] = metrics.time
        dataframes[METRIC_GATES].loc[num_qubits, column_name] = metrics.gates
        dataframes[METRIC_RATE].loc[num_qubits, column_name] = metrics.rate
        dataframes[METRIC_SWAPS].loc[num_qubits, column_name] = metrics.swaps

        status = "OK" if transpiled is not None else "FAILED"
        logging.info(f"    {column_name}: {status} ({elapsed:.2f}s)")


def run_benchmark_for_circuit_type(
    circuit_type: circuit_gen.CircuitType,
    output_dir: Path,
    base_seed: int,
) -> None:
    """Run full benchmark for one circuit type.

    Args:
        circuit_type: Type of circuit to benchmark.
        output_dir: Directory for output .dat files.
        base_seed: Base random seed for reproducibility.
    """
    logging.info(f"Benchmarking circuit type: {circuit_type.value}")

    dataframes = create_results_dataframes(bench_const.COMPREHENSIVE_QUBIT_COUNTS)

    for num_qubits in bench_const.COMPREHENSIVE_QUBIT_COUNTS:
        logging.info(f"  Qubits: {num_qubits}")
        benchmark_single_qubit_count(circuit_type, num_qubits, base_seed, dataframes)

    # Write .dat files
    for metric in [METRIC_TIME, METRIC_GATES, METRIC_RATE, METRIC_SWAPS]:
        write_results_dataframe(dataframes[metric], output_dir, metric, circuit_type)

    logging.info(f"  Wrote 4 .dat files for {circuit_type.value}")


def main() -> None:
    """Run synthetic benchmarks for all circuit types."""
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=bench_const.BENCHMARK_LOG_FORMAT,
    )

    logging.info("Starting synthetic benchmark suite")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Qubit counts: {bench_const.COMPREHENSIVE_QUBIT_COUNTS}")
    logging.info(f"Circuit types: {len(circuit_gen.CircuitType)}")
    logging.info(
        f"Enabled routers: {[r.column_name for r in bench_const.ENABLED_ROUTERS]}"
    )

    for circuit_type in circuit_gen.CircuitType:
        run_benchmark_for_circuit_type(circuit_type, args.output_dir, args.seed)

    logging.info("Benchmark suite complete")


if __name__ == "__main__":
    main()
