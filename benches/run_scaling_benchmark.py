#!/usr/bin/env python3
"""Scaling benchmarks for Quariadne routing algorithms.

Compares LpRouterEdges, LpRouterMapping, BipartiteAllocationRouter,
and Qiskit SABRE across different circuit sizes and types on Ourense topology.

Output files are in .dat format for direct use with pgfplots.

References:
    - Python 3.13 time.perf_counter: https://docs.python.org/3/library/time.html#time.perf_counter
    - Python 3.13 dataclasses: https://docs.python.org/3/library/dataclasses.html
    - Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
    - pgfplots manual: https://ctan.org/pkg/pgfplots

Usage:
    python benches/run_scaling_benchmark.py --output-dir benches/results/scaling
"""

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Callable

import networkx as nx
import qiskit
import qiskit.transpiler

import quariadne.benchmarks.circuit_generators as circuit_gen
import quariadne.benchmarks.constants as bench_const
import quariadne.bipartite_allocation_lp
import quariadne.circuit
import quariadne.milp
import quariadne.routers


# Type aliases for benchmark result serialisation
# Ref: https://docs.python.org/3/library/typing.html#type-aliases
type SerializedResult = dict[str, object]
type SampleResults = list[SerializedResult]
type QubitResults = dict[int, list[SampleResults]]
type BenchmarkSummary = dict[str, QubitResults]


# Logging configuration
logging.basicConfig(
    format=bench_const.BENCHMARK_LOG_FORMAT,
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a single benchmark run.

    Attributes:
        qubit_count: Number of qubits in the circuit.
        circuit_type: Type of circuit ("random_clifford", "linear_chain", "ring", "star").
        sample_index: Index of this sample (0 to samples_per_config - 1).
        backend_name: Name of the hardware backend topology.
    """

    qubit_count: int
    circuit_type: str
    sample_index: int
    backend_name: str


@dataclasses.dataclass
class RouterBenchmarkResult:
    """Result from timing a single router on a single circuit.

    Attributes:
        router_name: Name of the router algorithm.
        execution_time_seconds: Wall-clock time for routing.
        original_gate_count: Number of two-qubit gates before routing.
        final_gate_count: Number of two-qubit gates after routing (including SWAPs).
        swap_count: Number of SWAP operations inserted.
        success: Whether routing completed without error.
        error_message: Error message if routing failed.
    """

    router_name: str
    execution_time_seconds: float
    original_gate_count: int
    final_gate_count: int
    swap_count: int
    success: bool
    error_message: str | None = None

    @property
    def gate_increase_rate(self) -> float:
        """Calculate gate count increase ratio (final / original)."""
        if self.original_gate_count == 0:
            return 0.0
        return self.final_gate_count / self.original_gate_count


# SWAP operation is equivalent to 3 CX gates in standard decomposition
SWAP_TO_CX_COUNT = 3


def time_quariadne_router(
    router_class: type,
    coupling_graph: nx.DiGraph,
    circuit: quariadne.circuit.AbstractQuantumCircuit,
) -> RouterBenchmarkResult:
    """Time a Quariadne router (LpRouterEdges, LpRouterMapping, IlpRouter).

    Args:
        router_class: Router class to instantiate.
        coupling_graph: NetworkX DiGraph of physical connectivity.
        circuit: AbstractQuantumCircuit to route.

    Returns:
        RouterBenchmarkResult with timing and gate count data.
    """
    original_gate_count = len(circuit.get_two_qubit_operations())

    start_time = time.perf_counter()
    try:
        router = router_class(coupling_graph, circuit)
        swap_dict = router.get_inserted_swaps()
        swap_count = sum(len(swaps) for swaps in swap_dict.values())
        end_time = time.perf_counter()

        final_gate_count = original_gate_count + (swap_count * SWAP_TO_CX_COUNT)

        return RouterBenchmarkResult(
            router_name=router_class.__name__,
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_gate_count,
            final_gate_count=final_gate_count,
            swap_count=swap_count,
            success=True,
        )
    except Exception as e:
        end_time = time.perf_counter()
        logger.warning(f"Router {router_class.__name__} failed: {e}")
        return RouterBenchmarkResult(
            router_name=router_class.__name__,
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_gate_count,
            final_gate_count=0,
            swap_count=0,
            success=False,
            error_message=str(e),
        )


def time_bipartite_router(
    coupling_graph: nx.DiGraph,
    circuit: quariadne.circuit.AbstractQuantumCircuit,
) -> RouterBenchmarkResult:
    """Time the BipartiteAllocationRouter (different API from base Router).

    Args:
        coupling_graph: NetworkX DiGraph of physical connectivity.
        circuit: AbstractQuantumCircuit to route.

    Returns:
        RouterBenchmarkResult with timing and gate count data.
    """
    original_gate_count = len(circuit.get_two_qubit_operations())

    start_time = time.perf_counter()
    try:
        router = quariadne.bipartite_allocation_lp.BipartiteAllocationRouter(
            coupling_graph, circuit
        )
        result = router.run()
        swap_count = sum(len(swaps) for swaps in result.inserted_swaps.values())
        end_time = time.perf_counter()

        final_gate_count = original_gate_count + (swap_count * SWAP_TO_CX_COUNT)

        return RouterBenchmarkResult(
            router_name="BipartiteAllocationRouter",
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_gate_count,
            final_gate_count=final_gate_count,
            swap_count=swap_count,
            success=True,
        )
    except Exception as e:
        end_time = time.perf_counter()
        logger.warning(f"BipartiteAllocationRouter failed: {e}")
        return RouterBenchmarkResult(
            router_name="BipartiteAllocationRouter",
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_gate_count,
            final_gate_count=0,
            swap_count=0,
            success=False,
            error_message=str(e),
        )


def time_sabre_router(
    coupling_map: qiskit.transpiler.CouplingMap,
    qiskit_circuit: qiskit.QuantumCircuit,
) -> RouterBenchmarkResult:
    """Time Qiskit SABRE routing algorithm.

    Args:
        coupling_map: Qiskit CouplingMap for target backend.
        qiskit_circuit: Qiskit QuantumCircuit to route.

    Returns:
        RouterBenchmarkResult with timing and gate count data.
    """
    original_two_qubit_count = qiskit_circuit.num_nonlocal_gates()

    start_time = time.perf_counter()
    try:
        pass_manager = qiskit.transpiler.generate_preset_pass_manager(
            coupling_map=coupling_map,
            optimization_level=bench_const.DEFAULT_OPTIMISATION_LEVEL,
            routing_method="sabre",
            layout_method="sabre",
        )
        transpiled = pass_manager.run(qiskit_circuit)
        end_time = time.perf_counter()

        final_two_qubit_count = transpiled.num_nonlocal_gates()
        swap_count = transpiled.count_ops().get("swap", 0)

        return RouterBenchmarkResult(
            router_name="SABRE",
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_two_qubit_count,
            final_gate_count=final_two_qubit_count,
            swap_count=swap_count,
            success=True,
        )
    except Exception as e:
        end_time = time.perf_counter()
        logger.warning(f"SABRE failed: {e}")
        return RouterBenchmarkResult(
            router_name="SABRE",
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_two_qubit_count,
            final_gate_count=0,
            swap_count=0,
            success=False,
            error_message=str(e),
        )


def generate_circuit_pair(
    config: BenchmarkConfig,
) -> tuple[qiskit.QuantumCircuit, quariadne.circuit.AbstractQuantumCircuit]:
    """Generate both Qiskit and AbstractQuantumCircuit for a benchmark config.

    Args:
        config: Benchmark configuration specifying circuit parameters.

    Returns:
        Tuple of (Qiskit QuantumCircuit, AbstractQuantumCircuit).

    Raises:
        ValueError: If circuit_type is not recognised.
    """
    seed = bench_const.SCALING_SEED_BASE + config.sample_index
    num_qubits = config.qubit_count

    if config.circuit_type == "random_clifford":
        num_gates = num_qubits * bench_const.DEFAULT_GATES_PER_QUBIT
        qiskit_circuit = qiskit.circuit.random.random_clifford_circuit(
            num_qubits=num_qubits,
            num_gates=num_gates,
            gates=circuit_gen.CLIFFORD_GATE_SET,
            seed=seed,
        )
    elif config.circuit_type == "linear_chain":
        chain_length = num_qubits * 2
        qiskit_circuit = qiskit.QuantumCircuit(num_qubits)
        for i in range(chain_length):
            control = i % (num_qubits - 1)
            target = control + 1
            qiskit_circuit.cx(control, target)
    elif config.circuit_type == "ring":
        qiskit_circuit = qiskit.QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            control = i
            target = (i + 1) % num_qubits
            qiskit_circuit.cx(control, target)
    elif config.circuit_type == "star":
        hub_qubit = 0
        qiskit_circuit = qiskit.QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            if i != hub_qubit:
                qiskit_circuit.cx(hub_qubit, i)
    else:
        raise ValueError(f"Unknown circuit type: {config.circuit_type}")

    abstract_circuit = quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(
        qiskit_circuit
    )
    return qiskit_circuit, abstract_circuit


# Router names in output order (matches column order in .dat files)
ROUTER_NAMES = [
    "LpRouterEdges",
    "LpRouterMapping",
    "BipartiteAllocationRouter",
    "SABRE",
]


def run_single_benchmark(
    config: BenchmarkConfig,
    coupling_graph: nx.DiGraph,
    coupling_map: qiskit.transpiler.CouplingMap,
) -> list[RouterBenchmarkResult]:
    """Run all routers on a single circuit configuration.

    Args:
        config: Benchmark configuration.
        coupling_graph: NetworkX DiGraph for Quariadne routers.
        coupling_map: Qiskit CouplingMap for SABRE.

    Returns:
        List of RouterBenchmarkResult, one per router.
    """
    qiskit_circuit, abstract_circuit = generate_circuit_pair(config)

    results = []

    # Quariadne routers using Router interface
    for router_class in [
        quariadne.routers.LpRouterEdges,
        quariadne.routers.LpRouterMapping,
    ]:
        result = time_quariadne_router(router_class, coupling_graph, abstract_circuit)
        results.append(result)

    # BipartiteAllocationRouter (different API)
    bipartite_result = time_bipartite_router(coupling_graph, abstract_circuit)
    results.append(bipartite_result)

    # SABRE (Qiskit)
    sabre_result = time_sabre_router(coupling_map, qiskit_circuit)
    results.append(sabre_result)

    return results


def write_dat_file(
    filepath: Path,
    header_comment: str,
    column_names: list[str],
    data_rows: list[list[float]],
) -> None:
    """Write data in pgfplots-compatible .dat format.

    Args:
        filepath: Output file path.
        header_comment: Description comment for the file.
        column_names: Column header names.
        data_rows: List of data rows (each row is a list of values).

    Reference:
        - pgfplots manual: https://ctan.org/pkg/pgfplots
    """
    with open(filepath, "w") as f:
        f.write(f"# {header_comment}\n")
        f.write(f"# Columns: {', '.join(column_names)}\n")
        for row in data_rows:
            formatted_row = "\t".join(f"{val:.6f}" for val in row)
            f.write(f"{formatted_row}\n")


def run_full_benchmark(
    backend_name: str,
    output_dir: Path,
    samples_per_config: int = bench_const.DEFAULT_SAMPLES_PER_CONFIG,
) -> None:
    """Run the full scaling benchmark across all qubit counts and circuit types.

    Args:
        backend_name: Name of the backend topology.
        output_dir: Directory to write output files.
        samples_per_config: Number of samples per configuration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create coupling structures once
    coupling_map = bench_const.create_backend_coupling_map(backend_name)
    coupling_graph = quariadne.milp.get_coupling_graph(coupling_map)

    # Storage for aggregated results
    # results_by_type[circuit_type][qubit_count] = list of sample results
    results_by_type: dict[str, dict[int, list[list[RouterBenchmarkResult]]]] = {
        circuit_type: {qc: [] for qc in bench_const.SCALING_QUBIT_COUNTS}
        for circuit_type in bench_const.CIRCUIT_TYPES
    }

    total_configs = (
        len(bench_const.SCALING_QUBIT_COUNTS)
        * len(bench_const.CIRCUIT_TYPES)
        * samples_per_config
    )
    current_config = 0

    for circuit_type in bench_const.CIRCUIT_TYPES:
        for qubit_count in bench_const.SCALING_QUBIT_COUNTS:
            for sample_idx in range(samples_per_config):
                current_config += 1
                config = BenchmarkConfig(
                    qubit_count=qubit_count,
                    circuit_type=circuit_type,
                    sample_index=sample_idx,
                    backend_name=backend_name,
                )

                logger.info(
                    f"[{current_config}/{total_configs}] "
                    f"{circuit_type} q={qubit_count} sample={sample_idx}"
                )

                results = run_single_benchmark(config, coupling_graph, coupling_map)
                results_by_type[circuit_type][qubit_count].append(results)

    # Write .dat files for each circuit type and metric
    _write_all_dat_files(output_dir, results_by_type)

    # Write full JSON summary
    _write_json_summary(output_dir, results_by_type)

    logger.info(f"Benchmark complete. Results written to {output_dir}")


def _compute_averages(
    results_by_qubit: dict[int, list[list[RouterBenchmarkResult]]],
    metric_getter: Callable[[RouterBenchmarkResult], float],
) -> list[list[float]]:
    """Compute average metric values per qubit count across samples.

    Args:
        results_by_qubit: Results indexed by qubit count, then sample.
        metric_getter: Function to extract metric from RouterBenchmarkResult.

    Returns:
        List of rows, each row is [qubit_count, avg_router1, avg_router2, ...].
    """
    rows = []
    for qubit_count in bench_const.SCALING_QUBIT_COUNTS:
        sample_results = results_by_qubit[qubit_count]
        if not sample_results:
            continue

        # Compute average for each router across samples
        num_routers = len(ROUTER_NAMES)
        router_totals = [0.0] * num_routers
        router_counts = [0] * num_routers

        for sample in sample_results:
            for router_idx, result in enumerate(sample):
                if result.success:
                    router_totals[router_idx] += metric_getter(result)
                    router_counts[router_idx] += 1

        row = [float(qubit_count)]
        for router_idx in range(num_routers):
            if router_counts[router_idx] > 0:
                avg = router_totals[router_idx] / router_counts[router_idx]
            else:
                avg = 0.0
            row.append(avg)
        rows.append(row)

    return rows


def _write_all_dat_files(
    output_dir: Path,
    results_by_type: dict[str, dict[int, list[list[RouterBenchmarkResult]]]],
) -> None:
    """Write all .dat files for time, gates, and rate metrics.

    Args:
        output_dir: Directory to write files.
        results_by_type: Results indexed by circuit type, then qubit count.
    """
    column_names = ["qubit_count"] + ROUTER_NAMES

    for circuit_type in bench_const.CIRCUIT_TYPES:
        results_by_qubit = results_by_type[circuit_type]

        # Time .dat file
        time_rows = _compute_averages(
            results_by_qubit,
            lambda r: r.execution_time_seconds,
        )
        time_file = output_dir / f"time_{circuit_type}{bench_const.DAT_EXTENSION}"
        write_dat_file(
            time_file,
            f"Execution time (seconds) for {circuit_type} circuits",
            column_names,
            time_rows,
        )

        # Gate count .dat file
        gate_rows = _compute_averages(
            results_by_qubit,
            lambda r: float(r.final_gate_count),
        )
        gate_file = output_dir / f"gates_{circuit_type}{bench_const.DAT_EXTENSION}"
        write_dat_file(
            gate_file,
            f"Final gate count for {circuit_type} circuits",
            column_names,
            gate_rows,
        )

        # Gate increase rate .dat file
        rate_rows = _compute_averages(
            results_by_qubit,
            lambda r: r.gate_increase_rate,
        )
        rate_file = output_dir / f"rate_{circuit_type}{bench_const.DAT_EXTENSION}"
        write_dat_file(
            rate_file,
            f"Gate increase rate for {circuit_type} circuits",
            column_names,
            rate_rows,
        )


def _write_json_summary(
    output_dir: Path,
    results_by_type: dict[str, dict[int, list[list[RouterBenchmarkResult]]]],
) -> None:
    """Write full JSON summary of all benchmark results.

    Args:
        output_dir: Directory to write summary.json.
        results_by_type: Results indexed by circuit type, then qubit count.
    """
    summary: BenchmarkSummary = {}
    for circuit_type, by_qubit in results_by_type.items():
        summary[circuit_type] = {}
        for qubit_count, samples in by_qubit.items():
            summary[circuit_type][qubit_count] = [
                [dataclasses.asdict(r) for r in sample] for sample in samples
            ]

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=bench_const.JSON_INDENT)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run scaling benchmarks for Quariadne routing algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=bench_const.DEFAULT_OUTPUT_DIR / bench_const.SCALING_OUTPUT_SUBDIR,
        help="Output directory for .dat files",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="Ourense",
        choices=bench_const.AVAILABLE_BACKENDS,
        help="Backend topology to use",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=bench_const.DEFAULT_SAMPLES_PER_CONFIG,
        help="Number of samples per configuration",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=bench_const.BENCHMARK_DEFAULT_LOG_LEVEL,
        choices=bench_const.BENCHMARK_LOG_LEVELS,
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for scaling benchmark script."""
    args = parse_args()

    logging.getLogger().setLevel(args.log_level)

    logger.info(f"Starting scaling benchmark on {args.backend} topology")
    logger.info(f"Qubit counts: {bench_const.SCALING_QUBIT_COUNTS}")
    logger.info(f"Circuit types: {bench_const.CIRCUIT_TYPES}")
    logger.info(f"Samples per config: {args.samples}")
    logger.info(f"Output directory: {args.output_dir}")

    run_full_benchmark(
        backend_name=args.backend,
        output_dir=args.output_dir,
        samples_per_config=args.samples,
    )


if __name__ == "__main__":
    main()
