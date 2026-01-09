#!/usr/bin/env python3
"""Comprehensive benchmarking suite for Quariadne routing algorithms.

Implements split-phase benchmarking (token allocation vs token swapping),
supports multiple circuit generators, synthetic topologies, and produces
pgfplots-compatible output for LaTeX integration.

Key features:
    - Tiered qubit ranges: small (5-8), medium (9-12), large (13-16)
    - Split-phase timing: allocation time + swapping time
    - Multiple output formats: .dat (pgfplots), .json (raw)
    - Timeout handling with graceful degradation

References:
    - Python 3.13 argparse: https://docs.python.org/3/library/argparse.html
    - Python 3.13 dataclasses: https://docs.python.org/3/library/dataclasses.html
    - pgfplots manual: https://ctan.org/pkg/pgfplots
    - Google Python Style Guide: https://google.github.io/styleguide/pyguide.html

Usage:
    python benches/run_comprehensive_benchmark.py --output-dir benches/results/comprehensive
"""

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import qiskit
import qiskit.transpiler

import quariadne.benchmarks.circuit_generators as circuit_gen
import quariadne.benchmarks.constants as bench_const
import quariadne.bipartite_allocation_lp
import quariadne.circuit
import quariadne.milp
import quariadne.routers


# Logging configuration
logging.basicConfig(
    format=bench_const.BENCHMARK_LOG_FORMAT,
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# SWAP operation is equivalent to 3 CX gates in standard decomposition
SWAP_TO_CX_COUNT = 3


@dataclasses.dataclass(frozen=True)
class ComprehensiveBenchmarkConfig:
    """Configuration for a single benchmark run.

    Attributes:
        qubit_count: Number of qubits in the circuit.
        circuit_type: Type of circuit generator to use.
        topology_name: Name of hardware topology (or synthetic identifier).
        sample_index: Index for statistical significance sampling.
        seed: Random seed for reproducibility.
    """

    qubit_count: int
    circuit_type: str
    topology_name: str
    sample_index: int
    seed: int


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
    except Exception as exception:
        end_time = time.perf_counter()
        logger.warning(f"Router {router_class.__name__} failed: {exception}")
        return RouterBenchmarkResult(
            router_name=router_class.__name__,
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_gate_count,
            final_gate_count=0,
            swap_count=0,
            success=False,
            error_message=str(exception),
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
    except Exception as exception:
        end_time = time.perf_counter()
        logger.warning(f"BipartiteAllocationRouter failed: {exception}")
        return RouterBenchmarkResult(
            router_name="BipartiteAllocationRouter",
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_gate_count,
            final_gate_count=0,
            swap_count=0,
            success=False,
            error_message=str(exception),
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
        transpiled_circuit = pass_manager.run(qiskit_circuit)
        end_time = time.perf_counter()

        final_two_qubit_count = transpiled_circuit.num_nonlocal_gates()
        swap_count = transpiled_circuit.count_ops().get("swap", 0)

        return RouterBenchmarkResult(
            router_name="SABRE",
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_two_qubit_count,
            final_gate_count=final_two_qubit_count,
            swap_count=swap_count,
            success=True,
        )
    except Exception as exception:
        end_time = time.perf_counter()
        logger.warning(f"SABRE failed: {exception}")
        return RouterBenchmarkResult(
            router_name="SABRE",
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_two_qubit_count,
            final_gate_count=0,
            swap_count=0,
            success=False,
            error_message=str(exception),
        )


def generate_circuit(
    circuit_type: str,
    num_qubits: int,
    seed: int,
) -> quariadne.circuit.AbstractQuantumCircuit:
    """Generate an AbstractQuantumCircuit based on circuit type.

    Unified interface for all circuit generators.

    Args:
        circuit_type: One of the COMPREHENSIVE_CIRCUIT_TYPES.
        num_qubits: Number of qubits in the circuit.
        seed: Random seed for reproducibility.

    Returns:
        AbstractQuantumCircuit ready for routing.

    Raises:
        ValueError: If circuit_type is not recognised.
    """
    match circuit_type:
        case "random_clifford":
            num_gates = num_qubits * bench_const.DEFAULT_GATES_PER_QUBIT
            return circuit_gen.generate_random_clifford_circuit(
                num_qubits, num_gates, seed
            )
        case "linear_chain":
            chain_length = num_qubits * 2
            return circuit_gen.generate_linear_chain_circuit(num_qubits, chain_length)
        case "ring":
            return circuit_gen.generate_ring_circuit(num_qubits)
        case "star":
            return circuit_gen.generate_star_circuit(num_qubits, hub_qubit=0)
        case "qft_like":
            return circuit_gen.generate_qft_like_circuit(num_qubits, seed=seed)
        case "full_layer":
            num_layers = bench_const.FULL_LAYER_DEFAULT_LAYERS
            return circuit_gen.generate_full_layer_circuit(num_qubits, num_layers, seed)
        case "depth_controlled":
            target_depth = num_qubits * bench_const.DEPTH_MULTIPLIER_DEFAULT
            return circuit_gen.generate_depth_controlled_random_circuit(
                num_qubits, target_depth, seed
            )
        case _:
            raise ValueError(f"Unknown circuit type: {circuit_type}")


def generate_circuit_pair(
    config: ComprehensiveBenchmarkConfig,
) -> tuple[qiskit.QuantumCircuit, quariadne.circuit.AbstractQuantumCircuit]:
    """Generate both Qiskit and AbstractQuantumCircuit for a benchmark config.

    Args:
        config: Benchmark configuration specifying circuit parameters.

    Returns:
        Tuple of (Qiskit QuantumCircuit, AbstractQuantumCircuit).
    """
    abstract_circuit = generate_circuit(
        config.circuit_type, config.qubit_count, config.seed
    )
    qiskit_circuit = abstract_circuit.to_qiskit()
    return qiskit_circuit, abstract_circuit


# Router names in output order (matches column order in .dat files)
ROUTER_NAMES = [
    "IlpRouter",
    "LpRouterEdges",
    "LpRouterMapping",
    "BipartiteAllocationRouter",
    "SABRE",
]


def run_single_benchmark(
    config: ComprehensiveBenchmarkConfig,
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

    # IlpRouter (exact, may timeout)
    ilp_result = time_quariadne_router(
        quariadne.routers.IlpRouter, coupling_graph, abstract_circuit
    )
    results.append(ilp_result)

    # LP-based routers
    for router_class in [
        quariadne.routers.LpRouterEdges,
        quariadne.routers.LpRouterMapping,
    ]:
        result = time_quariadne_router(router_class, coupling_graph, abstract_circuit)
        results.append(result)

    # BipartiteAllocationRouter (different API)
    bipartite_result = time_bipartite_router(coupling_graph, abstract_circuit)
    results.append(bipartite_result)

    # SABRE (Qiskit baseline)
    sabre_result = time_sabre_router(coupling_map, qiskit_circuit)
    results.append(sabre_result)

    return results


# Metric definitions: (file_prefix, attribute_name)
BENCHMARK_METRICS = [
    ("time", "execution_time_seconds"),
    ("gates", "final_gate_count"),
    ("rate", "gate_increase_rate"),
    ("swaps", "swap_count"),
]


def aggregate_results_by_metric(
    results_by_qubit: dict[int, list[list[RouterBenchmarkResult]]],
    attribute_name: str,
) -> pd.DataFrame:
    """Aggregate benchmark results for a specific metric attribute.

    Args:
        results_by_qubit: Results indexed by qubit count, then samples.
        attribute_name: Name of RouterBenchmarkResult attribute to extract.

    Returns:
        DataFrame with qubit_count and averaged metric per router.
    """
    # Build columns separately for type consistency
    qubit_counts: list[int] = []
    router_metrics: dict[str, list[float]] = {name: [] for name in ROUTER_NAMES}

    for qubit_count in sorted(results_by_qubit.keys()):
        sample_results = results_by_qubit[qubit_count]
        if not sample_results:
            continue

        qubit_counts.append(qubit_count)
        for router_index, router_name in enumerate(ROUTER_NAMES):
            values = [
                getattr(sample[router_index], attribute_name)
                for sample in sample_results
                if router_index < len(sample) and sample[router_index].success
            ]
            average = sum(values) / len(values) if values else 0.0
            router_metrics[router_name].append(average)

    return pd.DataFrame({"qubit_count": qubit_counts, **router_metrics})


def write_all_dat_files(
    output_dir: Path,
    results_by_type: dict[str, dict[int, list[list[RouterBenchmarkResult]]]],
    topology_name: str,
) -> None:
    """Write all .dat files for all metrics and circuit types.

    Args:
        output_dir: Directory to write files.
        results_by_type: Results indexed by circuit type, then qubit count.
        topology_name: Name of topology for filename suffix.
    """
    for circuit_type, results_by_qubit in results_by_type.items():
        for file_prefix, attribute_name in BENCHMARK_METRICS:
            metric_df = aggregate_results_by_metric(results_by_qubit, attribute_name)
            output_file = (
                output_dir / f"{file_prefix}_{circuit_type}_{topology_name}.dat"
            )
            metric_df.to_csv(output_file, sep="\t", index=False, float_format="%.6f")


def write_json_summary(
    output_dir: Path,
    results_by_type: dict[str, dict[int, list[list[RouterBenchmarkResult]]]],
) -> None:
    """Write full JSON summary of all benchmark results.

    Args:
        output_dir: Directory to write summary.json.
        results_by_type: Results indexed by circuit type, then qubit count.
    """
    summary = {}
    for circuit_type, by_qubit in results_by_type.items():
        summary[circuit_type] = {
            qubit_count: [[dataclasses.asdict(r) for r in sample] for sample in samples]
            for qubit_count, samples in by_qubit.items()
        }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as file_handle:
        json.dump(summary, file_handle, indent=bench_const.JSON_INDENT)


def run_full_benchmark(
    backend_name: str,
    output_dir: Path,
    qubit_counts: list[int],
    circuit_types: list[str],
    samples_per_config: int,
) -> None:
    """Run the full comprehensive benchmark.

    Args:
        backend_name: Name of the backend topology.
        output_dir: Directory to write output files.
        qubit_counts: List of qubit counts to benchmark.
        circuit_types: List of circuit types to benchmark.
        samples_per_config: Number of samples per configuration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    coupling_map = bench_const.create_backend_coupling_map(backend_name)
    coupling_graph = quariadne.milp.get_coupling_graph(coupling_map)

    results_by_type: dict[str, dict[int, list[list[RouterBenchmarkResult]]]] = {
        circuit_type: {qc: [] for qc in qubit_counts} for circuit_type in circuit_types
    }

    total_configs = len(qubit_counts) * len(circuit_types) * samples_per_config
    current_config = 0

    for circuit_type in circuit_types:
        for qubit_count in qubit_counts:
            for sample_idx in range(samples_per_config):
                current_config += 1
                seed = bench_const.SCALING_SEED_BASE + sample_idx

                config = ComprehensiveBenchmarkConfig(
                    qubit_count=qubit_count,
                    circuit_type=circuit_type,
                    topology_name=backend_name,
                    sample_index=sample_idx,
                    seed=seed,
                )

                logger.info(
                    f"[{current_config}/{total_configs}] "
                    f"{circuit_type} q={qubit_count} sample={sample_idx}"
                )

                results = run_single_benchmark(config, coupling_graph, coupling_map)
                results_by_type[circuit_type][qubit_count].append(results)

    write_all_dat_files(output_dir, results_by_type, backend_name)
    write_json_summary(output_dir, results_by_type)

    logger.info(f"Benchmark complete. Results written to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmarks for Quariadne routing algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=bench_const.DEFAULT_OUTPUT_DIR
        / bench_const.COMPREHENSIVE_OUTPUT_SUBDIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="Aspen-4",
        choices=bench_const.AVAILABLE_BACKENDS,
        help="Backend topology to use",
    )
    parser.add_argument(
        "--qubit-counts",
        type=int,
        nargs="+",
        default=bench_const.COMPREHENSIVE_QUBIT_COUNTS,
        help="Qubit counts to benchmark",
    )
    parser.add_argument(
        "--circuit-types",
        type=str,
        nargs="+",
        default=bench_const.COMPREHENSIVE_CIRCUIT_TYPES,
        help="Circuit types to benchmark",
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
    """Main entry point for comprehensive benchmark script."""
    args = parse_args()

    logging.getLogger().setLevel(args.log_level)

    logger.info(f"Starting comprehensive benchmark on {args.backend} topology")
    logger.info(f"Qubit counts: {args.qubit_counts}")
    logger.info(f"Circuit types: {args.circuit_types}")
    logger.info(f"Samples per config: {args.samples}")
    logger.info(f"Output directory: {args.output_dir}")

    run_full_benchmark(
        backend_name=args.backend,
        output_dir=args.output_dir,
        qubit_counts=args.qubit_counts,
        circuit_types=args.circuit_types,
        samples_per_config=args.samples,
    )


if __name__ == "__main__":
    main()
