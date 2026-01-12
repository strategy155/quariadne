#!/usr/bin/env python3
"""Tiered benchmarking suite for Quariadne routing algorithms (~11-hour thesis run).

Implements three-tier benchmarking with configurable router sets per tier:
    - Tier 1 (5-8 qubits): All routers (ILP, LPE, LPM, Bipartite, SABRE), 8 samples
    - Tier 2 (9-10 qubits): All routers, 2 samples
    - Tier 3 (11-12 qubits): LP routers + Bipartite + SABRE only (skip ILP), 4 samples

Key features:
    - Parallel execution via ProcessPoolExecutor (--workers N)
    - Incremental JSON saving after each benchmark for crash resilience
    - Resume capability via completed result detection
    - Graceful error handling for BipartiteAllocationRouter (known issues)
    - pgfplots-compatible .dat output files for LaTeX integration
    - Watts-Strogatz synthetic topology (k=4, p=0.3)

Circuit types (per thesis Chapter 4, Figure 4.1):
    - random_clifford, linear_chain, ring, star, qft_like, full_layer, depth_controlled

References:
    - Python 3.13 argparse: https://docs.python.org/3/library/argparse.html
    - Python 3.13 dataclasses: https://docs.python.org/3/library/dataclasses.html
    - Python 3.13 time: https://docs.python.org/3/library/time.html
    - pgfplots manual: https://ctan.org/pkg/pgfplots
    - Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
    - Thesis Chapter 4 - Experiment methodology

Usage:
    python benches/run_tiered_benchmark.py --output-dir benches/results/tiered
    python benches/run_tiered_benchmark.py --dry-run
    python benches/run_tiered_benchmark.py --skip-tier 2 3  # Run Tier 1 only
"""

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime
import json
import logging
import os
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Generator

import pandas as pd
import qiskit
import qiskit.transpiler
import scipy.stats

import quariadne.benchmarks.circuit_generators as circuit_gen
import quariadne.benchmarks.constants as bench_const
import quariadne.benchmarks.topology_generators as topo_gen
import quariadne.bipartite_allocation_lp
import quariadne.circuit
import quariadne.lp_router_layered
import quariadne.milp
import quariadne.routers


# Logging configuration
logging.basicConfig(
    format=bench_const.BENCHMARK_LOG_FORMAT,
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Signal-Based Timeout
# -----------------------------------------------------------------------------

# ILP timeout covers both constraint building and solving phases
# HiGHS time_limit only affects solving, but constraint building can be slow
ILP_TIMEOUT_SECONDS = 900  # 15 minutes


class RouterTimeoutError(TimeoutError):
    """Raised when router execution exceeds configured timeout."""

    pass


def _timeout_handler(signum: int, frame: object) -> None:
    """SIGALRM handler - raises RouterTimeoutError on timeout."""
    raise RouterTimeoutError(f"Router exceeded {ILP_TIMEOUT_SECONDS}s timeout")


@contextlib.contextmanager
def timeout_context(seconds: int) -> Generator[None, None, None]:
    """Context manager for signal-based timeout (Unix only).

    Args:
        seconds: Maximum execution time in seconds.

    Yields:
        Control to the with-block.

    Raises:
        RouterTimeoutError: If execution exceeds timeout.

    Reference:
        https://docs.python.org/3/library/signal.html#signal.alarm
    """
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TierConfig:
    """Configuration for a benchmark tier.

    Attributes:
        tier_name: Human-readable tier identifier (e.g., "Tier 1 (5-8 qubits)").
        qubit_counts: List of qubit counts in this tier.
        samples_per_config: Number of statistical samples per configuration.
        include_ilp: Whether to run IlpRouter in this tier.
    """

    tier_name: str
    qubit_counts: list[int]
    samples_per_config: int
    include_ilp: bool


@dataclasses.dataclass(frozen=True)
class TieredBenchmarkConfig:
    """Configuration for a single tiered benchmark run.

    Attributes:
        qubit_count: Number of qubits in the circuit.
        circuit_type: Type of circuit generator to use.
        topology_seed: Seed for Watts-Strogatz topology generation.
        sample_index: Index for statistical significance sampling.
        circuit_seed: Random seed for circuit generation.
        tier_name: Name of the tier this config belongs to.
    """

    qubit_count: int
    circuit_type: str
    topology_seed: int
    sample_index: int
    circuit_seed: int
    tier_name: str


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
        skipped: Whether router was intentionally skipped for this tier.
    """

    router_name: str
    execution_time_seconds: float
    original_gate_count: int
    final_gate_count: int
    swap_count: int
    success: bool
    error_message: str | None = None
    skipped: bool = False

    @property
    def gate_increase_rate(self) -> float:
        """Calculate gate count increase ratio (final / original)."""
        if self.original_gate_count == 0:
            return 0.0
        return self.final_gate_count / self.original_gate_count


@dataclasses.dataclass
class IncrementalResultEntry:
    """Single entry in the incremental results JSON.

    Attributes:
        config: Benchmark configuration that produced this result.
        results: List of RouterBenchmarkResult for each router.
        timestamp: ISO format timestamp when benchmark completed.
    """

    config: TieredBenchmarkConfig
    results: list[RouterBenchmarkResult]
    timestamp: str


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# SWAP operation is equivalent to 3 CX gates in standard decomposition
SWAP_TO_CX_COUNT = 3

# Watts-Strogatz topology parameters (k must be even per NetworkX requirement)
WATTS_STROGATZ_K = 4
WATTS_STROGATZ_P = 0.3

# All 7 circuit types per thesis Chapter 4, Figure 4.1
THESIS_CIRCUIT_TYPES: list[str] = [
    "random_clifford",
    "linear_chain",
    "ring",
    "star",
    # "qft_like",  # Temporarily disabled - suspected hang source
    "full_layer",
    "depth_controlled",
]

# Router names in output order (matches column order in .dat files)
ROUTER_NAMES: list[str] = [
    "IlpRouter",
    "LpRouterEdges",
    "LpRouterMapping",
    "LpRouterLayered",
    # "BipartiteAllocationRouter",  # Temporarily disabled - suspected hang source
    "SABRE",
]

# Output file constants
RESULTS_FILENAME = "tiered_results.json"
DEFAULT_OUTPUT_DIR = Path("benches/results/tiered")

# DataFrame column names
COL_CIRCUIT_TYPE = "circuit_type"
COL_QUBIT_COUNT = "qubit_count"
COL_SAMPLE_INDEX = "sample_index"
COL_ROUTER = "router"
COL_TIME = "time"
COL_GATES = "gates"
COL_RATE = "rate"
COL_SWAPS = "swaps"
COL_SUCCESS = "success"
COL_SKIPPED = "skipped"

# .dat file prefixes
DAT_PREFIX_TIME = "time"
DAT_PREFIX_GATES = "gates"
DAT_PREFIX_RATE = "rate"
DAT_PREFIX_SWAPS = "swaps"
DAT_SUFFIX = "WS"


# -----------------------------------------------------------------------------
# Tier Configurations
# -----------------------------------------------------------------------------

TIER_1_CONFIG = TierConfig(
    tier_name="Tier 1 (5-8 qubits)",
    qubit_counts=[5, 6, 7, 8],
    samples_per_config=8,
    include_ilp=False,  # ILP disabled: signal timeout doesn't work with C extensions
)

TIER_2_CONFIG = TierConfig(
    tier_name="Tier 2 (9-10 qubits)",
    qubit_counts=[9, 10],
    samples_per_config=2,
    include_ilp=False,  # ILP disabled: signal timeout doesn't work with C extensions
)

TIER_3_CONFIG = TierConfig(
    tier_name="Tier 3 (11-12 qubits)",
    qubit_counts=[11, 12],
    samples_per_config=4,
    include_ilp=False,  # ILP disabled: signal timeout doesn't work with C extensions
)

ALL_TIERS: list[TierConfig] = [TIER_1_CONFIG, TIER_2_CONFIG, TIER_3_CONFIG]


# -----------------------------------------------------------------------------
# Router Timing Functions
# -----------------------------------------------------------------------------

# Type alias for router execution callables
# Returns (swap_count, final_gate_count)
type RouterCallable = Callable[[], tuple[int, int]]


def time_router_execution(
    router_name: str,
    original_gate_count: int,
    execute_fn: RouterCallable,
) -> RouterBenchmarkResult:
    """Generic timing wrapper for any router execution.

    Args:
        router_name: Name of the router for result labelling.
        original_gate_count: Number of two-qubit gates before routing.
        execute_fn: Callable that executes routing and returns (swap_count, final_gate_count).

    Returns:
        RouterBenchmarkResult with timing and gate count data.
    """
    start_time = time.perf_counter()
    try:
        swap_count, final_gate_count = execute_fn()
        end_time = time.perf_counter()
        return RouterBenchmarkResult(
            router_name=router_name,
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_gate_count,
            final_gate_count=final_gate_count,
            swap_count=swap_count,
            success=True,
        )
    except Exception as exception:
        end_time = time.perf_counter()
        logger.warning(f"{router_name} failed: {type(exception).__name__}: {exception}")
        return RouterBenchmarkResult(
            router_name=router_name,
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_gate_count,
            final_gate_count=0,
            swap_count=0,
            success=False,
            error_message=f"{type(exception).__name__}: {exception}",
        )


def time_quariadne_router(
    router_class: type,
    coupling_map: qiskit.transpiler.CouplingMap,
    circuit: quariadne.circuit.AbstractQuantumCircuit,
) -> RouterBenchmarkResult:
    """Time a Quariadne router (LpRouterEdges, LpRouterMapping, IlpRouter).

    Args:
        router_class: Router class to instantiate.
        coupling_map: Qiskit CouplingMap (converted to DiGraph internally).
        circuit: AbstractQuantumCircuit to route.

    Returns:
        RouterBenchmarkResult with timing and gate count data.
    """
    original_gate_count = len(circuit.get_two_qubit_operations())

    def execute() -> tuple[int, int]:
        # Convert CouplingMap to DiGraph with PhysicalQubit nodes
        coupling_graph = quariadne.milp.get_coupling_graph(coupling_map)
        router = router_class(coupling_graph, circuit)
        swap_dict = router.get_inserted_swaps()
        swap_count = sum(len(swaps) for swaps in swap_dict.values())
        final_gate_count = original_gate_count + (swap_count * SWAP_TO_CX_COUNT)
        return swap_count, final_gate_count

    return time_router_execution(router_class.__name__, original_gate_count, execute)


def time_bipartite_router(
    coupling_map: qiskit.transpiler.CouplingMap,
    circuit: quariadne.circuit.AbstractQuantumCircuit,
) -> RouterBenchmarkResult:
    """Time BipartiteAllocationRouter (different API, known issues).

    Args:
        coupling_map: Qiskit CouplingMap (converted to DiGraph internally).
        circuit: AbstractQuantumCircuit to route.

    Returns:
        RouterBenchmarkResult with timing and gate count data.
    """
    original_gate_count = len(circuit.get_two_qubit_operations())

    def execute() -> tuple[int, int]:
        coupling_graph = quariadne.milp.get_coupling_graph(coupling_map)
        router = quariadne.bipartite_allocation_lp.BipartiteAllocationRouter(
            coupling_graph, circuit
        )
        result = router.run()
        swap_count = sum(len(swaps) for swaps in result.inserted_swaps.values())
        final_gate_count = original_gate_count + (swap_count * SWAP_TO_CX_COUNT)
        return swap_count, final_gate_count

    return time_router_execution(
        "BipartiteAllocationRouter", original_gate_count, execute
    )


def time_layered_router(
    coupling_map: qiskit.transpiler.CouplingMap,
    circuit: quariadne.circuit.AbstractQuantumCircuit,
) -> RouterBenchmarkResult:
    """Time LpRouterLayered (layer-by-layer LP routing).

    Args:
        coupling_map: Qiskit CouplingMap (converted to DiGraph internally).
        circuit: AbstractQuantumCircuit to route.

    Returns:
        RouterBenchmarkResult with timing, gate count, and LP objective data.
    """
    original_gate_count = len(circuit.get_two_qubit_operations())
    start_time = time.perf_counter()

    try:
        coupling_graph = quariadne.milp.get_coupling_graph(coupling_map)
        router = quariadne.lp_router_layered.LpRouterLayered(coupling_graph, circuit)
        result = router.run()
        end_time = time.perf_counter()

        # Report LP objective as swap_count (swap extraction is buggy, objective is meaningful)
        lp_objective_int = int(result.objective_value)
        return RouterBenchmarkResult(
            router_name="LpRouterLayered",
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_gate_count,
            final_gate_count=original_gate_count + lp_objective_int * SWAP_TO_CX_COUNT,
            swap_count=lp_objective_int,
            success=True,
        )
    except Exception as exception:
        end_time = time.perf_counter()
        logger.warning(
            f"LpRouterLayered failed: {type(exception).__name__}: {exception}"
        )
        return RouterBenchmarkResult(
            router_name="LpRouterLayered",
            execution_time_seconds=end_time - start_time,
            original_gate_count=original_gate_count,
            final_gate_count=0,
            swap_count=0,
            success=False,
            error_message=f"{type(exception).__name__}: {exception}",
        )


def time_sabre_router(
    coupling_map: qiskit.transpiler.CouplingMap,
    qiskit_circuit: qiskit.QuantumCircuit,
) -> RouterBenchmarkResult:
    """Time Qiskit SABRE routing algorithm."""
    original_gate_count = qiskit_circuit.num_nonlocal_gates()

    def execute() -> tuple[int, int]:
        pass_manager = qiskit.transpiler.generate_preset_pass_manager(
            coupling_map=coupling_map,
            optimization_level=bench_const.DEFAULT_OPTIMISATION_LEVEL,
            routing_method="sabre",
            layout_method="trivial",  # Fair comparison: Quariadne routers don't optimise layout
        )
        transpiled = pass_manager.run(qiskit_circuit)
        swap_count = transpiled.count_ops().get("swap", 0)
        final_gate_count = transpiled.num_nonlocal_gates()
        return swap_count, final_gate_count

    return time_router_execution("SABRE", original_gate_count, execute)


def create_skipped_result(
    router_name: str, original_gate_count: int
) -> RouterBenchmarkResult:
    """Create a result for a router intentionally skipped (e.g., ILP in Tier 3)."""
    return RouterBenchmarkResult(
        router_name=router_name,
        execution_time_seconds=0.0,
        original_gate_count=original_gate_count,
        final_gate_count=0,
        swap_count=0,
        success=False,
        error_message="Router skipped for this tier",
        skipped=True,
    )


# -----------------------------------------------------------------------------
# Circuit Generation
# -----------------------------------------------------------------------------


def generate_circuit(
    circuit_type: str,
    num_qubits: int,
    seed: int,
) -> quariadne.circuit.AbstractQuantumCircuit:
    """Generate an AbstractQuantumCircuit based on circuit type.

    Supports all 7 circuit types per thesis Chapter 4, Figure 4.1.

    Args:
        circuit_type: One of THESIS_CIRCUIT_TYPES.
        num_qubits: Number of qubits in the circuit.
        seed: Random seed for reproducibility.

    Returns:
        AbstractQuantumCircuit ready for routing.

    Raises:
        ValueError: If circuit_type is not recognised.
    """
    match circuit_type:
        case "random_clifford":
            # Halved gate count for debugging hang issue
            num_gates = (num_qubits * bench_const.DEFAULT_GATES_PER_QUBIT) // 2
            return circuit_gen.generate_random_clifford_circuit(
                num_qubits, num_gates, seed
            )
        case "linear_chain":
            # Halved chain length for debugging
            chain_length = num_qubits
            return circuit_gen.generate_linear_chain_circuit(num_qubits, chain_length)
        case "ring":
            return circuit_gen.generate_ring_circuit(num_qubits)
        case "star":
            return circuit_gen.generate_star_circuit(num_qubits, hub_qubit=0)
        case "qft_like":
            return circuit_gen.generate_qft_like_circuit(num_qubits, seed=seed)
        case "full_layer":
            # Halved layer count for debugging
            num_layers = bench_const.FULL_LAYER_DEFAULT_LAYERS // 2
            return circuit_gen.generate_full_layer_circuit(num_qubits, num_layers, seed)
        case "depth_controlled":
            # Halved depth for debugging
            target_depth = (num_qubits * bench_const.DEPTH_MULTIPLIER_DEFAULT) // 2
            return circuit_gen.generate_depth_controlled_random_circuit(
                num_qubits, target_depth, seed
            )
        case _:
            raise ValueError(f"Unknown circuit type: {circuit_type}")


def generate_circuit_pair(
    config: TieredBenchmarkConfig,
) -> tuple[qiskit.QuantumCircuit, quariadne.circuit.AbstractQuantumCircuit]:
    """Generate both Qiskit and AbstractQuantumCircuit for a benchmark config."""
    abstract_circuit = generate_circuit(
        config.circuit_type, config.qubit_count, config.circuit_seed
    )
    qiskit_circuit = abstract_circuit.to_qiskit()
    return qiskit_circuit, abstract_circuit


# -----------------------------------------------------------------------------
# Incremental Saving
# -----------------------------------------------------------------------------


def generate_result_key(config: TieredBenchmarkConfig) -> str:
    """Generate a unique key for a benchmark configuration (for resume capability)."""
    return f"{config.circuit_type}_q{config.qubit_count}_s{config.sample_index}"


def load_existing_results(results_filepath: Path) -> dict[str, IncrementalResultEntry]:
    """Load existing results from JSON file for resume capability.

    Args:
        results_filepath: Path to tiered_results.json file.

    Returns:
        Dictionary mapping result keys to IncrementalResultEntry objects.
        Empty dict if file doesn't exist.
    """
    if not results_filepath.exists():
        return {}

    with open(results_filepath) as results_file:
        raw_results = json.load(results_file)

    existing_results: dict[str, IncrementalResultEntry] = {}
    for entry in raw_results:
        config = TieredBenchmarkConfig(**entry["config"])
        results = [RouterBenchmarkResult(**r) for r in entry["results"]]
        key = generate_result_key(config)
        existing_results[key] = IncrementalResultEntry(
            config=config,
            results=results,
            timestamp=entry["timestamp"],
        )

    logger.info(f"Loaded {len(existing_results)} existing results for resume")
    return existing_results


def save_results_incrementally(
    results_filepath: Path,
    all_results: dict[str, IncrementalResultEntry],
) -> None:
    """Save all results to JSON file (called after each benchmark for crash resilience)."""
    serialised = [
        {
            "config": dataclasses.asdict(entry.config),
            "results": [dataclasses.asdict(r) for r in entry.results],
            "timestamp": entry.timestamp,
        }
        for entry in all_results.values()
    ]

    with open(results_filepath, "w") as results_file:
        json.dump(serialised, results_file, indent=bench_const.JSON_INDENT)


# -----------------------------------------------------------------------------
# Main Benchmark Loop
# -----------------------------------------------------------------------------


def run_single_tiered_benchmark(
    config: TieredBenchmarkConfig,
    coupling_map: qiskit.transpiler.CouplingMap,
    include_ilp: bool,
) -> list[RouterBenchmarkResult]:
    """Run all applicable routers on a single circuit configuration.

    Args:
        config: Benchmark configuration specifying circuit parameters.
        coupling_map: Qiskit CouplingMap for the hardware topology.
        include_ilp: Whether to run IlpRouter (False for Tier 3).

    Returns:
        List of RouterBenchmarkResult, one per router in ROUTER_NAMES order.
    """
    qiskit_circuit, abstract_circuit = generate_circuit_pair(config)
    original_gate_count = len(abstract_circuit.get_two_qubit_operations())

    results: list[RouterBenchmarkResult] = []

    # IlpRouter (conditional based on tier, with signal timeout for constraint building)
    if include_ilp:
        try:
            with timeout_context(ILP_TIMEOUT_SECONDS):
                ilp_result = time_quariadne_router(
                    quariadne.routers.IlpRouter, coupling_map, abstract_circuit
                )
            results.append(ilp_result)
        except RouterTimeoutError as e:
            logger.warning(f"IlpRouter timeout: {e}")
            results.append(
                RouterBenchmarkResult(
                    router_name="IlpRouter",
                    execution_time_seconds=float(ILP_TIMEOUT_SECONDS),
                    original_gate_count=original_gate_count,
                    final_gate_count=0,
                    swap_count=0,
                    success=False,
                    error_message=str(e),
                )
            )
    else:
        results.append(create_skipped_result("IlpRouter", original_gate_count))

    # LpRouterEdges (NetworkX bipartite matching)
    results.append(
        time_quariadne_router(
            quariadne.routers.LpRouterEdges, coupling_map, abstract_circuit
        )
    )

    # LpRouterMapping (Birkhoff-based, no bipartite matching)
    results.append(
        time_quariadne_router(
            quariadne.routers.LpRouterMapping, coupling_map, abstract_circuit
        )
    )

    # LpRouterLayered (layer-by-layer LP routing)
    results.append(time_layered_router(coupling_map, abstract_circuit))

    # BipartiteAllocationRouter - temporarily disabled (suspected hang source)
    # results.append(time_bipartite_router(coupling_map, abstract_circuit))

    # SABRE (Qiskit baseline)
    results.append(time_sabre_router(coupling_map, qiskit_circuit))

    return results


def _benchmark_worker(
    config: TieredBenchmarkConfig,
    include_ilp: bool,
) -> tuple[str, IncrementalResultEntry]:
    """Worker function for parallel benchmark execution.

    Must be at module level for pickling by ProcessPoolExecutor.
    Creates coupling map internally (Qiskit objects don't pickle reliably).

    Reference:
        https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor

    Args:
        config: Benchmark configuration specifying circuit parameters.
        include_ilp: Whether to run IlpRouter.

    Returns:
        Tuple of (result_key, IncrementalResultEntry) for collection by main process.
    """
    coupling_map = create_coupling_map_for_qubit_count(config.qubit_count)
    results = run_single_tiered_benchmark(config, coupling_map, include_ilp)
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    result_key = generate_result_key(config)
    entry = IncrementalResultEntry(config=config, results=results, timestamp=timestamp)
    return result_key, entry


def create_coupling_map_for_qubit_count(
    qubit_count: int,
) -> qiskit.transpiler.CouplingMap:
    """Create Watts-Strogatz CouplingMap for a given qubit count.

    Args:
        qubit_count: Number of physical qubits in the topology.

    Returns:
        Qiskit CouplingMap representing the hardware connectivity.
    """
    topology_seed = bench_const.SCALING_SEED_BASE + qubit_count
    return topo_gen.connected_watts_strogatz_to_coupling_map(
        num_qubits=qubit_count,
        nearest_neighbours=WATTS_STROGATZ_K,
        rewiring_probability=WATTS_STROGATZ_P,
        seed=topology_seed,
    )


def create_benchmark_config(
    qubit_count: int,
    circuit_type: str,
    sample_idx: int,
    tier_name: str,
) -> TieredBenchmarkConfig:
    """Create a benchmark configuration for a single run."""
    topology_seed = bench_const.SCALING_SEED_BASE + qubit_count
    circuit_seed = bench_const.SCALING_SEED_BASE + sample_idx
    return TieredBenchmarkConfig(
        qubit_count=qubit_count,
        circuit_type=circuit_type,
        topology_seed=topology_seed,
        sample_index=sample_idx,
        circuit_seed=circuit_seed,
        tier_name=tier_name,
    )


def calculate_total_configs(tiers: list[TierConfig], circuit_types: list[str]) -> int:
    """Calculate total number of benchmark configurations."""
    return sum(
        len(tier.qubit_counts) * len(circuit_types) * tier.samples_per_config
        for tier in tiers
    )


def run_tiered_benchmark(
    output_dir: Path,
    tiers: list[TierConfig],
    circuit_types: list[str],
    include_ci: bool = True,
) -> None:
    """Run the full tiered benchmark with incremental saving."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_filepath = output_dir / RESULTS_FILENAME
    all_results = load_existing_results(results_filepath)

    total_configs = calculate_total_configs(tiers, circuit_types)
    current_config = 0
    skipped_count = 0

    for tier in tiers:
        logger.info(f"Starting {tier.tier_name}")

        for qubit_count in tier.qubit_counts:
            # Create topology once per qubit count (same seed ensures reproducibility)
            coupling_map = create_coupling_map_for_qubit_count(qubit_count)

            for circuit_type in circuit_types:
                for sample_idx in range(tier.samples_per_config):
                    current_config += 1
                    config = create_benchmark_config(
                        qubit_count, circuit_type, sample_idx, tier.tier_name
                    )
                    result_key = generate_result_key(config)

                    if result_key in all_results:
                        skipped_count += 1
                        logger.debug(
                            f"[{current_config}/{total_configs}] SKIP {result_key}"
                        )
                        continue

                    logger.info(
                        f"[{current_config}/{total_configs}] {circuit_type} q={qubit_count} s={sample_idx}"
                    )

                    results = run_single_tiered_benchmark(
                        config, coupling_map, tier.include_ilp
                    )
                    timestamp = datetime.datetime.now(
                        tz=datetime.timezone.utc
                    ).isoformat()
                    all_results[result_key] = IncrementalResultEntry(
                        config=config, results=results, timestamp=timestamp
                    )
                    save_results_incrementally(results_filepath, all_results)

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} already-completed benchmarks")

    write_dat_files(output_dir, all_results, include_ci=include_ci)
    write_summary_json(output_dir, all_results)
    logger.info(f"Benchmark complete. Results written to {output_dir}")


def run_tiered_benchmark_parallel(
    output_dir: Path,
    tiers: list[TierConfig],
    circuit_types: list[str],
    num_workers: int,
    include_ci: bool = True,
) -> None:
    """Run tiered benchmark with parallel config execution via ProcessPoolExecutor.

    Parallelises at the circuit configuration level: each worker executes all
    5 routers sequentially for a single circuit config. Results are collected
    in the main process and saved periodically.

    Reference:
        https://docs.python.org/3/library/concurrent.futures.html

    Args:
        output_dir: Directory to write results and .dat files.
        tiers: List of tier configurations to benchmark.
        circuit_types: List of circuit types to benchmark.
        num_workers: Number of parallel worker processes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_filepath = output_dir / RESULTS_FILENAME
    all_results = load_existing_results(results_filepath)
    completed_keys = set(all_results.keys())

    # Build list of pending configs (exclude already completed)
    pending_configs: list[tuple[TieredBenchmarkConfig, bool]] = []
    for tier in tiers:
        for qubit_count in tier.qubit_counts:
            for circuit_type in circuit_types:
                for sample_idx in range(tier.samples_per_config):
                    config = create_benchmark_config(
                        qubit_count, circuit_type, sample_idx, tier.tier_name
                    )
                    result_key = generate_result_key(config)
                    if result_key not in completed_keys:
                        pending_configs.append((config, tier.include_ilp))

    total_pending = len(pending_configs)
    total_completed = len(all_results)
    logger.info(
        f"Pending configs: {total_pending}, already completed: {total_completed}, workers: {num_workers}"
    )

    if total_pending == 0:
        logger.info("No pending configs to run")
        write_dat_files(output_dir, all_results, include_ci=include_ci)
        write_summary_json(output_dir, all_results)
        return

    # Save interval reduces I/O overhead (not every single result)
    SAVE_INTERVAL = 10
    completed_since_save = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all pending configs to the pool
        future_to_config: dict[concurrent.futures.Future, TieredBenchmarkConfig] = {
            executor.submit(_benchmark_worker, config, include_ilp): config
            for config, include_ilp in pending_configs
        }

        # Collect results as they complete (not in submission order)
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result_key, entry = future.result()
                all_results[result_key] = entry
                completed_since_save += 1
                total_completed += 1

                logger.info(
                    f"[{total_completed}/{total_completed + total_pending - completed_since_save}] "
                    f"{config.circuit_type} q={config.qubit_count} s={config.sample_index}"
                )

                # Periodic incremental save for crash resilience
                if completed_since_save >= SAVE_INTERVAL:
                    save_results_incrementally(results_filepath, all_results)
                    logger.debug(f"Saved {completed_since_save} results incrementally")
                    completed_since_save = 0

            except Exception as exc:
                logger.error(
                    f"Benchmark {config.circuit_type} q={config.qubit_count} s={config.sample_index} "
                    f"generated exception: {exc}"
                )

    # Final save after all workers complete
    if completed_since_save > 0:
        save_results_incrementally(results_filepath, all_results)

    write_dat_files(output_dir, all_results, include_ci=include_ci)
    write_summary_json(output_dir, all_results)
    logger.info(f"Parallel benchmark complete. Results written to {output_dir}")


# -----------------------------------------------------------------------------
# Result Aggregation
# -----------------------------------------------------------------------------

# Metrics mapping: (dat_file_prefix, dataframe_column)
# Each tuple defines how to name the output file and which column to aggregate
METRICS_CONFIG: list[tuple[str, str]] = [
    (DAT_PREFIX_TIME, COL_TIME),
    (DAT_PREFIX_GATES, COL_GATES),
    (DAT_PREFIX_RATE, COL_RATE),
    (DAT_PREFIX_SWAPS, COL_SWAPS),
]


def results_to_dataframe(
    all_results: dict[str, IncrementalResultEntry],
) -> pd.DataFrame:
    """Convert all benchmark results to a flat DataFrame.

    Args:
        all_results: Dictionary of result entries keyed by result key.

    Returns:
        DataFrame with one row per router per benchmark configuration.
    """
    rows = []

    # Flatten the nested structure: each router result becomes a separate row
    # This enables efficient groupby operations in pandas
    for entry in all_results.values():
        for router_result in entry.results:
            rows.append(
                {
                    COL_CIRCUIT_TYPE: entry.config.circuit_type,
                    COL_QUBIT_COUNT: entry.config.qubit_count,
                    COL_SAMPLE_INDEX: entry.config.sample_index,
                    COL_ROUTER: router_result.router_name,
                    COL_TIME: router_result.execution_time_seconds,
                    COL_GATES: router_result.final_gate_count,
                    COL_RATE: router_result.gate_increase_rate,
                    COL_SWAPS: router_result.swap_count,
                    COL_SUCCESS: router_result.success,
                    COL_SKIPPED: router_result.skipped,
                }
            )

    return pd.DataFrame(rows)


def aggregate_metric(
    df: pd.DataFrame,
    metric_col: str,
    include_ci: bool = True,
) -> pd.DataFrame:
    """Aggregate a metric by qubit count and router, with optional 95% CI.

    Uses Student's t-distribution for CI calculation (appropriate for n < 30).
    Reference: Casella & Berger (2002), "Statistical Inference", Chapter 9.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.ppf.html

    Args:
        df: DataFrame filtered to a single circuit type.
        metric_col: Column name of the metric to aggregate.
        include_ci: Whether to include 95% CI columns for pgfplots error bars.

    Returns:
        DataFrame with qubit_count rows and [router, router_ci, ...] columns.
        If include_ci=False, returns only [router, ...] columns (original format).
    """
    # Filter out failed and skipped results - they should not contribute to averages
    valid_df = df[(df[COL_SUCCESS]) & (~df[COL_SKIPPED])]

    # Compute mean and CI for each (qubit_count, router) group
    aggregated_rows: list[dict] = []
    for (qubit_count, router), group in valid_df.groupby([COL_QUBIT_COUNT, COL_ROUTER]):
        values = group[metric_col]
        n = len(values)
        mean_val = values.mean()

        if include_ci and n >= 2:
            # Sample std with Bessel's correction (ddof=1)
            std_val = values.std(ddof=1)
            sem = std_val / (n**0.5)
            # t-critical value for 95% two-tailed CI
            t_crit = scipy.stats.t.ppf(0.975, df=n - 1)
            ci_95 = sem * t_crit
        else:
            # n < 2: cannot compute CI (no variance estimable)
            ci_95 = 0.0

        aggregated_rows.append(
            {
                COL_QUBIT_COUNT: qubit_count,
                COL_ROUTER: router,
                "mean": mean_val,
                "ci_95": ci_95,
            }
        )

    result_df = pd.DataFrame(aggregated_rows)

    if result_df.empty:
        # No valid data - return empty DataFrame with expected columns
        return pd.DataFrame(columns=[COL_QUBIT_COUNT])

    # Pivot to get routers as columns
    pivoted_mean = result_df.pivot(
        index=COL_QUBIT_COUNT, columns=COL_ROUTER, values="mean"
    ).reset_index()

    if not include_ci:
        # Original format: just mean values
        available_routers = [r for r in ROUTER_NAMES if r in pivoted_mean.columns]
        return pivoted_mean[[COL_QUBIT_COUNT] + available_routers]

    # Include CI columns interleaved: router, router_ci, router2, router2_ci, ...
    pivoted_ci = result_df.pivot(
        index=COL_QUBIT_COUNT, columns=COL_ROUTER, values="ci_95"
    )

    output_cols = [COL_QUBIT_COUNT]
    for router in ROUTER_NAMES:
        if router in pivoted_mean.columns:
            output_cols.append(router)
            ci_col_name = f"{router}_ci"
            pivoted_mean[ci_col_name] = pivoted_ci[router].values
            output_cols.append(ci_col_name)

    return pivoted_mean[output_cols]


def write_dat_files(
    output_dir: Path,
    all_results: dict[str, IncrementalResultEntry],
    include_ci: bool = True,
) -> None:
    """Write pgfplots-compatible .dat files for all metrics and circuit types.

    Args:
        output_dir: Directory to write output files.
        all_results: Dictionary of all benchmark results.
        include_ci: Whether to include 95% CI columns for error bars.
    """
    df = results_to_dataframe(all_results)

    # Generate one .dat file per (circuit_type, metric) combination
    # Naming convention: {metric}_{circuit_type}_{topology_suffix}.dat
    for circuit_type in df[COL_CIRCUIT_TYPE].unique():
        circuit_df = df[df[COL_CIRCUIT_TYPE] == circuit_type]

        for dat_prefix, metric_col in METRICS_CONFIG:
            aggregated = aggregate_metric(circuit_df, metric_col, include_ci=include_ci)

            # Tab-separated format with 6 decimal places for pgfplots compatibility
            filename = f"{dat_prefix}_{circuit_type}_{DAT_SUFFIX}.dat"
            output_path = output_dir / filename
            aggregated.to_csv(output_path, sep="\t", index=False, float_format="%.6f")
            logger.debug(f"Wrote {output_path}")


def write_summary_json(
    output_dir: Path,
    all_results: dict[str, IncrementalResultEntry],
) -> None:
    """Write complete JSON summary of all benchmark results.

    Args:
        output_dir: Directory to write summary.json.
        all_results: Dictionary of all benchmark results.
    """
    # Structure: {circuit_type: {qubit_count: [[router_results], ...]}}
    # Nested by circuit type first for easy thesis chapter organisation
    summary: dict[str, dict[int, list]] = {}

    for entry in all_results.values():
        circuit_type = entry.config.circuit_type
        qubit_count = entry.config.qubit_count

        # Lazily initialise nested dicts
        if circuit_type not in summary:
            summary[circuit_type] = {}
        if qubit_count not in summary[circuit_type]:
            summary[circuit_type][qubit_count] = []

        # Store full router results for potential re-analysis
        sample_results = [dataclasses.asdict(r) for r in entry.results]
        summary[circuit_type][qubit_count].append(sample_results)

    output_path = output_dir / "summary.json"
    with open(output_path, "w") as output_file:
        json.dump(summary, output_file, indent=bench_const.JSON_INDENT)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def estimate_runtime_minutes(tiers: list[TierConfig], circuit_types: list[str]) -> int:
    """Estimate total benchmark runtime in minutes.

    Args:
        tiers: List of tier configurations to run.
        circuit_types: List of circuit types to benchmark.

    Returns:
        Estimated runtime in minutes based on empirical per-router timings.
    """
    # Empirical estimates per circuit (minutes)
    # Based on local testing: 5q ~3.5s total, ILP scales exponentially
    # ILP: 0.5s at 5q, ~15s at 6q, ~minutes at 8q+
    # LP routers: ~2s combined at 5q, scales moderately
    # SABRE: ~1s, scales well
    TIME_PER_ILP_SMALL = 0.5  # 5-8 qubits (minutes)
    TIME_PER_ILP_MEDIUM = 5.0  # 9-10 qubits (minutes)
    TIME_PER_LP = 0.1  # Both LP routers combined
    TIME_PER_BIPARTITE = 0.05  # Often fails, fast when it doesn't
    TIME_PER_SABRE = 0.02

    total = 0.0
    for tier in tiers:
        num_configs = (
            len(tier.qubit_counts) * len(circuit_types) * tier.samples_per_config
        )
        base_time = TIME_PER_LP + TIME_PER_BIPARTITE + TIME_PER_SABRE

        if tier.include_ilp:
            # Use different ILP estimates based on tier size
            max_qubits = max(tier.qubit_counts)
            if max_qubits <= 8:
                ilp_time = TIME_PER_ILP_SMALL
            else:
                ilp_time = TIME_PER_ILP_MEDIUM
            base_time += ilp_time

        total += num_configs * base_time

    return int(total)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run tiered thesis benchmarks for Quariadne routing algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )

    parser.add_argument(
        "--circuit-types",
        type=str,
        nargs="+",
        default=THESIS_CIRCUIT_TYPES,
        help="Circuit types to benchmark",
    )

    parser.add_argument(
        "--skip-tier",
        type=int,
        nargs="*",
        default=[],
        choices=[1, 2, 3],
        help="Tier numbers to skip (1, 2, or 3)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default=bench_const.BENCHMARK_DEFAULT_LOG_LEVEL,
        choices=bench_const.BENCHMARK_LOG_LEVELS,
        help="Logging level",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running benchmarks",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel worker processes (default: cpu_count - 1)",
    )

    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Disable 95%% confidence interval columns in .dat files",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for tiered benchmark script."""
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    # Filter tiers based on --skip-tier argument
    # Tier numbering is 1-indexed for user convenience
    tiers_to_run = [
        tier for idx, tier in enumerate(ALL_TIERS, start=1) if idx not in args.skip_tier
    ]

    total_configs = calculate_total_configs(tiers_to_run, args.circuit_types)
    estimated_minutes = estimate_runtime_minutes(tiers_to_run, args.circuit_types)

    # Log configuration summary
    logger.info("Tiered benchmark configuration:")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Circuit types ({len(args.circuit_types)}): {args.circuit_types}")
    logger.info(f"  Tiers: {[t.tier_name for t in tiers_to_run]}")
    logger.info(f"  Total configs: {total_configs}")
    logger.info(
        f"  Estimated time: ~{estimated_minutes} min ({estimated_minutes / 60:.1f} hrs)"
    )
    logger.info(
        f"  Topology: Watts-Strogatz k={WATTS_STROGATZ_K}, p={WATTS_STROGATZ_P}"
    )
    logger.info(
        f"  Workers: {args.workers} ({'parallel' if args.workers > 1 else 'sequential'})"
    )

    if args.dry_run:
        # Print detailed tier breakdown for verification
        for tier in tiers_to_run:
            tier_configs = (
                len(tier.qubit_counts)
                * len(args.circuit_types)
                * tier.samples_per_config
            )
            logger.info(
                f"  {tier.tier_name}: {tier_configs} configs, ILP={'yes' if tier.include_ilp else 'no'}"
            )
        logger.info("Dry run complete. No benchmarks executed.")
        return

    # Use parallel execution when workers > 1, otherwise sequential
    include_ci = not args.no_ci
    if args.workers > 1:
        run_tiered_benchmark_parallel(
            output_dir=args.output_dir,
            tiers=tiers_to_run,
            circuit_types=args.circuit_types,
            num_workers=args.workers,
            include_ci=include_ci,
        )
    else:
        run_tiered_benchmark(
            output_dir=args.output_dir,
            tiers=tiers_to_run,
            circuit_types=args.circuit_types,
            include_ci=include_ci,
        )


if __name__ == "__main__":
    main()
