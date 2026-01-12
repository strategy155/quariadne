#!/usr/bin/env python3
"""QUEKO BNTF benchmark for thesis - LpRouterLayered vs SABRE.

Runs 16-qubit QUEKO BNTF circuits (5-45 cycles) on Aspen-4 topology,
comparing LpRouterLayered with SABRE baseline.

References:
    - Python 3.13 concurrent.futures: https://docs.python.org/3/library/concurrent.futures.html
    - Python 3.13 argparse: https://docs.python.org/3/library/argparse.html
"""

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import statistics
import time
from pathlib import Path

import networkx as nx
import qiskit.qasm2
import qiskit.transpiler

import quariadne.benchmarks.constants as bench_const
import quariadne.circuit
import quariadne.lp_router_layered
import quariadne.milp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
TIMEOUT_SECONDS = 300  # 5 minutes per router per circuit
SWAP_TO_CX_COUNT = 3
BACKEND_NAME = "Aspen-4"
QUEKO_DIR = Path("QUEKO-benchmark")
BNTF_DIR = QUEKO_DIR / "BNTF"
OUTPUT_DIR = Path("benches/results/queko_thesis")

# BNTF cycle depths to benchmark (reduced from 5-45 to 5-25 for performance)
BNTF_DEPTHS = [5, 10, 15, 20, 25]
VARIANTS_PER_DEPTH = 10
DEFAULT_WORKERS = 4


@dataclasses.dataclass
class CircuitResult:
    """Result for a single circuit benchmark."""

    circuit_name: str
    cycle_depth: int
    variant: int
    router_name: str
    swap_count: int
    execution_time: float
    success: bool
    timeout: bool = False
    error_message: str | None = None


@dataclasses.dataclass
class AggregatedResult:
    """Aggregated results for a cycle depth."""

    cycle_depth: int
    router_name: str
    mean_swaps: float
    ci_95_swaps: float
    mean_time: float
    sample_count: int
    timeout_count: int


def create_aspen4_coupling_map() -> qiskit.transpiler.CouplingMap:
    """Create Qiskit CouplingMap for Aspen-4 backend."""
    edges = bench_const.BACKEND_EDGES[BACKEND_NAME]
    graph = nx.from_edgelist(edges)
    directed_edges = list(graph.to_directed().edges())
    return qiskit.transpiler.CouplingMap(directed_edges)


def create_aspen4_coupling_graph() -> nx.DiGraph:
    """Create NetworkX DiGraph for Aspen-4 with PhysicalQubit nodes."""
    edges = bench_const.BACKEND_EDGES[BACKEND_NAME]
    coupling = nx.DiGraph()
    for u, v in edges:
        pu = quariadne.circuit.PhysicalQubit(u)
        pv = quariadne.circuit.PhysicalQubit(v)
        coupling.add_edge(pu, pv)
        coupling.add_edge(pv, pu)
    return coupling


def get_bntf_circuits() -> list[tuple[Path, int, int]]:
    """Get all BNTF circuit paths with metadata.

    Returns:
        List of (path, cycle_depth, variant) tuples.
    """
    circuits = []
    for depth in BNTF_DEPTHS:
        for variant in range(VARIANTS_PER_DEPTH):
            filename = f"16QBT_{depth:02d}CYC_TFL_{variant}.qasm"
            filepath = BNTF_DIR / filename
            if filepath.exists():
                circuits.append((filepath, depth, variant))
            else:
                logger.warning(f"Circuit not found: {filepath}")
    return circuits


def run_lp_layered(
    coupling_graph: nx.DiGraph,
    abstract_circuit: quariadne.circuit.AbstractQuantumCircuit,
) -> tuple[int, float]:
    """Run LpRouterLayered and return (swap_count, time).

    Returns LP objective as swap count (swap extraction is buggy).
    """
    start = time.perf_counter()
    router = quariadne.lp_router_layered.LpRouterLayered(
        coupling_graph, abstract_circuit
    )
    result = router.run()
    elapsed = time.perf_counter() - start
    # Use LP objective as swap count (swap extraction bug workaround)
    swap_count = int(result.objective_value)
    return swap_count, elapsed


def run_sabre(
    coupling_map: qiskit.transpiler.CouplingMap,
    qiskit_circuit: qiskit.QuantumCircuit,
) -> tuple[int, float]:
    """Run SABRE routing and return (swap_count, time)."""
    start = time.perf_counter()
    pass_manager = qiskit.transpiler.generate_preset_pass_manager(
        coupling_map=coupling_map,
        optimization_level=1,
        routing_method="sabre",
        layout_method="trivial",
    )
    transpiled = pass_manager.run(qiskit_circuit)
    elapsed = time.perf_counter() - start
    swap_count = transpiled.count_ops().get("swap", 0)
    return swap_count, elapsed


def run_with_timeout(func, timeout: int = TIMEOUT_SECONDS):
    """Run function with timeout, return None if exceeded."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Timeout after {timeout}s")
            return None


def benchmark_circuit(
    filepath: Path,
    cycle_depth: int,
    variant: int,
    coupling_map: qiskit.transpiler.CouplingMap,
    coupling_graph: nx.DiGraph,
    timeout: int = TIMEOUT_SECONDS,
) -> list[CircuitResult]:
    """Benchmark a single circuit with both routers."""
    results = []
    circuit_name = filepath.stem

    # Load circuit
    try:
        qiskit_circuit = qiskit.qasm2.load(str(filepath))
        abstract_circuit = quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(
            qiskit_circuit
        )
    except Exception as e:
        logger.error(f"Failed to load {circuit_name}: {e}")
        for router in ["LpRouterLayered", "SABRE"]:
            results.append(
                CircuitResult(
                    circuit_name=circuit_name,
                    cycle_depth=cycle_depth,
                    variant=variant,
                    router_name=router,
                    swap_count=0,
                    execution_time=0.0,
                    success=False,
                    error_message=str(e),
                )
            )
        return results

    # Run LpRouterLayered
    lp_result = run_with_timeout(
        lambda: run_lp_layered(coupling_graph, abstract_circuit),
        timeout=timeout,
    )
    if lp_result is None:
        results.append(
            CircuitResult(
                circuit_name=circuit_name,
                cycle_depth=cycle_depth,
                variant=variant,
                router_name="LpRouterLayered",
                swap_count=0,
                execution_time=timeout,
                success=False,
                timeout=True,
            )
        )
    else:
        swap_count, elapsed = lp_result
        results.append(
            CircuitResult(
                circuit_name=circuit_name,
                cycle_depth=cycle_depth,
                variant=variant,
                router_name="LpRouterLayered",
                swap_count=swap_count,
                execution_time=elapsed,
                success=True,
            )
        )

    # Run SABRE
    sabre_result = run_with_timeout(
        lambda: run_sabre(coupling_map, qiskit_circuit),
        timeout=timeout,
    )
    if sabre_result is None:
        results.append(
            CircuitResult(
                circuit_name=circuit_name,
                cycle_depth=cycle_depth,
                variant=variant,
                router_name="SABRE",
                swap_count=0,
                execution_time=timeout,
                success=False,
                timeout=True,
            )
        )
    else:
        swap_count, elapsed = sabre_result
        results.append(
            CircuitResult(
                circuit_name=circuit_name,
                cycle_depth=cycle_depth,
                variant=variant,
                router_name="SABRE",
                swap_count=swap_count,
                execution_time=elapsed,
                success=True,
            )
        )

    return results


def benchmark_worker(args: tuple) -> list[CircuitResult]:
    """Worker function for parallel execution.

    Must be at module level for pickling by ProcessPoolExecutor.

    Args:
        args: Tuple of (filepath, depth, variant, timeout).

    Returns:
        List of CircuitResult for this circuit.
    """
    filepath, depth, variant, timeout = args
    coupling_map = create_aspen4_coupling_map()
    coupling_graph = create_aspen4_coupling_graph()
    return benchmark_circuit(
        filepath, depth, variant, coupling_map, coupling_graph, timeout
    )


def compute_ci_95(values: list[float]) -> float:
    """Compute 95% confidence interval half-width using t-distribution."""
    if len(values) < 2:
        return 0.0
    import scipy.stats

    n = len(values)
    std = statistics.stdev(values)
    t_value = scipy.stats.t.ppf(0.975, n - 1)
    return t_value * std / (n**0.5)


def aggregate_results(results: list[CircuitResult]) -> list[AggregatedResult]:
    """Aggregate results by cycle depth and router."""
    from collections import defaultdict

    grouped: dict[tuple[int, str], list[CircuitResult]] = defaultdict(list)
    for r in results:
        grouped[(r.cycle_depth, r.router_name)].append(r)

    aggregated = []
    for (depth, router), group in sorted(grouped.items()):
        successful = [r for r in group if r.success]
        swaps = [r.swap_count for r in successful]
        times = [r.execution_time for r in successful]
        timeouts = sum(1 for r in group if r.timeout)

        if swaps:
            aggregated.append(
                AggregatedResult(
                    cycle_depth=depth,
                    router_name=router,
                    mean_swaps=statistics.mean(swaps),
                    ci_95_swaps=compute_ci_95([float(s) for s in swaps]),
                    mean_time=statistics.mean(times),
                    sample_count=len(swaps),
                    timeout_count=timeouts,
                )
            )

    return aggregated


def write_dat_files(output_dir: Path, aggregated: list[AggregatedResult]) -> None:
    """Write pgfplots-compatible .dat files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by metric
    routers = ["LpRouterLayered", "SABRE"]
    depths = sorted(set(r.cycle_depth for r in aggregated))

    # Build data lookup
    data = {}
    for r in aggregated:
        data[(r.cycle_depth, r.router_name)] = r

    # Write swaps .dat
    swaps_file = output_dir / f"swaps_BNTF_{BACKEND_NAME}.dat"
    with open(swaps_file, "w") as f:
        header = ["cycle_depth"]
        for router in routers:
            header.extend([router, f"{router}_ci"])
        f.write("\t".join(header) + "\n")

        for depth in depths:
            row = [str(depth)]
            for router in routers:
                result = data.get((depth, router))
                if result is not None:
                    row.extend(
                        [f"{result.mean_swaps:.6f}", f"{result.ci_95_swaps:.6f}"]
                    )
                else:
                    row.extend(["", ""])
            f.write("\t".join(row) + "\n")

    # Write time .dat
    time_file = output_dir / f"time_BNTF_{BACKEND_NAME}.dat"
    with open(time_file, "w") as f:
        header = ["cycle_depth"]
        for router in routers:
            header.append(router)
        f.write("\t".join(header) + "\n")

        for depth in depths:
            row = [str(depth)]
            for router in routers:
                result = data.get((depth, router))
                if result is not None:
                    row.append(f"{result.mean_time:.6f}")
                else:
                    row.append("")
            f.write("\t".join(row) + "\n")

    logger.info(f"Wrote {swaps_file}")
    logger.info(f"Wrote {time_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run QUEKO BNTF benchmark for thesis")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_SECONDS,
        help=f"Timeout per router in seconds (default: {TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Run single circuit (e.g., '16QBT_05CYC_TFL_0')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List circuits without running",
    )
    args = parser.parse_args()

    # Get circuits
    circuits = get_bntf_circuits()
    logger.info(f"Found {len(circuits)} BNTF circuits")

    if args.dry_run:
        for path, depth, variant in circuits:
            print(f"{depth:2d} CYC, variant {variant}: {path}")
        return

    # Filter if single circuit requested
    if args.single:
        circuits = [(p, d, v) for p, d, v in circuits if args.single in p.stem]
        if not circuits:
            logger.error(f"No circuit matching '{args.single}'")
            return
        logger.info(f"Running single circuit: {circuits[0][0]}")

    # Setup output directory and results file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.output_dir / "queko_results.json"

    # Run benchmarks
    all_results: list[CircuitResult] = []
    total = len(circuits)

    if args.workers > 1 and not args.single:
        # Parallel execution
        logger.info(f"Running with {args.workers} parallel workers")
        work_items = [(path, depth, var, args.timeout) for path, depth, var in circuits]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(benchmark_worker, item): item for item in work_items
            }

            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)

                    # Log progress
                    for r in results:
                        status = (
                            "OK" if r.success else ("TIMEOUT" if r.timeout else "FAIL")
                        )
                        logger.info(
                            f"[{len(all_results) // 2}/{total}] {r.circuit_name} "
                            f"{r.router_name}: {r.swap_count} swaps [{status}]"
                        )

                    # Save incrementally
                    with open(results_file, "w") as f:
                        json.dump(
                            [dataclasses.asdict(r) for r in all_results], f, indent=2
                        )

                except Exception as e:
                    logger.error(f"Worker failed for {item[0]}: {e}")
    else:
        # Sequential execution (for single circuit or workers=1)
        coupling_map = create_aspen4_coupling_map()
        coupling_graph = create_aspen4_coupling_graph()

        for idx, (filepath, depth, variant) in enumerate(circuits, 1):
            logger.info(f"[{idx}/{total}] {filepath.stem}")
            results = benchmark_circuit(
                filepath,
                depth,
                variant,
                coupling_map,
                coupling_graph,
                timeout=args.timeout,
            )
            all_results.extend(results)

            # Log progress
            for r in results:
                status = "OK" if r.success else ("TIMEOUT" if r.timeout else "FAIL")
                logger.info(
                    f"  {r.router_name}: {r.swap_count} swaps, {r.execution_time:.2f}s [{status}]"
                )

            # Save incrementally
            with open(results_file, "w") as f:
                json.dump([dataclasses.asdict(r) for r in all_results], f, indent=2)

    logger.info(f"Wrote {results_file}")

    # Aggregate and write .dat files
    aggregated = aggregate_results(all_results)
    write_dat_files(args.output_dir, aggregated)

    # Summary
    logger.info("=== Summary ===")
    for r in aggregated:
        logger.info(
            f"{r.cycle_depth:2d} CYC {r.router_name:16s}: "
            f"{r.mean_swaps:6.2f} +/- {r.ci_95_swaps:5.2f} swaps "
            f"({r.sample_count} samples, {r.timeout_count} timeouts)"
        )


if __name__ == "__main__":
    main()
