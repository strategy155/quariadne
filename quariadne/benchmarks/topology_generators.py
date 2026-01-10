"""Hardware topology generators for benchmarking.

This module provides functions to generate synthetic coupling maps for
the thesis scaling study, using Watts-Strogatz small-world graphs.

References:
    - NetworkX watts_strogatz_graph:
      https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.watts_strogatz_graph.html
    - Watts & Strogatz 1998: Collective dynamics of 'small-world' networks
"""

import networkx

import qiskit.transpiler

import quariadne.benchmarks.constants as bench_const


def generate_watts_strogatz_coupling_map(
    num_qubits: int,
    k: int = bench_const.WATTS_STROGATZ_DEFAULT_K,
    p: float = bench_const.WATTS_STROGATZ_DEFAULT_P,
    seed: int | None = None,
) -> qiskit.transpiler.CouplingMap:
    """Generate Watts-Strogatz small-world topology as CouplingMap.

    Creates a connected small-world graph using the Watts-Strogatz model,
    then converts it to a Qiskit CouplingMap with bidirectional edges.

    Args:
        num_qubits: Number of nodes (physical qubits) in the topology.
        k: Number of nearest neighbours to connect (must be even, k >= 2).
        p: Rewiring probability (0 = ring lattice, 1 = random graph).
        seed: Random seed for reproducibility.

    Returns:
        Qiskit CouplingMap with bidirectional edges.

    Raises:
        networkx.NetworkXError: If k > num_qubits or k is odd.

    Reference:
        https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.watts_strogatz_graph.html
    """
    # Generate Watts-Strogatz graph (undirected)
    ws_graph = networkx.watts_strogatz_graph(num_qubits, k, p, seed=seed)

    # Convert to directed graph for Qiskit (bidirectional edges)
    directed_graph = ws_graph.to_directed()

    coupling_map = qiskit.transpiler.CouplingMap(list(directed_graph.edges()))
    return coupling_map
