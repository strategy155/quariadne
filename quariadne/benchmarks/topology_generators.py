"""Synthetic hardware topology generation for benchmarking.

This module provides functions to generate synthetic coupling graphs
for quantum hardware topology benchmarking. Supports Watts-Strogatz
small-world graphs and configurable grid/lattice structures.

Watts-Strogatz graphs are useful for testing router scalability on
synthetic hardware with tunable connectivity, replicating the typical
quantum hardware graph property of having a big connected component
with interleaving edges.

References:
    - NetworkX graph generators:
      https://networkx.org/documentation/stable/reference/generators.html
    - Python 3.13 random module:
      https://docs.python.org/3/library/random.html
    - Python 3.13 typing module:
      https://docs.python.org/3/library/typing.html#typing.TypedDict
    - Watts & Strogatz 1998: "Collective dynamics of 'small-world' networks"
    - Google Python Style Guide:
      https://google.github.io/styleguide/pyguide.html
"""

from typing import TypedDict

import networkx as nx
import qiskit.transpiler


class TopologyStatistics(TypedDict):
    """Statistics for characterising hardware topology complexity.

    Attributes:
        num_nodes: Number of nodes (physical qubits).
        num_edges: Number of undirected edges.
        avg_degree: Average node degree.
        diameter: Graph diameter (longest shortest path), inf if disconnected.
        clustering: Average clustering coefficient.
    """

    num_nodes: int
    num_edges: int
    avg_degree: float
    diameter: float
    clustering: float


def generate_connected_watts_strogatz_topology(
    num_qubits: int,
    nearest_neighbours: int,
    rewiring_probability: float,
    seed: int,
    max_connection_attempts: int = 100,
) -> nx.DiGraph:
    """Generate a connected Watts-Strogatz small-world graph as coupling topology.

    Creates a random graph with small-world properties: high clustering
    coefficient and short average path length. Uses NetworkX's
    connected_watts_strogatz_graph to guarantee connectivity.

    Args:
        num_qubits: Number of physical qubits (graph nodes). Must be > k.
        nearest_neighbours: Each node connects to k nearest neighbours in
            ring topology. Must be even and >= 2.
        rewiring_probability: Probability of rewiring each edge. p=0 gives
            ring lattice, p=1 gives random graph. Must be in [0, 1].
        seed: Random seed for reproducibility.
        max_connection_attempts: Maximum attempts to ensure connectivity
            (passed to NetworkX's tries parameter).

    Returns:
        NetworkX DiGraph with bidirectional edges suitable for routing.
        Guaranteed to be connected.

    Raises:
        ValueError: If parameters are out of valid ranges.
        NetworkXError: If connectivity cannot be achieved within max_attempts.

    Example:
        >>> graph = generate_connected_watts_strogatz_topology(16, 4, 0.3, 42)
        >>> nx.is_connected(graph.to_undirected())
        True

    Reference:
        - NetworkX connected_watts_strogatz_graph:
          https://networkx.org/documentation/stable/reference/generated/
          networkx.generators.random_graphs.connected_watts_strogatz_graph.html
    """
    if num_qubits < 3:
        raise ValueError(f"num_qubits must be at least 3, got {num_qubits}")
    if nearest_neighbours < 2 or nearest_neighbours % 2 != 0:
        raise ValueError(
            f"nearest_neighbours must be even and >= 2, got {nearest_neighbours}"
        )
    if nearest_neighbours >= num_qubits:
        raise ValueError(
            f"nearest_neighbours ({nearest_neighbours}) must be < "
            f"num_qubits ({num_qubits})"
        )
    if not 0 <= rewiring_probability <= 1:
        raise ValueError(
            f"rewiring_probability must be in [0, 1], got {rewiring_probability}"
        )

    # Generate connected undirected Watts-Strogatz graph
    undirected_graph = nx.connected_watts_strogatz_graph(
        n=num_qubits,
        k=nearest_neighbours,
        p=rewiring_probability,
        tries=max_connection_attempts,
        seed=seed,
    )

    # Convert to bidirectional DiGraph for Quariadne compatibility
    directed_graph = undirected_graph.to_directed()
    return directed_graph


def connected_watts_strogatz_to_coupling_map(
    num_qubits: int,
    nearest_neighbours: int,
    rewiring_probability: float,
    seed: int,
) -> qiskit.transpiler.CouplingMap:
    """Generate Qiskit CouplingMap from connected Watts-Strogatz parameters.

    Convenience function that wraps generate_connected_watts_strogatz_topology
    and converts the result to a Qiskit CouplingMap.

    Args:
        num_qubits: Number of physical qubits.
        nearest_neighbours: k parameter for Watts-Strogatz.
        rewiring_probability: p parameter for Watts-Strogatz.
        seed: Random seed for reproducibility.

    Returns:
        Qiskit CouplingMap for transpiler use.

    Example:
        >>> coupling_map = connected_watts_strogatz_to_coupling_map(16, 4, 0.3, 42)
        >>> isinstance(coupling_map, qiskit.transpiler.CouplingMap)
        True
    """
    directed_graph = generate_connected_watts_strogatz_topology(
        num_qubits=num_qubits,
        nearest_neighbours=nearest_neighbours,
        rewiring_probability=rewiring_probability,
        seed=seed,
    )
    edge_list = list(directed_graph.edges())
    return qiskit.transpiler.CouplingMap(edge_list)


def get_topology_statistics(graph: nx.Graph | nx.DiGraph) -> TopologyStatistics:
    """Compute topology statistics for benchmarking analysis.

    Calculates graph-theoretic metrics useful for characterising
    hardware topology complexity.

    Args:
        graph: NetworkX graph (directed or undirected).

    Returns:
        TopologyStatistics with computed metrics.

    Example:
        >>> graph = generate_connected_watts_strogatz_topology(10, 4, 0.3, 42)
        >>> stats = get_topology_statistics(graph)
        >>> stats["num_nodes"]
        10
    """
    # Convert to undirected for meaningful metrics
    if graph.is_directed():
        undirected_graph = graph.to_undirected()
    else:
        undirected_graph = graph

    num_nodes = undirected_graph.number_of_nodes()
    num_edges = undirected_graph.number_of_edges()

    # Average degree calculation
    degree_sum = sum(degree for _, degree in undirected_graph.degree())
    average_degree = degree_sum / num_nodes if num_nodes > 0 else 0.0

    # Diameter (requires connected graph)
    if nx.is_connected(undirected_graph):
        diameter = float(nx.diameter(undirected_graph))
    else:
        diameter = float("inf")

    # Average clustering coefficient
    average_clustering = nx.average_clustering(undirected_graph)

    return TopologyStatistics(
        num_nodes=num_nodes,
        num_edges=num_edges,
        avg_degree=average_degree,
        diameter=diameter,
        clustering=average_clustering,
    )
