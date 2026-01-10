"""Pre-computed shortest paths for known device topologies.

This module provides a caching layer for shortest path computations on
device coupling graphs. It supports:
- Compressed persistent storage (gzip pickle)
- Pre-registration of known backends (QUEKO benchmarks)
- Lazy computation for unknown topologies

References:
    - NetworkX all_pairs_shortest_path: https://networkx.org/documentation/stable/
    - Python 3.13 dataclasses: https://docs.python.org/3/library/dataclasses.html
    - Python 3.13 gzip: https://docs.python.org/3/library/gzip.html
"""

import gzip
import logging
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

import quariadne.benchmarks.constants

# Build cache file path from constants
_PACKAGE_DIR = Path(__file__).parent
CACHE_FILE = (
    _PACKAGE_DIR
    / quariadne.benchmarks.constants.TOPOLOGY_CACHE_SUBDIR
    / quariadne.benchmarks.constants.TOPOLOGY_CACHE_FILENAME
)

# Type aliases
ShortestPathsDict = dict[tuple[int, int], tuple[int, ...]]
DistancesDict = dict[tuple[int, int], int]


@dataclass(frozen=True)
class DeviceTopology:
    """Immutable container for cached device topology data.

    Stores pre-computed shortest paths and distances for a device's
    coupling graph. Frozen to ensure safe sharing across threads.

    Attributes:
        paths: Mapping from (source, target) pairs to shortest path as tuple.
        distances: Mapping from (source, target) pairs to path length.
    """

    paths: ShortestPathsDict
    distances: DistancesDict


def _compute_topology(edges: Iterable[tuple[int, int]]) -> DeviceTopology:
    """Compute shortest paths and distances for an edge set.

    Args:
        edges: Iterable of (source, target) tuples representing connectivity.

    Returns:
        DeviceTopology with computed paths and distances.
    """
    graph: nx.Graph[int] = nx.Graph()
    graph.add_edges_from(edges)

    paths: ShortestPathsDict = {}
    distances: DistancesDict = {}

    for source, target_paths in nx.all_pairs_shortest_path(graph):
        for target, path in target_paths.items():
            # Convert to tuple for immutability
            paths[(source, target)] = tuple(path)
            distances[(source, target)] = len(path) - 1

    return DeviceTopology(paths=paths, distances=distances)


def _load_cache() -> dict[str, DeviceTopology]:
    """Load pre-computed topologies from compressed file."""
    if not CACHE_FILE.exists():
        logging.warning(quariadne.benchmarks.constants.TOPOLOGY_CACHE_MISSING_WARNING)
        return quariadne.benchmarks.constants.EMPTY_TOPOLOGY_CACHE
    with gzip.open(CACHE_FILE, "rb") as f:
        loaded_cache: dict[str, DeviceTopology] = pickle.load(f)
        return loaded_cache


def _save_cache(cache: dict[str, DeviceTopology]) -> None:
    """Save topologies to compressed file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


# Load cache at import time
TOPOLOGY_CACHE: dict[str, DeviceTopology] = _load_cache()

# Build edge-based index for fast lookup by topology
_EDGE_INDEX: dict[frozenset[tuple[int, int]], DeviceTopology] = {}
for _name, _topo in TOPOLOGY_CACHE.items():
    _edges = frozenset(quariadne.benchmarks.constants.BACKEND_EDGES[_name])
    _EDGE_INDEX[_edges] = _topo


def get_topology(name: str) -> DeviceTopology | None:
    """Get cached topology for a known backend.

    Args:
        name: Backend name (e.g., "Aspen-4", "Sycamore").

    Returns:
        DeviceTopology if backend is cached, None otherwise.
    """
    return TOPOLOGY_CACHE.get(name)


def get_topology_by_edges(
    edges: frozenset[tuple[int, int]],
) -> DeviceTopology | None:
    """Get cached topology by edge set.

    Args:
        edges: Frozenset of (source, target) tuples defining the coupling graph.

    Returns:
        DeviceTopology if topology is cached, None otherwise.
    """
    return _EDGE_INDEX.get(edges)


def generate_cache() -> None:
    """Generate cache for all known backends."""
    cache = {
        name: _compute_topology(edges)
        for name, edges in quariadne.benchmarks.constants.BACKEND_EDGES.items()
    }
    _save_cache(cache)
    print(f"Generated cache with {len(cache)} backends at {CACHE_FILE}")


if __name__ == "__main__":
    generate_cache()
