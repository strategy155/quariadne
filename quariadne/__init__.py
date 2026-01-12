"""Quariadne: Quantum circuit routing optimisation library.

This package provides tools for optimising qubit routing in quantum circuits
using various LP and MILP formulations.

References:
    - Python 3.13 __all__: https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
"""

# Circuit primitives
from quariadne.circuit import (
    AbstractQuantumCircuit,
    LogicalQubit,
    PhysicalQubit,
    PhysicalSwap,
    QuantumOperation,
)

# Router base class and scipy-based implementations
from quariadne.routers import (
    IlpRouter,
    LpRouterEdges,
    LpRouterMapping,
    Router,
)

# HiGHS-based LP routers
from quariadne.bipartite_allocation_lp import BipartiteAllocationRouter
from quariadne.lp_router_layered import LayeredLPResult, LpRouterLayered
from quariadne.lp_router_unified import LpRouterUnified, UnifiedLPResult

# Configuration
from quariadne.lp_router_base import LPSolverOptions

__all__ = [
    # Circuit primitives
    "AbstractQuantumCircuit",
    "LogicalQubit",
    "PhysicalQubit",
    "PhysicalSwap",
    "QuantumOperation",
    # Router base
    "Router",
    # Scipy-based routers
    "IlpRouter",
    "LpRouterMapping",
    "LpRouterEdges",
    # HiGHS-based routers
    "BipartiteAllocationRouter",
    "LpRouterLayered",
    "LpRouterUnified",
    # Result types
    "LayeredLPResult",
    "UnifiedLPResult",
    # Configuration
    "LPSolverOptions",
]
