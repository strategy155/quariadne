"""Data structures for benchmark results and CSV export.

This module provides structured representations of quantum circuit routing
benchmark results, including circuit characteristics, algorithm performance
metrics, and status handling.
"""

import dataclasses
import enum


class AlgorithmStatus(enum.Enum):
    """Status of a routing algorithm execution.

    Attributes:
        SUCCESS: Algorithm completed successfully.
        TIMEOUT: Algorithm exceeded time limit.
        FAILED: Algorithm failed for other reasons (e.g., infeasibility).
    """

    SUCCESS = "success"
    TIMEOUT = "TIMEOUT"
    FAILED = "NA"


# CSV column name prefixes
_COL_COUNT = "count"
_COL_RATE = "rate"
_COL_TIME = "time"

# Algorithm names for CSV columns
_ALGO_EDGE = "edge"
_ALGO_QUBIT = "qubit"
_ALGO_ILP = "ilp"
_ALGO_SABRE = "sabre"
_ALGO_BASIC = "basic"
_ALGO_LOOKAHEAD = "lookahead"


@dataclasses.dataclass(frozen=True)
class CircuitCharacteristics:
    """Original quantum circuit characteristics before routing.

    Attributes:
        circuit_id: Unique identifier for the circuit (e.g., "bntf_5_10").
        gates: Total number of gates in original circuit.
        qubits: Number of logical qubits in circuit.
        depth: Circuit depth (longest path through operations).
        hardware_qubits: Number of physical qubits in coupling map.
        hardware_connectivity: Average node degree in coupling graph.
    """

    circuit_id: str
    gates: int
    qubits: int
    depth: int
    hardware_qubits: int
    hardware_connectivity: float


@dataclasses.dataclass(frozen=True)
class AlgorithmResult:
    """Performance metrics for a single routing algorithm.

    Attributes:
        status: Execution status (success, timeout, or failure).
        gate_count: Final gate count after routing (None if not successful).
        gate_increase_rate: Ratio of final to original gate count (None if not successful).
        solve_time_seconds: Execution time in seconds (None if not successful).
    """

    status: AlgorithmStatus
    gate_count: int | None
    gate_increase_rate: float | None
    solve_time_seconds: float | None


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """Complete benchmark result for one circuit across all algorithms.

    Attributes:
        circuit: Original circuit characteristics.
        edge_mapping: Results for LP with edge-based assignment (LpRouterEdges).
        qubit_mapping: Results for LP with Birkhoff decomposition (LpRouterMapping).
        ilp: Results for Integer Linear Programming (IlpRouter).
        sabre: Results for Qiskit Sabre routing.
        basic: Results for Qiskit Basic routing.
        lookahead: Results for Qiskit Lookahead routing.
    """

    circuit: CircuitCharacteristics
    edge_mapping: AlgorithmResult
    qubit_mapping: AlgorithmResult
    ilp: AlgorithmResult
    sabre: AlgorithmResult
    basic: AlgorithmResult
    lookahead: AlgorithmResult

    def to_csv_row(self) -> dict[str, str]:
        """Convert benchmark result to CSV row dictionary.

        Returns:
            Dictionary mapping CSV column names to formatted string values.
            Uses "TIMEOUT" or "NA" markers for failed algorithms.
        """
        row = {
            "circuit_id": self.circuit.circuit_id,
            "gates": str(self.circuit.gates),
            "qubits": str(self.circuit.qubits),
            "depth": str(self.circuit.depth),
            "hw_qubits": str(self.circuit.hardware_qubits),
            "hw_connectivity": f"{self.circuit.hardware_connectivity:.2f}",
        }

        # Add results for each algorithm
        for algo_name, algo_result in [
            (_ALGO_EDGE, self.edge_mapping),
            (_ALGO_QUBIT, self.qubit_mapping),
            (_ALGO_ILP, self.ilp),
            (_ALGO_SABRE, self.sabre),
            (_ALGO_BASIC, self.basic),
            (_ALGO_LOOKAHEAD, self.lookahead),
        ]:
            if algo_result.status == AlgorithmStatus.SUCCESS:
                row[f"{_COL_COUNT}_{algo_name}"] = str(algo_result.gate_count)
                row[f"{_COL_RATE}_{algo_name}"] = (
                    f"{algo_result.gate_increase_rate:.3f}"
                )
                row[f"{_COL_TIME}_{algo_name}"] = (
                    f"{algo_result.solve_time_seconds:.2f}"
                )
            else:
                marker = algo_result.status.value
                row[f"{_COL_COUNT}_{algo_name}"] = marker
                row[f"{_COL_RATE}_{algo_name}"] = marker
                row[f"{_COL_TIME}_{algo_name}"] = marker

        return row

    @staticmethod
    def csv_headers() -> list[str]:
        """Get CSV column headers in correct order.

        Returns:
            List of column header names matching to_csv_row() keys.
        """
        headers = [
            "circuit_id",
            "gates",
            "qubits",
            "depth",
            "hw_qubits",
            "hw_connectivity",
        ]

        # Add algorithm columns in order
        for algo_name in [
            _ALGO_EDGE,
            _ALGO_QUBIT,
            _ALGO_ILP,
            _ALGO_SABRE,
            _ALGO_BASIC,
            _ALGO_LOOKAHEAD,
        ]:
            headers.extend(
                [
                    f"{_COL_COUNT}_{algo_name}",
                    f"{_COL_RATE}_{algo_name}",
                    f"{_COL_TIME}_{algo_name}",
                ]
            )

        return headers
