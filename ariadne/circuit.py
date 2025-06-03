@dataclasses.dataclass(frozen=True)
class Qubit:
    is_start: bool
    index: int

@dataclasses.dataclass(frozen=True)
class Gate:
    qubits_participating: int
    name: str
    parent_id: int = dataclasses.field(repr=False)

