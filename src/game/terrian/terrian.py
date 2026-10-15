from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Terrian:
    """
    Immutable definition of a terrian type.
    """

    name: str
    traversable: bool
    buildable: bool
