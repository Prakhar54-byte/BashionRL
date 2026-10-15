from dataclasses import dataclass

from src.game.terrian.terrian import Terrian

@dataclass(slots=True)
class TerrianBlock:
    """
    A single cell in the terrian grid.
    """
    row: int
    col: int
    terrian: Terrian