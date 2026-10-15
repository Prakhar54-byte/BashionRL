from _typeshed import Self

from src.game.terrian.terrian import Terrian
from src.game.terrian.terrian_block import TerrianBlock


class TerrianMap:
    """
    Stores and provides access to a rectangular terrian grid.
    """

    def __init__(self, rows: int, cols: int, default_terrian: Terrian) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("Terrian Map dimensions must be positive")

        self.rows = rows
        self.cols = cols

        self._grid = [
            [
                TerrianBlock(row=row, col=col, terrian=default_terrian)
                for col in range(cols)
            ]
            for row in range(rows)
        ]

    def in_bounds(self, row: int, col: int) -> bool:
        """
        Return whether a cell coordinate is inside the map
        """
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_cell(self, row: int, col: int) -> TerrianBlock:
        """
        Given the row and the column get the TerrianBlock.
        """
        if not self.in_bounds(row, col):
            raise IndexError(f"Cell {row}, {col} is not inside the map bounds.")

        return self._grid[row][col]

    def set_terrian(self, row: int, col: int, terrian: Terrian) -> None:
        """Change the terrian type"""
        self.get_cell(row, col).terrian = terrian

    def get_neighbors(self, row: int, col: int) -> list[TerrianBlock]:
        """
        Get the list of neighbors around a particular block.
        """

        if not self.in_bounds(row, col):
            raise IndexError(f"Cell {row}, {col} is not inside the map bounds.")

        directions = (
            (-1, 0), # North
            (1, 0), # South
            (0, -1), # West
            (0, 1) # East
        )

        neighbors: list[TerrianBlock] = []

        for dr, dc in directions:
            nr = row + dr
            nc = row + dc

            if self.in_bounds(nr, nc):
                neighbors.append(self.get_cell(nr, nc))

        return neighbors
        
    