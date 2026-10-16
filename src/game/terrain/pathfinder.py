from pathlib import Path

Position = tuple[int, int]


class MapPathfinder:
    """Load a tiled JSON map and find a route across its road layer."""

    def __init__(self, map_path: str | Path) -> None:
        self.map_path = Path(map_path).resolve()
