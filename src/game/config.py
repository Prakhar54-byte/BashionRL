
import tomllib
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "game.toml"


@dataclass(frozen=True, slots=True)
class GameConfig:
    window_title: str
    window_width: int
    window_height: int
    fps: int
    rows: int
    cols: int
    tile_size: int
    grass_path: Path
    road_path: Path


def load_config(path: Path = CONFIG_PATH) -> GameConfig:
    with path.open("rb") as file:
        data = tomllib.load(file)

    window = data["window"]
    grid = data["grid"]
    assets = data["assets"]

    return GameConfig(
        window_title=window["title"],
        window_width=window["width"],
        window_height=window["height"],
        fps=window["fps"],
        rows=grid["rows"],
        cols=grid["cols"],
        tile_size=grid["tile_size"],
        grass_path=PROJECT_ROOT / assets["grass"],
        road_path=PROJECT_ROOT / assets["road"],
    )
