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
    map_path: Path
    pathway_path: Path
    enemy_assets: dict[str, Path]
    defense_slot_path: Path
    defense_assets: dict[str, Path]


def load_config(path: Path = CONFIG_PATH) -> GameConfig:
    config_path = path.resolve()

    with config_path.open("rb") as file:
        data = tomllib.load(file)

    window = data["window"]
    grid = data["grid"]
    map_assets = data["assets"]["map"]
    enemy_assets = data["assets"]["enemies"]
    defense_assets = data["assets"]["defense"]

    return GameConfig(
        window_title=window["title"],
        window_width=window["width"],
        window_height=window["height"],
        fps=window["fps"],
        rows=grid["rows"],
        cols=grid["cols"],
        tile_size=grid["tile_size"],
        map_path=(PROJECT_ROOT / map_assets["path"]).resolve(),
        pathway_path=(PROJECT_ROOT / map_assets["pathway"]).resolve(),
        enemy_assets={
            name: (PROJECT_ROOT / asset_path).resolve()
            for name, asset_path in enemy_assets.items()
        },
        defense_slot_path=(PROJECT_ROOT / defense_assets["slot"]).resolve(),
        defense_assets={
            name: (PROJECT_ROOT / asset_path).resolve()
            for name, asset_path in defense_assets.items()
            if name != "slot"
        },
    )
