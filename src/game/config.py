import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.game.entities.enemy import EnemyConfig
from src.game.systems.wave_director import WaveConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "game.toml"


@dataclass(frozen=True, slots=True)
class TowerConfig:
    name: str
    cost: int
    range_radius: float
    damage: float
    fire_rate: float
    projectile_speed: float
    size_scale: float = 0.5

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tower name cannot be empty.")
        if isinstance(self.cost, bool) or not isinstance(self.cost, int) or self.cost <= 0:
            raise ValueError("Tower cost must be a positive integer.")
        if any(not math.isfinite(value) or value <= 0 for value in (
            self.range_radius, self.damage, self.fire_rate,
            self.projectile_speed, self.size_scale,
        )):
            raise ValueError("Tower combat values must be positive.")


@dataclass(frozen=True, slots=True)
class DifficultyConfig:
    max_waves: int
    base_health: float
    starting_resources: int
    waves: WaveConfig
    enemy_health_growth: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.max_waves, bool) or not isinstance(self.max_waves, int) or self.max_waves < 1:
            raise ValueError("Maximum waves must be a positive integer.")
        if not math.isfinite(self.base_health) or self.base_health <= 0:
            raise ValueError("Invalid base health or starting resources.")
        if (isinstance(self.starting_resources, bool)
                or not isinstance(self.starting_resources, int)
                or self.starting_resources < 0):
            raise ValueError("Starting resources must be a non-negative integer.")
        if not math.isfinite(self.enemy_health_growth) or self.enemy_health_growth < 0:
            raise ValueError("Enemy health growth must be finite and non-negative.")


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
    enemies: dict[str, EnemyConfig]
    towers: dict[str, TowerConfig]
    difficulties: dict[str, DifficultyConfig]


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
        enemies={name: EnemyConfig(**values) for name, values in data["enemies"].items()},
        towers={name: TowerConfig(**values) for name, values in data["towers"].items()},
        difficulties={
            name: DifficultyConfig(
                max_waves=values["max_waves"],
                base_health=values["base_health"],
                starting_resources=values["starting_resources"],
                waves=WaveConfig(**values["waves"]),
                enemy_health_growth=values.get("enemy_health_growth", 0.0),
            )
            for name, values in data["difficulty"].items()
        },
    )
