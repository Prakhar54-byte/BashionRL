
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.game.entities.enemy import EnemyConfig
from src.game.systems.wave_director import WaveConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "game.toml"


@dataclass(frozen=True, slots=True)
class EnemyAssetConfig:
    """Asset configuration for a single-sprite enemy or a tank."""

    sprite_path: Path | None = None
    base_path: Path | None = None
    gun_path: Path | None = None

    def __post_init__(self) -> None:
        has_sprite = self.sprite_path is not None
        has_base = self.base_path is not None
        has_gun = self.gun_path is not None

        # Ordinary soldiers use one complete sprite.
        if has_sprite and not has_base and not has_gun:
            return

        # Tanks consist of a chassis and a separately rotating gun.
        if not has_sprite and has_base and has_gun:
            return

        raise ValueError(
            "Enemy assets must define either 'sprite' or both "
            "'base' and 'gun'."
        )

    @property
    def is_tank(self) -> bool:
        """Whether the enemy has separate base and gun sprites."""
        return self.base_path is not None and self.gun_path is not None


@dataclass(frozen=True, slots=True)
class TowerConfig:
    """Combat, economy, rendering, and firing configuration for a defense tower."""

    name: str
    cost: int
    range_radius: float
    damage: float
    fire_rate: float
    projectile_speed: float
    size_scale: float = 0.5
    base_size_scale: float = 0.5
    shots_per_burst: int = 1
    burst_interval: float = 0.12

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tower name cannot be empty.")

        if (
            isinstance(self.cost, bool)
            or not isinstance(self.cost, int)
            or self.cost <= 0
        ):
            raise ValueError("Tower cost must be a positive integer.")

        positive_values = {
            "range_radius": self.range_radius,
            "damage": self.damage,
            "fire_rate": self.fire_rate,
            "projectile_speed": self.projectile_speed,
            "size_scale": self.size_scale,
            "base_size_scale": self.base_size_scale,
        }

        for name, value in positive_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"Tower {name} must be finite and positive."
                )

        if (
            isinstance(self.shots_per_burst, bool)
            or not isinstance(self.shots_per_burst, int)
            or self.shots_per_burst < 1
        ):
            raise ValueError(
                "Shots per burst must be a positive integer."
            )

        if (
            isinstance(self.burst_interval, bool)
            or not isinstance(self.burst_interval, (int, float))
            or not math.isfinite(self.burst_interval)
            or self.burst_interval < 0
        ):
            raise ValueError(
                "Burst interval must be finite and non-negative."
            )


@dataclass(frozen=True, slots=True)
class ResourceTowerConfig:
    """Economy, durability, and rendering configuration for a resource tower."""

    name: str = "Resource Tower"
    cost: int = 125
    max_health: float = 250.0
    production_interval: float = 15.0
    production_amount: int = 10
    size_scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Resource tower name cannot be empty.")

        if (
            isinstance(self.cost, bool)
            or not isinstance(self.cost, int)
            or self.cost <= 0
        ):
            raise ValueError(
                "Resource tower cost must be a positive integer."
            )

        if (
            isinstance(self.max_health, bool)
            or not isinstance(self.max_health, (int, float))
            or not math.isfinite(self.max_health)
            or self.max_health <= 0
        ):
            raise ValueError(
                "Resource tower maximum health must be finite and positive."
            )

        if (
            isinstance(self.production_interval, bool)
            or not isinstance(self.production_interval, (int, float))
            or not math.isfinite(self.production_interval)
            or self.production_interval <= 0
        ):
            raise ValueError(
                "Resource tower production interval must be finite "
                "and positive."
            )

        if (
            isinstance(self.production_amount, bool)
            or not isinstance(self.production_amount, int)
            or self.production_amount <= 0
        ):
            raise ValueError(
                "Resource tower production amount must be a positive integer."
            )

        if (
            isinstance(self.size_scale, bool)
            or not isinstance(self.size_scale, (int, float))
            or not math.isfinite(self.size_scale)
            or self.size_scale <= 0
        ):
            raise ValueError(
                "Resource tower size scale must be finite and positive."
            )


@dataclass(frozen=True, slots=True)
class DifficultyConfig:
    """Episode limits, starting resources, and wave-generation settings."""

    max_waves: int
    base_health: float
    starting_resources: int
    waves: WaveConfig
    enemy_health_growth: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_waves, bool)
            or not isinstance(self.max_waves, int)
            or self.max_waves < 1
        ):
            raise ValueError(
                "Maximum waves must be a positive integer."
            )

        if (
            isinstance(self.base_health, bool)
            or not isinstance(self.base_health, (int, float))
            or not math.isfinite(self.base_health)
            or self.base_health <= 0
        ):
            raise ValueError(
                "Base health must be finite and positive."
            )

        if (
            isinstance(self.starting_resources, bool)
            or not isinstance(self.starting_resources, int)
            or self.starting_resources < 0
        ):
            raise ValueError(
                "Starting resources must be a non-negative integer."
            )

        if (
            isinstance(self.enemy_health_growth, bool)
            or not isinstance(self.enemy_health_growth, (int, float))
            or not math.isfinite(self.enemy_health_growth)
            or self.enemy_health_growth < 0
        ):
            raise ValueError(
                "Enemy health growth must be finite and non-negative."
            )


@dataclass(frozen=True, slots=True)
class GameConfig:
    """All configuration values required by the game and simulation."""

    window_title: str
    window_width: int
    window_height: int
    fps: int

    rows: int
    cols: int
    tile_size: int

    map_path: Path
    pathway_path: Path

    enemy_assets: dict[str, EnemyAssetConfig]

    defense_slot_path: Path
    defense_base_path: Path | None
    defense_assets: dict[str, Path]

    # Shared sprite used by the three King Towers.
    king_tower_sprite_path: Path

    # Optional sprite for the resource tower. The tower can use its
    # built-in placeholder drawing until an asset is configured.
    resource_tower_sprite_path: Path | None

    enemies: dict[str, EnemyConfig]
    towers: dict[str, TowerConfig]
    resource_tower: ResourceTowerConfig
    difficulties: dict[str, DifficultyConfig]


def _resolve_asset_path(asset_path: str) -> Path:
    """Resolve a project-relative asset path to an absolute path."""
    if not isinstance(asset_path, str) or not asset_path.strip():
        raise ValueError("Asset paths must be non-empty strings.")

    return (PROJECT_ROOT / asset_path).resolve()


def _parse_enemy_assets(
    enemy_assets: dict[str, Any],
) -> dict[str, EnemyAssetConfig]:
    """Parse enemy assets in single-sprite or base-and-gun format."""
    parsed: dict[str, EnemyAssetConfig] = {}

    for enemy_name, asset_data in enemy_assets.items():
        # Backward compatibility with the old TOML format:
        # enemy1 = "Assets/Main/Entities/Enemies/enemy1.png"
        if isinstance(asset_data, str):
            parsed[enemy_name] = EnemyAssetConfig(
                sprite_path=_resolve_asset_path(asset_data),
            )
            continue

        if not isinstance(asset_data, dict):
            raise ValueError(
                f"Assets for enemy '{enemy_name}' must be a path "
                "or a table containing sprite or base/gun paths."
            )

        allowed_keys = {"sprite", "base", "gun"}
        unexpected_keys = set(asset_data) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unknown asset keys for enemy '{enemy_name}': "
                f"{sorted(unexpected_keys)}"
            )

        sprite = asset_data.get("sprite")
        base = asset_data.get("base")
        gun = asset_data.get("gun")

        parsed[enemy_name] = EnemyAssetConfig(
            sprite_path=(
                _resolve_asset_path(sprite)
                if sprite is not None
                else None
            ),
            base_path=(
                _resolve_asset_path(base)
                if base is not None
                else None
            ),
            gun_path=(
                _resolve_asset_path(gun)
                if gun is not None
                else None
            ),
        )

    return parsed


def _parse_resource_tower_config(
    values: dict[str, Any],
) -> ResourceTowerConfig:
    """Parse resource tower settings while applying defaults to omitted keys."""
    if not isinstance(values, dict):
        raise ValueError(
            "The [resource_tower] configuration must be a TOML table."
        )

    allowed_keys = {
        "name",
        "cost",
        "max_health",
        "production_interval",
        "production_amount",
        "size_scale",
    }
    unexpected_keys = set(values) - allowed_keys

    if unexpected_keys:
        raise ValueError(
            "Unknown resource tower configuration keys: "
            f"{sorted(unexpected_keys)}"
        )

    return ResourceTowerConfig(**values)


def load_config(path: Path = CONFIG_PATH) -> GameConfig:
    """Load and validate game configuration from a TOML file."""
    config_path = Path(path).resolve()

    with config_path.open("rb") as file:
        data = tomllib.load(file)

    # Top-level configuration sections.
    window = data["window"]
    grid = data["grid"]
    assets = data["assets"]

    map_assets = assets["map"]
    enemy_assets = assets["enemies"]
    defense_assets = assets["defense"]

    # Parse soldier and tank assets.
    parsed_enemy_assets = _parse_enemy_assets(enemy_assets)

    # King Tower sprite: [assets.king] king = "..."
    king_assets = assets.get("king")

    if not isinstance(king_assets, dict):
        raise ValueError(
            "Missing [assets.king] configuration in game.toml."
        )

    if "king" not in king_assets:
        raise ValueError(
            "Missing 'king' asset in [assets.king] in game.toml."
        )

    king_tower_sprite_path = _resolve_asset_path(
        king_assets["king"]
    )

    # An optional [assets.resource_tower] sprite can be configured later.
    # When omitted, ResourceTower uses its built-in placeholder artwork.
    resource_assets = assets.get("resource_tower", {})

    if not isinstance(resource_assets, dict):
        raise ValueError(
            "[assets.resource_tower] must be a TOML table."
        )

    unexpected_resource_asset_keys = (
        set(resource_assets) - {"sprite"}
    )

    if unexpected_resource_asset_keys:
        raise ValueError(
            "Unknown resource tower asset keys: "
            f"{sorted(unexpected_resource_asset_keys)}"
        )

    resource_sprite = resource_assets.get("sprite")

    resource_tower_sprite_path = (
        _resolve_asset_path(resource_sprite)
        if resource_sprite is not None
        else None
    )

    # The cannon base is a shared visual asset, not a placeable tower.
    base_asset = defense_assets.get("cannon_base")

    defense_base_path = (
        _resolve_asset_path(base_asset)
        if base_asset is not None
        else None
    )

    # Keep only placeable defense tower sprites in this dictionary.
    tower_assets = {
        name: _resolve_asset_path(asset_path)
        for name, asset_path in defense_assets.items()
        if name not in {"slot", "cannon_base"}
    }

    # Parse gameplay configurations.
    enemy_configs = {
        name: EnemyConfig(**values)
        for name, values in data["enemies"].items()
    }

    tower_configs = {
        name: TowerConfig(**values)
        for name, values in data["towers"].items()
    }

    resource_tower_config = _parse_resource_tower_config(
        data.get("resource_tower", {})
    )

    difficulty_configs = {
        name: DifficultyConfig(
            max_waves=values["max_waves"],
            base_health=values["base_health"],
            starting_resources=values["starting_resources"],
            waves=WaveConfig(**values["waves"]),
            enemy_health_growth=values.get(
                "enemy_health_growth",
                0.0,
            ),
        )
        for name, values in data["difficulty"].items()
    }

    # Ensure each enemy configuration has a matching asset definition.
    if set(enemy_configs) != set(parsed_enemy_assets):
        raise ValueError(
            "Enemy configurations must match enemy asset definitions. "
            f"Missing assets: "
            f"{sorted(set(enemy_configs) - set(parsed_enemy_assets))}; "
            f"unused assets: "
            f"{sorted(set(parsed_enemy_assets) - set(enemy_configs))}."
        )

    # Ensure each tower configuration has a matching placeable sprite.
    if set(tower_configs) != set(tower_assets):
        raise ValueError(
            "Tower configurations must match tower assets. "
            f"Missing assets: "
            f"{sorted(set(tower_configs) - set(tower_assets))}; "
            f"unused assets: "
            f"{sorted(set(tower_assets) - set(tower_configs))}."
        )

    # Validate required asset entries.
    if "slot" not in defense_assets:
        raise ValueError(
            "Missing 'slot' asset in [assets.defense] in game.toml."
        )

    return GameConfig(
        window_title=window["title"],
        window_width=window["width"],
        window_height=window["height"],
        fps=window["fps"],

        rows=grid["rows"],
        cols=grid["cols"],
        tile_size=grid["tile_size"],

        map_path=_resolve_asset_path(map_assets["path"]),
        pathway_path=_resolve_asset_path(map_assets["pathway"]),

        enemy_assets=parsed_enemy_assets,

        defense_slot_path=_resolve_asset_path(
            defense_assets["slot"]
        ),
        defense_base_path=defense_base_path,
        defense_assets=tower_assets,

        king_tower_sprite_path=king_tower_sprite_path,
        resource_tower_sprite_path=resource_tower_sprite_path,

        enemies=enemy_configs,
        towers=tower_configs,
        resource_tower=resource_tower_config,
        difficulties=difficulty_configs,
    )
