from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from typing import Any

import pygame

from src.game.config import GameConfig, load_config
from src.game.entities.enemies import Enemy, Soldier, Tank
from src.game.entities.towers import DefenseTower, KingTower, ResourceTower
from src.game.rendering.asset_loader import AssetLoader
from src.game.state.game_state import GameState
from src.game.systems.wave_director import WaveDirector
from src.game.terrain.defense_slots import DefenseSlots
from src.game.terrain.pathway import Pathway


class Simulation:
    """Shared fixed-step simulation for manual gameplay and RL episodes."""

    timestep = 1.0 / 60.0

    def __init__(
        self,
        difficulty: str = "baseline",
        *,
        config: GameConfig | None = None,
        render_assets: bool = False,
        seed: int | None = None,
    ) -> None:
        self.config = config or load_config()

        if difficulty not in self.config.difficulties:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        self.difficulty = difficulty
        self.balance = self.config.difficulties[difficulty]

        if not self.config.enemies:
            raise ValueError("At least one enemy type is required.")

        if not self.config.towers:
            raise ValueError("At least one defense tower type is required.")

        if set(self.config.enemies) != set(self.config.enemy_assets):
            raise ValueError("Enemy configurations must match enemy asset definitions.")

        if set(self.config.towers) != set(self.config.defense_assets):
            raise ValueError(
                "Defense tower configurations must match tower asset definitions."
            )

        self.enemy_types = tuple(self.config.enemies)
        self.tower_types = tuple(self.config.towers)

        # Soldiers and tanks use independent Tiled route families.
        self.soldier_routes = self._normalize_routes(
            Pathway.load_routes(
                self.config.pathway_path,
                route_family="soldier",
            ),
            "soldier",
        )

        self.tank_routes = self._normalize_routes(
            Pathway.load_routes(
                self.config.pathway_path,
                route_family="tank",
            ),
            "tank",
        )

        if not self.soldier_routes:
            raise ValueError("The map must contain at least one soldier route.")

        if not self.tank_routes:
            raise ValueError("The map must contain at least one tank route.")

        # Keep a shared pathway reference for map dimensions.
        self.pathway = next(iter(self.soldier_routes.values()))
        self.map_width = self.pathway.map_width
        self.map_height = self.pathway.map_height

        map_path = self.config.pathway_path

        self.defense_slots = DefenseSlots(
            map_path,
            self.config.defense_slot_path,
            layer_name="defense_slots",
            load_image=render_assets,
        )

        self.resource_slots = DefenseSlots(
            map_path,
            self.config.resource_slot_path,
            layer_name="resource_slots",
            load_image=render_assets,
        )

        self.king_tower_slots = DefenseSlots(
            map_path,
            layer_name="king_tower_slots",
            load_image=False,
        )

        if not self.king_tower_slots.slots:
            raise ValueError("The map must contain King Tower slots.")

        self._load_assets(render_assets=render_assets)

        self.state = GameState(
            max_base_health=self.balance.base_health,
            starting_resources=self.balance.starting_resources,
            king_tower_count=len(self.king_tower_slots.slots),
        )

        self.enemies: list[Enemy] = []
        self.towers: list[DefenseTower] = []
        self.resource_towers: list[ResourceTower] = []
        self.king_towers: list[KingTower] = []

        self._king_towers_by_id: dict[int | str, KingTower] = {}
        self._rng = random.Random(seed)

        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Route handling

    @staticmethod
    def _normalize_routes(
        routes: Any,
        family: str,
    ) -> dict[str, Pathway]:
        """Normalize route-loader output into named pathways."""
        if isinstance(routes, Mapping):
            normalized = {str(name): pathway for name, pathway in routes.items()}
        elif isinstance(routes, Sequence) and not isinstance(routes, (str, bytes)):
            normalized = {}

            for index, pathway in enumerate(routes, start=1):
                name = getattr(pathway, "name", f"{family}_{index}")
                normalized[str(name)] = pathway
        else:
            raise TypeError(
                f"Pathway.load_routes() must return a route mapping "
                f"or sequence for the '{family}' family."
            )

        for name, pathway in normalized.items():
            if not isinstance(pathway, Pathway):
                raise TypeError(f"Route '{name}' is not a Pathway instance.")

            if len(pathway.points) < 2:
                raise ValueError(f"Route '{name}' must contain at least two points.")

        return normalized

    # ------------------------------------------------------------------
    # Asset loading

    def _load_assets(self, *, render_assets: bool) -> None:
        """Load enemy, defense, resource tower, and King Tower assets."""
        loader = AssetLoader()
        placeholder = pygame.Surface((1, 1), pygame.SRCALPHA)

        self.enemy_sprites: dict[str, pygame.Surface] = {}
        self.enemy_base_sprites: dict[str, pygame.Surface] = {}
        self.enemy_gun_sprites: dict[str, pygame.Surface] = {}

        for enemy_type, assets in self.config.enemy_assets.items():
            if assets.is_tank:
                if assets.base_path is None or assets.gun_path is None:
                    raise ValueError(
                        f"Tank '{enemy_type}' requires both base and gun assets."
                    )

                if render_assets:
                    base_sprite = loader.load_texture(assets.base_path)
                    gun_sprite = loader.load_texture(assets.gun_path)
                else:
                    base_sprite = placeholder
                    gun_sprite = placeholder

                self.enemy_sprites[enemy_type] = placeholder
                self.enemy_base_sprites[enemy_type] = base_sprite
                self.enemy_gun_sprites[enemy_type] = gun_sprite
            else:
                if assets.sprite_path is None:
                    raise ValueError(f"Enemy '{enemy_type}' requires a sprite asset.")

                self.enemy_sprites[enemy_type] = (
                    loader.load_texture(assets.sprite_path)
                    if render_assets
                    else placeholder
                )

        self.tower_sprites = {
            name: (loader.load_texture(path) if render_assets else placeholder)
            for name, path in self.config.defense_assets.items()
        }

        self.base_sprite = (
            loader.load_texture(self.config.defense_base_path)
            if render_assets and self.config.defense_base_path is not None
            else None
        )

        self.king_tower_sprite = (
            loader.load_texture(self.config.king_tower_sprite_path)
            if render_assets
            else None
        )

        self.resource_tower_sprite = (
            loader.load_texture(self.config.resource_tower_sprite_path)
            if (render_assets and self.config.resource_tower_sprite_path is not None)
            else None
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the episode, structures, economy, and wave scheduling."""
        self.state.reset()

        # Clear entities in place to preserve references held by main.py.
        self.enemies.clear()
        self.towers.clear()
        self.resource_towers.clear()
        self.king_towers.clear()
        self._king_towers_by_id.clear()

        if seed is not None:
            self._rng = random.Random(seed)

        # Reset buildable slots.
        for slot in self.defense_slots.slots:
            slot.occupied = False

        for slot in self.resource_slots.slots:
            slot.occupied = False

        # Recreate King Towers using the configured maximum HP.
        for slot in self.king_tower_slots.slots:
            slot.occupied = True

            king_tower = KingTower(
                structure_id=slot.slot_id,
                position=slot.center,
                sprite=self.king_tower_sprite,
                max_health=self.config.king_tower_max_health,
            )

            self.king_towers.append(king_tower)
            self._king_towers_by_id[slot.slot_id] = king_tower

        # Reset automatic wave scheduling.
        self.director = WaveDirector(
            self.enemy_types,
            self.balance.waves,
            seed=seed,
        )

        # Reset passive income and episode timers.
        self._passive_income_elapsed = 0.0
        self.waves_completed = 0
        self.ticks = 0
        self._accumulator = 0.0

    # ------------------------------------------------------------------
    # Building placement

    def place_tower(self, slot_index: int, tower_type: str) -> bool:
        """Place a configured defense tower if the slot is available and affordable."""
        if tower_type not in self.config.towers:
            raise ValueError(f"Unknown defense tower: {tower_type}")

        if not 0 <= slot_index < len(self.defense_slots.slots):
            raise ValueError("Defense slot index out of range.")

        if self.state.is_terminal:
            return False

        slot = self.defense_slots.slots[slot_index]
        tower_config = self.config.towers[tower_type]

        if slot.occupied:
            return False

        if not self.state.resources.can_afford(tower_config.cost):
            return False

        # Construct the tower using its configured combat and health values.
        tower = DefenseTower(
            slot_id=slot.slot_id,
            position=slot.center,
            sprite=self.tower_sprites[tower_type],
            base_sprite=self.base_sprite,
            range_radius=tower_config.range_radius,
            damage=tower_config.damage,
            fire_rate=tower_config.fire_rate,
            projectile_speed=tower_config.projectile_speed,
            max_health=tower_config.max_health,
            size_scale=tower_config.size_scale,
            base_size_scale=tower_config.base_size_scale,
            shots_per_burst=tower_config.shots_per_burst,
            burst_interval=tower_config.burst_interval,
        )

        tower.tower_type = tower_type

        # Spend resources only after successfully constructing the tower.
        if not self.state.resources.spend(tower_config.cost):
            return False

        self.towers.append(tower)
        slot.occupied = True

        return True

    def place_resource_tower(self, slot_index: int) -> bool:
        """Build a resource tower on an available resource slot."""
        if not 0 <= slot_index < len(self.resource_slots.slots):
            raise ValueError("Resource slot index out of range.")

        if self.state.is_terminal:
            return False

        slot = self.resource_slots.slots[slot_index]
        config = self.config.resource_tower

        if slot.occupied:
            return False

        if not self.state.resources.can_afford(config.cost):
            return False

        values = asdict(config)
        values.pop("name")
        values.pop("cost")

        tower = ResourceTower(
            tower_id=slot.slot_id,
            position=slot.center,
            sprite=self.resource_tower_sprite,
            **values,
        )

        if not self.state.resources.spend(config.cost):
            return False

        self.resource_towers.append(tower)
        slot.occupied = True

        return True

    def _update_passive_income(self, delta_time: float) -> None:
        """Grant passive resources at the configured time interval."""
        self._passive_income_elapsed += delta_time

        interval = self.config.passive_income_interval
        amount = self.config.passive_income_amount

        # Preserve elapsed time and support multiple payouts if necessary.
        while self._passive_income_elapsed + 1e-12 >= interval:
            self.state.resources.earn(amount)
            self._passive_income_elapsed -= interval

    # ------------------------------------------------------------------
    # King Tower repair and state
    # ------------------------------------------------------------------
    def start_king_tower_repair(
        self,
        tower_id: int | str,
    ) -> bool:
        """Start a King Tower repair if the episode and economy allow it."""
        if self.state.is_terminal:
            return False

        tower = self._king_towers_by_id.get(tower_id)

        if tower is None:
            raise ValueError(f"Unknown King Tower ID: {tower_id}")

        if tower.health >= tower.max_health:
            return False

        if tower.repair_time_remaining > 0:
            return False

        if not self.state.resources.can_afford(tower.repair_cost):
            return False

        started = tower.start_repair()

        # Some implementations return None on success.
        if started is False:
            return False

        if not self.state.resources.spend(tower.repair_cost):
            tower.cancel_repair()
            return False

        return True

    def _sync_king_tower_state(self) -> None:
        """Synchronize destroyed/restored King Tower state with the episode."""
        destroyed_ids = self.state.destroyed_king_tower_ids

        for tower_id, tower in self._king_towers_by_id.items():
            if tower.destroyed:
                if tower_id not in destroyed_ids:
                    self.state.register_king_tower_destroyed(tower_id)
            elif tower_id in destroyed_ids:
                self.state.register_king_tower_restored(tower_id)

    # ------------------------------------------------------------------
    # Enemy spawning

    def _spawn_enemy(self, enemy_type: str) -> None:
        """Spawn an enemy on a randomly selected route from its route family."""
        if self.state.is_terminal:
            return

        if enemy_type not in self.config.enemies:
            raise ValueError(f"Unknown enemy type: {enemy_type}")

        config = self.config.enemies[enemy_type]
        asset_config = self.config.enemy_assets[enemy_type]

        health_multiplier = 1.0 + self.balance.enemy_health_growth * max(
            0, self.director.wave_number - 1
        )

        scaled_config = replace(
            config,
            max_health=config.max_health * health_multiplier,
        )

        # Each enemy uses the route family assigned to its asset definition.
        routes = self.tank_routes if asset_config.is_tank else self.soldier_routes

        route_name = self._rng.choice(tuple(routes))
        pathway = routes[route_name]

        if asset_config.is_tank:
            base_sprite = self.enemy_base_sprites.get(enemy_type)
            gun_sprite = self.enemy_gun_sprites.get(enemy_type)

            if base_sprite is None or gun_sprite is None:
                raise ValueError(f"Tank '{enemy_type}' is missing base or gun sprites.")

            enemy: Enemy = Tank(
                pathway=pathway,
                sprite=self.enemy_sprites[enemy_type],
                config=scaled_config,
                base_sprite=base_sprite,
                gun_sprite=gun_sprite,
            )
        else:
            enemy = Soldier(
                pathway=pathway,
                sprite=self.enemy_sprites[enemy_type],
                config=scaled_config,
            )

        enemy.enemy_type = enemy_type
        enemy.route_name = route_name

        self.enemies.append(enemy)
        self.state.register_spawn()

    # ------------------------------------------------------------------
    # Combat helpers

    def _update_enemy_movement(self) -> bool:
        """Move enemies and handle soldier suicide attacks.

        Soldiers die when they cross a King Tower slot, regardless of
        whether the corresponding King Tower is still alive.

        Returns False if the episode becomes terminal during this phase.
        """
        for enemy in self.enemies:
            if not enemy.alive:
                continue

            previous_position = enemy.position.copy()
            enemy.update(self.timestep)

            # Tanks do not perform suicide attacks against King Towers.
            if isinstance(enemy, Tank):
                continue

            # An enemy that has already exited its route cannot attack.
            if not enemy.alive:
                continue

            start = (
                round(previous_position.x),
                round(previous_position.y),
            )
            end = (
                round(enemy.position.x),
                round(enemy.position.y),
            )

            for slot in self.king_tower_slots.slots:
                slot_rect = slot.rect.inflate(2, 2)

                crossed_slot = bool(slot_rect.clipline(start, end))

                if not crossed_slot:
                    crossed_slot = slot_rect.collidepoint(end)

                if not crossed_slot:
                    continue

                king_tower = self._king_towers_by_id.get(slot.slot_id)

                # Damage the tower only if it exists and is still alive.
                if king_tower is not None and king_tower.alive:
                    king_tower.take_damage(enemy.config.attack_damage)

                # The soldier dies whenever it crosses the King Tower slot,
                # even if the tower is already destroyed or missing.
                enemy.suicide_attacked = True
                enemy.alive = False

                break

            self._sync_king_tower_state()

            if self.state.is_terminal:
                return False

        return True

    def _update_tank_combat(self) -> None:
        """Update tank attacks against all eligible tower types."""
        active_defense_towers = tuple(
            tower for tower in self.towers if tower.alive and not tower.destroyed
        )

        active_resource_towers = tuple(
            tower
            for tower in self.resource_towers
            if tower.alive and not tower.destroyed
        )

        active_king_towers = tuple(
            tower for tower in self.king_towers if tower.alive and not tower.destroyed
        )

        for enemy in self.enemies:
            if isinstance(enemy, Tank) and enemy.alive:
                enemy.update_combat(
                    delta_time=self.timestep,
                    defense_towers=active_defense_towers,
                    king_towers=active_king_towers,
                    resource_towers=active_resource_towers,
                )

    def _update_defense_towers(self) -> None:
        """Update player defense towers and their projectiles."""
        for tower in self.towers:
            if tower.alive:
                tower.update(self.timestep, self.enemies)

        # Free the slots of destroyed defense towers and remove them.
        destroyed_slot_ids = {tower.slot_id for tower in self.towers if tower.destroyed}

        if destroyed_slot_ids:
            for slot in self.defense_slots.slots:
                if slot.slot_id in destroyed_slot_ids:
                    slot.occupied = False

            self.towers[:] = [tower for tower in self.towers if not tower.destroyed]

    def _update_resource_towers(self) -> None:
        """Generate income and remove destroyed Resource Towers."""
        for tower in self.resource_towers:
            if tower.alive:
                earned = tower.update(self.timestep)

                if earned > 0:
                    self.state.resources.earn(earned)

        destroyed_slot_ids = {
            tower.tower_id for tower in self.resource_towers if tower.destroyed
        }

        if destroyed_slot_ids:
            for slot in self.resource_slots.slots:
                if slot.slot_id in destroyed_slot_ids:
                    slot.occupied = False

            self.resource_towers[:] = [
                tower for tower in self.resource_towers if not tower.destroyed
            ]

    def _update_king_towers(self) -> None:
        """Advance timed King Tower repairs."""
        for tower in self.king_towers:
            tower.update(self.timestep)

        self._sync_king_tower_state()

    def _award_kills_and_remove_dead_enemies(self) -> None:
        """Reward actual kills, excluding suicide attacks and route exits."""
        for enemy in self.enemies:
            if (
                not enemy.alive
                and not enemy.suicide_attacked
                and not enemy.exited_route
            ):
                self.state.register_kill(enemy.config.kill_reward)

        # Route exits alone do not inflict global base damage.
        self.enemies[:] = [enemy for enemy in self.enemies if enemy.alive]

    # ------------------------------------------------------------------
    # Main simulation loop

    def tick(self) -> None:
        """Advance one fixed physics timestep."""
        if self.state.is_terminal:
            return

        self.ticks += 1

        # 1. Grant passive income during setup, countdowns, and combat.
        self._update_passive_income(self.timestep)

        # 2. Advance the automatic wave scheduler.
        # WaveDirector handles the initial countdown and inter-wave delays.
        self.director.update(
            delta_time=self.timestep,
            active_enemy_count=len(self.enemies),
            spawn_enemy=self._spawn_enemy,
        )

        # 3. Move enemies and resolve soldier suicide attacks.
        if not self._update_enemy_movement():
            return

        # 4. Tanks attack nearby defense towers and King Towers.
        # Their movement remains restricted to their assigned routes.
        self._update_tank_combat()

        # 5. Player defense towers attack enemies and update projectiles.
        self._update_defense_towers()

        # 6. Living Resource Towers generate their configured income.
        self._update_resource_towers()

        # 7. Update King Tower repairs and synchronize destruction state.
        self._update_king_towers()

        if self.state.is_terminal:
            return

        # 8. Award rewards for actual kills and remove dead or exited enemies.
        self._award_kills_and_remove_dead_enemies()

        # 9. Declare victory after the final wave has been completely cleared.
        if (
            not self.state.is_terminal
            and self.director.wave_number >= self.balance.max_waves
            and self.director.enemies_remaining_to_spawn == 0
            and not self.enemies
        ):
            self.waves_completed = self.director.wave_number
            self.state.mark_won()

    def advance(self, seconds: float) -> None:
        """Advance using a fixed timestep independent of rendering FPS."""
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("Elapsed time must be finite and non-negative.")

        self._accumulator += seconds

        while self._accumulator + 1e-12 >= self.timestep:
            self.tick()
            self._accumulator -= self.timestep
