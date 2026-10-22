from __future__ import annotations

import math
from dataclasses import asdict, replace

import pygame

from src.game.config import GameConfig, load_config
from src.game.entities.defense_tower import DefenseTower
from src.game.entities.enemy import Enemy
from src.game.entities.king_tower import KingTower
from src.game.entities.resource_tower import ResourceTower
from src.game.rendering.asset_loader import AssetLoader
from src.game.state.game_state import GameState
from src.game.systems.wave_director import WaveDirector
from src.game.terrain.defense_slots import DefenseSlots
from src.game.terrain.pathway import Pathway


class Simulation:
    """The same fixed-step game rules for keyboard play and RL episodes."""

    timestep = 1.0 / 60.0

    def __init__(self, difficulty="baseline", *, config: GameConfig | None = None,
                 render_assets=False, seed=None):
        self.config = config or load_config()
        if difficulty not in self.config.difficulties:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        self.difficulty = difficulty
        self.balance = self.config.difficulties[difficulty]
        if not self.config.enemies or not self.config.towers:
            raise ValueError("At least one enemy and tower type is required.")
        if set(self.config.enemies) != set(self.config.enemy_assets):
            raise ValueError("Enemy configurations must match enemy assets.")
        if set(self.config.towers) != set(self.config.defense_assets):
            raise ValueError("Tower configurations must match defense assets.")
        self.enemy_types = tuple(self.config.enemies)
        self.tower_types = tuple(self.config.towers)
        self.pathway = Pathway(self.config.pathway_path)
        self.defense_slots = DefenseSlots(
            self.config.pathway_path, self.config.defense_slot_path,
            load_image=render_assets,
        )
        loader = AssetLoader()
        placeholder = pygame.Surface((1, 1), pygame.SRCALPHA)
        self.enemy_sprites = {
            name: loader.load_texture(path) if render_assets else placeholder
            for name, path in self.config.enemy_assets.items()
        }
        self.tower_sprites = {
            name: loader.load_texture(path) if render_assets else placeholder
            for name, path in self.config.defense_assets.items()
        }
        self.state = GameState(sum(values["max_health"] for values in self.config.king_towers),
                               self.balance.starting_resources)
        self.enemies: list[Enemy] = []
        self.towers: list[DefenseTower] = []
        self.king_towers: list[KingTower] = []
        self.resource_towers: list[ResourceTower] = []
        self.reset(seed=seed)

    def reset(self, *, seed=None):
        self.state.reset()
        self.enemies.clear()
        self.towers.clear()
        self.king_towers[:] = [KingTower(**values, on_health_changed=self._health_changed)
                               for values in self.config.king_towers]
        self.resource_towers[:] = [ResourceTower(**values, on_health_changed=self._health_changed)
                                   for values in self.config.resource_towers]
        self.structures_destroyed = 0
        self._destroyed_ids = set()
        for slot in self.defense_slots.slots:
            slot.occupied = False
        self.director = WaveDirector(self.enemy_types, self.balance.waves, seed=seed)
        self.waves_completed = 0
        self.ticks = 0
        self._accumulator = 0.0

    @property
    def structures(self):
        return [*self.towers, *self.resource_towers, *self.king_towers]

    def _health_changed(self, structure):
        if structure.alive:
            self._destroyed_ids.discard(structure.structure_id)
        elif structure.structure_id not in self._destroyed_ids:
            self._destroyed_ids.add(structure.structure_id)
            self.structures_destroyed += 1
            if structure in self.towers:
                self.towers.remove(structure)
                structure.projectiles.clear()
                for slot in self.defense_slots.slots:
                    if slot.slot_id == structure.slot_id:
                        slot.occupied = False
        self.state.base_health = sum(king.health for king in self.king_towers)
        if all(king.destroyed for king in self.king_towers):
            self.state.mark_lost()
            for king in self.king_towers:
                king.repairing = False

    def repair_king_tower(self, structure_id):
        """Validate and pay once; the entity only handles timed healing."""
        if self.state.is_terminal:
            return False
        king = next((tower for tower in self.king_towers if tower.structure_id == structure_id), None)
        if king is None or not king.can_repair:
            return False
        if king.destroyed and not any(other.alive for other in self.king_towers if other is not king):
            return False
        if not self.state.resources.spend(king.repair_cost):
            return False
        king.start_repair()
        return True

    def place_tower(self, slot_index: int, tower_type: str) -> bool:
        if tower_type not in self.config.towers:
            raise ValueError(f"Unknown tower: {tower_type}")
        if not 0 <= slot_index < len(self.defense_slots.slots):
            raise ValueError("Slot index out of range.")
        slot = self.defense_slots.slots[slot_index]
        tower_config = self.config.towers[tower_type]
        if self.state.is_terminal or slot.occupied:
            return False
        if not self.state.resources.can_afford(tower_config.cost):
            return False
        values = asdict(tower_config)
        values.pop("name")
        values.pop("cost")
        tower = DefenseTower(slot.slot_id, slot.center,
                             self.tower_sprites[tower_type], on_health_changed=self._health_changed, **values)
        tower.tower_type = tower_type
        if not self.state.resources.spend(tower_config.cost):
            return False
        self._destroyed_ids.discard(tower.structure_id)
        self.towers.append(tower)
        slot.occupied = True
        return True

    def _spawn_enemy(self, enemy_type):
        if self.state.is_terminal:
            return
        config = self.config.enemies[enemy_type]
        multiplier = 1 + self.balance.enemy_health_growth * max(0, self.director.wave_number - 1)
        enemy = Enemy(self.pathway, self.enemy_sprites[enemy_type],
                      replace(config, max_health=config.max_health * multiplier))
        enemy.enemy_type = enemy_type
        self.enemies.append(enemy)
        self.state.register_spawn()

    def tick(self):
        if self.state.is_terminal:
            return
        self.ticks += 1
        self.director.update(self.timestep, len(self.enemies), self._spawn_enemy)
        for king in self.king_towers:
            king.update(self.timestep)
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            enemy.update(self.timestep, self.structures, self.config.target_priority)
            if self.state.is_terminal:
                return
            for projectile in enemy.projectiles:
                projectile.update(self.timestep, self.structures)
                if self.state.is_terminal:
                    return
            enemy.projectiles[:] = [shot for shot in enemy.projectiles if shot.alive]
            if enemy.reached_end and not enemy.leak_registered:
                self.state.register_leak(enemy.config.leak_damage)
                enemy.leak_registered = True
        if not self.state.is_terminal:
            for tower in self.towers:
                tower.update(self.timestep, self.enemies)
            for tower in self.resource_towers:
                self.state.resources.earn(tower.update(self.timestep))
        for enemy in self.enemies:
            if not enemy.alive:
                self.state.register_kill(enemy.config.kill_reward)
        self.enemies[:] = [enemy for enemy in self.enemies if enemy.alive]
        if (not self.state.is_terminal and self.director.wave_number > 0
                and self.director.enemies_remaining_to_spawn == 0 and not self.enemies):
            self.waves_completed = self.director.wave_number
            if self.waves_completed >= self.balance.max_waves:
                self.state.mark_won()

    def advance(self, seconds: float):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Elapsed time must be finite and non-negative.")
        self._accumulator += seconds
        while self._accumulator + 1e-12 >= self.timestep:
            self.tick()
            self._accumulator -= self.timestep
