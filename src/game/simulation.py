from __future__ import annotations

import math
from dataclasses import asdict, replace

import pygame

from src.game.config import GameConfig, load_config
from src.game.entities.defense_tower import DefenseTower
from src.game.entities.enemy import Enemy
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
        self.state = GameState(self.balance.base_health, self.balance.starting_resources)
        self.enemies: list[Enemy] = []
        self.towers: list[DefenseTower] = []
        self.reset(seed=seed)

    def reset(self, *, seed=None):
        self.state.reset()
        self.enemies.clear()
        self.towers.clear()
        for slot in self.defense_slots.slots:
            slot.occupied = False
        self.director = WaveDirector(self.enemy_types, self.balance.waves, seed=seed)
        self.waves_completed = 0
        self.ticks = 0
        self._accumulator = 0.0

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
                             self.tower_sprites[tower_type], **values)
        tower.tower_type = tower_type
        if not self.state.resources.spend(tower_config.cost):
            return False
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
        for enemy in self.enemies:
            enemy.update(self.timestep)
        for enemy in self.enemies:
            if enemy.alive and enemy.reached_end:
                self.state.register_leak(enemy.config.leak_damage)
                enemy.alive = False
        if not self.state.is_terminal:
            for tower in self.towers:
                tower.update(self.timestep, self.enemies)
        for enemy in self.enemies:
            if not enemy.alive and not enemy.reached_end:
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
