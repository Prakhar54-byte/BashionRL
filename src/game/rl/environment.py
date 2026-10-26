from __future__ import annotations

import math
from dataclasses import replace
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.game.simulation import Simulation
from src.game.state.game_state import GameStatus


def fraction(value, scale):
    """Bound a nonnegative quantity without dropping its ordering."""
    return value / (value + scale)


class BastionEnv(gym.Env):
    """Learn defense, economy, and repairs at legal decision points."""

    metadata: ClassVar[dict] = {"render_modes": []}
    decision_ticks = 30
    max_enemies = 64
    global_features = 15
    tower_features = 6
    enemy_type_features = 7
    slot_features = 12
    resource_features = 10
    king_features = 9
    enemy_features = 12
    reward_weights: ClassVar[dict] = {
        "enemy_damage": 0.05, "kills": 0.025, "base_damage": -1.0,
        "waves": 0.1, "win": 2.5, "loss": -1.0,
        "invalid_action": -0.005, "spending": -0.001,
    }
    reward_components = ("enemy_damage", "kills", "base_damage", "waves", "outcome",
                         "invalid_action", "spending")

    def __init__(self, difficulty="baseline", *, config=None, max_steps=2400, render_assets=False):
        super().__init__()
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        self.simulation = Simulation(difficulty, config=config, render_assets=render_assets)
        self.task_balance = self.simulation.balance
        self.wave_goal = self.task_balance.max_waves
        self.mixed_goals = False
        self.max_steps = max_steps
        self.steps = 0
        self._ended = True
        sim = self.simulation
        self.resource_action_start = 1 + len(sim.defense_slots.slots) * len(sim.tower_types)
        self.repair_action_start = self.resource_action_start + len(sim.resource_slots.slots)
        self.action_space = spaces.Discrete(self.repair_action_start + len(sim.king_towers))
        self.enemy_observation_start = (
            self.global_features + len(sim.tower_types) * self.tower_features
            + len(sim.enemy_types) * self.enemy_type_features
            + len(sim.defense_slots.slots) * self.slot_features
            + len(sim.resource_slots.slots) * self.resource_features
            + len(sim.king_towers) * self.king_features
        )
        size = self.enemy_observation_start + self.max_enemies * self.enemy_features
        self.observation_space = spaces.Box(0.0, 1.0, shape=(size,), dtype=np.float32)
        self.cost_scale = min(tower.cost for tower in sim.config.towers.values())
        self.health_scale = max(enemy.max_health for enemy in sim.config.enemies.values())
        self.map_diagonal = math.hypot(sim.map_width, sim.map_height)
        self.route_offsets = {}
        for route in (*sim.soldier_routes.values(), *sim.tank_routes.values()):
            segments = list(zip(route.points, route.points[1:]))
            lengths = [start.distance_to(end) for start, end in segments]
            offsets = np.concatenate(([0.0], np.cumsum(lengths)))
            if offsets[-1] <= 0:
                raise ValueError("The enemy route must have positive length.")
            self.route_offsets[route] = offsets
        self.soldier_route_distances = {
            slot.slot_id: self._route_distance(slot.center, sim.soldier_routes.values())
            for slot in [*sim.defense_slots.slots, *sim.resource_slots.slots]
        }
        self.tank_route_distances = {
            slot.slot_id: self._route_distance(slot.center, sim.tank_routes.values())
            for slot in [*sim.defense_slots.slots, *sim.resource_slots.slots]
        }
    def _route_distance(self, position, routes):
        distances = []
        segments = [(start, end) for route in routes for start, end in zip(route.points, route.points[1:])]
        for start, end in segments:
            segment = end - start
            length_squared = segment.length_squared()
            ratio = (position - start).dot(segment) / length_squared if length_squared else 0.0
            distances.append(position.distance_to(start + segment * min(1.0, max(0.0, ratio))))
        return min(distances) / self.map_diagonal

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        goal = self.wave_goal
        if self.mixed_goals:
            draw = self.np_random.random()
            if draw < 0.2:
                goal = self.task_balance.max_waves
            elif draw < 0.4 and goal > 1:
                goal = int(self.np_random.integers(1, goal))
        self.simulation.balance = replace(self.task_balance, max_waves=goal)
        episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        self.simulation.reset(seed=episode_seed)
        self.steps = 0
        self.reward_totals = dict.fromkeys(self.reward_components, 0.0)
        self.action_counts = dict.fromkeys(("defenses_built", "resources_built", "repairs_started"), 0)
        self.decision_counts = dict.fromkeys(("decisions_with_other_action_legal",
                                              "wait_when_other_action_legal",
                                              "forced_wait_decisions", "condensed_wait_ticks"), 0)
        self._ended = False
        return self.observation(), self.info()

    def set_wave_goal(self, waves, mixed=False):
        """Set the next training episode's objective; evaluation defaults to the full task."""
        if (isinstance(waves, bool) or not isinstance(waves, int)
                or not 1 <= waves <= self.task_balance.max_waves):
            raise ValueError("Wave goal must be within the configured task.")
        if not isinstance(mixed, bool):
            raise TypeError("Mixed-goal selection must be a boolean.")
        self.wave_goal = waves
        self.mixed_goals = mixed

    def action_masks(self):
        sim = self.simulation
        running = not sim.state.is_terminal
        funds = sim.state.resources
        return np.array([True] + [
            running and not slot.occupied and funds.can_afford(sim.config.towers[name].cost)
            for slot in sim.defense_slots.slots for name in sim.tower_types
        ] + [
            running and not slot.occupied and funds.can_afford(sim.config.resource_tower.cost)
            for slot in sim.resource_slots.slots
        ] + [
            running and king.destroyed and not king.repairing
            and funds.can_afford(king.repair_cost)
            and (king.alive or any(other.alive for other in sim.king_towers if other is not king))
            for king in sim.king_towers
        ], dtype=bool)

    @property
    def action_groups(self):
        return {
            "wait": slice(0, 1),
            "defense": slice(1, self.resource_action_start),
            "resource": slice(self.resource_action_start, self.repair_action_start),
            "repair": slice(self.repair_action_start, self.action_space.n),
        }

    def step(self, action):
        for result in self.step_ticks(action):
            pass
        return result

    def step_ticks(self, action):
        """Yield after each physics tick; only the last yield contains the step result."""
        if self._ended:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        sim = self.simulation
        previous = {
            "kills": sim.state.enemies_killed, "damage": sim.state.total_base_damage,
            "enemy_damage": sim.total_enemy_damage,
            "waves": sim.waves_completed, "spent": sim.state.resources.total_spent,
        }
        mask = self.action_masks()
        choice_available = bool(mask[1:].any())
        contained = self.action_space.contains(action)
        self.decision_counts["decisions_with_other_action_legal"] += int(choice_available)
        self.decision_counts["wait_when_other_action_legal"] += int(choice_available and contained and int(action) == 0)
        self.decision_counts["forced_wait_decisions"] += int(not choice_available)
        invalid = not (contained and mask[int(action)])
        skip_forced_waits = not invalid
        if not invalid and int(action) != 0:
            action = int(action)
            if action < self.resource_action_start:
                slot_index, tower_index = divmod(action - 1, len(sim.tower_types))
                sim.place_tower(slot_index, sim.tower_types[tower_index])
                self.action_counts["defenses_built"] += 1
            elif action < self.repair_action_start:
                sim.place_resource_tower(action - self.resource_action_start)
                self.action_counts["resources_built"] += 1
            else:
                king = sim.king_towers[action - self.repair_action_start]
                sim.start_king_tower_repair(king.structure_id)
                self.action_counts["repairs_started"] += 1
        tick = 0
        while True:
            sim.tick()
            tick += 1
            if sim.state.is_terminal or sim.ticks >= self.max_steps * self.decision_ticks:
                break
            if tick % self.decision_ticks == 0 and (not skip_forced_waits or self.action_masks()[1:].any()):
                break
            yield None
        self.steps += 1
        self.decision_counts["condensed_wait_ticks"] += max(0, tick - self.decision_ticks)
        terminated = sim.state.is_terminal
        truncated = sim.ticks >= self.max_steps * self.decision_ticks and not terminated
        weights = self.reward_weights
        enemy_damage = sim.total_enemy_damage - previous["enemy_damage"]
        terms = {
            "enemy_damage": weights["enemy_damage"] * max(0.0, enemy_damage),
            "kills": weights["kills"] * (sim.state.enemies_killed - previous["kills"]),
            "base_damage": weights["base_damage"] * (sim.state.total_base_damage - previous["damage"])
                           / sim.state.max_base_health,
            "waves": weights["waves"] * (sim.waves_completed - previous["waves"]),
            "outcome": weights["win"] if sim.state.status == GameStatus.WON else (
                weights["loss"] if sim.state.status == GameStatus.LOST or truncated else 0.0
            ),
            "invalid_action": weights["invalid_action"] if invalid else 0.0,
            "spending": weights["spending"] * (sim.state.resources.total_spent - previous["spent"])
                        / self.cost_scale,
        }
        self._ended = terminated or truncated
        for name, value in terms.items():
            self.reward_totals[name] += value
        info = self.info()
        info.update(reward_terms=terms, invalid_action=invalid, elapsed_ticks=tick)
        yield self.observation(), sum(terms.values()), terminated, truncated, info

    def observation(self):
        sim = self.simulation
        state, director = sim.state, sim.director
        values = [
            state.base_health / state.max_base_health, fraction(state.resource_balance, self.cost_scale),
            director.wave_number / self.task_balance.max_waves,
            sim.waves_completed / self.task_balance.max_waves,
            fraction(director.enemies_remaining_to_spawn, sim.balance.waves.initial_enemy_count),
            fraction(director.spawn_timer, 1.0), fraction(director.intermission_timer, 1.0),
            sim.ticks / (self.max_steps * self.decision_ticks), float(director.wave_active),
            state.king_towers_alive / state.king_tower_count,
            fraction(len(sim.enemies), sim.balance.waves.initial_enemy_count),
            fraction(sum(tower.production_amount / tower.production_interval
                         for tower in sim.resource_towers if tower.alive), 1.0),
            sim.balance.max_waves / self.task_balance.max_waves,
            sim._passive_income_elapsed / sim.config.passive_income_interval,
            fraction(director._calculate_enemy_count(director.wave_number + 1),
                     sim.balance.waves.initial_enemy_count)
            if director.wave_number < sim.balance.max_waves else 0.0,
        ]
        for config in sim.config.towers.values():
            values.extend([
                fraction(config.cost, self.cost_scale), config.range_radius / self.map_diagonal,
                fraction(config.damage, self.health_scale), fraction(config.fire_rate, 1.0),
                fraction(config.shots_per_burst, 1.0), fraction(config.max_health, self.health_scale),
            ])
        for name, config in sim.config.enemies.items():
            values.extend([
                fraction(config.max_health, self.health_scale), fraction(config.speed, 100.0),
                fraction(config.attack_damage, 100.0), config.attack_range / self.map_diagonal,
                fraction(config.attack_interval, 1.0), fraction(config.kill_reward, self.cost_scale),
                float(sim.config.enemy_assets[name].is_tank),
            ])
        maximum_range = max(config.range_radius for config in sim.config.towers.values())

        def proximity(position):
            distances = [(position.distance_to(enemy.position), enemy) for enemy in sim.enemies if enemy.alive]
            nearest = min((distance for distance, _ in distances), default=self.map_diagonal)
            health = sum(enemy.health / self.health_scale for distance, enemy in distances
                         if distance <= maximum_range)
            return nearest / self.map_diagonal, fraction(health, 1.0)

        towers = {tower.slot_id: tower for tower in sim.towers if tower.alive}
        for slot in sim.defense_slots.slots:
            tower = towers.get(slot.slot_id)
            values.extend([
                slot.center.x / sim.map_width, slot.center.y / sim.map_height, float(slot.occupied),
                (sim.tower_types.index(tower.tower_type) + 1) / len(sim.tower_types) if tower else 0.0,
                fraction(tower.cooldown, 1.0) if tower else 0.0,
                tower.health_fraction if tower else 0.0,
                tower.range_radius / self.map_diagonal if tower else 0.0,
                fraction(tower.damage * tower.fire_rate * tower.shots_per_burst, self.health_scale)
                if tower else 0.0,
                self.soldier_route_distances[slot.slot_id],
                self.tank_route_distances[slot.slot_id], *proximity(slot.center),
            ])
        resources = {tower.structure_id: tower for tower in sim.resource_towers if tower.alive}
        resource_config = sim.config.resource_tower
        for slot in sim.resource_slots.slots:
            tower = resources.get(slot.slot_id)
            values.extend([
                slot.center.x / sim.map_width, slot.center.y / sim.map_height, float(slot.occupied),
                tower.health_fraction if tower else 0.0, tower.production_progress if tower else 0.0,
                fraction(resource_config.production_amount / resource_config.production_interval, 1.0),
                fraction(resource_config.cost, self.cost_scale),
                self.soldier_route_distances[slot.slot_id],
                self.tank_route_distances[slot.slot_id], proximity(slot.center)[0],
            ])
        for king in sim.king_towers:
            pressure = sum(enemy.config.attack_damage / enemy.config.attack_interval
                           for enemy in sim.enemies if enemy.alive and enemy.target is king)
            incoming = sum(shot.damage for shot in sim.projectiles if shot.alive and shot.target is king)
            values.extend([
                king.position.x / sim.map_width, king.position.y / sim.map_height,
                king.health_fraction, float(king.alive), float(king.repairing), king.repair_progress,
                fraction(pressure, king.max_health / king.repair_duration), incoming / king.max_health,
                fraction(king.repair_cost, self.cost_scale),
            ])
        rows = []
        for enemy in sim.enemies:
            offsets = self.route_offsets[enemy.pathway]
            segment = min(enemy.waypoint_index - 1, len(offsets) - 2)
            travelled = offsets[segment] + enemy.position.distance_to(enemy.pathway.points[segment])
            rows.append([
                1.0, enemy.position.x / sim.map_width, enemy.position.y / sim.map_height,
                enemy.health / enemy.max_health,
                (sim.enemy_types.index(enemy.enemy_type) + 1) / len(sim.enemy_types),
                1.0 if enemy.reached_end else min(1.0, travelled / offsets[-1]),
                fraction(enemy.max_health, self.health_scale), fraction(enemy.speed, 100.0),
                fraction(enemy.config.attack_damage, 100.0), enemy.attack_range / self.map_diagonal,
                float(enemy.target is not None), enemy.attack_cooldown / enemy.config.attack_interval,
            ])
        rows.sort(key=lambda row: row[5], reverse=True)
        enemies = np.zeros((self.max_enemies, self.enemy_features), dtype=np.float32)
        # shortcut: retain the most advanced 64 enemies; increase capacity before training larger waves.
        for index, row in enumerate(rows[:self.max_enemies]):
            enemies[index] = row
        observation = np.concatenate((np.asarray(values, dtype=np.float32), enemies.ravel()))
        return np.clip(observation, 0.0, 1.0)

    def info(self):
        sim = self.simulation
        return {
            "difficulty": sim.difficulty, "status": sim.state.status.value,
            "is_success": sim.state.status == GameStatus.WON,
            "waves_completed": sim.waves_completed, "base_health": sim.state.base_health,
            "waves_reached": sim.director.wave_number, "survived": sim.state.base_health > 0,
            "enemies_omitted": max(0, len(sim.enemies) - self.max_enemies),
            "kills": sim.state.enemies_killed, "leaks": sim.state.enemies_leaked,
            "resources": sim.state.resource_balance, "spent": sim.state.resources.total_spent,
            "king_towers_alive": sim.state.king_towers_alive,
            "base_damage": sim.state.total_base_damage,
            "simulation_ticks": sim.ticks, "simulated_seconds": sim.ticks * sim.timestep,
            "wave_goal": sim.balance.max_waves,
            **self.action_counts,
            "defense_types_alive": {name: sum(tower.alive and tower.tower_type == name
                                               for tower in sim.towers) for name in sim.tower_types},
            "defense_types_built": dict(sim.defense_types_built),
            **self.decision_counts,
            "episode_events": list(sim.events) if self._ended else [],
            **{f"reward_{name}": value for name, value in self.reward_totals.items()},
        }
