from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.game.simulation import Simulation
from src.game.state.game_state import GameStatus


class BastionEnv(gym.Env):
    """Tower placement every half second; combat advances in 1/60-second ticks."""

    metadata: ClassVar[dict] = {"render_modes": []}
    decision_ticks = 30
    route_bins = 12

    def __init__(self, difficulty="medium", *, config=None, max_steps=2400, render_assets=False):
        super().__init__()
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        self.simulation = Simulation(difficulty, config=config, render_assets=render_assets)
        self.max_steps = max_steps
        self.steps = 0
        self._ended = True
        sim = self.simulation
        self.action_space = spaces.Discrete(
            1 + len(sim.defense_slots.slots) * len(sim.tower_types)
        )
        size = 8 + len(sim.defense_slots.slots) * 5 + self.route_bins * len(sim.enemy_types) * 2
        self.observation_space = spaces.Box(0.0, np.inf, shape=(size,), dtype=np.float32)
        self.segment_lengths = np.array([
            start.distance_to(end)
            for start, end in zip(sim.pathway.points, sim.pathway.points[1:])
        ])
        self.route_offsets = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        self.route_length = float(self.route_offsets[-1])
        if self.route_length <= 0:
            raise ValueError("The enemy route must have positive length.")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        self.simulation.reset(seed=episode_seed)
        self.steps = 0
        self.reward_totals = dict.fromkeys(("kills", "base_damage", "waves", "outcome"), 0.0)
        self._ended = False
        return self.observation(), self.info()

    def action_masks(self):
        sim = self.simulation
        return np.array([True] + [
            not sim.state.is_terminal and not slot.occupied
            and sim.state.resources.can_afford(sim.config.towers[name].cost)
            for slot in sim.defense_slots.slots for name in sim.tower_types
        ], dtype=bool)

    def step(self, action):
        if self._ended:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Action outside action space: {action}")
        sim = self.simulation
        previous = (sim.state.enemies_killed, sim.state.base_health, sim.waves_completed)
        invalid = not self.action_masks()[int(action)]
        if action and not invalid:
            slot_index, tower_index = divmod(int(action) - 1, len(sim.tower_types))
            sim.place_tower(slot_index, sim.tower_types[tower_index])
        for _ in range(self.decision_ticks):
            sim.tick()
            if sim.state.is_terminal:
                break
        self.steps += 1
        terms = {
            "kills": 0.1 * (sim.state.enemies_killed - previous[0]),
            "base_damage": -(previous[1] - sim.state.base_health) / 10.0,
            "waves": float(sim.waves_completed - previous[2]),
            "outcome": 20.0 if sim.state.status == GameStatus.WON else (
                -20.0 if sim.state.status == GameStatus.LOST else 0.0
            ),
        }
        terminated = sim.state.is_terminal
        truncated = self.steps >= self.max_steps and not terminated
        self._ended = terminated or truncated
        for name, value in terms.items():
            self.reward_totals[name] += value
        info = self.info()
        info.update(reward_terms=terms, invalid_action=invalid)
        return self.observation(), sum(terms.values()), terminated, truncated, info

    def observation(self):
        sim = self.simulation
        state, director = sim.state, sim.director
        values = [
            state.base_health / state.max_base_health, state.resource_balance / 200.0,
            director.wave_number / sim.balance.max_waves,
            sim.waves_completed / sim.balance.max_waves,
            director.enemies_remaining_to_spawn / 50.0,
            director.spawn_timer, director.intermission_timer, self.steps / self.max_steps,
        ]
        towers = {tower.slot_id: tower for tower in sim.towers}
        for slot in sim.defense_slots.slots:
            tower = towers.get(slot.slot_id)
            values.extend([
                slot.center.x / sim.pathway.map_width,
                slot.center.y / sim.pathway.map_height,
                float(slot.occupied),
                (sim.tower_types.index(tower.tower_type) + 1) / len(sim.tower_types) if tower else 0.0,
                tower.cooldown if tower else 0.0,
            ])
        # shortcut: route bins omit exact positions/projectiles; enrich if learning plateaus.
        bins = np.zeros((len(sim.enemy_types), self.route_bins, 2), dtype=np.float32)
        for enemy in sim.enemies:
            segment = min(enemy.waypoint_index - 1, len(self.segment_lengths) - 1)
            travelled = self.route_offsets[segment] + enemy.position.distance_to(
                sim.pathway.points[segment]
            )
            index = min(self.route_bins - 1, int(travelled / self.route_length * self.route_bins))
            enemy_type = sim.enemy_types.index(enemy.enemy_type)
            bins[enemy_type, index, 0] += 0.1
            bins[enemy_type, index, 1] += enemy.health / 100.0
        return np.concatenate((np.asarray(values, dtype=np.float32), bins.ravel()))

    def info(self):
        sim = self.simulation
        return {
            "difficulty": sim.difficulty, "status": sim.state.status.value,
            "is_success": sim.state.status == GameStatus.WON,
            "waves_completed": sim.waves_completed, "base_health": sim.state.base_health,
            "kills": sim.state.enemies_killed, "leaks": sim.state.enemies_leaked,
            "resources": sim.state.resource_balance,
            "spent": sim.state.resources.total_spent,
            **{f"reward_{name}": value for name, value in self.reward_totals.items()},
        }
