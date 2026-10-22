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
    max_enemies = 64
    reward_components = ("kills", "base_damage", "waves", "outcome", "invalid_action", "spending")

    def __init__(self, difficulty="baseline", *, config=None, max_steps=2400, render_assets=False):
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
        size = 8 + len(sim.defense_slots.slots) * 5 + self.max_enemies * 6
        self.observation_space = spaces.Box(0.0, np.inf, shape=(size,), dtype=np.float32)
        self.route_offsets = {}
        for route in (*sim.soldier_routes.values(), *sim.tank_routes.values()):
            lengths = [start.distance_to(end) for start, end in zip(route.points, route.points[1:])]
            offsets = np.concatenate(([0.0], np.cumsum(lengths)))
            if offsets[-1] <= 0:
                raise ValueError("The enemy route must have positive length.")
            self.route_offsets[route] = offsets

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        self.simulation.reset(seed=episode_seed)
        self.steps = 0
        self.reward_totals = dict.fromkeys(self.reward_components, 0.0)
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
        for result in self.step_ticks(action):
            pass
        return result

    def step_ticks(self, action):
        """Yield after each physics tick; only the last yield contains the step result."""
        if self._ended:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        sim = self.simulation
        previous = (sim.state.enemies_killed, sim.state.base_health, sim.waves_completed,
                    sim.state.resources.total_spent)
        invalid = not (self.action_space.contains(action) and self.action_masks()[int(action)])
        if not invalid and int(action) != 0:
            slot_index, tower_index = divmod(int(action) - 1, len(sim.tower_types))
            sim.place_tower(slot_index, sim.tower_types[tower_index])
        for tick in range(self.decision_ticks):
            sim.tick()
            if sim.state.is_terminal:
                break
            if tick + 1 < self.decision_ticks:
                yield None
        self.steps += 1
        terms = {
            "kills": 0.1 * (sim.state.enemies_killed - previous[0]),
            "base_damage": -max(0.0, previous[1] - sim.state.base_health) / 10.0,
            "waves": float(sim.waves_completed - previous[2]),
            "outcome": 20.0 if sim.state.status == GameStatus.WON else (
                -20.0 if sim.state.status == GameStatus.LOST else 0.0
            ),
            "invalid_action": -0.1 if invalid else 0.0,
            "spending": -(sim.state.resources.total_spent - previous[3]) / 1000.0,
        }
        terminated = sim.state.is_terminal
        truncated = self.steps >= self.max_steps and not terminated
        self._ended = terminated or truncated
        for name, value in terms.items():
            self.reward_totals[name] += value
        info = self.info()
        info.update(reward_terms=terms, invalid_action=invalid)
        yield self.observation(), sum(terms.values()), terminated, truncated, info

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
        rows = []
        for enemy in sim.enemies:
            offsets = self.route_offsets[enemy.pathway]
            segment = min(enemy.waypoint_index - 1, len(offsets) - 2)
            travelled = offsets[segment] + enemy.position.distance_to(
                enemy.pathway.points[segment]
            )
            rows.append([
                1.0, enemy.position.x / sim.pathway.map_width,
                enemy.position.y / sim.pathway.map_height, enemy.health / 100.0,
                (sim.enemy_types.index(enemy.enemy_type) + 1) / len(sim.enemy_types),
                min(1.0, travelled / offsets[-1]),
            ])
        rows.sort(key=lambda row: row[5], reverse=True)
        enemies = np.zeros((self.max_enemies, 6), dtype=np.float32)
        # shortcut: keep the nearest 64 enemies to the base; raise capacity for larger waves.
        for index, row in enumerate(rows[:self.max_enemies]):
            enemies[index] = row
        return np.concatenate((np.asarray(values, dtype=np.float32), enemies.ravel()))

    def info(self):
        sim = self.simulation
        return {
            "difficulty": sim.difficulty, "status": sim.state.status.value,
            "is_success": sim.state.status == GameStatus.WON,
            "waves_completed": sim.waves_completed, "base_health": sim.state.base_health,
            "waves_reached": sim.director.wave_number,
            "survived": sim.state.base_health > 0,
            "enemies_omitted": max(0, len(sim.enemies) - self.max_enemies),
            "kills": sim.state.enemies_killed, "leaks": sim.state.enemies_leaked,
            "resources": sim.state.resource_balance,
            "spent": sim.state.resources.total_spent,
            "king_towers_alive": sum(king.alive for king in sim.king_towers),
            "resource_towers_alive": sum(tower.alive for tower in sim.resource_towers),
            **{f"reward_{name}": value for name, value in self.reward_totals.items()},
        }
