"""Scripted winnability probe; never used by PPO training or model gameplay."""

import math

import numpy as np


class ScriptedDefense:
    def __init__(self, env, *, economy=False):
        self.economy = economy
        sim = env.simulation
        routes = [*sim.soldier_routes.values(), *sim.tank_routes.values()]
        points, weights, soldier_samples = [], [], []
        for index, route in enumerate(routes):
            samples = [start.lerp(end, float(ratio))
                       for start, end in zip(route.points, route.points[1:])
                       for ratio in np.linspace(0, 1, max(1, math.ceil(start.distance_to(end) / 90)), endpoint=False)]
            samples.append(route.points[-1])
            points.extend(samples)
            weights.extend([1 / (len(routes) * len(samples))] * len(samples))
            soldier_samples.extend([index < len(sim.soldier_routes)] * len(samples))
        self.weights = np.array(weights)
        soldiers = np.array(soldier_samples)
        self.soldier_weights = self.weights * soldiers
        self.soldier_weights /= self.soldier_weights.sum()
        self.tank_weights = self.weights * ~soldiers
        self.tank_weights /= self.tank_weights.sum()
        self.positions = np.array([tuple(slot.center) for slot in sim.defense_slots.slots])
        self.slot_indices = {slot.slot_id: index for index, slot in enumerate(sim.defense_slots.slots)}
        distances = ((self.positions[:, None, :] - np.array(points)[None, :, :]) ** 2).sum(-1)
        self.coverage = {name: distances <= config.range_radius ** 2
                         for name, config in sim.config.towers.items()}

    def __call__(self, env, mask):
        sim = env.simulation
        towers = [tower for tower in sim.towers if tower.alive]
        if (self.economy and len(towers) >= 2 and sim.ticks * sim.timestep < 120
                and not any(tower.alive for tower in sim.resource_towers)):
            candidates = [index for index in range(len(sim.resource_slots.slots))
                          if mask[env.resource_action_start + index]]
            if candidates:
                index = max(candidates, key=lambda index: (
                    env.soldier_route_distances[sim.resource_slots.slots[index].slot_id]
                    + env.tank_route_distances[sim.resource_slots.slots[index].slot_id], -index,
                ))
                return env.resource_action_start + index
        covered = np.zeros(len(self.weights))
        for tower in towers:
            covered += self.coverage[tower.tower_type][self.slot_indices[tower.slot_id]]
        best_score, best_action = -1.0, 0
        wave = sim.director.wave_number
        unlock = sim.balance.waves.light_tank_start_wave
        tank_share = 0.0 if wave < unlock else min(0.8, 0.2 + 0.1 * (wave - unlock))
        route_weights = (1 - tank_share) * self.soldier_weights + tank_share * self.tank_weights
        values = {}
        for name, config in sim.config.towers.items():
            if config.fire_rate > 0:
                cycle = 1 / config.fire_rate + (config.shots_per_burst - 1) * config.burst_interval
                values[name] = config.damage * config.shots_per_burst / (cycle * config.cost)
        preferred = max(values, key=values.get) if values else None
        for tower_index, (name, config) in enumerate(sim.config.towers.items()):
            if name != preferred:
                continue
            damage_per_cost = values[name]
            gains = self.coverage[name] @ (route_weights / (1 + covered))
            for enemy in sim.enemies:
                if not enemy.alive:
                    continue
                defenders = sum(tower.position.distance_to(enemy.position) <= tower.range_radius for tower in towers)
                in_range = ((self.positions - np.array(enemy.position)) ** 2).sum(-1) <= config.range_radius ** 2
                gains += in_range * enemy.health / (env.health_scale * (1 + defenders))
            for slot_index, gain in enumerate(gains):
                action = 1 + slot_index * len(sim.tower_types) + tower_index
                score = float(gain * damage_per_cost)
                if mask[action] and score > best_score:
                    best_score, best_action = score, action
        return best_action
