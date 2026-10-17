import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from src.game.config import PROJECT_ROOT


def manifest(env):
    sim = env.simulation
    config = asdict(sim.config)
    config.pop("difficulties")
    config = json.loads(json.dumps(config, default=lambda path: str(path.relative_to(PROJECT_ROOT))))
    return {
        "schema": 2,
        "config": config,
        "map_sha256": hashlib.sha256(sim.config.pathway_path.read_bytes()).hexdigest(),
        "slots": [slot.slot_id for slot in sim.defense_slots.slots],
        "tower_types": list(sim.tower_types),
        "enemy_types": list(sim.enemy_types),
        "observation_size": env.observation_space.shape[0],
        "max_enemies": env.max_enemies,
        "action_count": int(env.action_space.n),
        "decision_ticks": env.decision_ticks,
        "timestep": sim.timestep,
    }


def save_metadata(path, env, *, seed, steps):
    path = Path(path)
    data = {"environment": manifest(env), "training_difficulty": env.simulation.difficulty,
            "training_seed": seed, "requested_steps": steps,
            "training_max_steps": env.max_steps,
            "training_balance": asdict(env.simulation.balance)}
    path.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")


def load_model(path, env):
    from sb3_contrib import MaskablePPO

    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["environment"] != manifest(env):
        raise ValueError("Checkpoint environment differs from the current map/config/schema.")
    return MaskablePPO.load(path, env=env, device="cpu")
