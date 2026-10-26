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
        "schema": 7,
        "king_repair_requires_destroyed": True,
        "config": config,
        "map_sha256": hashlib.sha256(sim.config.pathway_path.read_bytes()).hexdigest(),
        "slots": [slot.slot_id for slot in sim.defense_slots.slots],
        "tower_types": list(sim.tower_types),
        "enemy_types": list(sim.enemy_types),
        "observation_size": env.observation_space.shape[0],
        "max_enemies": env.max_enemies,
        "reward_weights": env.reward_weights,
        "action_count": int(env.action_space.n),
        "decision_ticks": env.decision_ticks,
        "forced_wait_skip": True,
        "timestep": sim.timestep,
    }


def save_metadata(path, env, *, seed, steps, training_parameters=None, checkpoint_steps=None,
                  curriculum_state=None, normalization=False):
    path = Path(path)
    data = {"environment": manifest(env), "training_difficulty": env.simulation.difficulty,
            "training_seed": seed, "requested_steps": steps,
            "training_max_steps": env.max_steps,
            "training_balance": asdict(env.simulation.balance),
            "training_parameters": training_parameters}
    data["checkpoint_steps"] = checkpoint_steps
    data["curriculum_state"] = curriculum_state
    data["normalization"] = normalization
    path.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")


def save_checkpoint(path, model, env, *, seed, steps, training_parameters, curriculum=None):
    path = Path(path)
    normalizer = model.get_vec_normalize_env()
    if normalizer is not None:
        normalizer.save(path.with_suffix(".vecnormalize.pkl"))
    model.save(path)
    save_metadata(path, env, seed=seed, steps=steps, training_parameters=training_parameters,
                  checkpoint_steps=model.num_timesteps,
                  curriculum_state=curriculum.state_dict() if curriculum is not None else None,
                  normalization=normalizer is not None)


def policy_observation(model, observation):
    """Apply frozen checkpoint statistics to raw inference observations."""
    normalizer = model.get_vec_normalize_env()
    return normalizer.normalize_obs(observation) if normalizer is not None else observation


def load_model(path, env, *, device="cpu", vec_env=None):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["environment"] != manifest(env):
        raise ValueError("Checkpoint environment differs from the current map/config/schema.")
    if metadata.get("normalization", False):
        if vec_env is None:
            vec_env = VecNormalize.load(path.with_suffix(".vecnormalize.pkl"), DummyVecEnv([lambda: env]))
            vec_env.training = False
            vec_env.norm_reward = False
        elif not isinstance(vec_env, VecNormalize):
            raise ValueError("Normalized checkpoint requires its saved VecNormalize statistics.")
    torch.set_num_threads(1)
    return MaskablePPO.load(path, env=vec_env if vec_env is not None else env, device=device)
