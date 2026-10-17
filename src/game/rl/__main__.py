import argparse
import json
from pathlib import Path

import numpy as np

from src.game.rl.checkpoint import load_model, save_metadata
from src.game.rl.environment import BastionEnv


def coverage_scores(env):
    sim = env.simulation
    points = []
    for start, end in zip(sim.pathway.points, sim.pathway.points[1:]):
        count = max(1, int(start.distance_to(end) / 16))
        points.extend(start.lerp(end, index / count) for index in range(count))
    return np.array([0.0] + [
        sum(slot.center.distance_to(point) <= sim.config.towers[name].range_radius
            for point in points)
        * sim.config.towers[name].damage * sim.config.towers[name].fire_rate
        / sim.config.towers[name].cost
        for slot in sim.defense_slots.slots for name in sim.tower_types
    ])


def evaluate(args):
    env = BastionEnv(args.difficulty)
    model = load_model(args.model, env) if args.model else None
    scores = coverage_scores(env)
    episodes = []
    for seed in range(args.seed, args.seed + args.episodes):
        observation, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        total_reward = 0.0
        while True:
            mask = env.action_masks()
            if model:
                action, _ = model.predict(observation, deterministic=True, action_masks=mask)
            elif args.policy == "random":
                action = rng.choice(np.flatnonzero(mask))
            else:
                action = int(np.argmax(np.where(mask, scores, -np.inf)))
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            if terminated or truncated:
                episodes.append(dict(info, seed=seed, reward=total_reward,
                                     steps=env.steps, truncated=truncated))
                break
    result = {
        "difficulty": args.difficulty,
        "policy": str(args.model) if model else args.policy,
        "episodes": len(episodes),
        "win_rate": float(np.mean([item["is_success"] for item in episodes])),
        "mean_waves_completed": float(np.mean([item["waves_completed"] for item in episodes])),
        "mean_base_health": float(np.mean([item["base_health"] for item in episodes])),
        "mean_leaks": float(np.mean([item["leaks"] for item in episodes])),
        "mean_spent": float(np.mean([item["spent"] for item in episodes])),
        "results": episodes,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
    env.close()


def train(args):
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    args.output = args.output or Path("runs") / args.difficulty
    args.output.mkdir(parents=True, exist_ok=True)
    env = BastionEnv(args.difficulty)
    evaluation = BastionEnv(args.difficulty)
    evaluation.reset(seed=10000)
    monitored = Monitor(env, str(args.output / "episodes.csv"),
                        info_keywords=("is_success", "waves_completed", "base_health",
                                       "reward_kills", "reward_base_damage",
                                       "reward_waves", "reward_outcome"))
    model = load_model(args.resume, env) if args.resume else MaskablePPO(
        "MlpPolicy", monitored, seed=args.seed, device="cpu", verbose=1,
        n_steps=256, batch_size=64, gamma=0.995,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.set_env(monitored)
    model.set_random_seed(args.seed)
    model.set_logger(configure(str(args.output), ["stdout", "csv"]))
    callback = MaskableEvalCallback(
        Monitor(evaluation), best_model_save_path=str(args.output),
        log_path=str(args.output), eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes, deterministic=True,
    )
    save_metadata(args.output / "best_model.zip", env, seed=args.seed, steps=args.steps)
    try:
        model.learn(args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
        model.save(args.output / "final.zip")
        save_metadata(args.output / "final.zip", env, seed=args.seed, steps=args.steps)
    finally:
        monitored.close()
        evaluation.close()


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate BastionRL tower placement.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--steps", type=int, default=200000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--output", type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--eval-freq", type=int, default=10000)
    training.add_argument("--eval-episodes", type=int, default=5)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--policy", choices=("random", "heuristic"), default="heuristic")
    evaluation.add_argument("--model", type=Path)
    evaluation.add_argument("--episodes", type=int, default=100)
    evaluation.add_argument("--seed", type=int, default=30000)
    evaluation.add_argument("--output", type=Path)
    for command in (training, evaluation):
        command.add_argument("--difficulty", choices=("medium", "hard"), default="medium")
    args = parser.parse_args()
    for field in ("steps", "eval_freq", "eval_episodes", "episodes"):
        if hasattr(args, field) and getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
