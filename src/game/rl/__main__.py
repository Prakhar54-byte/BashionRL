import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path

from src.game.rl.checkpoint import load_model, save_checkpoint
from src.game.rl.environment import BastionEnv
from src.game.rl.evaluation import play_episodes, summarize_episodes


def evaluate(args):
    env = BastionEnv(args.difficulty, max_steps=args.max_steps)
    diagnostic = args.command == "diagnose"
    diagnostic_policy = None
    if diagnostic:
        from src.game.rl.diagnostics import ScriptedDefense

        model = None
        diagnostic_policy = ScriptedDefense(env, economy=args.economy)
        policy_name = "scripted_economy" if args.economy else "scripted_defense_only"
    else:
        model = load_model(args.model, env) if args.model else None
        policy_name = str(args.model) if model else args.policy
    episodes = play_episodes(env, model, range(args.seed, args.seed + args.episodes),
                             diagnostic_policy=diagnostic_policy)
    result = {
        "difficulty": args.difficulty,
        "policy": policy_name,
        "episodes": len(episodes),
        "seed": args.seed,
        "max_steps": args.max_steps,
        **summarize_episodes(episodes),
        "results": episodes,
    }
    prefix = policy_name if diagnostic else "evaluation"
    args.output = args.output or Path("runs") / f"{prefix}-{args.difficulty}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
    env.close()
    return result


def train(args):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from src.game.rl.curriculum import CurriculumCallback
    from src.game.rl.policy import HierarchicalPolicy
    from src.game.rl.training import (
        AnnealingCallback,
        OutcomeEvalCallback,
        TrainingMetricsCallback,
        initialize_flat_action_bias,
    )

    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or install a CUDA-enabled PyTorch on the server.")
    args.output = args.output or Path("runs") / args.difficulty
    args.output.mkdir(parents=True, exist_ok=True)
    resume_metadata = json.loads(args.resume.with_suffix(".json").read_text()) if args.resume else {}
    if args.seed is None:
        args.seed = resume_metadata.get("training_seed", 0)
    saved_curriculum = resume_metadata.get("curriculum_state")
    saved_mode = saved_curriculum["mode"] if saved_curriculum else "staged"
    saved_threshold = saved_curriculum["promotion_successes"] if saved_curriculum else 40
    mode = args.curriculum or saved_mode
    threshold = args.promotion_successes or saved_threshold
    normalize = bool(resume_metadata.get("normalization")) if args.normalize is None else args.normalize
    if args.resume:
        if normalize != bool(resume_metadata.get("normalization")):
            raise ValueError("Changing normalization requires fresh training, not resume.")
        if saved_curriculum and (mode != saved_mode or threshold != saved_threshold):
            raise ValueError("Changing saved curriculum settings requires a fresh experiment.")
    with ExitStack() as resources:
        env = BastionEnv(args.difficulty, max_steps=args.max_steps)
        resources.callback(env.close)
        evaluation = BastionEnv(args.difficulty, max_steps=args.max_steps)
        resources.callback(evaluation.close)
        monitored = make_vec_env(
            BastionEnv, n_envs=args.n_envs, seed=args.seed, monitor_dir=str(args.output / "episodes"),
            env_kwargs={"difficulty": args.difficulty, "max_steps": args.max_steps},
            vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
            vec_env_kwargs={"start_method": "spawn"} if args.n_envs > 1 else {},
            monitor_kwargs={"info_keywords": ("is_success", "waves_completed", "base_health", "wave_goal",
                                             *(f"reward_{name}" for name in env.reward_components))},
        )
        resources.callback(monitored.close)
        vector_backend = type(monitored).__name__
        if normalize:
            monitored = (VecNormalize.load(args.resume.with_suffix(".vecnormalize.pkl"), monitored)
                         if args.resume else VecNormalize(monitored, gamma=0.999))
            monitored.training, monitored.norm_reward = True, True
        curriculum = CurriculumCallback(monitored, env.task_balance.max_waves, mode=mode,
                                         promotion_successes=threshold, state=saved_curriculum)
        strategy = args.strategy or "hierarchical"
        policy_kwargs = {"net_arch": [128, 128]}
        if strategy == "hierarchical":
            policy_kwargs["action_group_sizes"] = [
                int(group.stop - group.start) for group in env.action_groups.values()
            ]
        training_parameters = {"n_steps": args.n_steps or 256, "batch_size": args.batch_size or 64,
                               "n_epochs": args.n_epochs or 10, "gamma": 0.999, "gae_lambda": 0.98,
                               "ent_coef": args.ent_coef if args.ent_coef is not None else 0.02,
                               "learning_rate": args.learning_rate if args.learning_rate is not None else 0.0003,
                               "target_kl": 0.03,
                               "policy_kwargs": policy_kwargs}
        rollout_size = training_parameters["n_steps"] * args.n_envs
        if not args.resume and (rollout_size < 2 or training_parameters["batch_size"] > rollout_size
                               or rollout_size % training_parameters["batch_size"]):
            raise ValueError("Rollout size (n_steps × n_envs) must be at least 2 and divisible by batch_size.")
        model = load_model(args.resume, env, device=args.device, vec_env=monitored) if args.resume else MaskablePPO(
            HierarchicalPolicy if strategy == "hierarchical" else "MlpPolicy", monitored,
            seed=args.seed, device=args.device, verbose=1, **training_parameters,
        )
        if args.resume:
            strategy = "hierarchical" if isinstance(model.policy, HierarchicalPolicy) else "flat"
            if args.strategy is not None and args.strategy != strategy:
                raise ValueError("Policy strategy differs from the checkpoint; start fresh to change architecture.")
            for option in ("n_steps", "batch_size", "n_epochs"):
                value = getattr(args, option)
                if value is not None and value != getattr(model, option):
                    raise ValueError(f"Changing {option} on resume requires a fresh experiment.")
            if args.ent_coef is not None:
                model.ent_coef = args.ent_coef
            if args.learning_rate is not None:
                model.learning_rate = args.learning_rate
                model._setup_lr_schedule()
        elif strategy == "flat":
            initialize_flat_action_bias(model.policy, env)
        rollout_size = model.n_steps * monitored.num_envs
        if rollout_size % model.batch_size or model.batch_size > rollout_size:
            raise ValueError("Resumed rollout size must be divisible by the saved batch_size.")
        if (not hasattr(model, "training_schedule") or any(value is not None for value in
                (args.learning_rate, args.lr_final, args.ent_coef, args.ent_final))):
            model.training_schedule = {
                "start_step": model.num_timesteps, "end_step": model.num_timesteps + args.steps,
                "lr_start": float(model.learning_rate),
                "lr_end": args.lr_final if args.lr_final is not None else float(model.learning_rate),
                "entropy_start": model.ent_coef,
                "entropy_end": args.ent_final if args.ent_final is not None else model.ent_coef,
            }
        training_parameters = {
            "n_steps": model.n_steps, "batch_size": model.batch_size, "gamma": model.gamma,
            "gae_lambda": model.gae_lambda, "ent_coef": model.ent_coef, "target_kl": model.target_kl,
            "learning_rate": model.learning_rate,
            "policy": type(model.policy).__name__, "policy_kwargs": model.policy_kwargs,
            "n_envs": monitored.num_envs, "device": str(model.device),
            "n_epochs": model.n_epochs, "rollout_size": rollout_size,
            "vector_backend": vector_backend, "normalization": normalize,
            "schedule": model.training_schedule,
        }
        training_parameters["curriculum"] = {"goals": curriculum.goals, "window": 50,
                                             "promotion_successes": threshold, "mode": mode}
        training_parameters["strategy"] = strategy
        training_parameters["deterministic_selection"] = "kind_then_target" if strategy == "hierarchical" else "flat_argmax"
        training_parameters["validation"] = {
            "seeds": list(range(args.eval_seed, args.eval_seed + args.eval_episodes)),
            "selection": ["win_rate", "mean_waves_completed", "mean_base_health"],
            "requested_frequency": args.eval_freq,
            "effective_frequency": math.ceil(args.eval_freq / rollout_size) * rollout_size,
            "timing": "after_ppo_update",
        }
        model.set_env(monitored)
        model.set_random_seed(args.seed)
        model.set_logger(configure(str(args.output), ["stdout", "csv"]))
        callback = OutcomeEvalCallback(
            evaluation, args.output, eval_freq=args.eval_freq, episodes=args.eval_episodes,
            seed=args.eval_seed, training_seed=args.seed, steps=args.steps,
            training_parameters=training_parameters,
            curriculum=curriculum,
        )
        initial_steps, initial_epochs = model.num_timesteps, model._n_updates
        optimizer_steps = 0

        def count_optimizer_step(*_):
            nonlocal optimizer_steps
            optimizer_steps += 1
            model.logger.record("train/completed_optimizer_steps", optimizer_steps)

        hook = model.policy.optimizer.register_step_post_hook(count_optimizer_step)
        resources.callback(hook.remove)
        print(f"PPO rollout: {monitored.num_envs} envs × {model.n_steps} steps = {rollout_size} samples; "
              f"batch={model.batch_size}, epochs≤{model.n_epochs}, strategy={strategy}, "
              f"backend={vector_backend}, normalize={normalize}, curriculum={mode}")
        if args.resume and saved_curriculum is None:
            print("Checkpoint has no curriculum state: starting at goal 1; policy and optimizer were restored.")
        model.learn(args.steps, callback=[curriculum, AnnealingCallback(), TrainingMetricsCallback(args.output), callback],
                    reset_num_timesteps=not bool(args.resume))
        save_checkpoint(args.output / "final.zip", model, env, seed=args.seed, steps=args.steps,
                        training_parameters=training_parameters, curriculum=curriculum)
        actual_steps = model.num_timesteps - (initial_steps if args.resume else 0)
        summary = {"requested_steps": args.steps, "actual_steps": actual_steps,
                   "starting_timesteps": initial_steps if args.resume else 0,
                   "ending_timesteps": model.num_timesteps,
                   "resumed_from": str(args.resume) if args.resume else None,
                   "rollout_size": rollout_size, "completed_rollouts": actual_steps // rollout_size,
                   "completed_ppo_epochs": model._n_updates - initial_epochs,
                   "completed_optimizer_steps": optimizer_steps,
                   "final_learning_rate": float(model.learning_rate),
                   "final_entropy_coefficient": model.ent_coef,
                   "training_parameters": training_parameters}
        (args.output / "training-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate learned BastionRL defense, economy, and repairs.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--steps", type=int, default=2000000)
    training.add_argument("--seed", type=int, help="Fresh default: 0; resume preserves the saved seed")
    training.add_argument("--output", type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--strategy", choices=("flat", "hierarchical"),
                          help="Fresh default: hierarchical; resume keeps the saved policy unless specified")
    training.add_argument("--n-steps", type=int, help="Rollout steps per environment; fresh default: 256")
    training.add_argument("--batch-size", type=int, help="PPO minibatch size; fresh default: 64")
    training.add_argument("--n-epochs", type=int, help="Maximum PPO epochs per rollout; fresh default: 10")
    training.add_argument("--eval-freq", type=int, default=10000)
    training.add_argument("--eval-episodes", type=int, default=20)
    training.add_argument("--eval-seed", type=int, default=10000)
    training.add_argument("--n-envs", type=int, default=1,
                          help="Independent CPU environments; use multiple workers on a server")
    training.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    training.add_argument("--ent-coef", type=float,
                          help="Fresh default: 0.02; on resume keep the saved value unless specified")
    training.add_argument("--learning-rate", type=float,
                          help="Fresh default: 0.0003; on resume keep the saved value unless specified")
    training.add_argument("--lr-final", type=float, help="Linearly anneal learning rate to this endpoint")
    training.add_argument("--ent-final", type=float, help="Linearly anneal entropy coefficient to this endpoint")
    training.add_argument("--curriculum", choices=("staged", "mixed"),
                          help="Mixed: 60%% current, 20%% easier, 20%% full-task episodes")
    training.add_argument("--promotion-successes", type=int, help="Successes required in a 50-episode window")
    training.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None,
                          help="Use VecNormalize for observations and rewards; resume preserves saved setting")
    evaluation = commands.add_parser("evaluate")
    policy = evaluation.add_mutually_exclusive_group(required=True)
    policy.add_argument("--model", type=Path)
    policy.add_argument("--policy", choices=("random",), help="Random legal-action sanity baseline")
    evaluation.add_argument("--episodes", type=int, default=100)
    evaluation.add_argument("--seed", type=int, default=30000)
    evaluation.add_argument("--output", type=Path)
    diagnostic = commands.add_parser("diagnose", help="Scripted winnability check; separate from RL")
    diagnostic.add_argument("--episodes", type=int, default=10)
    diagnostic.add_argument("--seed", type=int, default=80000)
    diagnostic.add_argument("--economy", action="store_true", help="Also try one early resource tower")
    diagnostic.add_argument("--output", type=Path)
    for command in (training, evaluation, diagnostic):
        command.add_argument("--difficulty", choices=("baseline", "medium", "hard"), default="baseline")
        command.add_argument("--max-steps", type=int, default=2400,
                             help="Episode horizon in half-second units; forced waits can span several units")
    args = parser.parse_args()
    for field in ("steps", "eval_freq", "eval_episodes", "episodes", "max_steps", "n_envs",
                  "n_steps", "batch_size", "n_epochs"):
        if getattr(args, field, None) is not None and getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.command == "train":
        for name in ("ent_coef", "ent_final"):
            value = getattr(args, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
        for name in ("learning_rate", "lr_final"):
            value = getattr(args, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                parser.error(f"--{name.replace('_', '-')} must be finite and positive")
        if args.promotion_successes is not None and not 1 <= args.promotion_successes <= 50:
            parser.error("--promotion-successes must be between 1 and 50")
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
