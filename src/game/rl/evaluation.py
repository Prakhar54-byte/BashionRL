"""Seeded, full-task evaluation shared by training and the CLI."""

import numpy as np

from src.game.rl.checkpoint import policy_observation


def play_episodes(env, model, seeds, *, diagnostic_policy=None):
    if model is not None and diagnostic_policy is not None:
        raise ValueError("A scripted diagnostic cannot be combined with an RL model.")
    episodes = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        total_reward = 0.0
        diagnostics = {name: {"legal_decisions": 0, "chosen": 0, "probability_sum": 0.0}
                       for name in env.action_groups}
        while True:
            mask = env.action_masks()
            if diagnostic_policy is not None:
                action = int(diagnostic_policy(env, mask))
                if not env.action_space.contains(action) or not mask[action]:
                    raise ValueError("Diagnostic selected an illegal action.")
                probabilities = np.zeros(len(mask))
                probabilities[action] = 1.0
            elif model is None:
                probabilities = mask / mask.sum()
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                import torch

                model.policy.set_training_mode(False)
                tensor, _ = model.policy.obs_to_tensor(policy_observation(model, observation))
                with torch.no_grad():
                    distribution = model.policy.get_distribution(
                        tensor, action_masks=mask,
                    )
                    probabilities = distribution.distribution.probs.cpu().numpy()[0]
                    action = int(distribution.get_actions(deterministic=True).cpu().item())
            for name, group in env.action_groups.items():
                if mask[group].any():
                    diagnostics[name]["legal_decisions"] += 1
                    diagnostics[name]["chosen"] += int(group.start <= action < group.stop)
                    diagnostics[name]["probability_sum"] += float(probabilities[group].sum())
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                episodes.append(dict(info, seed=seed, reward=total_reward,
                                     steps=env.steps, truncated=truncated,
                                     action_diagnostics=diagnostics))
                break
    return episodes


def summarize_episodes(episodes):
    metrics = {
        "mean_episode_reward": "reward", "mean_waves_reached": "waves_reached",
        "mean_waves_completed": "waves_completed", "mean_base_health": "base_health",
        "mean_leaks": "leaks", "mean_kills": "kills", "mean_spent": "spent",
        "mean_repairs_started": "repairs_started", "mean_resources_built": "resources_built",
        "win_rate": "is_success", "survival_rate": "survived", "truncation_rate": "truncated",
    }
    result = {name: float(np.mean([episode[key] for episode in episodes]))
              for name, key in metrics.items()}
    for key in episodes[0]:
        if key.startswith("reward_") and key != "reward_terms":
            result[f"mean_{key}"] = float(np.mean([episode[key] for episode in episodes]))
    result["std_episode_reward"] = float(np.std([episode["reward"] for episode in episodes]))
    for key in ("decisions_with_other_action_legal", "wait_when_other_action_legal",
                "forced_wait_decisions", "condensed_wait_ticks"):
        result[key] = sum(episode[key] for episode in episodes)
    choices = result["decisions_with_other_action_legal"]
    result["wait_rate_when_other_action_legal"] = (
        result["wait_when_other_action_legal"] / choices if choices else None
    )
    result["action_diagnostics"] = {}
    for name in episodes[0]["action_diagnostics"]:
        totals = {key: sum(episode["action_diagnostics"][name][key] for episode in episodes)
                  for key in ("legal_decisions", "chosen", "probability_sum")}
        count = totals["legal_decisions"]
        result["action_diagnostics"][name] = {
            "legal_decisions": count, "chosen": totals["chosen"],
            "mean_probability_when_legal": totals["probability_sum"] / count if count else None,
            "selection_rate_when_legal": totals["chosen"] / count if count else None,
        }
    return result
