import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from sb3_contrib import MaskablePPO

from src.game.rl.__main__ import main
from src.game.rl.checkpoint import load_model, save_metadata
from src.game.rl.environment import BastionEnv
from src.game.rl.evaluation import play_episodes
from src.game.rl.policy import HierarchicalDistribution, HierarchicalPolicy


def test_hierarchical_decision_avoids_flat_argmax_wait_and_evaluation_matches():
    env = BastionEnv(max_steps=1)
    model = MaskablePPO(HierarchicalPolicy, env, n_steps=8, batch_size=8, seed=0,
                        policy_kwargs={"action_group_sizes": [1, 92, 6, 3]})
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.bias[:4] = torch.tensor([0.3, 0.6, 0.05, 0.05]).log()
    obs, _ = env.reset(seed=0)
    tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(tensor, action_masks=env.action_masks())
        probabilities = dist.distribution.probs[0]
        assert probabilities[1:93].sum() > probabilities[0]
        assert probabilities.argmax().item() == 0
        assert dist.mode().item() == 1
    action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
    assert int(action) == 1
    episode = play_episodes(env, model, [0])[0]
    assert episode["action_diagnostics"]["defense"]["chosen"] == 1
    assert episode["action_diagnostics"]["wait"]["chosen"] == 0


def test_hierarchical_joint_probabilities_entropy_masks_and_gradients():
    torch.manual_seed(0)
    logits = torch.randn(2, 11, requires_grad=True)
    masks = torch.tensor([[True, False, True, True, False, True, False],
                          [True, False, False, False, False, False, False]])
    dist = HierarchicalDistribution([1, 3, 2, 1]).proba_distribution(logits)
    dist.apply_masking(masks)
    probs = dist.distribution.probs
    torch.testing.assert_close(probs.sum(-1), torch.ones(2))
    assert (probs[~masks] == 0).all()
    expected = torch.cat([dist.kinds.probs[:, i:i + 1] * target.probs
                          for i, target in enumerate(dist.targets)], dim=-1)
    torch.testing.assert_close(probs, expected)
    expected_entropy = dist.kinds.entropy() + sum(
        dist.kinds.probs[:, i] * target.entropy() for i, target in enumerate(dist.targets)
    )
    torch.testing.assert_close(dist.entropy(), expected_entropy)
    for _ in range(32):
        actions = dist.sample()
        assert masks.gather(1, actions[:, None]).all()
        torch.testing.assert_close(dist.log_prob(actions), probs.gather(1, actions[:, None]).squeeze(1).log())
    assert dist.mode()[1].item() == 0
    loss = -dist.log_prob(torch.tensor([2, 0])).mean() - 0.02 * dist.entropy().mean()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, :4].abs().sum() > 0
    assert logits.grad[0, 4:].abs().sum() > 0


def test_hierarchical_ppo_updates_and_round_trips(tmp_path):
    env = BastionEnv(max_steps=2)
    model = MaskablePPO(HierarchicalPolicy, env, n_steps=8, batch_size=8, n_epochs=1, seed=0,
                        policy_kwargs={"action_group_sizes": [1, 92, 6, 3], "net_arch": [16]})
    initial = model.policy.action_net.weight.detach().clone()
    model.learn(16)
    assert not torch.equal(initial, model.policy.action_net.weight)
    path = tmp_path / "hierarchical.zip"
    model.save(path)
    save_metadata(path, env, seed=0, steps=16)
    restored = load_model(path, env)
    assert isinstance(restored.policy, HierarchicalPolicy)
    obs, _ = env.reset(seed=42)
    for deterministic in (False, True):
        action, _ = restored.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
        assert env.action_masks()[int(action)]
    tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        first = model.policy.get_distribution(tensor, env.action_masks()).distribution.probs
        second = restored.policy.get_distribution(tensor, env.action_masks()).distribution.probs
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("sizes", [[1, 92, 6, 0], [1, 92, 6, 3.0], [1, 2]])
def test_hierarchical_policy_rejects_invalid_group_sizes(sizes):
    with pytest.raises(ValueError, match="Action group sizes"):
        MaskablePPO(HierarchicalPolicy, BastionEnv(), policy_kwargs={"action_group_sizes": sizes})


def test_parallel_training_evaluation_checkpoint_and_worker_count_resume(tmp_path, monkeypatch):
    output = tmp_path / "parallel"
    monkeypatch.setattr("sys.argv", ["rl", "train", "--n-envs", "2", "--steps", "512",
                                    "--max-steps", "2", "--eval-freq", "256", "--eval-episodes", "1",
                                    "--output", str(output)])
    main()
    metadata = json.loads((output / "final.json").read_text())
    assert metadata["training_parameters"]["n_envs"] == 2
    assert len(list((output / "episodes").glob("*.monitor.csv"))) == 2
    history = json.loads((output / "evaluations.json").read_text())
    assert [item["timesteps"] for item in history] == [512]
    assert all(item["results"][0]["wave_goal"] == 10 for item in history)
    resumed = tmp_path / "resumed"
    monkeypatch.setattr("sys.argv", ["rl", "train", "--resume", str(output / "latest.zip"),
                                    "--steps", "256", "--max-steps", "2", "--eval-episodes", "1",
                                    "--output", str(resumed)])
    main()
    model = load_model(resumed / "final.zip", BastionEnv())
    assert model.num_timesteps == 768 and model.n_envs == 1


def test_training_rejects_flat_resume_but_keeps_checkpoint_evaluable(tmp_path, monkeypatch):
    env = BastionEnv(max_steps=1)
    model = MaskablePPO("MlpPolicy", env, n_steps=8, batch_size=8)
    path = tmp_path / "flat.zip"
    model.save(path)
    save_metadata(path, env, seed=0, steps=0)
    assert play_episodes(env, load_model(path, env), [42])
    monkeypatch.setattr("sys.argv", ["rl", "train", "--resume", str(path),
                                    "--strategy", "hierarchical",
                                    "--output", str(tmp_path / "fresh")])
    with pytest.raises(ValueError, match="start fresh"):
        main()


def test_training_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("sys.argv", ["rl", "train", "--device", "cuda"])
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        main()
