"""A learned action kind followed by a learned legal target, trained jointly by PPO."""

import torch
from sb3_contrib.common.maskable.distributions import (
    MaskableCategorical,
    MaskableCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn


class HierarchicalDistribution(MaskableCategoricalDistribution):
    def __init__(self, group_sizes):
        super().__init__(sum(group_sizes))
        self.group_sizes = tuple(group_sizes)

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, len(self.group_sizes) + self.action_dim)

    def proba_distribution(self, action_logits):
        self.kind_logits = action_logits[:, :len(self.group_sizes)]
        self.target_logits = action_logits[:, len(self.group_sizes):]
        self.apply_masking(None)
        return self

    def apply_masking(self, masks):
        mask = (torch.ones_like(self.target_logits, dtype=torch.bool) if masks is None else
                torch.as_tensor(masks, dtype=torch.bool, device=self.target_logits.device)
                .reshape(self.target_logits.shape))
        group_masks = mask.split(self.group_sizes, dim=-1)
        self.kinds = MaskableCategorical(
            logits=self.kind_logits, masks=torch.stack([group.any(-1) for group in group_masks], dim=-1),
        )
        self.targets = [MaskableCategorical(logits=logits, masks=group_mask)
                        for logits, group_mask in zip(self.target_logits.split(self.group_sizes, dim=-1),
                                                      group_masks, strict=True)]
        joint_logits = torch.cat([self.kinds.logits[:, index:index + 1] + target.logits
                                  for index, target in enumerate(self.targets)], dim=-1)
        self.distribution = MaskableCategorical(logits=joint_logits, masks=mask)

    def mode(self):
        kind = self.kinds.probs.argmax(dim=-1)
        choices, offset = [], 0
        for target, size in zip(self.targets, self.group_sizes, strict=True):
            choices.append(target.probs.argmax(dim=-1) + offset)
            offset += size
        return torch.stack(choices, dim=-1).gather(1, kind[:, None]).squeeze(1)


class HierarchicalPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, action_group_sizes, **kwargs):
        if not action_group_sizes or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0
                                         for size in action_group_sizes):
            raise ValueError("Action group sizes must be positive integers.")
        self.action_group_sizes = tuple(action_group_sizes)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        if sum(self.action_group_sizes) != self.action_space.n:
            raise ValueError("Action group sizes must cover the action space.")
        self.action_dist = HierarchicalDistribution(self.action_group_sizes)
        super()._build(lr_schedule)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(), action_group_sizes=self.action_group_sizes)
