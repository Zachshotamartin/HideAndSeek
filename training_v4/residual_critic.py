"""Training-only zero-initialized correction to detached partial value estimates."""
import torch
from torch import nn
from central_critic import CentralCritic

SCHEMA = 'hide-seek-residual-central-value-v1'


class ResidualCentralCritic(nn.Module):
    def __init__(self, dropout=.1, memory_size=64):
        super().__init__()
        self.dropout = float(dropout)
        self.correction = CentralCritic(use_actor_memory=True, memory_size=memory_size)
        original = self.correction.value
        self.correction.value = nn.Sequential(
            original[0], original[1], nn.Dropout(dropout),
            original[2], original[3], nn.Dropout(dropout), original[4])
        nn.init.zeros_(self.correction.value[-1].weight)
        nn.init.zeros_(self.correction.value[-1].bias)

    @staticmethod
    def baseline(partial_values):
        if partial_values.shape[-1] != 2:
            raise ValueError('Expected both same-pre-action partial value estimates')
        return .5 * (partial_values[..., 0] - partial_values[..., 1]).detach()

    def residual(self, state, actor_memories):
        return self.correction(state, actor_memories)

    def forward(self, state, actor_memories, partial_values):
        return self.baseline(partial_values) + self.residual(state, actor_memories)

    def role_values(self, state, actor_memories, partial_values):
        value = self(state, actor_memories, partial_values)
        return torch.stack((value, -value), -1)
