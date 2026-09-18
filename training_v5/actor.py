"""Original recurrent actor/critic for the physical hide-and-seek arena.

The actor gets only the observation assembled by the environment: masked
entities, range sensors and its own state. No path planner or expert actions.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Normal


class PhysicalActor(nn.Module):
    def __init__(self, observation_size, hidden_size=64):
        super().__init__()
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(observation_size, 96)
        self.memory = nn.GRUCell(96, hidden_size)
        self.movement = nn.Linear(hidden_size, 3)
        self.tools = nn.Linear(hidden_size, 2)
        self.value = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((3,), -0.6))
        for layer in (self.encoder, self.movement, self.tools, self.value):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.movement.weight, 0.01)
        nn.init.orthogonal_(self.tools.weight, 0.01)

    def forward(self, observation, memory):
        encoded = torch.tanh(self.encoder(observation))
        memory = self.memory(encoded, memory)
        normal = Normal(self.movement(memory), self.log_std.clamp(-2.5, 0.3).exp())
        tools = Bernoulli(logits=self.tools(memory))
        return normal, tools, self.value(memory).squeeze(-1), memory

    @staticmethod
    def statistics(normal, tools, raw_action, include_entropy=True):
        raw_movement = raw_action[..., :3]
        # Stable log|d tanh(u)/du| even when float32 tanh rounds to +/-1.
        def log_jacobian(value):
            return 2 * (math.log(2) - value - torch.nn.functional.softplus(-2 * value))
        logp = (normal.log_prob(raw_movement) - log_jacobian(raw_movement)).sum(-1)
        logp += tools.log_prob(raw_action[..., 3:]).sum(-1)
        entropy = None
        if include_entropy:
            # Entropy belongs to the bounded force distribution, not its raw
            # Gaussian. Unsquashed entropy rewards variance even when a large
            # mean makes every applied force indistinguishably saturate.
            entropy = (normal.entropy() + log_jacobian(normal.rsample())).sum(-1)
            entropy += tools.entropy().sum(-1)
        return logp, entropy

    def act(self, observation, memory, deterministic=False):
        normal, tools, value, next_memory = self(observation, memory)
        movement = normal.mean if deterministic else normal.sample()
        bits = (tools.probs >= 0.5).float() if deterministic else tools.sample()
        raw_action = torch.cat((movement, bits), -1)
        action = torch.cat((torch.tanh(movement), bits), -1)
        logp, _ = self.statistics(normal, tools, raw_action, include_entropy=False)
        return action, raw_action, logp, value, next_memory


def export_actor(model):
    return {
        'observationSize': model.observation_size,
        'hiddenSize': model.hidden_size,
        'encoderSize': 96,
        'weights': {key: value.detach().cpu().numpy().astype(np.float32).tolist()
                    for key, value in model.state_dict().items() if not key.startswith('value.')},
    }
