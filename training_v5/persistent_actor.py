"""Original, isolated PPO pilot with generic persistent button commands.

Physics remains unchanged. Keep/Press/Release are commands for a requested
button state, not object-specific instructions. Uniform initial logits have
no preferred command. The actor and critic both observe their two button states.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

PHYSICS_OBSERVATIONS = 208
OBSERVATIONS = 210
COMMANDS = ('keep', 'press', 'release')
FORMAT = 'original-mujoco-persistent-buttons-ppo-v1'


def advance_buttons(previous, commands, blind=False):
    previous = np.asarray(previous, dtype=np.float32)
    commands = np.asarray(commands)
    if previous.shape != commands.shape or previous.shape[-1] != 2:
        raise ValueError('Two previous button states and two commands are required')
    if not np.isin(previous, [0, 1]).all() or not np.isin(commands, [0, 1, 2]).all():
        raise ValueError('Button states must be binary; commands are Keep/Press/Release')
    result = np.where(commands == 1, 1, np.where(commands == 2, 0, previous)).astype(np.float32)
    result[np.asarray(blind, dtype=bool)] = 0
    return result


def augment(observations, buttons, noise=None, physical_size=PHYSICS_OBSERVATIONS):
    """Actor input: the first ``physical_size`` measurements, the two own buttons, then any observed noise.

    A worker may return more measurements than an older actor was trained on;
    each actor takes the prefix its schema defines.
    """
    observations = np.asarray(observations, dtype=np.float32)
    buttons = np.asarray(buttons, dtype=np.float32)
    if observations.shape[:-1] != buttons.shape[:-1] or observations.shape[-1] < physical_size or buttons.shape[-1] != 2:
        raise ValueError(f'Expected at least {physical_size} physical measurements and two own button states')
    parts = [observations[..., :physical_size], buttons]
    if noise is not None:
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape[:-1] != buttons.shape[:-1] or noise.shape[-1] != 4:
            raise ValueError('Expected four observed noise values per actor')
        parts.append(noise)
    return np.concatenate(parts, axis=-1)


class PersistentActor(nn.Module):
    def __init__(self, hidden_size=64, encoder_size=96, observation_size=OBSERVATIONS, noise_rho=0.):
        super().__init__()
        self.observation_size, self.hidden_size = int(observation_size), hidden_size
        self.encoder_size = encoder_size
        # v6 actors observe their own AR(1) exploration noise in the last four
        # columns; the per-step distribution conditions on it, so the likelihood
        # stays exact while exploration is coherent over a fraction of a second.
        if not 0 <= noise_rho < 1:
            raise ValueError('The noise correlation must lie in [0, 1)')
        self.noise_rho = float(noise_rho)
        self.physical_size = self.observation_size - 2 - (4 if self.noise_rho else 0)
        self.encoder = nn.Linear(self.observation_size, encoder_size)
        self.memory = nn.GRUCell(encoder_size, hidden_size)
        self.movement = nn.Linear(hidden_size, 4)
        self.tools = nn.Linear(hidden_size, 6)
        self.value = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((4,), -.6))
        for layer in (self.encoder, self.movement, self.value):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.movement.weight, .01)
        nn.init.zeros_(self.tools.weight)
        nn.init.zeros_(self.tools.bias)

    def warm_start(self, source):
        """Preserve all compatible own learned parameters; reset the new head.

        Two new controller columns are exactly zero. The value head is retained
        because the physical game and reward are identical. No optimizer state
        is accepted here. A zero-experience source stays a genuine initial model.
        """
        if source['encoder.weight'].shape != (96, PHYSICS_OBSERVATIONS):
            raise ValueError('Warm start requires the original 208-observation architecture')
        with torch.no_grad():
            for name, target in self.state_dict().items():
                if name.startswith('tools.'):
                    target.zero_()
                elif name == 'encoder.weight':
                    target.zero_()
                    target[:, :208].copy_(source[name])
                else:
                    target.copy_(source[name])
        return self

    def forward(self, observation, memory):
        memory = self.memory(torch.tanh(self.encoder(observation)), memory)
        base = self.movement(memory)
        scale = self.log_std.clamp(-2.5, .3).exp()
        if self.noise_rho:
            base = base + self.noise_rho * observation[..., -4:]
            scale = scale * math.sqrt(1 - self.noise_rho ** 2)
        normal = Normal(base, scale)
        tools = Categorical(logits=self.tools(memory).reshape(*memory.shape[:-1], 2, 3))
        return normal, tools, self.value(memory).squeeze(-1), memory

    @staticmethod
    def statistics(normal, tools, raw_action, include_entropy=True):
        movement, commands = raw_action[..., :4], raw_action[..., 4:].long()
        log_jacobian = lambda value: 2 * (math.log(2) - value - nn.functional.softplus(-2 * value))
        logp = (normal.log_prob(movement) - log_jacobian(movement)).sum(-1)
        logp += tools.log_prob(commands).sum(-1)
        entropy = None
        if include_entropy:
            entropy = (normal.entropy() + log_jacobian(normal.rsample())).sum(-1)
            entropy += 0.1 * tools.entropy().sum(-1)
        return logp, entropy

    def act_with_noise(self, observation, memory, deterministic=False):
        """Sample an action; also return the exploration noise carried into the next step."""
        normal, tools, value, next_memory = self(observation, memory)
        carried = self.noise_rho * observation[..., -4:] if self.noise_rho else torch.zeros_like(normal.mean)
        base = normal.mean - carried
        # Deterministic playback carries no noise: the head's own mean.
        movement = base if deterministic else normal.sample()
        commands = tools.logits.argmax(-1) if deterministic else tools.sample()
        raw = torch.cat((movement, commands.float()), -1)
        logp, _ = self.statistics(normal, tools, raw, include_entropy=False)
        noise = (movement - base) if self.noise_rho else torch.zeros_like(movement)
        return torch.tanh(movement), commands, raw, logp, value, next_memory, noise

    def act(self, observation, memory, deterministic=False):
        return self.act_with_noise(observation, memory, deterministic)[:6]

    def observe(self, physical, buttons, noise=None):
        """This actor's own input from a worker observation that may be wider than its schema."""
        if self.noise_rho and noise is None:
            raise ValueError('This actor observes its own exploration noise; pass the carried noise')
        return augment(physical, buttons, noise if self.noise_rho else None, self.physical_size)


def export_actor(model):
    return {'observationSize': model.observation_size, 'physicsObservationSize': model.physical_size,
            'hiddenSize': model.hidden_size, 'encoderSize': model.encoder_size,
            'weights': {key: value.detach().cpu().numpy().astype(np.float32).tolist()
                        for key, value in model.state_dict().items() if not key.startswith('value.')}}
