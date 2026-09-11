"""Training-only, full-state value baseline for the original zero-sum game.

This module is deliberately separate from both actor implementations. Its scene
state must never enter actor observations, recurrence, action sampling, or an
exported browser policy. It predicts value; it does not choose actions or change
the visibility reward. Existing training runs do not import this prototype.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

SCHEMA = 'hide-seek-central-value-v1'
MAX_OBJECTS = 10
MAX_WALLS = 16
FEATURES = {'global': (8,), 'agents': (2, 14),
            'objects': (MAX_OBJECTS, 21), 'objectMask': (MAX_OBJECTS,),
            'walls': (MAX_WALLS, 8), 'wallMask': (MAX_WALLS,)}


def central_state(env, buttons=None):
    """Read the *current* physical state without stepping or mutating the world.

    Positions and velocities use fixed physical scales, not fitted statistics.
    No future actions, episode returns, search, or construction targets appear.
    Object identity is represented by attributes, never by an object index.
    Requested controller buttons are Markov state for the persistent-button
    pilot; the binary-action baseline can omit them.
    """
    arena = env.arena
    if len(arena['objects']) > MAX_OBJECTS or len(arena['walls']) > MAX_WALLS:
        raise ValueError('Scene exceeds the declared central-critic schema')
    buttons = np.zeros((2, 2), np.float32) if buttons is None else np.asarray(buttons)
    if buttons.shape != (2, 2) or not np.isin(buttons, [0, 1]).all():
        raise ValueError('Expected two binary requested buttons for each agent')
    state = {key: np.zeros(shape, np.float32) for key, shape in FEATURES.items()}
    width, height = arena['width'], arena['height']
    position_scale = np.array([width, height, 2.0])
    velocity_scale = np.array([5.0, 5.0, 5.0, 8.0])
    state['global'][:] = [width / 12, height / 12,
        min(1, env.t / max(1, env.prep)),
        max(0, env.t - env.prep) / max(1, env.play),
        float(env.t < env.prep), float(env.done),
        float(env.visible), float(env.seen[0, 1])]
    for role in range(2):
        offset = role * 4
        yaw = env.data.qpos[offset + 3]
        grip = env.grips[role]
        state['agents'][role] = [
            *(env.data.qpos[offset:offset + 3] / position_scale),
            *(env.data.qvel[offset:offset + 4] / velocity_scale),
            math.cos(yaw), math.sin(yaw), float(grip >= 0),
            float(grip >= 0 and env.locks[grip] >= 0), *buttons[role], float(role)]
    for index, obj in enumerate(arena['objects']):
        offset = 8 + index * 4
        yaw = env.data.qpos[offset + 3]
        lock = env.locks[index]
        state['objects'][index] = [
            *(env.data.qpos[offset:offset + 3] / position_scale),
            *(env.data.qvel[offset:offset + 4] / velocity_scale),
            math.cos(yaw), math.sin(yaw), *(np.array(obj['size']) / 2),
            obj['mass'] / 2,
            *[float(obj['kind'] == kind) for kind in ('box', 'plank', 'ramp')],
            *[float(lock == owner) for owner in (0, 1, 2)],
            *[float(env.grips[owner] == index) for owner in (0, 1)]]
        state['objectMask'][index] = 1
    for index, wall in enumerate(arena['walls']):
        yaw = wall.get('yaw', 0)
        state['walls'][index] = [*(np.array(wall['position']) / position_scale),
            *(np.array(wall['size']) / position_scale), math.cos(yaw), math.sin(yaw)]
        state['wallMask'][index] = 1
    if not all(np.isfinite(value).all() for value in state.values()):
        raise FloatingPointError('Nonfinite central-critic state')
    return state


def batch_states(states):
    if not states:
        raise ValueError('At least one physical state is required')
    for state in states:
        if set(state) != set(FEATURES) or any(
                np.shape(state[key]) != shape for key, shape in FEATURES.items()):
            raise ValueError('Central-critic state schema mismatch')
    return {key: torch.from_numpy(np.stack([state[key] for state in states]).astype(np.float32))
            for key in FEATURES}


class CentralCritic(nn.Module):
    """A separate permutation-invariant critic, with exact zero-sum values.

    Shared object/wall encoders and masked mean/max pooling avoid assigning
    meaning to the nearest-object slot order used by the actor. A single value
    predicts the hider's discounted return; the seeker's return is its negative
    under this environment's identical horizons and opposite visibility reward.
    """
    def __init__(self, use_actor_memory=False, memory_size=64):
        super().__init__()
        self.use_actor_memory = bool(use_actor_memory)
        self.memory_size = memory_size
        self.objects = nn.Sequential(nn.Linear(21, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.walls = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.value = nn.Sequential(nn.Linear(228 + (2 * memory_size if self.use_actor_memory else 0), 128), nn.Tanh(),
                                   nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value[-1].weight, 1)

    @staticmethod
    def pool(embedding, mask):
        weights = mask[..., None]
        count = weights.sum(-2)
        mean = (embedding * weights).sum(-2) / count.clamp_min(1)
        maximum = embedding.masked_fill(weights == 0, -1e9).amax(-2)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), -1)

    def forward(self, state, actor_memories=None):
        objects = self.pool(self.objects(state['objects']), state['objectMask'])
        walls = self.pool(self.walls(state['walls']), state['wallMask'])
        features = torch.cat((state['global'], state['agents'].flatten(-2), objects, walls), -1)
        if self.use_actor_memory:
            if actor_memories is None or actor_memories.shape != (*features.shape[:-1], 2, self.memory_size):
                raise ValueError(f'Expected both pre-action actor memories with shape [..., 2, {self.memory_size}]')
            # Value fitting must never backpropagate privileged scene information
            # through actor recurrence, even if the caller forgets to detach.
            features = torch.cat((features, actor_memories.detach().flatten(-2)), -1)
        return self.value(features).squeeze(-1)

    def role_values(self, state, actor_memories=None):
        value = self(state, actor_memories)
        return torch.stack((value, -value), -1)
