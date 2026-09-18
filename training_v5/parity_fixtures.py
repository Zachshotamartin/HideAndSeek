"""Seeded native decisions of a v6 actor pair on real physics, replayed by the browser parity tests.

The rollout closes the loop through the physics so the observations, the
last-seen memory and the clock are realistic, and every decision records the
uniform tape it consumed, the noise it carried in and the noise it carries out.
"""
import numpy as np
import torch

from game import NOISE, observe
from persistent_evaluate import sample
from physics import PhysicsEnv

STEPS = 132
PREP = 24
RESET_TICKS = (44, 88)
ARENA = dict(seed=1723, scenario='rooms', size=8, n_boxes=3, n_ramps=1)


class Tape:
    """Uniform tape recorded so the browser can replay the identical draws."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.values = []

    def random(self, count):
        values = self.rng.random(count)
        self.values = values.tolist()
        return values


def rollout(actors, deterministic, tape_seed):
    """One closed-loop episode: rows of observation, tape, carried noise and the expected decision."""
    play = STEPS - PREP
    env = PhysicsEnv(**ARENA, prep=PREP, play=play)
    physical = observe(env)
    memory = [torch.zeros(1, actor.hidden_size) for actor in actors]
    buttons = np.zeros((2, 2), np.float32)
    noises = np.zeros((2, NOISE), np.float32)
    tapes = [Tape(tape_seed + role) for role in range(2)]
    rows = []
    counts = dict(rows=0, blind=0, resets=0)
    with torch.no_grad():
        for tick in range(STEPS):
            actions = np.zeros((2, 6), np.float32)
            for role, actor in enumerate(actors):
                reset = tick in RESET_TICKS
                if reset:
                    memory[role].zero_()
                    buttons[role] = 0
                    noises[role] = 0
                    counts['resets'] += 1
                row = dict(role=role, reset=reset, observation=physical[role].tolist(), noiseIn=noises[role].tolist())
                actions[role], memory[role], buttons[role], commands, noises[role] = sample(
                    actor, physical[role], memory[role], buttons[role], tapes[role], deterministic, noises[role])
                row.update(tape=tapes[role].values, action=actions[role].tolist(), memory=memory[role][0].tolist(),
                           mean=actor.movement(memory[role])[0].tolist(), toolLogits=actor.tools(memory[role])[0].tolist(),
                           commands=[int(x) for x in commands], buttons=buttons[role].tolist(), noiseOut=noises[role].tolist())
                rows.append(row)
                counts['rows'] += 1
                counts['blind'] += int(physical[role][7] < .5 and physical[role][5] < 1)
            env.step(actions)
            physical = observe(env)
    env.close()
    return dict(deterministic=deterministic, rows=rows), counts


def policy_rollouts(actors):
    """Sampled and deterministic episodes with mid-rollout resets and blind preparation rows."""
    rollouts = []
    counts = dict(rows=0, blind=0, resets=0)
    for deterministic in (False, True):
        episode, episode_counts = rollout([actor.eval() for actor in actors], deterministic, 500 + 10 * deterministic)
        rollouts.append(episode)
        for key in counts:
            counts[key] += episode_counts[key]
    return rollouts, counts
