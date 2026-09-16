"""Frozen action sampling from persistent_evaluate without evaluation dependencies."""
import math
import numpy as np
import torch
from persistent_actor import PersistentActor, advance_buttons, augment
TAPE_LENGTH=10

def is_blind(physical):
    """A seeker in the preparation phase sees nothing and may not act."""
    return physical[7] < .5 and physical[5] < 1


def gaussian(tape, axis):
    """Box-Muller sample from two uniform tape entries."""
    return math.sqrt(-2 * math.log(max(1e-12, tape[axis * 2]))) * math.cos(2 * math.pi * tape[axis * 2 + 1])


def sample(model, physical, memory, buttons, generator, deterministic=False):
    """One stochastic action from a seeded uniform tape; identical tapes across modes."""
    persistent = isinstance(model, PersistentActor)
    blind = is_blind(physical)
    if blind:
        buttons = np.zeros(2, np.float32)
    observation = augment(physical, buttons) if persistent else physical
    normal, tools, _, memory = model(torch.from_numpy(observation[None]), memory)
    tape = generator.random(TAPE_LENGTH)
    action = np.zeros(6, np.float32)
    for axis in range(4):
        action[axis if axis < 3 else 5] = math.tanh(float(normal.mean[0, axis]) + (0 if deterministic else float(normal.scale[0, axis]) * gaussian(tape, axis)))
    if persistent:
        probabilities = tools.probs[0].numpy()
        commands = (probabilities.argmax(-1) if deterministic else np.asarray([min(2, np.searchsorted(np.cumsum(probabilities[t]), tape[8 + t], side='right')) for t in range(2)]))
        buttons = advance_buttons(buttons, commands, blind)
    else:
        buttons = ((.5 if deterministic else tape[8:]) < tools.probs[0].numpy()).astype(np.float32)
        commands = np.where(buttons, 1, 2)
    action[3:5] = buttons
    if blind:
        action.fill(0)
        buttons.fill(0)
    return action, memory, buttons, commands

