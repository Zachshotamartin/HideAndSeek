"""Native truth for the browser's tag-round runtime, from the training_v5 modules.

Writes tests/fixtures/tag-rounds-native.json: a physics rollout with the
last-seen memory, the six extra observations and the capture flag per frame,
plus seeded decisions of a synthetic v6 actor pair that observes its own
exploration noise. Nothing here loads or changes training checkpoints.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training_v5'))
from capture import captured                                     # noqa: E402
from entity_actor import EntityActor, TAG_FORMAT, export_actor   # noqa: E402
from game import NOISE, PHYSICAL_OBSERVATIONS, SCHEMA, extras, observe, preparation_steps  # noqa: E402
from persistent_evaluate import sample                           # noqa: E402
from physics import PhysicsEnv, generate_arena                   # noqa: E402

torch.set_num_threads(1)
SOURCES = ['physics.py', 'game.py', 'capture.py', 'entity_actor.py', 'persistent_actor.py', 'persistent_evaluate.py']
STEPS = 132
PREP = 24
RESET_TICKS = (44, 88)
NOISE_RHO = .7
OUTPUT = ROOT / 'tests/fixtures/tag-rounds-native.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tape:
    """Uniform tape recorded so the browser can replay the identical draws."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.values = []

    def random(self, count):
        values = self.rng.random(count)
        self.values = values.tolist()
        return values


def scripted_action(tick, caught):
    """The hider creeps forward and turns; the seeker advances along the line until the tag, then wanders."""
    hider = [.3 if tick < 16 else 0, 0, .35 if 22 <= tick < 40 else 0, int(tick < 18), int(12 <= tick < 26), 0]
    seeker = [.8 if not caught else -.3, 0, 0 if not caught else .3, 0, 0, int(tick == 50)]
    return np.array([hider, seeker], dtype=float)


def physics_rollout():
    arena = generate_arena(923, 'open', 8, 2, 1)
    arena['agents'] = [dict(position=[2, 3, .25], yaw=0), dict(position=[6, 3, .25], yaw=math.pi)]
    for prop in arena['objects']:
        # Keep the props off the approach line so the scripted rollout reaches a tag on any layout.
        prop['position'][1] = max(prop['position'][1], 5.5)
    env = PhysicsEnv(arena=arena, prep=12, play=52)
    frames, actions = [], []
    caught = False
    for tick in range(64):
        action = scripted_action(tick, caught)
        env.step(action)
        actions.append(action.tolist())
        frames.append(dict(t=env.t, qpos=env.data.qpos.tolist(), qvel=env.data.qvel.tolist(),
                           lastSeen=[None if seen is None else dict(position=seen['position'], t=seen['t']) for seen in env.last_seen],
                           extras=extras(env).tolist(), captured=bool(captured(env)), observation=observe(env).tolist()))
        caught = caught or bool(captured(env))
    assert any(frame['lastSeen'][1] is not None for frame in frames), 'the seeker must see the hider at least once'
    assert any(frame['captured'] for frame in frames), 'the rollout must reach a capture'
    return dict(arena=arena, prep=12, play=52, actions=actions, frames=frames)


def synthetic_pair():
    with torch.random.fork_rng():
        torch.manual_seed(931772)
        actors = [EntityActor(64, 96, 32, PHYSICAL_OBSERVATIONS + 2 + NOISE, NOISE_RHO) for _ in range(2)]
        for actor in actors:
            with torch.no_grad():
                actor.encoder.residual.weight.normal_(0, .04)
                actor.encoder.residual.bias.normal_(0, .03)
                actor.tools.weight.normal_(0, .05)
                actor.movement.weight.normal_(0, .05)
    return [actor.eval() for actor in actors]


def policy_rollouts(actors):
    model = dict(format=TAG_FORMAT, observationSchema=SCHEMA, observationSize=PHYSICAL_OBSERVATIONS + 2 + NOISE,
                 physicsObservationSize=PHYSICAL_OBSERVATIONS, noiseRho=NOISE_RHO, actionSize=6,
                 commands=['keep', 'press', 'release'], actors=[export_actor(actor) for actor in actors],
                 auditLabel='Synthetic v6 parity fixture; not trained or selected')
    rollouts = []
    counts = dict(rows=0, blind=0, resets=0)
    with torch.no_grad():
        for deterministic in (False, True):
            play = STEPS - PREP
            env = PhysicsEnv(seed=1723, scenario='rooms', size=8, n_boxes=3, n_ramps=1, prep=PREP, play=play)
            physical = observe(env)
            memory = [torch.zeros(1, 64), torch.zeros(1, 64)]
            buttons = np.zeros((2, 2), np.float32)
            noises = np.zeros((2, NOISE), np.float32)
            tapes = [Tape(500 + role + 10 * deterministic) for role in range(2)]
            rows = []
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
            rollouts.append(dict(deterministic=deterministic, rows=rows))
    return model, rollouts, counts


def main():
    physics = physics_rollout()
    actors = synthetic_pair()
    model, rollouts, counts = policy_rollouts(actors)
    fixture = dict(schema=SCHEMA, sources={name: digest(ROOT / 'training_v5' / name) for name in SOURCES},
                   generator=digest(Path(__file__)), physics=physics, model=model, rollouts=rollouts, counts=counts)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(fixture, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(OUTPUT.relative_to(ROOT)), counts=counts, bytes=OUTPUT.stat().st_size)))


if __name__ == '__main__':
    main()
