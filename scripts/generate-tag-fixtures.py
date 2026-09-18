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
from parity_fixtures import policy_rollouts as native_rollouts     # noqa: E402
from physics import PhysicsEnv, generate_arena                   # noqa: E402

torch.set_num_threads(1)
SOURCES = ['physics.py', 'game.py', 'capture.py', 'entity_actor.py', 'persistent_actor.py', 'persistent_evaluate.py', 'parity_fixtures.py']
NOISE_RHO = .7
OUTPUT = ROOT / 'tests/fixtures/tag-rounds-native.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    rollouts, counts = native_rollouts(actors)
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
