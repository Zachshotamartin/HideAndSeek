"""Evaluation-only targets. No fixture or target controller is used in training."""
import math
import numpy as np
from physics import generate_arena, local_xy

FIXTURES = ('visible-target', 'blocked-path', 'reacquisition')


def fixture_arena(name, seed):
    if name not in FIXTURES:
        raise ValueError('Unknown diagnostic fixture')
    arena = generate_arena(seed, 'open', 8, 0, 0)
    # Retain exactly the generated outside walls; lay out simple diagnostic scenes.
    arena['walls'] = [dict(position=p, size=s, yaw=0) for p, s in [
        ([4,-.1,1.1],[8.4,.2,2.2]), ([4,8.1,1.1],[8.4,.2,2.2]),
        ([-.1,4,1.1],[.2,8,2.2]), ([8.1,4,1.1],[.2,8,2.2])]]
    arena['objects'] = []
    # A small paired translation varies start positions without changing the task.
    shift = float(np.random.default_rng(seed).uniform(-.2, .2))
    y = 2.5 + shift
    arena['agents'] = [dict(position=[5.7,y,.25],yaw=math.pi), dict(position=[2.3,y,.25],yaw=0)]
    if name != 'visible-target':
        arena['walls'].append(dict(position=[4,4,1.1],size=[.25,2.4,2.2],yaw=0))
    if name == 'blocked-path':
        for agent in arena['agents']:
            agent['position'][1] = 4 + shift
    arena['scenario'] = 'open'
    return arena


def target_action(name, env):
    """Stationary hider, or a target that moves behind a wall and stops.

    The seeker always uses its learned policy. Forces, collision and sensors
    remain physical. This is a diagnostic target, not a learned hider score.
    """
    action = np.zeros(6, np.float32)
    if name == 'reacquisition' and env.t >= 4 and env.data.qpos[1] < 4.2:
        action[:2] = local_xy([0, .75], env.data.qpos[3])
    return action
