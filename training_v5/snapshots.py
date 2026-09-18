"""Exact physical snapshots including mutable grab welds and RNG state."""
import copy

import mujoco
import numpy as np

from physics import PhysicsEnv

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
LIVE_HANDLES = ('model', 'data')


def dump(env):
    """Everything needed to rebuild ``env`` mid-episode, as plain picklable values."""
    state = np.empty(mujoco.mj_stateSize(env.model, SPEC))
    mujoco.mj_getState(env.model, env.data, state, SPEC)
    attributes = copy.deepcopy({k: v for k, v in vars(env).items() if k not in LIVE_HANDLES})
    return dict(attributes=attributes, integration=state, eqData=env.model.eq_data.copy())


def restore(record):
    a = record['attributes']
    env = PhysicsEnv(arena=a['arena'], prep=a['prep'], play=a['play'], disable_tools=a['disable_tools'],
                     immovable=a['immovable'], **a['config'])
    for k, v in a.items():
        setattr(env, k, copy.deepcopy(v))
    env.model.eq_data[:] = record['eqData']
    mujoco.mj_setState(env.model, env.data, record['integration'], SPEC)
    mujoco.mj_forward(env.model, env.data)
    return env
