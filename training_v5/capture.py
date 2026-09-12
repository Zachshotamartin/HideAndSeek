"""Capture rule: a seeker within reach and in sight of the hider ends the play phase.

The reward stays the original zero-sum visibility reward; a capture credits the
seeker with every remaining play step as seen and the hider with the same
steps as found. Under a visibility-only reward a seeker was right to keep its
distance as long as it could see the hider, which is not seeking. The rule
lives beside physics.py so the physical contract stays byte-identical: no
world, sensor or per-step reward changes.
"""
import numpy as np

CAPTURE_DISTANCE = .7   # metres between agent centres; the two 0.25 m spheres touch at 0.5 m


def captured(env):
    """True during play when the seeker sees the hider from within reach."""
    if env.t <= env.prep or not env.seen[1, 0]:
        return False
    q = env.data.qpos
    return float(np.hypot(q[0] - q[4], q[1] - q[5])) < CAPTURE_DISTANCE


def resolve_capture(env, observation, reward, done, info):
    """Apply the capture rule to one environment step's results."""
    if done or not captured(env):
        return observation, reward, done, info
    remaining = env.prep + env.play - env.t
    reward = np.asarray(reward, dtype=float) + np.array([-remaining, remaining], dtype=float)
    info = dict(info, captured=True, captureStep=int(env.t - env.prep), play_steps=env.play)
    return observation, reward, True, info
