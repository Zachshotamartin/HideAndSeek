"""Tag-round game rules and the v6 actor observation, kept beside the byte-identical physics.

Round structure: preparation (the seeker is blind and frozen), then play with a
time limit. A seeker within reach and in sight of the hider tags it, which ends
the round with the remaining play credited as seen (capture.py). The hider wins
by not being tagged before the clock runs out. The per-step visibility reward
stays as the dense term.

The v6 actor sees, in addition to the 208 physical measurements, what a real
player remembers and knows: where the opponent was last seen (in its own frame),
how long ago, and how much play time remains. Neither agent is told where an
unseen opponent is now, and nothing here tells either agent what to do.
"""
import numpy as np

from physics import DT, local_xy

SCHEMA = 'tag-rounds-last-seen-214-v6'
PHYSICS_OBSERVATIONS = 208
EXTRAS = 6                    # last-seen flag, relative x, y, z, age, time remaining
PHYSICAL_OBSERVATIONS = PHYSICS_OBSERVATIONS + EXTRAS
NOISE = 4                     # observed AR(1) exploration noise of the four movement outputs
MEMORY_AGE = 10.              # seconds at which the last-seen age saturates
CLOCK = 60.                   # seconds that scale the remaining play time
PREP_FRACTION = .4            # preparation is at least this fraction of the play length
MINIMUM_PREP = 96             # the original 7.7 s preparation
POSITION_SCALE = 6.           # metres, the scale physics uses for relative positions
HEIGHT_SCALE = 2.
SEEKER_SPEED_ABLATION = .85   # movement force scale of the seeker in the 'slow-seeker' arm
NEVER_SEEN = [0., 0., 0., 0., 1.]


def preparation_steps(play):
    """Preparation grows with the play length so a long game leaves time to build."""
    return int(max(MINIMUM_PREP, round(PREP_FRACTION * int(play))))


def remaining_play_seconds(env):
    return max(0, env.prep + env.play - max(env.t, env.prep)) * DT


def agent_extras(env, agent):
    """One agent's six extra measurements: last-seen opponent, its age, time remaining."""
    clock = min(1., remaining_play_seconds(env) / CLOCK)
    seen = env.last_seen[agent]
    if seen is None:
        return [*NEVER_SEEN, clock]
    position = env.data.qpos[agent * 4:agent * 4 + 3]
    yaw = env.data.qpos[agent * 4 + 3]
    delta = np.asarray(seen['position'], dtype=float) - position
    local = local_xy(delta, yaw)
    age = min(MEMORY_AGE, (env.t - seen['t']) * DT) / MEMORY_AGE
    return [1., local[0] / POSITION_SCALE, local[1] / POSITION_SCALE, delta[2] / HEIGHT_SCALE, age, clock]


def extras(env):
    return np.asarray([agent_extras(env, agent) for agent in range(2)], dtype=np.float32)


def observe(env):
    """Physical observation plus the extras, for both agents: shape (2, 214)."""
    return np.concatenate([env.observe(), extras(env)], axis=-1)


def scale_actions(actions, seeker_speed=1.):
    """Optional asymmetry ablation: scale the seeker's movement force outside the physics."""
    if seeker_speed == 1.:
        return actions
    scaled = np.array(actions, dtype=float, copy=True)
    scaled[..., 1, :2] *= seeker_speed
    return scaled


def seeker_found(info):
    """True when the seeker saw the hider at least once during play of a finished round."""
    return int(info['hidden']) < int(info['play_steps'])
