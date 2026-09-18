"""Contact-aware diagnostics. Geometry and sensors are never changed."""
import numpy as np


def blocked_attempt(command, speed, contact, contacted_speed=0):
    # Stillness in cover, successful sliding and actually pushing a moving prop
    # are not blocked movement. Costs, if enabled, accumulate only during futile effort.
    return bool(np.linalg.norm(command) > .5 and speed < .05 and contact and contacted_speed < .05)


def blocked_roles(env):
    if env.t <= env.prep:
        return [False, False]
    result = []
    for role, geom in enumerate(env.agent_geoms):
        touching = False
        object_speed = 0.
        for contact in env.data.contact:
            other = contact.geom2 if contact.geom1 == geom else contact.geom1 if contact.geom2 == geom else -1
            if contact.dist > 0 or other < 0 or env.model.geom_group[other] not in (1, 3):
                continue
            touching = True
            body = env.model.geom_bodyid[other]
            object_speed = max(object_speed, float(np.linalg.norm(env.data.cvel[body, 3:6])))
        speed = float(np.linalg.norm(env.data.qvel[role * 4:role * 4 + 2]))
        result.append(blocked_attempt(env.actions[role, :2], speed, touching, object_speed))
    return result


def blocked_adjustment(blocked, weight):
    """Optional zero-sum effort-cost ablation, separate from reported game outcomes."""
    if not np.isfinite(weight) or not 0 <= weight <= .1:
        raise ValueError('Blocked cost must be between zero and 0.1 per decision')
    b = np.asarray(blocked, dtype=float)
    delta = weight * (b[..., 1] - b[..., 0])
    return np.stack([delta, -delta], axis=-1)
