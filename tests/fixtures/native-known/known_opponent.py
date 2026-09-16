"""Known opponent positions; unchanged rigid-body physics and visibility reward."""
from physics import PhysicsEnv as BasePhysicsEnv, local_xy, DT
OBSERVATION_SCHEMA = 'known-opponent-position-210-v1'
TASK_CONTRACT = {
    'version': 'known-position-sustained-visibility-v1',
    'observationSchema': OBSERVATION_SCHEMA,
    'opponentPosition': 'Current relative XYZ for both roles, including behind walls.',
    'opponentVisibility': 'Slot 10 is actual own field-of-view and ray visibility.',
    'opponentMotion': 'Velocity and heading remain visible-only.',
    'reward': 'Each play step: seeker +1 if hider visible, otherwise -1; hider opposite.',
    'termination': 'Training time horizon only; no capture or future visibility credit.',
    'physicalGame': 'Unchanged body, jump, tool and collision physics.',
}
class PhysicsEnv(BasePhysicsEnv):
    def observe(self):
        observation = super().observe()
        for role in range(2):
            own, other = role * 4, (1 - role) * 4
            delta = self.data.qpos[other:other + 3] - self.data.qpos[own:own + 3]
            xy = local_xy(delta, self.data.qpos[own + 3])
            observation[role, 11:14] = [xy[0] / 6, xy[1] / 6, delta[2] / 2]
        return observation
