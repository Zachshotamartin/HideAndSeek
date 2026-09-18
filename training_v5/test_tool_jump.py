"""Grab, lock and jump semantics of the physical game, plus the legacy migration path."""
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from entity_actor import load_pair
from league_ppo import act_grouped, actor_update
from migrate_environment import migrate
from physics import PhysicsEnv, generate_arena


def test_arena():
    """Two agents facing one box on an otherwise empty floor."""
    arena = generate_arena(31, 'open', n_boxes=0, n_ramps=0)
    arena['agents'] = [dict(position=[3, 3, .25], yaw=0), dict(position=[4.25, 3, .25], yaw=np.pi)]
    arena['objects'] = [dict(id='test', kind='box', position=[3.62, 3, .353], size=[.7, .7, .7], yaw=0, mass=1.2)]
    return PhysicsEnv(arena=arena, prep=0, play=1000)


class InteractionTests(unittest.TestCase):
    def test_exclusive_grab_release_and_lock(self):
        e = test_arena()
        a = np.zeros((2, 6))
        a[:, 3] = 1
        e._tools(a)
        self.assertEqual(sum(i >= 0 for i in e.grips), 1)
        a[0, 4] = 1
        e._tools(a)
        self.assertEqual(e.grips, [-1, -1])
        self.assertEqual(e.locks[0], 0)
        e._tools(a)
        self.assertEqual(e.grips, [-1, -1])
        a[0, 4] = 0
        e._tools(a)
        e._tools(a)
        self.assertEqual(sum(i >= 0 for i in e.grips), 1)
        a[:] = 0
        e._tools(a)
        self.assertEqual(e.grips, [-1, -1])

    def test_jump_grounded_and_height(self):
        e = test_arena()
        a = np.zeros((2, 6))
        e.step(a)
        a[0, 5] = 1
        e.step(a)
        self.assertEqual(e.jump_events[0], 1)
        peak = e.data.qpos[2]
        for _ in range(5):
            e.step(a)
            peak = max(peak, e.data.qpos[2])
        self.assertEqual(e.jump_events[0], 1)
        self.assertGreater(peak - .25, .7)
        self.assertLess(peak - .25, .81)

    def test_jump_from_prop_has_wall_safe_apex(self):
        e = test_arena()
        e.data.qpos[:3] = [3.62, 3, 1.003]
        mujoco.mj_forward(e.model, e.data)
        a = np.zeros((2, 6))
        e.step(a)
        a[0, 5] = 1
        peak = e.data.qpos[2]
        for _ in range(12):
            e.step(a)
            peak = max(peak, e.data.qpos[2])
        self.assertGreater(e.jump_events[0], 0)
        self.assertLess(peak - .25, 2.05)

    def test_can_land_on_box(self):
        e = test_arena()
        e.data.qpos[0] = 2.6
        mujoco.mj_forward(e.model, e.data)
        a = np.zeros((2, 6))
        e.step(a)
        a[0, 0] = 1
        a[0, 5] = 1
        for t in range(15):
            e.step(a)
            a[0, 5] = 0
            if t == 6:
                a[0, 0] = 0
        self.assertGreater(e.data.qpos[0], 3.27)
        self.assertAlmostEqual(e.data.qpos[2], .95, places=2)

    def test_jump_does_not_cross_wall(self):
        e = test_arena()
        e.arena['objects'] = []
        e.arena['walls'].append(dict(position=[3.62, 3, 1.1], size=[.18, 5, 2.2], yaw=0))
        e.reset(arena=e.arena)
        a = np.zeros((2, 6))
        a[0, 0] = 1
        a[0, 5] = 1
        for _ in range(120):
            e.step(a)
        self.assertLess(e.data.qpos[0], 3.62 - .09)

    def test_seeker_cannot_jump_in_preparation(self):
        e = test_arena()
        e.prep = 96
        a = np.ones((2, 6))
        e.step(a)
        self.assertEqual(e.jump_events[1], 0)

    def test_migration_and_ppo(self):
        torch.set_num_threads(2)
        from synthetic import legacy_movement_source
        with tempfile.TemporaryDirectory() as folder:
            old = torch.load(legacy_movement_source(folder + '/legacy.pt'), map_location='cpu', weights_only=False)
        record, models = migrate(old)
        for before, after in zip(old['models'], models):
            torch.testing.assert_close(before['movement.weight'], after.movement.weight[:3])
            self.assertEqual(after.movement.weight.shape[0], 4)
            torch.testing.assert_close(after.movement.weight[3], torch.zeros_like(after.movement.weight[3]))
        e = test_arena()
        obs = np.stack([e.observe()] * 2)
        mem = [torch.zeros(2, models[0].hidden_size) for _ in range(2)]
        buttons = np.zeros((2, 2, 2), np.float32)
        roles = np.full((2, 2), -1)
        actions, states, buttons, _, records = act_grouped(models, [], obs, mem, buttons, roles)
        self.assertEqual(actions.shape, (2, 2, 6))
        e.step(actions[0])
        ob, raw, logp, _ = records[0]
        result = actor_update(models[0], torch.optim.Adam(models[0].parameters(), lr=1e-4),
                              (ob[None], mem[0][None], torch.ones(1, 2), raw[None], logp[None]), torch.tensor([[1., -1.]]),
                              torch.ones(1, 2), 0, epochs=1, sequence_length=1, sequence_batch=2)
        self.assertTrue(np.isfinite(result['loss']))


if __name__ == '__main__':
    unittest.main()
