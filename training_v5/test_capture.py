"""Capture ends play with the remaining steps credited to the seeker; physics.py is untouched."""
import unittest

import mujoco
import numpy as np

from capture import CAPTURE_DISTANCE, resolve_capture
from env_pool import PhysicsEnvPool
from physics import PhysicsEnv, generate_arena


def facing_pair(gap):
    arena = generate_arena(31, 'open', n_boxes=0, n_ramps=0)
    arena['agents'] = [dict(position=[3, 3, .25], yaw=0), dict(position=[3 + gap, 3, .25], yaw=np.pi)]
    arena['objects'] = []
    return arena


class CaptureTests(unittest.TestCase):
    def test_capture_credits_remaining_play_and_ends_the_episode(self):
        e = PhysicsEnv(arena=facing_pair(.55), prep=2, play=20)
        for _ in range(3):
            row = e.step(np.zeros((2, 6)))
        observation, reward, done, info = resolve_capture(e, *row)
        self.assertTrue(done)
        self.assertTrue(info['captured'])
        self.assertEqual(info['play_steps'], 20)
        remaining = 2 + 20 - e.t
        self.assertGreater(remaining, 0)
        np.testing.assert_array_equal(reward, row[1] + np.array([-remaining, remaining]))
        self.assertEqual(reward.sum(), 0)
        self.assertEqual(info['hidden'], row[3]['hidden'])

    def test_no_capture_during_preparation_out_of_reach_or_out_of_sight(self):
        e = PhysicsEnv(arena=facing_pair(.55), prep=6, play=20)
        row = e.step(np.zeros((2, 6)))
        self.assertFalse(resolve_capture(e, *row)[2], 'the seeker is blind in preparation')
        far = PhysicsEnv(arena=facing_pair(CAPTURE_DISTANCE + .5), prep=1, play=20)
        for _ in range(2):
            row = far.step(np.zeros((2, 6)))
        self.assertFalse(resolve_capture(far, *row)[2])
        blocked = facing_pair(.55)
        blocked['walls'].append(dict(position=[3.275, 3, 1.1], size=[.05, 1.5, 2.2], yaw=0))
        walled = PhysicsEnv(arena=blocked, prep=1, play=20)
        for _ in range(2):
            row = walled.step(np.zeros((2, 6)))
        self.assertFalse(walled.seen[1, 0])
        self.assertFalse(resolve_capture(walled, *row)[2])

    def test_pool_applies_capture_and_the_trainer_reset_path_follows(self):
        with PhysicsEnvPool([dict(arena=facing_pair(.55), prep=1, play=20)], workers=1) as pool:
            for _ in range(2):
                obs, reward, done, infos = pool.step(np.zeros((1, 2, 6)))
            self.assertTrue(done[0])
            self.assertTrue(infos[0]['captured'])
            self.assertEqual(reward.sum(), 0)
            self.assertLess(reward[0, 0], 0)
            fresh = pool.reset_at([0], [11])
            self.assertTrue(np.isfinite(fresh).all())


if __name__ == '__main__':
    unittest.main()
