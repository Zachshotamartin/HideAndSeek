"""Tag-round rules and the v6 actor observation, measured on real physics states."""
import unittest

import mujoco
import numpy as np

from game import (CLOCK, EXTRAS, MEMORY_AGE, MINIMUM_PREP, NOISE, PHYSICAL_OBSERVATIONS, extras, observe, preparation_steps,
                  scale_actions, seeker_found)
from physics import DT, PhysicsEnv, generate_arena


def open_pair(seed=41, hider_yaw=0.):
    arena = generate_arena(seed, 'open', n_boxes=0, n_ramps=0)
    arena['agents'] = [dict(position=[2, 3, .25], yaw=hider_yaw), dict(position=[5, 3, .25], yaw=np.pi)]
    arena['objects'] = []
    return arena


class GameTests(unittest.TestCase):
    def test_preparation_follows_the_play_length(self):
        self.assertEqual(preparation_steps(144), MINIMUM_PREP)
        self.assertEqual(preparation_steps(188), MINIMUM_PREP)
        self.assertEqual(preparation_steps(375), 150)
        self.assertEqual(preparation_steps(750), 300)

    def test_extras_before_any_sighting_and_the_clock(self):
        # The hider faces away from the seeker and the seeker is blind in preparation: nobody has seen anybody.
        env = PhysicsEnv(arena=open_pair(hider_yaw=np.pi), prep=4, play=100)
        try:
            self.assertEqual(env.last_seen, [None, None])
            rows = extras(env)
            self.assertEqual(rows.shape, (2, EXTRAS))
            np.testing.assert_allclose(rows[:, :5], [[0, 0, 0, 0, 1]] * 2)
            np.testing.assert_allclose(rows[:, 5], min(1, 100 * DT / CLOCK))
            self.assertEqual(observe(env).shape, (2, PHYSICAL_OBSERVATIONS))
            np.testing.assert_array_equal(observe(env)[:, :208], env.observe())
            env.t = 4 + 40
            np.testing.assert_allclose(extras(env)[:, 5], 60 * DT / CLOCK)
        finally:
            env.close()

    def test_last_seen_offset_is_local_and_ages_until_saturation(self):
        env = PhysicsEnv(arena=open_pair(), prep=0, play=400)
        try:
            env.step(np.zeros((2, 6)))
            self.assertTrue(env.seen[1, 0], 'the facing pair sees each other')
            row = extras(env)[1]
            self.assertEqual(row[0], 1)
            hider = env.data.qpos[:3]
            seeker = env.data.qpos[4:7]
            yaw = env.data.qpos[7]
            delta = hider - seeker
            expected_x = (np.cos(yaw) * delta[0] + np.sin(yaw) * delta[1]) / 6
            expected_y = (-np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1]) / 6
            np.testing.assert_allclose(row[1:4], [expected_x, expected_y, delta[2] / 2], atol=1e-6)
            self.assertAlmostEqual(row[4], 0)
            seen_at = env.t
            env.t = seen_at + 40
            self.assertAlmostEqual(extras(env)[1, 4], 40 * DT / MEMORY_AGE, places=5)
            env.t = seen_at + int(30 / DT)
            self.assertAlmostEqual(extras(env)[1, 4], 1)
            # Turning the seeker rotates the remembered offset into its own frame.
            env.data.qpos[7] = yaw + np.pi / 2
            mujoco.mj_forward(env.model, env.data)
            turned = extras(env)[1]
            np.testing.assert_allclose(turned[1:3], [expected_y, -expected_x], atol=1e-6)
        finally:
            env.close()

    def test_seeker_scaling_and_found_rule(self):
        actions = np.ones((3, 2, 6))
        scaled = scale_actions(actions, .85)
        np.testing.assert_array_equal(actions, 1)
        np.testing.assert_allclose(scaled[:, 1, :2], .85)
        np.testing.assert_array_equal(scaled[:, 0], 1)
        np.testing.assert_array_equal(scaled[:, 1, 2:], 1)
        self.assertIs(scale_actions(actions), actions)
        self.assertTrue(seeker_found(dict(hidden=100, play_steps=188)))
        self.assertFalse(seeker_found(dict(hidden=188, play_steps=188)))
        self.assertEqual(NOISE, 4)


if __name__ == '__main__':
    unittest.main()
