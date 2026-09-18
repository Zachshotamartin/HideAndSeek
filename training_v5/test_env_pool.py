"""The process pool must match single-process physics exactly and fail loudly."""
import unittest

import numpy as np

from env_pool import PhysicsEnvPool, PhysicsPoolError
from physics import PhysicsEnv


def random_actions(rng):
    a = rng.uniform(-1, 1, (4, 2, 6))
    a[:, :, 3:5] = (a[:, :, 3:5] > 0).astype(float)
    a[:, :, 5] = (a[:, :, 5] > .5).astype(float)
    return a


class PoolTests(unittest.TestCase):
    def test_native_single_multi_parity_and_explicit_reset(self):
        configs = [dict(seed=771 + i, scenario=['shelter', 'rooms', 'open'][i % 3], size=8, n_boxes=3, n_ramps=1, prep=4, play=12)
                   for i in range(4)]
        single = [PhysicsEnv(**config) for config in configs]
        with PhysicsEnvPool(configs, workers=2) as pool:
            np.testing.assert_array_equal(pool.observations, np.stack([e.observe() for e in single]))
            rng = np.random.default_rng(12)
            for step in range(18):
                a = random_actions(rng)
                obs, rew, done, infos = pool.step(a)
                expected = [e.step(action) for e, action in zip(single, a)]
                np.testing.assert_allclose(obs, np.stack([r[0] for r in expected]), rtol=0, atol=1e-7)
                np.testing.assert_array_equal(rew, np.stack([r[1] for r in expected]))
                np.testing.assert_array_equal(done, [r[2] for r in expected])
                for info in infos:
                    self.assertEqual(len(info['blockedMotion']), 2)
                    self.assertTrue(all(isinstance(x, bool) for x in info['blockedMotion']))
                self.assertEqual([{k: v for k, v in info.items() if k != 'blockedMotion'} for info in infos],
                                 [r[3] for r in expected])
            self.assertTrue(done.all())
            self.assertEqual(infos[0]['t'], 16, 'no silent auto-reset')
            reset = pool.reset_at([3, 1], [123456, 78910])
            np.testing.assert_array_equal(reset[0], single[3].reset(123456))
            np.testing.assert_array_equal(reset[1], single[1].reset(78910))
            traces = pool.traces()
            self.assertEqual(traces[0]['t'], 16)
            self.assertEqual(traces[1]['t'], 0)
            self.assertEqual(traces[3]['t'], 0)
            changed = pool.reset_at([2], [42], [dict(scenario='open', n_boxes=0, n_ramps=0)])
            np.testing.assert_array_equal(changed[0], PhysicsEnv(seed=42, scenario='open', n_boxes=0, n_ramps=0, prep=4, play=12).observe())

    def test_game_observation_mode_matches_the_single_process_rules(self):
        from game import PHYSICAL_OBSERVATIONS, observe
        configs = [dict(seed=901 + i, scenario='open', size=8, n_boxes=1, n_ramps=0, prep=2, play=30) for i in range(3)]
        single = [PhysicsEnv(**config) for config in configs]
        with PhysicsEnvPool(configs, workers=2, observation='game') as pool:
            self.assertEqual(pool.observations.shape, (3, 2, PHYSICAL_OBSERVATIONS))
            np.testing.assert_array_equal(pool.observations, np.stack([observe(e) for e in single]))
            rng = np.random.default_rng(4)
            for _ in range(12):
                a = random_actions(rng)[:3]
                obs, _, done, _ = pool.step(a)
                for e, action in zip(single, a):
                    e.step(action)
                np.testing.assert_allclose(obs, np.stack([observe(e) for e in single]), rtol=0, atol=1e-7)
            reset = pool.reset_at([1], [3131])
            single[1].reset(3131)
            np.testing.assert_array_equal(reset[0], observe(single[1]))
            snapshot = pool.snapshot()
            restored = pool.restore(snapshot)
            self.assertEqual(restored.shape, (3, 2, PHYSICAL_OBSERVATIONS))
        with self.assertRaises(ValueError):
            PhysicsEnvPool(configs, workers=1, observation='unknown')

    def test_invalid_batch_and_worker_failure_cleanup(self):
        pool = PhysicsEnvPool([dict(seed=4)], workers=1)
        with self.assertRaises(ValueError):
            pool.step(np.zeros((1, 2, 5)))
        pool.processes[0].terminate()
        pool.processes[0].join()
        with self.assertRaises(PhysicsPoolError):
            pool.step(np.zeros((1, 2, 6)))
        self.assertTrue(pool.closed)
        pool.close()

    def test_diverged_world_ends_its_episode_instead_of_crashing(self):
        with PhysicsEnvPool([dict(seed=9, scenario='open', n_boxes=1, n_ramps=0, prep=1, play=20)], workers=1) as pool:
            snap = pool.snapshot()
            world = snap['worlds'][0]
            nq = len(world['attributes']['initial_objects']) * 4 + 8
            # The integration state is [time, qpos, qvel, ...]; velocities beyond
            # mjMAXVAL trigger MuJoCo's silent instability reset.
            world['integration'][1 + nq:1 + nq + 8] = 1e12
            pool.restore(snap)
            obs, reward, done, infos = pool.step(np.zeros((1, 2, 6)))
            self.assertTrue(done[0])
            self.assertTrue(infos[0].get('diverged'))
            np.testing.assert_array_equal(reward[0], [0, 0])
            self.assertTrue(np.isfinite(obs).all())
            fresh = pool.reset_at([0], [10])
            self.assertTrue(np.isfinite(fresh).all())
            obs, reward, done, infos = pool.step(np.zeros((1, 2, 6)))
            self.assertFalse(done[0])
            self.assertFalse(infos[0].get('diverged', False))

    def test_bad_startup_cleans_up(self):
        with self.assertRaises(PhysicsPoolError):
            PhysicsEnvPool([dict(scenario='not-an-arena')], workers=1)


if __name__ == '__main__':
    unittest.main()
