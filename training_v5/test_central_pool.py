"""Gated privileged-value transport must not alter simulation or actor observations."""
import unittest
import numpy as np
from physics import PhysicsEnv
from env_pool import PhysicsEnvPool
from central_critic import central_state


class CentralPoolTests(unittest.TestCase):
    def test_before_after_and_partial_reset_states_match_direct_world(self):
        configs=[dict(seed=17+i,scenario='open',n_boxes=2,n_ramps=0,prep=2,play=8) for i in range(3)]
        worlds=[PhysicsEnv(**config) for config in configs]
        with PhysicsEnvPool(configs,workers=2,with_central_state=True) as pool:
            for i,world in enumerate(worlds):
                for key,value in central_state(world).items():
                    np.testing.assert_array_equal(pool.central_states[i][key],value)
            previous=pool.central_states
            actions=np.zeros((3,2,6));actions[:,:,:3]=[.5,.2,-.1];actions[:,:,3:5]=[1,0]
            for _ in range(5):
                result=pool.step(actions)
                direct=[world.step(action) for world,action in zip(worlds,actions)]
                self.assertEqual(len(result),4)
                np.testing.assert_allclose(result[0],np.stack([row[0] for row in direct]),atol=0,rtol=0)
                np.testing.assert_array_equal(result[1],np.stack([row[1] for row in direct]))
                for i,world in enumerate(worlds):
                    for key,value in central_state(world,(world.actions[:,3:5]>.5).astype(np.float32)).items():
                        np.testing.assert_array_equal(pool.central_states[i][key],value)
            self.assertEqual(previous[0]['global'][2],0)
            untouched=pool.central_states[0]
            pool.reset_at([2],[444],[{'n_boxes':0}])
            reset=PhysicsEnv(**{**configs[2],'seed':444,'n_boxes':0})
            for key,value in central_state(reset).items():
                np.testing.assert_array_equal(pool.central_states[2][key],value)
            self.assertIs(pool.central_states[0],untouched)
            np.testing.assert_array_equal(pool.central_states[2]['agents'][:,11:13],0)

    def test_default_pool_retains_original_public_api(self):
        with PhysicsEnvPool([dict(seed=99)],workers=1) as pool:
            self.assertIsNone(pool.central_states)
            self.assertEqual(len(pool.step(np.zeros((1,2,6)))),4)
            self.assertIsNone(pool.central_states)


if __name__=='__main__':
    unittest.main()
