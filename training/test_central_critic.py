"""Behavioral checks for an isolated training-only value-baseline prototype."""
import unittest
import mujoco
import numpy as np
import torch
from central_critic import CentralCritic, central_state, batch_states
from physics import PhysicsEnv
from persistent_actor import PersistentActor, augment

torch.set_num_threads(1)


class CentralCriticTests(unittest.TestCase):
    def test_reads_current_state_without_mutating_physics(self):
        env = PhysicsEnv(seed=117)
        before = (env.data.qpos.copy(), env.data.qvel.copy(), env.observe().copy(), env.t)
        state = central_state(env, [[1, 0], [0, 1]])
        self.assertTrue(all(np.isfinite(value).all() for value in state.values()))
        np.testing.assert_array_equal(state['agents'][:, 11:13], [[1, 0], [0, 1]])
        np.testing.assert_array_equal(env.data.qpos, before[0])
        np.testing.assert_array_equal(env.data.qvel, before[1])
        np.testing.assert_array_equal(env.observe(), before[2])
        self.assertEqual(env.t, before[3])

    def test_value_is_invariant_to_object_and_wall_slot_permutations(self):
        critic = CentralCritic()
        original = batch_states([central_state(PhysicsEnv(seed=823))])
        permuted = {key: value.clone() for key, value in original.items()}
        for entity, mask in [('objects', 'objectMask'), ('walls', 'wallMask')]:
            order = torch.randperm(permuted[entity].shape[-2])
            permuted[entity] = permuted[entity][:, order]
            permuted[mask] = permuted[mask][:, order]
        torch.testing.assert_close(critic(original), critic(permuted), atol=1e-6, rtol=1e-6)

    def test_empty_prop_set_has_finite_values_and_gradients(self):
        critic = CentralCritic()
        states = batch_states([central_state(PhysicsEnv(seed=44, n_boxes=0, n_ramps=0)),
                               central_state(PhysicsEnv(seed=45, n_boxes=8, n_ramps=2))])
        values = critic.role_values(states)
        self.assertEqual(tuple(values.shape), (2, 2))
        torch.testing.assert_close(values.sum(-1), torch.zeros(2), atol=0, rtol=0)
        values[:, 0].square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in critic.parameters()))

    def test_hidden_opponent_changes_only_privileged_input(self):
        env = PhysicsEnv(seed=391, scenario='open', n_boxes=0, n_ramps=0)
        # Choose legal positions away from boundary walls. Hider looks east;
        # seeker is to its west and is excluded from actor lidar rays.
        env.data.qpos[:4] = [5, 4, .25, 0]
        env.data.qpos[4:8] = [2, 4, .25, 0]
        mujoco.mj_forward(env.model, env.data)
        env._senses()
        observed_before = env.observe()[0].copy()
        privileged_before = central_state(env)
        env.data.qpos[4:6] = [2, 3]
        mujoco.mj_forward(env.model, env.data)
        env._senses()
        np.testing.assert_array_equal(env.observe()[0], observed_before)
        self.assertFalse(np.array_equal(central_state(env)['agents'], privileged_before['agents']))

        actor, critic = PersistentActor(), CentralCritic()
        observed = torch.from_numpy(augment(env.observe(), np.zeros((2, 2), np.float32))[0:1])
        memory = torch.zeros(1, 64)
        before = actor.act(observed, memory, deterministic=True)[0].detach().clone()
        optimizer = torch.optim.Adam(critic.parameters(), lr=.001)
        for _ in range(3):
            loss = (critic(batch_states([central_state(env)])) - .75).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.testing.assert_close(actor.act(observed, memory, deterministic=True)[0], before, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_optional_memory_is_value_only_and_cannot_receive_gradients(self):
        critic = CentralCritic(use_actor_memory=True)
        state = batch_states([central_state(PhysicsEnv(seed=161))])
        memories = torch.randn(1, 2, 64, requires_grad=True)
        critic(state, memories).square().sum().backward()
        self.assertIsNone(memories.grad)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in critic.parameters()))
        with self.assertRaisesRegex(ValueError, 'pre-action actor memories'):
            critic(state)


if __name__ == '__main__':
    unittest.main()
