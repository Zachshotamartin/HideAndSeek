"""Observed AR(1) exploration noise: exact likelihoods, the right correlation, carried through acting."""
import math
import unittest

import numpy as np
import torch

from entity_actor import EntityActor
from game import NOISE
from league_ppo import act_grouped
from persistent_actor import PersistentActor

torch.set_num_threads(1)
RHO = .7
WIDE = 214 + 2 + NOISE


class CoherentNoiseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77)

    def test_noise_chain_has_the_declared_correlation_and_stationary_scale(self):
        actor = PersistentActor(observation_size=WIDE, noise_rho=RHO).eval()
        physical = torch.zeros(2000, 214)
        buttons = torch.zeros(2000, 2)
        noise = torch.zeros(2000, NOISE)
        memory = torch.zeros(2000, 64)
        chain = []
        with torch.no_grad():
            for _ in range(40):
                observation = torch.cat([physical, buttons, noise], 1)
                _, _, raw, logp, _, memory, noise = actor.act_with_noise(observation, memory)
                chain.append(noise.clone())
                # The recorded log-probability is exactly the conditional likelihood of the sampled movement.
                normal, tools, _, _ = actor(observation, memory)
        chain = torch.stack(chain)
        later, earlier = chain[10:].flatten(1), chain[9:-1].flatten(1)
        correlation = float((later * earlier).mean() / earlier.square().mean())
        self.assertAlmostEqual(correlation, RHO, delta=.03)
        scale = float(actor.log_std.clamp(-2.5, .3).exp().mean())
        self.assertAlmostEqual(float(chain[20:].std()), scale, delta=.03 * scale)

    def test_log_probability_conditions_on_the_carried_noise(self):
        actor = PersistentActor(observation_size=WIDE, noise_rho=RHO).eval()
        observation = torch.randn(5, WIDE) * .3
        memory = torch.randn(5, 64)
        with torch.no_grad():
            _, _, raw, logp, _, _, noise = actor.act_with_noise(observation, memory)
            normal, tools, _, _ = actor(observation, memory)
            recomputed, _ = actor.statistics(normal, tools, raw, include_entropy=False)
        torch.testing.assert_close(recomputed, logp, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(normal.mean - RHO * observation[:, -NOISE:], actor.movement(actor(observation, memory)[3]),
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(normal.scale[0], actor.log_std.clamp(-2.5, .3).exp() * math.sqrt(1 - RHO ** 2), atol=1e-6, rtol=1e-6)
        with torch.no_grad():
            deterministic = actor.act_with_noise(observation, memory, deterministic=True)
        self.assertEqual(float(deterministic[6].abs().sum()), 0)
        torch.testing.assert_close(deterministic[0], torch.tanh(actor.movement(deterministic[5])), atol=1e-6, rtol=1e-6)

    def test_grouped_acting_carries_noise_per_role_and_legacy_opponents_ignore_it(self):
        models = [EntityActor(observation_size=WIDE, noise_rho=RHO), EntityActor(observation_size=WIDE, noise_rho=RHO)]
        histories = [[PersistentActor(), PersistentActor()]]
        physical = np.random.default_rng(3).normal(size=(6, 2, 214)).astype(np.float32) * .2
        physical[:, :, 5] = 1
        memories = [torch.zeros(6, 64), torch.zeros(6, 64)]
        buttons = np.zeros((6, 2, 2), np.float32)
        roles = np.array([[-1, -1], [-1, 0], [0, -1], [0, 0], [-1, -1], [-1, 0]])
        noises = np.zeros((6, 2, NOISE), np.float32)
        first = act_grouped(models, histories, physical, memories, buttons, roles, noises=noises)
        actions, next_memories, next_buttons, next_noises, records = first
        self.assertEqual(next_noises.shape, (6, 2, NOISE))
        current = roles == -1
        self.assertTrue((np.abs(next_noises[current]).sum(-1) > 0).all(), 'current v6 actors carry sampled noise')
        self.assertTrue((next_noises[~current] == 0).all(), 'legacy opponents never observe noise')
        self.assertEqual(records[0][0].shape, (6, WIDE))
        np.testing.assert_array_equal(records[0][0][:, 214:216].numpy(), buttons[:, 0])
        np.testing.assert_array_equal(records[0][0][:, 216:].numpy(), noises[:, 0])
        second = act_grouped(models, histories, physical, next_memories, next_buttons, roles, noises=next_noises)
        np.testing.assert_array_equal(second[4][1][0][:, 216:].numpy(), next_noises[:, 1])
        bootstrap = act_grouped(models, histories, physical, memories, buttons, roles, sample=False, noises=next_noises)
        np.testing.assert_array_equal(bootstrap[3], next_noises)


if __name__ == '__main__':
    unittest.main()
