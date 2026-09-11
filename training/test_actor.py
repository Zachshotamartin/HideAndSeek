"""Distribution regressions for bounded physical actions, independent of PPO."""
import unittest
import torch
from torch.distributions import Bernoulli, Normal, TransformedDistribution, TanhTransform
from actor import PhysicalActor


class ActorDistributionTests(unittest.TestCase):
    def test_log_density_matches_transformed_distribution(self):
        mean = torch.tensor([[.2, -.5, 1.2]], dtype=torch.float64)
        normal = Normal(mean, torch.full_like(mean, .6))
        tools = Bernoulli(logits=torch.zeros(1, 2, dtype=torch.float64))
        raw = torch.tensor([[.4, -.8, 2., 1., 0.]], dtype=torch.float64)
        logp, _ = PhysicalActor.statistics(normal, tools, raw)
        bounded = TransformedDistribution(normal, [TanhTransform(cache_size=1)])
        expected = bounded.log_prob(raw[:, :3].tanh()).sum(-1) + tools.log_prob(raw[:, 3:]).sum(-1)
        torch.testing.assert_close(logp, expected, atol=1e-12, rtol=1e-12)

    def test_saturated_policy_has_low_entropy_and_restoring_gradient(self):
        torch.manual_seed(91)
        samples = 16000
        mean = torch.tensor([6., -6., 0.], requires_grad=True)
        normal = Normal(mean.expand(samples, 3), torch.full((samples, 3), .5))
        tools = Bernoulli(logits=torch.zeros(samples, 2))
        raw = torch.zeros(samples, 5)
        _, saturated = PhysicalActor.statistics(normal, tools, raw)
        _, centered = PhysicalActor.statistics(Normal(torch.zeros(samples, 3), .5), tools, raw)
        self.assertLess(saturated.mean().item(), centered.mean().item() - 15)
        saturated.mean().backward()
        self.assertLess(mean.grad[0].item(), -1.9)
        self.assertGreater(mean.grad[1].item(), 1.9)

    def test_extreme_means_remain_finite(self):
        mean = torch.tensor([[100., -100., 0.]], requires_grad=True)
        normal = Normal(mean, .5)
        tools = Bernoulli(logits=torch.zeros(1, 2))
        raw = torch.cat((mean.detach(), torch.zeros(1, 2)), -1)
        logp, entropy = PhysicalActor.statistics(normal, tools, raw)
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue(torch.isfinite(entropy).all())
        entropy.sum().backward()
        self.assertTrue(torch.isfinite(mean.grad).all())

    def test_deterministic_inference_does_not_consume_randomness(self):
        actor = PhysicalActor(138)
        rng = torch.get_rng_state()
        actor.act(torch.zeros(1, 138), torch.zeros(1, 64), deterministic=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
