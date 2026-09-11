import copy
import unittest
import torch
from central_critic import central_state, batch_states
from persistent_actor import PersistentActor
from physics import PhysicsEnv
from residual_critic import ResidualCentralCritic
from fit_residual_value import assess


class ResidualValueTests(unittest.TestCase):
    def test_zero_initialization_preserves_averaged_prediction_baseline(self):
        critic = ResidualCentralCritic()
        state = batch_states([central_state(PhysicsEnv(seed=732))] * 3)
        partial = torch.tensor([[2., -4.], [-2., 4.], [1., 3.]])
        memory = torch.randn(3, 2, 64)
        for training in [True, False]:
            critic.train(training)
            torch.testing.assert_close(critic(state, memory, partial), torch.tensor([3., -3., -1.]), atol=0, rtol=0)

    def test_central_update_detaches_partial_values_and_actor_memories(self):
        actor = PersistentActor()
        before = copy.deepcopy(actor.state_dict())
        _, _, value, memory = actor(torch.randn(2, 140), torch.zeros(2, 64))
        critic = ResidualCentralCritic()
        torch.nn.init.normal_(critic.correction.value[-1].weight, std=.1)
        state = batch_states([central_state(PhysicsEnv(seed=733))] * 2)
        prediction = critic(state, torch.stack([memory, memory], dim=1), torch.stack([value, -value], dim=-1))
        prediction.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters()))
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_pair_prediction_mse_differs_from_mean_individual_mse(self):
        state = batch_states([central_state(PhysicsEnv(seed=734))] * 240)
        data = dict(states=state, memories=torch.zeros(240, 2, 64),
                    partialValues=torch.tensor([[2., 2.]]).repeat(240, 1),
                    targets=torch.zeros(240), phases=torch.arange(240), episodes=1)
        result = assess(ResidualCentralCritic(), data, torch.zeros(240))
        self.assertEqual(result['mse']['partialPairPrediction'], 0.)
        self.assertEqual(result['mse']['meanIndividualPartialMSE'], 4.)
        self.assertEqual(result['mse']['residualCentral'], 0.)


if __name__ == '__main__':
    unittest.main()
