import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from entity_actor import prepare_pair, FORMAT, TAG_FORMAT, load_pair
from persistent_actor import PersistentActor, FORMAT as LEGACY_FORMAT
from persistent_train import file_hash
from residual_critic import ResidualCentralCritic, SCHEMA
from train_entity import initialize_critic, parser, train
from widen_policy import widen_actor, widen_critic


torch.set_num_threads(1)


class EntityTrainingTests(unittest.TestCase):
    def test_inherited_critic_weights_are_exact_but_both_optimizers_reset(self):
        model = ResidualCentralCritic(.1)
        old = torch.optim.AdamW(model.parameters(), lr=.001)
        sum(value.square().sum() for value in model.parameters()).backward()
        old.step()
        source = dict(centralCriticSchema=SCHEMA, centralCritic=model.state_dict(),
                      centralCriticOptimizer=old.state_dict())
        first, first_optimizer = initialize_critic(source, .0001)
        second, second_optimizer = initialize_critic(source, .0001)
        self.assertFalse(first_optimizer.state)
        self.assertFalse(second_optimizer.state)
        self.assertTrue(old.state)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, source['centralCritic'][name], atol=0, rtol=0)
            torch.testing.assert_close(value, second.state_dict()[name], atol=0, rtol=0)

    def test_actual_rollout_terminal_checkpoint_resume_and_protocol_guards(self):
        self.actual_rollout(False)

    def test_scaled_rollout_terminal_checkpoint_resume_and_protocol_guards(self):
        self.actual_rollout(True)

    def test_tag_round_schema_rollout_with_noise_and_staged_curriculum(self):
        self.actual_rollout(False, widened=True)

    def actual_rollout(self, scaled, widened=False):
        torch.manual_seed(65591)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            physics = file_hash(Path(__file__).with_name('physics.py'))
            actors = [PersistentActor(), PersistentActor()]
            parent = dict(format=LEGACY_FORMAT, models=[model.state_dict() for model in actors],
                decisions=123, torchRNG=torch.get_rng_state(),
                provenance=dict(physicsSHA256=physics, roleSources=[], sourceSelectedPairSHA256='test-parent'))
            noise_rho = .7 if widened else 0.
            prepared = prepare_pair(parent, projected=True, observation_size=220 if widened else 210, noise_rho=noise_rho)
            if scaled:
                small,_=load_pair(prepared)
                prepared['models']=[widen_actor(m).state_dict() for m in small]
            initial = copy.deepcopy(parent)
            initial['decisions'] = 0
            torch.save(prepared, root / 'entity.pt')
            torch.save(initial, root / 'initial.pt')
            critic = ResidualCentralCritic(.1)
            if scaled: critic=widen_critic(critic.state_dict())
            torch.save(dict(centralCriticSchema=SCHEMA, centralCritic=critic.state_dict(),
                provenance=dict(physicsSHA256=physics)), root / 'critic.pt')
            args = parser().parse_args([
                '--parent', str(root / 'entity.pt'), '--critic', str(root / 'critic.pt'),
                '--initial', str(root / 'initial.pt'), '--history', str(root / 'initial.pt'),
                '--output', str(root / 'run'), '--protocol', str(root / 'protocol.json'),
                '--encoder', 'entity', '--variant', 'short', '--envs', '4', '--workers', '2', '--horizon', '256',
                '--sequence-length', '128' if scaled else '32', '--sequence-batch', '8', '--critic-batch-size', '256', '--epochs', '1',
                '--target-interactions', '4096', '--retain-updates', '1,2', '--noise-rho', str(noise_rho),
                '--curriculum', 'staged' if widened else 'none'])
            settings = {name: getattr(args, name) for name in [
                'seed', 'arm', 'envs', 'workers', 'horizon', 'sequence_length', 'sequence_batch',
                'critic_batch_size', 'epochs', 'learning_rate', 'critic_learning_rate',
                'entropy', 'kl_limit', 'target_interactions', 'noise_rho', 'curriculum']}
            protocol = dict(reward='Zero-sum visibility with capture credit; no tool bonuses.', training=settings, physicsSHA256=physics, history=['initial'],
                assets={name: dict(sha256=file_hash(root / (name + '.pt')))
                        for name in ['entity', 'initial', 'critic']})
            (root / 'protocol.json').write_text(json.dumps(protocol))
            with contextlib.redirect_stdout(io.StringIO()):
                train(args, stop_requested=lambda: True)
            first = torch.load(root / 'run/latest.pt', weights_only=False)
            self.assertEqual(first['format'], TAG_FORMAT if widened else FORMAT)
            self.assertEqual(first['observationSize'], 220 if widened else 210)
            self.assertEqual(first['noiseRho'], noise_rho)
            self.assertEqual(first['rolloutState']['noises'].shape, (4, 2, 4))
            self.assertEqual(first['curriculumState']['stage'], 0 if widened else 2)
            self.assertEqual(first['log'][-1]['curriculum']['stage'], 'small' if widened else 'full')
            self.assertEqual(first['totalPolicyInteractions'], 2048)
            self.assertEqual(first['pilotEpisodes'], 4)
            self.assertEqual(sum(first['currentPolicyDecisions']) +
                             sum(first['historicalPolicyDecisions']), 2048)
            self.assertLess(first['activePolicySamples'][1], first['currentPolicyDecisions'][1])
            self.assertTrue(all(state['state'] for state in first['optimizers']))
            self.assertTrue((root / 'run/checkpoint-2048.pt').exists())
            load_pair(first)
            args.resume = str(root / 'run/latest.pt')
            with contextlib.redirect_stdout(io.StringIO()):
                train(args)
            second = torch.load(root / 'run/latest.pt', weights_only=False)
            self.assertEqual(second['totalPolicyInteractions'], 4096)
            self.assertEqual(second['pilotEpisodes'], 8)
            self.assertEqual(second['decisions'], 4219)
            self.assertEqual(len(second['provenance']['resumes']), 1)
            self.assertEqual(second['provenance']['resumes'][0]['worldsRestarted'], 0)
            for previous, current in zip(first['optimizers'], second['optimizers']):
                self.assertGreater(max(value['step'] for value in current['state'].values()),
                                   max(value['step'] for value in previous['state'].values()))
            self.assertEqual(first['log'][0], second['log'][0])
            self.assertTrue((root / 'run/checkpoint-4096.pt').exists())
            args.critic_batch_size += 1
            with self.assertRaisesRegex(ValueError, 'setting changed'):
                train(args)


if __name__ == '__main__':
    unittest.main()
