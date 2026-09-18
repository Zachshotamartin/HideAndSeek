"""PopArt invariance, explicit capture objective, and fresh map replay."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from capture_objective import calibrate_capture, future_credit
from central_critic import FEATURES
from level_replay import LevelReplay
from synthetic import prepared_setup
from test_exact_training import small_arguments
from train_entity import train
from value_learning import PopArtCritic, initialize_critic, NORMALIZED_SCHEMA


class LearningOptions(unittest.TestCase):
    def test_capture_credit_matches_discounted_future_and_keeps_zero_sum(self):
        for n in [0, 1, 10, 188, 375, 750]:
            self.assertAlmostEqual(future_credit(n), sum(.998 ** k for k in range(1, n + 1)), places=9)
        reward = np.array([[-751., 751.], [-1., 1.]])
        info = [dict(captured=True, play_steps=750, captureStep=0), dict(captured=False)]
        adjusted = calibrate_capture(reward, info, 'discounted-equivalent')
        np.testing.assert_array_equal(adjusted.sum(1), [0, 0])
        self.assertAlmostEqual(adjusted[0, 1], 1 + future_credit(750))
        np.testing.assert_array_equal(reward, [[-751, 751], [-1, 1]])
        np.testing.assert_array_equal(calibrate_capture(reward, info), reward)

    def test_popart_update_preserves_raw_prediction_with_partial_baseline(self):
        torch.manual_seed(11)
        critic = PopArtCritic(dropout=0, memory_size=64).double().eval()
        with torch.no_grad():
            critic.correction.value[-1].weight.normal_(0, .1)
            critic.correction.value[-1].bias.fill_(.4)
        states = {key: torch.randn(5, *shape, dtype=torch.float64) for key, shape in FEATURES.items()}
        # Visibility/masks are ordinarily 0/1, not arbitrary Gaussian masks.
        for key in states:
            if 'mask' in key.lower():
                states[key] = torch.ones_like(states[key])
        memories = torch.randn(5, 2, 64, dtype=torch.float64)
        partial = torch.tensor([[1., -4.]] * 5, dtype=torch.float64)
        optimizer = torch.optim.AdamW(critic.parameters())
        before = critic(states, memories, partial).detach()
        critic.update_scale(torch.tensor([-100., 5., 30., 60., 90.], dtype=torch.float64), optimizer)
        after = critic(states, memories, partial).detach()
        torch.testing.assert_close(before, after, atol=1e-10, rtol=1e-10)
        self.assertGreater(float(critic.return_std), 1.)

    def test_map_replay_has_fresh_maps_and_reproducible_state(self):
        archive = LevelReplay()
        for seed in range(12):
            config = dict(seed=seed, scenario='rooms' if seed % 2 else 'open', play=188, size=8, n_boxes=2, n_ramps=0, prep=96)
            archive.observe(config, .1)
            archive.observe(config, .8)
        saved = archive.state_dict()
        restored = LevelReplay(saved)
        rng1, rng2 = np.random.default_rng(99), np.random.default_rng(99)
        fresh = lambda: dict(seed=999, scenario='corridors', play=750)
        a = [archive.sample(rng1, fresh) for _ in range(100)]
        b = [restored.sample(rng2, fresh) for _ in range(100)]
        self.assertEqual(a, b)
        self.assertTrue(any(row['seed'] == 999 for row in a))
        self.assertTrue(any(row['seed'] != 999 for row in a))
        self.assertEqual({row['scenario'] for row in a}, {'open', 'rooms', 'corridors'})
        with self.assertRaises(ValueError):
            archive.observe(dict(seed=1760100000), .5)

    def test_combined_options_resume_exactly_with_real_world_resets(self):
        def short_world(*args, **kwargs):
            return dict(scenario='open', size=8, n_boxes=1, n_ramps=0, prep=0, play=8)
        with tempfile.TemporaryDirectory() as folder, patch('train_entity.environment_config', short_world):
            setup = prepared_setup(folder, target=512)
            whole, split = [small_arguments(setup, folder, name) for name in ['whole', 'split']]
            for args in [whole, split]:
                args.value_normalization = 'popart'
                args.map_replay = 'progress'
                args.capture_credit = 'discounted-equivalent'
                args.matchmaking = 'seeker-curriculum'
                protocol = json.loads(Path(args.protocol).read_text())
                protocol['training'] = {key: getattr(args, key) for key in protocol['training']}
                Path(args.protocol).write_text(json.dumps(protocol))
            train(whole)
            split.stop_after_updates = 1
            train(split)
            split.resume = str(Path(split.output) / 'latest.pt')
            split.stop_after_updates = None
            train(split)
            a = torch.load(Path(whole.output) / 'latest.pt', weights_only=False)
            b = torch.load(split.resume, weights_only=False)
            self.assertEqual(a['centralCriticSchema'], NORMALIZED_SCHEMA)
            self.assertTrue(a['levelReplayState']['entries'])
            self.assertEqual(a['levelReplayState'], b['levelReplayState'])
            for pair in [(a['centralCritic'], b['centralCritic']), *zip(a['models'], b['models'])]:
                for key in pair[0]:
                    torch.testing.assert_close(pair[0][key], pair[1][key], atol=0, rtol=0)
            self.assertEqual(a['worldRNG'], b['worldRNG'])
