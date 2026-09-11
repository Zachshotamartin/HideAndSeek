"""A run split by a checkpoint must equal the same run done in one piece."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch

from synthetic import prepared_setup
from train_entity import train

SMALL_RUN = dict(envs=2, workers=1, horizon=64, sequence_length=32, burn_in=8, snapshot_every=1, archive_limit=13,
                 sequence_batch=4, epochs=1, kl_limit=1, target_interactions=512, stop_after_updates=None)


def small_arguments(setup, folder, name):
    a = argparse.Namespace(**setup['arguments'])
    for key, value in SMALL_RUN.items():
        setattr(a, key, value)
    a.output = str(Path(folder) / name)
    protocol = json.loads(Path(a.protocol).read_text())
    protocol['training'] = {k: getattr(a, k) for k in protocol['training']}
    a.protocol = str(Path(folder) / (name + '.json'))
    Path(a.protocol).write_text(json.dumps(protocol))
    return a


class ExactTraining(unittest.TestCase):
    def test_split_optimizer_physics_and_burn_in_prefix_match(self):
        with tempfile.TemporaryDirectory() as folder:
            setup = prepared_setup(folder, target=512)
            whole = small_arguments(setup, folder, 'whole')
            train(whole)
            w = torch.load(Path(whole.output) / 'latest.pt', weights_only=False)
            split = small_arguments(setup, folder, 'split')
            split.stop_after_updates = 1
            train(split)
            split.resume = str(Path(split.output) / 'latest.pt')
            split.stop_after_updates = None
            train(split)
            s = torch.load(split.resume, weights_only=False)
            for wm, sm in zip(w['models'], s['models']):
                for k in wm:
                    torch.testing.assert_close(wm[k], sm[k], atol=0, rtol=0)
            for x, y in zip(w['rolloutState']['memories'], s['rolloutState']['memories']):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            for x, y in zip(w['rolloutState']['burnInPrefix'], s['rolloutState']['burnInPrefix']):
                for a, b in zip(x, y):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertEqual(w['rolloutState']['recent'], s['rolloutState']['recent'])
            self.assertEqual(w['log'][-1]['hider']['burnInSteps'], s['log'][-1]['hider']['burnInSteps'])
            self.assertGreater(w['log'][-1]['hider']['burnInSteps'], 0)
            self.assertEqual(w['pilotUpdates'], 2)
            self.assertEqual(len(w['leagueModels']), 3)


if __name__ == '__main__':
    unittest.main()
