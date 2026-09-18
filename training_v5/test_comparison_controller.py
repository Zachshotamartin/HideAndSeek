"""A paused comparison keeps its declared protocol and comparable batch units."""
import argparse
import tempfile
import unittest
from pathlib import Path
from compare_curriculum import arms_for, prepare_trial
from synthetic import synthetic_source, synthetic_cohort


class ComparisonController(unittest.TestCase):
    def test_repreparation_is_idempotent_after_a_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = synthetic_source(root / 'source.pt')
            cohort = synthetic_cohort(root / 'cohort.json')
            heldout = synthetic_source(root / 'heldout.pt', seed=777)
            args = argparse.Namespace(source=str(source), cohort=str(cohort), heldout=str(heldout), history=[], steps=65536)
            arm, overrides = arms_for('research')[0]
            dest = root / 'combined'
            dest.mkdir()
            first = prepare_trial(args, dest, 11, arm, overrides)
            protocol = Path(first['protocol'])
            before = protocol.read_bytes()
            (dest / 'run').mkdir()
            # Model restoration is independently tested with real simulator checkpoints.
            (dest / 'run/latest.pt').touch()
            second = prepare_trial(args, dest, 11, arm, overrides)
            self.assertEqual(protocol.read_bytes(), before)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, 'already-started'):
                prepare_trial(args, dest, 11, arm, dict(overrides, capture_credit='immediate'))

    def test_larger_batch_preserves_opponent_refresh_interval_in_experience(self):
        arms = dict(arms_for('research'))
        baseline_interactions = 32 * 256 * 2
        large_interactions = arms['large-batch']['envs'] * 256 * 2
        self.assertEqual(large_interactions, 4 * baseline_interactions)
        self.assertEqual(large_interactions * arms['large-batch']['snapshot_every'], baseline_interactions * 4)
        self.assertEqual(arms['large-batch']['sequence_batch'] * 128, 4 * 32 * 128)
