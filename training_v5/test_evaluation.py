"""Evaluation interventions on a synthetic prepared pair."""
import tempfile
import unittest
from pathlib import Path

from entity_actor import load_pair
from persistent_evaluate import BLACKOUT_STEPS, episode
from synthetic import prepared_setup


class EvaluationTests(unittest.TestCase):
    def test_variable_time_memory_and_blackout_interventions(self):
        with tempfile.TemporaryDirectory() as folder:
            prepared_setup(folder, target=512)
            models = load_pair(str(Path(folder) / 'run/prepared/entity.pt'))[0]
            for mode in ['learned', 'no-seeker-memory', 'seeker-blackout']:
                r = episode(models, 1750500000, 'multi-exit', mode, arena_config=dict(size=8, n_boxes=2, n_ramps=1, prep=4, play=15))
                self.assertEqual(r['info']['play_steps'], 15)
                self.assertIn('reacquisitionSeconds', r['roles'][1])
                self.assertIn('captured', r)
                self.assertIn('playStuckFrames', r['roles'][0])
                self.assertLessEqual(r['blackoutTicks'], BLACKOUT_STEPS)
                if mode != 'seeker-blackout':
                    self.assertEqual(r['blackoutTicks'], 0)
            with self.assertRaises(ValueError):
                episode(models, 1750500000, 'open', 'learned', arena_config=dict(size=8, walls=3))


if __name__ == '__main__':
    unittest.main()
