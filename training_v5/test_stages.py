"""The arena curriculum advances only on measured seeker success and survives a checkpoint."""
import unittest

import numpy as np

from protocol import environment_config, LONG_PLAY
from stages import MINIMUM_EPISODES, STAGES, THRESHOLD, WINDOW, Progression


class StageTests(unittest.TestCase):
    def test_advance_needs_enough_rounds_and_a_high_find_rate(self):
        progression = Progression()
        self.assertEqual(progression.name, 'small')
        for _ in range(MINIMUM_EPISODES - 1):
            self.assertFalse(progression.observe(True))
        self.assertEqual(progression.stage, 0)
        self.assertTrue(progression.observe(True))
        self.assertEqual(progression.name, 'medium')
        self.assertEqual(progression.episodes, 0)
        self.assertIsNone(progression.find_rate())
        # A low find rate holds the stage however many rounds pass.
        for _ in range(2 * MINIMUM_EPISODES):
            progression.observe(np.random.default_rng(1).random() < THRESHOLD - .3)
        self.assertEqual(progression.stage, 1)
        for _ in range(WINDOW):
            progression.observe(True)
        self.assertEqual(progression.stage, 2)
        for _ in range(MINIMUM_EPISODES + WINDOW):
            self.assertFalse(progression.observe(True), 'the last stage never advances')

    def test_state_round_trip_and_disabled_progression(self):
        progression = Progression()
        for _ in range(37):
            progression.observe(_ % 3 == 0)
        restored = Progression(progression.state_dict())
        self.assertEqual(restored.state_dict(), progression.state_dict())
        self.assertEqual(restored.find_rate(), progression.find_rate())
        disabled = Progression(enabled=False)
        self.assertEqual(disabled.name, 'full')
        self.assertFalse(disabled.observe(True))
        self.assertEqual(disabled.scope(), STAGES[-1])

    def test_scopes_only_narrow_the_arena_distribution(self):
        rng = np.random.default_rng(5)
        small = [environment_config(rng, scope=STAGES[0]) for _ in range(300)]
        self.assertEqual({row['play'] for row in small}, {188})
        self.assertTrue({row['size'] for row in small} <= {8., 9.})
        self.assertTrue(all(row['prep'] == 96 for row in small))
        self.assertTrue(all(2 <= row['n_boxes'] < 6 for row in small))
        full = [environment_config(rng, scope=STAGES[-1]) for _ in range(600)]
        self.assertEqual({row['play'] for row in full}, set(LONG_PLAY))
        self.assertEqual({row['prep'] for row in full if row['play'] == 750}, {300})
        self.assertEqual({row['prep'] for row in full if row['play'] == 375}, {150})
        self.assertEqual({row['size'] for row in full}, {8., 9., 10., 12.})


if __name__ == '__main__':
    unittest.main()
