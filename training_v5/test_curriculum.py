"""Outcome sampling and blocked movement invariants, independent of learned weights."""
import unittest
import numpy as np
from matchmaking import SeekerMatchmaker
from motion_diagnostics import blocked_attempt, blocked_adjustment


class CurriculumTests(unittest.TestCase):
    def test_mixed_difficulty_keeps_all_opponents(self):
        m = SeekerMatchmaker(3, dict(success=[0., .5, 1.], counts=[50]*3))
        p = m.probabilities()
        self.assertGreater(p[1], p[0])
        self.assertTrue((p >= .25/3).all())
        roles = np.array([[-1, -1], [-1, 0], [0, -1]] * 500)
        result = m.assign(roles, np.random.default_rng(6))
        np.testing.assert_array_equal(result[::3], roles[::3])
        np.testing.assert_array_equal(result[1::3], roles[1::3])
        self.assertEqual(set(result[2::3, 0]), {0, 1, 2})

    def test_outcomes_pruning_and_resume_preserve_identity(self):
        m = SeekerMatchmaker(3)
        m.observe(2, True)
        m.append()
        m.prune([2, 0, 3])
        restored = SeekerMatchmaker(3, m.state_dict())
        self.assertGreater(restored.success[0], restored.success[1])
        np.testing.assert_array_equal(restored.probabilities(), m.probabilities())
        self.assertEqual(restored.counts, [1, 0, 0])

    def test_standing_sliding_and_successful_pushing_are_not_blocked(self):
        self.assertFalse(blocked_attempt([0, 0], 0, True))
        self.assertFalse(blocked_attempt([1, 0], .2, True))
        self.assertFalse(blocked_attempt([1, 0], 0, True, .2))
        self.assertFalse(blocked_attempt([1, 0], 0, False))
        self.assertTrue(blocked_attempt([1, 0], 0, True))
        np.testing.assert_array_equal(blocked_adjustment([[1, 0], [0, 1], [1, 1]], .02),
                                      [[-.02, .02], [.02, -.02], [0, 0]])
