"""The seeker milestone is measured and recorded; it never stops the controller."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_suite
from train_scaled import Controller, GATE_FOUND_RATE, GATE_INTERACTIONS


class MilestoneTests(unittest.TestCase):
    def controller(self):
        controller = Controller.__new__(Controller)
        controller.status = {}
        controller.stopped = False
        controller.args = SimpleNamespace(stop_after_updates=None)
        controller.save = lambda **changes: controller.status.update(changes)
        return controller

    def measure(self, controller, found, regressions):
        with patch('train_scaled.utility', return_value=dict(seekerFoundRate=found, roleRegressions=regressions)):
            controller.gate({}, GATE_INTERACTIONS)
        return controller.status['gates'][-1]

    def test_a_missed_milestone_is_recorded_and_the_run_keeps_training(self):
        controller = self.controller()
        row = self.measure(controller, GATE_FOUND_RATE - .3, ['Seeker change'])
        self.assertFalse(row['passed'])
        self.assertEqual(len(row['failures']), 2)
        self.assertEqual(row['steps'], GATE_INTERACTIONS)
        self.assertFalse(controller.stopped)
        self.assertEqual(controller.final_phase(), 'completed-awaiting-review')

    def test_a_met_milestone_records_that_it_passed(self):
        controller = self.controller()
        row = self.measure(controller, GATE_FOUND_RATE + .3, [])
        self.assertTrue(row['passed'])
        self.assertEqual(row['failures'], [])
        self.assertFalse(controller.stopped)

    def test_no_control_path_ends_a_run_on_a_milestone(self):
        for phase in ('paused', 'completed-awaiting-review', 'pilot-complete', 'finished-awaiting-review'):
            self.assertNotIn('gate', phase)
        for name in ('train_scaled.py', 'run_suite.py'):
            self.assertNotIn('gate-failed', Path(__file__).with_name(name).read_text(), name)
        self.assertNotIn('gateFailedTrial', Path(run_suite.__file__).read_text())


if __name__ == '__main__':
    unittest.main()
