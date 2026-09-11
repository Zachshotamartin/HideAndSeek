import unittest
import numpy as np
import torch
from evaluate_object_mobility import episode, neutral_lock_cues
from persistent_evaluate import episode as original_episode
from persistent_actor import PersistentActor
from physics import PhysicsEnv


class ObjectMobilityTests(unittest.TestCase):
    def test_neutral_fixed_cues_match_initial_push_only_observation(self):
        moving = PhysicsEnv(seed=7413, scenario='open', disable_tools=True)
        fixed = PhysicsEnv(seed=7413, scenario='open', disable_tools=True, immovable=True)
        np.testing.assert_array_equal(moving.observe(), neutral_lock_cues(fixed.observe()))
        np.testing.assert_array_equal(fixed.observe()[1, 10:], np.r_[np.zeros(104), np.ones(24)])

    def test_probes_leave_original_learned_trajectory_unchanged(self):
        torch.manual_seed(31)
        actors = [PersistentActor().eval(), PersistentActor().eval()]
        original = original_episode(actors, 7813, 'shelter', 'learned', trace=True)
        checked = episode(actors, 7813, 'shelter', 'tools', trace=True, probes=True)
        np.testing.assert_array_equal(original['hiddenFraction'], checked['hiddenFraction'])
        for first, second in zip(original['frames'], checked['frames']):
            np.testing.assert_array_equal(first['qpos'], second['qpos'])
            np.testing.assert_array_equal(first['qvel'], second['qvel'])
            np.testing.assert_array_equal(first['actions'], second['actions'])

    def test_explicit_push_only_matches_prior_no_tools_ablation(self):
        torch.manual_seed(37)
        actors = [PersistentActor().eval(), PersistentActor().eval()]
        original = original_episode(actors, 8117, 'open', 'no-tools', trace=True)
        checked = episode(actors, 8117, 'open', 'push-only', trace=True)
        for first, second in zip(original['frames'], checked['frames']):
            np.testing.assert_array_equal(first['qpos'], second['qpos'])
        self.assertEqual(checked['info']['grabs'], [0, 0])
        self.assertEqual(checked['info']['locks'], [0, 0])


if __name__ == '__main__':
    unittest.main()
