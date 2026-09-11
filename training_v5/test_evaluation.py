import unittest
from pathlib import Path
from entity_actor import load_pair
from persistent_evaluate import episode
class EvaluationTests(unittest.TestCase):
 def test_variable_time_and_memory_interventions(self):
  models=load_pair('output/v5-smoke/prepared/entity.pt')[0]
  for mode in ['learned','no-seeker-memory']:
   r=episode(models,1750500000,'multi-exit',mode,arena_config=dict(size=8,n_boxes=2,n_ramps=1,prep=4,play=15))
   self.assertEqual(r['info']['play_steps'],15)
   self.assertIn('reacquisitionSeconds',r['roles'][1])
if __name__=='__main__':unittest.main()
