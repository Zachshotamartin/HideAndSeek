import unittest
import numpy as np
from protocol import *
class ProtocolTests(unittest.TestCase):
 def test_reset_distribution(self):
  r=np.random.default_rng(31); rows=[environment_config(r,0) for _ in range(1000)]
  self.assertEqual({x['scenario'] for x in rows},set(FAMILIES))
  self.assertEqual({x['play'] for x in rows},{188,375,750})
 def test_balance(self):
  roles=assign_balanced_roles(np.random.default_rng(3),100000,6,.8)
  rates=(roles==-1).mean(0)*[1,.8]
  self.assertLess(abs(rates[0]-rates[1]),.015)
  self.assertGreater(np.mean((roles==-1).all(1)),.69)
 def test_archive(self):
  d=np.zeros((40,10));d[8,0]=100
  keep=archive_indices(d,[[12,-1]])
  self.assertIn(0,keep);self.assertIn(8,keep);self.assertIn(12,keep)
  self.assertTrue(set(range(32,40)).issubset(keep));self.assertLessEqual(len(keep),25)
 def test_extended_cohort_adds_long_play_maps_with_fresh_seeds(self):
  base=[dict(seed=1750100000+i,scenario='open',arenaConfig=dict(size=8,n_boxes=3,n_ramps=1)) for i in range(24)]
  maps=extended_cohort(base)
  self.assertEqual(len(maps),48)
  extra=maps[24:]
  self.assertEqual({m['arenaConfig']['play'] for m in extra},set(LONG_PLAY))
  self.assertTrue(all(1500000000<=m['seed']<1900000000 for m in extra))
  self.assertEqual(len({(m['seed'],m['scenario']) for m in maps}),48)
  with self.assertRaises(ValueError):extended_cohort(base,seed_base=1750100000)
 def test_pilot_exercises_archive_truncation(self):
  self.assertTrue(archive_exercised(160,4,1,16))
  self.assertFalse(archive_exercised(160,16,2,24))
if __name__=='__main__':unittest.main()
