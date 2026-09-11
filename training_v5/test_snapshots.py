import unittest
import numpy as np
from physics import PhysicsEnv
from snapshots import dump,restore
class SnapshotTest(unittest.TestCase):
 def test_exact_replay(self):
  e=PhysicsEnv(seed=851,scenario='rooms',prep=5,play=375,n_boxes=5,n_ramps=2)
  rng=np.random.default_rng(65)
  for _ in range(23):e.step(rng.uniform(-1,1,(2,6)))
  snap=dump(e);f=restore(snap)
  for _ in range(20):
   a=rng.uniform(-1,1,(2,6));x=e.step(a);y=f.step(a)
   np.testing.assert_allclose(x[0],y[0],atol=1e-12,rtol=0);np.testing.assert_array_equal(x[1],y[1])
  e.close();f.close()
if __name__=='__main__':unittest.main()
