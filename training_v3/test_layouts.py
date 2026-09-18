import unittest,collections
import numpy as np
from physics import generate_arena,PhysicsEnv,local_xy
class Layouts(unittest.TestCase):
 def test_connected_clearance_and_variety(self):
  fingerprints=set()
  for scenario in ['connected-rooms','corridors','multi-exit']:
   for seed in range(40):
    a=generate_arena(seed+901,scenario,size=8+seed%5,n_boxes=6,n_ramps=2)
    self.assertLessEqual(len(a['walls']),16)
    size=a['width'];step=.15;coords=np.arange(.3,size-.29,step)
    free=set()
    for i,x in enumerate(coords):
     for j,y in enumerate(coords):
      clear=True
      for w in a['walls']:
       u,v=local_xy([x-w['position'][0],y-w['position'][1]],w['yaw'])
       if np.hypot(max(0,abs(u)-w['size'][0]/2),max(0,abs(v)-w['size'][1]/2))<.27:clear=False;break
      if clear:free.add((i,j))
    seen={next(iter(free))};q=collections.deque(seen)
    while q:
     i,j=q.popleft()
     for point in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
      if point in free and point not in seen:seen.add(point);q.append(point)
    self.assertEqual(len(seen),len(free),(scenario,seed))
    for agent in a['agents']:
     for w in a['walls']+a['objects']:
      u,v=local_xy(np.array(agent['position'])-w['position'],w['yaw'])
      self.assertGreaterEqual(np.hypot(max(0,abs(u)-w['size'][0]/2),max(0,abs(v)-w['size'][1]/2)),.25)
    fingerprints.add(str(a['walls']))
   for seed in range(3):
    e=PhysicsEnv(seed=700+seed,scenario=scenario,size=10,n_boxes=6,n_ramps=2)
    for _ in range(240):obs,reward,done,info=e.step(np.array([[.4,.2,.1,0,0],[-.3,.1,.1,0,0]]))
    self.assertTrue(done);self.assertTrue(np.isfinite(obs).all())
  self.assertEqual(len(fingerprints),120)
if __name__=='__main__':unittest.main()
