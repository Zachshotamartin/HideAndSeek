import unittest
import numpy as np,torch,mujoco
from entity_actor import EntityActor
from physics import PhysicsEnv,OBS_DIM
from league_ppo import assign_roles
class V2(unittest.TestCase):
 def test_all_objects_and_permutation(self):
  torch.set_num_threads(1);torch.manual_seed(24);m=EntityActor(64,96,32).eval()
  with torch.no_grad():m.encoder.residual.weight.normal_(0,.1);m.encoder.relation_scale.fill_(1)
  row=torch.randn(1,210);row[:,10]=1;objects=row[:,18:178].reshape(1,10,16);objects[:,:,0]=1
  base=m.encoder(row)
  for _ in range(10):
   other=row.clone();other[:,18:178]=objects[:,torch.randperm(10)].reshape(1,160)
   torch.testing.assert_close(base,m.encoder(other),atol=3e-6,rtol=3e-6)
  other=row.clone();other[:,18+9*16+1]+=1
  self.assertFalse(torch.allclose(base,m.encoder(other)))
  row[:,18:178]=0;self.assertTrue(torch.isfinite(m.encoder(row)).all())
 def test_long_sight_and_occlusion(self):
  e=PhysicsEnv(seed=123,scenario='open',size=12,n_boxes=0,n_ramps=0)
  e.data.qpos[:8]=[2,6,.25,0,10,6,.25,np.pi];e.t=e.prep;mujoco.mj_forward(e.model,e.data);e._senses()
  self.assertTrue(e._sees(1,e.agent_bodies[0]));self.assertEqual(e.observe().shape,(2,208));self.assertTrue(np.isfinite(e.observe()).all())
  e.data.qpos[7]=0;mujoco.mj_forward(e.model,e.data);e._senses();self.assertFalse(e._sees(1,e.agent_bodies[0]))
 def test_visibility_only_and_preparation(self):
  e=PhysicsEnv(seed=341,scenario='shelter',size=10,n_boxes=6,n_ramps=2)
  for t in range(240):
   _,r,done,info=e.step(np.zeros((2,5)))
   self.assertEqual(float(r.sum()),0)
   if t<96:np.testing.assert_array_equal(r,[0,0])
   else:np.testing.assert_array_equal(r,[-1,1] if info['visible'] else [1,-1])
  self.assertTrue(done)
 def test_real_grab_lock_and_ramp_tools(self):
  import json
  from pathlib import Path
  from physics import generate_arena
  f=json.loads(Path('training/fixtures/native-physics.json').read_text())
  e=PhysicsEnv(arena=f['arena'],prep=f['prep'],play=f['play'])
  for action in f['actions']:e.step(action)
  self.assertGreater(e.grab_events[0],0);self.assertGreater(e.lock_events[0],0)
  self.assertGreater(max(e.info()['object_displacement']),.05)
  arena=generate_arena(4,'open',8,0,0)
  arena['agents']=[dict(position=[2.6,4,.25],yaw=0),dict(position=[6.5,6.5,.25],yaw=0)]
  arena['objects']=[dict(id='ramp',kind='ramp',position=[4,4,.353],size=[1.7,1.2,.7],yaw=0,mass=1.2)]
  e=PhysicsEnv(arena=arena,prep=0,play=100,immovable=True);peak=0
  for _ in range(50):e.step([[1,0,0,0,0],[0,0,0,0,0]]);peak=max(peak,e.data.qpos[2])
  self.assertGreater(peak,.65)
 def test_barrier_changes_visibility(self):
  from physics import generate_arena
  arena=generate_arena(4,'open',8,0,0)
  arena['agents']=[dict(position=[2,4,.25],yaw=0),dict(position=[6,4,.25],yaw=np.pi)]
  arena['objects']=[dict(id='barrier',kind='plank',position=[4,4,.353],size=[.25,1.7,.7],yaw=0,mass=1.8)]
  e=PhysicsEnv(arena=arena,prep=0)
  self.assertFalse(e._sees(1,e.agent_bodies[0]));self.assertTrue(e._sees(1,e.agent_bodies[0],ignore_props=True))
  e.data.qpos[9]=6;mujoco.mj_forward(e.model,e.data)
  self.assertTrue(e._sees(1,e.agent_bodies[0]))
 def test_league_distribution(self):
  roles=assign_roles(np.random.default_rng(31),20000,'B',8)
  self.assertLess(abs(float(np.mean(np.all(roles<0,axis=1)))-.8),.02)
if __name__=='__main__':unittest.main()
