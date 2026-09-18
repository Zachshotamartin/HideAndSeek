"""Generate native truth fixtures for the same scene, controls and observations."""
import json,sys,numpy as np
from physics import PhysicsEnv,generate_arena
arena=generate_arena(923,'open',8,2,1)
arena['agents']=[dict(position=[2,3,.25],yaw=0),dict(position=[6,5,.25],yaw=3.141592653589793)]
arena['objects'][0].update(position=[2.66,3,.353],size=[.7,.7,.7],kind='box',yaw=0,mass=1.2)
e=PhysicsEnv(arena=arena,prep=12,play=52)
frames=[];actions=[]
for t in range(64):
 a=np.array([[.7 if t<18 else -.3,.1,.15 if t<14 else 0,1 if t<23 else 0,1 if 18<=t<28 else 0],[.5,.25,-.1,0,0]])
 actions.append(a.tolist());obs,rew,done,info=e.step(a)
 frames.append(dict(qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),obs=obs.tolist(),info=info,grips=list(e.grips),locks=e.locks.tolist()))
with open('training/fixtures/native-physics.json','w')as f:json.dump(dict(engine='MuJoCo3.13.0',arena=arena,prep=12,play=52,actions=actions,frames=frames),f)
print('Exported',len(frames),'frames; grabs',e.grab_events,'locks',e.lock_events,'max motion',max(e.info()['object_displacement']))
