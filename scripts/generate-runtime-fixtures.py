"""Reproducible independent native truth for the current browser's v4 contract.
Does not load or modify training checkpoints. Uses only committed actor exports.
"""
import sys,json,hashlib,math
from pathlib import Path
import numpy as np,torch,mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'training_v4'))
from physics import PhysicsEnv,generate_arena,OBS_DIM
from entity_actor import EntityActor
from persistent_actor import advance_buttons
from persistent_evaluate import sample

torch.set_num_threads(1)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
sources={name:digest(ROOT/'training_v4'/name) for name in ['physics.py','entity_actor.py','persistent_actor.py','persistent_evaluate.py']}
sources['../scripts/generate-runtime-fixtures.py']=digest(Path(__file__))
arena=generate_arena(923,'open',8,2,1)
arena['agents']=[dict(position=[2,3,.25],yaw=0),dict(position=[6,5,.25],yaw=math.pi)]
arena['objects'][0].update(position=[2.66,3,.353],size=[.7,.7,.7],kind='box',yaw=0,mass=1.2)
e=PhysicsEnv(arena=arena,prep=12,play=52);frames=[];actions=[]
for t in range(64):
 a=np.array([[.7 if t<18 else -.3,.1,.15 if t<14 else 0,int(t<23),int(18<=t<28),0],[.5,.25,-.1,0,0,0]])
 obs,_,_,info=e.step(a);actions.append(a.tolist());frames.append(dict(qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),obs=obs.tolist(),grips=e.grips.copy(),locks=e.locks.tolist(),jumps=e.jump_events.tolist()))
assert e.grab_events[0]>0 and e.lock_events[0]>0
write(ROOT/'training/fixtures/native-physics.json',dict(engine='MuJoCo'+mujoco.__version__,sources=sources,observationSize=OBS_DIM,actionSize=6,arena=arena,prep=12,play=52,actions=actions,frames=frames))
# Distinct airborne contact case: jump onto a box, not a no-op sixth action.
jump_arena=generate_arena(31,'open',8,0,0);jump_arena['agents']=[dict(position=[2.6,3,.25],yaw=0),dict(position=[4.25,3,.25],yaw=math.pi)];jump_arena['objects']=[dict(id='test',kind='box',position=[3.62,3,.353],size=[.7,.7,.7],yaw=0,mass=1.2)]
e=PhysicsEnv(arena=jump_arena,prep=0,play=100);frames=[]
for t in range(20):
 a=np.zeros((2,6));a[0,0]=int(1<=t<=7);a[0,5]=int(t==1);obs,_,_,_=e.step(a)
 frames.append(dict(action=a.tolist(),qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),obs=obs.tolist(),jumps=e.jump_events.tolist()))
assert e.jump_events[0]==1 and e.data.qpos[2]>.9
write(ROOT/'tests/fixtures/jump-native.json',dict(sources=sources,arena=jump_arena,prep=0,play=100,frames=frames))
public=ROOT/'public/models';manifest=json.loads((public/'MANIFEST.json').read_text());manifest.update(status='LOCAL DEVELOPMENT PREVIEW',localPreview=dict(only=True,qualified=False),actionSize=6)
class Tape:
 def __init__(self,seed):self.rng=np.random.default_rng(seed);self.values=[]
 def random(self,n):v=self.rng.random(n);self.values=v.tolist();return v
for entry in manifest['checkpoints']:
 path=public/entry['file'];model=json.loads(path.read_text());actors=[]
 for definition in model['actors']:
  a=EntityActor(definition['hiddenSize'],definition['encoderSize'],definition['embeddingSize'])
  missing=a.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in definition['weights'].items()},strict=False)
  assert set(missing.missing_keys)=={'value.weight','value.bias'} and not missing.unexpected_keys
  actors.append(a.eval())
 rollouts=[]
 with torch.no_grad():
  for deterministic in [False,True]:
   e=PhysicsEnv(arena=arena,prep=4,play=40);memory=[torch.zeros(1,256) for _ in actors];buttons=np.zeros((2,2),np.float32);tapes=[Tape(717+r) for r in range(2)];rows=[]
   for t in range(16):
    physical=e.observe();actions=[]
    for role,a in enumerate(actors):
     reset=t in (0,8)
     if reset:memory[role].zero_();buttons[role].fill(0)
     blind=physical[role,7]<.5 and physical[role,5]<1
     previous=np.zeros(2,np.float32) if blind else buttons[role].copy()
     obs=torch.tensor(np.r_[physical[role],previous],dtype=torch.float32)[None]
     normal,tools,_,next_memory=a(obs,memory[role]);logits=a.tools(next_memory)[0].tolist()
     if deterministic:
      commands=tools.probs[0].argmax(-1).numpy();b=advance_buttons(previous,commands,blind);action=np.zeros(6,np.float32);action[[0,1,2,5]]=normal.mean[0].tanh().numpy();action[3:5]=b
      if blind:action.fill(0)
      tape=[]
     else:
      action,actual_memory,b,commands=sample(a,physical[role],memory[role],buttons[role],tapes[role]);torch.testing.assert_close(actual_memory,next_memory);tape=tapes[role].values
     rows.append(dict(role=role,reset=reset,observation=physical[role].tolist(),tape=tape,action=action.tolist(),memory=next_memory[0].tolist(),mean=normal.mean[0].tolist(),toolLogits=logits,commands=commands.tolist(),buttons=b.tolist()))
     memory[role]=next_memory;buttons[role]=b;actions.append(action)
    e.step(actions)
   rollouts.append(dict(deterministic=deterministic,rows=rows))
 fixture=ROOT/'tests/fixtures'/('policy-'+entry['id']+'-native.json')
 write(fixture,dict(modelSHA256=digest(path),sources=sources,rollouts=rollouts))
 entry.update(bytes=path.stat().st_size,sha256=digest(path),checkpointSHA256=model['localPreview']['checkpointSHA256'],parityFixture=str(fixture.relative_to(ROOT)),parityFixtureSHA256=digest(fixture))
manifest['sha256']=digest(public/manifest['file']);write(public/'MANIFEST.json',manifest)
print('Generated v4 physical and live-weight policy fixtures; refreshed manifest integrity metadata.')
