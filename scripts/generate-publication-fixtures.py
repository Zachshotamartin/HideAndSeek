"""Reproducible independent native truth for the current browser's v4 contract.
Does not load or modify training checkpoints. Uses only committed actor exports.
"""
import sys,json,hashlib,math
from pathlib import Path
import numpy as np,torch,mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'training_v5'));sys.path.insert(0,str(ROOT/'tests/fixtures/native-known'))
from physics import generate_arena,OBS_DIM
from known_opponent import PhysicsEnv
from entity_actor import EntityActor
from persistent_actor import advance_buttons
from sample import sample

torch.set_num_threads(1)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
sources={name:digest(ROOT/'tests/fixtures/native-known'/name) for name in ['physics.py','entity_actor.py','persistent_actor.py','sample.py','known_opponent.py']}
arena=generate_arena(923,'open',8,2,1)
public=ROOT/'public/models';manifest=json.loads((public/'MANIFEST.json').read_text());manifest.update(status='DEVELOPMENT',localPreview=dict(only=False,qualified=False),actionSize=6)
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
   # 132 steps per mode with a 24-step preparation (blind seeker rows) and a
   # mid-rollout memory reset: 528 rows, 48 blind rows and 8 resets per fixture.
   e=PhysicsEnv(arena=arena,prep=24,play=120);memory=[torch.zeros(1,256) for _ in actors];buttons=np.zeros((2,2),np.float32);tapes=[Tape(717+r) for r in range(2)];rows=[]
   for t in range(132):
    physical=e.observe();actions=[]
    for role,a in enumerate(actors):
     reset=t in (0,66)
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
 counts=dict(rows=sum(len(r['rows']) for r in rollouts),blind=sum(1 for r in rollouts for row in r['rows'] if row['observation'][7]<.5 and row['observation'][5]<1),resets=sum(1 for r in rollouts for row in r['rows'] if row['reset']))
 assert counts['rows']>=500 and counts['blind']>=40 and counts['resets']>=8,counts
 write(fixture,dict(modelSHA256=digest(path),sourceRoot='tests/fixtures/native-known/',sources=sources,counts=counts,rollouts=rollouts))
 entry.update(bytes=path.stat().st_size,sha256=digest(path),checkpointSHA256=model.get('localPreview',model.get('provenance',{}))['checkpointSHA256'],parityFixture=str(fixture.relative_to(ROOT)),parityFixtureSHA256=digest(fixture))
write(public/'MANIFEST.json',manifest)
print('Generated exact action and recurrence fixtures.')
