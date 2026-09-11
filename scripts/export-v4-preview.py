"""Export an evaluated development checkpoint with explicit v4 provenance."""
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'training_v4'))
from entity_actor import load_pair,export_actor,FORMAT
from physics import generate_arena
from persistent_train import file_hash
from test_tool_jump import InteractionTests
preview=Path('/Users/zacharymartin/Desktop/Portfolio-about-skills/output/latest-ai-preview/HideAndSeek')
registry=ROOT/'output/tool-control-jump-v4/best'
best=json.loads((registry/'best-evaluated.json').read_text())
source=registry/best['file'];models,record=load_pair(source)
training={k:record.get(k,0) for k in ['decisions','totalPolicyInteractions','currentPolicyDecisions','activePolicySamples','seconds']}
training['newRuleInteractions']=0 if best.get('selection','').startswith('Evaluated fixed reference') else record.get('totalPolicyInteractions',0)
model=dict(format=FORMAT,observationSize=210,physicsObservationSize=208,actionSize=6,commands=['keep','press','release'],actors=[export_actor(m) for m in models],training=training,localPreview=dict(only=True,qualified=False,checkpointSHA256=file_hash(source),evaluation=best['evidence']))
raw=(json.dumps(model,separators=(',',':'))+'\n').encode();digest=hashlib.sha256(raw).hexdigest();name='physical-policy-trained-'+digest[:12]+'.json';public=preview/'public/models';(public/name).write_bytes(raw)
initial_models,initial=load_pair(ROOT/'output/tool-control-jump-v4/prepared/initial.pt');initial_model={**model,'training':dict(decisions=0),'localPreview':dict(only=True,qualified=False,checkpointSHA256=file_hash(ROOT/'output/tool-control-jump-v4/prepared/initial.pt')),'actors':[export_actor(m) for m in initial_models]};raw_initial=(json.dumps(initial_model,separators=(',',':'))+'\n').encode();initial_name='physical-policy-initial-'+hashlib.sha256(raw_initial).hexdigest()[:12]+'.json';(public/initial_name).write_bytes(raw_initial)
entries=[dict(id='trained',label='Evaluated reference · jump untrained' if training['newRuleInteractions']==0 else 'Best evaluated · jump & tool control',file=name,sha256=digest,training=training),dict(id='initial',label='Before training · jump enabled',file=initial_name,training=dict(decisions=0))]
for entry, payload, definition in zip(entries, [raw, raw_initial], [model, initial_model]):
 entry.update(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),checkpointSHA256=definition['localPreview']['checkpointSHA256'])
manifest=dict(format=FORMAT,file=name,sha256=digest,status='LOCAL DEVELOPMENT PREVIEW',localPreview=dict(only=True,qualified=False),observationSize=210,physicsObservationSize=208,actionSize=6,checkpoints=entries,training=training)
(public/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copyfile(registry/best['evidence'],public/'development-v4-evaluation.json')
details=dict(training='Evaluated reference adapted to the new jump and tool rules. It retains previous learned movement; the jump output is untrained. Continued training uses reduced tool exploration. New candidates replace this reference only after a better evaluation.',evaluation='Evaluated against the same migrated reference on 120 layouts. Useful tool strategies and reliable jumping are not yet established.',modes='Both sight overlays are shown. Playback continues until paused or reset; training uses finite episodes for evaluation.')
(preview/'src/core/policyAsset.js').write_text('export const PHYSICAL_POLICY_FILE = '+json.dumps('models/'+name)+';\nexport const PHYSICAL_POLICY_DETAILS = '+json.dumps(details,indent=2)+';\n')
fixtures=[]
for role,m in enumerate(models):
 for seed in [0,1,2]:
  env=InteractionTests().env();physical=env.observe()[role];obs=torch.tensor(np.r_[physical,[seed%2,0]],dtype=torch.float32)[None];mem=torch.zeros(1,256)
  with torch.no_grad():normal,tools,_,state=m(obs,mem);encoded=m.encoder(obs)
  fixtures.append(dict(role=role,observation=obs[0].tolist(),previousMemory=mem[0].tolist(),encoded=encoded[0].tolist(),mean=normal.mean[0].tolist(),memory=state[0].tolist()))
(preview/'parity-fixtures.json').write_text(json.dumps(fixtures))
(preview/'arenas-native.json').write_text(json.dumps([generate_arena(123,x,8,6,2) for x in ['shelter','rooms','open','connected-rooms','corridors','multi-exit']]))
# Physical trajectories, including a jump onto a box and conflicting grabs.
trajectories=[]
for mode in ['jump','grab']:
 env=InteractionTests().env()
 if mode=='jump':env.arena['agents'][0]['position'][0]=2.6;env.reset(arena=env.arena)
 rows=[]
 for t in range(20):
  a=np.zeros((2,6))
  if mode=='jump':a[0,0]=int(1<=t<=7);a[0,5]=int(t==1)
  else:a[:,3]=int(t<8);a[0,4]=int(3<=t<6)
  env.step(a);rows.append(dict(actions=a.tolist(),qpos=env.data.qpos.tolist(),grips=env.grips.copy(),locks=env.locks.tolist(),jumps=env.jump_events.tolist()))
 trajectories.append(dict(arena=env.arena,rows=rows))
(preview/'physics-v4-fixtures.json').write_text(json.dumps(trajectories))
print(json.dumps(dict(file=name,source=str(source),steps=training['totalPolicyInteractions'])))
