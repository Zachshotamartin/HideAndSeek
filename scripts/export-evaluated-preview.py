"""Export an explicitly selected, evaluated v5 pair without altering training."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'training_v5'))
from entity_actor import load_pair,export_actor,FORMAT
p=argparse.ArgumentParser();p.add_argument('registry');a=p.parse_args();registry=Path(a.registry)
best=json.loads((registry/'best-evaluated.json').read_text());source=registry/best['file']
assert best['eligible'] and hashlib.sha256(source.read_bytes()).hexdigest()==best['sha256']
actors,record=load_pair(source);torch.set_num_threads(1)
training={k:record.get(k,0)for k in ['decisions','totalPolicyInteractions','currentPolicyDecisions','activePolicySamples','seconds']}
model=dict(format=FORMAT,observationSize=210,physicsObservationSize=208,actionSize=6,commands=['keep','press','release'],actors=[export_actor(x)for x in actors],training=training,provenance=dict(checkpointSHA256=best['sha256'],selection=best['selection']),qualified=False)
raw=(json.dumps(model,separators=(',',':'))+'\n').encode();sha=hashlib.sha256(raw).hexdigest();name=f'physical-policy-trained-{sha[:12]}.json';public=ROOT/'public/models';(public/name).write_bytes(raw)
manifest=json.loads((public/'MANIFEST.json').read_text());initial=next(e for e in manifest['checkpoints']if e['id']=='initial')
entry=dict(id='trained',label='Best evaluated · 10.5M new interactions',file=name,bytes=len(raw),sha256=sha,training=training,checkpointSHA256=best['sha256'])
manifest.update(status='DEVELOPMENT',localPreview=dict(only=False,qualified=False),file=name,sha256=sha,bytes=len(raw),training=training,checkpoints=[entry,initial])
(public/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');report=json.loads((registry/best['evidence']).read_text());(public/'development-v5-evaluation.json').write_text(json.dumps(report)+'\n')
details=dict(training='Frozen development pair: 10,485,760 new self-play interactions in the shorter-round comparison. Both policies use recurrent memory and restricted physical observations. Training continues separately.',evaluation=f"Selected from completed fixed-opponent evaluations: mean utility {best['score']:.3f}. At least one role improved with a 95% bootstrap interval excluding zero, with no clear regression in the other. This is development evidence, not a claim of mastered tool use.",modes='Both sight overlays are shown. Playback continues until paused or reset. Each agent can move, turn, grab, lock and jump onto low objects.')
(ROOT/'src/core/policyAsset.js').write_text('export const PHYSICAL_POLICY_FILE = '+json.dumps('models/'+name)+';\nexport const PHYSICAL_POLICY_DETAILS = '+json.dumps(details,indent=2)+';\n')
print('Exported',name,training)
