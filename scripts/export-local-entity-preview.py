"""Explicit development-only preview of the completed 16M entity comparison.

Does not select or deploy the newer, still-training larger model.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from entity_actor import load_pair, export_actor, FORMAT
from persistent_train import file_hash


def main():
    spec=importlib.util.spec_from_file_location('exporter',ROOT/'scripts/export-entity-policy.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    folder=ROOT/'output/local-entity-preview';folder.mkdir(exist_ok=True)
    public=ROOT/'public/models'
    entries=[]
    for kind,source in [('trained',ROOT/'output/entity-encoder/comparison/entity/checkpoint-15990784.pt'),
                        ('initial',ROOT/'output/persistent-button-pilot/run/initial.pt')]:
        models,record=load_pair(source)
        training={key:record.get(key,0) for key in ['decisions','totalPolicyInteractions',
            'currentPolicyDecisions','activePolicySamples','seconds']}
        model=dict(format=FORMAT,observationSize=140,physicsObservationSize=138,
            commands=['keep','press','release'],training=training,
            localPreview=dict(only=True,qualified=False,checkpointSHA256=file_hash(source)),
            actors=[export_actor(m) for m in models])
        path=folder/(kind+'.json');path.write_text(json.dumps(model,separators=(',',':'))+'\n')
        fixture=folder/(kind+'.fixtures.json')
        fixture.write_text(json.dumps(exporter.fixtures(models),separators=(',',':'))+'\n')
        report=json.loads(subprocess.check_output(['node','scripts/test-entity-parity.mjs',str(path),str(fixture)],cwd=ROOT))
        digest=file_hash(path);name=f'physical-policy-{"initial-" if kind=="initial" else ""}{digest[:12]}.json'
        (public/name).write_bytes(path.read_bytes())
        report.update(exportSHA256=digest,checkpointSHA256=file_hash(source),bytes=path.stat().st_size)
        parity=name.replace('.json','.parity.json')
        (public/parity).write_text(json.dumps(report,indent=2)+'\n')
        entries.append(dict(id=kind,label='Local preview · 16M entity continuation' if kind=='trained' else 'Before game experience',
            status='DEVELOPMENT' if kind=='trained' else 'UNTRAINED',file=name,bytes=path.stat().st_size,
            sha256=digest,checkpointSHA256=file_hash(source),parityFile=parity,training=training))
    manifest=dict(format=FORMAT,observationSize=140,physicsObservationSize=138,commands=['keep','press','release'],
        status='LOCAL DEVELOPMENT PREVIEW',localPreview=dict(only=True,qualified=False),
        **{key:entries[0][key] for key in ['file','bytes','sha256','training']},checkpoints=entries)
    (public/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    details=dict(training='Local development preview: completed 15,990,784-interaction entity-encoder continuation. Current hider: 5,993,840 decisions; current seeker: 6,012,672 decisions, including 4,004,480 active play samples. A larger 256-memory model is training separately.',
        evaluation='The 96-map development comparison did not establish a clear improvement over the earlier pair. Useful grab/lock strategies remain unproven. This preview is not a qualified release.',
        modes='Seeded sampling uses the learned action distribution. Push-only lets you test whether disabling grab and lock changes the outcome on the same scene.')
    (ROOT/'src/core/policyAsset.js').write_text('export const PHYSICAL_POLICY_FILE = '+json.dumps('models/'+entries[0]['file'])+';\nexport const PHYSICAL_POLICY_DETAILS = '+json.dumps(details,indent=2)+';\n')
    print(json.dumps(dict(checkpoints=[dict(id=e['id'],sha256=e['sha256'],bytes=e['bytes']) for e in entries])))


if __name__=='__main__':main()
