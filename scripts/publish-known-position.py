"""Export the eligible evaluated champion for the explicit known-position task.

Usage: PYTHON scripts/publish-known-position.py REGISTRY_DIRECTORY
Reads an immutable selected checkpoint; never opens a trainer checkpoint for writing.
Afterward regenerate publication fixtures and run the JS test suite before release.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/fixtures/native-known'))
from entity_actor import FORMAT, export_actor, load_pair
from known_opponent import OBSERVATION_SCHEMA, TASK_CONTRACT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


def public_paths(value):
    if isinstance(value, dict):
        return {k: public_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public_paths(v) for v in value]
    return Path(value).name if isinstance(value, str) and value.startswith('/') else value


def publish(registry):
    selected = json.loads((registry / 'best-evaluated.json').read_text())
    checkpoint = registry / selected['file']
    evidence = registry / selected['evidence']
    assert selected['eligible'] and selected['entirePairPreserved']
    assert digest(checkpoint) == selected['sha256']
    assert digest(evidence) == selected['reportSHA256']
    torch.set_num_threads(1)
    actors, saved = load_pair(checkpoint)
    assert digest(checkpoint) == selected['sha256']
    assert saved['observationSchema'] == OBSERVATION_SCHEMA
    assert saved['taskContract'] == TASK_CONTRACT
    report = json.loads(evidence.read_text())
    assert report['checkpointSHA256'] == selected['sha256']
    assert report['taskContract'] == TASK_CONTRACT
    public = ROOT / 'public/models'
    manifest = json.loads((public / 'MANIFEST.json').read_text())
    initial_entry = next(e for e in manifest['checkpoints'] if e['id'] == 'initial')
    initial = json.loads((public / initial_entry['file']).read_text())
    initial.update(observationSchema=OBSERVATION_SCHEMA, taskContract=TASK_CONTRACT)
    # Same genuine zero-experience tensors, evaluated under the new observation
    # contract. No optimization or new actor initialization takes place here.
    initial['provenance'] = {**initial.get('provenance', initial.get('localPreview', {})),
                            'observationMigration': 'Original zero-experience actor tensors unchanged; known-position observations enabled for comparison.'}
    training = {key: saved[key] for key in ['decisions', 'totalPolicyInteractions',
                 'currentPolicyDecisions', 'activePolicySamples', 'seconds']}
    model = dict(format=FORMAT, observationSchema=OBSERVATION_SCHEMA,
                 taskContract=TASK_CONTRACT, observationSize=210, physicsObservationSize=208,
                 actionSize=6, commands=['keep', 'press', 'release'], qualified=False,
                 actors=[export_actor(actor) for actor in actors], training=training,
                 provenance={'checkpointSHA256': selected['sha256'],
                             'selection': selected['selection'],
                             'taskTransitions': public_paths(saved['provenance']['taskTransitions'])})
    details = dict(training=f"{training['totalPolicyInteractions']:,} cumulative self-play interactions, including inherited training. Frozen September 15 evaluated checkpoint; training continues separately.",
                   evaluation=f"Best eligible pair within the known-position visibility task; mean fixed-opponent utility {selected['score']:.3f} across {len(report['maps'])} development maps. Development selection, not a final test or proof of mastered tool use.",
                   modes='Both agents know the opponent’s position behind walls. Visibility still depends on sight and occlusion. Playback continues until paused or reset.')
    entries = []
    for role, value in [('trained', model), ('initial', initial)]:
        raw = encode(value)
        sha = hashlib.sha256(raw).hexdigest()
        name = f"physical-policy-{'initial-' if role == 'initial' else ''}{sha[:12]}.json"
        (public / name).write_bytes(raw)
        entries.append(dict(id=role, label=(f"Best evaluated · {training['totalPolicyInteractions']/1e6:.2f}M interactions" if role == 'trained' else 'Before training · same position sensors'),
                            file=name, bytes=len(raw), sha256=sha, training=value['training'],
                            checkpointSHA256=value['provenance']['checkpointSHA256'],
                            details=(details if role == 'trained' else dict(training='Original zero-experience actor tensors, with the current known-position observation contract.', evaluation='Untrained comparison; not a validated candidate.'))))
    raw = encode(public_paths(report))
    sha = hashlib.sha256(raw).hexdigest()
    name = f'development-fixed-opponent-{sha[:12]}.json'
    (public / name).write_bytes(raw)
    manifest.update(file=entries[0]['file'], sha256=entries[0]['sha256'], bytes=entries[0]['bytes'],
                    training=training, checkpoints=entries, observationSchema=OBSERVATION_SCHEMA,
                    taskContract=TASK_CONTRACT, status='DEVELOPMENT', qualified=False,
                    localPreview=dict(only=False, qualified=False),
                    evaluation=dict(file=name, bytes=len(raw), sha256=sha,
                        sourceReportSHA256=selected['reportSHA256'], shippedRole='candidate',
                        referenceSHA256=report['referenceSHA256'], candidateSHA256=selected['sha256'],
                        maps=len(report['maps']), episodes=len(report['episodes']),
                        note='Fixed-opponent development evidence, not qualification. Local path strings reduced to basenames; all measurements and hashes retained.'))
    (public / 'MANIFEST.json').write_bytes(encode(manifest))
    (ROOT / 'src/core/policyAsset.js').write_text('export const PHYSICAL_POLICY_FILE = ' + json.dumps('models/' + entries[0]['file']) + ';\nexport const PHYSICAL_POLICY_DETAILS = ' + json.dumps(details) + ';\n')
    print(json.dumps({'checkpoint': selected['sha256'], 'interactions': training['totalPolicyInteractions'], 'score': selected['score'], 'file': entries[0]['file']}, indent=2))


if __name__ == '__main__':
    publish(Path(sys.argv[1]))
