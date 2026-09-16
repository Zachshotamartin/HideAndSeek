"""Export an evaluated tag-round (v6) pair for the browser without touching training.

Reads a registry written by training_v5/evaluated_models.py (``best-evaluated.json``
plus the immutable checkpoint it names), writes the actor-only model file, the
manifest entry, the evidence copy and ``src/core/policyAsset.js``. The export is a
development preview and is never marked qualified.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training_v5'))
from capture import CAPTURE_DISTANCE                                   # noqa: E402
from entity_actor import TAG_FORMAT, export_actor, load_pair          # noqa: E402
from game import CLOCK, MEMORY_AGE, MINIMUM_PREP, PREP_FRACTION, SCHEMA  # noqa: E402

COMMANDS = ['keep', 'press', 'release']
COUNTERS = ['decisions', 'totalPolicyInteractions', 'currentPolicyDecisions', 'activePolicySamples', 'seconds']
ROUND_PLAY = 375


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_best(registry):
    best = json.loads((registry / 'best-evaluated.json').read_text())
    source = registry / best['file']
    if not best.get('eligible'):
        raise ValueError('Only an eligible evaluated pair may be exported')
    if sha256(source.read_bytes()) != best['sha256']:
        raise ValueError('The registry checkpoint does not match its recorded hash')
    return best, source


def model_document(actors, record, best):
    if record['format'] != TAG_FORMAT or record.get('schema') != SCHEMA:
        raise ValueError('Only a tag-round (v6) pair can be exported with this script')
    training = {key: record.get(key, 0) for key in COUNTERS}
    return dict(
        format=TAG_FORMAT, observationSchema=SCHEMA, observationSize=record['observationSize'],
        physicsObservationSize=record['physicsObservationSize'], noiseRho=record['noiseRho'], actionSize=6, commands=COMMANDS,
        actors=[export_actor(actor) for actor in actors], training=training,
        gameRules=dict(captureDistance=CAPTURE_DISTANCE, preparationFraction=PREP_FRACTION, minimumPreparationSteps=MINIMUM_PREP,
                       roundPlaySteps=ROUND_PLAY, memoryAgeSeconds=MEMORY_AGE, clockSeconds=CLOCK),
        provenance=dict(checkpointSHA256=best['sha256'], selection=best['selection'], evidence=best.get('evidence')),
        qualified=False)


def manifest_entry(name, raw, model, best, label):
    return dict(id='trained', label=label, file=name, bytes=len(raw), sha256=sha256(raw), training=model['training'],
                checkpointSHA256=best['sha256'])


def details(best, model):
    steps = model['training'].get('totalPolicyInteractions', 0)
    return dict(
        training=f'Frozen development pair: {steps:,} new self-play interactions of the tag-round game. Both policies use '
                 'recurrent memory, restricted physical observations, the remembered last sighting and the clock.',
        evaluation=f"Selected from completed fixed-opponent evaluations: mean utility {best['score']:.3f}. At least one role "
                   'improved with a 95% bootstrap interval excluding zero, with no clear regression in the other. This is '
                   'development evidence, not a claim of mastered tool use.',
        modes=f'Rounds last {ROUND_PLAY * .08:.0f} s after a {max(MINIMUM_PREP, round(PREP_FRACTION * ROUND_PLAY)) * .08:.0f} s '
              'preparation. A tag ends the round; the next one starts on its own. Each agent can move, turn, grab, lock and '
              'jump onto low objects.')


def export(registry, public, asset_module, label):
    best, source = load_best(registry)
    actors, record = load_pair(source)
    model = model_document(actors, record, best)
    raw = (json.dumps(model, separators=(',', ':')) + '\n').encode()
    digest = sha256(raw)
    name = f'physical-policy-tag-{digest[:12]}.json'
    public.mkdir(parents=True, exist_ok=True)
    (public / name).write_bytes(raw)
    manifest_path = public / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(checkpoints=[])
    others = [entry for entry in manifest.get('checkpoints', []) if entry.get('id') != 'trained']
    entry = manifest_entry(name, raw, model, best, label)
    manifest.update(format=TAG_FORMAT, observationSchema=SCHEMA, file=name, sha256=digest, bytes=len(raw),
                    observationSize=model['observationSize'], physicsObservationSize=model['physicsObservationSize'],
                    actionSize=6, training=model['training'], status='DEVELOPMENT', localPreview=dict(only=False, qualified=False),
                    checkpoints=[entry, *others])
    evidence = None
    if best.get('evidence') and (registry / best['evidence']).exists():
        report = (registry / best['evidence']).read_bytes()
        evidence = 'development-v6-evaluation.json'
        (public / evidence).write_bytes(report)
        manifest['evaluation'] = dict(file=evidence, sha256=sha256(report), shippedRole='candidate',
                                      referenceSHA256=json.loads(report)['referenceSHA256'], candidateSHA256=best['sha256'],
                                      maps=len(json.loads(report)['maps']), episodes=len(json.loads(report)['episodes']),
                                      note='The shipped pair is the evaluated candidate; development evidence, not qualification.')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    asset_module.write_text('export const PHYSICAL_POLICY_FILE = ' + json.dumps('models/' + name) + ';\n'
                            + 'export const PHYSICAL_POLICY_DETAILS = ' + json.dumps(details(best, model), indent=2) + ';\n')
    return dict(file=name, sha256=digest, bytes=len(raw), evidence=evidence, checkpointSHA256=best['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry', help='Directory holding best-evaluated.json and the evaluated checkpoints')
    parser.add_argument('--public', default=str(ROOT / 'public/models'))
    parser.add_argument('--asset-module', default=str(ROOT / 'src/core/policyAsset.js'))
    parser.add_argument('--label', default='Best evaluated tag-round pair')
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = export(Path(args.registry), Path(args.public), Path(args.asset_module), args.label)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
