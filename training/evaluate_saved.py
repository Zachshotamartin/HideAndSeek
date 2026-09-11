"""Explicit read-only assessment of a full saved pair against fixed opponents."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from entity_actor import load_pair
from checkpoint_store import copy_immutable, load_native, stage_dependencies
from evaluated_models import REPORT_FORMAT, register
from persistent_evaluate import episode, summarize, plain
from persistent_train import file_hash


ACTORS = None


def initialize(candidate, reference):
    global ACTORS
    learned = load_pair(candidate)[0]
    frozen = load_pair(reference)[0]
    ACTORS = {
        'reference': (frozen, 'learned'),
        'candidate-hider': ([learned[0], frozen[1]], 'learned'),
        'candidate-seeker': ([frozen[0], learned[1]], 'learned'),
        'candidate-pair': (learned, 'learned'),
        'no-hider-tools': (learned, 'no-hider-tools'),
        'no-seeker-tools': (learned, 'no-seeker-tools')}


def evaluate_map(configuration):
    rows = []
    for name, (actors, mode) in ACTORS.items():
        row = episode(actors, configuration['seed'], configuration['scenario'], mode,
                      arena_config=configuration['arenaConfig'])
        row['mode'] = name
        rows.append(row)
    return rows


def main(args):
    cohort = json.loads(Path(args.cohort).read_text())
    maps = cohort['maps']
    if not maps or len({(row['seed'], row['scenario']) for row in maps}) != len(maps):
        raise ValueError('Use distinct declared development map identities')
    for row in maps:
        config = row['arenaConfig']
        if (not 1500000000 <= row['seed'] < 1900000000
                or row['scenario'] not in ['open', 'shelter', 'rooms']
                or not 6 <= config['size'] <= 12
                or not 0 <= config['n_boxes'] <= 8 or not 0 <= config['n_ramps'] <= 2):
            raise ValueError('Use the declared physical range and reserve final-test seeds')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'evaluation.json'
    if report_path.exists():
        raise ValueError('Keep completed evaluations immutable')
    frozen_paths = []
    for path in [args.checkpoint, args.reference]:
        digest = file_hash(path)
        frozen = output / 'inputs' / (digest + '.pt')
        if copy_immutable(path, frozen) != digest:
            raise ValueError('Use an immutable checkpoint; the supplied latest file changed')
        saved = load_native(frozen)
        stage_dependencies(path, saved, output)
        frozen_paths.append(str(frozen.resolve()))
    args.checkpoint, args.reference = frozen_paths
    candidate, reference = [load_pair(path)[1] for path in frozen_paths]
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in [
        'evaluate_saved.py', 'persistent_evaluate.py', 'entity_actor.py',
        'persistent_actor.py', 'actor.py', 'physics.py']}
    if any(saved['provenance']['physicsSHA256'] != sources['physics.py'] for saved in [candidate, reference]):
        raise ValueError('Checkpoint and evaluation physics must agree')
    identity = dict(checkpointSHA256=file_hash(args.checkpoint), referenceSHA256=file_hash(args.reference),
                    maps=maps, sources=sources)
    identity_path = output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Partial evaluation identity changed')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    partial = output / 'episodes.partial.json'
    rows = json.loads(partial.read_text()) if partial.exists() else []
    completed = {(row['seed'], row['scenario']) for row in rows}
    if len(rows) != 6 * len(completed):
        raise ValueError('Resume only complete evaluation maps')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(args.checkpoint, args.reference)) as executor:
        for result in executor.map(evaluate_map, [row for row in maps if (row['seed'], row['scenario']) not in completed]):
            rows.extend(result)
            temporary = partial.with_suffix('.tmp')
            temporary.write_text(json.dumps(rows, default=plain) + '\n')
            temporary.replace(partial)
            print(json.dumps(dict(completeMaps=len(rows) // 6)), flush=True)
    summary, contrasts = summarize(rows, [
        ('Hider change', 'candidate-hider', 'reference'),
        ('Seeker change', 'reference', 'candidate-seeker'),
        ('Hider grab/lock benefit', 'candidate-pair', 'no-hider-tools'),
        ('Seeker grab/lock benefit', 'no-seeker-tools', 'candidate-pair')])
    for mode, value in summary.items():
        value['completeSearchMisses'] = sum(row['info']['hidden'] == row['info']['play_steps']
                                           for row in rows if row['mode'] == mode)
    report = dict(format=REPORT_FORMAT, **identity, summary=summary, contrasts=contrasts, episodes=rows,
        evaluationActorDecisions=len(rows) * 480,
        sampling='Exact seeded stochastic policy; identical independent role uniform tapes per map across conditions.',
        scope='Explicit fixed-opponent development evaluation. No training-return selection; maps reused for ranking are not held-out evidence.',
        provenance=dict(candidate=candidate['provenance'], reference=reference['provenance']))
    report_path.write_text(json.dumps(report, indent=2, default=plain) + '\n')
    if args.registry:
        print(json.dumps(register(args.checkpoint, args.reference, report_path, args.registry), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'reference', 'cohort', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--registry')
    parser.add_argument('--workers', type=int, default=2)
    main(parser.parse_args())
