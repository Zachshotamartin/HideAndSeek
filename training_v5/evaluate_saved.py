"""Explicit read-only assessment of a full saved pair against fixed opponents."""
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from checkpoint_store import copy_immutable, load_native, stage_dependencies
from entity_actor import load_pair
from evaluated_models import REPORT_FORMAT, register
from persistent_evaluate import BLACKOUT_STEPS, episode, plain, summarize
from persistent_train import file_hash

ACTORS = None
SCENARIOS = ['shelter', 'rooms', 'open', 'connected-rooms', 'corridors', 'multi-exit']
SOURCE_NAMES = ['evaluate_saved.py', 'persistent_evaluate.py', 'entity_actor.py', 'persistent_actor.py', 'actor.py', 'physics.py']
SEED_RANGE = (1500000000, 1900000000)   # development seeds; final-test seeds are reserved
SIZE_RANGE = (6, 12)
BOX_LIMIT = 8
RAMP_LIMIT = 2
PLAY_LIMIT = 750
TOOL_CONTRASTS = [('Hider tool benefit', 'candidate-pair', 'no-hider-tools'), ('Seeker tool benefit', 'no-seeker-tools', 'candidate-pair')]
CHANGE_CONTRASTS = [('Hider change', 'candidate-hider', 'reference'), ('Seeker change', 'reference', 'candidate-seeker')]
SAMPLING = 'Exact seeded stochastic policy; identical independent role uniform tapes per map across conditions.'
SCOPE = ('Explicit fixed-opponent development evaluation. No training-return selection; maps reused for ranking are not '
         'held-out evidence. The stateless conditions zero recurrent memory every step and measure policy breakage, not '
         'memory content; the blackout condition hides the opponent block for a bounded window after first sight.')


def condition_table(learned, frozen, heldout=None):
    """Named actor pairs and physical/observation modes for every evaluated condition."""
    table = {
        'reference': (frozen, 'learned'),
        'candidate-hider': ([learned[0], frozen[1]], 'learned'),
        'candidate-seeker': ([frozen[0], learned[1]], 'learned'),
        'candidate-pair': (learned, 'learned'),
        'no-hider-tools': (learned, 'no-hider-tools'),
        'no-seeker-tools': (learned, 'no-seeker-tools'),
        'no-hider-memory': (learned, 'no-hider-memory'),
        'no-seeker-memory': (learned, 'no-seeker-memory'),
        'seeker-blackout': (learned, 'seeker-blackout')}
    if heldout is not None:
        table.update({
            'heldout-pair': (heldout, 'learned'),
            'candidate-hider-heldout': ([learned[0], heldout[1]], 'learned'),
            'candidate-seeker-heldout': ([heldout[0], learned[1]], 'learned')})
    return table


def contrast_definitions(heldout=False):
    rows = [
        ('Hider change', 'candidate-hider', 'reference'),
        ('Seeker change', 'reference', 'candidate-seeker'),
        ('Hider grab/lock benefit', 'candidate-pair', 'no-hider-tools'),
        ('Seeker grab/lock benefit', 'no-seeker-tools', 'candidate-pair'),
        ('Hider stateless cost', 'candidate-pair', 'no-hider-memory'),
        ('Seeker stateless cost', 'no-seeker-memory', 'candidate-pair'),
        ('Seeker blackout cost', 'seeker-blackout', 'candidate-pair')]
    if heldout:
        rows += [('Hider vs held-out opponent', 'candidate-hider-heldout', 'heldout-pair'),
                 ('Seeker vs held-out opponent', 'heldout-pair', 'candidate-seeker-heldout')]
    return rows


def initialize(candidate, reference, heldout):
    global ACTORS
    learned = load_pair(candidate)[0]
    frozen = load_pair(reference)[0]
    ACTORS = condition_table(learned, frozen, load_pair(heldout)[0] if heldout else None)


def evaluate_map(configuration):
    rows = []
    for name, (actors, mode) in ACTORS.items():
        row = episode(actors, configuration['seed'], configuration['scenario'], mode, arena_config=configuration['arenaConfig'])
        row['mode'] = name
        rows.append(row)
    return rows


def validate_maps(maps):
    if not maps or len({(row['seed'], row['scenario']) for row in maps}) != len(maps):
        raise ValueError('Use distinct declared development map identities')
    for row in maps:
        config = row['arenaConfig']
        if (not SEED_RANGE[0] <= row['seed'] < SEED_RANGE[1]
                or row['scenario'] not in SCENARIOS
                or not SIZE_RANGE[0] <= config['size'] <= SIZE_RANGE[1]
                or not 0 <= config['n_boxes'] <= BOX_LIMIT or not 0 <= config['n_ramps'] <= RAMP_LIMIT
                or not 1 <= config.get('play', 144) <= PLAY_LIMIT):
            raise ValueError('Use the declared physical range and reserve final-test seeds')


def freeze_inputs(args, output):
    """Copy every input checkpoint under ``output/inputs`` by hash and point ``args`` at the copies."""
    inputs = [args.checkpoint, args.reference] + ([args.heldout] if args.heldout else [])
    frozen_paths = []
    for path in inputs:
        digest = file_hash(path)
        frozen = output / 'inputs' / (digest + '.pt')
        if copy_immutable(path, frozen) != digest:
            raise ValueError('Use an immutable checkpoint; the supplied latest file changed')
        saved = load_native(frozen)
        stage_dependencies(path, saved, output)
        frozen_paths.append(str(frozen.resolve()))
    args.checkpoint, args.reference = frozen_paths[:2]
    args.heldout = frozen_paths[2] if args.heldout else None
    return frozen_paths


def resume_rows(partial, conditions):
    rows = json.loads(partial.read_text()) if partial.exists() else []
    completed = {(row['seed'], row['scenario']) for row in rows}
    if len(rows) != conditions * len(completed):
        raise ValueError('Resume only complete evaluation maps')
    return rows, completed


def breakdowns(rows, conditions):
    """Summaries and contrasts per scenario family and per play length."""
    by_environment = {}
    for scenario in SCENARIOS:
        subset = [row for row in rows if row['scenario'] == scenario]
        if not subset:
            by_environment[scenario] = dict(episodes=0, summary={}, contrasts={})
            continue
        local_summary, local_contrasts = summarize(subset, TOOL_CONTRASTS)
        by_environment[scenario] = dict(summary=local_summary, contrasts=local_contrasts)
    by_play = {}
    for play in sorted({row['info']['play_steps'] for row in rows}):
        subset = [row for row in rows if row['info']['play_steps'] == play]
        local_summary, local_contrasts = summarize(subset, CHANGE_CONTRASTS + TOOL_CONTRASTS)
        by_play[str(play)] = dict(maps=len(subset) // conditions, summary=local_summary, contrasts=local_contrasts)
    return by_environment, by_play


def main(args):
    cohort = json.loads(Path(args.cohort).read_text())
    maps = cohort['maps']
    validate_maps(maps)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'evaluation.json'
    if report_path.exists():
        raise ValueError('Keep completed evaluations immutable')
    frozen_paths = freeze_inputs(args, output)
    records = [load_pair(path)[1] for path in frozen_paths]
    candidate, reference = records[:2]
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in SOURCE_NAMES}
    if any(saved['provenance']['physicsSHA256'] != sources['physics.py'] for saved in records):
        raise ValueError('Checkpoint and evaluation physics must agree')
    identity = dict(checkpointSHA256=file_hash(args.checkpoint), referenceSHA256=file_hash(args.reference),
                    heldoutSHA256=file_hash(args.heldout) if args.heldout else None, maps=maps, sources=sources,
                    blackoutSteps=BLACKOUT_STEPS)
    identity_path = output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Partial evaluation identity changed')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    partial = output / 'episodes.partial.json'
    conditions = len(condition_table([None, None], [None, None], [None, None] if args.heldout else None))
    rows, completed = resume_rows(partial, conditions)
    pending = [row for row in maps if (row['seed'], row['scenario']) not in completed]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(args.checkpoint, args.reference, args.heldout)) as executor:
        for result in executor.map(evaluate_map, pending):
            rows.extend(result)
            temporary = partial.with_suffix('.tmp')
            temporary.write_text(json.dumps(rows, default=plain) + '\n')
            temporary.replace(partial)
            print(json.dumps(dict(completeMaps=len(rows) // conditions)), flush=True)
    summary, contrasts = summarize(rows, contrast_definitions(bool(args.heldout)))
    by_environment, by_play = breakdowns(rows, conditions)
    for mode, value in summary.items():
        value['completeSearchMisses'] = sum(row['info']['hidden'] == row['info']['play_steps'] for row in rows if row['mode'] == mode)
    report = dict(format=REPORT_FORMAT, **identity, summary=summary, contrasts=contrasts, byEnvironment=by_environment,
                  byPlayLength=by_play, episodes=rows, evaluationActorDecisions=sum(2 * (row['info']['t']) for row in rows),
                  sampling=SAMPLING, scope=SCOPE,
                  provenance=dict(candidate=candidate['provenance'], reference=reference['provenance'],
                                  heldout=records[2]['provenance'] if args.heldout else None))
    report_path.write_text(json.dumps(report, indent=2, default=plain, allow_nan=False) + '\n')
    if args.registry:
        print(json.dumps(register(args.checkpoint, args.reference, report_path, args.registry), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'reference', 'cohort', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--heldout')
    parser.add_argument('--registry')
    parser.add_argument('--workers', type=int, default=2)
    main(parser.parse_args())
