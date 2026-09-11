"""Fixed-map read-only A/B validation; no centralized critic runs at inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil

from persistent_evaluate import episode, load, plain, summarize
from persistent_actor import FORMAT
from persistent_train import file_hash


ACTORS = None


def conditions(a, b, parent, initial):
    compared = {
        'parent-pair': (parent, 'learned'),
        'initial-pair': (initial, 'learned'),
        'parent-hider-v-initial': ([parent[0], initial[1]], 'learned'),
        'parent-seeker-v-initial': ([initial[0], parent[1]], 'learned'),
    }
    for arm, actors in [('A', a), ('B', b)]:
        for mode in ['learned', 'no-hider-tools', 'no-seeker-tools', 'no-tools']:
            compared[f'{arm}/{mode}'] = (actors, mode)
        compared[f'{arm}/hider-v-parent'] = ([actors[0], parent[1]], 'learned')
        compared[f'{arm}/seeker-v-parent'] = ([parent[0], actors[1]], 'learned')
        compared[f'{arm}/hider-v-initial'] = ([actors[0], initial[1]], 'learned')
        compared[f'{arm}/seeker-v-initial'] = ([initial[0], actors[1]], 'learned')
    compared['A-hider/B-seeker'] = ([a[0], b[1]], 'learned')
    compared['B-hider/A-seeker'] = ([b[0], a[1]], 'learned')
    return compared


def contrast_definitions():
    definitions = []
    for arm in ['A', 'B']:
        definitions.extend([
            (f'{arm}: Hider grab/lock benefit', f'{arm}/learned', f'{arm}/no-hider-tools'),
            (f'{arm}: Seeker grab/lock benefit', f'{arm}/no-seeker-tools', f'{arm}/learned'),
            (f'{arm}: Hider change vs fixed parent seeker', f'{arm}/hider-v-parent', 'parent-pair'),
            (f'{arm}: Seeker change vs fixed parent hider', 'parent-pair', f'{arm}/seeker-v-parent'),
            (f'{arm}: Hider change vs fixed initial seeker', f'{arm}/hider-v-initial', 'parent-hider-v-initial'),
            (f'{arm}: Seeker change vs fixed initial hider', 'parent-seeker-v-initial', f'{arm}/seeker-v-initial'),
            (f'{arm}: Total hider learning vs initial', f'{arm}/hider-v-initial', 'initial-pair'),
            (f'{arm}: Total seeker learning vs initial', 'initial-pair', f'{arm}/seeker-v-initial'),
        ])
    definitions.extend([
        ('B minus A: Hider vs same parent seeker', 'B/hider-v-parent', 'A/hider-v-parent'),
        ('B minus A: Seeker vs same parent hider', 'A/seeker-v-parent', 'B/seeker-v-parent'),
        ('B minus A: Hider vs same initial seeker', 'B/hider-v-initial', 'A/hider-v-initial'),
        ('B minus A: Seeker vs same initial hider', 'A/seeker-v-initial', 'B/seeker-v-initial'),
    ])
    return definitions


def initialize(paths):
    global ACTORS
    ACTORS = conditions(*(load(path)[0] for path in paths))


def evaluate_map(task):
    seed, scenario = task
    records = []
    for name, (actors, physical_mode) in ACTORS.items():
        result = episode(actors, seed, scenario, physical_mode)
        result['mode'] = name
        records.append(result)
    return records


def main(args):
    if not 1500000000 <= args.seed < 1899000000:
        raise ValueError('This experiment uses validation seeds; final 1.9B remains untouched')
    paths = [args.a, args.b, args.parent, args.initial]
    saved = [load(path)[1] for path in paths]
    a, b, parent, initial = saved
    parent_hash = file_hash(args.parent)
    physics_hash = file_hash(Path(__file__).with_name('physics.py'))
    if any(record['format'] != FORMAT for record in saved) or initial['decisions'] != 0:
        raise ValueError('Compatible actors and genuine zero-experience initial policies required')
    if a['totalPolicyInteractions'] != b['totalPolicyInteractions']:
        raise ValueError('Compare checkpoints at identical total interaction budgets')
    for arm, record in [('A', a), ('B', b)]:
        if record['arguments']['arm'] != arm:
            raise ValueError('Checkpoint assigned to wrong experimental arm')
        if record['provenance']['parentSHA256'] != parent_hash:
            raise ValueError('Use the exact frozen original parent as comparator')
        if record['provenance']['physicsSHA256'] != physics_hash:
            raise ValueError('Evaluation physics must match the frozen training environment')
        if sum(record['currentPolicyDecisions']) + sum(record['historicalPolicyDecisions']) != record['totalPolicyInteractions']:
            raise ValueError('Invalid interaction accounting')
    if a['worldRNG'] != b['worldRNG'] or a['opponentRNG'] != b['opponentRNG']:
        raise ValueError('The paired world/assignment schedules diverged')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation.json').exists():
        raise ValueError('Preserve previously assessed output')
    source_names = ['evaluate_league.py', 'persistent_evaluate.py', 'persistent_actor.py',
                    'persistent_train.py', 'actor.py', 'physics.py']
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in source_names}
    (output / 'source').mkdir(exist_ok=True)
    for name in source_names:
        shutil.copyfile(Path(__file__).with_name(name), output / 'source' / name)
    tasks = [(args.seed + scenario_index * 1000 + index, scenario)
             for scenario_index, scenario in enumerate(['shelter', 'rooms', 'open'])
             for index in range(args.episodes_per_scenario)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(paths,)) as executor:
        for index, rows in enumerate(executor.map(evaluate_map, tasks)):
            records.extend(rows)
            (output / 'episodes.partial.json').write_text(json.dumps(records, default=plain) + '\n')
            print(json.dumps(dict(maps=index + 1, games=len(records))), flush=True)
    summary, contrasts = summarize(records, contrast_definitions())
    rejections = {arm: [name for name, result in contrasts.items()
                       if name.startswith(f'{arm}:') and 'change vs fixed' in name
                       and result['bootstrap95Percent'][1] < 0] for arm in ['A', 'B']}
    report = dict(
        format='matched-larger-batch-league-validation-v1',
        checkpoints={name: dict(sha256=file_hash(path), training={
            key: record.get(key, 0) for key in ['parentDecisions', 'criticWarmupDecisions',
                'totalPolicyInteractions', 'currentPolicyDecisions', 'historicalPolicyDecisions',
                'activePolicySamples', 'actorUpdateDecisions', 'decisions']},
            provenance=record.get('provenance'))
            for name, path, record in zip(['A', 'B', 'parent', 'initial'], paths, saved)},
        seedStart=args.seed, episodesPerScenario=args.episodes_per_scenario,
        evaluationActorDecisions=len(records) * 240 * 2,
        scope='Fixed development cohort: three scenarios at8m, three boxes and one ramp. No final-test or full supported-range claim.',
        sampling='Identical independent role-specific uniform tapes per map across all22 conditions; Gaussian movement plus categorical commands.',
        ablation='Grab/lock disabled for named roles only; requested controller states unchanged, physical pushing remains enabled.',
        privacy='No centralized critic or full-scene state enters inference; actual restricted actor inputs and recurrent memory only.',
        rejectedForRoleRegression=rejections,
        decision='No automatic promotion. Negative role intervals reject candidates; inconclusive intervals cannot establish useful tools.',
        summary=summary, contrasts=contrasts, episodes=records,
        sources=sources)
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, default=plain) + '\n')
    print(json.dumps(contrasts, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ['a', 'b', 'parent', 'initial', 'output']:
        parser.add_argument('--' + argument, required=True)
    parser.add_argument('--episodes-per-scenario', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1500110000)
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args())
