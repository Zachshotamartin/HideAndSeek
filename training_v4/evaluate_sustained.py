"""Assess sustained checkpoints on the predeclared broad development cohort.

Unchanged reference games are reused verbatim from the frozen mixed assessment.
Only new actor conditions collect fresh evaluation rollouts; no optimizer runs.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil

from persistent_evaluate import episode, load, plain, summarize
from persistent_train import file_hash

ACTORS = None


def initialize(paths):
    current, parent, league, initial, warm, middle = [load(path)[0] for path in paths]
    ACTORS_NAMES = [
        ('new_pair', current, 'learned'),
        ('new_hider_fixed_mixed_seeker', [current[0], league[1]], 'learned'),
        ('fixed_parent_hider_new_seeker', [parent[0], current[1]], 'learned'),
        ('new_hider_fixed_parent_seeker', [current[0], parent[1]], 'learned'),
        ('new_hider_initial_seeker', [current[0], initial[1]], 'learned'),
        ('initial_hider_new_seeker', [initial[0], current[1]], 'learned'),
        ('warm_hider_new_seeker', [warm[0], current[1]], 'learned'),
        ('middle_hider_new_seeker', [middle[0], current[1]], 'learned'),
        ('new_no_hider_tools', current, 'no-hider-tools'),
        ('new_no_seeker_tools', current, 'no-seeker-tools'),
        ('new_push_only', current, 'no-tools'),
    ]
    global ACTORS
    ACTORS = {name: (actors, mode) for name, actors, mode in ACTORS_NAMES}


def evaluate_map(configuration):
    results = []
    for name, (actors, mode) in ACTORS.items():
        result = episode(actors, configuration['seed'], configuration['scenario'], mode,
                         arena_config=configuration['arenaConfig'])
        result['mode'] = name
        results.append(result)
    return results


def definitions():
    return [
        ('Hider change vs fixed mixed seeker', 'new_hider_fixed_mixed_seeker', 'mixed'),
        ('Hider change vs fixed original seeker', 'new_hider_fixed_parent_seeker', 'parent'),
        ('Hider change vs fixed initial seeker', 'new_hider_initial_seeker', 'parent_hider_initial_seeker'),
        ('Seeker change vs fixed original hider', 'mixed', 'fixed_parent_hider_new_seeker'),
        ('Seeker change vs fixed initial hider', 'initial_hider_mixed_seeker', 'initial_hider_new_seeker'),
        ('Seeker change vs fixed warm-start hider', 'warm_hider_mixed_seeker', 'warm_hider_new_seeker'),
        ('Seeker change vs fixed intermediate hider', 'middle_hider_mixed_seeker', 'middle_hider_new_seeker'),
        ('New hider grab/lock benefit', 'new_pair', 'new_no_hider_tools'),
        ('New seeker grab/lock benefit', 'new_no_seeker_tools', 'new_pair'),
        ('New hider total learning vs initial', 'new_hider_initial_seeker', 'initial'),
        ('New seeker total learning vs initial', 'initial', 'initial_hider_new_seeker'),
    ]


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation.json').exists():
        raise ValueError('Do not rerun or overwrite an assessed checkpoint')
    paths = [args.checkpoint, args.parent, args.seeker, args.initial, args.warm, args.middle]
    records = [load(path)[1] for path in paths]
    saved = records[0]
    physics_hash = file_hash(Path(__file__).with_name('physics.py'))
    if any(record['provenance']['physicsSHA256'] != physics_hash for record in records):
        raise ValueError('All models must use the same frozen physical game')
    baseline = json.loads(Path(args.baseline).read_text())
    if baseline['format'] != 'mixed-role-varied-layout-validation-v1':
        raise ValueError('Use the predeclared complete mixed-role baseline report')
    if baseline['sources']['physics.py'] != physics_hash:
        raise ValueError('Baseline physics must match exactly')
    if saved['provenance']['assessedMixedCheckpointSHA256'] != baseline['mixedCheckpointSHA256']:
        raise ValueError('The continuation must begin from the exact assessed mixed pair')
    for path in paths[1:]:
        if file_hash(path) not in baseline['fixedOpponentSHA256'].values():
            raise ValueError('Cached reference games must belong to the exact frozen opponents')
    if records[3]['decisions'] != 0:
        raise ValueError('Initial comparison must have zero training experience')
    maps = baseline['maps']
    if len(maps) != 96 or any(not 1500000000 <= row['seed'] < 1900000000 for row in maps):
        raise ValueError('This protocol uses the fixed 96-map development cohort only')
    identity = dict(checkpointSHA256=file_hash(args.checkpoint), baselineReportSHA256=file_hash(args.baseline),
                    physicsSHA256=physics_hash, maps=maps)
    identity_path = output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Partial results belong to another model or baseline')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    partial = output / 'new-episodes.partial.json'
    games = json.loads(partial.read_text()) if partial.exists() else []
    completed = {(row['seed'], row['scenario']) for row in games}
    if len(games) != len(completed) * 11:
        raise ValueError('Resume only complete maps with all 11 new conditions')
    pending = [row for row in maps if (row['seed'], row['scenario']) not in completed]
    sources = ['evaluate_sustained.py', 'persistent_evaluate.py', 'persistent_actor.py',
               'persistent_train.py', 'actor.py', 'physics.py']
    source_dir = output / 'source'
    source_dir.mkdir(exist_ok=True)
    source_hashes = {name: file_hash(Path(__file__).with_name(name)) for name in sources}
    for name in sources:
        source = Path(__file__).with_name(name)
        if (source_dir / name).exists() and file_hash(source_dir / name) != source_hashes[name]:
            raise ValueError('Evaluation code changed during a partial run')
        shutil.copyfile(source, source_dir / name)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(paths,)) as executor:
        for rows in executor.map(evaluate_map, pending):
            games.extend(rows)
            temporary = partial.with_suffix('.tmp')
            temporary.write_text(json.dumps(games, default=plain) + '\n')
            temporary.replace(partial)
            print(json.dumps(dict(maps=len(games) // 11, newGames=len(games))), flush=True)
    cached = copy.deepcopy(baseline['episodes'])
    combined = cached + games
    summary, contrasts = summarize(combined, definitions())
    for mode, result in summary.items():
        selected = [row for row in combined if row['mode'] == mode]
        result['completeSearchMisses'] = sum(row['info']['hidden'] == row['info']['play_steps'] for row in selected)
    breakdowns = {}
    for name, selector in [
        ('scenario', lambda row: row['scenario']),
        ('size', lambda row: str(row['arenaConfig']['size'])),
        ('objects', lambda row: '0' if row['actualObjectCount'] == 0 else '1–3' if row['actualObjectCount'] <= 3 else '4–6' if row['actualObjectCount'] <= 6 else '7–10'),
    ]:
        breakdowns[name] = {}
        for value in dict.fromkeys(selector(row) for row in combined):
            selected = [row for row in combined if selector(row) == value]
            _, effects = summarize(selected, definitions())
            breakdowns[name][value] = dict(maps=len(selected) // 25, contrasts=effects)
    fixed_role_changes = {key: value for key, value in contrasts.items()
                          if key.startswith('Hider change') or key.startswith('Seeker change')}
    regressions = [key for key, value in fixed_role_changes.items() if value['bootstrap95Percent'][1] < 0]
    improvements = [key for key, value in fixed_role_changes.items() if value['bootstrap95Percent'][0] > 0]
    report = dict(format='sustained-mixed-role-validation-v1', **identity,
        trainingCounters={key: saved.get(key) for key in ['parentDecisions', 'decisions', 'totalPolicyInteractions',
            'currentPolicyDecisions', 'historicalPolicyDecisions', 'activePolicySamples']},
        inheritedRoleSources=saved['provenance']['inheritedRoleSources'],
        newEvaluationGames=len(games), newEvaluationActorDecisions=len(games) * 480,
        cachedGamesReused=len(cached), cachedActorDecisionsNotRepeated=len(cached) * 480,
        scope='Repeated broad development cohort and known frozen opponents; final test seeds remain untouched.',
        sampling='Identical role-specific action tapes, consistent with the cached mixed assessment.',
        toolAblation='Grab/lock disabled for the named role; physical pushing and requested controller state remain enabled.',
        selection=dict(credibleRoleRegressions=regressions, credibleRoleImprovements=improvements,
                       automaticPromotion=False, unchangedBrowserReference=True),
        summary=summary, contrasts=contrasts, breakdowns=breakdowns, newEpisodes=games, sources=source_hashes)
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, default=plain) + '\n')
    print(json.dumps(dict(contrasts=contrasts, selection=report['selection']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'baseline', 'parent', 'seeker', 'initial', 'warm', 'middle', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args())
