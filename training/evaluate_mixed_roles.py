"""Fresh varied-layout assessment of unchanged37M hider plus leagueB8M seeker."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil

import numpy as np
import torch
from persistent_evaluate import episode, load, plain, summarize
from persistent_train import file_hash


ACTORS = None


def initialize(paths):
    global ACTORS
    parent, league, initial, warm, middle = [load(path)[0] for path in paths]
    mixed = [parent[0], league[1]]
    ACTORS = dict(
        parent=(parent, 'learned'), mixed=(mixed, 'learned'), initial=(initial, 'learned'),
        parent_hider_initial_seeker=([parent[0], initial[1]], 'learned'),
        initial_hider_parent_seeker=([initial[0], parent[1]], 'learned'),
        initial_hider_mixed_seeker=([initial[0], league[1]], 'learned'),
        warm_hider_parent_seeker=([warm[0], parent[1]], 'learned'),
        warm_hider_mixed_seeker=([warm[0], league[1]], 'learned'),
        middle_hider_parent_seeker=([middle[0], parent[1]], 'learned'),
        middle_hider_mixed_seeker=([middle[0], league[1]], 'learned'),
        mixed_no_hider_tools=(mixed, 'no-hider-tools'),
        mixed_no_seeker_tools=(mixed, 'no-seeker-tools'),
        mixed_push_only=(mixed, 'no-tools'), league_pair=(league, 'learned'))


def definitions():
    return [
        ('Seeker change vs same37M hider', 'parent', 'mixed'),
        ('Seeker change vs same initial hider', 'initial_hider_parent_seeker', 'initial_hider_mixed_seeker'),
        ('Seeker change vs same warm-start hider', 'warm_hider_parent_seeker', 'warm_hider_mixed_seeker'),
        ('Seeker change vs same intermediate hider', 'middle_hider_parent_seeker', 'middle_hider_mixed_seeker'),
        ('Retained37M hider vs league hider, same league seeker', 'mixed', 'league_pair'),
        ('Mixed hider grab/lock benefit', 'mixed', 'mixed_no_hider_tools'),
        ('Mixed seeker grab/lock benefit', 'mixed_no_seeker_tools', 'mixed'),
        ('Mixed hider total learning vs initial', 'parent_hider_initial_seeker', 'initial'),
        ('Mixed seeker total learning vs initial', 'initial', 'initial_hider_mixed_seeker'),
    ]


def evaluate_map(configuration):
    records = []
    for name, (actors, mode) in ACTORS.items():
        result = episode(actors, configuration['seed'], configuration['scenario'], mode,
                         arena_config=configuration['arenaConfig'])
        result['mode'] = name
        records.append(result)
    return records


def main(args):
    if not 1500000000 <= args.seed < 1899000000:
        raise ValueError('Development validation only; final1.9B remains untouched')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation.json').exists():
        raise ValueError('Preserve previous assessed cohorts')
    paths = [args.parent, args.seeker, args.initial, args.warm, args.middle]
    records = [load(path)[1] for path in paths]
    parent, league, initial, _, _ = records
    physics_hash = file_hash(Path(__file__).with_name('physics.py'))
    if initial['decisions'] != 0:
        raise ValueError('Initial comparison requires actual zero experience')
    if any(record['provenance']['physicsSHA256'] != physics_hash for record in records):
        raise ValueError('Every fixed opponent must use the same physical game')
    if league['arguments']['arm'] != 'B' or league['totalPolicyInteractions'] != 7995392:
        raise ValueError('This protocol assesses the exact predeclared leagueB8M source')
    lineage = []
    for role, path, source in [('hider', args.parent, parent), ('seeker', args.seeker, league)]:
        lineage.append(dict(role=role, sourceCheckpointSHA256=file_hash(path),
            sourceFile=Path(path).name, sourceRunCounters={key: source.get(key) for key in
                ['decisions', 'parentDecisions', 'pilotDecisions', 'criticWarmupDecisions',
                 'totalPolicyInteractions', 'currentPolicyDecisions', 'historicalPolicyDecisions', 'activePolicySamples']}))
    mixed = dict(format=parent['format'], observationSize=140, physicsObservationSize=138,
        models=[copy.deepcopy(parent['models'][0]), copy.deepcopy(league['models'][1])],
        optimizers=[copy.deepcopy(parent['optimizers'][0]), copy.deepcopy(league['optimizers'][1])],
        decisions=league['decisions'],
        decisionsDefinition='Union source-run budget including shared ancestry, critic-only warmup and historical opponents. Per-role sources are authoritative; these roles did not train equally.',
        newActorUpdates=0, provenance=dict(physicsSHA256=physics_hash, roleSources=lineage,
            operation='Exact role selection only; no averaging, optimization, or actor modification'),
        status='Mixed-role development candidate pending this separate evaluation; not a browser promotion')
    mixed_path = output / 'mixed-role-candidate.pt'
    torch.save(mixed, mixed_path)
    for source, selected in [(parent['models'][0], mixed['models'][0]), (league['models'][1], mixed['models'][1])]:
        for name in source:
            torch.testing.assert_close(source[name], selected[name], atol=0, rtol=0)
    generator = np.random.default_rng(913779)
    maps = [dict(seed=args.seed + scenario_index * 1000 + index, scenario=scenario,
                arenaConfig=dict(size=[6, 7, 8, 9, 10, 12][index % 6],
                                 n_boxes=int(generator.integers(0, 9)), n_ramps=int(generator.integers(0, 3))))
            for scenario_index, scenario in enumerate(['shelter', 'rooms', 'open'])
            for index in range(32)]
    (output / 'predeclared-maps.json').write_text(json.dumps(maps, indent=2) + '\n')
    names = ['evaluate_mixed_roles.py', 'persistent_evaluate.py', 'persistent_actor.py',
             'persistent_train.py', 'actor.py', 'physics.py']
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in names}
    (output / 'source').mkdir(exist_ok=True)
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), output / 'source' / name)
    games = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(paths,)) as executor:
        for index, rows in enumerate(executor.map(evaluate_map, maps)):
            games.extend(rows)
            (output / 'episodes.partial.json').write_text(json.dumps(games, default=plain) + '\n')
            print(json.dumps(dict(maps=index + 1, games=len(games))), flush=True)
    summary, contrasts = summarize(games, definitions())
    breakdowns = {}
    for label, selector in [('scenario', lambda row: row['scenario']),
                            ('size', lambda row: str(row['arenaConfig']['size'])),
                            ('objects', lambda row: '0' if row['actualObjectCount'] == 0 else '1–3' if row['actualObjectCount'] <= 3 else '4–6' if row['actualObjectCount'] <= 6 else '7–10')]:
        breakdowns[label] = {}
        for value in dict.fromkeys(selector(row) for row in games):
            selected = [row for row in games if selector(row) == value]
            _, effects = summarize(selected, definitions())
            breakdowns[label][value] = dict(maps=len(selected) // 14, contrasts=effects)
    for mode, values in summary.items():
        selected = [row for row in games if row['mode'] == mode]
        values['completeSearchMisses'] = sum(row['info']['hidden'] == row['info']['play_steps'] for row in selected)
    report = dict(format='mixed-role-varied-layout-validation-v1',
        mixedCheckpointSHA256=file_hash(mixed_path), roleSources=lineage,
        fixedOpponentSHA256={Path(path).name: file_hash(path) for path in paths},
        maps=maps, games=len(games), evaluationActorDecisions=len(games) * 480,
        scope='96 fresh development maps across supported size/count range; fixed known opponent policies. Modest subgroup samples; final1.9B untouched.',
        sampling='Common role-specific uniform action tapes per map across14 conditions; no critic at inference.',
        toolAblation='Disable named role grab/lock; requested controller remains stateful and physical pushing remains active.',
        summary=summary, contrasts=contrasts, breakdowns=breakdowns, episodes=games, sources=sources)
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, default=plain) + '\n')
    print(json.dumps(contrasts, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ['parent', 'seeker', 'initial', 'warm', 'middle', 'output']:
        parser.add_argument('--' + argument, required=True)
    parser.add_argument('--seed', type=int, default=1500140000)
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args())
