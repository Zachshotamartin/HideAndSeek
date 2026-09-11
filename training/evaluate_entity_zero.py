"""Read-only projected encoder versus exact legacy selected roles, before training."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil

from entity_actor import load_pair
from persistent_evaluate import episode, summarize, plain
from persistent_train import file_hash

ACTORS = None


def initialize(paths):
    entity, legacy, initial = [load_pair(path)[0] for path in paths]
    global ACTORS
    ACTORS = dict(entity_pair=entity,
        entity_hider_fixed_legacy_seeker=[entity[0], legacy[1]],
        legacy_hider_entity_seeker=[legacy[0], entity[1]],
        entity_hider_fixed_initial_seeker=[entity[0], initial[1]],
        initial_hider_entity_seeker=[initial[0], entity[1]])


def evaluate_map(configuration):
    rows = []
    for name, actors in ACTORS.items():
        row = episode(actors, configuration['seed'], configuration['scenario'], 'learned',
                      arena_config=configuration['arenaConfig'])
        row['mode'] = name
        rows.append(row)
    return rows


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation.json').exists():
        raise ValueError('Keep zero-update assessments immutable')
    entity, legacy, initial = [load_pair(path)[1] for path in [args.entity, args.legacy, args.initial]]
    cohort = json.loads(Path(args.cohort).read_text())
    selected = json.loads(Path(args.selected_evaluation).read_text())
    hashes = [row['sourceCheckpointSHA256'] for row in legacy['provenance']['roleSources']]
    if hashes[0] != cohort['fixedOpponentSHA256']['checkpoint-14098432.pt'] or hashes[1] != selected['checkpointSHA256']:
        raise ValueError('Cached outcomes must belong to these exact selected roles')
    if entity['provenance']['sourceSelectedPairSHA256'] != legacy['provenance']['sourceSelectedPairSHA256']:
        raise ValueError('Both arms must start from the same selected pair')
    if any(record['newActorUpdates'] != 0 or any(opt['state'] for opt in record['optimizers'])
           for record in [entity, legacy]):
        raise ValueError('Both arms must have zero new actor updates and equally reset optimizers')
    if initial['decisions'] != 0:
        raise ValueError('The fixed initial opponent must have zero experience')
    physics = file_hash(Path(__file__).with_name('physics.py'))
    if any(record['provenance']['physicsSHA256'] != physics for record in [entity, legacy, initial]):
        raise ValueError('Physics is frozen')
    cached = []
    for records, original, name in [
        (selected['newEpisodes'], 'fixed_parent_hider_new_seeker', 'legacy_pair'),
        (selected['newEpisodes'], 'initial_hider_new_seeker', 'initial_hider_legacy_seeker'),
        (cohort['episodes'], 'parent_hider_initial_seeker', 'legacy_hider_fixed_initial_seeker')]:
        cached += [dict(copy.deepcopy(row), mode=name) for row in records if row['mode'] == original]
    if len(cached) != 288 or selected['baselineReportSHA256'] != file_hash(args.cohort):
        raise ValueError('Reuse the same complete 96-map cohort')
    names = ['evaluate_entity_zero.py', 'entity_actor.py', 'persistent_evaluate.py', 'persistent_actor.py', 'physics.py']
    identity = dict(entitySHA256=file_hash(args.entity), legacySHA256=file_hash(args.legacy),
        initialSHA256=file_hash(args.initial), cohortSHA256=file_hash(args.cohort),
        selectedEvaluationSHA256=file_hash(args.selected_evaluation),
        sources={name: file_hash(Path(__file__).with_name(name)) for name in names})
    identity_path = output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Partial evaluation identity changed')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    (output / 'source').mkdir(exist_ok=True)
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), output / 'source' / name)
    partial = output / 'new-episodes.partial.json'
    rows = json.loads(partial.read_text()) if partial.exists() else []
    complete = {(row['seed'], row['scenario']) for row in rows}
    if len(rows) != len(complete) * 5:
        raise ValueError('Resume only complete maps')
    pending = [row for row in cohort['maps'] if (row['seed'], row['scenario']) not in complete]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=([args.entity, args.legacy, args.initial],)) as executor:
        for result in executor.map(evaluate_map, pending):
            rows.extend(result)
            temporary = partial.with_suffix('.tmp')
            temporary.write_text(json.dumps(rows, default=plain) + '\n'); temporary.replace(partial)
            print(json.dumps(dict(maps=len(rows) // 5)), flush=True)
    summary, contrasts = summarize(cached + rows, [
        ('Projected hider change, same selected seeker', 'entity_hider_fixed_legacy_seeker', 'legacy_pair'),
        ('Projected seeker change, same selected hider', 'legacy_pair', 'legacy_hider_entity_seeker'),
        ('Projected hider change, same initial seeker', 'entity_hider_fixed_initial_seeker', 'legacy_hider_fixed_initial_seeker'),
        ('Projected seeker change, same initial hider', 'initial_hider_legacy_seeker', 'initial_hider_entity_seeker')])
    for mode, value in summary.items():
        value['completeSearchMisses'] = sum(row['info']['hidden'] == 160 for row in cached + rows if row['mode'] == mode)
    report = dict(format='entity-encoder-zero-update-evaluation-v1', **identity,
        scope='96 previously reused development maps; a zero-update projection test, not architecture learning evidence.',
        optimizerUpdates=0, newEvaluationActorDecisions=len(rows) * 480, cachedGamesReused=len(cached),
        optimizerResetBothArms=True, summary=summary, contrasts=contrasts, newEpisodes=rows,
        publicAssetsChanged=False)
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, default=plain) + '\n')
    print(json.dumps(contrasts, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['entity', 'legacy', 'initial', 'cohort', 'selected-evaluation', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=2)
    main(parser.parse_args())
