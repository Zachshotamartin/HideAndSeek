"""Measure grab/lock interventions on the exact selected pair in both UI modes.

The unchanged, already recorded selected-pair games are reused. Only physical
grab/lock actions are disabled; requested controller state and pushing remain.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil

import numpy as np
import torch
from evaluate_policy_modes import deterministic, contrast
from persistent_evaluate import load, sample, runs, plain
from persistent_train import file_hash
from physics import PhysicsEnv, DT

MODELS = None


def initialize(path):
    global MODELS
    MODELS = load(path)[0]


@torch.no_grad()
def episode(configuration, action_mode, intervention):
    env = PhysicsEnv(seed=configuration['seed'], scenario=configuration['scenario'],
                     **configuration['arenaConfig'], disable_tools=intervention == 'push_only')
    physical = env.observe()
    memories = [torch.zeros(1, 64), torch.zeros(1, 64)]
    buttons = np.zeros((2, 2), np.float32)
    generators = [np.random.default_rng(configuration['seed'] + offset)
                  for offset in [912341, 2912341]]
    grips = [[], []]
    for _ in range(env.prep + env.play):
        actions = np.zeros((2, 5), np.float32)
        for role in range(2):
            if action_mode == 'mean':
                actions[role], memories[role], buttons[role], _ = deterministic(
                    MODELS[role], physical[role], memories[role], buttons[role])
            else:
                actions[role], memories[role], buttons[role], _ = sample(
                    MODELS[role], physical[role], memories[role], buttons[role], generators[role])
            if (intervention == 'no_hider_tools' and role == 0
                    or intervention == 'no_seeker_tools' and role == 1):
                actions[role, 3:] = 0
        physical, _, _, info = env.step(actions)
        for role in range(2):
            grips[role].append(int(env.grips[role]))
    result = dict(**configuration, mode=f'{action_mode}_{intervention}',
                  hiddenFraction=info['hidden'] / env.play, info=info,
                  roles=[dict(gripDurationsTicks=runs(values)) for values in grips])
    env.close()
    return result


def evaluate_map(configuration):
    return [episode(configuration, mode, intervention)
            for mode in ['sample', 'mean']
            for intervention in ['no_hider_tools', 'no_seeker_tools', 'push_only']]


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation.json').exists():
        raise ValueError('Preserve completed selected-pair assessments')
    baseline = json.loads(Path(args.baseline).read_text())
    modes = json.loads(Path(args.modes).read_text())
    cache = json.loads(Path(args.cache).read_text())
    digest = file_hash(args.candidate)
    if any(record['candidateSHA256'] != digest for record in [modes, cache]):
        raise ValueError('Mode reports must contain the exact selected pair')
    if any(record['baselineSHA256'] != file_hash(args.baseline) for record in [modes, cache]):
        raise ValueError('Use the identical predeclared map cohort')
    if load(args.candidate)[1]['provenance']['physicsSHA256'] != file_hash(Path(__file__).with_name('physics.py')):
        raise ValueError('The selected policy must match the frozen physical game')
    names = ['evaluate_selected_tools.py', 'evaluate_policy_modes.py',
             'persistent_evaluate.py', 'persistent_actor.py', 'persistent_train.py', 'actor.py', 'physics.py']
    identity = dict(candidateSHA256=digest, modeReportSHA256=file_hash(args.modes),
                    cacheSHA256=file_hash(args.cache), baselineSHA256=file_hash(args.baseline),
                    sources={name: file_hash(Path(__file__).with_name(name)) for name in names})
    identity_path = output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Partial results belong to another immutable selection or source')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    (output / 'source').mkdir(exist_ok=True)
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), output / 'source' / name)
    cached = [dict(row, mode='sample_learned') for row in cache['episodes']
              if row['mode'] == 'candidate_pair_sample']
    cached += [dict(row, mode='mean_learned') for row in modes['newEpisodes']
               if row['mode'] == 'candidate_pair_mean']
    if len(cached) != 192:
        raise ValueError('Reuse the complete 96-map selected pair in both modes')
    partial = output / 'new-episodes.partial.json'
    rows = json.loads(partial.read_text()) if partial.exists() else []
    complete = {(row['seed'], row['scenario']) for row in rows}
    if len(rows) != len(complete) * 6:
        raise ValueError('Resume only complete maps with all six interventions')
    pending = [row for row in baseline['maps'] if (row['seed'], row['scenario']) not in complete]
    with ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.candidate,)) as executor:
        for results in executor.map(evaluate_map, pending):
            rows.extend(results)
            temporary = partial.with_suffix('.tmp')
            temporary.write_text(json.dumps(rows, default=plain) + '\n')
            temporary.replace(partial)
            print(json.dumps(dict(maps=len(rows) // 6, newGames=len(rows))), flush=True)
    all_rows = cached + rows
    effects = {}
    for mode in ['sample', 'mean']:
        effects[f'{mode}: hider grab/lock benefit'] = contrast(
            all_rows, f'{mode}_learned', f'{mode}_no_hider_tools')
        effects[f'{mode}: seeker grab/lock benefit'] = contrast(
            all_rows, f'{mode}_no_seeker_tools', f'{mode}_learned')
        effects[f'{mode}: joint tools hidden-fraction change'] = contrast(
            all_rows, f'{mode}_learned', f'{mode}_push_only')
    summary = {}
    for mode in dict.fromkeys(row['mode'] for row in all_rows):
        selected = [row for row in all_rows if row['mode'] == mode]
        summary[mode] = dict(hiddenFraction=float(np.mean([row['hiddenFraction'] for row in selected])),
            completeSearchMisses=sum(row['info']['hidden'] == 160 for row in selected),
            propShieldedFraction=float(np.mean([row['info']['shielded'] / 160 for row in selected])),
            meanPropDisplacement=float(np.mean([sum(row['info']['object_displacement']) for row in selected])),
            roles=[])
        for role in range(2):
            lengths = [value * DT for row in selected for value in row['roles'][role]['gripDurationsTicks']]
            summary[mode]['roles'].append(dict(grips=len(lengths),
                medianGripSeconds=float(np.median(lengths)) if lengths else 0, maxGripSeconds=max(lengths, default=0)))
    report = dict(format='selected-pair-tools-and-action-modes-v1', **identity,
        scope='Exact selected pair on 96 repeated development maps; final test seeds remain untouched.',
        intervention='Only physical grab/lock controls disabled for the named roles. Requested buttons and ordinary pushing persist. Joint removal is not an isolated role-learning effect.',
        newEvaluationActorDecisions=len(rows) * 480, cachedGamesReused=len(cached),
        summary=summary, contrasts=effects, newEpisodes=rows, optimizerUpdates=0,
        automaticPromotion=False)
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, default=plain) + '\n')
    print(json.dumps(effects, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['candidate', 'baseline', 'modes', 'cache', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=2)
    main(parser.parse_args())
