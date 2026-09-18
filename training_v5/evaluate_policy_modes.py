"""Read-only sample/mean comparison with fixed learned and initial references.

Role comparisons hold the browser opponent in its existing sampled mode. The
three complete-pair mean conditions additionally match the global UI setting.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import shutil

import numpy as np
import torch
from persistent_actor import augment, advance_buttons
from persistent_evaluate import load, sample, runs, plain
from persistent_train import file_hash
from physics import PhysicsEnv, DT

POLICIES = None


@torch.no_grad()
def deterministic(model, physical, memory, buttons):
    blind = physical[7] < .5 and physical[5] < 1
    if blind:
        buttons = np.zeros(2, np.float32)
    observation = augment(physical, buttons)
    normal, tools, _, memory = model(torch.from_numpy(observation[None]), memory)
    commands = tools.logits[0].argmax(-1).numpy()
    buttons = advance_buttons(buttons, commands, blind)
    action = np.zeros(5, np.float32)
    action[:3] = [math.tanh(float(value)) for value in normal.mean[0]]
    action[3:] = buttons
    if blind:
        action.fill(0)
    return action, memory, buttons, commands


def initialize(candidate, browser, initial):
    candidate, browser, initial = [load(path)[0] for path in [candidate, browser, initial]]
    global POLICIES
    POLICIES = dict(
        candidate_hider_mean_fixed_browser_seeker=([candidate[0], browser[1]], ['mean', 'sample']),
        initial_hider_mean_fixed_browser_seeker=([initial[0], browser[1]], ['mean', 'sample']),
        fixed_browser_hider_candidate_seeker_mean=([browser[0], candidate[1]], ['sample', 'mean']),
        fixed_browser_hider_initial_seeker_mean=([browser[0], initial[1]], ['sample', 'mean']),
        candidate_pair_mean=(candidate, ['mean', 'mean']),
        browser_pair_mean=(browser, ['mean', 'mean']),
        initial_pair_mean=(initial, ['mean', 'mean']))


@torch.no_grad()
def episode(models, modes, configuration):
    env = PhysicsEnv(seed=configuration['seed'], scenario=configuration['scenario'], **configuration['arenaConfig'])
    physical = env.observe()
    memory = [torch.zeros(1, 64), torch.zeros(1, 64)]
    buttons = np.zeros((2, 2), np.float32)
    generators = [np.random.default_rng(configuration['seed'] + offset) for offset in [912341, 2912341]]
    grips, holding, counts = [[], []], [[], []], np.zeros((2, 2, 3), np.int64)
    for tick in range(240):
        actions = np.zeros((2, 5), np.float32)
        for role in range(2):
            if modes[role] == 'mean':
                actions[role], memory[role], buttons[role], commands = deterministic(
                    models[role], physical[role], memory[role], buttons[role])
            else:
                actions[role], memory[role], buttons[role], commands = sample(
                    models[role], physical[role], memory[role], buttons[role], generators[role])
            holding[role].append(buttons[role].copy())
            if role == 0 or tick >= 80:
                for tool in range(2):
                    counts[role, tool, commands[tool]] += 1
        physical, _, _, info = env.step(actions)
        for role in range(2):
            grips[role].append(int(env.grips[role]))
    result = dict(**configuration, roleModes=modes, hiddenFraction=info['hidden'] / 160,
        completeSearchMiss=info['hidden'] == 160, info=info,
        roles=[dict(gripDurationsTicks=runs(grips[role]),
                    requestedHoldDurationsTicks=[runs([row[tool] for row in holding[role]], lambda value: value == 1) for tool in range(2)],
                    commandCounts=counts[role].tolist()) for role in range(2)])
    env.close()
    return result


def evaluate_map(configuration):
    rows = []
    for name, (models, modes) in POLICIES.items():
        result = episode(models, modes, configuration)
        result['mode'] = name
        rows.append(result)
    return rows


def contrast(rows, left, right):
    by_map = {}
    for row in rows:
        by_map.setdefault((row['seed'], row['scenario']), {})[row['mode']] = row
    values = np.array([record[left]['hiddenFraction'] - record[right]['hiddenFraction'] for record in by_map.values()])
    boot = np.random.default_rng(319881).choice(values, size=(10000, len(values)), replace=True).mean(1)
    return dict(mean=float(values.mean()), bootstrap95Percent=np.quantile(boot, [.025, .975]).tolist())


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation.json').exists():
        raise ValueError('Preserve completed mode assessments')
    candidate, browser, initial = [load(path)[1] for path in
                                  [args.candidate, args.browser, args.initial]]
    baseline = json.loads(Path(args.baseline).read_text())
    cached = json.loads(Path(args.cached_candidate).read_text())
    # A selected-pair cache explicitly resolves sampled role outcomes to their
    # already recorded conditions; no repeated sampled games are needed.
    if cached['candidateSHA256'] != file_hash(args.candidate):
        raise ValueError('Cached selected-role records belong to another candidate')
    if cached['baselineSHA256'] != file_hash(args.baseline):
        raise ValueError('Cached selected-role records use another cohort')
    physics_hash = file_hash(Path(__file__).with_name('physics.py'))
    if any(record['provenance']['physicsSHA256'] != physics_hash
           for record in [candidate, browser, initial]):
        raise ValueError('All mode comparisons must use the same frozen physical game')
    if baseline['sources']['physics.py'] != physics_hash:
        raise ValueError('Cached baseline physics does not match')
    if any(file_hash(path) not in baseline['fixedOpponentSHA256'].values()
           for path in [args.browser, args.initial]):
        raise ValueError('Mode references must be the exact cached browser and initial actors')
    if initial['decisions'] != 0:
        raise ValueError('Initial counterpart must have zero experience')
    maps = baseline['maps']
    if len(maps) != 96:
        raise ValueError('Use the same 96-map broad development cohort')
    expected_modes = {
        'candidate_pair_sample', 'candidate_hider_sample_fixed_browser_seeker',
        'fixed_browser_hider_candidate_seeker_sample',
        'initial_hider_sample_fixed_browser_seeker',
        'fixed_browser_hider_initial_seeker_sample', 'browser_pair_sample',
        'initial_pair_sample'}
    expected_keys = {(row['seed'], row['scenario'], mode)
                     for row in maps for mode in expected_modes}
    cached_keys = {(row['seed'], row['scenario'], row['mode'])
                   for row in cached['episodes']}
    if cached_keys != expected_keys or len(cached['episodes']) != len(expected_keys):
        raise ValueError('Cache must contain each sampled condition exactly once on every map')
    source_names = ['evaluate_policy_modes.py', 'persistent_evaluate.py',
                    'persistent_actor.py', 'persistent_train.py', 'actor.py', 'physics.py']
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in source_names}
    identity = dict(candidateSHA256=file_hash(args.candidate), browserSHA256=file_hash(args.browser),
                    initialSHA256=file_hash(args.initial), baselineSHA256=file_hash(args.baseline),
                    cachedCandidateSHA256=file_hash(args.cached_candidate), sources=sources)
    identity_path = output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Partial modes belong to another selection')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    source_dir = output / 'source'
    source_dir.mkdir(exist_ok=True)
    for name in source_names:
        shutil.copyfile(Path(__file__).with_name(name), source_dir / name)
    partial = output / 'new-episodes.partial.json'
    rows = json.loads(partial.read_text()) if partial.exists() else []
    complete = {(row['seed'], row['scenario']) for row in rows}
    if len(rows) != len(complete) * 7:
        raise ValueError('Resume only whole maps with seven conditions')
    pending = [row for row in maps if (row['seed'], row['scenario']) not in complete]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(args.candidate, args.browser, args.initial)) as executor:
        for results in executor.map(evaluate_map, pending):
            rows.extend(results)
            temporary = partial.with_suffix('.tmp')
            temporary.write_text(json.dumps(rows, default=plain) + '\n')
            temporary.replace(partial)
            print(json.dumps(dict(maps=len(rows) // 7, newGames=len(rows))), flush=True)
    all_rows = cached['episodes'] + rows
    definitions = [
        ('Hider mean vs sampled, same sampled browser seeker', 'candidate_hider_mean_fixed_browser_seeker', 'candidate_hider_sample_fixed_browser_seeker'),
        ('Seeker mean vs sampled, same sampled browser hider', 'fixed_browser_hider_candidate_seeker_sample', 'fixed_browser_hider_candidate_seeker_mean'),
        ('Mean hider learning vs genuine mean initial, same opponent', 'candidate_hider_mean_fixed_browser_seeker', 'initial_hider_mean_fixed_browser_seeker'),
        ('Sampled hider learning vs genuine sampled initial, same opponent', 'candidate_hider_sample_fixed_browser_seeker', 'initial_hider_sample_fixed_browser_seeker'),
        ('Mean seeker learning vs genuine mean initial, same opponent', 'fixed_browser_hider_initial_seeker_mean', 'fixed_browser_hider_candidate_seeker_mean'),
        ('Sampled seeker learning vs genuine sampled initial, same opponent', 'fixed_browser_hider_initial_seeker_sample', 'fixed_browser_hider_candidate_seeker_sample'),
    ]
    effects = {name: contrast(all_rows, left, right) for name, left, right in definitions}
    summaries = {}
    for mode in dict.fromkeys(row['mode'] for row in all_rows):
        selected = [row for row in all_rows if row['mode'] == mode]
        summaries[mode] = dict(maps=len(selected), hiddenFraction=float(np.mean([row['hiddenFraction'] for row in selected])),
            completeSearchMisses=sum(row['info']['hidden'] == 160 for row in selected), roles=[])
        for role in range(2):
            durations = [value * DT for row in selected for value in row['roles'][role]['gripDurationsTicks']]
            requested = [[value * DT for row in selected
                          for value in row['roles'][role]['requestedHoldDurationsTicks'][tool]]
                         for tool in range(2)]
            summaries[mode]['roles'].append(dict(grips=len(durations), medianGripSeconds=float(np.median(durations)) if durations else 0,
                meanGripSeconds=float(np.mean(durations)) if durations else 0, maxGripSeconds=max(durations, default=0),
                fractionGripsShorterThanThreeTicks=float(np.mean(np.asarray(durations) < 3 * DT)) if durations else 0,
                requestedHoldMedianSeconds=[float(np.median(values)) if values else 0 for values in requested],
                commandCounts=np.sum([row['roles'][role]['commandCounts'] for row in selected], axis=0).tolist()))
    report = dict(format='frozen-policy-action-mode-validation-v1', **identity,
        scope='Same 96 repeated development maps. Role-mode comparisons keep the browser opponent sampled; whole-pair mean conditions match the UI global setting.',
        semantics='Mean means tanh Gaussian mean plus categorical argmax, with the actual recurrent requested-button controller. No buttons suppressed beyond the unchanged preparation mask.',
        candidateRoleSources=candidate['provenance']['roleSources'],
        newEvaluationActorDecisions=len(rows) * 480, cachedGamesReused=len(cached['episodes']),
        summary=summaries, contrasts=effects, newEpisodes=rows,
        automaticPromotion=False, sourceSHA256=file_hash(__file__))
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, default=plain) + '\n')
    print(json.dumps(effects, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['candidate', 'browser', 'initial', 'baseline', 'cached-candidate', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=2)
    main(parser.parse_args())
