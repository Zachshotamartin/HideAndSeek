"""Predeclared fixed-opponent and physical-tool checks for the encoder comparison.

This evaluator runs frozen restricted actors only. It reuses a previous episode
only when both actor tensors, controller implementation, physical mode and exact
map configuration match; aliases within a cohort share one physical rollout.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil

import numpy as np

from entity_actor import load_pair
from persistent_evaluate import episode, summarize, plain
from persistent_train import file_hash


ACTORS = None
SIGNATURES = None


def actor_signature(model):
    digest = hashlib.sha256(type(model).__name__.encode())
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def conditions(pairs):
    selected = pairs['selected']
    compared = {'selected/pair': (selected, 'learned'),
                'initial/pair': (pairs['initial'], 'learned')}
    for opponent in ['initial', 'warm', 'mid']:
        compared[f'selected/hider-v-{opponent}'] = ([selected[0], pairs[opponent][1]], 'learned')
        compared[f'selected/seeker-v-{opponent}'] = ([pairs[opponent][0], selected[1]], 'learned')
    for arm in ['legacy', 'entity']:
        actors = pairs[arm]
        for mode in ['learned', 'no-hider-tools', 'no-seeker-tools', 'no-tools']:
            compared[f'{arm}/{mode}'] = (actors, mode)
        for opponent in ['selected', 'initial', 'warm', 'mid']:
            compared[f'{arm}/hider-v-{opponent}'] = ([actors[0], pairs[opponent][1]], 'learned')
            compared[f'{arm}/seeker-v-{opponent}'] = ([pairs[opponent][0], actors[1]], 'learned')
    return compared


def initialize(paths):
    global ACTORS, SIGNATURES
    pairs = {name: load_pair(path)[0] for name, path in paths.items()}
    ACTORS = conditions(pairs)
    identities = {id(model): actor_signature(model) for pair in pairs.values() for model in pair}
    SIGNATURES = {name: ':'.join([identities[id(actors[0])], identities[id(actors[1])], mode])
                  for name, (actors, mode) in ACTORS.items()}


def evaluate_map(task):
    configuration, cached = task
    prior = {row['policyPairSignature']: row for row in cached}
    rows = []
    simulated = 0
    for name, (actors, mode) in ACTORS.items():
        signature = SIGNATURES[name]
        if signature not in prior:
            prior[signature] = episode(actors, configuration['seed'], configuration['scenario'], mode,
                                       arena_config=configuration['arenaConfig'])
            prior[signature]['policyPairSignature'] = signature
            simulated += 1
        row = dict(copy.deepcopy(prior[signature]), mode=name)
        rows.append(row)
    return dict(episodes=rows, simulatedGames=simulated)


def contrast_definitions():
    compared = []
    for arm in ['legacy', 'entity']:
        compared.extend([
            (f'{arm}: Hider grab/lock benefit', f'{arm}/learned', f'{arm}/no-hider-tools'),
            (f'{arm}: Seeker grab/lock benefit', f'{arm}/no-seeker-tools', f'{arm}/learned')])
        for opponent in ['selected', 'initial', 'warm', 'mid']:
            h_reference = 'selected/pair' if opponent == 'selected' else f'selected/hider-v-{opponent}'
            s_reference = 'selected/pair' if opponent == 'selected' else f'selected/seeker-v-{opponent}'
            compared.extend([
                (f'{arm}: Hider change vs fixed {opponent} seeker', f'{arm}/hider-v-{opponent}', h_reference),
                (f'{arm}: Seeker change vs fixed {opponent} hider', s_reference, f'{arm}/seeker-v-{opponent}')])
    for opponent in ['selected', 'initial', 'warm', 'mid']:
        compared.extend([
            (f'Entity minus legacy: Hider vs fixed {opponent} seeker', f'entity/hider-v-{opponent}', f'legacy/hider-v-{opponent}'),
            (f'Entity minus legacy: Seeker vs fixed {opponent} hider', f'legacy/seeker-v-{opponent}', f'entity/seeker-v-{opponent}')])
    return compared


def breakdown(rows):
    def summarize_group(group):
        return dict(games=len(group), hiddenFraction=float(np.mean([row['hiddenFraction'] for row in group])),
                    completeSearchMisses=sum(row['info']['hidden'] == row['info']['play_steps'] for row in group))
    result = {}
    for mode in dict.fromkeys(row['mode'] for row in rows):
        selected = [row for row in rows if row['mode'] == mode]
        result[mode] = dict(all=summarize_group(selected),
            scenario={scenario: summarize_group([row for row in selected if row['scenario'] == scenario])
                      for scenario in ['shelter', 'rooms', 'open']},
            actualObjectCount={str(count): summarize_group([row for row in selected if row['actualObjectCount'] == count])
                               for count in sorted(set(row['actualObjectCount'] for row in selected))})
    return result


def main(args):
    protocol = json.loads(Path(args.protocol).read_text())
    paths = {name: protocol['assets'][name]['path'] for name in ['selected', 'initial', 'warm', 'mid']}
    paths.update(legacy=args.legacy, entity=args.entity)
    records = {name: load_pair(path)[1] for name, path in paths.items()}
    source_names = ['evaluate_entity.py', 'persistent_evaluate.py', 'entity_actor.py',
                    'persistent_actor.py', 'persistent_train.py', 'actor.py', 'physics.py']
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in source_names}
    if sources['physics.py'] != protocol['physicsSHA256']:
        raise ValueError('Frozen physics identity changed')
    if records['initial']['decisions'] != 0:
        raise ValueError('Keep the fixed initial opponent genuinely zero-experience')
    for arm in ['legacy', 'entity']:
        record = records[arm]
        if record['provenance']['physicsSHA256'] != protocol['physicsSHA256']:
            raise ValueError('Actors must use the same physical game')
        if args.milestone == 0:
            if record['newActorUpdates'] != 0 or file_hash(paths[arm]) != protocol['assets'][arm]['sha256']:
                raise ValueError('Zero-update assessment needs both original prepared actors')
        elif (record['totalPolicyInteractions'] != args.milestone
              or record['arguments']['encoder'] != arm
              or record['provenance']['protocolSHA256'] != file_hash(args.protocol)):
            raise ValueError('Compare the matched immutable milestone from this protocol')
    if args.milestone and any(records['legacy'][key] != records['entity'][key] for key in ['worldRNG', 'opponentRNG']):
        raise ValueError('Matched world/assignment schedules diverged')
    if args.milestone not in protocol['milestones']:
        raise ValueError('This milestone was not predeclared')
    if any(not 1500000000 <= row['seed'] < 1900000000 for row in protocol['maps']):
        raise ValueError('Only the declared development seeds are allowed')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation.json').exists():
        raise ValueError('Preserve completed assessments')
    identity = dict(protocolSHA256=file_hash(args.protocol), milestone=args.milestone,
        checkpointSHA256={name: file_hash(path) for name, path in paths.items()}, sources=sources)
    identity_path = output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Partial evaluation identity changed')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    (output / 'source').mkdir(exist_ok=True)
    for name in source_names:
        shutil.copyfile(Path(__file__).with_name(name), output / 'source' / name)
    cache = {}
    if args.cache:
        cached = json.loads(Path(args.cache).read_text())
        if cached['protocolSHA256'] != file_hash(args.protocol) or cached['sources'] != sources:
            raise ValueError('Cache needs the same cohort and exact inference/physics implementation')
        for row in cached['episodes']:
            key = (row['seed'], row['scenario'])
            cache.setdefault(key, []).append(row)
    partial = output / 'maps.partial.json'
    finished = json.loads(partial.read_text()) if partial.exists() else []
    completed = {(result['episodes'][0]['seed'], result['episodes'][0]['scenario']) for result in finished}
    tasks = [(row, cache.get((row['seed'], row['scenario']), [])) for row in protocol['maps']
             if (row['seed'], row['scenario']) not in completed]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(paths,)) as executor:
        for result in executor.map(evaluate_map, tasks):
            finished.append(result)
            temporary = partial.with_suffix('.tmp')
            temporary.write_text(json.dumps(finished, default=plain) + '\n')
            temporary.replace(partial)
            print(json.dumps(dict(maps=len(finished), conditions=32,
                simulatedGames=sum(row['simulatedGames'] for row in finished))), flush=True)
    episodes = [row for result in finished for row in result['episodes']]
    summary, contrasts = summarize(episodes, contrast_definitions())
    counts = breakdown(episodes)
    for mode, row in summary.items():
        row['completeSearchMisses'] = counts[mode]['all']['completeSearchMisses']
    simulated = sum(result['simulatedGames'] for result in finished)
    result = dict(format='matched-entity-encoder-development-v1', **identity,
        training={name: {key: records[name].get(key, 0) for key in ['decisions', 'parentDecisions',
            'totalPolicyInteractions', 'currentPolicyDecisions', 'historicalPolicyDecisions', 'activePolicySamples']}
            for name in ['legacy', 'entity']},
        maps=protocol['maps'], simulatedGames=simulated, newEvaluationActorDecisions=simulated * 480,
        cachedOrAliasedConditionGames=len(episodes) - simulated,
        scope='One paired training seed, 96 predeclared development maps reused across zero/8/16M. No final-test claim. Each role faces identical frozen selected/initial/warm/intermediate opponents.',
        sampling='Exact sampled policy, independent role-specific common uniform tapes across all modes.',
        ablation='Disable physical grab/lock for named roles; preserve requested controller state and physical pushing.',
        decision='No automatic promotion. Credible role regression blocks that role; inconclusive tool intervals do not demonstrate useful tools.',
        summary=summary, contrasts=contrasts, breakdown=counts, episodes=episodes, publicAssetsChanged=False)
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2, default=plain) + '\n')
    print(json.dumps(contrasts, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'legacy', 'entity', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--milestone', type=int, choices=[0, 7995392, 15990784], required=True)
    parser.add_argument('--cache')
    parser.add_argument('--workers', type=int, default=2)
    main(parser.parse_args())
