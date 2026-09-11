"""Compare entity-order events in exact failed and successful room searches.

The assessed hider/seeker actors stay frozen. Each replay must reproduce its
cached outcome. Counterfactual slot swaps are inference probes only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import kl_divergence
from persistent_actor import augment
from persistent_evaluate import load, sample
from persistent_train import file_hash
from physics import PhysicsEnv


@torch.no_grad()
def replay(models, expected):
    env = PhysicsEnv(seed=expected['seed'], scenario=expected['scenario'], **expected['arenaConfig'])
    observations = env.observe()
    memories = [torch.zeros(1, 64), torch.zeros(1, 64)]
    buttons = np.zeros((2, 2), np.float32)
    generators = [np.random.default_rng(expected['seed'] + offset) for offset in [912341, 2912341]]
    previous = []
    counts = dict(playFrames=0, noVisibleObjects=0, oneVisibleObject=0, multiObjectFrames=0,
                  sameVisibleSetRankSwaps=0, persistentEntitySlotChanges=0)
    divergences = []
    for tick in range(240):
        position = env.data.qpos[4:7]
        ids = [i for i in range(env.n_obj) if env.object_seen[1, i]]
        ids.sort(key=lambda i: np.linalg.norm(env.data.qpos[8 + 4 * i:11 + 4 * i] - position))
        ids = ids[:6]
        if tick >= 80:
            counts['playFrames'] += 1
            counts['noVisibleObjects'] += len(ids) == 0
            counts['oneVisibleObject'] += len(ids) == 1
            counts['multiObjectFrames'] += len(ids) >= 2
            counts['sameVisibleSetRankSwaps'] += len(ids) >= 2 and set(ids) == set(previous) and ids != previous
            common = set(ids) & set(previous)
            counts['persistentEntitySlotChanges'] += any(ids.index(i) != previous.index(i) for i in common)
            if len(ids) >= 2:
                observed = torch.from_numpy(augment(observations[1], buttons[1])[None])
                modified = observed.clone()
                modified[:, 18:34], modified[:, 34:50] = observed[:, 34:50].clone(), observed[:, 18:34].clone()
                base, changed = models[1](observed, memories[1]), models[1](modified, memories[1])
                divergence = (kl_divergence(base[0], changed[0]).sum(-1) + kl_divergence(base[1], changed[1]).sum(-1)).item()
                divergences.append(divergence)
        previous = ids
        actions = np.zeros((2, 5), np.float32)
        for role in range(2):
            actions[role], memories[role], buttons[role], _ = sample(
                models[role], observations[role], memories[role], buttons[role], generators[role])
        observations, _, _, info = env.step(actions)
    if info != expected['info']:
        raise AssertionError('Read-only replay differs from the cached assessed outcome')
    result = dict(seed=expected['seed'], size=env.arena['width'], objects=env.n_obj,
        completeSearchMiss=info['hidden'] == 160, hiddenFraction=info['hidden'] / 160,
        counts=counts, swappedEntitySetKL=divergences)
    env.close()
    return result


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.report).read_text())
    parent, _ = load(args.parent)
    seeker, _ = load(args.seeker)
    if file_hash(args.seeker) != report['checkpointSHA256']:
        raise ValueError('Use the exact assessed sustained checkpoint')
    expected = [row for row in report['newEpisodes']
                if row['scenario'] == 'rooms' and row['mode'] == 'fixed_parent_hider_new_seeker']
    if len(expected) != 32:
        raise ValueError('The predeclared room cohort must have 32 maps')
    rows = [replay([parent[0], seeker[1]], row) for row in expected]
    groups = {}
    for label, predicate in [('completeMisses', lambda row: row['completeSearchMiss']),
                             ('sawHider', lambda row: not row['completeSearchMiss'])]:
        selected = [row for row in rows if predicate(row)]
        counts = {key: sum(row['counts'][key] for row in selected) for key in rows[0]['counts']}
        divergences = [value for row in selected for value in row['swappedEntitySetKL']]
        groups[label] = dict(maps=len(selected), mapsWithZeroProps=sum(row['objects'] == 0 for row in selected),
            mapsWithNoPersistentSlotChanges=sum(row['counts']['persistentEntitySlotChanges'] == 0 for row in selected),
            counts=counts, eventFraction={key: value / counts['playFrames'] for key, value in counts.items() if key != 'playFrames'},
            swappedEntitySetKL=dict(count=len(divergences), mean=float(np.mean(divergences)) if divergences else None,
                                   p95=float(np.quantile(divergences, .95)) if divergences else None))
    result = dict(format='read-only-entity-order-failure-comparison-v1',
        checkpointSHA256=file_hash(args.seeker), hiderSHA256=file_hash(args.parent), sourceReportSHA256=file_hash(args.report),
        newReadOnlyActorDecisions=len(rows) * 480, exactCachedReplays=len(rows), optimizerSteps=0,
        groups=groups, episodes=rows,
        interpretation='Descriptive association in32 fixed development rooms, not causality. Slot-swap inference changes neither game state nor evaluated actions.',
        sourceSHA256=file_hash(__file__))
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['report', 'parent', 'seeker', 'output']:
        parser.add_argument('--' + name, required=True)
    main(parser.parse_args())
