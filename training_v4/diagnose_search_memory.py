"""Read-only memory, entity-order, and delayed-credit diagnostics.

Frozen recorded observations are used for representation counterfactuals; these
are not rollout success rates. No optimizer is created and no model is changed.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal, Categorical, kl_divergence
from diagnose_league_gradients import collect
from persistent_evaluate import sample
from persistent_train import file_hash
from residual_critic import ResidualCentralCritic
from train_league import load_actors
from physics import PhysicsEnv, DT
from fit_central_value import GAMMA


def describe(values):
    values = torch.as_tensor(values).flatten()
    if not values.numel():
        return dict(count=0)
    return dict(count=values.numel(), mean=float(values.mean()), median=float(values.median()),
                p95=float(torch.quantile(values, .95)), maximum=float(values.max()))


def difference(reference, changed, selected):
    normal_a, tools_a = reference[:2]
    normal_b, tools_b = changed[:2]
    return dict(
        jointKLnats=describe((kl_divergence(normal_a, normal_b).sum(-1) + kl_divergence(tools_a, tools_b).sum(-1))[selected]),
        deterministicMovementRMS=describe((normal_a.mean.tanh() - normal_b.mean.tanh()).square().mean(-1).sqrt()[selected]),
        toolTotalVariation=describe((tools_a.probs - tools_b.probs).abs().sum(-1).mean(-1)[selected] / 2))


@torch.no_grad()
def replay(model, observations, starts):
    memory = torch.zeros(observations.shape[1], 64)
    before = []
    for tick in range(len(observations)):
        memory = memory * (1 - starts[tick, :, None])
        before.append(memory.clone())
        memory = model(observations[tick], memory)[3]
    return torch.stack(before)


@torch.no_grad()
def memory_and_slots(model, newer, batch):
    observations, stored, starts, _, _ = batch
    observations, stored, starts = observations[:240], stored[:240], starts[:240]
    rebuilt = replay(model, observations, starts)
    newer_rebuilt = replay(newer, observations, starts)
    result = dict(maxStoredVersusRebuiltMemoryError=float((rebuilt - stored).abs().max()))
    # All selected rows lie in the complete first episode and beyond128 ticks.
    times = torch.arange(128, 240, 4).repeat_interleave(observations.shape[1])
    worlds = torch.arange(observations.shape[1]).repeat(28)
    chosen = observations[times, worlds]
    reference = model(chosen, rebuilt[times, worlds])
    hidden = chosen[:, 10] == 0
    result['truncatedObservationHistory'] = {}
    for length in [1, 4, 8, 16, 32, 64, 128]:
        memory = torch.zeros(len(times), 64)
        for offset in range(length, 0, -1):
            memory = model(observations[times - offset, worlds], memory)[3]
        changed = model(chosen, memory)
        result['truncatedObservationHistory'][str(length)] = difference(reference, changed, hidden)
    # Actual production chunks begin at0,32,...,224. Recompute under the next
    # trained checkpoint, once from its full history and once from the old
    # checkpoint's stored chunk-start memory. No further optimizer is run here.
    stale = []
    for begin in range(0, 240, 32):
        memory = stored[begin].clone()
        for tick in range(begin, min(begin + 32, 240)):
            memory = memory * (1 - starts[tick, :, None])
            stale.append(memory.clone())
            memory = newer(observations[tick], memory)[3]
    stale = torch.stack(stale)
    flat = observations.flatten(0, 1)
    active_hidden = (flat[:, 10] == 0) & (flat[:, 5] >= 1)
    result['adjacentCheckpointStoredChunkState'] = difference(
        newer(flat, newer_rebuilt.flatten(0, 1)), newer(flat, stale.flatten(0, 1)), active_hidden)
    result['adjacentCheckpointMemoryRMS'] = describe((newer_rebuilt - stale).square().mean(-1).sqrt())
    # Swap the first two visible object records, keeping every feature and
    # existing pre-action memory identical. Semantic entity set is unchanged.
    entities = flat[:, 18:114].reshape(-1, 6, 16)
    changed = flat.clone()
    other = changed[:, 18:114].reshape(-1, 6, 16)
    other[:, 0], other[:, 1] = entities[:, 1].clone(), entities[:, 0].clone()
    has_pair = (entities[:, 0, 0] > .5) & (entities[:, 1, 0] > .5) & (flat[:, 5] >= 1)
    lengths = (entities[:, :2, 1:4] * torch.tensor([6, 6, 2])).norm(dim=-1)
    near_tie = has_pair & ((lengths[:, 0] - lengths[:, 1]).abs() < .05)
    baseline = model(flat, rebuilt.flatten(0, 1))
    altered = model(changed, rebuilt.flatten(0, 1))
    result['sameEntitySetDifferentSlotOrder'] = dict(
        allVisiblePairs=difference(baseline, altered, has_pair),
        pairsWithinFiveCentimetresOfEqualDistance=difference(baseline, altered, near_tie),
        scope='One-step representation sensitivity on recorded observations, not an evaluated alternative policy or a physical intervention.')
    return result


@torch.no_grad()
def natural_rank_changes(models):
    results = []
    for index in range(12):
        env = PhysicsEnv(seed=1500183000 + index, scenario='rooms', size=[6, 8, 10, 12][index % 4], n_boxes=8, n_ramps=2)
        observations = env.observe()
        memories = [torch.zeros(1, 64), torch.zeros(1, 64)]
        buttons = np.zeros((2, 2), np.float32)
        generators = [np.random.default_rng(1500183000 + index + offset) for offset in [912341, 2912341]]
        previous = [[], []]
        counts = [dict(playFrames=0, multiObjectFrames=0, sameVisibleSetRankSwaps=0,
                       persistentEntitySlotChanges=0, visibleOpponentFrames=0) for _ in range(2)]
        for tick in range(240):
            actions = np.zeros((2, 5), np.float32)
            for role in range(2):
                position = env.data.qpos[role * 4:role * 4 + 3]
                ids = [i for i in range(env.n_obj) if env.object_seen[role, i]]
                ids.sort(key=lambda i: np.linalg.norm(env.data.qpos[8 + 4 * i:11 + 4 * i] - position))
                ids = ids[:6]
                if tick >= 80:
                    count = counts[role]
                    count['playFrames'] += 1
                    count['multiObjectFrames'] += len(ids) >= 2
                    count['visibleOpponentFrames'] += bool(env.seen[role, 1 - role])
                    count['sameVisibleSetRankSwaps'] += len(ids) >= 2 and set(ids) == set(previous[role]) and ids != previous[role]
                    common = set(ids) & set(previous[role])
                    count['persistentEntitySlotChanges'] += any(ids.index(i) != previous[role].index(i) for i in common)
                previous[role] = ids
                actions[role], memories[role], buttons[role], _ = sample(
                    models[role], observations[role], memories[role], buttons[role], generators[role])
            observations, _, _, info = env.step(actions)
        results.append(dict(seed=1500183000 + index, size=env.arena['width'], objects=env.n_obj,
                            hiddenFraction=info['hidden'] / 160, roles=counts))
        env.close()
    return dict(newReadOnlyActorDecisions=12 * 480, episodes=results,
                roleTotals=[{key: sum(row['roles'][role][key] for row in results) for key in results[0]['roles'][role]} for role in range(2)])


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    before = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    after = torch.load(args.newer_checkpoint, map_location='cpu', weights_only=False)
    if after['pilotUpdates'] - before['pilotUpdates'] != 1:
        raise ValueError('The staleness comparison needs adjacent actual training updates')
    models, newer = load_actors(before), load_actors(after)
    snapshots = [copy.deepcopy(model.state_dict()) for model in models + newer]
    fitted = torch.load(args.critic_fit, map_location='cpu', weights_only=False)
    critic = ResidualCentralCritic(fitted['dropout']).eval()
    critic.load_state_dict(before['centralCritic'])
    torch.manual_seed(87223)
    cache = output / 'rollout.pt'
    if cache.exists():
        data = torch.load(cache, map_location='cpu', weights_only=False)
        if data['checkpointSHA256'] != file_hash(args.checkpoint):
            raise ValueError('Diagnostic cache belongs to another checkpoint')
    else:
        data = collect(models, critic, args)
        data['checkpointSHA256'] = file_hash(args.checkpoint)
        torch.save(data, cache)
    roles = [memory_and_slots(models[role], newer[role], data['buffers'][role]) for role in range(2)]
    total_prep, relies_on_bootstrap = 0, 0
    for tick in range(3840):
        if tick % 240 < 80:
            total_prep += 1
            first_play = (tick // 240) * 240 + 80
            relies_on_bootstrap += first_play // 256 != tick // 256
    credit = dict(gamma=GAMMA, gaeLambda=.98, bpttTicks=32, bpttSeconds=32 * DT,
        preparationTicks=80, playTicks=160, rolloutTicks=256,
        directRewardGAEWeightAtDelay={str(delay): (GAMMA * .98) ** delay for delay in [16, 32, 40, 64, 80, 120, 160]},
        discountedObjectiveWeightAfterPreparation=GAMMA ** 80,
        preparationStepsAcrossAlignmentCycle=total_prep,
        preparationStepsWithFirstPlayBeyondRollout=relies_on_bootstrap,
        measuredPreparationAdvantage=describe(data['advantages'][:80].abs()),
        note='GAE supplies delayed scalar credit beyond BPTT. Pre-play actions whose first play reward falls outside a rollout depend on the critic bootstrap.')
    ranks = natural_rank_changes(models)
    for old, model in zip(snapshots, models + newer):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(old[name], value, atol=0, rtol=0)
    report = dict(format='read-only-search-memory-diagnostic-v1',
        beforeCheckpointSHA256=file_hash(args.checkpoint), afterCheckpointSHA256=file_hash(args.newer_checkpoint),
        beforeUpdate=before['pilotUpdates'], afterUpdate=after['pilotUpdates'], optimizerStepsMadeByAudit=0,
        cachedDiagnosticActorDecisions=data['diagnosticActorDecisions'], roles=roles, delayedCredit=credit,
        naturalEntityOrdering=ranks,
        scope='Frozen-policy representation probes and12 diagnostic room games. Counterfactual observation ordering/history is not a performance claim or a training change.',
        sourceSHA256=file_hash(__file__), physicsSHA256=file_hash(Path(__file__).with_name('physics.py')))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'newer-checkpoint', 'critic-fit', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1500181000)
    main(parser.parse_args())
