"""Read-only complete-game value errors before the matched encoder continuation."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from central_critic import batch_states
from entity_actor import load_pair
from env_pool import PhysicsEnvPool
from fit_central_value import GAMMA, REWARD_SCALE, geometry_hash
from league_ppo import assign_roles, act_grouped
from persistent_train import file_hash
from residual_critic import ResidualCentralCritic
from train_entity import initialize_critic


torch.set_num_threads(1)


def collect(protocol, arm, envs, workers):
    models, record = load_pair(protocol['assets'][arm]['path'])
    histories = [load_pair(protocol['assets'][name]['path'])[0] for name in protocol['history']]
    source = torch.load(protocol['assets']['critic']['path'], map_location='cpu', weights_only=False)
    critic, _ = initialize_critic(source, .0001)
    critic.eval().requires_grad_(False)
    original = [copy.deepcopy(model.state_dict()) for model in models]
    maps = protocol['maps']
    if len(maps) % envs:
        raise ValueError('Collect complete equal-size episode batches')
    role_rng = np.random.default_rng(971531)
    torch.set_rng_state(record['torchRNG'])
    all_predictions, all_partials, all_targets, all_roles = [], [], [], []
    configs = [dict(seed=row['seed'], scenario=row['scenario'], **row['arenaConfig']) for row in maps]
    with PhysicsEnvPool(configs[:envs], workers=workers, with_central_state=True) as pool:
        for begin in range(0, len(maps), envs):
            if begin:
                changes = configs[begin:begin + envs]
                pool.reset_at(range(envs), [row['seed'] for row in changes], changes)
            physical = pool.observations.copy()
            memories = [torch.zeros(envs, 64), torch.zeros(envs, 64)]
            buttons = np.zeros((envs, 2, 2), np.float32)
            roles = assign_roles(role_rng, envs, 'B', len(histories))
            predictions, partials, rewards = [], [], []
            with torch.no_grad():
                for tick in range(240):
                    state = batch_states(pool.central_states)
                    before = torch.stack(memories, 1)
                    actions, memories, buttons, records = act_grouped(
                        models, histories, physical, memories, buttons, roles)
                    partial = torch.stack([row[3] for row in records], -1)
                    predictions.append(critic(state, before, partial))
                    partials.append(ResidualCentralCritic.baseline(partial))
                    physical, returned, done, _ = pool.step(actions)
                    if bool(done.any()) != (tick == 239):
                        raise AssertionError('Only complete games enter this check')
                    np.testing.assert_array_equal(returned.sum(1), 0)
                    rewards.append(torch.from_numpy(returned[:, 0].copy()) * REWARD_SCALE)
            target = torch.zeros(240, envs)
            following = torch.zeros(envs)
            for tick in reversed(range(240)):
                following = rewards[tick] + GAMMA * following
                target[tick] = following
            all_predictions.append(torch.stack(predictions).T.numpy())
            all_partials.append(torch.stack(partials).T.numpy())
            all_targets.append(target.T.numpy())
            all_roles.append(roles)
            print(json.dumps(dict(arm=arm, completeGames=begin + envs)), flush=True)
    for before, model in zip(original, models):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        assert all(parameter.grad is None for parameter in model.parameters())
    return dict(central=np.concatenate(all_predictions), partial=np.concatenate(all_partials),
                target=np.concatenate(all_targets), roles=np.concatenate(all_roles), configs=configs)


def assess(data, groups):
    # Assignment depends on complete static geometry, never on measured returns.
    fit = np.array([int(group[:8], 16) % 2 == 0 for group in groups])
    if min(fit.sum(), (~fit).sum()) < 16:
        raise ValueError('Insufficient separate geometry groups for the time baseline')
    mean_time_return = data['target'][fit].mean(0)
    errors = {name: (data[name] - data['target']) ** 2 for name in ['central', 'partial']}
    errors['time'] = (mean_time_return - data['target']) ** 2
    report = dict(fitGames=int(fit.sum()), assessmentGames=int((~fit).sum()),
                  fitGeometryGroups=len(set(np.asarray(groups)[fit])),
                  assessmentGeometryGroups=len(set(np.asarray(groups)[~fit])))
    for split, selected in [('timeBaselineFit', fit), ('assessment', ~fit)]:
        report[split] = {name: dict(meanGameMSE=float(error[selected].mean()),
            preparationMSE=float(error[selected, :80].mean()),
            playMSE=float(error[selected, 80:].mean())) for name, error in errors.items()}
    # Bootstrap whole geometry groups so repeated empty-room geometries cannot
    # contribute independent samples to this diagnostic uncertainty estimate.
    assessed_groups = sorted(set(np.asarray(groups)[~fit]))
    rng = np.random.default_rng(197391)
    comparisons = {}
    for baseline in ['partial', 'time']:
        differences = (errors['central'] - errors[baseline]).mean(1)
        group_values = np.asarray([differences[np.asarray(groups) == group].mean()
                                   for group in assessed_groups])
        boot = rng.choice(group_values, (10000, len(group_values)), replace=True).mean(1)
        comparisons[f'centralMinus{baseline.title()}MSE'] = dict(
            mean=float(group_values.mean()), bootstrap95Percent=np.quantile(boot, [.025, .975]).tolist())
    report['assessmentContrasts'] = comparisons
    report['sourceWeightsUnchanged'] = True
    return report


def main(args):
    protocol = json.loads(Path(args.protocol).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        raise ValueError('Keep this zero-update value assessment immutable')
    for entry in protocol['assets'].values():
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('Frozen input identity changed')
    source_names = ['check_entity_value.py', 'train_entity.py', 'entity_actor.py',
        'persistent_actor.py', 'actor.py', 'league_ppo.py', 'central_critic.py',
        'residual_critic.py', 'env_pool.py', 'fit_central_value.py', 'physics.py']
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in source_names}
    (output / 'source').mkdir(exist_ok=True)
    for name in source_names:
        shutil.copyfile(Path(__file__).with_name(name), output / 'source' / name)
    reports = {}
    for arm in ['legacy', 'entity']:
        data = collect(protocol, arm, args.envs, args.workers)
        groups = [geometry_hash(config) for config in data['configs']]
        reports[arm] = assess(data, groups)
        np.savez_compressed(output / f'{arm}.npz', **{name: value for name, value in data.items()
                                                    if name != 'configs'})
        (output / f'{arm}-maps.json').write_text(json.dumps(dict(configs=data['configs'], geometryGroups=groups), indent=2))
    result = dict(format='entity-zero-update-value-error-v1', protocolSHA256=file_hash(args.protocol),
        sources=sources, arms=reports, actorUpdates=0, criticUpdates=0,
        evaluationActorDecisions=len(protocol['maps']) * 480 * 2,
        target='Exact complete-game discounted original hider reward, gamma .998 and optimization scale .05; seeker values are its negative.',
        split='Time-only baseline fit and assessment use disjoint static geometry groups; current actor memories and partial values are from the same actual pre-action role in league B.',
        scope='Small diagnostic of inherited value estimates, not calibrated advantage certification or a learned critic fit. Maps are the predeclared development cohort; final test untouched.')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--envs', type=int, default=24)
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args())
