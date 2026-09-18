"""Whole-map validation of a residual CTDE baseline; no actor optimization.

Includes the zero-correction baseline as epoch zero. Assessment maps are never
used to choose weights, regularization, learning rate, or stopping epoch.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from fit_central_value import collect, geometry_hash, STEPS
from persistent_actor import PersistentActor, FORMAT
from persistent_train import file_hash
from residual_critic import ResidualCentralCritic, SCHEMA

torch.set_num_threads(1)


def predictions(critic, data, batch_size=2048):
    critic.eval()
    output = []
    with torch.no_grad():
        for begin in range(0, len(data['targets']), batch_size):
            selected = slice(begin, begin + batch_size)
            output.append(critic(
                {key: value[selected] for key, value in data['states'].items()},
                data['memories'][selected], data['partialValues'][selected]))
    return torch.cat(output)


def assess(critic, data, time_mean):
    target = data['targets']
    baseline = ResidualCentralCritic.baseline(data['partialValues'])
    errors = {
        'residualCentral': (predictions(critic, data) - target).square(),
        'partialPairPrediction': (baseline - target).square(),
        'partialHider': (data['partialValues'][:, 0] - target).square(),
        'partialSeeker': (data['partialValues'][:, 1] + target).square(),
        'timeMean': (time_mean[data['phases']] - target).square(),
    }
    errors['meanIndividualPartialMSE'] = .5 * (errors['partialHider'] + errors['partialSeeker'])
    per_episode = {key: value.reshape(data['episodes'], STEPS).mean(-1).numpy()
                   for key, value in errors.items()}
    generator = np.random.default_rng(170922)
    comparisons = {}
    for name in errors:
        if name == 'residualCentral':
            continue
        difference = per_episode['residualCentral'] - per_episode[name]
        samples = generator.choice(difference, size=(10000, len(difference)), replace=True).mean(1)
        comparisons[name] = dict(centralMinusBaseline=float(difference.mean()),
            bootstrap95Percent=np.quantile(samples, [.025, .975]).tolist())
    return dict(mse={key: float(value.mean()) for key, value in errors.items()},
                comparisons=comparisons,
                perEpisodeMSE={key: value.tolist() for key, value in per_episode.items()})


def fit(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'assessment.json').exists():
        raise ValueError('Use a fresh output directory; preserve assessed results')
    saved = torch.load(args.parent, map_location='cpu', weights_only=False)
    if saved['format'] != FORMAT:
        raise ValueError('Expected our persistent-button parent')
    if saved['provenance']['physicsSHA256'] != file_hash(Path(__file__).with_name('physics.py')):
        raise ValueError('Parent and current physical environment differ')
    models = [PersistentActor().eval() for _ in range(2)]
    for model, state in zip(models, saved['models']):
        model.load_state_dict(state)
    actor_before = [copy.deepcopy(model.state_dict()) for model in models]
    used = set()
    if args.exclude_dataset:
        previous = torch.load(args.exclude_dataset, map_location='cpu', weights_only=False)
        for split in previous.values():
            used.update(geometry_hash(config) for config in split['configs'])
        del previous
    excluded_maps = len(used)
    dataset = {}
    for name, episodes, offset, namespace in [
        ('train', args.train_episodes, 0, 0),
        ('selection', args.selection_episodes, 1, 1470000000),
        ('heldout', args.test_episodes, 2, 1480000000),
    ]:
        dataset[name] = collect(models, episodes, args.envs, args.workers,
                                args.seed + offset, namespace, used)
        torch.save(dataset[name], output / f'{name}-dataset.pt')
    train, selection, heldout = [dataset[name] for name in ['train', 'selection', 'heldout']]
    torch.manual_seed(args.seed + 3)
    critic = ResidualCentralCritic(args.dropout)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    time_mean = train['targets'].reshape(args.train_episodes, STEPS).mean(0)
    best_error = float((predictions(critic, selection) - selection['targets']).square().mean())
    best = dict(epoch=0, critic=copy.deepcopy(critic.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()))
    log = [dict(epoch=0, trainMSE=float((predictions(critic, train) - train['targets']).square().mean()),
                selectionMSE=best_error, note='Exact detached partial-pair baseline; zero correction')]
    print(json.dumps(log[0]), flush=True)
    stale = 0
    for epoch in range(args.epochs):
        ordering = torch.randperm(len(train['targets']))
        errors = []
        critic.train()
        for offset in range(0, len(ordering), args.batch_size):
            selected = ordering[offset:offset + args.batch_size]
            residual = critic.residual({key: value[selected] for key, value in train['states'].items()},
                                       train['memories'][selected])
            value = critic.baseline(train['partialValues'][selected]) + residual
            error = (value - train['targets'][selected]).square().mean()
            loss = error + args.correction_penalty * residual.square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
            optimizer.step()
            errors.append(float(error.detach()))
        error = float((predictions(critic, selection) - selection['targets']).square().mean())
        row = dict(epoch=epoch + 1, trainMSE=float(np.mean(errors)), selectionMSE=error)
        log.append(row)
        print(json.dumps(row), flush=True)
        if error < best_error - args.minimum_improvement:
            best_error = error
            best = dict(epoch=epoch + 1, critic=copy.deepcopy(critic.state_dict()),
                        optimizer=copy.deepcopy(optimizer.state_dict()))
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    critic.load_state_dict(best['critic'])
    for before, model in zip(actor_before, models):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(before[name], value, atol=0, rtol=0)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError('Privileged fitting reached actor gradients')
    report = dict(format='residual-central-value-fit-assessment-v1',
        parentSHA256=file_hash(args.parent), parentDecisions=saved['decisions'],
        actorUpdates=0, actorParametersUnchanged=True, selectedEpoch=best['epoch'],
        episodes={name: part['episodes'] for name, part in dataset.items()},
        environmentActorDecisions={name: part['episodes'] * STEPS * 2 for name, part in dataset.items()},
        previousDevelopmentMapsExcluded=excluded_maps,
        split='Complete episodes; static map geometry is disjoint modulo rotation/reflection and ignores agent spawn. Fresh selection 1.47B and assessment 1.48B. Final game tests at 1.9B remain untouched.',
        baseline='Detached .5*(current pre-action partial hider V - current pre-action partial seeker V). Correction starts exactly zero.',
        units='MSE of discounted returns, uniform .05 reward scale and gamma .998. Negative residual-minus-baseline is better.',
        selection=assess(critic, selection, time_mean), heldout=assess(critic, heldout, time_mean),
        mapHashes={name: part['mapHashes'] for name, part in dataset.items()}, log=log,
        sources={name: file_hash(Path(__file__).with_name(name)) for name in [
            'fit_residual_value.py', 'residual_critic.py', 'fit_central_value.py', 'central_critic.py',
            'persistent_actor.py', 'physics.py', 'env_pool.py']})
    checkpoint = dict(format=SCHEMA, critic=best['critic'], criticOptimizer=best['optimizer'],
        dropout=args.dropout, timeMean=time_mean, parentSHA256=file_hash(args.parent),
        parentPath=str(Path(args.parent).resolve()), report=report, arguments=vars(args))
    torch.save(checkpoint, output / 'fitted-critic.pt')
    (output / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(selectedEpoch=best['epoch'], heldout=report['heldout']['mse'],
                         comparisons=report['heldout']['comparisons']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--exclude-dataset')
    parser.add_argument('--train-episodes', type=int, default=4096)
    parser.add_argument('--selection-episodes', type=int, default=256)
    parser.add_argument('--test-episodes', type=int, default=256)
    parser.add_argument('--envs', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--minimum-improvement', type=float, default=.001)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--learning-rate', type=float, default=.0001)
    parser.add_argument('--weight-decay', type=float, default=.001)
    parser.add_argument('--correction-penalty', type=float, default=.1)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=491039)
    fit(parser.parse_args())
