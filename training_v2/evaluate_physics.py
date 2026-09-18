"""Held-out evaluation and causal tool ablations for our physical policies."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from actor import PhysicalActor
from physics import PhysicsEnv

torch.set_num_threads(1)


def reactive_action(observation, role, step):
    """Transparent observation-only comparator, with no map or object planner."""
    action = np.zeros(5, np.float32)
    if observation[10] > .5:
        direction = observation[11:13].copy() * (1 if role else -1)
        norm = max(.1, np.linalg.norm(direction))
        action[:2] = direction / norm * .55
        angle = math.atan2(direction[1], direction[0])
        action[2] = np.clip(angle * .6, -.7, .7)
    else:
        rays = observation[-24:]
        # Persistent exploratory movement while turning away from close obstacles.
        action[0] = .45 if min(rays[0], rays[1], rays[-1]) > .13 else -.15
        action[2] = .28 if rays[1:7].mean() >= rays[-6:].mean() else -.28
        if role == 0:
            action[0] *= .5
    return action


def load_models(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    models = []
    for state in saved['models']:
        model = PhysicalActor(saved['observationSize'])
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, saved


def episode(models, seed, scenario, mode='learned', trace=False, size=8, n_boxes=3, n_ramps=1, stochastic=False):
    torch.manual_seed(seed + 912341)
    env = PhysicsEnv(seed=seed, scenario=scenario, size=size, n_boxes=n_boxes, n_ramps=n_ramps,
                     disable_tools=mode == 'no-tools', immovable=mode == 'fixed-props')
    observations = env.observe()
    memories = [torch.zeros(1, model.hidden_size) for model in models]
    frames = [env.trace()] if trace else None
    with torch.no_grad():
        for step in range(env.prep + env.play):
            actions = np.zeros((2, 5), np.float32)
            for role, model in enumerate(models):
                baseline = mode == 'reactive' or (mode == 'hider-v-reactive' and role == 1) or (mode == 'seeker-v-reactive' and role == 0)
                if baseline:
                    actions[role] = reactive_action(observations[role], role, step)
                else:
                    prediction, _, _, _, memories[role] = model.act(torch.from_numpy(observations[role][None]), memories[role], deterministic=not stochastic)
                    actions[role] = prediction[0].numpy()
                if mode == 'no-hider-tools' and role == 0 or mode == 'no-seeker-tools' and role == 1:
                    actions[role, 3:] = 0
            observations, _, done, info = env.step(actions)
            if trace:
                frames.append(env.trace())
            if done:
                break
    play = max(1, info['play_steps'])
    result = {'seed': seed, 'scenario': scenario, 'mode': mode, 'stochastic': stochastic,
              'size': size, 'boxes': n_boxes, 'ramps': n_ramps,
              'hiddenFraction': info['hidden'] / play, 'propShieldedFraction': info['shielded'] / play,
              'grabs': info['grabs'], 'locks': info['locks'], 'path': info['path'],
              'propDisplacement': sum(info['object_displacement'])}
    if trace:
        result['arena'] = env.arena
        result['frames'] = frames
    env.close()
    return result


def paired_intervals(records, modes, seed=94271):
    """Seed-paired bootstrap intervals; individual episodes are the resampling unit."""
    generator = np.random.default_rng(seed)
    baseline = {(r['seed'], r['scenario']): r for r in records if r['mode'] == 'learned'}
    intervals = {}
    for mode in modes:
        if mode == 'learned':
            continue
        pairs = [r['hiddenFraction'] - baseline[(r['seed'], r['scenario'])]['hiddenFraction']
                 for r in records if r['mode'] == mode]
        if not pairs:
            continue
        values = np.asarray(pairs)
        resampled = generator.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
        intervals[mode] = {'episodes': len(values), 'differenceFromLearned': float(values.mean()),
                           'bootstrap95Percent': np.quantile(resampled, [.025, .975]).tolist()}
    return intervals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--episodes-per-scenario', type=int, default=12)
    # Training resets use seeds below 2**30. This range cannot be sampled there.
    p.add_argument('--seed', type=int, default=1900000000)
    p.add_argument('--modes', default='learned,no-tools,fixed-props,reactive,hider-v-reactive,seeker-v-reactive')
    p.add_argument('--initial-checkpoint', help='Adds one trained-role-versus-original-role comparison for each role')
    p.add_argument('--output', default='training/physical-evaluation.json')
    p.add_argument('--trace-dir')
    p.add_argument('--stochastic', action='store_true', help='Sample the learned PPO action distribution with a fixed episode seed')
    p.add_argument('--size', type=float, default=8)
    p.add_argument('--boxes', type=int, default=3)
    p.add_argument('--ramps', type=int, default=1)
    args = p.parse_args()
    models, saved = load_models(args.checkpoint)
    modes = args.modes.split(',')
    compared = {}
    if args.initial_checkpoint:
        initial, original = load_models(args.initial_checkpoint)
        if original.get('decisions', 0) != 0:
            raise ValueError('The initial comparison must contain zero training decisions')
        compared = {'initial-pair': initial,
                    'trained-hider-v-initial': [models[0], initial[1]],
                    'trained-seeker-v-initial': [initial[0], models[1]]}
        modes.extend(compared)
    records = []
    for scenario_index, scenario in enumerate(['shelter', 'rooms', 'open']):
        for index in range(args.episodes_per_scenario):
            seed = args.seed + scenario_index * 1000 + index
            for mode in modes:
                record = episode(compared.get(mode, models), seed, scenario, mode, size=args.size,
                                 n_boxes=args.boxes, n_ramps=args.ramps, stochastic=args.stochastic)
                records.append(record)
        subset = [r for r in records if r['scenario'] == scenario]
        print(json.dumps({'scenario': scenario, 'means': {mode: round(float(np.mean([r['hiddenFraction'] for r in subset if r['mode'] == mode])), 4) for mode in modes}}), flush=True)
    summary = {}
    for mode in modes:
        group = [r for r in records if r['mode'] == mode]
        summary[mode] = {key: float(np.mean([r[key] for r in group])) for key in ['hiddenFraction', 'propShieldedFraction', 'propDisplacement']}
        summary[mode]['hiderGrabs'] = float(np.mean([r['grabs'][0] for r in group]))
        summary[mode]['seekerGrabs'] = float(np.mean([r['grabs'][1] for r in group]))
    report = {'format': 'original-physical-hide-seek-evaluation-v1',
              'checkpointSHA256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'trainingDecisions': saved.get('decisions'), 'heldOutSeedStart': args.seed,
              'stochastic': args.stochastic,
              'episodesPerScenario': args.episodes_per_scenario, 'summary': summary, 'episodes': records,
              'pairedHiddenFractionDifferences': paired_intervals(records, modes)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    if args.trace_dir:
        destination = Path(args.trace_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for scenario_index, scenario in enumerate(['shelter', 'rooms', 'open']):
            result = episode(models, args.seed + scenario_index * 1000, scenario, trace=True,
                             size=args.size, n_boxes=args.boxes, n_ramps=args.ramps, stochastic=args.stochastic)
            (destination / f'{scenario}.json').write_text(json.dumps(result, separators=(',', ':'),
                default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
