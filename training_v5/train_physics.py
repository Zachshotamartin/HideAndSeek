"""Recurrent PPO training from random weights in the original MuJoCo arena.

Runs on CPU, saves optimizer/recurrent-training state and an auditable log.
Physical conditions vary between episodes; neither actor receives expert moves.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from actor import PhysicalActor
from env_pool import PhysicsEnvPool

torch.set_num_threads(1)


def recurrent_update(model, optimizer, batch, bootstrap, epochs=3, sequence_length=16,
                     entropy_weight=.012, sequence_batch=32, role=0):
    observations, memories, starts, actions, old_logps, rewards, values, dones = batch
    steps, count = rewards.shape
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros(count)
    for step in reversed(range(steps)):
        following = bootstrap if step == steps - 1 else values[step + 1]
        live = 1 - dones[step]
        delta = rewards[step] + .998 * following * live - values[step]
        accumulator = delta + .998 * .98 * live * accumulator
        advantages[step] = accumulator
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    sequences = [(begin, environment) for begin in range(0, steps, sequence_length) for environment in range(count)]
    losses, divergences = [], []
    for _ in range(epochs):
        ordering = torch.randperm(len(sequences)).tolist()
        for offset in range(0, len(ordering), sequence_batch):
            selected = [sequences[index] for index in ordering[offset:offset + sequence_batch]]
            begin = torch.tensor([item[0] for item in selected])
            env = torch.tensor([item[1] for item in selected])
            memory = memories[begin, env].detach()
            total_loss = 0
            kl_sum = 0
            for inner in range(sequence_length):
                step = begin + inner
                memory = memory * (1 - starts[step, env, None])
                normal, tools, value, memory = model(observations[step, env], memory)
                logp, entropy = model.statistics(normal, tools, actions[step, env])
                log_ratio = logp - old_logps[step, env]
                ratio = log_ratio.exp()
                # The seeker cannot act during preparation. Its memory and
                # critic still learn, but discarded actions carry no policy loss.
                active = (observations[step, env, 5] >= 1).float() if role == 1 else torch.ones_like(ratio)
                actor_loss = -(torch.minimum(ratio * advantages[step, env],
                    ratio.clamp(.8, 1.2) * advantages[step, env]) * active).mean()
                # Value clipping prevents an isolated long episode from destabilizing a batch.
                clipped_value = values[step, env] + (value - values[step, env]).clamp(-.2, .2)
                critic_loss = torch.maximum((value - returns[step, env]).square(),
                    (clipped_value - returns[step, env]).square()).mean()
                total_loss += actor_loss + .5 * critic_loss - entropy_weight * (entropy * active).mean()
                kl_sum += (((ratio - 1) - log_ratio) * active).mean().detach()
            total_loss /= sequence_length
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            losses.append(float(total_loss.detach()))
            divergences.append(float(kl_sum / sequence_length))
        if np.mean(divergences[-max(1, (len(ordering) + sequence_batch - 1) // sequence_batch):]) > .025:
            break
    return {'loss': float(np.mean(losses)), 'approximateKL': float(np.mean(divergences))}


def scenario_for(index):
    return ['shelter', 'rooms', 'open'][index % 3]


def sample_environment(generator, index, fixed=False):
    if fixed:
        return dict(scenario=scenario_for(index), size=8, n_boxes=3, n_ramps=1)
    # Domain randomization changes physical conditions, never rewards or actions.
    return dict(scenario=scenario_for(index),
                size=float(generator.choice([6, 7, 8, 9, 10, 12])),
                n_boxes=int(generator.integers(0, 9)),
                n_ramps=int(generator.integers(0, 3)))


def train(args):
    assert args.horizon % args.sequence_length == 0
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configs = [dict(seed=args.seed + i, **sample_environment(generator, i, args.fixed_arena))
               for i in range(args.envs)]
    with PhysicsEnvPool(configs, workers=args.workers) as pool:
        run_training(args, pool, output, generator)


def run_training(args, pool, output, generator):
    obs = pool.observations.copy()
    if obs.ndim != 3 or obs.shape[1] != 2:
        raise ValueError(f'Expected [environments, 2, observations], got {obs.shape}')
    observation_size = obs.shape[-1]
    models = [PhysicalActor(observation_size), PhysicalActor(observation_size)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5) for model in models]
    first_update, decisions, episodes, elapsed_before = 0, 0, 0, 0
    log = []
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['observationSize'] != observation_size:
            raise ValueError('Checkpoint observation schema differs from this environment')
        for role in range(2):
            models[role].load_state_dict(saved['models'][role])
            optimizers[role].load_state_dict(saved['optimizers'][role])
        first_update, decisions, episodes = saved['updates'], saved['decisions'], saved['episodes']
        elapsed_before, log = saved['seconds'], saved['log']
        torch.set_rng_state(saved['torchRNG'])
        generator.bit_generator.state = saved['numpyRNG']
    else:
        torch.save({'models': [m.state_dict() for m in models], 'observationSize': observation_size,
                    'seed': args.seed}, output / 'initial.pt')
    memories = [torch.zeros(args.envs, model.hidden_size) for model in models]
    starts = torch.ones(args.envs)
    recent = []
    start_time = time.monotonic()
    for update in range(first_update, first_update + args.updates):
        buffers = []
        for model in models:
            h, n = args.horizon, args.envs
            buffers.append([torch.zeros(h, n, observation_size), torch.zeros(h, n, model.hidden_size),
                torch.zeros(h, n), torch.zeros(h, n, 5)] + [torch.zeros(h, n) for _ in range(4)])
        reward_terms = {}
        for step in range(args.horizon):
            env_actions = np.zeros((args.envs, 2, 5), dtype=np.float32)
            for role, model in enumerate(models):
                memories[role] *= (1 - starts[:, None])
                tensor = torch.from_numpy(obs[:, role].copy())
                with torch.no_grad():
                    action, raw, logp, value, next_memory = model.act(tensor, memories[role])
                b = buffers[role]
                b[0][step], b[1][step], b[2][step] = tensor, memories[role], starts
                b[3][step], b[4][step], b[6][step] = raw, logp, value
                memories[role] = next_memory
                env_actions[:, role] = action.numpy()
            new_starts = np.zeros(args.envs, dtype=np.float32)
            next_observations, environment_rewards, terminals, infos = pool.step(env_actions)
            finished, reset_seeds, reset_configs = [], [], []
            for index in range(args.envs):
                next_obs, rewards, done, info = (next_observations[index], environment_rewards[index],
                                                terminals[index], infos[index])
                # Constant scaling improves critic conditioning without changing
                # the objective. There is no movement, tool-use or strategy bonus.
                unscaled_rewards = np.asarray(rewards)
                rewards = unscaled_rewards * .05
                for role in range(2):
                    key = ['hiderVisibility', 'seekerVisibility'][role]
                    reward_terms[key] = reward_terms.get(key, 0) + float(unscaled_rewards[role])
                for role in range(2):
                    buffers[role][5][step, index] = float(rewards[role])
                    buffers[role][7][step, index] = float(done)
                if not np.isfinite(next_obs).all() or not np.isfinite(rewards).all():
                    raise FloatingPointError(f'Nonfinite simulator output, update {update}, env {index}')
                if done:
                    summary = {k: v for k, v in info.items() if isinstance(v, (int, float, bool))}
                    summary['hiddenFraction'] = info.get('hidden', 0) / max(1, info.get('play_steps', 1))
                    summary['propShieldedFraction'] = info.get('shielded', 0) / max(1, info.get('play_steps', 1))
                    for key in ('grabs', 'locks', 'path', 'collisions', 'returns'):
                        for role, value in enumerate(info.get(key, [])):
                            summary[f'{["hider", "seeker"][role]}_{key}'] = value
                    summary['propDisplacement'] = sum(info.get('object_displacement', []))
                    recent.append(summary)
                    recent = recent[-128:]
                    episodes += 1
                    finished.append(index)
                    reset_seeds.append(int(generator.integers(1, 2**30)))
                    reset_configs.append(sample_environment(generator, index, args.fixed_arena))
                    new_starts[index] = 1
                obs[index] = next_obs
            if finished:
                obs[finished] = pool.reset_at(finished, reset_seeds, reset_configs)
            starts = torch.from_numpy(new_starts)
        update_results = []
        for role, model in enumerate(models):
            with torch.no_grad():
                bootstrap = model(torch.from_numpy(obs[:, role].copy()),
                                  memories[role] * (1 - starts[:, None]))[2]
            update_results.append(recurrent_update(model, optimizers[role], buffers[role], bootstrap,
                epochs=args.epochs, sequence_length=args.sequence_length, entropy_weight=args.entropy,
                sequence_batch=args.sequence_batch, role=role))
        decisions += args.horizon * args.envs * 2
        if update % args.log_every == 0 or update == first_update + args.updates - 1:
            row = {'update': update + 1, 'decisions': decisions, 'episodes': episodes,
                   'seconds': round(elapsed_before + time.monotonic() - start_time, 2),
                   'hider': update_results[0], 'seeker': update_results[1],
                   'episodeMeans': {key: float(np.mean([r[key] for r in recent if key in r]))
                                    for key in sorted(set().union(*(r.keys() for r in recent)))},
                   'rewardTerms': {k: v / (args.horizon * args.envs) for k, v in reward_terms.items()}}
            log.append(row)
            print(json.dumps(row), flush=True)
        if (update + 1) % args.save_every == 0 or update == first_update + args.updates - 1:
            saved = {'format': 'original-mujoco-recurrent-ppo-v1', 'observationSize': observation_size,
                'models': [m.state_dict() for m in models], 'optimizers': [o.state_dict() for o in optimizers],
                'updates': update + 1, 'decisions': decisions, 'episodes': episodes,
                'seconds': elapsed_before + time.monotonic() - start_time, 'log': log, 'seed': args.seed,
                'torchRNG': torch.get_rng_state(), 'numpyRNG': generator.bit_generator.state,
                'arguments': vars(args),
                'physicsSHA256': __import__('hashlib').sha256(Path(__file__).with_name('physics.py').read_bytes()).hexdigest(),
                'rewardDescription': 'Only the zero-sum hide/seek visibility outcome, scaled uniformly by .05 for critic conditioning. No shaped rewards, expert actions, scripted opponents or tool-use bonuses.'}
            temporary = output / 'latest.tmp'
            torch.save(saved, temporary)
            os.replace(temporary, output / 'latest.pt')
            (output / 'training-log.json').write_text(json.dumps(log, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--updates', type=int, default=500)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--fixed-arena', action='store_true', help='Use the early fixed-size/count development distribution')
    p.add_argument('--envs', type=int, default=16)
    p.add_argument('--horizon', type=int, default=64)
    p.add_argument('--sequence-length', type=int, default=16)
    p.add_argument('--sequence-batch', type=int, default=32)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--entropy', type=float, default=.005)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=90210)
    p.add_argument('--output', default='training/runs/physical-ppo')
    p.add_argument('--resume')
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--save-every', type=int, default=20)
    train(p.parse_args())
