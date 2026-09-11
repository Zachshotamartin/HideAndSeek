"""Read-only reward/entropy and value diagnostics for frozen original actors.

No optimizer is constructed. Raw diagnostic rollouts are saved for reuse. These
gradients describe the unclipped PPO objective at ratio1, not a new model update.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from central_critic import batch_states
from central_persistent_train import advantages_and_returns, normalize_active_advantages
from env_pool import PhysicsEnvPool
from fit_central_value import GAMMA, REWARD_SCALE
from league_ppo import act_grouped
from persistent_train import environment_config, file_hash
from residual_critic import ResidualCentralCritic
from train_league import load_actors

torch.set_num_threads(1)


def collect(models, critic, args):
    generator = np.random.default_rng(773993)
    configs = [dict(seed=args.seed + index, **environment_config(generator, index))
               for index in range(args.envs)]
    steps, count = 256, args.envs
    buffers = [[torch.zeros(steps, count, 140), torch.zeros(steps, count, 64),
                torch.zeros(steps, count), torch.zeros(steps, count, 5),
                torch.zeros(steps, count)] for _ in range(2)]
    values = torch.zeros(steps, count)
    partial = torch.zeros(steps, count, 2)
    rewards = torch.zeros(steps, count)
    dones = torch.zeros(steps, count)
    with PhysicsEnvPool(configs, workers=args.workers, with_central_state=True) as pool:
        physical = pool.observations.copy()
        memories = [torch.zeros(count, 64), torch.zeros(count, 64)]
        buttons = np.zeros((count, 2, 2), np.float32)
        roles = np.full((count, 2), -1, np.int64)
        starts = torch.ones(count)
        with torch.no_grad():
            for tick in range(steps):
                memories = [memory * (1 - starts[:, None]) for memory in memories]
                before = torch.stack(memories, dim=1)
                actions, next_memories, next_buttons, records = act_grouped(
                    models, [], physical, memories, buttons, roles)
                partial[tick] = torch.stack([record[3] for record in records], dim=-1)
                values[tick] = critic(batch_states(pool.central_states), before, partial[tick])
                for role in range(2):
                    observed, raw, logp, _ = records[role]
                    for destination, value in zip(buffers[role], [observed, memories[role], starts, raw, logp]):
                        destination[tick] = value
                physical, returned, done, _ = pool.step(actions)
                rewards[tick] = torch.from_numpy(returned[:, 0].copy()) * REWARD_SCALE
                dones[tick] = torch.from_numpy(done.astype(np.float32))
                memories, buttons = next_memories, next_buttons
                finished = np.flatnonzero(done).tolist()
                if finished:
                    physical[finished] = pool.reset_at(finished, [args.seed + 10000 + index for index in finished],
                        [environment_config(generator, index) for index in finished])
                    buttons[finished] = 0
                starts = torch.from_numpy(done.astype(np.float32))
            memories = [memory * (1 - starts[:, None]) for memory in memories]
            _, _, _, records = act_grouped(models, [], physical, memories, buttons, roles, sample=False)
            last_partial = torch.stack([record[3] for record in records], dim=-1)
            bootstrap = critic(batch_states(pool.central_states), torch.stack(memories, dim=1), last_partial)
    advantages, returns = advantages_and_returns(rewards, values, dones, bootstrap)
    return dict(buffers=buffers, values=values, partial=partial, rewards=rewards,
                dones=dones, advantages=advantages, returns=returns, configs=configs,
                diagnosticActorDecisions=steps * count * 2)


def calibration(data):
    # First240 ticks form complete independently initialized episodes. Exclude
    # the16-tick prefix following their resets from Monte Carlo calibration.
    if not (data['dones'][239] == 1).all() or data['dones'][:239].any():
        raise ValueError('Calibration requires complete fixed-horizon episodes')
    target = torch.zeros_like(data['values'][:240])
    future = torch.zeros(target.shape[1])
    for tick in reversed(range(240)):
        future = data['rewards'][tick] + GAMMA * future
        target[tick] = future
    baseline = .5 * (data['partial'][:240, :, 0] - data['partial'][:240, :, 1])
    result = {}
    for name, selected in [('complete', slice(None)), ('preparation', slice(0, 80)), ('play', slice(80, 240))]:
        result[name] = {label: float(((prediction[selected] - target[selected]) ** 2).mean())
                       for label, prediction in [('centralMSE', data['values'][:240]), ('partialPairMSE', baseline)]}
    return result


def gradients(model, batch, advantages, role):
    observations, memories, starts, raw, old_logps = batch
    active = torch.ones_like(advantages, dtype=torch.bool)
    if role:
        active &= observations[:, :, 5] >= 1
    advantages = normalize_active_advantages(advantages.detach(), active)
    total_active = int(active.sum())
    named = list(model.named_parameters())
    parameters = [parameter for _, parameter in named]
    totals = {name: [torch.zeros_like(parameter) for parameter in parameters]
              for name in ['toolReward', 'movementReward', 'toolEntropy', 'movementEntropy']}
    probability_sums = {label: torch.zeros(2, 3) for label in ['all', 'holding', 'notHolding', 'preparation', 'play']}
    counts = {label: 0 for label in probability_sums}
    entropy_sums = torch.zeros(2)
    reward_tool_chunks = []
    maximum_logp_error = 0.0
    sequence_length = 32
    sequences = [(begin, env) for begin in range(0, len(active), sequence_length)
                 for env in range(active.shape[1])]
    ordering = torch.randperm(len(sequences)).tolist()
    for offset in range(0, len(sequences), 256):
        selected = [sequences[index] for index in ordering[offset:offset + 256]]
        begin = torch.tensor([item[0] for item in selected])
        env = torch.tensor([item[1] for item in selected])
        memory = memories[begin, env].detach()
        losses = {name: 0 for name in totals}
        for inner in range(sequence_length):
            tick = begin + inner
            memory = memory * (1 - starts[tick, env, None])
            normal, tools, _, memory = model(observations[tick, env], memory)
            action = raw[tick, env]
            jacobian = lambda value: 2 * (math.log(2) - value - torch.nn.functional.softplus(-2 * value))
            movement_logp = (normal.log_prob(action[:, :3]) - jacobian(action[:, :3])).sum(-1)
            tool_logp = tools.log_prob(action[:, 3:].long()).sum(-1)
            active_rows = active[tick, env]
            error = (movement_logp + tool_logp - old_logps[tick, env]).abs()
            maximum_logp_error = max(maximum_logp_error, float(error.detach().max()))
            advantage = advantages[tick, env]
            losses['toolReward'] -= (tool_logp[active_rows] * advantage[active_rows]).sum() / total_active
            losses['movementReward'] -= (movement_logp[active_rows] * advantage[active_rows]).sum() / total_active
            losses['toolEntropy'] -= .005 * tools.entropy().sum(-1)[active_rows].sum() / total_active
            movement_entropy = (normal.entropy() + jacobian(normal.rsample())).sum(-1)
            losses['movementEntropy'] -= .005 * movement_entropy[active_rows].sum() / total_active
            with torch.no_grad():
                holding = observations[tick, env, 8] > .5
                preparation = observations[tick, env, 5] < 1
                masks = dict(all=active_rows, holding=active_rows & holding,
                             notHolding=active_rows & ~holding,
                             preparation=active_rows & preparation, play=active_rows & ~preparation)
                for label, mask in masks.items():
                    probability_sums[label] += tools.probs[mask].sum(0)
                    counts[label] += int(mask.sum())
                entropy_sums += tools.entropy()[active_rows].sum(0)
        for index, (name, loss) in enumerate(losses.items()):
            parts = torch.autograd.grad(loss, parameters, retain_graph=index < 3, allow_unused=True)
            for accumulated, part in zip(totals[name], parts):
                if part is not None:
                    accumulated += part.detach()
            if name == 'toolReward':
                reward_tool_chunks.append(torch.cat([part.detach().flatten() for (key, _), part in zip(named, parts)
                                                       if key.startswith('tools.') and part is not None]))
    groups = {'toolHead': lambda name: name.startswith('tools.'),
              'movementHead': lambda name: name.startswith('movement.') or name == 'log_std',
              'sharedEncoderMemory': lambda name: name.startswith(('encoder.', 'memory.'))}
    norms = {}
    for label, predicate in groups.items():
        vectors = {component: torch.cat([gradient.flatten() for (name, _), gradient in zip(named, parts) if predicate(name)])
                   for component, parts in totals.items()}
        policy = vectors['toolReward'] + vectors['movementReward']
        entropy = vectors['toolEntropy'] + vectors['movementEntropy']
        cosine = lambda first, second: float(torch.dot(first, second) / (first.norm() * second.norm() + 1e-20))
        norms[label] = {name: float(vector.norm()) for name, vector in vectors.items()}
        norms[label].update(totalRewardNorm=float(policy.norm()), totalEntropyNorm=float(entropy.norm()),
                            entropyToRewardNorm=float(entropy.norm() / (policy.norm() + 1e-20)),
                            rewardEntropyCosine=cosine(policy, entropy))
    return dict(activeSamples=total_active, toolProbabilities={label: (probability_sums[label] / counts[label]).tolist()
                if counts[label] else None for label in counts}, conditionCounts=counts,
                meanToolEntropyNats=(entropy_sums / total_active).tolist(), gradients=norms,
                toolRewardChunkCosine=float(torch.nn.functional.cosine_similarity(*reward_tool_chunks, dim=0))
                if len(reward_tool_chunks) == 2 else None,
                maximumRecomputedLogpError=maximum_logp_error)


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    models = load_actors(saved)
    before = [copy.deepcopy(model.state_dict()) for model in models]
    fitted = torch.load(args.critic_fit, map_location='cpu', weights_only=False)
    critic = ResidualCentralCritic(fitted['dropout']).eval()
    critic.load_state_dict(saved.get('centralCritic', fitted['critic']))
    torch.manual_seed(913833)
    cache = output / 'rollout.pt'
    if cache.exists():
        data = torch.load(cache, map_location='cpu', weights_only=False)
        if (data['checkpointSHA256'] != file_hash(args.checkpoint)
                or data['criticFitSHA256'] != file_hash(args.critic_fit)
                or data['physicsSHA256'] != file_hash(Path(__file__).with_name('physics.py'))):
            raise ValueError('Do not reuse diagnostic buffers for different actors, critic, or physics')
    else:
        data = collect(models, critic, args)
        data['checkpointSHA256'] = file_hash(args.checkpoint)
        data['criticFitSHA256'] = file_hash(args.critic_fit)
        data['physicsSHA256'] = file_hash(Path(__file__).with_name('physics.py'))
        torch.save(data, cache)
    torch.manual_seed(329911)
    roles = [gradients(model, data['buffers'][role], data['advantages'] if role == 0 else -data['advantages'], role)
             for role, model in enumerate(models)]
    for old, model in zip(before, models):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(old[name], value, atol=0, rtol=0)
    result = dict(checkpointSHA256=data['checkpointSHA256'],
                  diagnosticActorDecisions=data['diagnosticActorDecisions'], optimizerSteps=0,
                  scope=f'{len(data["configs"])} seeded varied-size/count current/current diagnostic games plus16 reset ticks; no performance-selection claim.',
                  interpretation='Unclipped PPO gradient at ratio1 before Adam or clipping, averaged across diagnostic recurrent chunks. Entropy coefficient .005. Not a causal test of a new optimization setting.',
                  roleOrder=['hider', 'seeker'], commandOrder=['Keep', 'Press', 'Release'],
                  criticCalibration=calibration(data), roles=roles,
                  sources={name: file_hash(Path(__file__).with_name(name)) for name in
                           ['diagnose_league_gradients.py', 'league_ppo.py', 'central_persistent_train.py',
                            'persistent_actor.py', 'central_critic.py', 'residual_critic.py', 'physics.py']})
    (output / 'diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ['checkpoint', 'critic-fit', 'output']:
        parser.add_argument('--' + argument, required=True)
    parser.add_argument('--envs', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1500130000)
    main(parser.parse_args())
