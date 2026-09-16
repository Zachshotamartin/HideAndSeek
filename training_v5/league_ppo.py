"""Isolated recurrent PPO helpers for current actors versus frozen own opponents."""
import numpy as np
import torch
from torch.distributions import Normal, Categorical
from persistent_actor import advance_buttons
from central_persistent_train import normalize_active_advantages, policy_objective

NOISE = 4   # observed exploration-noise values carried per actor between steps


def assign_roles(generator, count, arm, history_count):
    """-1 means current actor; other entries identify frozen own checkpoints."""
    draws = generator.random(count)
    historical = generator.integers(history_count, size=count)
    roles = np.full((count, 2), -1, np.int64)
    if arm == 'B':
        roles[(draws >= .8) & (draws < .9), 1] = historical[(draws >= .8) & (draws < .9)]
        roles[draws >= .9, 0] = historical[draws >= .9]
    elif arm != 'A':
        raise ValueError('The matched comparison has arms A and B only')
    return roles


def active_masks(physical, roles):
    current = roles == -1
    active = current.copy()
    active[:, 1] &= physical[:, 1, 5] >= 1
    return current, active


def actor_input(actor, physical, buttons, noises):
    """One actor's own observation from the worker rows: its physical prefix, buttons and any carried noise."""
    return torch.from_numpy(actor.observe(physical, buttons, noises).copy())


@torch.no_grad()
def act_grouped(models, histories, physical, memories, buttons, roles, sample=True, noises=None):
    """Act for every environment with the current or frozen actor each role slot names.

    ``physical`` may be wider than an actor's schema; every actor takes the
    prefix it was trained on. ``noises`` holds the exploration noise each role
    carried out of the previous step; the returned array carries it forward.
    The recorded observation of each role is the current actor's input for all
    rows; rows played by frozen actors are masked out of every update.
    """
    count = len(physical)
    if noises is None:
        noises = np.zeros((count, 2, NOISE), np.float32)
    actions = np.zeros((count, 2, 6), np.float32)
    next_memories = [memory.clone() for memory in memories]
    next_buttons = buttons.copy()
    next_noises = noises.copy()
    records = []
    for role in range(2):
        observed = actor_input(models[role], physical[:, role], buttons[:, role], noises[:, role])
        raw = torch.zeros(count, 6)
        logp = torch.zeros(count)
        values = torch.zeros(count)
        for identity in np.unique(roles[:, role]):
            indices = np.flatnonzero(roles[:, role] == identity)
            actor = models[role] if identity == -1 else histories[identity][role]
            inputs = observed[indices] if identity == -1 else actor_input(
                actor, physical[indices, role], buttons[indices, role], noises[indices, role])
            memory_in = memories[role][indices, :actor.hidden_size]
            if sample:
                movement, commands, raw_group, logp_group, value, memory, noise = actor.act_with_noise(inputs, memory_in)
                blind = physical[indices, role, 5] < 1 if role else np.zeros(len(indices), bool)
                next_buttons[indices, role] = advance_buttons(buttons[indices, role], commands.numpy(), blind)
                actions[indices, role, :3] = movement.numpy()[:, :3]
                actions[indices, role, 5] = movement.numpy()[:, 3]
                actions[indices, role, 3:5] = next_buttons[indices, role]
                actions[indices[blind], role] = 0
                raw[indices] = raw_group
                logp[indices] = logp_group
                next_noises[indices, role] = noise.numpy()
            else:
                _, _, value, memory = actor(inputs, memory_in)
            values[indices] = value
            next_memories[role][indices] = 0
            next_memories[role][indices, :actor.hidden_size] = memory
        records.append((observed, raw, logp, values))
    return actions, next_memories, next_buttons, next_noises, records


def actor_update(model, optimizer, batch, advantages, current_rows, role,
                 epochs=2, sequence_length=32, sequence_batch=256,
                 entropy_weight=.005, kl_limit=.008, burn_in=0, prefix=None):
    """Recurrent PPO update over complete sequences with real burn-in.

    `prefix` holds the last `burn_in` steps of the previous rollout
    (observations, stored memories, episode starts) so that the first sequence
    of a horizon is also warmed up through the current parameters instead of
    reusing a stale stored state. Without a prefix only sequences that begin
    inside the horizon get burn-in. Burn-in never contributes gradients.
    """
    observations, memories, starts, raw_actions, old_logps = batch
    if observations.shape[-1] != model.observation_size or current_rows.shape != advantages.shape:
        raise ValueError('Actor observations of this schema and exact per-row policy identity required')
    if prefix is not None:
        prefix_observations, prefix_memories, prefix_starts = prefix
        if burn_in <= 0 or prefix_observations.shape != (burn_in, *observations.shape[1:]) \
                or prefix_memories.shape != (burn_in, *memories.shape[1:]) or prefix_starts.shape != (burn_in, *starts.shape[1:]):
            raise ValueError('The burn-in prefix must hold exactly burn_in steps for every environment')
        extended_observations = torch.cat((prefix_observations, observations))
        extended_memories = torch.cat((prefix_memories, memories))
        extended_starts = torch.cat((prefix_starts, starts))
        offset = burn_in
    else:
        extended_observations, extended_memories, extended_starts, offset = observations, memories, starts, 0
    active = current_rows.bool().clone()
    if role == 1:
        active &= observations[:, :, 5] >= 1
    advantages = normalize_active_advantages(advantages.detach(), active)
    steps, count = active.shape
    if steps % sequence_length:
        raise ValueError('Recurrent horizon must contain complete sequences')
    sequences = [(begin, env) for begin in range(0, steps, sequence_length) for env in range(count)]
    losses, divergences, sizes = [], [], []
    optimizer_steps = 0
    burn_in_steps = 0
    for _ in range(epochs):
        ordering = torch.randperm(len(sequences)).tolist()
        epoch_kls, epoch_sizes = [], []
        for start in range(0, len(ordering), sequence_batch):
            selected = [sequences[index] for index in ordering[start:start + sequence_batch]]
            begin = torch.tensor([entry[0] for entry in selected])
            env = torch.tensor([entry[1] for entry in selected])
            if not any(bool(active[begin + inner, env].any()) for inner in range(sequence_length)):
                continue
            extended_begin = begin + offset
            warm_start = (extended_begin - burn_in).clamp(min=0)
            memory = extended_memories[warm_start, env].detach().clone()
            with torch.no_grad():
                for warm in range(burn_in):
                    warm_step = warm_start + warm
                    live = warm_step < extended_begin
                    if live.any():
                        ids = warm_step[live]; ee = env[live]
                        mm = memory[live] * (1 - extended_starts[ids, ee, None])
                        memory[live] = model(extended_observations[ids, ee], mm)[-1]
                        burn_in_steps += int(live.sum())
            loss = 0
            kl_sum = 0
            active_count = 0
            for inner in range(sequence_length):
                step = begin + inner
                memory = memory * (1 - starts[step, env, None])
                normal, tools, _, memory = model(observations[step, env], memory)
                selected_rows = active[step, env]
                if not selected_rows.any():
                    continue
                # Never evaluate importance ratios for an off-policy historical
                # action. Masking an infinite ratio afterward could produce NaN.
                active_normal = Normal(normal.loc[selected_rows], normal.scale[selected_rows])
                active_tools = Categorical(logits=tools.logits[selected_rows])
                logp, entropy = model.statistics(active_normal, active_tools, raw_actions[step, env][selected_rows])
                objective, kl, number = policy_objective(logp, old_logps[step, env][selected_rows],
                    entropy, advantages[step, env][selected_rows], torch.ones_like(logp, dtype=torch.bool), entropy_weight)
                loss += objective
                kl_sum += kl.detach()
                active_count += int(number)
            loss /= active_count
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach()))
            divergence = float(kl_sum / active_count)
            divergences.append(divergence)
            sizes.append(active_count)
            epoch_kls.append(divergence)
            epoch_sizes.append(active_count)
        if epoch_kls and np.average(epoch_kls, weights=epoch_sizes) > kl_limit:
            break
    return dict(loss=float(np.average(losses, weights=sizes)) if sizes else 0,
        approximateKL=float(np.average(divergences, weights=sizes)) if sizes else 0,
        optimizerSteps=optimizer_steps, activeSamples=int(active.sum()), burnInSteps=burn_in_steps)
