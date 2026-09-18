"""Isolated recurrent PPO helpers for current actors versus frozen own opponents."""
import numpy as np
import torch
from torch.distributions import Normal, Categorical
from persistent_actor import augment, advance_buttons
from central_persistent_train import normalize_active_advantages, policy_objective


def assign_roles(generator, count, arm, history_count):
    """-1 means current actor; other entries identify frozen own checkpoints."""
    draws = generator.random(count)
    historical = generator.integers(history_count, size=count)
    roles = np.full((count, 2), -1, np.int64)
    if arm == 'B':
        roles[(draws >= .5) & (draws < .75), 1] = historical[(draws >= .5) & (draws < .75)]
        roles[draws >= .75, 0] = historical[draws >= .75]
    elif arm != 'A':
        raise ValueError('The matched comparison has arms A and B only')
    return roles


def active_masks(physical, roles):
    current = roles == -1
    active = current.copy()
    active[:, 1] &= physical[:, 1, 5] >= 1
    return current, active


@torch.no_grad()
def act_grouped(models, histories, physical, memories, buttons, roles, sample=True):
    observations = augment(physical, buttons)
    count = len(physical)
    actions = np.zeros((count, 2, 5), np.float32)
    next_memories = [memory.clone() for memory in memories]
    next_buttons = buttons.copy()
    records = []
    for role in range(2):
        observed = torch.from_numpy(observations[:, role].copy())
        raw = torch.zeros(count, 5)
        logp = torch.zeros(count)
        values = torch.zeros(count)
        for identity in np.unique(roles[:, role]):
            indices = np.flatnonzero(roles[:, role] == identity)
            actor = models[role] if identity == -1 else histories[identity][role]
            if sample:
                movement, commands, raw_group, logp_group, value, memory = actor.act(
                    observed[indices], memories[role][indices, :actor.hidden_size])
                blind = physical[indices, role, 5] < 1 if role else np.zeros(len(indices), bool)
                next_buttons[indices, role] = advance_buttons(buttons[indices, role], commands.numpy(), blind)
                actions[indices, role, :3] = movement.numpy()
                actions[indices, role, 3:] = next_buttons[indices, role]
                actions[indices[blind], role] = 0
                raw[indices] = raw_group
                logp[indices] = logp_group
            else:
                _, _, value, memory = actor(observed[indices], memories[role][indices, :actor.hidden_size])
            values[indices] = value
            next_memories[role][indices] = 0
            next_memories[role][indices, :actor.hidden_size] = memory
        records.append((observed, raw, logp, values))
    return actions, next_memories, next_buttons, records


def actor_update(model, optimizer, batch, advantages, current_rows, role,
                 epochs=2, sequence_length=32, sequence_batch=256,
                 entropy_weight=.005, kl_limit=.008):
    observations, memories, starts, raw_actions, old_logps = batch
    if observations.shape[-1] != 140 or current_rows.shape != advantages.shape:
        raise ValueError('Restricted actor observations and exact per-row policy identity required')
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
    for _ in range(epochs):
        ordering = torch.randperm(len(sequences)).tolist()
        epoch_kls, epoch_sizes = [], []
        for offset in range(0, len(ordering), sequence_batch):
            selected = [sequences[index] for index in ordering[offset:offset + sequence_batch]]
            begin = torch.tensor([entry[0] for entry in selected])
            env = torch.tensor([entry[1] for entry in selected])
            if not any(bool(active[begin + inner, env].any()) for inner in range(sequence_length)):
                continue
            memory = memories[begin, env].detach()
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
        optimizerSteps=optimizer_steps, activeSamples=int(active.sum()))
