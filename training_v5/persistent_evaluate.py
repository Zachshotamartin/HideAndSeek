"""Read-only validation of the persistent-button pilot and fixed opponents.

All modes use role-specific common random tapes for the same maps. Disabling
tools changes the physical buttons, not the actor's requested controller state;
ordinary physical pushing remains enabled. No scoring or training is changed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from actor import PhysicalActor
from capture import resolve_capture
from game import NOISE, observe as game_observe
from motion_diagnostics import blocked_roles
from env_pool import unstable
from persistent_actor import FORMAT, PHYSICS_OBSERVATIONS, PersistentActor, advance_buttons
from physics import DT, PhysicsEnv

torch.set_num_threads(1)
BLACKOUT_STEPS = 24  # 1.9 s without the opponent block after the seeker's first sight
OPPONENT_BLOCK = slice(10, 18)
DEFAULT_ARENA = dict(size=8, n_boxes=3, n_ramps=1)
TAPE_LENGTH = 10
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 371473
SEED_OFFSETS = (912341, 2912341)   # role-specific random tapes derived from the map seed
WALL_GROUP = 1
PROP_GROUP = 3
DEFAULT_CONTRASTS = [
    ('Hider tools benefit', 'learned', 'no-hider-tools'),
    ('Seeker tools benefit', 'no-seeker-tools', 'learned'),
    ('Pilot hider learning vs fixed initial', 'trained-hider-v-initial', 'warm-hider-v-initial'),
    ('Pilot seeker learning vs fixed initial', 'warm-seeker-v-initial', 'trained-seeker-v-initial'),
    ('Total hider learning vs fixed initial', 'trained-hider-v-initial', 'initial-pair'),
    ('Total seeker learning vs fixed initial', 'initial-pair', 'trained-seeker-v-initial'),
    ('Pilot hider vs original parent at fixed initial', 'trained-hider-v-initial', 'parent-hider-v-initial'),
    ('Pilot seeker vs original parent at fixed initial', 'parent-seeker-v-initial', 'trained-seeker-v-initial')]
BINARY_CONTRASTS = [
    ('Pilot hider vs frozen trained binary seeker', 'pilot-hider-v-binary', 'binary-pair'),
    ('Pilot seeker vs frozen trained binary hider', 'binary-pair', 'pilot-seeker-v-binary')]


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def load(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    persistent = saved['format'] == FORMAT
    models = [PersistentActor() if persistent else PhysicalActor(208) for _ in range(2)]
    for model, state in zip(models, saved['models']):
        model.load_state_dict(state)
        model.eval()
    return models, saved


def is_blind(physical):
    """A seeker in the preparation phase sees nothing and may not act."""
    return physical[7] < .5 and physical[5] < 1


def gaussian(tape, axis):
    """Box-Muller sample from two uniform tape entries."""
    return math.sqrt(-2 * math.log(max(1e-12, tape[axis * 2]))) * math.cos(2 * math.pi * tape[axis * 2 + 1])


def movement_sample(model, normal, noise, tape, deterministic):
    """Raw movement from the tape, and the exploration noise carried into the next step.

    An actor that observes its noise already includes ``rho * noise`` in the
    mean it returns; the deterministic action is the head's own mean and
    carries no noise, exactly as the training actor does.
    """
    rho = float(getattr(model, 'noise_rho', 0.))
    mean = normal.mean[0].numpy().astype(np.float64)
    scale = normal.scale[0].numpy().astype(np.float64)
    base = mean - rho * np.asarray(noise, np.float64)[:len(mean)]
    if deterministic:
        return base, np.zeros(NOISE, np.float32)
    movement = np.array([mean[axis] + scale[axis] * gaussian(tape, axis) for axis in range(len(mean))])
    carried = np.zeros(NOISE, np.float32)
    if rho:
        carried[:len(mean)] = movement - base
    return movement, carried


def sample(model, physical, memory, buttons, generator, deterministic=False, noise=None):
    """One stochastic action from a seeded uniform tape; identical tapes across modes."""
    persistent = isinstance(model, PersistentActor)
    blind = is_blind(physical)
    if blind:
        buttons = np.zeros(2, np.float32)
    if noise is None:
        noise = np.zeros(NOISE, np.float32)
    physical = np.asarray(physical, np.float32)
    observation = model.observe(physical, buttons, noise) if persistent else physical[:PHYSICS_OBSERVATIONS]
    normal, tools, _, memory = model(torch.from_numpy(observation[None]), memory)
    tape = generator.random(TAPE_LENGTH)
    movement, next_noise = movement_sample(model, normal, noise, tape, deterministic)
    action = np.zeros(6, np.float32)
    for axis, value in enumerate(movement):
        action[axis if axis < 3 else 5] = math.tanh(float(value))
    if persistent:
        probabilities = tools.probs[0].numpy()
        commands = (probabilities.argmax(-1) if deterministic else np.asarray([min(2, np.searchsorted(np.cumsum(probabilities[t]), tape[8 + t], side='right')) for t in range(2)]))
        buttons = advance_buttons(buttons, commands, blind)
    else:
        buttons = ((.5 if deterministic else tape[8:]) < tools.probs[0].numpy()).astype(np.float32)
        commands = np.where(buttons, 1, 2)
    action[3:5] = buttons
    if blind:
        action.fill(0)
        buttons.fill(0)
    return action, memory, buttons, commands, next_noise


def runs(values, enabled=lambda value: value >= 0):
    """Lengths of consecutive stretches of enabled values."""
    lengths = []
    previous = None
    length = 0
    for value in values:
        if value != previous:
            if length:
                lengths.append(length)
            length = 0
        if enabled(value):
            length += 1
        previous = value
    if length:
        lengths.append(length)
    return lengths


def evaluation_arena(arena_config):
    configuration = dict(DEFAULT_ARENA)
    if arena_config is not None:
        if set(arena_config) - (set(configuration) | {'prep', 'play'}):
            raise ValueError('Only declared geometry and episode durations may vary in evaluation')
        configuration.update(arena_config)
    return configuration


def agent_contacts(env):
    """Per role: whether the agent touches a wall and whether it touches a prop this tick."""
    wall = [False, False]
    prop = [False, False]
    for contact in env.data.contact:
        if contact.dist > 0:
            continue
        for role in range(2):
            geom = env.agent_geoms[role]
            other = contact.geom2 if contact.geom1 == geom else contact.geom1 if contact.geom2 == geom else -1
            if other >= 0:
                wall[role] |= env.model.geom_group[other] == WALL_GROUP
                prop[role] |= env.model.geom_group[other] == PROP_GROUP
    return wall, prop


def tick_record(env, role, buttons, commands, wall, prop):
    speed = float(np.linalg.norm(env.data.qvel[role * 4:role * 4 + 2]))
    own = env.data.qpos[role * 4:role * 4 + 2]
    other = env.data.qpos[(1 - role) * 4:(1 - role) * 4 + 2]
    pushing = np.linalg.norm(env.actions[role, :2]) > .5
    return dict(grip=env.grips[role], buttons=buttons.tolist(), commands=commands, wall=bool(wall), prop=bool(prop),
                blockedMotion=blocked_roles(env)[role], speed=speed, wallPress=bool(wall and speed < .1 and pushing), stuck=bool(speed < .05 and pushing),
                seesOpponent=bool(env.seen[role, 1 - role]), opponentDistance=float(np.linalg.norm(own - other)),
                yawRate=float(env.data.qvel[role * 4 + 3]), action=env.actions[role].tolist())


def role_report(history, role, prep):
    """Pursuit, grip, wall-press and command statistics of one role's episode history."""
    grip = runs([row['grip'] for row in history])
    holding = [runs([row['buttons'][tool] for row in history], lambda value: value == 1) for tool in range(2)]
    action = np.asarray([row['action'] for row in history])
    seen = [x['seesOpponent'] for x in history[prep:]]
    gaps = []
    gap = None
    for tick, visible in enumerate(seen):
        if tick and seen[tick - 1] and not visible:
            gap = tick
        if visible and gap is not None:
            gaps.append((tick - gap) * DT)
            gap = None
    progress = sum(max(-1, min(1, history[t - 1]['opponentDistance'] - history[t]['opponentDistance']))
                   for t in range(prep + 1, len(history)) if history[t - 1]['seesOpponent'])
    return dict(reacquisitionSeconds=gaps, unresolvedLostSight=gap is not None, visiblePursuitProgress=progress,
                gripDurationsTicks=grip, requestedHoldDurationsTicks=holding,
                wallPressFrames=sum(row['wallPress'] for row in history),
                playWallPressFrames=sum(row['wallPress'] for row in history[prep:]),
                playStuckFrames=sum(row['stuck'] for row in history[prep:]),
                playBlockedFrames=sum(row.get('blockedMotion', False) for row in history[prep:]),
                visibleRetreatFrames=sum(history[t]['opponentDistance'] > history[t - 1]['opponentDistance'] + .01
                                         for t in range(prep + 1, len(history)) if history[t - 1]['seesOpponent']),
                sightLosses=len(gaps) + int(gap is not None),
                propContactFrames=sum(row['prop'] for row in history),
                meanSpeed=float(np.mean([row['speed'] for row in history])),
                absoluteTurns=float(sum(abs(row['yawRate']) * DT for row in history) / (2 * math.pi)),
                saturatedXYFraction=float((np.abs(action[:, :2]) > .98).mean()),
                commandCounts=[[sum(row['commands'][tool] == command for row in history[(prep if role else 0):])
                                for command in range(3)] for tool in range(2)])


def episode(models, seed, scenario, mode, trace=False, arena_config=None, role_modes=("sample", "sample"), fixture=None):
    """Play one seeded episode in ``mode`` and return its measurements."""
    if len(role_modes) != 2 or any(x not in ('mean', 'sample') for x in role_modes):
        raise ValueError('Specify mean/sample for both roles')
    configuration = evaluation_arena(arena_config)
    arena = None
    if fixture:
        from behavior_fixtures import fixture_arena, target_action
        arena = fixture_arena(fixture, seed)
        configuration.update(prep=0)
    env = PhysicsEnv(seed=seed, scenario=scenario, **configuration, arena=arena, disable_tools=mode == 'no-tools')
    physical = game_observe(env)
    memory = [torch.zeros(1, model.hidden_size) for model in models]
    buttons = np.zeros((2, 2), np.float32)
    noises = np.zeros((2, NOISE), np.float32)
    random = [np.random.default_rng(seed + offset) for offset in SEED_OFFSETS]
    histories = [[], []]
    frames = [env.trace()] if trace else None
    first_sight = None
    blackout_ticks = 0
    diverged = False
    with torch.no_grad():
        for tick in range(env.prep + env.play):
            actions = np.zeros((2, 6), np.float32)
            commands = []
            view = physical.copy()
            if first_sight is None and env.t >= env.prep and env.seen[1, 0]:
                first_sight = tick
            if mode == 'seeker-blackout':
                if first_sight is None and env.seen[1, 0]:
                    first_sight = tick
                if first_sight is not None and first_sight < tick <= first_sight + BLACKOUT_STEPS:
                    # Observation-only probe: the physical world and the recurrence
                    # are untouched; only the seeker's opponent block is hidden.
                    view[1, OPPONENT_BLOCK] = 0
                    blackout_ticks += 1
            for role, model in enumerate(models):
                if mode == f'no-{["hider", "seeker"][role]}-memory':
                    memory[role].zero_()
                actions[role], memory[role], buttons[role], chosen, noises[role] = sample(
                    model, view[role], memory[role], buttons[role], random[role], deterministic=role_modes[role] == 'mean', noise=noises[role])
                commands.append(chosen.tolist())
                if mode == 'no-hider-tools' and role == 0 or mode == 'no-seeker-tools' and role == 1:
                    actions[role, 3:5] = 0
            if fixture:
                actions[0] = target_action(fixture, env)
                buttons[0].fill(0)
            _, reward, done, info = env.step(actions)
            if unstable(env):
                diverged = True
                break
            physical, reward, done, info = resolve_capture(env, game_observe(env), reward, done, info)
            wall, prop = agent_contacts(env)
            for role in range(2):
                histories[role].append(tick_record(env, role, buttons[role], commands[role], wall[role], prop[role]))
            if trace:
                frame = env.trace()
                frame['controller'] = dict(buttons=buttons.copy(), commands=commands)
                frames.append(frame)
            if done:
                break
    roles = [role_report(history, role, env.prep) for role, history in enumerate(histories)]
    result = dict(seed=seed, scenario=scenario, mode=mode, hiddenFraction=info['hidden'] / info['play_steps'],
                  propShieldedFraction=info['shielded'] / info['play_steps'], propDisplacement=sum(info['object_displacement']),
                  info=info, roles=roles, blackoutTicks=blackout_ticks, firstSightTick=first_sight, diverged=diverged,
                  roleModes=list(role_modes), diagnosticFixture=fixture,
                  captured=bool(info.get('captured', False)), captureSecond=None if not info.get('captured') else info['captureStep'] * DT,
                  seekerFound=first_sight is not None, hiderEscaped=first_sight is not None and not info.get('captured', False))
    if arena_config is not None:
        result.update(arenaConfig=configuration, actualObjectCount=len(env.arena['objects']))
    if trace:
        result.update(arena=env.arena, frames=frames)
    env.close()
    return result


def role_summary(details):
    grip = [length for row in details for length in row['gripDurationsTicks']]
    holding = [[length for row in details for length in row['requestedHoldDurationsTicks'][tool]] for tool in range(2)]
    reacquired = [x for row in details for x in row['reacquisitionSeconds']]
    return dict(meanReacquisitionSeconds=float(np.mean(reacquired)) if reacquired else None,
                unresolvedLostSightRate=float(np.mean([row['unresolvedLostSight'] for row in details])),
                meanVisiblePursuitProgress=float(np.mean([row['visiblePursuitProgress'] for row in details])),
                meanVisibleRetreatFrames=float(np.mean([row.get('visibleRetreatFrames', 0) for row in details])),
                meanPlayBlockedFrames=float(np.mean([row.get('playBlockedFrames', 0) for row in details])),
                gripCount=len(grip),
                medianGripSeconds=float(np.median(grip) * DT) if grip else 0,
                meanGripSeconds=float(np.mean(grip) * DT) if grip else 0,
                maxGripSeconds=float(max(grip) * DT) if grip else 0,
                fractionGripsShorterThanThreeTicks=float(np.mean(np.asarray(grip) < 3)) if grip else 0,
                requestedHoldMedianSeconds=[float(np.median(values) * DT) if values else 0 for values in holding],
                meanPlayWallPressFrames=float(np.mean([row['playWallPressFrames'] for row in details])),
                meanPlayStuckFrames=float(np.mean([row.get('playStuckFrames', 0) for row in details])),
                meanAbsoluteTurns=float(np.mean([row['absoluteTurns'] for row in details])),
                meanSpeed=float(np.mean([row['meanSpeed'] for row in details])),
                commandCounts=np.sum([row['commandCounts'] for row in details], axis=0).tolist())


def mode_summary(group):
    summary = {key: float(np.mean([row[key] for row in group])) for key in ['hiddenFraction', 'propShieldedFraction', 'propDisplacement']}
    capture_times = [row['captureSecond'] for row in group if row.get('captured')]
    summary['meanCaptureSeconds'] = float(np.mean(capture_times)) if capture_times else None
    summary['captureRate'] = float(np.mean([row.get('captured', False) for row in group]))
    summary['foundRate'] = float(np.mean([row.get('seekerFound', row.get('firstSightTick') is not None) for row in group]))
    summary['escapeRate'] = float(np.mean([row.get('hiderEscaped', False) for row in group]))
    sightings = [row['firstSightTick'] for row in group if row.get('firstSightTick') is not None]
    summary['meanFirstSightSeconds'] = float(np.mean(sightings) * DT) if sightings else None
    summary['roles'] = [role_summary([row['roles'][role] for row in group]) for role in range(2)]
    return summary


def summarize(records, contrast_definitions=None):
    """Per-mode means plus bootstrapped paired contrasts of the hidden fraction."""
    summary = {}
    for mode in dict.fromkeys(row['mode'] for row in records):
        summary[mode] = mode_summary([row for row in records if row['mode'] == mode])
    by_seed = {}
    for row in records:
        by_seed.setdefault((row['seed'], row['scenario']), {})[row['mode']] = row
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrast_modes = list(DEFAULT_CONTRASTS)
    if 'binary-pair' in summary:
        contrast_modes.extend(BINARY_CONTRASTS)
    if contrast_definitions is not None:
        contrast_modes = contrast_definitions
    contrasts = {}
    for label, left, right in contrast_modes:
        values = np.asarray([rows[left]['hiddenFraction'] - rows[right]['hiddenFraction'] for rows in by_seed.values()])
        boot = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
        contrasts[label] = dict(mean=float(values.mean()), bootstrap95Percent=np.quantile(boot, [.025, .975]).tolist(),
                                positiveCases=int((values > 0).sum()), negativeCases=int((values < 0).sum()),
                                unchangedCases=int((values == 0).sum()))
    return summary, contrasts


def compared_models(args, learned, saved):
    """Every actor pairing of the pilot validation, plus the optional trained binary baseline."""
    warm, _ = load(args.warm_start)
    initial, initial_saved = load(args.initial)
    parent, _ = load(args.parent)
    if initial_saved['decisions'] != 0:
        raise ValueError('Initial must have zero experience')
    compared = {'learned': learned, 'no-hider-tools': learned, 'no-seeker-tools': learned, 'no-tools': learned,
                'warm-start': warm, 'initial-pair': initial,
                'trained-hider-v-initial': [learned[0], initial[1]], 'trained-seeker-v-initial': [initial[0], learned[1]],
                'warm-hider-v-initial': [warm[0], initial[1]], 'warm-seeker-v-initial': [initial[0], warm[1]],
                'parent-hider-v-initial': [parent[0], initial[1]], 'parent-seeker-v-initial': [initial[0], parent[1]]}
    strong_metadata = None
    if args.strong_baseline:
        strong, strong_saved = load(args.strong_baseline)
        if strong_saved.get('physicsSHA256') != saved['provenance']['physicsSHA256']:
            raise ValueError('The trained binary comparison must use the same physical environment')
        compared.update({'binary-pair': strong, 'pilot-hider-v-binary': [learned[0], strong[1]],
                         'pilot-seeker-v-binary': [strong[0], learned[1]]})
        strong_metadata = dict(checkpointSHA256=hashlib.sha256(Path(args.strong_baseline).read_bytes()).hexdigest(),
                               decisions=strong_saved['decisions'], format=strong_saved['format'],
                               physicsSHA256=strong_saved['physicsSHA256'])
    return compared, strong_metadata


def main():
    parser = argparse.ArgumentParser()
    for name in ['checkpoint', 'warm-start', 'initial', 'parent', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--episodes-per-scenario', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1500010000)
    parser.add_argument('--strong-baseline', help='Optional additional frozen trained binary pair and role comparisons')
    args = parser.parse_args()
    if not 1500000000 <= args.seed < 1900000000:
        raise ValueError('Pilot uses validation seeds only')
    learned, saved = load(args.checkpoint)
    compared, strong_metadata = compared_models(args, learned, saved)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for scenario_index, scenario in enumerate(['shelter', 'rooms', 'open']):
        for index in range(args.episodes_per_scenario):
            seed = args.seed + scenario_index * 1000 + index
            for mode, models in compared.items():
                records.append(episode(models, seed, scenario, mode))
            (destination / 'episodes.partial.json').write_text(json.dumps(records, separators=(',', ':'), default=plain) + '\n')
        subset = [row for row in records if row['scenario'] == scenario]
        means = {mode: float(np.mean([row['hiddenFraction'] for row in subset if row['mode'] == mode])) for mode in compared}
        print(json.dumps(dict(scenario=scenario, hiddenMeans=means)), flush=True)
    summary, contrasts = summarize(records)
    report = dict(format='persistent-button-pilot-validation-v1',
                  checkpointSHA256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  parentSHA256=hashlib.sha256(Path(args.parent).read_bytes()).hexdigest(), provenance=saved['provenance'],
                  parentDecisions=saved['parentDecisions'], pilotDecisions=saved['pilotDecisions'], seedStart=args.seed,
                  seedUsage='Repeated validation maps, not untouched final-test data. Training seeds are below 2**30.',
                  sampling='Independent role-specific seeded uniform tapes; Box–Muller Gaussian plus categorical inverse CDF. Same tapes across modes.',
                  initial='Actual zero-experience original backbone with new uniform categorical head. Identical initial actors are fixed in all role comparisons.',
                  additionalFrozenTrainedBaseline=strong_metadata, episodesPerScenario=args.episodes_per_scenario,
                  summary=summary, contrasts=contrasts, episodes=records)
    (destination / 'evaluation.json').write_text(json.dumps(report, indent=2, default=plain) + '\n')
    print(json.dumps(contrasts, indent=2), flush=True)


if __name__ == '__main__':
    main()
