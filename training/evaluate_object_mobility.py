"""Frozen-actor object-mobility evaluation, without training or reward changes.

The neutral-cue fixed condition is an explicitly labeled diagnostic: only the
externally-fixed lock indicators are censored, never hidden object positions.
A separate MuJoCo data instance probes occlusion, so probes cannot perturb play.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from persistent_evaluate import load, sample, plain
from persistent_actor import FORMAT
from persistent_train import file_hash
from physics import PhysicsEnv

CONDITIONS = ('tools', 'push-only', 'fixed-props', 'fixed-neutral-cues')


def neutral_lock_cues(observations):
    result = observations.copy()
    for slot in range(6):
        result[:, 18 + slot * 16 + 13:18 + slot * 16 + 15] = 0
    return result


def episode(models, seed, scenario, condition, trace=False, probes=True):
    fixed = condition.startswith('fixed-')
    env = PhysicsEnv(seed=seed, scenario=scenario, size=8, n_boxes=3, n_ramps=1,
                     disable_tools=condition != 'tools', immovable=fixed)
    probe = PhysicsEnv(arena=env.arena) if probes else None
    initial_qpos = env.data.qpos.copy()
    initial_xy = initial_qpos[8:].reshape(-1, 4)[:, :2]
    previous_xy = initial_xy.copy()
    observations = env.observe()
    memories = [torch.zeros(1, 64), torch.zeros(1, 64)]
    buttons = np.zeros((2, 2), np.float32)
    generators = [np.random.default_rng(seed + 912341), np.random.default_rng(seed + 2912341)]
    object_index = {int(geom): index for index, geom in enumerate(env.object_geoms)}
    path = np.zeros(env.n_obj)
    maximum = np.zeros(env.n_obj)
    contact_steps = np.zeros(2, int)
    moving_contacts = np.zeros(2, int)
    contact_motion = np.zeros(2)
    moved_hidden = moved_exposed = 0
    frames = [env.trace()] if trace else None
    with torch.no_grad():
        for tick in range(env.prep + env.play):
            inputs = neutral_lock_cues(observations) if condition == 'fixed-neutral-cues' else observations
            actions = np.zeros((2, 5), np.float32)
            commands = []
            for role, model in enumerate(models):
                actions[role], memories[role], buttons[role], command = sample(
                    model, inputs[role], memories[role], buttons[role], generators[role])
                commands.append(command.tolist())
            observations, reward, done, info = env.step(actions)
            np.testing.assert_array_equal(reward.sum(), 0)
            current_xy = env.data.qpos[8:].reshape(-1, 4)[:, :2].copy()
            motion = np.linalg.norm(current_xy - previous_xy, axis=1)
            path += motion
            maximum = np.maximum(maximum, np.linalg.norm(current_xy - initial_xy, axis=1))
            previous_xy = current_xy
            contacts = [set(), set()]
            for contact in env.data.contact:
                if contact.dist > 0:
                    continue
                for role in range(2):
                    geom = env.agent_geoms[role]
                    other = contact.geom2 if contact.geom1 == geom else contact.geom1 if contact.geom2 == geom else -1
                    index = object_index.get(int(other))
                    if index is not None and env.grips[role] != index:
                        contacts[role].add(index)
            for role in range(2):
                contact_steps[role] += bool(contacts[role])
                moving_contacts[role] += any(motion[index] > .002 for index in contacts[role])
                contact_motion[role] += sum(motion[index] for index in contacts[role])
            if probe is not None and env.t > env.prep:
                probe.data.qpos[:] = env.data.qpos
                probe.data.qpos[8:] = initial_qpos[8:]
                mujoco.mj_forward(probe.model, probe.data)
                visible_at_initial_positions = probe._sees(1, probe.agent_bodies[0])
                moved_hidden += bool(not env.visible and visible_at_initial_positions)
                moved_exposed += bool(env.visible and not visible_at_initial_positions)
            if trace:
                frame = env.trace()
                frame['controller'] = dict(buttons=buttons.copy(), commands=commands)
                frames.append(frame)
            if done:
                break
    net = np.linalg.norm(current_xy - initial_xy, axis=1)
    result = dict(seed=seed, scenario=scenario, condition=condition,
        hiddenFraction=info['hidden'] / info['play_steps'],
        visibleFraction=1 - info['hidden'] / info['play_steps'],
        propShieldedFraction=info['shielded'] / info['play_steps'],
        movedPropsCausedHidingTicks=moved_hidden,
        movedPropsCausedExposureTicks=moved_exposed,
        objectNetDisplacementXY=net.tolist(), objectPathXY=path.tolist(),
        objectMaxDisplacementXY=maximum.tolist(),
        pushContactSteps=contact_steps.tolist(), pushMotionSteps=moving_contacts.tolist(),
        movementDuringPushContactXY=contact_motion.tolist(), info=info)
    if trace:
        result.update(arena=env.arena, frames=frames)
    env.close()
    if probe is not None:
        probe.close()
    return result


def summarize(records):
    generator = np.random.default_rng(901003)
    summary = {}
    contrasts = {}
    for pair in dict.fromkeys(record['pair'] for record in records):
        subset = [record for record in records if record['pair'] == pair]
        summary[pair] = {}
        for condition in CONDITIONS:
            cases = [record for record in subset if record['condition'] == condition]
            summary[pair][condition] = {
                key: float(np.mean([record[key] for record in cases])) for key in
                ['hiddenFraction', 'visibleFraction', 'propShieldedFraction',
                 'movedPropsCausedHidingTicks', 'movedPropsCausedExposureTicks']}
            for key in ['pushContactSteps', 'pushMotionSteps', 'movementDuringPushContactXY']:
                summary[pair][condition][key] = np.mean([record[key] for record in cases], axis=0).tolist()
            for key in ['objectNetDisplacementXY', 'objectPathXY', 'objectMaxDisplacementXY']:
                summary[pair][condition]['meanSum' + key[0].upper() + key[1:]] = float(np.mean([sum(record[key]) for record in cases]))
            summary[pair][condition]['mapsWithAtLeastFiveMovedCoverTicks'] = sum(record['movedPropsCausedHidingTicks'] >= 5 for record in cases)
        paired = {}
        for record in subset:
            paired.setdefault((record['seed'], record['scenario']), {})[record['condition']] = record
        contrasts[pair] = {}
        for label, left, right in [
            ('Grab-lock effect beyond pushing', 'tools', 'push-only'),
            ('Mobility effect with neutral lock cues', 'push-only', 'fixed-neutral-cues'),
            ('Mobility and truthful fixed-lock cue effect', 'push-only', 'fixed-props'),
            ('Fixed-lock cue effect at identical fixed physics', 'fixed-props', 'fixed-neutral-cues'),
        ]:
            delta = np.array([row[left]['hiddenFraction'] - row[right]['hiddenFraction'] for row in paired.values()])
            draws = generator.choice(delta, size=(10000, len(delta)), replace=True).mean(-1)
            contrasts[pair][label] = dict(hiddenFractionDifference=float(delta.mean()),
                pairedMap95PercentCI=np.quantile(draws, [.025, .975]).tolist(),
                hiderPositiveCases=int((delta > 0).sum()), seekerPositiveCases=int((delta < 0).sum()),
                unchangedCases=int((delta == 0).sum()))
    return summary, contrasts


def main(args):
    if not 1500000000 <= args.seed < 1899000000:
        raise ValueError('Use validation seeds and reserve final 1.9B')
    models, saved = load(args.checkpoint)
    initial, original = load(args.initial)
    if saved['format'] != FORMAT or original['format'] != FORMAT or original['decisions'] != 0:
        raise ValueError('Expected frozen persistent actors and genuine zero-experience comparator')
    if saved['provenance']['physicsSHA256'] != file_hash(Path(__file__).with_name('physics.py')):
        raise ValueError('The frozen model requires its unchanged native physical environment')
    pairs = {'trained-pair': models, 'trained-hider-v-initial': [models[0], initial[1]],
             'trained-seeker-v-initial': [initial[0], models[1]], 'initial-pair': initial}
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'evaluation.json').exists():
        raise ValueError('Preserve the previous assessment')
    records = []
    for offset, scenario in enumerate(['shelter', 'rooms', 'open']):
        for index in range(args.episodes_per_scenario):
            seed = args.seed + offset * 1000 + index
            for pair, actors in pairs.items():
                for condition in CONDITIONS:
                    result = episode(actors, seed, scenario, condition)
                    result['pair'] = pair
                    records.append(result)
            if (index + 1) % 8 == 0 or index + 1 == args.episodes_per_scenario:
                (destination / 'episodes.partial.json').write_text(json.dumps(records, default=plain) + '\n')
                print(json.dumps(dict(scenario=scenario, completedMaps=index + 1, totalGames=len(records))), flush=True)
    summary, contrasts = summarize(records)
    report = dict(format='frozen-policy-object-mobility-assessment-v1',
        checkpointSHA256=file_hash(args.checkpoint), initialSHA256=file_hash(args.initial),
        trainingDecisions=saved['decisions'], seedStart=args.seed, episodesPerScenario=args.episodes_per_scenario,
        scope='Three scenario families at8m with3 boxes and1 ramp; validation only. Same role-specific uniform action tapes across all paired conditions.',
        conditions={
            'tools': 'Original game: props can be pushed, grabbed and locked.',
            'push-only': 'Original movable props; physical grab and lock disabled. Actor-requested button state still advances normally.',
            'fixed-props': 'Original immovable=True diagnostic, physical grab/lock disabled. Props are truthfully observed as externally locked.',
            'fixed-neutral-cues': 'Same fixed physical props and disabled grab/lock, but externally-fixed lock indicators are censored to zero. This separates lock-cue changes from the mobility intervention; no hidden positions are exposed.'},
        causalLimits='Both agents encounter each global mobility intervention; changes are paired game outcomes, not a role-isolated pushing treatment. Same-pose initial-prop occlusion probes establish geometric consequences, not intentionality or a substitute physical trajectory. Moving-contact attribution is conservative, based on final-substep non-grab contacts and more than2mm of object movement that tick.',
        summary=summary, contrasts=contrasts, episodes=records,
        sources={name: file_hash(Path(__file__).with_name(name)) for name in
                 ['evaluate_object_mobility.py', 'persistent_evaluate.py', 'persistent_actor.py', 'physics.py']})
    (destination / 'evaluation.json').write_text(json.dumps(report, indent=2, default=plain) + '\n')
    print(json.dumps(contrasts['trained-pair'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--initial', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes-per-scenario', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1500090000)
    main(parser.parse_args())
