"""Validation-only CTDE comparison; actors run without any privileged critic."""
import argparse
import json
from pathlib import Path
from persistent_evaluate import load, episode, summarize, plain
from persistent_actor import FORMAT
from persistent_train import file_hash


def main(args):
    if not 1500000000 <= args.seed < 1899000000:
        raise ValueError('Use development validation namespaces; reserve final 1.9B')
    learned, saved = load(args.checkpoint)
    previous, parent = load(args.parent)
    initial, initial_saved = load(args.initial)
    for record in [saved, parent, initial_saved]:
        if record['format'] != FORMAT:
            raise ValueError('All compared actors must use the same persistent-button architecture')
    if initial_saved['decisions'] != 0:
        raise ValueError('The initial opponent must have zero experience')
    if saved['provenance']['parentSHA256'] != file_hash(args.parent):
        raise ValueError('The frozen comparison must be this exact actor parent')
    if any(record['provenance']['physicsSHA256'] != file_hash(Path(__file__).with_name('physics.py'))
           for record in [saved, parent]):
        raise ValueError('Compared checkpoints require identical current physics')
    modes = {
        'learned': learned, 'no-hider-tools': learned, 'no-seeker-tools': learned, 'no-tools': learned,
        'parent-pair': previous, 'initial-pair': initial,
        'trained-hider-v-initial': [learned[0], initial[1]],
        'trained-seeker-v-initial': [initial[0], learned[1]],
        'parent-hider-v-initial': [previous[0], initial[1]],
        'parent-seeker-v-initial': [initial[0], previous[1]],
        'trained-hider-v-parent': [learned[0], previous[1]],
        'trained-seeker-v-parent': [previous[0], learned[1]],
    }
    definitions = [
        ('Hider tools benefit', 'learned', 'no-hider-tools'),
        ('Seeker tools benefit', 'no-seeker-tools', 'learned'),
        ('Hider change vs fixed parent seeker', 'trained-hider-v-parent', 'parent-pair'),
        ('Seeker change vs fixed parent hider', 'parent-pair', 'trained-seeker-v-parent'),
        ('Hider change vs fixed initial seeker', 'trained-hider-v-initial', 'parent-hider-v-initial'),
        ('Seeker change vs fixed initial hider', 'parent-seeker-v-initial', 'trained-seeker-v-initial'),
        ('Total hider learning vs initial', 'trained-hider-v-initial', 'initial-pair'),
        ('Total seeker learning vs initial', 'initial-pair', 'trained-seeker-v-initial'),
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation.json').exists():
        raise ValueError('Preserve previously evaluated reports in their own directory')
    records = []
    for offset, scenario in enumerate(['shelter', 'rooms', 'open']):
        for index in range(args.episodes_per_scenario):
            seed = args.seed + offset * 1000 + index
            for mode, actors in modes.items():
                records.append(episode(actors, seed, scenario, mode))
            (output / 'episodes.partial.json').write_text(json.dumps(records, default=plain) + '\n')
        print(json.dumps(dict(scenario=scenario, completedGames=len(records))), flush=True)
    summary, contrasts = summarize(records, definitions)
    report = dict(format='residual-ctde-pilot-validation-v1',
        checkpointSHA256=file_hash(args.checkpoint), parentSHA256=file_hash(args.parent),
        initialSHA256=file_hash(args.initial), provenance=saved['provenance'],
        training={key: saved.get(key, 0) for key in
                  ['parentDecisions', 'criticWarmupDecisions', 'actorUpdateDecisions', 'decisions']},
        seedStart=args.seed, episodesPerScenario=args.episodes_per_scenario,
        scope='Three scenarios at 8m with three boxes and one ramp; validation only, not full size/count coverage.',
        sampling='Same role-specific uniform action tapes per paired map, sampled Gaussian movement and categorical tool commands.',
        toolsAblation='Requested actor button state remains unchanged; physical grab/lock disabled only for the named role. Pushing remains possible.',
        privacy='Only exported-compatible restricted recurrent actor inputs enter inference; no central critic executes during evaluation.',
        summary=summary, contrasts=contrasts, episodes=records,
        sources={name: file_hash(Path(__file__).with_name(name)) for name in
                 ['evaluate_central_pilot.py', 'persistent_evaluate.py', 'persistent_actor.py', 'physics.py']})
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, default=plain) + '\n')
    print(json.dumps(contrasts, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--initial', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes-per-scenario', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1500080000)
    main(parser.parse_args())
