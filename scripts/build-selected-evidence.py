"""Package the exact selected-role development evidence without new rollouts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
from persistent_train import file_hash


def build(args):
    inputs = {name: json.loads(Path(getattr(args, name)).read_text())
              for name in ['selection', 'baseline', 'cache', 'modes', 'tools', 'browser_comparison']}
    selection, baseline, cache, modes, tools, browser = [inputs[name] for name in inputs]
    digest = selection['candidateSHA256']
    if any(record['candidateSHA256'] != digest for record in [cache, modes, tools]):
        raise ValueError('All records must describe the exact selected pair')
    if browser['checkpointSHA256'] != selection['roleSources'][1]['sourceCheckpointSHA256']:
        raise ValueError('Browser comparison belongs to another seeker')
    records = cache['episodes'] + modes['newEpisodes'] + tools['newEpisodes']
    conditions = list(dict.fromkeys(row['mode'] for row in records))
    paired = {}
    for row in records:
        key = (row['seed'], row['scenario'])
        values = paired.setdefault(key, {})
        if row['mode'] in values:
            raise ValueError('A logical condition must occur once per map')
        if row['hiddenFraction'] != row['info']['hidden'] / row['info']['play_steps']:
            raise ValueError('Each summary must match the actual physical visibility count')
        values[row['mode']] = row['hiddenFraction']
    if len(paired) != 96 or any(set(values) != set(conditions) for values in paired.values()):
        raise ValueError('All twenty conditions must cover the same 96 development maps')
    comparisons = [
        ('Selected seeker vs original browser seeker', 'browser_pair_sample', 'candidate_pair_sample',
         browser['contrasts']['Seeker change vs browser seeker, same37M hider']),
        ('Sampled hider vs genuine initial, same browser seeker', 'candidate_hider_sample_fixed_browser_seeker',
         'initial_hider_sample_fixed_browser_seeker', modes['contrasts']['Sampled hider learning vs genuine sampled initial, same opponent']),
        ('Sampled seeker vs genuine initial, same browser hider', 'fixed_browser_hider_initial_seeker_sample',
         'fixed_browser_hider_candidate_seeker_sample', modes['contrasts']['Sampled seeker learning vs genuine sampled initial, same opponent']),
        ('Mean seeker vs sampled, same sampled browser hider', 'fixed_browser_hider_candidate_seeker_sample',
         'fixed_browser_hider_candidate_seeker_mean', modes['contrasts']['Seeker mean vs sampled, same sampled browser hider']),
        ('Sampled hider grab/lock benefit', 'candidate_pair_sample', 'sample_no_hider_tools',
         tools['contrasts']['sample: hider grab/lock benefit']),
        ('Sampled seeker grab/lock benefit', 'sample_no_seeker_tools', 'candidate_pair_sample',
         tools['contrasts']['sample: seeker grab/lock benefit']),
        ('Mean hider grab/lock benefit', 'candidate_pair_mean', 'mean_no_hider_tools',
         tools['contrasts']['mean: hider grab/lock benefit']),
        ('Mean seeker grab/lock benefit', 'mean_no_seeker_tools', 'candidate_pair_mean',
         tools['contrasts']['mean: seeker grab/lock benefit']),
    ]
    effects = {}
    for label, left, right, result in comparisons:
        mean = sum(values[left] - values[right] for values in paired.values()) / len(paired)
        if abs(mean - result['mean']) > 1e-12:
            raise ValueError(f'{label} does not match saved physical games')
        effects[label] = dict(result, leftCondition=left, rightCondition=right)
    evidence = dict(format='selected-role-development-evidence-v1', status='DEVELOPMENT',
        checkpointSHA256=digest, roleSources=selection['roleSources'],
        sourceReports={name: file_hash(getattr(args, name)) for name in inputs},
        maps=96, scope='Three scenarios, sizes 6–12 m, zero to eight boxes and zero to two ramps. These maps were reused for selection; they are not an untouched final test.',
        qualification='Search improved descriptively against the original reference, with an interval crossing zero. Reliable useful grab/lock strategies remain unproven. This is a development checkpoint.',
        effectUnits='Role objective fraction; multiply by 100 for percentage points. Paired-map bootstrap 95% intervals; repeated selection is not adjusted away.',
        sampling='Seeded action distributions remain the default. Mean is the actual tanh Gaussian mean plus categorical argmax, without additional control changes.',
        modes=modes['summary'], toolInterventions=tools['summary'], contrasts=effects,
        conditionOrder=conditions,
        pairedMaps=[dict(seed=seed, scenario=scenario,
                        hiddenFractions=[values[name] for name in conditions])
                    for (seed, scenario), values in paired.items()],
        newOptimizationDuringSelection=0,
        preservation='The final 31,981,568-interaction native continuation remains latest; the earlier seeker is retained for broader measured behavior.')
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(dict(file=str(destination), bytes=destination.stat().st_size,
                         sha256=file_hash(destination), maps=96)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['selection', 'baseline', 'cache', 'modes', 'tools', 'browser-comparison', 'output']:
        parser.add_argument('--' + name, required=True)
    build(parser.parse_args())
