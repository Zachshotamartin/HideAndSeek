"""Current entity-policy playback and targeted pursuit diagnostics. Never trains or publishes."""
import argparse
import json
import io
import hashlib
from pathlib import Path
import torch
from assessment_stats import paired_interval
from behavior_fixtures import FIXTURES
from checkpoint_store import atomic_json
from entity_actor import load_pair
from persistent_evaluate import episode, plain
from persistent_train import file_hash

# Paired role-mode tests keep the opponent sampled; pair mean matches deterministic UI.
MODES = dict(sample=('sample','sample'), mean=('mean','mean'),
             hider_mean=('mean','sample'), seeker_mean=('sample','mean'))
SOURCES = ('assess_behavior.py', 'behavior_fixtures.py', 'persistent_evaluate.py', 'physics.py',
           'capture.py', 'entity_actor.py', 'persistent_actor.py', 'actor.py', 'assessment_stats.py')


def assess(checkpoint, seeds, play=188):
    if len(seeds) != len(set(seeds)) or not seeds or any(not 1900000000 <= s < 2000000000 for s in seeds):
        raise ValueError('Use distinct reserved diagnostic seeds in [1900000000, 2000000000)')
    if play < 1:
        raise ValueError('Play length must be positive')
    raw = Path(checkpoint).read_bytes()
    models, saved = load_pair(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False))
    if saved['provenance']['physicsSHA256'] != file_hash(Path(__file__).with_name('physics.py')):
        raise ValueError('Checkpoint physical schema does not match this evaluator')
    rows = []
    for seed in seeds:
        for name, modes in MODES.items():
            row = episode(models, seed, 'multi-exit', 'learned', role_modes=modes,
                          arena_config=dict(size=8, n_boxes=4, n_ramps=1, prep=96, play=play))
            rows.append(dict(condition=name, fixture='learned-pair', **row))
        for fixture in FIXTURES:
            for name in ('sample','mean'):
                row = episode(models, seed, 'open', 'learned', role_modes=MODES[name], fixture=fixture,
                              arena_config=dict(size=8, n_boxes=0, n_ramps=0, prep=0, play=play))
                rows.append(dict(condition=name, fixture=fixture, **row))
    contrasts = {}
    for fixture in ('learned-pair', *FIXTURES):
        selected = [r for r in rows if r['fixture'] == fixture]
        metrics = dict(capture=lambda r: float(r['captured']), hiddenFraction=lambda r: r['hiddenFraction'],
                       retreatFrames=lambda r: r['roles'][1]['visibleRetreatFrames'],
                       blockedFrames=lambda r: r['roles'][1]['playBlockedFrames'])
        for metric, value in metrics.items():
            a, b = [{str(r['seed']): value(r) for r in selected if r['condition'] == mode} for mode in ('mean','sample')]
            contrasts[fixture + ':' + metric] = paired_interval(a,b)
    paired = [r for r in rows if r['fixture']=='learned-pair']
    for role, left, right in [('hider','hider_mean','sample'),('seeker','sample','seeker_mean')]:
        a,b=[{str(r['seed']):r['hiddenFraction'] for r in paired if r['condition']==mode} for mode in (left,right)]
        contrasts[role + ':mean-utility-minus-sampled'] = paired_interval(a,b)
    return dict(format='entity-behavior-assessment-v1', checkpointSHA256=hashlib.sha256(raw).hexdigest(),
                sources={n:file_hash(Path(__file__).with_name(n)) for n in SOURCES}, episodes=rows, contrasts=contrasts,
                diagnostics={name:dict(episodes=len([r for r in rows if r['fixture']==name]),
                    initialSightExpected=name != 'blocked-path',
                    exercisedSightLossEpisodes=sum(r['roles'][1]['sightLosses'] > 0 for r in rows if r['fixture']==name)) for name in FIXTURES},
                scope='Diagnostic/development evidence only. Target fixtures use an explicitly controlled hider; only the seeker is assessed there. A reacquisition case with no sight loss is unexercised, not a pass.',
                automaticPublication=False, optimizerSteps=0)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(1900100100,1900100108)))
    p.add_argument('--play',type=int,default=188)
    a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise ValueError('Preserve completed assessments; choose a new output')
    torch.set_num_threads(1)
    report=assess(a.checkpoint,a.seeds,a.play)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,default=plain,indent=2,allow_nan=False)+'\n')
