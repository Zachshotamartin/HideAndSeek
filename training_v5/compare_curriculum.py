"""Matched baseline, outcome sampler and blocked-effort ablations; no publication."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from checkpoint_store import atomic_json
from persistent_train import file_hash
from train_entity import SOURCE_NAMES
from train_scaled import prepare

HERE = Path(__file__).resolve().parent
def arms_for(suite):
    if suite == 'mechanisms':
        return [('seeker-curriculum', dict(matchmaking='seeker-curriculum')),
                ('baseline', {}), ('blocked-cost', dict(blocked_cost=.02))]
    large = dict(envs=128, sequence_batch=128, snapshot_every=1)
    if suite == 'initialization':
        settings = dict(**large, value_normalization='popart', map_replay='progress', capture_credit='discounted-equivalent', matchmaking='seeker-curriculum')
        return [('continued', settings), ('fresh', settings)]
    return [
        ('combined', dict(**large, value_normalization='popart', map_replay='progress',
                          capture_credit='discounted-equivalent', matchmaking='seeker-curriculum')),
        ('baseline', {}),
        ('popart', dict(value_normalization='popart')),
        ('map-progress', dict(map_replay='progress')),
        ('discounted-capture', dict(capture_credit='discounted-equivalent')),
        ('large-batch', large),
    ]


def prepare_trial(args, dest, seed, arm, overrides):
    setup = prepare(Path(args.source), dest, args.steps, seed, 'full', Path(args.cohort), args.history, args.heldout)
    config = setup['arguments']
    config.update(envs=32, workers=1, matchmaking='uniform', blocked_cost=0., save_every=4)
    config.update(overrides)
    if getattr(args, 'suite', None) == 'initialization':
        from initialization_comparison import prepare_initialization
        prepare_initialization(config, arm, seed)
    block = config['envs'] * config['horizon'] * 2
    if args.steps % block:
        raise ValueError('Budget must contain complete rollout batches')
    stride = 1048576 // block
    config['retain_updates'] = ','.join(str(i) for i in range(stride, args.steps // block + 1, stride))
    protocol = Path(config['protocol'])
    declaration = json.loads(protocol.read_text())
    declaration['format'] = 'hide-seek-controlled-training-v1'
    declaration['training'].update({k: config[k] for k in declaration['training']})
    declaration['reward'] = ('Unchanged capture/visibility outcome; optional zero-sum blocked effort cost '
                             f"{config['blocked_cost']} per play decision; capture credit {config['capture_credit']}. "
                             'Evaluation always reports original capture/visibility outcomes.')
    declaration['comparisonArm'] = arm
    for key, field in [('entity','parent'),('critic','critic')]:
        declaration['assets'][key] = dict(path=config[field], sha256=file_hash(config[field]))
    declaration['selfPlay'] = declaration['selfPlay'].split('; historical seeker opponents:')[0]
    declaration['selfPlay'] += '; historical seeker opponents: ' + config['matchmaking']
    declaration['comparisonSettings'] = overrides
    declaration['batchUnits'] = dict(environmentTransitions=block // 2, roleInteractions=block,
                                    recurrentSequencesPerOptimizerBatch=config['sequence_batch'])
    if protocol.read_text() != json.dumps(declaration, indent=2) + '\n':
        if (dest / 'run/latest.pt').exists():
            raise ValueError('Refusing to alter an already-started protocol')
        atomic_json(protocol, declaration)
    return config


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    arms = arms_for(args.suite)
    identity = dict(arguments=vars(args), arms=arms,
                    sources={n: file_hash(HERE / n) for n in [*SOURCE_NAMES, 'compare_curriculum.py', 'initialization_comparison.py', 'comparison_summary.py', 'assessment_stats.py']},
                    inputs={p: file_hash(p) for p in [args.source, args.cohort, args.heldout, *args.history]})
    identity = json.loads(json.dumps(identity))
    plan = out / 'PLAN.json'
    if plan.exists() and json.loads(plan.read_text()) != identity:
        raise ValueError('Comparison inputs changed; choose a new folder')
    atomic_json(plan, identity)
    stopped = False
    child = None
    evaluating = False

    def stop(*_):
        nonlocal stopped
        stopped = True
        if child and child.poll() is None:
            if evaluating:
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def launch(command, label):
        nonlocal child, evaluating
        evaluating = label.endswith('-evaluate')
        with (out / (label + '.log')).open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(out / 'STATUS.json', dict(phase=label, childPID=child.pid))
            code = child.wait()
        if stopped:
            atomic_json(out / 'STATUS.json', dict(phase='paused', childPID=None))
            return False
        if code:
            atomic_json(out / 'STATUS.json', dict(phase='failed', task=label, exitCode=code))
            raise RuntimeError('Comparison child failed; inspect its log')
        return True

    for seed in args.seeds:
        for arm, overrides in arms:
            label = f'{arm}-{seed}'
            dest = out / label
            dest.mkdir(exist_ok=True)
            report = dest / 'evaluation' / 'evaluation.json'
            if report.exists():
                continue
            config = prepare_trial(args, dest, seed, arm, overrides)
            config['resume'] = str(dest / 'run/latest.pt') if (dest / 'run/latest.pt').exists() else None
            command = [sys.executable, str(HERE / 'train_entity.py')]
            # These arguments originate from the trainer's own parser; preserve lists and defaults.
            for key, value in config.items():
                if value is None:
                    continue
                command.append('--' + key.replace('_', '-'))
                if isinstance(value, list):
                    command.extend(map(str, value))
                else:
                    command.append(str(value))
            if not launch(command, label):
                return
            command = [sys.executable, str(HERE / 'evaluate_saved.py'), '--checkpoint', str(dest / 'run/latest.pt'),
                       '--reference', str(dest / 'prepared/reference.pt'), '--heldout', str(dest / 'prepared/heldout.pt'),
                       '--cohort', str(dest / 'prepared/cohort.json'), '--output', str(dest / 'evaluation'), '--workers', '1']
            if not launch(command, label + '-evaluate'):
                return
    reports = {}
    for seed in args.seeds:
        reports[str(seed)] = {}
        for arm, _ in arms:
            data = json.loads((out / f'{arm}-{seed}' / 'evaluation/evaluation.json').read_text())
            reports[str(seed)][arm] = dict(contrasts=data['contrasts'], byPlayLength=data['byPlayLength'])
    from comparison_summary import summarize_comparison
    stat_rows = []
    for seed in args.seeds:
        metrics = {}
        for arm, _ in arms:
            blocks = reports[str(seed)][arm]['byPlayLength'].values()
            blocks = [b for b in blocks if b.get('maps')]
            hider = sum(b['summary']['candidate-hider']['hiddenFraction'] for b in blocks) / len(blocks)
            seeker = sum(1-b['summary']['candidate-seeker']['hiddenFraction'] for b in blocks) / len(blocks)
            metrics[arm] = dict(hiderUtility=hider,seekerUtility=seeker,score=.5*(hider+seeker))
        stat_rows.append(dict(seed=seed,metrics=metrics))
    statistics = summarize_comparison(stat_rows, 'continued' if args.suite == 'initialization' else 'baseline')
    atomic_json(out / 'RESULTS.json', dict(pairedSeeds=stat_rows, details=reports, statistics=statistics, publication='Manual review; never select on training reward'))
    atomic_json(out / 'STATUS.json', dict(phase='complete-awaiting-review', childPID=None))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=['mechanisms', 'research', 'initialization'], default='mechanisms')
    for name in ['output', 'source', 'cohort', 'heldout']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--history', nargs='*', default=[])
    p.add_argument('--seeds', nargs='+', type=int, default=[91201, 91202, 91203, 91204, 91205])
    p.add_argument('--steps', type=int, default=5242880)
    args = p.parse_args()
    block = 65536 if args.suite in ('research', 'initialization') else 16384
    if args.steps <= 0 or args.steps % block:
        p.error(f'Use a positive whole number of {block}-interaction updates')
    run(args)
