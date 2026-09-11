"""Larger original entity policies, long recurrent sequences, saved evaluations.

One native process owns the training. No strategy scripts or tool-use bonuses.
Resume with the same command; budgets can subsequently be extended using
train_saved.py, preserving complete actor/critic/optimizer checkpoints.

Every external input is explicit: the source pair, the evaluation cohort, the
frozen league anchors and the held-out evaluation opponent. Nothing is read
from paths remembered inside a checkpoint, so a fresh clone reproduces a run.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import torch

from checkpoint_store import atomic_json, copy_immutable, utc_now
from entity_actor import load_pair
from evaluated_models import register
from persistent_train import file_hash
from protocol import archive_exercised, extended_cohort
from train_entity import train, parser as training_parser

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ['full', 'baseline', 'short', 'unbalanced', 'recent-only', 'short-memory']
SNAPSHOT_EVERY = 4        # opponent snapshots per update; a 160-update pilot adds 40
ARCHIVE_LIMIT = 16        # anchors + eight newest + behaviour-diverse picks
SEQUENCE_LENGTH = {v: 128 for v in VARIANTS} | {'short-memory': 64}
BURN_IN = {v: 32 for v in VARIANTS} | {'baseline': 0, 'short-memory': 0}


def resolve_source_history(source, source_path):
    paths = []
    for original in source.get('arguments', {}).get('history', []):
        candidate = Path(original)
        if not candidate.is_absolute():
            candidate = source_path.parent / candidate
        paths.append(candidate)
    return paths


def prepare(source_path, output, target, seed=1091252, variant='full', cohort_path=None, history_paths=None, heldout_path=None):
    prepared = output / 'prepared'
    if cohort_path is None:
        raise ValueError('Pass --cohort: the fixed development maps are not read from the source checkpoint')
    cohort_hash = file_hash(cohort_path)
    if prepared.exists():
        setup = json.loads((prepared / 'SETUP.json').read_text())
        if setup['sourceSHA256'] != file_hash(source_path) or setup['target'] != target or setup.get('seed', 1091252) != seed \
                or setup.get('variant', 'full') != variant or setup.get('sourceCohortSHA256') != cohort_hash:
            raise ValueError('Preserve this run configuration; branch or extend with train_saved.py')
        return setup
    from entity_actor import EntityActor
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    large = load_pair(source)[0]
    parent = copy.deepcopy(source)
    parent['provenance']['roleSources'] = source['provenance'].get('roleSources', source['provenance'].get('inheritedRoleSources'))
    parent.update(newActorUpdates=0, optimizers=[torch.optim.Adam(m.parameters()).state_dict() for m in large])
    # Physical game is byte-identical. Only training distribution is changed.
    if parent['provenance']['physicsSHA256'] != file_hash(ROOT / 'training_v5/physics.py'):
        raise ValueError('New training requires the same v4 physical contract')
    critic_source = dict(centralCritic=source['centralCritic'], centralCriticSchema=source['centralCriticSchema'], provenance=copy.deepcopy(parent['provenance']))
    torch.manual_seed(seed)
    parent['torchRNG'] = torch.get_rng_state()
    prepared.mkdir(parents=True)
    torch.save(parent, prepared / 'entity.pt'); torch.save(critic_source, prepared / 'critic.pt')
    torch.save(parent, prepared / 'reference.pt')
    initial_record = copy.deepcopy(parent)
    initial_record.update(decisions=0, models=[EntityActor(m.hidden_size, m.encoder_size, m.encoder.embedding_size).state_dict() for m in large])
    torch.save(initial_record, prepared / 'initial.pt')
    seen = {file_hash(prepared / 'entity.pt')}
    candidates = [Path(p) for p in history_paths] if history_paths is not None else resolve_source_history(source, source_path)
    missing = [str(c) for c in candidates if not c.exists()]
    if missing:
        raise ValueError(f'Frozen opponent checkpoints are missing; copy them or pass --history explicitly: {missing}')
    if heldout_path is None and history_paths is None and candidates:
        # The source's own earlier parent is held out of the league so that
        # evaluation has an opponent the learner never trained against.
        heldout_path, candidates = candidates[0], candidates[1:]
    histories = [str(prepared / 'entity.pt')]
    for original in candidates:
        digest = file_hash(original)
        if digest in seen:
            continue
        record = torch.load(original, map_location='cpu', weights_only=False)
        if record['provenance']['physicsSHA256'] != parent['provenance']['physicsSHA256']:
            raise ValueError(f'Frozen opponent {original} belongs to a different physical game')
        destination = prepared / f'history-{len(histories)}.pt'; copy_immutable(original, destination); histories.append(str(destination)); seen.add(digest)
    heldout = None
    if heldout_path is not None:
        record = torch.load(heldout_path, map_location='cpu', weights_only=False)
        if record['provenance']['physicsSHA256'] != parent['provenance']['physicsSHA256']:
            raise ValueError('The held-out opponent belongs to a different physical game')
        if file_hash(heldout_path) in seen:
            raise ValueError('The held-out opponent must not be a league anchor')
        copy_immutable(heldout_path, prepared / 'heldout.pt'); heldout = str(prepared / 'heldout.pt')
    initial = str(prepared / 'initial.pt')
    args = training_parser().parse_args([
        '--parent', str(prepared / 'entity.pt'), '--critic', str(prepared / 'critic.pt'),
        '--initial', initial, '--history', *histories, '--output', str(output / 'run'),
        '--protocol', str(prepared / 'PROTOCOL.json'), '--encoder', 'entity',
        '--envs', '128', '--workers', '4', '--horizon', '256', '--sequence-length', str(SEQUENCE_LENGTH[variant]),
        '--burn-in', str(BURN_IN[variant]), '--snapshot-every', str(SNAPSHOT_EVERY), '--archive-limit', str(ARCHIVE_LIMIT),
        '--sequence-batch', '32', '--target-interactions', str(target),
        '--save-every', '1', '--retain-updates', '', '--seed', str(seed), '--variant', variant])
    fields = ['variant', 'seed', 'arm', 'envs', 'workers', 'horizon', 'sequence_length', 'burn_in', 'snapshot_every', 'archive_limit',
              'sequence_batch', 'critic_batch_size', 'epochs', 'learning_rate', 'critic_learning_rate', 'entropy', 'kl_limit', 'target_interactions']
    protocol = dict(format='hide-seek-balanced-long-league-v5.1', physicsSHA256=parent['provenance']['physicsSHA256'],
        training={k: getattr(args, k) for k in fields}, sourceCheckpointSHA256=file_hash(source_path),
        comparisons='Unchanged environment and sensor contract. Longer randomized episodes, balanced active sampling, diverse archives and real recurrent burn-in. Inherited weights, fresh optimizers. Current versus fixed migrated reference, a held-out opponent outside the league, and counterfactual tools-disabled evaluations; not an isolated architecture ablation.',
        architecture=[dict(hidden=m.hidden_size, encoder=m.encoder_size, embedding=m.encoder.embedding_size, actorParameters=sum(p.numel() for k, p in m.named_parameters() if not k.startswith('value.'))) for m in large],
        reward='Visibility-only zero-sum; zero preparation reward; no tool bonuses.', toolEntropyScale=0.1,
        selfPlay='70% current-current, 30% active-sample-balanced historical; fixed anchor, latest eight and behavioral diversity',
        evaluation=dict(sourceCohortSHA256=cohort_hash, longPlayMaps=24, heldOutOpponent=heldout is not None), assets={}, history=[])
    for key, path in [('entity', args.parent), ('critic', args.critic), ('initial', initial)]:
        protocol['assets'][key] = dict(path=path, sha256=file_hash(path))
    for i, path in enumerate(histories):
        key = f'history-{i}'; protocol['history'].append(key); protocol['assets'][key] = dict(path=path, sha256=file_hash(path))
    if heldout:
        protocol['assets']['heldout'] = dict(path=heldout, sha256=file_hash(heldout))
    atomic_json(prepared / 'PROTOCOL.json', protocol)
    # Retain the old fixed maps; add construction layouts with 15/30/60 s play on fresh seeds.
    maps = json.loads(Path(cohort_path).read_text())['maps']
    atomic_json(prepared / 'cohort.json', dict(maps=extended_cohort(maps)))
    setup = dict(seed=seed, variant=variant, sourceSHA256=file_hash(source_path), target=target, arguments=vars(args), createdUTC=utc_now(),
                 architecture=protocol['architecture'], sourceCohortSHA256=cohort_hash, cohortSHA256=file_hash(prepared / 'cohort.json'),
                 anchors=histories, heldout=heldout)
    for name in ['train_scaled.py', 'protocol.py']:
        copy_immutable(ROOT / 'training_v5' / name, prepared / 'controller-source' / name)
    atomic_json(prepared / 'SETUP.json', setup)
    return setup


def main(options):
    output = Path(options.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if options.stop_after_updates and not archive_exercised(options.stop_after_updates, SNAPSHOT_EVERY, 1, ARCHIVE_LIMIT):
        raise ValueError('Pilots must run long enough to exceed the opponent archive limit, otherwise the archive ablation is inert')
    setup = prepare(Path(options.source).resolve(), output, options.target, getattr(options, 'seed', 1091252), getattr(options, 'variant', 'full'),
                    cohort_path=Path(options.cohort).resolve(), history_paths=getattr(options, 'history', None), heldout_path=getattr(options, 'heldout', None))
    args = argparse.Namespace(**setup['arguments'])
    args.graceful_worker_signals = True
    args.stop_after_updates = getattr(options, 'stop_after_updates', None)
    args.training_description = 'Relational attention; 10 object slots; 30 rays; visibility-only reward; refreshed self-play. V5.1 balanced active samples, 15/30/60s randomized play, diverse opponent archive, 128-step sequences with 32-step burn-in through the previous rollout; 0.1 tool entropy scale; bounded grounded jump; 2.2m walls.'
    status_path = output / 'STATUS.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else dict(
        createdUTC=utc_now(), evaluated=[], snapshots=[], architecture=setup['architecture'],
        targetInteractions=options.target, publication='No automatic browser promotion')
    stopped = False; child = None
    def save(**changes):
        status.update(changes, pid=os.getpid(), updatedUTC=utc_now()); atomic_json(status_path, status)
    def stop(sig, _):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None: child.send_signal(sig)
    for sig in [signal.SIGINT, signal.SIGTERM]: signal.signal(sig, stop)
    heldout = output / 'prepared/heldout.pt'
    def evaluate(path, steps):
        nonlocal child
        destination = output / 'evaluations' / str(steps)
        if steps in status['evaluated']: return
        if not (destination / 'evaluation.json').exists():
            command = [sys.executable, str(ROOT / 'training_v5/evaluate_saved.py'),
                '--checkpoint', str(path), '--reference', str(output / 'prepared/reference.pt'),
                '--cohort', str(output / 'prepared/cohort.json'), '--output', str(destination),
                '--registry', str(output / 'best'), '--workers', '4']
            if heldout.exists(): command += ['--heldout', str(heldout)]
            save(phase='evaluating', evaluationSteps=steps)
            with (output / f'evaluate-{steps}.log').open('a') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                code = child.wait(); child = None
            if stopped: return
            if code: raise RuntimeError(f'Evaluation failed at {steps}; inspect evaluation log')
        # Registration is idempotent, including a restart between report and registry writes.
        register(path, output / 'prepared/reference.pt', destination / 'evaluation.json', output / 'best')
        status['evaluated'].append(steps); save(phase='training')
    def checkpoint(saved, directory):
        steps = saved['totalPolicyInteractions']
        # Every ~1M interactions, plus first/last/paused: immutable full state.
        pilot_end = bool(args.stop_after_updates and saved['pilotUpdates'] >= args.stop_after_updates)
        retain = pilot_end or steps % 1048576 == 0 or steps == 65536 or steps >= options.target or stopped
        if retain:
            path = output / 'checkpoints' / f'{steps}.pt'
            digest = copy_immutable(directory / 'latest.pt', path)
            if steps not in status['snapshots']: status['snapshots'].append(steps)
            save(latestSnapshot=dict(steps=steps, path=str(path), sha256=digest))
        save(phase='training', interactions=steps, trainingSeconds=saved['seconds'],
             currentPolicyDecisions=saved['currentPolicyDecisions'], activePolicySamples=saved['activePolicySamples'],
             lastUpdate=saved['log'][-1])
        if not stopped and (pilot_end or steps == 65536 or steps % 5242880 == 0 or steps >= options.target):
            evaluate(output / 'checkpoints' / f'{steps}.pt', steps)
    try:
        latest = output / 'run/latest.pt'
        if latest.exists():
            args.resume = str(latest)
            # Re-run a pending complete milestone evaluation before further PPO.
            saved = torch.load(latest, map_location='cpu', weights_only=False)
            checkpoint(saved, latest.parent)
        if not stopped:
            save(phase='training'); train(args, stop_requested=lambda: stopped, on_checkpoint=checkpoint)
        save(phase='paused' if stopped else 'pilot-complete' if args.stop_after_updates else 'completed-awaiting-review')
    except BaseException as error:
        save(phase='failed', error=repr(error)); raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=1091252); p.add_argument('--variant', default='full', choices=VARIANTS)
    p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    p.add_argument('--cohort', required=True, help='JSON with the fixed development maps (the earlier cohort.json)')
    p.add_argument('--history', nargs='*', default=None, help='Explicit frozen opponent checkpoints; default derives them from the source and requires them to exist')
    p.add_argument('--heldout', default=None, help='Held-out evaluation opponent never placed in the league')
    p.add_argument('--stop-after-updates', type=int); p.add_argument('--target', type=int, default=1048576000)
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    if a.target < 65536 or a.target % 65536: a.error('Use complete 65,536-interaction PPO batches')
    main(a)
