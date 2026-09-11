"""Portable extension of saved native actor pairs, outside fixed experiments.

The existing trainers remain unchanged. Entity extensions receive a new,
explicit continuation protocol linked to the immutable original protocol and
checkpoint. Only paths/budget/provenance metadata is adapted before resuming.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import uuid

import torch

from checkpoint_store import atomic_json, copy_immutable, read_native, stage_dependencies, sync_file, utc_now
from entity_actor import FORMAT as ENTITY_FORMAT
from persistent_actor import FORMAT as PERSISTENT_FORMAT
from persistent_train import file_hash
from train_entity import train as train_entity
from train_league import train as train_legacy


def _run(options, locks):
    source = Path(options.resume).resolve()
    saved, source_hash = read_native(source)
    if saved['format'] not in [ENTITY_FORMAT, PERSISTENT_FORMAT]:
        raise ValueError('This workflow supports our persistent and typed entity native pairs')
    original_directory = Path.cwd()
    prior_settings = copy.deepcopy(saved['arguments'])
    old_managed = prior_settings.get('savedRun', {})
    destination = Path(options.output).resolve() if options.output else None
    if destination is None and old_managed:
        for root in source.parents:
            if (root / 'RUN.json').is_file():
                record = json.loads((root / 'RUN.json').read_text())
                if record.get('id') == old_managed.get('id'):
                    destination = root
                    break
    if destination is None:
        raise ValueError('Use --output to continue a fixed experiment or distributed bundle in a new directory')
    destination.mkdir(parents=True, exist_ok=True)
    lock = open(destination / '.training.lock', 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('Another trainer is writing this run; use a separate output branch') from None
    locks.append(lock)
    lock.seek(0)
    lock.truncate()
    lock.write(json.dumps(dict(pid=os.getpid(), acquiredUTC=utc_now())) + '\n')
    lock.flush()
    run_path = destination / 'RUN.json'
    existing = json.loads(run_path.read_text()) if run_path.exists() else None
    if not existing and (destination / 'latest.pt').exists():
        raise ValueError('Keep the existing experiment intact; choose a new --output directory')
    if existing and (destination / 'latest.pt').exists():
        if file_hash(destination / 'latest.pt') != source_hash:
            raise ValueError('Resuming an older snapshot/best model requires a new --output branch')
    identifier = existing['id'] if existing else uuid.uuid4().hex
    record = existing or dict(format='hide-seek-managed-native-run-v1', id=identifier,
        createdUTC=utc_now(), initialCheckpointSHA256=source_hash, sessions=[], snapshots=[],
        bestSelection='Separate explicit fixed-opponent evaluation registry; training return never selects best.')
    settings = copy.deepcopy(prior_settings)
    block = settings['envs'] * settings['horizon'] * 2
    before = saved['totalPolicyInteractions']
    if options.until_stop:
        target = ((2**53 - 1) // block) * block
        additional = None
    else:
        additional = (options.steps // block) * block
        if additional < block:
            raise ValueError(f'At least {block} requested interactions are needed for one complete rollout')
        target = before + additional
    archive_interval = max(block, ((options.snapshot_every + block - 1) // block) * block)
    if options.snapshot_every < 1:
        raise ValueError('Snapshot interval must be positive')
    started = time.monotonic()
    assets = stage_dependencies(source, saved, destination)
    if copy_immutable(source, destination / 'resume-inputs' / (source_hash + '.pt')) != source_hash:
        raise ValueError('The source checkpoint changed; retry with its immutable step snapshot')
    if not existing:
        copy_immutable(source, destination / 'best-reference.pt')
    settings.update(assets)
    settings.update(output='.', target_interactions=target, retain_updates='', save_every=1,
                    stop_after_updates=None, graceful_worker_signals=True)
    session = dict(id=uuid.uuid4().hex, sourceCheckpointSHA256=source_hash,
        startUTC=utc_now(), startedAtInteractions=before, requestedAdditionalInteractions=options.steps,
        plannedAdditionalInteractions=additional, targetInteractions=target,
        requestedSnapshotInterval=options.snapshot_every, actualSnapshotInterval=archive_interval,
        inheritedTrainingSeconds=float(saved['seconds']), stopSignal=None,
        resumeSemantics='Actor/optimizer/critic/RNG state retained; in-flight worlds, memories and buttons restart together.')
    settings['savedRun'] = dict(id=identifier, sessionID=session['id'], driverSHA256=file_hash(__file__))
    adapted = copy.deepcopy(saved)
    if saved['format'] == ENTITY_FORMAT:
        original_protocol = json.loads((destination / assets['protocol']).read_text())
        protocol = copy.deepcopy(original_protocol)
        protocol['format'] = 'hide-seek-entity-continuation-protocol-v1'
        protocol['training'] = {name: settings[name] for name in original_protocol['training']}
        protocol['training']['target_interactions'] = target
        protocol['extension'] = dict(sourceCheckpointSHA256=source_hash,
            sourceProtocolSHA256=saved['provenance']['protocolSHA256'],
            originalTarget=original_protocol['training']['target_interactions'],
            newTarget=target, weightsAndOptimizersChanged=False,
            reason='Explicit user-requested continuation; the original matched experiment remains unchanged.')
        protocol['assets'] = {}
        for name, key in [('parent', settings['encoder']), ('critic', 'critic'), ('initial', 'initial')]:
            protocol['assets'][key] = dict(path=settings[name], sha256=file_hash(destination / settings[name]))
        protocol['history'] = []
        for index, relative in enumerate(settings['history']):
            key = f'history-{index}'
            protocol['history'].append(key)
            protocol['assets'][key] = dict(path=relative, sha256=file_hash(destination / relative))
        protocol_path = destination / 'protocols' / (session['id'] + '.json')
        atomic_json(protocol_path, protocol)
        settings['protocol'] = str(protocol_path.relative_to(destination))
        adapted['provenance'].setdefault('continuations', []).append(protocol['extension'])
        adapted['provenance']['protocolSHA256'] = file_hash(protocol_path)
    # Preserve source bytes above. This derived input changes only administrative
    # metadata; the next actual rollout/update is performed by the old trainer.
    adapted['arguments'] = copy.deepcopy(settings)
    adapted_path = destination / 'resume-inputs' / (session['id'] + '-adapted.pt')
    torch.save(adapted, adapted_path)
    sync_file(adapted_path)
    settings['resume'] = str(adapted_path.relative_to(destination))
    session['derivedResumeSHA256'] = file_hash(adapted_path)
    record['sessions'].append(session)
    record['status'] = 'starting'
    atomic_json(run_path, record)
    source_folder = destination / 'managed-source' / session['id']
    for name in ['train_saved.py', 'checkpoint_store.py']:
        copy_immutable(Path(__file__).with_name(name), source_folder / name)
    requested = False

    def stop(signum, _frame):
        nonlocal requested
        requested = True
        session['stopSignal'] = signal.Signals(signum).name
        print('Stop requested; finish the current rollout/update and save.', flush=True)

    def checkpoint(current, output):
        sync_file(output / 'latest.pt')
        steps = current['totalPolicyInteractions']
        final = requested or steps >= target
        if final or steps % archive_interval == 0:
            relative = f'checkpoints/{steps}.pt'
            digest = copy_immutable(output / 'latest.pt', output / relative)
            if not any(row['file'] == relative for row in record['snapshots']):
                record['snapshots'].append(dict(file=relative, sha256=digest,
                    interactions=steps, trainingSeconds=float(current['seconds']),
                    savedUTC=utc_now(), sessionID=session['id']))
        session.update(lastInteractions=steps, cumulativeTrainingSeconds=float(current['seconds']),
            activeSessionWallSeconds=time.monotonic() - started,
            rolloutSecondsThisSession=sum(row['rolloutSeconds'] for row in current['log'][saved['pilotUpdates']:]),
            optimizerSecondsThisSession=sum(row['optimizerSeconds'] for row in current['log'][saved['pilotUpdates']:]))
        record.update(status='saved-stopping' if final else 'training',
            latest=dict(file='latest.pt', sha256=file_hash(output / 'latest.pt'),
                interactions=steps, trainingSeconds=float(current['seconds']), savedUTC=utc_now()))
        atomic_json(output / 'RUN.json', record)

    previous = {sig: signal.getsignal(sig) for sig in [signal.SIGINT, signal.SIGTERM]}
    for sig in previous:
        signal.signal(sig, stop)
    print(json.dumps(dict(output=str(destination), resumeInteractions=before,
        plannedAdditionalInteractions=additional, actualSnapshotInterval=archive_interval,
        mode='until-stop' if options.until_stop else 'additional-interactions')), flush=True)
    try:
        os.chdir(destination)
        trainer = train_entity if saved['format'] == ENTITY_FORMAT else train_legacy
        trainer(argparse.Namespace(**settings), stop_requested=lambda: requested, on_checkpoint=checkpoint)
        record['status'] = 'paused' if requested else 'completed-requested-extension'
    except BaseException as error:
        record.update(status='interrupted-by-error', error=repr(error))
        raise
    finally:
        os.chdir(original_directory)
        session.update(endUTC=utc_now(), activeSessionWallSeconds=time.monotonic() - started)
        atomic_json(run_path, record)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run(options):
    locks = []
    try:
        return _run(options, locks)
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--output')
    parser.add_argument('--snapshot-every', type=int, default=1048576, help='Actor interactions, rounded up to a complete rollout')
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--steps', type=int, help='Additional fresh interactions, rounded down to complete rollouts')
    budget.add_argument('--until-stop', action='store_true')
    run(parser.parse_args())
