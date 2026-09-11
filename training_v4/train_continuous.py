"""Extendable fresh-rollout training with graceful stop and durable archives.

This driver never selects a model by training reward or changes the browser's
accepted reference. It reuses the reviewed original PPO implementation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal

import torch
from persistent_train import file_hash
from train_league import train


DEFAULTS = dict(envs=128, workers=8, horizon=256, sequence_length=32,
                sequence_batch=256, critic_batch_size=2048, epochs=2,
                learning_rate=.0001, critic_learning_rate=.0001,
                entropy=.005, kl_limit=.008, seed=773119, arm='A')


def copy_immutable(source, destination):
    destination = Path(destination)
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ValueError(f'Refusing to overwrite an immutable checkpoint: {destination}')
    else:
        shutil.copyfile(source, destination)


def run(options):
    original_directory = Path.cwd()
    resume_path = Path(options.resume).resolve() if options.resume else None
    if resume_path and (resume_path.parent / 'MANIFEST.json').exists():
        manifest = json.loads((resume_path.parent / 'MANIFEST.json').read_text())
        if manifest.get('format') == 'hide-seek-native-resume-bundle-v1':
            if manifest['resumeFile'] != resume_path.name:
                raise ValueError('Resume the checkpoint named by the bundle manifest')
            for relative, record in manifest['files'].items():
                asset_path = (resume_path.parent / relative).resolve()
                if not asset_path.is_relative_to(resume_path.parent) or file_hash(asset_path) != record['sha256']:
                    raise ValueError(f'Portable bundle integrity check failed: {relative}')
    resumed = torch.load(options.resume, map_location='cpu', weights_only=False) if options.resume else None
    settings = dict(DEFAULTS)
    if resumed:
        settings.update(resumed['arguments'])
    for field, value in vars(options).items():
        if value is not None and field not in ['steps', 'until_stop', 'archive_every']:
            settings[field] = value
    if not all(settings.get(field) for field in ['parent', 'critic', 'initial', 'history', 'output']):
        raise ValueError('A new run needs --parent, --critic, --initial, --history and --output; resume inherits them')
    if resumed and not options.output:
        if resumed['arguments'].get('portableBundle'):
            raise ValueError('Choose --output for a new run; keep the distributed bundle immutable')
        destination = resume_path.parent
    else:
        destination = Path(settings['output']).resolve()
    if resumed and not resumed['arguments'].get('continuation') and destination == Path(options.resume).resolve().parent:
        raise ValueError('Preserve the fixed experiment; continue it in a new --output directory')
    destination.mkdir(parents=True, exist_ok=True)
    if not resumed and (destination / 'latest.pt').exists():
        raise ValueError('Use --resume to extend an existing run')
    assets = destination / 'assets'
    assets.mkdir(exist_ok=True)

    def portable_asset(value, explicit=False):
        source = Path(value)
        if not source.is_absolute():
            source = (original_directory if explicit or not resumed else resume_path.parent) / source
        source = source.resolve()
        name = f'{file_hash(source)[:16]}.pt'
        copy_immutable(source, assets / name)
        return f'assets/{name}'

    for field in ['parent', 'critic', 'initial']:
        settings[field] = portable_asset(settings[field], explicit=getattr(options, field) is not None)
    settings['history'] = [portable_asset(path, explicit=options.history is not None) for path in settings['history']]
    if resume_path:
        saved_sources = destination / 'resume-inputs'
        saved_sources.mkdir(exist_ok=True)
        name = f'{file_hash(resume_path)[:16]}.pt'
        copy_immutable(resume_path, saved_sources / name)
        settings['resume'] = f'resume-inputs/{name}'
    else:
        settings['resume'] = None
    settings['output'] = '.'
    settings.pop('portableBundle', None)
    block = settings['envs'] * settings['horizon'] * 2
    before = resumed['totalPolicyInteractions'] if resumed else 0
    if options.until_stop:
        # A safely serializable integer ceiling exceeds 28,000 years at 10,000 decisions/s.
        # The operational stop condition is the user's signal, not this bound.
        target = ((2**53 - 1) // block) * block
        additional = None
    else:
        additional = (options.steps // block) * block
        if additional < block:
            raise ValueError(f'At least one fresh rollout ({block} actor decisions) is required')
        target = before + additional
    settings.update(target_interactions=target, stop_after_updates=None,
                    retain_updates='', save_every=1, graceful_worker_signals=True)
    wrapper_hash = file_hash(__file__)
    source_dir = destination / 'continuation-source'
    source_dir.mkdir(exist_ok=True)
    copy_immutable(__file__, source_dir / f'{wrapper_hash}.py')
    copy_immutable(destination / settings['parent'], destination / 'best-reference.pt')
    settings['continuation'] = dict(wrapperSHA256=wrapper_hash,
        mode='until-stop' if options.until_stop else 'additional-steps',
        requestedAdditionalInteractions=options.steps, plannedAdditionalInteractions=additional,
        startedAtInteractions=before, archiveEveryRollouts=options.archive_every,
        bestReferenceSHA256=file_hash(destination / 'best-reference.pt'),
        selection='best-reference is the frozen parent comparison; latest and archives are unqualified training progress. Acceptance requires a separate evaluation.',
        stopSignal=None)
    args = argparse.Namespace(**settings)
    requested = False

    def stop(signum, _frame):
        nonlocal requested
        requested = True
        args.continuation['stopSignal'] = signal.Signals(signum).name
        print('Stop requested. Finishing the current rollout/update, then saving all training state.', flush=True)

    def archive(saved, output):
        final = saved['totalPolicyInteractions'] >= target
        if requested or final or saved['pilotUpdates'] % options.archive_every == 0:
            folder = output / 'checkpoints'
            folder.mkdir(exist_ok=True)
            copy_immutable(output / 'latest.pt', folder / f'{saved["totalPolicyInteractions"]}.pt')

    old_handlers = {sig: signal.getsignal(sig) for sig in [signal.SIGINT, signal.SIGTERM]}
    for sig in old_handlers:
        signal.signal(sig, stop)
    print(json.dumps(dict(mode=args.continuation['mode'], resumeInteractions=before,
                          freshRolloutActorDecisions=block, plannedAdditionalInteractions=additional,
                          output=str(destination), stop='Ctrl-C or SIGTERM saves at the next completed update')), flush=True)
    try:
        # Dependencies are stored relative to the run directory. A copied run
        # remains resumable after all original machine paths disappear.
        os.chdir(destination)
        train(args, stop_requested=lambda: requested, on_checkpoint=archive)
    finally:
        os.chdir(original_directory)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    latest = torch.load(destination / 'latest.pt', map_location='cpu', weights_only=False)
    print(json.dumps(dict(status='saved', latest=str(destination / 'latest.pt'),
        totalPolicyInteractions=latest['totalPolicyInteractions'],
        currentPolicyDecisions=latest['currentPolicyDecisions'],
        historicalPolicyDecisions=latest['historicalPolicyDecisions'],
        acceptedReferenceChanged=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ['resume', 'parent', 'critic', 'initial', 'output']:
        parser.add_argument('--' + field)
    parser.add_argument('--history', nargs='+')
    parser.add_argument('--arm', choices=['A', 'B'])
    for field in ['envs', 'workers', 'horizon', 'sequence-length', 'sequence-batch',
                  'critic-batch-size', 'epochs', 'seed']:
        parser.add_argument('--' + field, type=int)
    for field in ['learning-rate', 'critic-learning-rate', 'entropy', 'kl-limit']:
        parser.add_argument('--' + field, type=float)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--steps', type=int, help='Additional actor decisions, rounded down to complete fresh rollouts')
    budget.add_argument('--until-stop', action='store_true')
    parser.add_argument('--archive-every', type=int, default=16, help='Immutable checkpoint interval in completed rollouts')
    options = parser.parse_args()
    if options.archive_every < 1:
        parser.error('--archive-every must be positive')
    run(options)
