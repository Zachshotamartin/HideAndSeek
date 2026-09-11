"""Prepare an original mixed-role continuation without changing either actor.

The hider and seeker, including their Adam moments, are selected by role from
explicit source checkpoints. The existing league critic is inherited as a
training initialization, not represented as a new fit on the mixed pair.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
from persistent_actor import FORMAT
from persistent_train import file_hash
from residual_critic import SCHEMA
from train_continuous import DEFAULTS


def prepare(args):
    output = Path(args.output).resolve()
    if (output / 'resume.pt').exists():
        raise ValueError('Keep each prepared phase immutable')
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)
    hider, seeker, mixed, fitted, initial = [
        torch.load(path, map_location='cpu', weights_only=False)
        for path in [args.hider, args.seeker, args.mixed, args.critic, args.initial]
    ]
    physics_hash = file_hash(ROOT / 'training/physics.py')
    if initial['decisions'] != 0 or initial['format'] != FORMAT:
        raise ValueError('Use the genuine zero-experience persistent reference')
    for record in [hider, seeker, mixed, initial]:
        if record['format'] != FORMAT or record['provenance']['physicsSHA256'] != physics_hash:
            raise ValueError('Every source must use the same frozen physical game')
    source_hashes = [file_hash(args.hider), file_hash(args.seeker)]
    if [row['sourceCheckpointSHA256'] for row in mixed['provenance']['roleSources']] != source_hashes:
        raise ValueError('The evaluated mixed pair must contain these exact role sources')
    for role, source in enumerate([hider, seeker]):
        for name, value in source['models'][role].items():
            torch.testing.assert_close(value, mixed['models'][role][name], atol=0, rtol=0)
    if fitted['format'] != SCHEMA or fitted['parentSHA256'] != source_hashes[0]:
        raise ValueError('The critic architecture reference must belong to the frozen hider parent')

    copied = {}
    def asset(path):
        digest = file_hash(path)
        relative = f'assets/{digest[:16]}.pt'
        if digest not in copied:
            shutil.copyfile(path, output / relative)
            copied[digest] = relative
        return relative

    # Keep the assessed file untouched. This native parent adds only the Torch
    # RNG needed by the trainer, explicitly inherited from the seeker phase.
    parent = copy.deepcopy(mixed)
    parent['torchRNG'] = seeker['torchRNG'].clone()
    parent['provenance']['assessedMixedCheckpointSHA256'] = file_hash(args.mixed)
    parent['provenance']['torchRNGSourceSHA256'] = source_hashes[1]
    parent['status'] = 'Frozen mixed-role initialization for a longer development continuation'
    torch.save(parent, output / 'mixed-parent.pt')
    parent_hash = file_hash(output / 'mixed-parent.pt')

    critic_initialization = dict(
        kind='Inherited existing central critic and optimizer; no new value fitting',
        sourceCheckpointSHA256=source_hashes[1], fittedOnMixedParent=False,
        newCriticWarmupDecisions=0,
        originalArchitectureReferenceSHA256=file_hash(args.critic))
    critic = dict(format=SCHEMA, parentSHA256=parent_hash, dropout=fitted['dropout'],
        critic=copy.deepcopy(seeker['centralCritic']),
        criticOptimizer=copy.deepcopy(seeker['centralCriticOptimizer']),
        initialization=critic_initialization,
        report=dict(environmentActorDecisions=dict(train=0)))
    torch.save(critic, output / 'critic-initialization.pt')
    history = [torch.load(path, map_location='cpu', weights_only=False) for path in args.history]
    if any(record['format'] != FORMAT or record['provenance']['physicsSHA256'] != physics_hash for record in history):
        raise ValueError('Historical opponents must use the unchanged physical game')
    settings = dict(DEFAULTS, arm='B', seed=args.seed,
        parent=asset(output / 'mixed-parent.pt'), critic=asset(output / 'critic-initialization.pt'),
        initial=asset(args.initial), history=[asset(path) for path in args.history],
        output='.', resume=None, target_interactions=0, stop_after_updates=None,
        retain_updates='', save_every=1, graceful_worker_signals=True,
        portableBundle=True, continuation=dict(prepared=True),
        training_description='Sustained mixed-role historical-opponent continuation; original visibility reward only')
    optimizers = copy.deepcopy(mixed['optimizers'])
    for optimizer in optimizers:
        for group in optimizer['param_groups']:
            group['lr'] = settings['learning_rate']
    provenance = dict(parentSHA256=parent_hash, parentDecisions=mixed['decisions'],
        parentDecisionsDefinition=mixed['decisionsDefinition'],
        inheritedRoleSources=copy.deepcopy(mixed['provenance']['roleSources']),
        assessedMixedCheckpointSHA256=file_hash(args.mixed),
        initialSHA256=file_hash(args.initial), physicsSHA256=physics_hash,
        criticFitSHA256=file_hash(output / 'critic-initialization.pt'),
        criticInitialization=critic_initialization, criticWarmupDecisions=0,
        criticHashFieldNote='criticFitSHA256 is the legacy asset-identity field; this phase inherits a trained critic rather than claiming a new fit on its mixed parent.',
        history=[dict(file=relative, sha256=file_hash(path), decisions=record['decisions'])
                 for relative, path, record in zip(settings['history'], args.history, history)],
        preparation='Zero new actor updates. Exact selected role weights and Adam moments; Torch RNG inherited from the seeker source. Fresh world and opponent streams begin at the documented seed.',
        actorInput='Unchanged 140 restricted observations; centralized state is training-only',
        rule='Only original zero-sum visibility reward; unchanged physical grab/lock semantics',
        selection='Retained per-role references stay frozen; latest checkpoints require separate broad assessment',
        sourceHistory=[], resumes=[])
    saved = dict(format=FORMAT, observationSize=140, physicsObservationSize=138,
        trainingMethod=settings['training_description'], models=copy.deepcopy(mixed['models']),
        optimizers=optimizers, centralCritic=critic['critic'], centralCriticOptimizer=critic['criticOptimizer'],
        centralCriticSchema=SCHEMA, parentDecisions=mixed['decisions'], criticWarmupDecisions=0,
        pilotDecisions=0, totalPolicyInteractions=0, actorUpdateDecisions=0,
        currentPolicyDecisions=[0, 0], historicalPolicyDecisions=[0, 0], activePolicySamples=[0, 0],
        decisions=mixed['decisions'], pilotUpdates=0, pilotEpisodes=0, seconds=0,
        provenance=provenance, arguments=settings, log=[], torchRNG=parent['torchRNG'],
        worldRNG=np.random.default_rng(args.seed).bit_generator.state,
        opponentRNG=np.random.default_rng(args.seed + 1).bit_generator.state)
    torch.save(saved, output / 'resume.pt')
    reloaded = torch.load(output / 'resume.pt', map_location='cpu', weights_only=False)
    for role, source in enumerate([hider, seeker]):
        for name, value in source['models'][role].items():
            torch.testing.assert_close(value, reloaded['models'][role][name], atol=0, rtol=0)
    manifest = dict(format='hide-seek-native-resume-bundle-v1', resumeFile='resume.pt',
        phase='Sustained mixed-role continuation', newActorUpdates=0,
        inheritedRoleSources=provenance['inheritedRoleSources'],
        inheritedUnionSourceBudget=mixed['decisions'], criticInitialization=critic_initialization,
        seed=args.seed, physicsSHA256=physics_hash,
        files={str(path.relative_to(output)): dict(sha256=file_hash(path), bytes=path.stat().st_size)
               for path in sorted(output.rglob('*.pt'))},
        preparationSourceSHA256=file_hash(__file__))
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copyfile(__file__, output / 'prepare-source.py')
    print(json.dumps(dict(resume=str(output / 'resume.pt'), sha256=file_hash(output / 'resume.pt'),
                         newActorUpdates=0, physicsUnchanged=True), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['hider', 'seeker', 'mixed', 'critic', 'initial', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--history', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=774113)
    prepare(parser.parse_args())
