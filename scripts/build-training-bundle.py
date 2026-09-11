"""Package our native development parent for portable fresh-rollout continuation.

This prepares zero new actor updates. Model weights, optimizer moments and Torch
RNG come from our exact frozen parent; a separately assessed fitted critic is
included. It is not a model promotion or a reconstruction from browser weights.
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


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'assets').mkdir(exist_ok=True)
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    fitted = torch.load(args.critic, map_location='cpu', weights_only=False)
    initial = torch.load(args.initial, map_location='cpu', weights_only=False)
    parent_hash = file_hash(args.parent)
    physics_hash = file_hash(ROOT / 'training/physics.py')
    if parent['format'] != FORMAT or initial['format'] != FORMAT or initial['decisions'] != 0:
        raise ValueError('Compatible own persistent parent and genuine zero-experience initial required')
    if fitted['format'] != SCHEMA or fitted['parentSHA256'] != parent_hash:
        raise ValueError('Fitted critic must belong to the exact frozen parent')
    if parent['provenance']['physicsSHA256'] != physics_hash:
        raise ValueError('Frozen physics changed')
    files = {}
    by_hash = {}

    def asset(source, label):
        digest = file_hash(source)
        if digest in by_hash:
            return by_hash[digest]
        name = f'assets/{label}.pt'
        shutil.copyfile(source, output / name)
        files[name] = dict(sha256=digest, bytes=(output / name).stat().st_size)
        by_hash[digest] = name
        return name

    settings = dict(DEFAULTS, parent=asset(args.parent, 'parent'),
                    critic=asset(args.critic, 'critic'), initial=asset(args.initial, 'initial'),
                    history=[asset(path, f'history-{index}') for index, path in enumerate(args.history)],
                    output='.', resume=None, target_interactions=0, stop_after_updates=None,
                    retain_updates='', save_every=1, graceful_worker_signals=True,
                    portableBundle=True, continuation=dict(prepared=True))
    history = [torch.load(path, map_location='cpu', weights_only=False) for path in args.history]
    for record in history:
        if record['format'] != FORMAT or record['provenance']['physicsSHA256'] != physics_hash:
            raise ValueError('Historical models must use the same physical game')
    optimizers = copy.deepcopy(parent['optimizers'])
    for optimizer in optimizers:
        for group in optimizer['param_groups']:
            group['lr'] = settings['learning_rate']
    warmup = fitted['report']['environmentActorDecisions']['train']
    provenance = dict(parentSHA256=parent_hash, parentDecisions=parent['decisions'],
        initialSHA256=file_hash(args.initial), physicsSHA256=physics_hash,
        criticFitSHA256=file_hash(args.critic), criticWarmupDecisions=warmup,
        history=[dict(file=path, sha256=file_hash(original), decisions=record['decisions'])
                 for path, original, record in zip(settings['history'], args.history, history)],
        preparation='Zero new actor updates. Exact parent weights/moments/Torch RNG; new-phase world RNG seed 773119. Actor LR explicitly set to 0.0001. Fitted critic retained.',
        actorInput='Unchanged restricted 140 observations; no central state enters actor inference',
        rule='Only original zero-sum visibility reward; no tool or pursuit shaping',
        sourceHistory=[], resumes=[])
    saved = dict(format=FORMAT, observationSize=140, physicsObservationSize=138,
        trainingMethod='Prepared zero-update continuation of frozen 37M development parent',
        models=copy.deepcopy(parent['models']), optimizers=optimizers,
        centralCritic=copy.deepcopy(fitted['critic']), centralCriticOptimizer=copy.deepcopy(fitted['criticOptimizer']),
        centralCriticSchema=SCHEMA, parentDecisions=parent['decisions'],
        criticWarmupDecisions=warmup, pilotDecisions=0, totalPolicyInteractions=0,
        actorUpdateDecisions=0, currentPolicyDecisions=[0, 0], historicalPolicyDecisions=[0, 0],
        activePolicySamples=[0, 0], decisions=parent['decisions'] + warmup,
        pilotUpdates=0, pilotEpisodes=0, seconds=0, provenance=provenance, arguments=settings,
        log=[], torchRNG=parent['torchRNG'], worldRNG=np.random.default_rng(settings['seed']).bit_generator.state,
        opponentRNG=np.random.default_rng(settings['seed'] + 1).bit_generator.state)
    torch.save(saved, output / 'resume.pt')
    reloaded = torch.load(output / 'resume.pt', map_location='cpu', weights_only=False)
    for original, bundled in zip(parent['models'], reloaded['models']):
        for key in original:
            torch.testing.assert_close(original[key], bundled[key], atol=0, rtol=0)
    files['resume.pt'] = dict(sha256=file_hash(output / 'resume.pt'), bytes=(output / 'resume.pt').stat().st_size)
    manifest = dict(format='hide-seek-native-resume-bundle-v1',
        status='Frozen 37M development parent, not reliable useful-tool strategy or a new model promotion',
        parentActorLineageDecisions=parent['decisions'], criticOnlyWarmupDecisions=warmup,
        newActorUpdates=0, resumeFile='resume.pt', files=files,
        totalBytes=sum(entry['bytes'] for entry in files.values()),
        preparation=provenance['preparation'], physicsSHA256=physics_hash,
        command='npm run train:continuous -- --resume training/resume-bundle/resume.pt --output output/continuous-run --until-stop',
        sources={name: file_hash(ROOT / name) for name in
                 ['scripts/build-training-bundle.py', 'training/train_continuous.py', 'training/train_league.py',
                  'training/league_ppo.py', 'training/env_pool.py', 'training/persistent_actor.py',
                  'training/central_critic.py', 'training/residual_critic.py', 'training/physics.py']})
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ['parent', 'critic', 'initial', 'output']:
        parser.add_argument('--' + argument, required=True)
    parser.add_argument('--history', nargs='+', required=True)
    main(parser.parse_args())
