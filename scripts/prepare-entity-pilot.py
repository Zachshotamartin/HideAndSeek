"""Create immutable zero-update encoder/reference snapshots with matched Adam reset."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
from entity_actor import EntityActor, FORMAT, ENTITY, prepare_pair
from persistent_train import file_hash


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'MANIFEST.json').exists():
        raise ValueError('Preserve each prepared comparison')
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    physics_hash = file_hash(ROOT / 'training/physics.py')
    if parent['provenance']['physicsSHA256'] != physics_hash:
        raise ValueError('The native physical game must remain unchanged')
    files = {}
    for name, projected in [('legacy', False), ('entity', True)]:
        record = prepare_pair(parent, projected=projected, seed=args.seed)
        record['provenance'].update(sourceSelectedPairSHA256=file_hash(args.parent),
                                    sourceFiles={name: file_hash(ROOT / name) for name in
                                                 ['training/entity_actor.py', 'training/persistent_actor.py', 'training/physics.py']})
        path = output / f'{name}.pt'
        torch.save(record, path)
        files[name] = dict(file=path.name, sha256=file_hash(path), bytes=path.stat().st_size,
                           newActorUpdates=0, emptyOptimizerStates=all(not state['state'] for state in record['optimizers']))
    with torch.random.fork_rng():
        torch.manual_seed(args.seed + 1)
        initial = [EntityActor(), EntityActor()]
        record = dict(format=FORMAT, encoderTypes=[ENTITY, ENTITY],
            models=[actor.state_dict() for actor in initial], observationSize=140, physicsObservationSize=138,
            optimizers=[torch.optim.Adam(actor.parameters(), lr=.0001, eps=1e-5).state_dict() for actor in initial],
            torchRNG=torch.get_rng_state(), decisions=0, parentDecisions=0, pilotDecisions=0,
            newActorUpdates=0, provenance=dict(physicsSHA256=physics_hash,
                initializationSeed=args.seed + 1, description='Actual zero-experience entity architecture, distinct from the projected warm start'))
        path = output / 'initial.pt'; torch.save(record, path)
        files['initial'] = dict(file=path.name, sha256=file_hash(path), bytes=path.stat().st_size,
                                newActorUpdates=0, emptyOptimizerStates=True)
    manifest = dict(format='isolated-entity-encoder-zero-update-v1',
        selectedParentSHA256=file_hash(args.parent), physicsSHA256=physics_hash,
        ownProjectionSeed=args.seed, files=files, sourceSHA256=file_hash(__file__),
        publicAssetsChanged=False, optimizerResetBothArms=True,
        status='Untrained representation change on our learned parameters; not a promoted model')
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=785611)
    main(parser.parse_args())
