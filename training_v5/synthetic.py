"""Small self-contained source pairs and cohorts so tests run from a fresh clone.

Nothing here is trained; the records only carry the schema that the trainer,
the controller and the evaluator require.
"""
import json
from pathlib import Path
import torch
from entity_actor import EntityActor, FORMAT, ENTITY
from residual_critic import ResidualCentralCritic, SCHEMA
from persistent_train import file_hash


def synthetic_source(path, hidden=16, encoder=32, embedding=8, seed=3):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        models = [EntityActor(hidden, encoder, embedding) for _ in range(2)]
        critic = ResidualCentralCritic(.1, memory_size=hidden)
        rng = torch.get_rng_state()
    physics = file_hash(Path(__file__).with_name('physics.py'))
    record = dict(format=FORMAT, encoderTypes=[ENTITY, ENTITY], observationSize=210, physicsObservationSize=208,
        models=[m.state_dict() for m in models], optimizers=[torch.optim.Adam(m.parameters()).state_dict() for m in models],
        centralCritic=critic.state_dict(), centralCriticOptimizer=torch.optim.AdamW(critic.parameters()).state_dict(),
        centralCriticSchema=SCHEMA, decisions=0, totalPolicyInteractions=0, pilotUpdates=0, seconds=0., newActorUpdates=0,
        provenance=dict(physicsSHA256=physics, roleSources=[], inheritedRoleSources=[], sourceSelectedPairSHA256='synthetic',
            encoderPreparation=dict(description='synthetic zero-experience pair for tests', optimizerReset=True), history=[]),
        arguments=dict(history=[]), log=[], torchRNG=rng, worldRNG=None, opponentRNG=None)
    torch.save(record, path)
    return Path(path)


def legacy_movement_source(path, hidden=16, encoder=32, embedding=8, seed=4):
    """A pre-jump relational pair: three movement outputs, otherwise identical."""
    source = torch.load(synthetic_source(path, hidden, encoder, embedding, seed), map_location='cpu', weights_only=False)
    for state in source['models']:
        for key in ['movement.weight', 'movement.bias', 'log_std']:
            state[key] = state[key][:3].clone()
    torch.save(source, path)
    return Path(path)


def synthetic_cohort(path, count=2):
    maps = [dict(seed=1500900000 + i, scenario=['open', 'rooms'][i % 2], arenaConfig=dict(size=8, n_boxes=2, n_ramps=1)) for i in range(count)]
    Path(path).write_text(json.dumps(dict(maps=maps)) + '\n')
    return Path(path)


def prepared_setup(folder, target=512, seed=1, variant='full'):
    """Run the real controller preparation on a synthetic source in `folder`."""
    from train_scaled import prepare
    folder = Path(folder)
    source = synthetic_source(folder / 'source.pt')
    cohort = synthetic_cohort(folder / 'cohort.json')
    return prepare(source, folder / 'run', target, seed, variant, cohort_path=cohort, history_paths=[])
