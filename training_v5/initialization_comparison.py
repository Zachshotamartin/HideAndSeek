"""Fresh versus inherited actors with matched fresh critics, optimizers and league anchors."""
from pathlib import Path
import torch
from checkpoint_store import atomic_json
from entity_actor import load_pair
from persistent_train import file_hash
from residual_critic import ResidualCentralCritic, SCHEMA


def prepare_initialization(config, mode, seed):
    if mode not in ('fresh','continued'):
        raise ValueError('Unknown initialization arm')
    prepared = Path(config['parent']).parent
    identity_path = prepared / 'INITIALIZATION.json'
    expected = dict(mode=mode, seed=seed, originalParentSHA256=file_hash(config['parent']),
                    criticInitialization='fresh in both arms', optimizerInitialization='fresh in both arms',
                    league='same inherited frozen anchors in both arms; fresh is not training without experienced opponents')
    if identity_path.exists():
        import json
        previous = json.loads(identity_path.read_text())
        if any(previous.get(k) != v for k,v in expected.items()):
            raise ValueError('Initialization comparison identity changed')
        for key in ('parent','critic'):
            path = prepared / previous[key]['name']
            if file_hash(path) != previous[key]['sha256']:
                raise ValueError('Prepared initialization checkpoint changed')
            config[key] = str(path)
        return
    parent = torch.load(config['parent'],map_location='cpu',weights_only=False)
    actors = load_pair(parent)[0]
    if mode == 'fresh':
        initial = torch.load(config['initial'],map_location='cpu',weights_only=False)
        parent['models'] = initial['models']
        for key in ('decisions','parentDecisions','totalPolicyInteractions','pilotDecisions','pilotUpdates','newActorUpdates'):
            parent[key] = 0
        parent['provenance']['roleSources'] = []
        parent['provenance']['inheritedRoleSources'] = []
        parent['provenance']['encoderPreparation'] = dict(description='Fresh random actors; no inherited actor weights',optimizerReset=True)
    parent['provenance']['initializationComparison'] = expected
    with torch.random.fork_rng():
        torch.manual_seed(seed + 419)
        critic = ResidualCentralCritic(.1, actors[0].hidden_size)
    parent['centralCritic'] = critic.state_dict()
    parent['centralCriticSchema'] = SCHEMA
    # Rollout RNG and frozen league files stay identical between matched arms.
    critic_record = dict(centralCritic=critic.state_dict(), centralCriticSchema=SCHEMA,
                         provenance=parent['provenance'])
    parent_path, critic_path = prepared / 'comparison-parent.pt', prepared / 'comparison-critic.pt'
    torch.save(parent,parent_path);torch.save(critic_record,critic_path)
    for key,path in [('parent',parent_path),('critic',critic_path)]:
        config[key] = str(path)
        expected[key] = dict(name=path.name,sha256=file_hash(path))
    atomic_json(identity_path,expected)
