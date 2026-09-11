"""Explicit weights-only migration. Old experience is lineage, not v2 training."""
import copy
import torch
from entity_actor import EntityActor, FORMAT, ENTITY
from persistent_train import file_hash
from pathlib import Path

def migrate(source):
    models=[]
    for state in source['models']:
        h=state['memory.weight_hh'].shape[1];e=state['memory.weight_ih'].shape[1]
        model=EntityActor(h,e,state['encoder.embedding.0.weight'].shape[0])
        target=model.state_dict()
        # Same sensor/action/architecture contract; keep every learned tensor.
        for key,value in state.items():
            if key not in target or target[key].shape!=value.shape:raise ValueError('Incompatible v2 checkpoint: '+key)
            target[key]=value.clone()
        model.load_state_dict(target);models.append(model)
    result=copy.deepcopy(source)
    result.update(format=FORMAT,encoderTypes=[ENTITY]*2,observationSize=210,physicsObservationSize=208,
        models=[m.state_dict() for m in models],newActorUpdates=0,
        optimizers=[torch.optim.Adam(m.parameters(),lr=.0001,eps=1e-5).state_dict() for m in models],torchRNG=torch.get_rng_state())
    provenance=result['provenance'];provenance['sourcePhysicsSHA256']=provenance['physicsSHA256']
    provenance['physicsSHA256']=file_hash(Path(__file__).with_name('physics.py'))
    provenance['roleSources']=copy.deepcopy(provenance.get('inheritedRoleSources',[]))
    provenance['encoderPreparation']=dict(description='Explicit weights-only migration: expanded randomized wall layouts; identical actor tensors and sensor semantics. Environment distribution changed.',optimizerReset=True)
    return result,models
