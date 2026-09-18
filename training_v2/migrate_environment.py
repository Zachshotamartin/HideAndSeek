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
        for key,value in state.items():
            if key in ('encoder.fixed.weight','encoder.query.weight'):
                target[key].zero_();target[key][:,:18]=value[:,:18]
                # Thirty new ray directions: reuse the nearest old direction explicitly.
                for i in range(30): target[key][:,18+i]=value[:,18+round(i*24/30)%24]
                target[key][:,-2:]=value[:,-2:]
            elif key in target and target[key].shape==value.shape:target[key]=value.clone()
            else:raise ValueError('Unrecognized migration parameter '+key)
        model.load_state_dict(target);models.append(model)
    result=copy.deepcopy(source)
    result.update(format=FORMAT,encoderTypes=[ENTITY]*2,observationSize=210,physicsObservationSize=208,
        models=[m.state_dict() for m in models],newActorUpdates=0,
        optimizers=[torch.optim.Adam(m.parameters(),lr=.0001,eps=1e-5).state_dict() for m in models],torchRNG=torch.get_rng_state())
    provenance=result['provenance'];provenance['sourcePhysicsSHA256']=provenance['physicsSHA256']
    provenance['physicsSHA256']=file_hash(Path(__file__).with_name('physics.py'))
    provenance['roleSources']=copy.deepcopy(provenance.get('inheritedRoleSources',[]))
    provenance['encoderPreparation']=dict(description='Explicit weights-only migration: 10 objects, 30 lidar rays, no sight-distance cap, zero-gated relational attention. Sensor semantics changed; not function-preserving.',optimizerReset=True)
    return result,models
