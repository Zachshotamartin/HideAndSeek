"""Fit an isolated central value baseline from complete, fixed-policy games.

Actor inputs/actions are unchanged. Training, selection and held-out assessment
use disjoint whole maps and episodes. No policy update occurs in this program.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from central_critic import CentralCritic, FEATURES
from env_pool import PhysicsEnvPool
from persistent_actor import PersistentActor, augment, advance_buttons, FORMAT
from persistent_train import environment_config, file_hash
from physics import generate_arena

torch.set_num_threads(1)
GAMMA = .998
REWARD_SCALE = .05
STEPS = 240


def geometry_hash(config):
    """Static map identity, modulo square rotation/reflection and agent spawns.

    The first four outside walls always have the same solid boundary, although
    rotation changes which rectangular segment includes the outside corners.
    Width/height represent that boundary; hash the interior geometry only.
    """
    arena = generate_arena(**config)
    def number(value):
        rounded=round(float(value),6)
        return 0.0 if rounded==0 else rounded
    candidates=[]
    for mirror in [False,True]:
        for quarter in range(4):
            angle=quarter*math.pi/2; co,si=math.cos(angle),math.sin(angle)
            def transform(x,y):
                x-=arena['width']/2;y-=arena['height']/2
                if mirror:x=-x
                return number(co*x-si*y),number(si*x+co*y)
            entries=[]
            for kind,entities in [('walls',arena['walls'][4:]),('objects',arena['objects'])]:
                for entity in entities:
                    x,y,z=entity['position'];sx,sy,sz=entity['size'];yaw=entity.get('yaw',0)
                    corners=sorted(transform(x+math.cos(yaw)*dx*sx/2-math.sin(yaw)*dy*sy/2,
                                             y+math.sin(yaw)*dx*sx/2+math.cos(yaw)*dy*sy/2)
                                   for dx,dy in [(-1,-1),(1,-1),(1,1),(-1,1)])
                    label=entity.get('kind','wall')
                    uphill=(number(co*(-1 if mirror else 1)*math.cos(yaw)-si*math.sin(yaw)),
                            number(si*(-1 if mirror else 1)*math.cos(yaw)+co*math.sin(yaw))) if label=='ramp' else (0.0,0.0)
                    entries.append((kind,label,corners,number(z),number(sz),number(entity.get('mass',0)),uphill))
            candidates.append(json.dumps([arena['width'],arena['height'],sorted(entries)],separators=(',',':')))
    return hashlib.sha256(min(candidates).encode()).hexdigest()


def choose_configs(generator, count, start, namespace, used):
    configs=[]; hashes=[]
    while len(configs)<count:
        config=environment_config(generator,start+len(configs))
        # Validation namespaces are never sampled by the on-policy trainer.
        seed=int(generator.integers(1,2**30)) if namespace==0 else namespace+int(generator.integers(0,1000000))
        config={'seed':seed,**config}; digest=geometry_hash(config)
        if digest in used: continue
        used.add(digest); configs.append(config); hashes.append(digest)
    return configs,hashes


def sample_restricted(models, physical, memories, buttons):
    """The only actor path: 208 restricted measurements + two own buttons."""
    observations=augment(physical,buttons)
    count=len(physical); actions=np.zeros((count,2,6),np.float32)
    next_buttons=buttons.copy(); next_memories=[]; records=[]
    for role,model in enumerate(models):
        observed=torch.from_numpy(observations[:,role].copy())
        movement,commands,raw,logp,value,memory=model.act(observed,memories[role])
        blind=(physical[:,role,5]<1) if role else np.zeros(count,bool)
        next_buttons[:,role]=advance_buttons(buttons[:,role],commands.numpy(),blind)
        actions[:,role,:3]=movement.numpy()[:,:3];actions[:,role,5]=movement.numpy()[:,3];actions[:,role,3:5]=next_buttons[:,role]
        actions[blind,role]=0
        next_memories.append(memory)
        records.append((observed,raw,logp,value))
    return actions,next_memories,next_buttons,records


def collect(models, episodes, env_count, workers, seed, namespace, used):
    generator=np.random.default_rng(seed); torch.manual_seed(seed)
    if episodes%env_count: raise ValueError('Complete-episode collection must be divisible by environment count')
    chunks={key:[] for key in FEATURES}; memories_out=[]; partial_out=[]; targets_out=[]; configs_out=[]; hashes_out=[]
    configs,hashes=choose_configs(generator,env_count,0,namespace,used)
    with PhysicsEnvPool(configs,workers=workers,with_central_state=True) as pool:
        for begin in range(0,episodes,env_count):
            if begin:
                configs,hashes=choose_configs(generator,env_count,begin,namespace,used)
                pool.reset_at(range(env_count),[config['seed'] for config in configs],configs)
            configs_out.extend(configs);hashes_out.extend(hashes)
            physical=pool.observations.copy();buttons=np.zeros((env_count,2,2),np.float32)
            memories=[torch.zeros(env_count,64),torch.zeros(env_count,64)]
            frames={key:[] for key in FEATURES}; history=[]; partial=[]; rewards=[]
            with torch.no_grad():
                for step in range(STEPS):
                    for key in FEATURES:
                        frames[key].append(torch.from_numpy(np.stack([state[key] for state in pool.central_states])))
                    np.testing.assert_array_equal(frames['agents'][-1][:,:,11:13].numpy(),buttons)
                    history.append(torch.stack(memories,dim=1).clone())
                    actions,memories,buttons,records=sample_restricted(models,physical,memories,buttons)
                    partial.append(torch.stack([record[3] for record in records],dim=-1))
                    physical,reward,done,info=pool.step(actions)
                    if bool(done.any()) != (step==STEPS-1): raise AssertionError('Unexpected episode boundary')
                    np.testing.assert_array_equal(reward.sum(axis=1),np.zeros(env_count))
                    rewards.append(torch.from_numpy(reward[:,0].copy())*REWARD_SCALE)
            discounted=torch.zeros(env_count); targets=torch.zeros(STEPS,env_count)
            for step in reversed(range(STEPS)):
                discounted=rewards[step]+GAMMA*discounted;targets[step]=discounted
            for key in FEATURES:
                stacked=torch.stack(frames[key]).transpose(0,1).contiguous()
                chunks[key].append(stacked.reshape(env_count*STEPS,*FEATURES[key]))
            memories_out.append(torch.stack(history).transpose(0,1).contiguous().reshape(-1,2,64))
            partial_out.append(torch.stack(partial).transpose(0,1).contiguous().reshape(-1,2))
            targets_out.append(targets.T.contiguous().reshape(-1))
            print(json.dumps({'collectionNamespace':namespace,'completeEpisodes':begin+env_count,'planned':episodes}),flush=True)
    return dict(states={key:torch.cat(value) for key,value in chunks.items()},
        memories=torch.cat(memories_out),partialValues=torch.cat(partial_out),targets=torch.cat(targets_out),
        phases=torch.arange(STEPS).repeat(episodes),episodes=episodes,configs=configs_out,mapHashes=hashes_out)


def predict(critic, data, batch_size=1024):
    predictions=[]
    with torch.no_grad():
        for begin in range(0,len(data['targets']),batch_size):
            selected=slice(begin,begin+batch_size)
            predictions.append(critic({key:value[selected] for key,value in data['states'].items()},data['memories'][selected]))
    return torch.cat(predictions)


def assess(critic, data, time_mean):
    central=predict(critic,data);target=data['targets'];timed=time_mean[data['phases']]
    errors={'central':(central-target).square(),
        'oldPartialHider':(data['partialValues'][:,0]-target).square(),
        'oldPartialSeeker':(data['partialValues'][:,1]+target).square(),
        'timeMean':(timed-target).square()}
    errors['oldPartialPair']=(errors['oldPartialHider']+errors['oldPartialSeeker'])/2
    per_episode={key:value.reshape(data['episodes'],STEPS).mean(-1).numpy() for key,value in errors.items()}
    rng=np.random.default_rng(133197);comparisons={}
    for name in ['oldPartialHider','oldPartialSeeker','oldPartialPair','timeMean']:
        values=per_episode['central']-per_episode[name]
        boot=rng.choice(values,size=(10000,len(values)),replace=True).mean(axis=1)
        comparisons[name]=dict(centralMinusBaseline=float(values.mean()),
            bootstrap95Percent=np.quantile(boot,[.025,.975]).tolist())
    return dict(mse={key:float(value.mean()) for key,value in errors.items()},
        perEpisodeMSE={key:value.tolist() for key,value in per_episode.items()},comparisons=comparisons)


def fit(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    saved=torch.load(args.parent,map_location='cpu',weights_only=False)
    if saved['format']!=FORMAT: raise ValueError('Expected our frozen persistent-button parent')
    if saved['provenance']['physicsSHA256']!=file_hash(Path(__file__).with_name('physics.py')):
        raise ValueError('Parent and current physics differ')
    models=[PersistentActor().eval() for _ in range(2)]
    for model,state in zip(models,saved['models']):model.load_state_dict(state)
    actor_before=[copy.deepcopy(model.state_dict()) for model in models]
    used=set()
    train=collect(models,args.train_episodes,args.envs,args.workers,args.seed,0,used)
    selection=collect(models,args.selection_episodes,args.envs,args.workers,args.seed+1,1450000000,used)
    heldout=collect(models,args.test_episodes,args.envs,args.workers,args.seed+2,1460000000,used)
    assert not(set(train['mapHashes'])&set(selection['mapHashes']) or set(train['mapHashes'])&set(heldout['mapHashes']) or set(selection['mapHashes'])&set(heldout['mapHashes']))
    dataset=dict(train=train,selection=selection,heldout=heldout)
    torch.save(dataset,output/'value-dataset.pt')
    torch.manual_seed(args.seed+3)
    critic=CentralCritic(use_actor_memory=True);optimizer=torch.optim.Adam(critic.parameters(),lr=args.learning_rate)
    time_mean=train['targets'].reshape(args.train_episodes,STEPS).mean(0)
    best_error=float('inf');best=None;log=[]
    for epoch in range(args.epochs):
        ordering=torch.randperm(len(train['targets']));losses=[]
        for offset in range(0,len(ordering),args.batch_size):
            selected=ordering[offset:offset+args.batch_size]
            value=critic({key:item[selected] for key,item in train['states'].items()},train['memories'][selected])
            loss=(value-train['targets'][selected]).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1);optimizer.step()
            losses.append(float(loss.detach()))
        error=float((predict(critic,selection)-selection['targets']).square().mean())
        row=dict(epoch=epoch+1,trainMSE=float(np.mean(losses)),selectionMSE=error);log.append(row);print(json.dumps(row),flush=True)
        if error<best_error:
            best_error=error;best=dict(epoch=epoch+1,critic=copy.deepcopy(critic.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()))
    critic.load_state_dict(best['critic'])
    for before,model in zip(actor_before,models):
        for name,value in model.state_dict().items():torch.testing.assert_close(before[name],value,atol=0,rtol=0)
        if any(parameter.grad is not None for parameter in model.parameters()):raise AssertionError('Value fitting touched actor gradients')
    report=dict(format='central-value-fit-assessment-v1',parentSHA256=file_hash(args.parent),parentDecisions=saved['decisions'],
        actorUpdates=0,actorParametersUnchanged=True,selectedEpoch=best['epoch'],
        episodes=dict(train=args.train_episodes,selection=args.selection_episodes,heldout=args.test_episodes),
        environmentActorDecisions=dict(train=args.train_episodes*STEPS*2,selection=args.selection_episodes*STEPS*2,heldout=args.test_episodes*STEPS*2),
        units='Mean squared discounted return; rewards uniformly scaled .05 and gamma .998. Negative central-minus-baseline is better.',
        split='Disjoint complete episodes and canonical static map geometry. Selection chooses epoch; heldout is evaluated once afterward.',
        mapHashes={name:value['mapHashes'] for name,value in dataset.items()},
        selection=assess(critic,selection,time_mean),heldout=assess(critic,heldout,time_mean),log=log,
        sources={name:file_hash(Path(__file__).with_name(name)) for name in
            ['fit_central_value.py','central_critic.py','persistent_actor.py','physics.py','env_pool.py']})
    checkpoint=dict(format='central-value-fit-v1',critic=best['critic'],criticOptimizer=best['optimizer'],
        useActorMemory=True,timeMean=time_mean,parentSHA256=file_hash(args.parent),parentPath=str(Path(args.parent).resolve()),
        report=report,arguments=vars(args))
    torch.save(checkpoint,output/'fitted-critic.pt')
    (output/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'selectedEpoch':best['epoch'],'heldout':report['heldout']['mse'],'comparisons':report['heldout']['comparisons']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--train-episodes',type=int,default=384);parser.add_argument('--selection-episodes',type=int,default=64)
    parser.add_argument('--test-episodes',type=int,default=96);parser.add_argument('--envs',type=int,default=32)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--epochs',type=int,default=24)
    parser.add_argument('--batch-size',type=int,default=512);parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=881033);fit(parser.parse_args())
