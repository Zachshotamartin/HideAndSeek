"""Separate CTDE PPO experiment; restricted actors and game scoring unchanged.

Only the value baseline sees privileged physical state. Actor optimization has
no central-state argument and no gradient path from the separate critic.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from central_critic import CentralCritic, FEATURES, batch_states
from env_pool import PhysicsEnvPool
from fit_central_value import sample_restricted, GAMMA, REWARD_SCALE
from persistent_actor import PersistentActor, FORMAT, augment
from persistent_train import environment_config, file_hash
from residual_critic import ResidualCentralCritic, SCHEMA as RESIDUAL_SCHEMA

torch.set_num_threads(1)


def advantages_and_returns(rewards, values, dones, bootstrap):
    advantages=torch.zeros_like(rewards);accumulator=torch.zeros_like(bootstrap)
    for step in reversed(range(len(rewards))):
        following=bootstrap if step==len(rewards)-1 else values[step+1]
        live=1-dones[step]
        delta=rewards[step]+GAMMA*following*live-values[step]
        accumulator=delta+GAMMA*.98*live*accumulator;advantages[step]=accumulator
    return advantages,advantages+values


def normalize_active_advantages(advantages, active):
    selected=advantages[active]
    if not selected.numel():return torch.zeros_like(advantages)
    return torch.where(active,(advantages-selected.mean())/(selected.std(unbiased=False)+1e-8),0)


def policy_objective(logp, old_logp, entropy, advantages, active, entropy_weight=.005):
    """Sums over active actions only; blind recurrence is handled by the caller."""
    ratio=(logp-old_logp).exp()
    surrogate=torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages)
    objective=(-(surrogate+entropy_weight*entropy)*active).sum()
    kl=(((ratio-1)-(logp-old_logp))*active).sum()
    return objective,kl,active.sum()


def actor_update(model,optimizer,batch,advantages,role,epochs=3,sequence_length=32,
                 sequence_batch=64,entropy_weight=.005):
    observations,memories,starts,raw_actions,old_logps=batch
    if observations.shape[-1]!=140:raise ValueError('Actor accepts only the original restricted140 observations')
    active=(observations[:,:,5]>=1) if role else torch.ones_like(advantages,dtype=torch.bool)
    advantages=normalize_active_advantages(advantages,active)
    steps,count=advantages.shape
    sequences=[(begin,env) for begin in range(0,steps,sequence_length) for env in range(count)]
    losses=[];divergences=[];sample_counts=[];updates=0
    for _ in range(epochs):
        ordering=torch.randperm(len(sequences)).tolist();epoch_divergences=[];epoch_counts=[]
        for offset in range(0,len(ordering),sequence_batch):
            selected=[sequences[index] for index in ordering[offset:offset+sequence_batch]]
            begin=torch.tensor([entry[0] for entry in selected]);env=torch.tensor([entry[1] for entry in selected])
            if role==1 and not any(bool((observations[begin+i,env,5]>=1).any()) for i in range(sequence_length)):
                # Adam momentum must not update an actor for an all-blind batch.
                continue
            memory=memories[begin,env].detach();loss=0;kl_sum=0;active_count=0
            for inner in range(sequence_length):
                step=begin+inner;memory=memory*(1-starts[step,env,None])
                normal,tools,_,memory=model(observations[step,env],memory)
                logp,entropy=model.statistics(normal,tools,raw_actions[step,env])
                objective,kl,count_active=policy_objective(logp,old_logps[step,env],entropy,
                    advantages[step,env],active[step,env],entropy_weight)
                loss+=objective;kl_sum+=kl.detach();active_count+=int(count_active)
            loss/=active_count;optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step();updates+=1
            losses.append(float(loss.detach()));sample_counts.append(active_count)
            epoch_divergences.append(float(kl_sum/active_count));epoch_counts.append(active_count)
        divergences.extend(epoch_divergences)
        if epoch_divergences and np.average(epoch_divergences,weights=epoch_counts)>.025:break
    return dict(loss=float(np.average(losses,weights=sample_counts)) if losses else 0,
        approximateKL=float(np.average(divergences,weights=sample_counts)) if divergences else 0,optimizerSteps=updates)


def partial_value_estimates(models, physical, memories, buttons):
    """Current pre-action values without sampling actions or advancing RNG."""
    observations=augment(physical,buttons)
    return torch.stack([model(torch.from_numpy(observations[:,role].copy()),memories[role])[2]
                        for role,model in enumerate(models)],dim=-1).detach()


def critic_update(critic,optimizer,states,memories,old_values,returns,epochs=3,batch_size=512,
                  partial_values=None):
    states={key:value.flatten(0,1) for key,value in states.items()}
    memories=memories.flatten(0,1).detach();old_values=old_values.flatten();returns=returns.flatten()
    if partial_values is not None:partial_values=partial_values.flatten(0,1).detach()
    critic.train()
    losses=[]
    for _ in range(epochs):
        order=torch.randperm(len(returns))
        for offset in range(0,len(order),batch_size):
            selected=order[offset:offset+batch_size]
            inputs=({key:item[selected] for key,item in states.items()},memories[selected])
            value=critic(*inputs,partial_values[selected]) if partial_values is not None else critic(*inputs)
            clipped=old_values[selected]+(value-old_values[selected]).clamp(-.2,.2)
            loss=torch.maximum((value-returns[selected]).square(),(clipped-returns[selected]).square()).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),.5);optimizer.step()
            losses.append(float(loss.detach()))
    return dict(clippedValueMSE=float(np.mean(losses)))


def train(args):
    if args.updates<1 or args.horizon%args.sequence_length:raise ValueError('Positive updates and divisible recurrent horizon required')
    parent=torch.load(args.parent,map_location='cpu',weights_only=False)
    fitted=torch.load(args.critic,map_location='cpu',weights_only=False)
    parent_hash=file_hash(args.parent)
    if parent['format']!=FORMAT or fitted['format'] not in ['central-value-fit-v1',RESIDUAL_SCHEMA] or fitted['parentSHA256']!=parent_hash:
        raise ValueError('The central critic must be fitted on this exact persistent parent')
    physics_hash=file_hash(Path(__file__).with_name('physics.py'))
    if parent['provenance']['physicsSHA256']!=physics_hash:raise ValueError('Physical environment changed')
    original=torch.load(args.initial,map_location='cpu',weights_only=False)
    if original['format']!=FORMAT or original['decisions']!=0:raise ValueError('Initial comparison must have zero experience')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'latest.pt').exists():raise ValueError('Use a new output directory; this experiment does not overwrite previous runs')
    shutil.copyfile(args.parent,output/'warm-start.pt');shutil.copyfile(args.initial,output/'initial.pt')
    models=[PersistentActor() for _ in range(2)]
    for model,state in zip(models,parent['models']):model.load_state_dict(state)
    optimizers=[torch.optim.Adam(model.parameters(),lr=.0003,eps=1e-5) for model in models]
    for optimizer,state in zip(optimizers,parent['optimizers']):optimizer.load_state_dict(state)
    residual=fitted['format']==RESIDUAL_SCHEMA
    critic=ResidualCentralCritic(fitted['dropout']) if residual else CentralCritic(use_actor_memory=True)
    critic.load_state_dict(fitted['critic'])
    critic_optimizer=(torch.optim.AdamW if residual else torch.optim.Adam)(critic.parameters(),lr=args.critic_learning_rate)
    critic_optimizer.load_state_dict(fitted['criticOptimizer'])
    previous_critic_lr=critic_optimizer.param_groups[0]['lr']
    for group in critic_optimizer.param_groups:group['lr']=args.critic_learning_rate
    torch.set_rng_state(parent['torchRNG']);generator=np.random.default_rng(args.seed)
    warmup_decisions=fitted['report']['environmentActorDecisions']['train']
    provenance=dict(parentSHA256=parent_hash,parentDecisions=parent['decisions'],parentProvenance=parent['provenance'],
        actorOptimizerRestored=True,actorSamplingRNGRestored=True,rolloutEnvironmentsRestarted=True,
        initialSHA256=file_hash(args.initial),criticFitSHA256=file_hash(args.critic),
        criticWarmup=fitted['report']['environmentActorDecisions'],criticOptimizerRestored=True,
        criticLearningRate=dict(fitted=previous_critic_lr,ppo=args.critic_learning_rate),
        actorInput='Unchanged138 restricted physical measurements plus two own requested button states',
        criticInput='Training-only current full scene and detached pre-action actor memories and partial value estimates; no policy gradient path',
        criticSchema=fitted['format'],
        physicsSHA256=physics_hash,sourceSHA256={name:file_hash(Path(__file__).with_name(name)) for name in
            ['central_persistent_train.py','fit_central_value.py','central_critic.py','residual_critic.py','persistent_actor.py','env_pool.py','physics.py']})
    configs=[dict(seed=int(generator.integers(1,2**30)),**environment_config(generator,index)) for index in range(args.envs)]
    with PhysicsEnvPool(configs,workers=args.workers,with_central_state=True) as pool:
        physical=pool.observations.copy();buttons=np.zeros((args.envs,2,2),np.float32)
        memories=[torch.zeros(args.envs,64),torch.zeros(args.envs,64)];starts=torch.ones(args.envs)
        decisions=episodes=0;recent=[];log=[];started=time.monotonic()
        retained={int(value) for value in args.retain_updates.split(',') if value}
        for update in range(args.updates):
            critic.eval()
            h,n=args.horizon,args.envs
            actor_buffers=[[torch.zeros(h,n,140),torch.zeros(h,n,64),torch.zeros(h,n),torch.zeros(h,n,5),torch.zeros(h,n)] for _ in range(2)]
            central={key:torch.zeros(h,n,*shape) for key,shape in FEATURES.items()}
            both_memories=torch.zeros(h,n,2,64);values=torch.zeros(h,n);rewards=torch.zeros(h,n);dones=torch.zeros(h,n)
            partial_values=torch.zeros(h,n,2)
            unscaled=np.zeros(2)
            for step in range(h):
                for role in range(2):memories[role]*=1-starts[:,None]
                state=batch_states(pool.central_states);before=torch.stack(memories,dim=1).clone()
                for key in FEATURES:central[key][step]=state[key]
                both_memories[step]=before
                with torch.no_grad():
                    actions,next_memories,next_buttons,records=sample_restricted(models,physical,memories,buttons)
                    partial_values[step]=torch.stack([record[3] for record in records],dim=-1)
                    values[step]=critic(state,before,partial_values[step]) if residual else critic(state,before)
                for role in range(2):
                    observed,raw,logp,_=records[role]
                    buffer=actor_buffers[role]
                    buffer[0][step],buffer[1][step],buffer[2][step]=observed,memories[role],starts
                    buffer[3][step],buffer[4][step]=raw,logp
                physical,returned,done,infos=pool.step(actions)
                np.testing.assert_array_equal(returned.sum(axis=1),np.zeros(n))
                rewards[step]=torch.from_numpy(returned[:,0].copy())*REWARD_SCALE
                dones[step]=torch.from_numpy(done.astype(np.float32));unscaled+=returned.sum(axis=0)
                buttons=next_buttons;memories=next_memories
                finished=np.flatnonzero(done).tolist()
                if finished:
                    for index in finished:recent.append(infos[index]['hidden']/infos[index]['play_steps'])
                    recent=recent[-128:];episodes+=len(finished)
                    seeds=[int(generator.integers(1,2**30)) for _ in finished]
                    changes=[environment_config(generator,index) for index in finished]
                    physical[finished]=pool.reset_at(finished,seeds,changes);buttons[finished]=0
                starts=torch.from_numpy(done.astype(np.float32))
            with torch.no_grad():
                bootstrap_memories=[memory*(1-starts[:,None]) for memory in memories]
                bootstrap_inputs=(batch_states(pool.central_states),torch.stack(bootstrap_memories,dim=1))
                bootstrap=critic(*bootstrap_inputs,partial_value_estimates(models,physical,bootstrap_memories,buttons)) if residual else critic(*bootstrap_inputs)
            advantages,returns=advantages_and_returns(rewards,values,dones,bootstrap)
            actors=[]
            for role,model in enumerate(models):
                actors.append(actor_update(model,optimizers[role],actor_buffers[role],advantages if role==0 else -advantages,
                    role,epochs=args.epochs,sequence_length=args.sequence_length,sequence_batch=args.sequence_batch,entropy_weight=args.entropy))
            value_stats=critic_update(critic,critic_optimizer,central,both_memories,values,returns,
                epochs=args.epochs,batch_size=args.critic_batch_size,partial_values=partial_values if residual else None)
            decisions+=h*n*2
            if (update+1)%args.log_every==0 or update+1==args.updates:
                row=dict(update=update+1,actorUpdateDecisions=decisions,criticWarmupDecisions=warmup_decisions,
                    parentDecisions=parent['decisions'],totalTrainingDecisions=parent['decisions']+warmup_decisions+decisions,
                    episodes=episodes,seconds=round(time.monotonic()-started,3),hider=actors[0],seeker=actors[1],
                    centralValue=value_stats,meanHiddenFraction=float(np.mean(recent)) if recent else None,unscaledRewardSum=unscaled.tolist())
                log.append(row);print(json.dumps(row),flush=True)
            if (update+1)%args.save_every==0 or update+1==args.updates or update+1 in retained:
                saved=dict(format=FORMAT,observationSize=140,physicsObservationSize=138,
                    trainingMethod='Central value baseline with unchanged restricted recurrent PPO actors',
                    models=[model.state_dict() for model in models],optimizers=[optimizer.state_dict() for optimizer in optimizers],
                    centralCritic=critic.state_dict(),centralCriticOptimizer=critic_optimizer.state_dict(),
                    centralCriticSchema=fitted['format'],
                    parentDecisions=parent['decisions'],criticWarmupDecisions=warmup_decisions,
                    pilotDecisions=decisions,actorUpdateDecisions=decisions,decisions=parent['decisions']+warmup_decisions+decisions,
                    pilotUpdates=update+1,pilotEpisodes=episodes,seconds=time.monotonic()-started,
                    provenance=provenance,arguments=vars(args),log=log,torchRNG=torch.get_rng_state(),numpyRNG=generator.bit_generator.state,
                    rewardDescription='Only unchanged zero-sum physical visibility, scaled.05. No tool, pursuit, or holding bonus.')
                torch.save(saved,output/'latest.tmp');os.replace(output/'latest.tmp',output/'latest.pt')
                if update+1 in retained:torch.save(saved,output/f'checkpoint-{decisions}.pt')
                (output/'training-log.json').write_text(json.dumps(log,indent=2)+'\n')
                (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--critic',required=True)
    p.add_argument('--initial',required=True);p.add_argument('--output',required=True);p.add_argument('--updates',type=int,required=True)
    p.add_argument('--envs',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--horizon',type=int,default=128);p.add_argument('--sequence-length',type=int,default=32)
    p.add_argument('--sequence-batch',type=int,default=64);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--entropy',type=float,default=.005);p.add_argument('--critic-learning-rate',type=float,default=.0003)
    p.add_argument('--critic-batch-size',type=int,default=512);p.add_argument('--seed',type=int,default=663071)
    p.add_argument('--log-every',type=int,default=16);p.add_argument('--save-every',type=int,default=32)
    p.add_argument('--retain-updates',default='');train(p.parse_args())
