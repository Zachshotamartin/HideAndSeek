"""Matched larger-batch self-play A/B trial with frozen own opponent policies.

A: current versus current. B: half current/current, one quarter per role against
frozen historical opponents. Only current actor rows receive policy gradients.
No physics, actor input, reward, or button semantics change in this experiment.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from central_critic import FEATURES, batch_states
from residual_critic import ResidualCentralCritic, SCHEMA
from persistent_actor import PersistentActor, FORMAT
from persistent_train import environment_config, file_hash
from central_persistent_train import advantages_and_returns, critic_update
from fit_central_value import REWARD_SCALE
from league_ppo import assign_roles, active_masks, act_grouped, actor_update
from env_pool import PhysicsEnvPool

torch.set_num_threads(1)


def load_actors(saved, frozen=False):
    if saved['format'] != FORMAT:
        raise ValueError('This comparison only uses our compatible persistent actors')
    models = [PersistentActor() for _ in range(2)]
    for model, state in zip(models, saved['models']):
        model.load_state_dict(state)
        if frozen:
            model.eval().requires_grad_(False)
    return models


def train(args, *, stop_requested=None, on_checkpoint=None):
    if args.horizon % args.sequence_length or args.envs < 1:
        raise ValueError('Positive environment count and complete recurrent sequences required')
    block = args.envs * args.horizon * 2
    target_updates = args.target_interactions // block
    if not target_updates:
        raise ValueError('Interaction target is smaller than one complete rollout')
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    fitted = torch.load(args.critic, map_location='cpu', weights_only=False)
    initial = torch.load(args.initial, map_location='cpu', weights_only=False)
    if initial['format'] != FORMAT or initial['decisions'] != 0:
        raise ValueError('The fixed initial comparison must be a compatible zero-experience policy')
    parent_hash = file_hash(args.parent)
    physics_hash = file_hash(Path(__file__).with_name('physics.py'))
    if fitted['format'] != SCHEMA or fitted['parentSHA256'] != parent_hash:
        raise ValueError('Use a residual critic initialization tied to this exact parent')
    if parent['provenance']['physicsSHA256'] != physics_hash:
        raise ValueError('Frozen physics identity changed')
    history_paths = [Path(path) for path in args.history]
    history_records = [torch.load(path, map_location='cpu', weights_only=False) for path in history_paths]
    if not history_records or any(record['provenance']['physicsSHA256'] != physics_hash for record in history_records):
        raise ValueError('Historical opponents must use this unchanged physical game')
    histories = [load_actors(record, frozen=True) for record in history_records]
    history_hashes = [file_hash(path) for path in history_paths]
    frozen_before = [[copy.deepcopy(model.state_dict()) for model in pair] for pair in histories]
    resumed = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if resumed and (resumed['provenance']['parentSHA256'] != parent_hash or resumed['arguments']['arm'] != args.arm):
        raise ValueError('Resume requires the same parent and trial arm')
    if resumed:
        if [entry['sha256'] for entry in resumed['provenance']['history']] != history_hashes:
            raise ValueError('Resume requires the identical ordered frozen opponent pool')
        if resumed['provenance']['criticFitSHA256'] != file_hash(args.critic):
            raise ValueError('Resume requires the identical initial critic asset')
        if resumed['provenance']['initialSHA256'] != file_hash(args.initial):
            raise ValueError('Resume requires the identical fixed initial comparison')
        for field in ['envs','horizon','sequence_length','sequence_batch','critic_batch_size','epochs','learning_rate','critic_learning_rate','entropy','kl_limit','seed']:
            if resumed['arguments'][field] != getattr(args, field):
                raise ValueError(f'Cannot silently change resumed setting: {field}')
    models = load_actors(resumed or parent)
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5) for model in models]
    for optimizer, state in zip(optimizers, (resumed or parent)['optimizers']):
        optimizer.load_state_dict(state)
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate
    critic = ResidualCentralCritic(fitted['dropout'])
    critic.load_state_dict(resumed['centralCritic'] if resumed else fitted['critic'])
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_learning_rate)
    critic_optimizer.load_state_dict(resumed['centralCriticOptimizer'] if resumed else fitted['criticOptimizer'])
    for group in critic_optimizer.param_groups:
        group['lr'] = args.critic_learning_rate
    world_generator = np.random.default_rng(args.seed)
    opponent_generator = np.random.default_rng(args.seed + 1)
    torch.set_rng_state((resumed or parent)['torchRNG'])
    if resumed:
        world_generator.bit_generator.state = resumed['worldRNG']
        opponent_generator.bit_generator.state = resumed['opponentRNG']
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'latest.pt').exists() and not resumed:
        raise ValueError('Do not overwrite an existing comparison arm')
    if not resumed:
        shutil.copyfile(args.parent, destination / 'parent.pt')
        shutil.copyfile(args.initial, destination / 'initial.pt')
    source_names = ['train_league.py','league_ppo.py','central_persistent_train.py','residual_critic.py',
                    'central_critic.py','persistent_actor.py','persistent_train.py','fit_central_value.py',
                    'env_pool.py','physics.py']
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in source_names}
    provenance = copy.deepcopy(resumed['provenance']) if resumed else dict(
        parentSHA256=parent_hash, parentDecisions=parent['decisions'],
        physicsSHA256=physics_hash, criticFitSHA256=file_hash(args.critic),
        initialSHA256=file_hash(args.initial),
        criticWarmupDecisions=fitted['report']['environmentActorDecisions']['train'],
        history=[dict(file=str(path.resolve()),sha256=file_hash(path),decisions=record['decisions'])
                 for path,record in zip(history_paths,history_records)],
        actorInput='Unchanged restricted140 observations; no opponent identity or central state',
        rule='Only original zero-sum visibility reward; unchanged physical grab/lock state semantics',
        comparisons='Equal total interaction budget; B has fewer current-actor samples, reported separately',
        sourceHistory=[], resumes=[])
    provenance['sourceHistory'].append(sources)
    if resumed:
        provenance['resumes'].append(dict(afterInteractions=resumed['totalPolicyInteractions'],
            worldsRestarted=args.envs, reason='Explicit process restart; memories/buttons reset. No claim of exact uninterrupted rollout parity.'))
    source_dir = destination / 'source' / str(len(provenance['sourceHistory']))
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in source_names:
        shutil.copyfile(Path(__file__).with_name(name), source_dir / name)
    (source_dir / 'SHA256.json').write_text(json.dumps(sources, indent=2) + '\n')
    first_update = resumed['pilotUpdates'] if resumed else 0
    interactions = resumed['totalPolicyInteractions'] if resumed else 0
    current_counts = np.array(resumed['currentPolicyDecisions'] if resumed else [0,0], np.int64)
    historical_counts = np.array(resumed['historicalPolicyDecisions'] if resumed else [0,0], np.int64)
    active_counts = np.array(resumed['activePolicySamples'] if resumed else [0,0], np.int64)
    episodes = resumed['pilotEpisodes'] if resumed else 0
    previous_seconds = resumed['seconds'] if resumed else 0
    log = list(resumed['log']) if resumed else []
    recent = []
    configs = [dict(seed=int(world_generator.integers(1,2**30)),
                    **environment_config(world_generator,index)) for index in range(args.envs)]
    role_ids = assign_roles(opponent_generator,args.envs,args.arm,len(histories))
    with PhysicsEnvPool(configs,workers=args.workers,with_central_state=True,
                        ignore_parent_signals=getattr(args,'graceful_worker_signals',False)) as pool:
        physical = pool.observations.copy()
        buttons = np.zeros((args.envs,2,2),np.float32)
        memories = [torch.zeros(args.envs,64),torch.zeros(args.envs,64)]
        starts = torch.ones(args.envs)
        started = time.monotonic()
        retained = {int(value) for value in args.retain_updates.split(',') if value}
        for update in range(first_update,target_updates):
            critic.eval()
            h,n = args.horizon,args.envs
            actor_buffers = [[torch.zeros(h,n,140),torch.zeros(h,n,64),torch.zeros(h,n),
                              torch.zeros(h,n,5),torch.zeros(h,n)] for _ in range(2)]
            current_rows = torch.zeros(h,n,2,dtype=torch.bool)
            central = {key:torch.zeros(h,n,*shape) for key,shape in FEATURES.items()}
            before_memories = torch.zeros(h,n,2,64)
            partial_values = torch.zeros(h,n,2)
            values = torch.zeros(h,n)
            rewards = torch.zeros(h,n)
            dones = torch.zeros(h,n)
            rollout_started = time.monotonic()
            unscaled = np.zeros(2)
            for step in range(h):
                for role in range(2):
                    memories[role] *= 1 - starts[:,None]
                before = torch.stack(memories,dim=1).clone()
                state = batch_states(pool.central_states)
                for key in FEATURES:
                    central[key][step] = state[key]
                before_memories[step] = before
                current,active = active_masks(physical,role_ids)
                current_rows[step] = torch.from_numpy(current)
                current_counts += current.sum(0)
                historical_counts += (~current).sum(0)
                active_counts += active.sum(0)
                with torch.no_grad():
                    actions,next_memories,next_buttons,records = act_grouped(
                        models,histories,physical,memories,buttons,role_ids)
                    partial_values[step] = torch.stack([record[3] for record in records],dim=-1)
                    values[step] = critic(state,before,partial_values[step])
                for role in range(2):
                    observed,raw,logp,_ = records[role]
                    actor_buffers[role][0][step] = observed
                    actor_buffers[role][1][step] = memories[role]
                    actor_buffers[role][2][step] = starts
                    actor_buffers[role][3][step] = raw
                    actor_buffers[role][4][step] = logp
                physical,returned,done,infos = pool.step(actions)
                np.testing.assert_array_equal(returned.sum(1),np.zeros(n))
                rewards[step] = torch.from_numpy(returned[:,0].copy()) * REWARD_SCALE
                dones[step] = torch.from_numpy(done.astype(np.float32))
                unscaled += returned.sum(0)
                buttons,memories = next_buttons,next_memories
                finished = np.flatnonzero(done).tolist()
                if finished:
                    recent.extend(infos[index]['hidden']/infos[index]['play_steps'] for index in finished)
                    recent = recent[-256:]
                    episodes += len(finished)
                    seeds = [int(world_generator.integers(1,2**30)) for _ in finished]
                    changes = [environment_config(world_generator,index) for index in finished]
                    physical[finished] = pool.reset_at(finished,seeds,changes)
                    buttons[finished] = 0
                    role_ids[finished] = assign_roles(opponent_generator,len(finished),args.arm,len(histories))
                starts = torch.from_numpy(done.astype(np.float32))
            rollout_seconds = time.monotonic()-rollout_started
            with torch.no_grad():
                bootstrap_memories = [memory*(1-starts[:,None]) for memory in memories]
                _,_,_,records = act_grouped(models,histories,physical,bootstrap_memories,buttons,role_ids,sample=False)
                bootstrap_partial = torch.stack([record[3] for record in records],dim=-1)
                bootstrap = critic(batch_states(pool.central_states),torch.stack(bootstrap_memories,dim=1),bootstrap_partial)
            advantages,returns = advantages_and_returns(rewards,values,dones,bootstrap)
            optimizer_started = time.monotonic()
            actor_stats = []
            for role,model in enumerate(models):
                actor_stats.append(actor_update(model,optimizers[role],actor_buffers[role],
                    advantages if role==0 else -advantages,current_rows[:,:,role],role,
                    epochs=args.epochs,sequence_length=args.sequence_length,sequence_batch=args.sequence_batch,
                    entropy_weight=args.entropy,kl_limit=args.kl_limit))
            value_stats = critic_update(critic,critic_optimizer,central,before_memories,values,returns,
                epochs=args.epochs,batch_size=args.critic_batch_size,partial_values=partial_values)
            optimizer_seconds = time.monotonic()-optimizer_started
            interactions += block
            assert int(current_counts.sum()+historical_counts.sum()) == interactions
            row = dict(arm=args.arm,update=update+1,totalPolicyInteractions=interactions,
                currentPolicyDecisions=current_counts.tolist(),historicalPolicyDecisions=historical_counts.tolist(),
                activePolicySamples=active_counts.tolist(),episodes=episodes,
                seconds=previous_seconds+time.monotonic()-started,
                rolloutSeconds=rollout_seconds,optimizerSeconds=optimizer_seconds,
                hider=actor_stats[0],seeker=actor_stats[1],critic=value_stats,
                meanHiddenFraction=float(np.mean(recent)) if recent else None,
                unscaledRewardSum=unscaled.tolist())
            log.append(row)
            print(json.dumps(row),flush=True)
            stopping = bool(args.stop_after_updates and update+1>=args.stop_after_updates)
            stopping |= bool(stop_requested and stop_requested())
            if (update+1)%args.save_every==0 or update+1 in retained or stopping or update+1==target_updates:
                for before,pair in zip(frozen_before,histories):
                    for old,model in zip(before,pair):
                        for name,value in model.state_dict().items():
                            torch.testing.assert_close(old[name],value,atol=0,rtol=0)
                        assert all(parameter.grad is None for parameter in model.parameters())
                saved = dict(format=FORMAT,observationSize=140,physicsObservationSize=138,
                    trainingMethod=getattr(args, 'training_description', f'Matched larger-batch residual CTDE trial arm {args.arm}; only current actors optimized'),
                    models=[model.state_dict() for model in models],optimizers=[optimizer.state_dict() for optimizer in optimizers],
                    centralCritic=critic.state_dict(),centralCriticOptimizer=critic_optimizer.state_dict(),
                    centralCriticSchema=SCHEMA,parentDecisions=parent['decisions'],
                    criticWarmupDecisions=provenance['criticWarmupDecisions'],
                    pilotDecisions=interactions,totalPolicyInteractions=interactions,
                    actorUpdateDecisions=int(current_counts.sum()),currentPolicyDecisions=current_counts.tolist(),
                    historicalPolicyDecisions=historical_counts.tolist(),activePolicySamples=active_counts.tolist(),
                    decisions=parent['decisions']+provenance['criticWarmupDecisions']+interactions,
                    pilotUpdates=update+1,pilotEpisodes=episodes,seconds=row['seconds'],
                    provenance=provenance,arguments=vars(args),log=log,
                    torchRNG=torch.get_rng_state(),worldRNG=world_generator.bit_generator.state,
                    opponentRNG=opponent_generator.bit_generator.state)
                torch.save(saved,destination/'latest.tmp')
                os.replace(destination/'latest.tmp',destination/'latest.pt')
                if update+1 in retained:
                    torch.save(saved,destination/f'checkpoint-{interactions}.pt')
                (destination/'training-log.json').write_text(json.dumps(log,indent=2)+'\n')
                (destination/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
                if on_checkpoint is not None:
                    on_checkpoint(saved, destination)
            if stopping:
                break


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for argument in ['parent','critic','initial','output']:
        parser.add_argument('--'+argument,required=True)
    parser.add_argument('--history',nargs='+',required=True)
    parser.add_argument('--arm',choices=['A','B'],required=True)
    parser.add_argument('--resume')
    parser.add_argument('--envs',type=int,default=128)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--horizon',type=int,default=256)
    parser.add_argument('--sequence-length',type=int,default=32)
    parser.add_argument('--sequence-batch',type=int,default=256)
    parser.add_argument('--critic-batch-size',type=int,default=2048)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--learning-rate',type=float,default=.0001)
    parser.add_argument('--critic-learning-rate',type=float,default=.0001)
    parser.add_argument('--entropy',type=float,default=.005)
    parser.add_argument('--kl-limit',type=float,default=.008)
    parser.add_argument('--target-interactions',type=int,default=7995392)
    parser.add_argument('--stop-after-updates',type=int)
    parser.add_argument('--retain-updates',default='61,122')
    parser.add_argument('--save-every',type=int,default=8)
    parser.add_argument('--seed',type=int,default=773119)
    train(parser.parse_args())
