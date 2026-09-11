"""Matched encoder continuation with unchanged visibility-only physical self-play.

Both encoder arms use league B, identical inherited critic weights and fresh
actor/critic optimizers. Runtime models and earlier trainers are untouched.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import time
import numpy as np
import torch
from central_critic import FEATURES, batch_states
from residual_critic import ResidualCentralCritic, SCHEMA
from entity_actor import FORMAT, ENTITY, LEGACY, load_pair
from persistent_train import file_hash
from protocol import environment_config, assign_for_variant, archive_indices
from central_persistent_train import advantages_and_returns, critic_update
from fit_central_value import REWARD_SCALE
from league_ppo import assign_roles, active_masks, act_grouped, actor_update
from env_pool import PhysicsEnvPool

torch.set_num_threads(1)


def load_actors(saved, frozen=False):
    return load_pair(saved, frozen=frozen)[0]


def initialize_critic(source, learning_rate, resumed=None):
    if source.get('centralCriticSchema') != SCHEMA:
        raise ValueError('Expected our frozen residual-critic checkpoint')
    critic_state = (resumed or source)['centralCritic']
    memory_size = (critic_state['correction.value.0.weight'].shape[1] - 228) // 2
    critic = ResidualCentralCritic(.1, memory_size=memory_size)
    critic.load_state_dict((resumed or source)['centralCritic'])
    optimizer = torch.optim.AdamW(critic.parameters(), lr=learning_rate)
    if resumed:
        optimizer.load_state_dict(resumed['centralCriticOptimizer'])
    # Fresh means fresh for both encoder arms: no inherited AdamW moments.
    return critic, optimizer



def train(args, *, stop_requested=None, on_checkpoint=None):
    if not hasattr(args,'variant'):args.variant='full'
    if args.horizon % args.sequence_length or args.envs < 1:
        raise ValueError('Positive environment count and complete recurrent sequences required')
    block = args.envs * args.horizon * 2
    target_updates = args.target_interactions // block
    if not target_updates:
        raise ValueError('Interaction target is smaller than one complete rollout')
    protocol = json.loads(Path(args.protocol).read_text())
    for field, expected in protocol['training'].items():
        if getattr(args, field) != expected:
            raise ValueError(f'Predeclared comparison setting changed: {field}')
    for field, asset in [('parent', args.encoder), ('critic', 'critic'), ('initial', 'initial')]:
        if file_hash(getattr(args, field)) != protocol['assets'][asset]['sha256']:
            raise ValueError(f'Predeclared comparison asset changed: {field}')
    if [file_hash(path) for path in args.history] != [
            protocol['assets'][name]['sha256'] for name in protocol['history']]:
        raise ValueError('Predeclared frozen opponent pool changed')
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    critic_source = torch.load(args.critic, map_location='cpu', weights_only=False)
    initial = torch.load(args.initial, map_location='cpu', weights_only=False)
    if initial['decisions'] != 0:
        raise ValueError('The fixed initial comparison must be a compatible zero-experience policy')
    parent_hash = file_hash(args.parent)
    physics_hash = file_hash(Path(__file__).with_name('physics.py'))
    if physics_hash != protocol['physicsSHA256']:
        raise ValueError('Predeclared physics changed')
    expected_type = ENTITY if args.encoder == 'entity' else LEGACY
    if parent.get('encoderTypes') != [expected_type] * 2 or parent.get('newActorUpdates') != 0:
        raise ValueError('Use the matching immutable zero-update prepared arm')
    if any(state['state'] for state in parent['optimizers']):
        raise ValueError('Both arms require equally empty initial actor Adam states')
    if critic_source['provenance']['physicsSHA256'] != physics_hash:
        raise ValueError('Inherited critic must belong to this same physical game')
    if args.arm != 'B':
        raise ValueError('Both encoder arms use the same frozen-opponent league B')
    if parent['provenance']['physicsSHA256'] != physics_hash:
        raise ValueError('Versioned physics identity changed')
    history_paths = [Path(path) for path in args.history]
    history_records = [torch.load(path, map_location='cpu', weights_only=False) for path in history_paths]
    if not history_records or any(record['provenance']['physicsSHA256'] != physics_hash for record in history_records):
        raise ValueError('Historical opponents must use this unchanged physical game')
    histories = [load_actors(record, frozen=True) for record in history_records]
    history_hashes = [file_hash(path) for path in history_paths]
    frozen_before = [[copy.deepcopy(model.state_dict()) for model in pair] for pair in histories]
    resumed = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if resumed and (resumed['provenance']['parentSHA256'] != parent_hash or resumed['arguments']['encoder'] != args.encoder):
        raise ValueError('Resume requires the same parent and trial arm')
    if resumed:
        if resumed['provenance']['protocolSHA256'] != file_hash(args.protocol):
            raise ValueError('Resume requires the identical predeclared protocol')
        if [entry['sha256'] for entry in resumed['provenance']['history']] != history_hashes:
            raise ValueError('Resume requires the identical ordered frozen opponent pool')
        if resumed['provenance']['criticSourceSHA256'] != file_hash(args.critic):
            raise ValueError('Resume requires the identical initial critic asset')
        if resumed['provenance']['initialSHA256'] != file_hash(args.initial):
            raise ValueError('Resume requires the identical fixed initial comparison')
        for field in ['encoder','arm','envs','horizon','sequence_length','sequence_batch','critic_batch_size','epochs','learning_rate','critic_learning_rate','entropy','kl_limit','seed']:
            if resumed['arguments'][field] != getattr(args, field):
                raise ValueError(f'Cannot silently change resumed setting: {field}')
    if resumed and 'leagueModels' in resumed:
        histories = [load_actors({'format':FORMAT,'models': pair, 'encoderTypes': parent['encoderTypes']}, frozen=True) for pair in resumed['leagueModels']]
        frozen_before = [[copy.deepcopy(m.state_dict()) for m in pair] for pair in histories]
    models = load_actors(resumed or parent)
    hidden_size = models[0].hidden_size
    if models[1].hidden_size != hidden_size or any(m.hidden_size > hidden_size for pair in histories for m in pair):
        raise ValueError('Current actors need equal memory capacity, no smaller than historical opponents')
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5) for model in models]
    if resumed:
        for optimizer, state in zip(optimizers, resumed['optimizers']):
            optimizer.load_state_dict(state)
    critic, critic_optimizer = initialize_critic(critic_source, args.critic_learning_rate, resumed)
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
    source_names = ['train_entity.py','entity_actor.py','actor.py','league_ppo.py','central_persistent_train.py','residual_critic.py',
                    'central_critic.py','persistent_actor.py','persistent_train.py','fit_central_value.py',
                    'env_pool.py','physics.py','protocol.py','snapshots.py']
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in source_names}
    provenance = copy.deepcopy(resumed['provenance']) if resumed else dict(
        parentSHA256=parent_hash, parentDecisions=parent['decisions'], protocolSHA256=file_hash(args.protocol),
        sourceSelectedPairSHA256=parent['provenance']['sourceSelectedPairSHA256'],
        inheritedRoleSources=parent['provenance']['roleSources'],
        encoderPreparation=parent['provenance']['encoderPreparation'],
        physicsSHA256=physics_hash, criticSourceSHA256=file_hash(args.critic),
        initialSHA256=file_hash(args.initial), criticWarmupDecisions=0,
        criticInitialization='Same inherited residual critic weights in both arms; new AdamW state. No claim of calibration to the projected actor.',
        optimizerInitialization='Both actor Adam states and central critic AdamW state start empty in both arms.',
        history=[dict(file=str(path.resolve()),sha256=file_hash(path),decisions=record['decisions'])
                 for path,record in zip(history_paths,history_records)],
        actorInput='Restricted 210 observations; all ten object slots and thirty range sensors; no opponent identity or central state',
        rule='Only original zero-sum visibility reward; physical grab/lock semantics; no distance-only hiding',
        comparisons=protocol.get('comparisons', 'Same league B and unchanged visibility reward.'),
        architecture=[dict(memory=m.hidden_size, encoder=m.encoder_size, parameters=sum(p.numel() for p in m.parameters())) for m in models],
        sourceHistory=[], resumes=[])
    if resumed and resumed['provenance']['sourceHistory'][-1]!=sources:
        raise ValueError('Training source changed; exact resume refused. Create an explicit new protocol instead.')
    provenance['sourceHistory'].append(sources)
    if resumed:
        provenance['resumes'].append(dict(afterInteractions=resumed['totalPolicyInteractions'],
            worldsRestarted=0 if 'rolloutState' in resumed else args.envs, reason='Full physical/weld/RNG/memory restoration when rolloutState is present; legacy checkpoints explicitly reset worlds.'))
    source_dir = destination / 'source' / str(len(provenance['sourceHistory']))
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.protocol, source_dir / 'PROTOCOL.json')
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
                    **environment_config(world_generator,index,args.variant)) for index in range(args.envs)]
    descriptors = list(resumed.get('leagueDescriptors', [])) if resumed else []
    if len(descriptors)!=len(histories): descriptors=[[0.]*10 for _ in histories]
    role_ids = assign_for_variant(opponent_generator,args.envs,len(histories),.8,args.variant)
    if resumed and 'rolloutState' in resumed:configs=resumed['rolloutState']['pool']['configs']
    with PhysicsEnvPool(configs,workers=args.workers,with_central_state=True,
                        ignore_parent_signals=getattr(args,'graceful_worker_signals',False)) as pool:
        physical = pool.observations.copy()
        buttons = np.zeros((args.envs,2,2),np.float32)
        memories = [torch.zeros(args.envs,hidden_size),torch.zeros(args.envs,hidden_size)]
        starts = torch.ones(args.envs)
        if resumed and 'rolloutState' in resumed:
            rs=resumed['rolloutState'];physical=pool.restore(rs['pool']);buttons=rs['buttons'];memories=rs['memories'];starts=rs['starts'];role_ids=rs['roleIds']
            world_generator.bit_generator.state=resumed['worldRNG'];opponent_generator.bit_generator.state=resumed['opponentRNG'];torch.set_rng_state(resumed['torchRNG'])
        started = time.monotonic()
        retained = {int(value) for value in args.retain_updates.split(',') if value}
        for update in range(first_update,target_updates):
            critic.eval()
            h,n = args.horizon,args.envs
            actor_buffers = [[torch.zeros(h,n,210),torch.zeros(h,n,hidden_size),torch.zeros(h,n),
                              torch.zeros(h,n,6),torch.zeros(h,n)] for _ in range(2)]
            current_rows = torch.zeros(h,n,2,dtype=torch.bool)
            central = {key:torch.zeros(h,n,*shape) for key,shape in FEATURES.items()}
            before_memories = torch.zeros(h,n,2,hidden_size)
            partial_values = torch.zeros(h,n,2)
            values = torch.zeros(h,n)
            rewards = torch.zeros(h,n)
            dones = torch.zeros(h,n)
            rollout_started = time.monotonic()
            unscaled = np.zeros(2)
            behavior = np.zeros(10); behavior_count=0
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
                behavior[:6] += np.mean(np.abs(actions),axis=(0,1)); behavior_count+=1
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
                    for index in finished:
                        r=infos[index]; behavior[6:8]+=np.asarray(r['path'])/max(1,r['play_steps']); behavior[8:]+=np.asarray(r['grabs'])/max(1,r['play_steps'])
                    seeds = [int(world_generator.integers(1,2**30)) for _ in finished]
                    changes = [environment_config(world_generator,index,args.variant) for index in finished]
                    physical[finished] = pool.reset_at(finished,seeds,changes)
                    buttons[finished] = 0
                    role_ids[finished] = assign_for_variant(opponent_generator,len(finished),len(histories),active_counts[1]/max(1,current_counts[1]),args.variant)
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
                    entropy_weight=args.entropy,kl_limit=args.kl_limit,burn_in=0 if args.variant in ('baseline','short-memory') else 32))
            value_stats = critic_update(critic,critic_optimizer,central,before_memories,values,returns,
                epochs=args.epochs,batch_size=args.critic_batch_size,partial_values=partial_values)
            optimizer_seconds = time.monotonic()-optimizer_started
            interactions += block
            assert int(current_counts.sum()+historical_counts.sum()) == interactions
            row = dict(encoder=args.encoder,arm=args.arm,update=update+1,totalPolicyInteractions=interactions,
                currentPolicyDecisions=current_counts.tolist(),historicalPolicyDecisions=historical_counts.tolist(),
                activePolicySamples=active_counts.tolist(),episodes=episodes,
                seconds=previous_seconds+time.monotonic()-started,
                rolloutSeconds=rollout_seconds,optimizerSeconds=optimizer_seconds,
                hider=actor_stats[0],seeker=actor_stats[1],critic=value_stats,
                meanHiddenFraction=float(np.mean(recent)) if recent else None,
                unscaledRewardSum=unscaled.tolist())
            if (update + 1) % 16 == 0:
                pair = [copy.deepcopy(m).eval().requires_grad_(False) for m in models]
                histories.append(pair)
                frozen_before.append([copy.deepcopy(m.state_dict()) for m in pair])
                descriptors.append((behavior/max(1,behavior_count)).tolist())
            # Retain the latest eight plus any opponent still serving an episode.
            keep=sorted(set(range(max(0,len(histories)-8),len(histories)))|{int(x) for x in role_ids.ravel() if x>=0}) if args.variant in ('baseline','recent-only') else archive_indices(descriptors,role_ids)
            remap={old:new for new,old in enumerate(keep)}
            for role in range(2):
                role_ids[:,role]=np.array([remap[int(x)] if x>=0 else -1 for x in role_ids[:,role]])
            histories=[histories[i] for i in keep];frozen_before=[frozen_before[i] for i in keep];descriptors=[descriptors[i] for i in keep]
            row['leagueSize'] = len(histories)
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
                saved = dict(format=FORMAT,encoderTypes=parent['encoderTypes'],observationSize=210,physicsObservationSize=208,
                    trainingMethod=getattr(args, 'training_description', f'Matched encoder comparison: {args.encoder}, league B; original visibility-only rewards'),
                    rolloutState=dict(pool=pool.snapshot(),buttons=buttons.copy(),memories=[x.clone() for x in memories],starts=starts.clone(),roleIds=role_ids.copy()),leagueDescriptors=descriptors,leagueModels=[[m.state_dict() for m in pair] for pair in histories],models=[model.state_dict() for model in models],optimizers=[optimizer.state_dict() for optimizer in optimizers],
                    centralCritic=critic.state_dict(),centralCriticOptimizer=critic_optimizer.state_dict(),
                    centralCriticSchema=SCHEMA,parentDecisions=parent['decisions'],
                    criticWarmupDecisions=provenance['criticWarmupDecisions'],
                    pilotDecisions=interactions,totalPolicyInteractions=interactions,
                    actorUpdateDecisions=int(current_counts.sum()),currentPolicyDecisions=current_counts.tolist(),
                    historicalPolicyDecisions=historical_counts.tolist(),activePolicySamples=active_counts.tolist(),
                    decisions=parent['decisions']+interactions,
                    decisionsDefinition='Inherited selected-pair resource lineage plus this arm total current/historical policy interactions; not a per-role count.',
                    newActorUpdates=update+1,
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


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--variant',choices=['full','baseline','short','unbalanced','recent-only','short-memory'],default='full')
    for argument in ['parent', 'critic', 'initial', 'output', 'protocol']:
        result.add_argument('--' + argument, required=True)
    result.add_argument('--history', nargs='+', required=True)
    result.add_argument('--encoder', choices=['legacy', 'entity'], required=True)
    result.add_argument('--arm', choices=['B'], default='B')
    result.add_argument('--resume')
    result.add_argument('--envs', type=int, default=128)
    result.add_argument('--workers', type=int, default=8)
    result.add_argument('--horizon', type=int, default=256)
    result.add_argument('--sequence-length', type=int, default=32)
    result.add_argument('--sequence-batch', type=int, default=256)
    result.add_argument('--critic-batch-size', type=int, default=2048)
    result.add_argument('--epochs', type=int, default=2)
    result.add_argument('--learning-rate', type=float, default=.0001)
    result.add_argument('--critic-learning-rate', type=float, default=.0001)
    result.add_argument('--entropy', type=float, default=.005)
    result.add_argument('--kl-limit', type=float, default=.008)
    result.add_argument('--target-interactions', type=int, default=15990784)
    result.add_argument('--stop-after-updates', type=int)
    result.add_argument('--retain-updates', default='122,244')
    result.add_argument('--save-every', type=int, default=8)
    result.add_argument('--seed', type=int, default=885713)
    return result


def main(args):
    stopped = False
    def stop(signum, _frame):
        nonlocal stopped
        stopped = True
        print(json.dumps(dict(stopRequested=signal.Signals(signum).name,
            behavior='Finish this rollout/update, atomically save optimizer/RNG state, then stop.')), flush=True)
    previous = {sig: signal.getsignal(sig) for sig in [signal.SIGINT, signal.SIGTERM]}
    for sig in previous:
        signal.signal(sig, stop)
    args.graceful_worker_signals = True
    try:
        train(args, stop_requested=lambda: stopped)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main(parser().parse_args())
