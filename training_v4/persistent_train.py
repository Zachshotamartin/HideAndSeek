"""Isolated persistent-button PPO pilot. No strategy reward or physics changes."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from env_pool import PhysicsEnvPool
from persistent_actor import PersistentActor, advance_buttons, augment, FORMAT

torch.set_num_threads(1)

# Same clipped recurrent PPO update as our main trainer, frozen here so ongoing
# main-run changes cannot alter the pilot. Its model supplies categorical command
# likelihoods; discarded seeker-preparation commands carry no actor/entropy loss.
def recurrent_update(model, optimizer, batch, bootstrap, epochs=3, sequence_length=16,
                     entropy_weight=.012, sequence_batch=32, role=0):
    observations, memories, starts, actions, old_logps, rewards, values, dones = batch
    steps, count = rewards.shape
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros(count)
    for step in reversed(range(steps)):
        following = bootstrap if step == steps - 1 else values[step + 1]
        live = 1 - dones[step]
        delta = rewards[step] + .998 * following * live - values[step]
        accumulator = delta + .998 * .98 * live * accumulator
        advantages[step] = accumulator
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    sequences = [(begin, environment) for begin in range(0, steps, sequence_length) for environment in range(count)]
    losses, divergences = [], []
    for _ in range(epochs):
        ordering = torch.randperm(len(sequences)).tolist()
        for offset in range(0, len(ordering), sequence_batch):
            selected = [sequences[index] for index in ordering[offset:offset + sequence_batch]]
            begin = torch.tensor([item[0] for item in selected])
            env = torch.tensor([item[1] for item in selected])
            memory = memories[begin, env].detach()
            total_loss = 0
            kl_sum = 0
            for inner in range(sequence_length):
                step = begin + inner
                memory = memory * (1 - starts[step, env, None])
                normal, tools, value, memory = model(observations[step, env], memory)
                logp, entropy = model.statistics(normal, tools, actions[step, env])
                log_ratio = logp - old_logps[step, env]
                ratio = log_ratio.exp()
                # The seeker cannot act during preparation. Its memory and
                # critic still learn, but discarded actions carry no policy loss.
                active = (observations[step, env, 5] >= 1).float() if role == 1 else torch.ones_like(ratio)
                actor_loss = -(torch.minimum(ratio * advantages[step, env],
                    ratio.clamp(.8, 1.2) * advantages[step, env]) * active).mean()
                # Value clipping prevents an isolated long episode from destabilizing a batch.
                clipped_value = values[step, env] + (value - values[step, env]).clamp(-.2, .2)
                critic_loss = torch.maximum((value - returns[step, env]).square(),
                    (clipped_value - returns[step, env]).square()).mean()
                total_loss += actor_loss + .5 * critic_loss - entropy_weight * (entropy * active).mean()
                kl_sum += (((ratio - 1) - log_ratio) * active).mean().detach()
            total_loss /= sequence_length
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            losses.append(float(total_loss.detach()))
            divergences.append(float(kl_sum / sequence_length))
        if np.mean(divergences[-max(1, (len(ordering) + sequence_batch - 1) // sequence_batch):]) > .025:
            break
    return {'loss': float(np.mean(losses)), 'approximateKL': float(np.mean(divergences))}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_config(generator, index):
    return dict(scenario=['shelter', 'rooms', 'open','connected-rooms','corridors','multi-exit'][index % 6],
                size=float(generator.choice([8, 9, 10, 12])),
                n_boxes=int(generator.integers(5, 9)), n_ramps=2)


def train(args):
    if args.horizon % args.sequence_length:
        raise ValueError('Horizon must be divisible by recurrent sequence length')
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    original = torch.load(args.initial, map_location='cpu', weights_only=False)
    resumed = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if original.get('decisions', 0) != 0 or original['observationSize'] != 208:
        raise ValueError('The initial comparator must have zero game experience')
    source_hash = file_hash(Path(__file__).with_name('physics.py'))
    if parent.get('physicsSHA256') != source_hash:
        raise ValueError('The parent checkpoint and current physical environment differ')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(parentSHA256=file_hash(args.parent), parentDecisions=parent['decisions'],
        parentUpdates=parent.get('updates'), parentFormat=parent['format'],
        initialSourceSHA256=file_hash(args.initial), initialSourceDecisions=0,
        optimizerReset=True, newToolHead='Exactly uniform zero logits; no Keep/Press/Release prior',
        copiedParameters='Encoder first138columns+bias, GRU, movement, log_std, value; new2columns zero',
        physicsSHA256=source_hash, sourceSHA256={name:file_hash(Path(__file__).with_name(name))
            for name in ['persistent_actor.py','persistent_train.py','physics.py','env_pool.py']})
    if resumed:
        if resumed['format'] != FORMAT or resumed['provenance']['parentSHA256'] != provenance['parentSHA256'] or \
            resumed['provenance']['physicsSHA256'] != source_hash or \
            resumed['provenance']['initialSourceSHA256'] != provenance['initialSourceSHA256']:
            raise ValueError('Continuation must preserve the original pilot parent, initial model and physics')
        current_sources = provenance['sourceSHA256']
        provenance = dict(resumed['provenance'])
        provenance['continuations'] = [*provenance.get('continuations', []), dict(
            checkpointSHA256=file_hash(args.resume), pilotDecisions=resumed['pilotDecisions'],
            optimizerRestored=True, rolloutEnvironmentsRestarted=True, sourceSHA256=current_sources)]
    models = [PersistentActor().warm_start(state) for state in parent['models']]
    initial_models = [PersistentActor().warm_start(state) for state in original['models']]
    initial_saved = dict(format=FORMAT, observationSize=210, physicsObservationSize=208,
        models=[m.state_dict() for m in initial_models], decisions=0, parentDecisions=0,
        pilotDecisions=0, provenance={**provenance,'parentSHA256':file_hash(args.initial),
        'parentDecisions':0,'parentUpdates':0}, seed=original.get('seed'))
    if not resumed:
        torch.save(initial_saved, output/'initial.pt')
        torch.save(dict(format=FORMAT,observationSize=210,physicsObservationSize=208,
            models=[m.state_dict() for m in models],decisions=parent['decisions'],
            parentDecisions=parent['decisions'],pilotDecisions=0,provenance=provenance),output/'warm-start.pt')
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5) for model in models]
    if resumed:
        for model, state in zip(models, resumed['models']): model.load_state_dict(state)
        for optimizer, state in zip(optimizers, resumed['optimizers']): optimizer.load_state_dict(state)
        torch.set_rng_state(resumed['torchRNG'])
        generator.bit_generator.state = resumed['numpyRNG']
        print(json.dumps(dict(resumeSHA256=file_hash(args.resume), pilotDecisions=resumed['pilotDecisions'],
            optimizerRestored=True, rolloutEnvironmentsRestarted=True)), flush=True)
    configs=[dict(seed=int(generator.integers(1,2**30)),**environment_config(generator,i)) for i in range(args.envs)]
    with PhysicsEnvPool(configs, workers=args.workers) as pool:
        physical=pool.observations.copy();buttons=np.zeros((args.envs,2,2),np.float32)
        memories=[torch.zeros(args.envs,64) for _ in range(2)];starts=torch.ones(args.envs)
        decisions=resumed['pilotDecisions'] if resumed else 0
        episodes=resumed['pilotEpisodes'] if resumed else 0
        start_updates=resumed['pilotUpdates'] if resumed else 0
        log=list(resumed['log']) if resumed else []
        recent=[];started=time.monotonic()-(resumed['seconds'] if resumed else 0)
        retained_updates={int(value) for value in args.retain_updates.split(',') if value}
        for local_update in range(args.updates):
            update=start_updates+local_update
            buffers=[]
            for role in range(2):
                h,n=args.horizon,args.envs
                buffers.append([torch.zeros(h,n,210),torch.zeros(h,n,64),torch.zeros(h,n),
                    torch.zeros(h,n,5)]+[torch.zeros(h,n) for _ in range(4)])
            reward_sum=np.zeros(2);command_counts=np.zeros((2,2,3),np.int64)
            for step in range(args.horizon):
                observations=augment(physical,buttons)
                applied=np.zeros((args.envs,2,5),np.float32)
                for role,model in enumerate(models):
                    memories[role]*=1-starts[:,None]
                    observation=torch.from_numpy(observations[:,role].copy())
                    with torch.no_grad():
                        movement,commands,raw,logp,value,next_memory=model.act(observation,memories[role])
                    b=buffers[role]
                    b[0][step],b[1][step],b[2][step]=observation,memories[role],starts
                    b[3][step],b[4][step],b[6][step]=raw,logp,value
                    memories[role]=next_memory
                    blind=(physical[:,role,5]<1) if role==1 else np.zeros(args.envs,bool)
                    buttons[:,role]=advance_buttons(buttons[:,role],commands.numpy(),blind)
                    applied[:,role,:3]=movement.numpy();applied[:,role,3:]=buttons[:,role]
                    applied[blind,role]=0
                    for tool in range(2):
                        command_counts[role,tool]+=np.bincount(commands.numpy()[~blind,tool],minlength=3)
                next_physical,rewards,done,infos=pool.step(applied)
                if not np.isfinite(next_physical).all() or not np.isfinite(rewards).all():
                    raise FloatingPointError('Nonfinite rollout')
                reward_sum+=rewards.sum(axis=0)
                for role in range(2):
                    buffers[role][5][step]=torch.from_numpy(rewards[:,role].copy())*.05
                    buffers[role][7][step]=torch.from_numpy(done.astype(np.float32))
                finished=np.flatnonzero(done).tolist()
                physical=next_physical
                if finished:
                    for index in finished:
                        info=infos[index];recent.append({'hiddenFraction':info['hidden']/info['play_steps'],
                            'grabs':info['grabs'],'locks':info['locks'],'path':info['path']})
                    recent=recent[-128:];episodes+=len(finished)
                    seeds=[int(generator.integers(1,2**30)) for _ in finished]
                    changes=[environment_config(generator,index) for index in finished]
                    physical[finished]=pool.reset_at(finished,seeds,changes)
                    buttons[finished]=0
                starts=torch.from_numpy(done.astype(np.float32))
            updates=[]
            observations=augment(physical,buttons)
            for role,model in enumerate(models):
                with torch.no_grad():
                    bootstrap=model(torch.from_numpy(observations[:,role].copy()),memories[role]*(1-starts[:,None]))[2]
                updates.append(recurrent_update(model,optimizers[role],buffers[role],bootstrap,
                    epochs=args.epochs,sequence_length=args.sequence_length,entropy_weight=args.entropy,
                    sequence_batch=args.sequence_batch,role=role))
            decisions+=args.horizon*args.envs*2
            if (update+1)%args.log_every==0 or local_update+1==args.updates:
                row=dict(update=update+1,pilotDecisions=decisions,parentDecisions=parent['decisions'],
                    totalDecisions=parent['decisions']+decisions,pilotEpisodes=episodes,
                    seconds=round(time.monotonic()-started,3),hider=updates[0],seeker=updates[1],
                    meanHiddenFraction=float(np.mean([r['hiddenFraction'] for r in recent])) if recent else None,
                    commandCounts=command_counts.tolist(),unscaledRewardSum=reward_sum.tolist())
                log.append(row);print(json.dumps(row),flush=True)
            if (update+1)%args.save_every==0 or local_update+1==args.updates or update+1 in retained_updates:
                saved=dict(format=FORMAT,observationSize=210,physicsObservationSize=208,
                    models=[m.state_dict() for m in models],optimizers=[o.state_dict() for o in optimizers],
                    decisions=parent['decisions']+decisions,parentDecisions=parent['decisions'],
                    pilotDecisions=decisions,pilotUpdates=update+1,pilotEpisodes=episodes,
                    seconds=time.monotonic()-started,provenance=provenance,log=log,arguments=vars(args),
                    torchRNG=torch.get_rng_state(),numpyRNG=generator.bit_generator.state,
                    rewardDescription='Only unchanged zero-sum physical visibility reward, uniformly scaled .05. No tool/holding bonus or minimum hold.')
                torch.save(saved,output/'latest.tmp');os.replace(output/'latest.tmp',output/'latest.pt')
                if update+1 in retained_updates:
                    torch.save(saved,output/f'checkpoint-{decisions}.pt')
                (output/'training-log.json').write_text(json.dumps(log,indent=2)+'\n')
                (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--initial',required=True)
    p.add_argument('--output',default='output/persistent-button-pilot/run');p.add_argument('--updates',type=int,default=256)
    p.add_argument('--envs',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--horizon',type=int,default=128);p.add_argument('--sequence-length',type=int,default=32)
    p.add_argument('--sequence-batch',type=int,default=64);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--entropy',type=float,default=.005);p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=940313);p.add_argument('--log-every',type=int,default=16)
    p.add_argument('--save-every',type=int,default=32)
    p.add_argument('--resume',help='Continue weights, optimizers, RNG and counters; restart physical rollouts')
    p.add_argument('--retain-updates',default='',help='Comma-separated cumulative pilot update numbers to freeze')
    train(p.parse_args())
