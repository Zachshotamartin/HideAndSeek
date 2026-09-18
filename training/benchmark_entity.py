"""Full real-physics rollout and disposable PPO-update benchmark; no model retained.

Each rollout uses the immutable zero-update actors. PPO executes on copies to
measure actual backward/optimizer work without beginning an unapproved phase.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from central_critic import FEATURES, batch_states
from central_persistent_train import advantages_and_returns, critic_update
from entity_actor import load_pair
from env_pool import PhysicsEnvPool
from fit_central_value import REWARD_SCALE
from league_ppo import assign_roles, active_masks, act_grouped, actor_update
from persistent_train import environment_config, file_hash
from residual_critic import ResidualCentralCritic

torch.set_num_threads(1)


def benchmark(path, args):
    models, record = load_pair(path)
    histories = [load_pair(value)[0] for value in args.history]
    critic_source = torch.load(args.critic_source, map_location='cpu', weights_only=False)
    critic = ResidualCentralCritic(.1).eval().requires_grad_(False)
    critic.load_state_dict(critic_source['centralCritic'])
    original = [copy.deepcopy(model.state_dict()) for model in models]
    rng = np.random.default_rng(args.seed)
    opponents = np.random.default_rng(args.seed + 1)
    torch.set_rng_state(record['torchRNG'])
    configurations = [dict(seed=int(rng.integers(1, 2**30)), **environment_config(rng, index))
                      for index in range(args.envs)]
    roles = assign_roles(opponents, args.envs, 'B', len(histories))
    startup = time.perf_counter()
    records = []
    with PhysicsEnvPool(configurations, workers=args.workers, with_central_state=True) as pool:
        startup = time.perf_counter() - startup
        physical = pool.observations.copy()
        memories = [torch.zeros(args.envs, 64), torch.zeros(args.envs, 64)]
        buttons = np.zeros((args.envs, 2, 2), np.float32)
        starts = torch.ones(args.envs)
        for cycle in range(args.cycles):
            h, n = args.horizon, args.envs
            buffers = [[torch.zeros(h,n,140), torch.zeros(h,n,64), torch.zeros(h,n),
                        torch.zeros(h,n,5), torch.zeros(h,n)] for _ in range(2)]
            current_rows = torch.zeros(h,n,2,dtype=torch.bool)
            central = {key: torch.zeros(h,n,*shape) for key,shape in FEATURES.items()}
            before_memories = torch.zeros(h,n,2,64)
            partial = torch.zeros(h,n,2)
            values, rewards, dones = [torch.zeros(h,n) for _ in range(3)]
            actor_time = physics_time = value_time = reset_time = 0
            counts, active_counts = np.zeros((2,2), np.int64), np.zeros(2,np.int64)
            start = time.perf_counter()
            for step in range(h):
                memories = [memory * (1 - starts[:,None]) for memory in memories]
                before = torch.stack(memories,1)
                clock = time.perf_counter()
                state = batch_states(pool.central_states)
                for key in FEATURES:
                    central[key][step] = state[key]
                before_memories[step] = before
                value_time += time.perf_counter() - clock
                current, active = active_masks(physical, roles)
                current_rows[step] = torch.from_numpy(current)
                counts[0] += current.sum(0); counts[1] += (~current).sum(0)
                active_counts += active.sum(0)
                clock = time.perf_counter()
                actions, next_memories, next_buttons, actor_records = act_grouped(
                    models, histories, physical, memories, buttons, roles)
                actor_time += time.perf_counter() - clock
                clock = time.perf_counter()
                partial[step] = torch.stack([entry[3] for entry in actor_records], -1)
                with torch.no_grad(): values[step] = critic(state, before, partial[step])
                value_time += time.perf_counter() - clock
                for role in range(2):
                    observed, raw, logp, _ = actor_records[role]
                    for destination, value in zip(buffers[role], [observed, memories[role], starts, raw, logp]):
                        destination[step] = value
                clock = time.perf_counter()
                physical, returned, done, _ = pool.step(actions)
                physics_time += time.perf_counter() - clock
                np.testing.assert_array_equal(returned.sum(1), np.zeros(n))
                rewards[step] = torch.from_numpy(returned[:,0].copy()) * REWARD_SCALE
                dones[step] = torch.from_numpy(done.astype(np.float32))
                memories, buttons = next_memories, next_buttons
                finished = np.flatnonzero(done).tolist()
                if finished:
                    clock = time.perf_counter()
                    physical[finished] = pool.reset_at(finished,
                        [int(rng.integers(1,2**30)) for _ in finished],
                        [environment_config(rng,index) for index in finished])
                    buttons[finished] = 0
                    roles[finished] = assign_roles(opponents,len(finished),'B',len(histories))
                    reset_time += time.perf_counter() - clock
                starts = torch.from_numpy(done.astype(np.float32))
            with torch.no_grad():
                bootstrap_memories = [memory * (1-starts[:,None]) for memory in memories]
                _,_,_,actor_records = act_grouped(models,histories,physical,bootstrap_memories,buttons,roles,sample=False)
                bootstrap_partial = torch.stack([entry[3] for entry in actor_records],-1)
                bootstrap = critic(batch_states(pool.central_states),torch.stack(bootstrap_memories,1),bootstrap_partial)
            advantage, returns = advantages_and_returns(rewards,values,dones,bootstrap)
            rollout_time = time.perf_counter() - start
            # Only disposable copies receive optimizer updates. All future
            # physical rollouts keep the same frozen zero-update source actors.
            updates = copy.deepcopy(models)
            for model in updates: model.train().requires_grad_(True)
            optimizers = [torch.optim.Adam(model.parameters(),lr=.0001,eps=1e-5) for model in updates]
            update_critic = copy.deepcopy(critic).train().requires_grad_(True)
            critic_optimizer = torch.optim.AdamW(update_critic.parameters(),lr=.0001)
            critic_optimizer.load_state_dict(critic_source['centralCriticOptimizer'])
            for group in critic_optimizer.param_groups: group['lr'] = .0001
            clock = time.perf_counter()
            actor_stats = [actor_update(model,optimizer,buffers[role],advantage if role==0 else -advantage,
                current_rows[:,:,role],role,epochs=2,sequence_length=32,sequence_batch=256,
                entropy_weight=.005,kl_limit=.008) for role,(model,optimizer) in enumerate(zip(updates,optimizers))]
            actor_optimizer_time = time.perf_counter() - clock
            clock = time.perf_counter()
            critic_stats = critic_update(update_critic,critic_optimizer,central,before_memories,values,returns,
                                         epochs=2,batch_size=2048,partial_values=partial)
            critic_optimizer_time = time.perf_counter() - clock
            for before, model in zip(original,models):
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value,before[name],atol=0,rtol=0)
                assert all(parameter.grad is None for parameter in model.parameters())
            total_time = rollout_time + actor_optimizer_time + critic_optimizer_time
            row = dict(cycle=cycle, warmup=cycle==0, interactions=h*n*2,
                currentDecisions=counts[0].tolist(),historicalDecisions=counts[1].tolist(),activeSamples=active_counts.tolist(),
                rolloutSeconds=rollout_time,actorInferenceSeconds=actor_time,physicsTransportSeconds=physics_time,
                valueFeatureSeconds=value_time,resetSeconds=reset_time,actorOptimizerSeconds=actor_optimizer_time,
                criticOptimizerSeconds=critic_optimizer_time,totalSeconds=total_time,
                interactionsPerSecond=h*n*2/total_time,actorStats=actor_stats,criticStats=critic_stats)
            records.append(row)
            print(json.dumps(dict(arm=Path(path).stem,**row)),flush=True)
    measured = records[1:]
    return dict(sourceSHA256=file_hash(path),startupSeconds=startup,cycles=records,
        meanInteractionsPerSecond=sum(row['interactions'] for row in measured)/sum(row['totalSeconds'] for row in measured),
        allSourceWeightsUnchanged=True,retainedOptimizerUpdates=0,
        diagnosticRolloutInteractions=sum(row['interactions'] for row in records))


def main(args):
    output = Path(args.output); output.mkdir(parents=True,exist_ok=True)
    if (output/'report.json').exists(): raise ValueError('Preserve benchmarks')
    if args.cycles < 2: raise ValueError('Warm up once and measure at least one full rollout')
    reports = {}
    for label,path in [('legacy',args.legacy),('entity',args.entity)]:
        reports[label] = benchmark(path,args)
    result = dict(format='entity-full-rollout-disposable-update-benchmark-v1',settings=vars(args),arms=reports,
        entityRelativeThroughput=reports['entity']['meanInteractionsPerSecond']/reports['legacy']['meanInteractionsPerSecond'],
        scope='Frozen source-policy rollouts and disposable real PPO updates; no updated actor or critic retained. Startup and snapshot-copy overhead reported/excluded as documented.',
        sourceSHA256=file_hash(__file__),publicAssetsChanged=False)
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['legacy','entity','critic-source','output']: parser.add_argument('--'+name,required=True)
    parser.add_argument('--history',nargs='+',required=True)
    parser.add_argument('--envs',type=int,default=128)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--horizon',type=int,default=256)
    parser.add_argument('--cycles',type=int,default=4)
    parser.add_argument('--seed',type=int,default=886711)
    main(parser.parse_args())
