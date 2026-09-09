"""PPO-Clip training. Opponents see exactly the same 23 actor inputs.
No expert actions, paths, hidden opponent coordinates, or map IDs enter actors.
"""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from sim import BatchArena, OBS_SIZE

torch.set_num_threads(1)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(OBS_SIZE,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,5))
        self.critic=nn.Sequential(nn.Linear(OBS_SIZE,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,1))
        for net in [self.actor,self.critic]:
            for layer in net:
                if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,np.sqrt(2));nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01)
    def forward(self,x):return Categorical(logits=self.actor(x)), self.critic(x).squeeze(-1)

class Baseline:
    """Persistent walker, observation-only chase/search or flee. No map access."""
    def __init__(self,n,seed=2):self.rng=np.random.default_rng(seed);self.actions=self.rng.integers(1,5,n)
    def act(self,obs,role,kind='reactive'):
        n=len(obs);a=self.actions.copy();change=self.rng.random(n)<.09
        # Cardial directions use ray indices up6,right0,down2,left4.
        rays=obs[:,[6,0,2,4]]
        blocked=rays[np.arange(n),np.clip(a-1,0,3)]<.12
        change|=blocked;a[change]=self.rng.integers(1,5,np.count_nonzero(change))
        if kind=='stationary':a[:]=0
        elif kind=='reactive':
            visible=obs[:,8]>.5;known=obs[:,11]>.5
            target=np.where(visible[:,None],obs[:,9:11],obs[:,12:14])
            use=visible if role==0 else (visible|known)
            if role==0:target=-target
            scores=np.stack([-target[:,1],target[:,0],target[:,1],-target[:,0]],axis=1)
            scores[rays<.12]=-20
            choices=scores.argmax(1)+1;a[use]=choices[use]
        self.actions=a;return a

def ppo_update(model,optimizer,buff,last,epochs=4,entropy=.016):
    obs,actions,logps,rewards,values,dones=buff
    advantages=torch.zeros_like(rewards);gae=torch.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        nxt=last if t==len(rewards)-1 else values[t+1]
        live=1-dones[t];delta=rewards[t]+.99*nxt*live-values[t]
        gae=delta+.99*.95*live*gae;advantages[t]=gae
    returns=(advantages+values).flatten();advantages=advantages.flatten();advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    obs=obs.flatten(0,1);actions=actions.flatten();logps=logps.flatten();losses=[];kl=0
    for _ in range(epochs):
        order=torch.randperm(len(actions))
        for indexes in order.split(512):
            distribution,value=model(obs[indexes]);logp=distribution.log_prob(actions[indexes]);ratio=(logp-logps[indexes]).exp()
            policy_loss=-torch.minimum(ratio*advantages[indexes],ratio.clamp(.8,1.2)*advantages[indexes]).mean()
            value_loss=(value-returns[indexes]).square().mean()
            loss=policy_loss+.5*value_loss-entropy*distribution.entropy().mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            kl=((ratio-1)-(logp-logps[indexes])).mean().item();losses.append(float(loss.item()))
        if kl>.03:break
    return float(np.mean(losses)),kl

def train(args):
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    models=[ActorCritic(),ActorCritic()];optimizers=[torch.optim.Adam(m.parameters(),lr=3e-4,eps=1e-5) for m in models]
    log=[];start=time.time();total=0;episodes=0;offset=0
    if args.resume:
        checkpoint=torch.load(args.resume,weights_only=False)
        for r in range(2):models[r].load_state_dict(checkpoint['models'][r]);optimizers[r].load_state_dict(checkpoint['optimizers'][r])
        log=checkpoint['log'];total=checkpoint['steps'];episodes=checkpoint['episodes'];offset=checkpoint['updates'];start-=checkpoint['seconds']
    else:
        torch.save({'models':[m.state_dict() for m in models]},output/'initial.pt')
    env=BatchArena(args.envs,args.seed+offset,0 if not args.resume else 1);obs=env.observe();baseline=Baseline(args.envs,args.seed+5)
    pool=[copy.deepcopy(m) for m in models]
    if args.resume:env.map_hashes.update(checkpoint.get('mapHashes',[]))
    recent=[]
    for local in range(args.updates):
        update=local+offset;stage=0 if update<args.curriculum else 1
        if env.stage!=stage:
            env.stage=stage
            for i in range(args.envs):env.reset(i)
            obs=env.observe()
        # Alternate role updates. Curriculum starts seeker against stationary/random targets;
        # then hider against the fixed visible-only pursuer. Self-play includes a fixed
        # baseline quarter of batches, and snapshots so the opponent isn't moving per PPO update.
        role=1 if update<args.curriculum//2 else (update%2)
        opponent=1-role
        fixed=update<args.curriculum or update%8<2
        buffers=[torch.zeros((args.horizon,args.envs,OBS_SIZE)),torch.zeros((args.horizon,args.envs),dtype=torch.long)]+[torch.zeros((args.horizon,args.envs)) for _ in range(4)]
        for t in range(args.horizon):
            tensor=torch.from_numpy(obs[:,role]);buffers[0][t]=tensor
            with torch.no_grad():
                distribution,value=models[role](tensor);action=distribution.sample();logp=distribution.log_prob(action)
                if fixed:
                    kind=('stationary' if update<args.curriculum//4 else 'walker') if opponent==0 else 'reactive'
                    other=baseline.act(obs[:,opponent],opponent,kind)
                else:other=pool[opponent](torch.from_numpy(obs[:,opponent]))[0].sample().numpy()
            actions=np.empty((args.envs,2),int);actions[:,role]=action.numpy();actions[:,opponent]=other
            obs,rewards,done,infos=env.step(actions)
            buffers[1][t]=action;buffers[2][t]=logp;buffers[3][t]=torch.from_numpy(rewards[:,role]);buffers[4][t]=value;buffers[5][t]=torch.from_numpy(done.astype(np.float32))
            episodes+=len(infos);recent.extend(infos);recent=recent[-200:]
        with torch.no_grad():last=models[role](torch.from_numpy(obs[:,role]))[1]
        loss,kl=ppo_update(models[role],optimizers[role],buffers,last,entropy=.018 if stage==0 else .012)
        total+=args.envs*args.horizon
        if update%16==15:pool=[copy.deepcopy(m) for m in models]
        if local%10==0 or local==args.updates-1:
            row={'update':update+1,'role':['hider','seeker'][role],'steps':total,'episodes':episodes,'seconds':round(time.time()-start,2),'capture':round(float(np.mean([r['capture'] for r in recent])),3) if recent else None,'hidden':round(float(np.mean([r['hidden_fraction'] for r in recent])),3) if recent else None,'loss':round(loss,4),'kl':round(kl,4)}
            log.append(row);print(json.dumps(row),flush=True)
        if local%args.save_every==args.save_every-1 or local==args.updates-1:
            checkpoint={'models':[m.state_dict() for m in models],'optimizers':[o.state_dict() for o in optimizers],'steps':total,'episodes':episodes,'seconds':time.time()-start,'updates':update+1,'log':log,'seed':args.seed,'mapHashes':sorted(env.map_hashes)}
            torch.save(checkpoint,output/'latest.pt');torch.save(checkpoint,output/f'checkpoint-{update+1:04d}.pt');(output/'training-log.json').write_text(json.dumps(log,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=400);p.add_argument('--curriculum',type=int,default=120);p.add_argument('--envs',type=int,default=48);p.add_argument('--horizon',type=int,default=128);p.add_argument('--seed',type=int,default=2709);p.add_argument('--output',default='training/runs/ppo');p.add_argument('--resume');p.add_argument('--save-every',type=int,default=400);train(p.parse_args())
