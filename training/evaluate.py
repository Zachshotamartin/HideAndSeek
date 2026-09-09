"""Evaluation-only held-out arenas. Never passes true maps to any controller."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from train import ActorCritic,Baseline
from sim import BatchArena,generate_arena,PLAY_STEPS

def load(path):
    ck=torch.load(path,weights_only=False);models=[ActorCritic(),ActorCritic()]
    for r in range(2):models[r].load_state_dict(ck['models'][r]);models[r].eval()
    return models,ck

def evaluate(models,role,n=100,seed=9000000,opponent='reactive'):
    torch.manual_seed(722);env=BatchArena(n,482,1);rng=np.random.default_rng(184);rows=[];done=np.zeros(n,bool)
    for i in range(n):
        w=int(rng.integers(10,19));h=int(rng.integers(10,17));c=int(rng.integers(2,10));arena,_=generate_arena(seed+i,w,h,c);env.reset(i,arena)
    obs=env.observe();baseline=Baseline(n,672);first=np.full(n,PLAY_STEPS,int)
    for _ in range(204):
        actions=np.zeros((n,2),int)
        with torch.no_grad():
            for r in range(2):
                if r==role or opponent=='self':actions[:,r]=models[r](torch.from_numpy(obs[:,r]))[0].sample().numpy()
                else:actions[:,r]=baseline.act(obs[:,r],r,opponent)
        obs,reward,ends,infos=env.step(actions,auto_reset=False)
        first=np.where((~done)&env.visible&(env.t>=env.prep)&(first==PLAY_STEPS),np.maximum(0,env.t-env.prep),first)
        for info in infos:
            i=info['index']
            if not done[i]:info['first_detection']=int(first[i]);rows.append(info);done[i]=True
        if done.all():break
    capture=float(np.mean([r['capture'] for r in rows]));steps=float(np.mean([r['play_steps'] for r in rows]));hidden=float(np.mean([r['hidden_fraction']*r['play_steps']/PLAY_STEPS for r in rows]));collisions=np.mean([r['collisions'] for r in rows],axis=0).tolist()
    return {'role':['hider','seeker'][role],'opponent':opponent,'episodes':len(rows),'seedRange':[seed,seed+n-1],'captureRate':round(capture,4),'hiderSurvivalRate':round(1-capture,4),'meanPlaySteps':round(steps,2),'hiddenFractionOfFullRound':round(hidden,4),'meanFirstDetectionStep':round(float(np.mean([r['first_detection'] for r in rows])),2),'meanCollisions':np.round(collisions,2).tolist(),'meanDistance':np.round(np.mean([r['distance'] for r in rows],axis=0),2).tolist()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--count',type=int,default=100);p.add_argument('--seed',type=int,default=9000000);p.add_argument('--output');a=p.parse_args();models,ck=load(a.checkpoint)
    out={'checkpoint':a.checkpoint,'trainingSteps':ck.get('steps',0),'results':[]}
    for role,opponent in [(0,'reactive'),(1,'walker'),(1,'reactive'),(0,'self')]:
        row=evaluate(models,role,a.count,a.seed,opponent);out['results'].append(row);print(json.dumps(row),flush=True)
    if a.output:Path(a.output).write_text(json.dumps(out,indent=2))
