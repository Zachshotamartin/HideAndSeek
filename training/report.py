"""One untouched test pass after validation-only checkpoint selection.
200 distinct geometry seeds × 3 independent spawn seeds, with geometry-hash exclusion.
"""
import json,argparse,time
from pathlib import Path
import numpy as np
import torch
from sim import BatchArena,generate_arena,arena_hash,PLAY_STEPS
from train import Baseline
from evaluate import load

def test_arenas(training_hashes,count=200):
    maps=[];seen=set();rng=np.random.default_rng(6812);seed=11000000
    while len(maps)<count:
        w=int(rng.integers(10,19));h=int(rng.integers(10,17));blocks=int(rng.integers(2,10));a,_=generate_arena(seed,w,h,blocks);key=arena_hash(a);seed+=1
        if key in training_hashes or key in seen:continue
        seen.add(key);variants=[]
        for v in range(3):
            b,_=generate_arena(a['seed'],w,h,blocks,spawn_seed=21000000+len(maps)*3+v);variants.append(b)
        maps.append(variants)
    return maps

def run(models,role,opponent,maps):
    arenas=[a for variants in maps for a in variants];n=len(arenas);env=BatchArena(n,777,1)
    for i,a in enumerate(arenas):env.reset(i,a)
    obs=env.observe();baseline=Baseline(n,903);done=np.zeros(n,bool);first=np.full(n,180,int);rows=[None]*n;torch.manual_seed(14822)
    for _ in range(204):
        actions=np.zeros((n,2),int)
        with torch.no_grad():
            for r in range(2):
                actions[:,r]=models[r](torch.from_numpy(obs[:,r]))[0].sample().numpy() if r==role or opponent=='self' else baseline.act(obs[:,r],r,opponent)
        obs,_,_,infos=env.step(actions,False)
        first=np.where((~done)&env.visible&(env.t>=env.prep)&(first==180),np.maximum(0,env.t-env.prep),first)
        for row in infos:
            i=row['index']
            if done[i]:continue
            rows[i]={'mapIndex':i//3,'spawnIndex':i%3,'capture':int(row['capture']),'survivalSteps':row['play_steps'],'unseenSteps':round(row['hidden_fraction']*row['play_steps']),'firstDetectionStep':int(first[i]),'collisions':row['collisions'],'distance':row['distance']};done[i]=True
        if done.all():break
    assert done.all()
    def mean(key):return float(np.mean([r[key] for r in rows]))
    return {'episodes':n,'arenas':len(maps),'captureRate':round(mean('capture'),4),'hiderSurvivalRate':round(1-mean('capture'),4),'meanSurvivalSteps':round(mean('survivalSteps'),2),'hiddenFractionOfFullRound':round(mean('unseenSteps')/180,4),'meanFirstDetectionStep':round(mean('firstDetectionStep'),2),'meanCollisions':np.round(np.mean([r['collisions'] for r in rows],axis=0),2).tolist(),'meanDistance':np.round(np.mean([r['distance'] for r in rows],axis=0),2).tolist(),'rows':rows}

def ci_delta(before,after,role):
    sign=-1 if role==0 else 1
    b=np.array([r['capture'] for r in before['rows']]).reshape(-1,3).mean(1);a=np.array([r['capture'] for r in after['rows']]).reshape(-1,3).mean(1)
    delta=(a-b)*sign;rng=np.random.default_rng(934);samples=delta[rng.integers(0,len(delta),(3000,len(delta)))].mean(1)
    return {'absoluteRateImprovement':round(float(delta.mean()),4),'pairedMapBootstrap95CI':np.round(np.quantile(samples,[.025,.975]),4).tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',default='evaluation.json');a=p.parse_args();trained,ck=load(a.checkpoint);initial,_=load('training/runs/final/initial.pt');maps=test_arenas(set(ck['mapHashes']));results={}
    for name,models,role,opponent in [('initialHider',initial,0,'reactive'),('trainedHider',trained,0,'reactive'),('initialSeeker',initial,1,'reactive'),('trainedSeeker',trained,1,'reactive'),('hiderVsWalker',trained,0,'walker'),('seekerVsWalker',trained,1,'walker'),('trainedPair',trained,0,'self')]:
        result=run(models,role,opponent,maps);result['role']=['hider','seeker'][role];result['opponent']=opponent;results[name]=result;print(name,json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    report={'format':'hide-seek-evaluation-v1','checkpoint':Path(a.checkpoint).name,'training':{'steps':ck['steps'],'episodes':ck['episodes'],'seconds':round(ck['seconds'],2),'updates':ck['updates'],'seed':ck['seed'],'uniqueGeometryHashes':len(ck['mapHashes']),'arenaSeedRange':[1,999999]},'protocol':{'geometryCount':200,'spawnVariants':3,'episodesPerMatchup':600,'mapSeedRange':[maps[0][0]['seed'],maps[-1][0]['seed']],'spawnSeedRange':[21000000,21000599],'geometryHashes':[arena_hash(m[0]) for m in maps],'geometryDisjointFromTraining':True,'validationSeedRange':[9000000,9000099],'widthRange':[10,18],'heightRange':[10,16],'requestedCoverRange':[2,9],'note':'Selected final checkpoint using validation only. These test seeds and independent spawn variants were evaluated after the checkpoint was frozen. Caption percentages compare separate observation-only fixed opponents, not each other. Capture requires distance <0.65 and exact unoccluded LoS. Each full play round is180 steps. Hidden fraction counts all post-capture remaining steps as0.'},'summary':{'episodesPerMatchup':600,'arenas':200,'seekerCapture':results['trainedSeeker']['captureRate'],'hiderSurvival':results['trainedHider']['hiderSurvivalRate']},'improvement':{'hider':ci_delta(results['initialHider'],results['trainedHider'],0),'seeker':ci_delta(results['initialSeeker'],results['trainedSeeker'],1)},'results':{k:{a:b for a,b in v.items() if a!='rows'} for k,v in results.items()}}
    Path(a.output).write_text(json.dumps(report,indent=2));Path('training/test-episodes.json').write_text(json.dumps({k:v['rows'] for k,v in results.items()},separators=(',',':')))
if __name__=='__main__':main()
