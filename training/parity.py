"""Create checked-in fixtures from the independent Python simulator and actor."""
import json
from pathlib import Path
import numpy as np
from sim import BatchArena,generate_arena,Random
from evaluate import load
import torch

def main():
    arenas=[generate_arena(s,14,12,6)[0] for s in [1,2709,9013,17761]]
    env=BatchArena(1);env.reset(0,arenas[1]);rng=Random(824);states=[]
    for t in range(90):
        actions=[rng.integer(5),rng.integer(5)]
        obs,rewards,done,infos=env.step(np.array([actions]),False)
        states.append({'actions':actions,'pos':env.pos[0].tolist(),'obs':obs[0].tolist(),'rewards':rewards[0].tolist(),'visible':bool(env.visible[0])})
        if done[0]:break
    models,_=load('training/runs/final/latest.pt')
    samples=[]
    for role in range(2):
        observation=states[30]['obs'][role]
        with torch.no_grad():logits=models[role].actor(torch.tensor(observation)).tolist()
        samples.append({'role':role,'observation':observation,'logits':logits})
    Path('tests/fixtures').mkdir(exist_ok=True)
    Path('tests/fixtures/python-parity.json').write_text(json.dumps({'arenas':arenas,'states':states,'logits':samples},separators=(',',':')))
if __name__=='__main__':main()
