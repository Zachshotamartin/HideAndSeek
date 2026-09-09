"""Portable frozen actor parameters: no optimizer, critic or environment state."""
import argparse,json
from pathlib import Path
from evaluate import load

def export(path,label):
    models,ck=load(path);policies=[]
    for model in models:
        layers=[]
        for i in [0,2,4]:
            l=model.actor[i];layers.append({'input':l.in_features,'output':l.out_features,'weights':[round(float(v),7) for v in l.weight.detach().flatten()],'bias':[round(float(v),7) for v in l.bias.detach()]})
        policies.append({'layers':layers})
    return {'format':'hide-seek-ppo-v1','label':label,'policies':policies},ck
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',default='public/models/hide-seek.json');a=p.parse_args()
    trained,ck=export(a.checkpoint,'Pretrained pair');initial,_=export('training/runs/final/initial.pt','Before training')
    data={'version':1,'trained':trained,'initial':initial,'training':{'algorithm':'PPO-Clip','steps':ck.get('steps',0),'episodes':ck.get('episodes',0),'seconds':round(ck.get('seconds',0),1),'seed':ck.get('seed',2709)},'evaluation':json.loads(Path('evaluation.json').read_text()) if Path('evaluation.json').exists() and 'hide-seek' in Path('evaluation.json').read_text() else None}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,separators=(',',':')))
