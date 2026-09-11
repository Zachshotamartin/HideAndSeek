import unittest,tempfile,json,argparse,copy
from pathlib import Path
import torch
from train_entity import train
class ExactTraining(unittest.TestCase):
 def test_split_optimizer_and_physics_match(self):
  setup=json.loads(Path('output/v5-smoke/prepared/SETUP.json').read_text())
  with tempfile.TemporaryDirectory() as folder:
   def args(name):
    a=argparse.Namespace(**setup['arguments']);a.envs=2;a.workers=1;a.horizon=64;a.sequence_length=32;a.sequence_batch=4;a.epochs=1;a.kl_limit=1;a.target_interactions=512;a.stop_after_updates=None
    a.output=str(Path(folder)/name);p=json.loads(Path(a.protocol).read_text());p['training']={k:getattr(a,k) for k in p['training']};a.protocol=str(Path(folder)/(name+'.json'));Path(a.protocol).write_text(json.dumps(p));return a
   whole=args('whole');train(whole);w=torch.load(Path(whole.output)/'latest.pt',weights_only=False)
   split=args('split');split.stop_after_updates=1;train(split);split.resume=str(Path(split.output)/'latest.pt');split.stop_after_updates=None;train(split);s=torch.load(split.resume,weights_only=False)
   for wm,sm in zip(w['models'],s['models']):
    for k in wm:torch.testing.assert_close(wm[k],sm[k],atol=0,rtol=0)
   for x,y in zip(w['rolloutState']['memories'],s['rolloutState']['memories']):torch.testing.assert_close(x,y,atol=0,rtol=0)
if __name__=='__main__':unittest.main()
