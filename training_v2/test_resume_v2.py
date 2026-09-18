import argparse,copy,json,unittest
from pathlib import Path
import tempfile,torch
from train_scaled import prepare
from train_entity import train
from checkpoint_store import atomic_json
from entity_actor import load_pair
class ResumeV2(unittest.TestCase):
 def test_migration_refresh_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   output=Path(tmp);setup=prepare(Path('output/scaled-memory-256/run/latest.pt').resolve(),output,1048576000)
   args=argparse.Namespace(**setup['arguments']);args.envs=2;args.workers=1;args.horizon=32;args.sequence_length=16;args.sequence_batch=4;args.epochs=1;args.kl_limit=1.;args.target_interactions=2560
   protocol=json.loads(Path(args.protocol).read_text())
   for k in protocol['training']:protocol['training'][k]=getattr(args,k)
   atomic_json(args.protocol,protocol)
   args.stop_after_updates=16;train(args)
   path=Path(args.output)/'latest.pt';a=torch.load(path,weights_only=False)
   self.assertEqual(a['totalPolicyInteractions'],2048);self.assertEqual(len(a['leagueModels']),2)
   load_pair(a);args.resume=str(path);args.stop_after_updates=0;train(args)
   b=torch.load(path,weights_only=False);self.assertEqual(b['totalPolicyInteractions'],2560);self.assertEqual(len(b['leagueModels']),2)
   self.assertTrue(all(x['state'] for x in b['optimizers']))
   for x,y in zip(a['optimizers'],b['optimizers']):self.assertGreater(max(v['step'] for v in y['state'].values()),max(v['step'] for v in x['state'].values()))
if __name__=='__main__':unittest.main()
