"""One-update training smoke test on a synthetic pair in a temporary directory."""
import argparse,json,tempfile
from pathlib import Path
from train_entity import train
from synthetic import prepared_setup
if __name__=='__main__':
 with tempfile.TemporaryDirectory() as folder:
  setup=prepared_setup(folder,target=1024);a=argparse.Namespace(**setup['arguments'])
  a.envs=4;a.workers=2;a.horizon=64;a.sequence_length=32;a.burn_in=8;a.snapshot_every=1;a.archive_limit=13;a.sequence_batch=4;a.target_interactions=1024;a.stop_after_updates=1
  p=json.loads(Path(a.protocol).read_text());p['training']={k:getattr(a,k) for k in p['training']};Path(a.protocol).write_text(json.dumps(p))
  train(a);print('smoke: one synthetic update completed')
