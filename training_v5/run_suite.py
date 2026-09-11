"""Matched training ablations, followed by exact continuation of the full system.
No browser asset promotion, tool-use reward, or agent monitoring is involved.
"""
import argparse,fcntl,json,os,signal,subprocess,sys
from pathlib import Path
import numpy as np
from checkpoint_store import atomic_json,utc_now
from protocol import archive_exercised
from train_scaled import VARIANTS,SNAPSHOT_EVERY,ARCHIVE_LIMIT
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def main(a):
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 lock=(out/'.suite.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 file=out/'SUITE.json';state=json.loads(file.read_text()) if file.exists() else dict(config=vars(a),trials=[],publication='No automatic browser promotion')
 if state['config']!=vars(a):raise ValueError('Resume with identical suite configuration')
 if not archive_exercised(a.pilot_updates,SNAPSHOT_EVERY,1,ARCHIVE_LIMIT):raise ValueError('Pilot length must exceed the opponent archive limit so the archive ablation is exercised')
 stop=False;child=None
 def save(**changes):state.update(changes,pid=os.getpid(),updatedUTC=utc_now());atomic_json(file,state)
 def halt(sig,frame):
  nonlocal stop
  stop=True
  if child is not None and child.poll() is None:child.send_signal(sig)
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,halt)
 def run(variant,seed,pilot):
  nonlocal child
  name=f'{variant}-seed-{seed}';folder=out/name;status=folder/'STATUS.json'
  if status.exists():
   phase=json.loads(status.read_text())['phase']
   if phase=='completed-awaiting-review' or (pilot and phase=='pilot-complete'):return
  command=[sys.executable,str(HERE/'train_scaled.py'),'--source',a.source,'--cohort',a.cohort,'--output',str(folder),'--seed',str(seed),'--variant',variant,'--target',str(a.total_steps)]
  if a.history is not None:command+=['--history',*a.history]
  if a.heldout:command+=['--heldout',a.heldout]
  if pilot:command+=['--stop-after-updates',str(a.pilot_updates)]
  save(phase='training',current=name,stage='pilot' if pilot else 'continuation')
  with (out/(name+'.log')).open('a') as log:
   child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});save(childPID=child.pid)
   code=child.wait();child=None
  if code and not stop:raise RuntimeError(f'{name} failed; see log')
 try:
  for variant in VARIANTS:
   for seed in a.seeds:
    run(variant,seed,True)
    if stop:save(phase='paused',childPID=None);return
    name=f'{variant}-seed-{seed}'
    report=json.loads((out/name/'evaluations'/str(a.pilot_updates*65536)/'evaluation.json').read_text())
    h=report['summary']['candidate-hider']['hiddenFraction'];s=1-report['summary']['candidate-seeker']['hiddenFraction']
    row=dict(name=name,variant=variant,seed=seed,hiderUtility=h,seekerUtility=s,score=(h+s)/2,contrasts=report['contrasts'])
    state['trials']=[x for x in state['trials'] if x['name']!=name]+[row];save(phase='pilot-evaluated')
  comparison={v:dict(meanUtility=float(np.mean([r['score'] for r in state['trials'] if r['variant']==v])),medianUtility=float(np.median([r['score'] for r in state['trials'] if r['variant']==v]))) for v in VARIANTS}
  # Continue the median full-system seed. Ablations are evidence for later review,
  # not a license to discard changes on a noisy early development score.
  candidates=sorted([r for r in state['trials'] if r['variant']=='full'],key=lambda r:r['score']);chosen=candidates[len(candidates)//2]
  save(phase='comparison-complete',comparison=comparison,selected=chosen,interpretation='Matched ablations across seeds; fixed opponents, identical cohorts. Full-system continuation; no claim of qualified tool use.')
  run('full',chosen['seed'],False)
  save(phase='paused' if stop else 'finished-awaiting-review',childPID=None)
 except BaseException as error:save(phase='failed',error=repr(error),childPID=None);raise
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--cohort',required=True)
 p.add_argument('--history',nargs='*',default=None);p.add_argument('--heldout',default=None)
 p.add_argument('--seeds',type=int,nargs='+',default=[109310,109311,109312]);p.add_argument('--pilot-updates',type=int,default=160);p.add_argument('--total-steps',type=int,default=1048576000)
 a=p.parse_args()
 if a.pilot_updates<1 or a.total_steps<=a.pilot_updates*65536 or a.total_steps%65536:p.error('Use complete PPO batches and a larger continuation budget')
 main(a)
