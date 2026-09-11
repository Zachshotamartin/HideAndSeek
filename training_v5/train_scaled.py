"""Larger original entity policies, long recurrent sequences, saved evaluations.

One native process owns the training. No strategy scripts or tool-use bonuses.
Resume with the same command; budgets can subsequently be extended using
train_saved.py, preserving complete actor/critic/optimizer checkpoints.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import torch

from checkpoint_store import atomic_json, copy_immutable, utc_now
from entity_actor import load_pair
from evaluated_models import register
from persistent_train import file_hash
from train_entity import train, parser as training_parser
from widen_policy import widen_actor, widen_critic

ROOT = Path(__file__).resolve().parents[1]


def prepare(source_path, output, target, seed=1091252, variant="full"):
    prepared = output/'prepared'
    if prepared.exists():
        setup = json.loads((prepared/'SETUP.json').read_text())
        if setup['sourceSHA256'] != file_hash(source_path) or setup['target'] != target or setup.get('seed',1091252)!=seed or setup.get('variant','full')!=variant:
            raise ValueError('Preserve this run configuration; branch or extend with train_saved.py')
        return setup
    from entity_actor import EntityActor, FORMAT, ENTITY
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    large=load_pair(source)[0]
    parent=copy.deepcopy(source)
    parent['provenance']['roleSources']=source['provenance'].get('roleSources', source['provenance'].get('inheritedRoleSources'))
    parent.update(newActorUpdates=0, optimizers=[torch.optim.Adam(m.parameters()).state_dict() for m in large])
    # Physical game is byte-identical. Only training distribution is changed.
    if parent['provenance']['physicsSHA256']!=file_hash(ROOT/'training_v5/physics.py'):
        raise ValueError('New training requires the same v4 physical contract')
    critic_source=dict(centralCritic=source['centralCritic'],centralCriticSchema=source['centralCriticSchema'],provenance=copy.deepcopy(parent['provenance']))
    torch.manual_seed(seed)
    parent['torchRNG']=torch.get_rng_state()
    prepared.mkdir(parents=True)
    torch.save(parent,prepared/'entity.pt');torch.save(critic_source,prepared/'critic.pt')
    torch.save(parent,prepared/'reference.pt')
    initial_record=copy.deepcopy(parent)
    initial_record.update(decisions=0,models=[EntityActor(m.hidden_size,m.encoder_size,m.encoder.embedding_size).state_dict() for m in large])
    torch.save(initial_record,prepared/'initial.pt')
    histories=[str(prepared/'entity.pt')];initial=str(prepared/'initial.pt')
    seen={file_hash(prepared/'entity.pt')}
    candidates=source.get('arguments',{}).get('history',[])
    for original in candidates[:3]:
        original=Path(original)
        if not original.is_absolute():original=source_path.parent/original
        if not original.exists():continue
        digest=file_hash(original)
        if digest in seen:continue
        record=torch.load(original,map_location='cpu',weights_only=False)
        if record['provenance']['physicsSHA256']!=parent['provenance']['physicsSHA256']:continue
        destination=prepared/f'history-{len(histories)}.pt';copy_immutable(original,destination);histories.append(str(destination));seen.add(digest)
    args = training_parser().parse_args([
        '--parent',str(prepared/'entity.pt'),'--critic',str(prepared/'critic.pt'),
        '--initial',initial,'--history',*histories,'--output',str(output/'run'),
        '--protocol',str(prepared/'PROTOCOL.json'),'--encoder','entity',
        '--envs','128','--workers','4','--horizon','256','--sequence-length','128' if variant in ('baseline','short-memory') else '256',
        '--sequence-batch','32','--target-interactions',str(target),
        '--save-every','1','--retain-updates','', '--seed',str(seed),'--variant',variant])
    fields = ['variant','seed','arm','envs','workers','horizon','sequence_length','sequence_batch',
              'critic_batch_size','epochs','learning_rate','critic_learning_rate','entropy','kl_limit','target_interactions']
    protocol=dict(format='hide-seek-balanced-long-league-v5',physicsSHA256=parent['provenance']['physicsSHA256'],
        training={k:getattr(args,k) for k in fields},sourceCheckpointSHA256=file_hash(source_path),
        comparisons='Unchanged environment and sensor contract. Longer randomized episodes, balanced active sampling and diverse archives. Inherited weights, fresh optimizers. Current versus fixed migrated reference and counterfactual tools-disabled evaluations; not an isolated architecture ablation.',
        architecture=[dict(hidden=m.hidden_size,encoder=m.encoder_size,embedding=m.encoder.embedding_size,actorParameters=sum(p.numel() for k,p in m.named_parameters() if not k.startswith('value.'))) for m in large],
        reward='Visibility-only zero-sum; zero preparation reward; no tool bonuses.',toolEntropyScale=0.1,selfPlay='70% current-current, 30% active-sample-balanced historical; fixed anchor, latest eight and behavioral diversity',assets={},history=[])
    for key,path in [('entity',args.parent),('critic',args.critic),('initial',initial)]:protocol['assets'][key]=dict(path=path,sha256=file_hash(path))
    for i,path in enumerate(histories):
        key=f'history-{i}';protocol['history'].append(key);protocol['assets'][key]=dict(path=path,sha256=file_hash(path))
    atomic_json(prepared/'PROTOCOL.json',protocol)
    # Retain the old fixed maps, add separate richer construction layouts.
    cohort=Path(source['arguments']['protocol']).parent/'cohort.json'
    maps=json.loads(cohort.read_text())['maps']
    extra=[dict(seed=1750100000+i,scenario=['connected-rooms','corridors','multi-exit'][i%3],arenaConfig=dict(size=8+(i%5),n_boxes=5+i%4,n_ramps=2)) for i in range(24)]
    atomic_json(prepared/'cohort.json',dict(maps=maps+[dict(m,arenaConfig={**m['arenaConfig'],'play':[188,375,750][i%3]}) for i,m in enumerate(extra) if m['seed'] not in {row['seed'] for row in maps}]))
    setup=dict(seed=seed,variant=variant,sourceSHA256=file_hash(source_path),target=target,arguments=vars(args),createdUTC=utc_now(),architecture=protocol['architecture'])
    for name in ['train_scaled.py','protocol.py']:copy_immutable(ROOT/'training_v5'/name,prepared/'controller-source'/name)
    atomic_json(prepared/'SETUP.json',setup)
    return setup


def main(options):
    output=Path(options.output).resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    setup=prepare(Path(options.source).resolve(),output,options.target,getattr(options,'seed',1091252),getattr(options,'variant','full'))
    args=argparse.Namespace(**setup['arguments'])
    args.graceful_worker_signals=True
    args.stop_after_updates=getattr(options,"stop_after_updates",None)
    args.training_description='Relational attention; 10 object slots; 30 rays; visibility-only reward; refreshed self-play. V5 balanced active samples, 15/30/60s randomized play, diverse opponent archive, 256-step memory with burn-in; 0.1 tool entropy scale; bounded grounded jump; 2.2m walls.'
    status_path=output/'STATUS.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else dict(
        createdUTC=utc_now(),evaluated=[],snapshots=[],architecture=setup['architecture'],
        targetInteractions=options.target,publication='No automatic browser promotion')
    stopped=False;child=None
    def save(**changes):
        status.update(changes,pid=os.getpid(),updatedUTC=utc_now());atomic_json(status_path,status)
    def stop(sig,_):
        nonlocal stopped
        stopped=True
        if child is not None and child.poll() is None: child.send_signal(sig)
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    def evaluate(path,steps):
        nonlocal child
        destination=output/'evaluations'/str(steps)
        if steps in status['evaluated']:return
        if not (destination/'evaluation.json').exists():
            command=[sys.executable,str(ROOT/'training_v5/evaluate_saved.py'),
                '--checkpoint',str(path),'--reference',str(output/'prepared/reference.pt'),
                '--cohort',str(output/'prepared/cohort.json'),'--output',str(destination),
                '--registry',str(output/'best'),'--workers','4']
            save(phase='evaluating',evaluationSteps=steps)
            with (output/f'evaluate-{steps}.log').open('a') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                code=child.wait();child=None
            if stopped:return
            if code:raise RuntimeError(f'Evaluation failed at {steps}; inspect evaluation log')
        # Registration is idempotent, including a restart between report and registry writes.
        register(path, output/'prepared/reference.pt', destination/'evaluation.json', output/'best')
        status['evaluated'].append(steps);save(phase='training')
    def checkpoint(saved,directory):
        steps=saved['totalPolicyInteractions']
        # Every ~1M interactions, plus first/last/paused: immutable full state.
        pilot_end=bool(args.stop_after_updates and saved['pilotUpdates']>=args.stop_after_updates)
        retain=pilot_end or steps%1048576==0 or steps==65536 or steps>=options.target or stopped
        if retain:
            path=output/'checkpoints'/f'{steps}.pt'
            digest=copy_immutable(directory/'latest.pt',path)
            if steps not in status['snapshots']:status['snapshots'].append(steps)
            save(latestSnapshot=dict(steps=steps,path=str(path),sha256=digest))
        save(phase='training',interactions=steps,trainingSeconds=saved['seconds'],
             currentPolicyDecisions=saved['currentPolicyDecisions'],activePolicySamples=saved['activePolicySamples'],
             lastUpdate=saved['log'][-1])
        if not stopped and (pilot_end or steps==65536 or steps%5242880==0 or steps>=options.target):
            evaluate(output/'checkpoints'/f'{steps}.pt',steps)
    try:
        latest=output/'run/latest.pt'
        if latest.exists():
            args.resume=str(latest)
            # Re-run a pending complete milestone evaluation before further PPO.
            saved=torch.load(latest,map_location='cpu',weights_only=False)
            checkpoint(saved,latest.parent)
        if not stopped:
            save(phase='training');train(args,stop_requested=lambda:stopped,on_checkpoint=checkpoint)
        save(phase='paused' if stopped else 'pilot-complete' if args.stop_after_updates else 'completed-awaiting-review')
    except BaseException as error:
        save(phase='failed',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=1091252);p.add_argument('--variant',default='full',choices=['full','baseline','short','unbalanced','recent-only','short-memory']);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stop-after-updates',type=int);p.add_argument('--target',type=int,default=1048576000)
    a=p.parse_args()
    if a.target<65536 or a.target%65536:p.error('Use complete 65,536-interaction PPO batches')
    main(a)
