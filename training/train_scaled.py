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


def prepare(source_path, output, target):
    prepared = output/'prepared'
    if prepared.exists():
        setup = json.loads((prepared/'SETUP.json').read_text())
        if setup['sourceSHA256'] != file_hash(source_path) or setup['target'] != target:
            raise ValueError('Preserve this run configuration; branch or extend with train_saved.py')
        return setup
    source_models, source = load_pair(source_path)
    torch.manual_seed(1091251)
    large = [widen_actor(m) for m in source_models]
    critic = widen_critic(source['centralCritic'])
    parent = copy.deepcopy(source)
    parent.update(models=[m.state_dict() for m in large], newActorUpdates=0,
        optimizers=[torch.optim.Adam(m.parameters(),lr=.0001,eps=1e-5).state_dict() for m in large],
        torchRNG=torch.get_rng_state())
    parent['provenance']['roleSources'] = copy.deepcopy(source['provenance'].get('inheritedRoleSources', []))
    parent['provenance']['encoderPreparation'] = dict(sourceSHA256=file_hash(source_path),
        description='Function-preserving expansion of own learned entity actors; fresh optimizer moments.',
        hiddenSize=256, encoderSize=256, embeddingSize=128, optimizerReset=True)
    critic_source = dict(centralCritic=critic.state_dict(), centralCriticSchema=source['centralCriticSchema'],
                        provenance=copy.deepcopy(parent['provenance']))
    prepared.mkdir(parents=True)
    torch.save(parent,prepared/'entity.pt');torch.save(critic_source,prepared/'critic.pt')
    # Historical training opponents include the current source in addition to
    # earlier own snapshots. Current-v-current self-play still receives half
    # of the episodes, so neither side is tied to a fixed opponent strategy.
    histories = list(source['arguments']['history']) + [str(source_path)]
    unique = {}
    for path in histories: unique[file_hash(path)] = str(Path(path).resolve())
    histories = list(unique.values())
    initial = str(Path(source['arguments']['initial']).resolve())
    args = training_parser().parse_args([
        '--parent',str(prepared/'entity.pt'),'--critic',str(prepared/'critic.pt'),
        '--initial',initial,'--history',*histories,'--output',str(output/'run'),
        '--protocol',str(prepared/'PROTOCOL.json'),'--encoder','entity',
        '--envs','128','--workers','4','--horizon','256','--sequence-length','128',
        '--sequence-batch','32','--target-interactions',str(target),
        '--save-every','1','--retain-updates','', '--seed','1091251'])
    fields = ['seed','arm','envs','workers','horizon','sequence_length','sequence_batch',
              'critic_batch_size','epochs','learning_rate','critic_learning_rate','entropy','kl_limit','target_interactions']
    protocol = dict(format='hide-seek-capacity-continuation-v1',
        physicsSHA256=source['provenance']['physicsSHA256'], training={k:getattr(args,k) for k in fields},
        comparisons='Capacity and recurrent horizon changed together in a longer continuation. Fixed-reference comparisons measure combined improvement, not an isolated causal architecture effect.',
        sourceCheckpointSHA256=file_hash(source_path),
        architecture=[dict(hidden=m.hidden_size,encoder=m.encoder_size,embedding=m.encoder.embedding_size,
            actorParameters=sum(p.numel() for k,p in m.named_parameters() if not k.startswith('value.')))
            for m in large],
        reward='Unchanged visibility-only zero-sum game; no reward for moving, grabbing, locking, or building.',
        assets={}, history=[])
    for key, path in [('entity',args.parent),('critic',args.critic),('initial',initial)]:
        protocol['assets'][key]=dict(path=path,sha256=file_hash(path))
    for i,path in enumerate(histories):
        key=f'history-{i}';protocol['history'].append(key)
        protocol['assets'][key]=dict(path=path,sha256=file_hash(path))
    atomic_json(prepared/'PROTOCOL.json',protocol)
    # Reuse the declared development cohort. These maps are not a final test.
    prior_protocol=json.loads(Path(source['arguments']['protocol']).read_text())
    atomic_json(prepared/'cohort.json',dict(maps=prior_protocol['maps']))
    copy_immutable(source_path,prepared/'reference.pt')
    setup=dict(sourceSHA256=file_hash(source_path),target=target,arguments=vars(args),
               createdUTC=utc_now(),architecture=protocol['architecture'])
    for name in ['train_scaled.py', 'widen_policy.py']:
        copy_immutable(ROOT/'training'/name, prepared/'controller-source'/name)
    atomic_json(prepared/'SETUP.json',setup)
    return setup


def main(options):
    output=Path(options.output).resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    setup=prepare(Path(options.source).resolve(),output,options.target)
    args=argparse.Namespace(**setup['arguments'])
    args.graceful_worker_signals=True
    args.training_description='Original enlarged entity actors; 256-unit recurrent memory; 128-step PPO sequences; unchanged visibility-only rewards.'
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
            command=[sys.executable,str(ROOT/'training/evaluate_saved.py'),
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
        retain=steps%1048576==0 or steps==65536 or steps>=options.target or stopped
        if retain:
            path=output/'checkpoints'/f'{steps}.pt'
            digest=copy_immutable(directory/'latest.pt',path)
            if steps not in status['snapshots']:status['snapshots'].append(steps)
            save(latestSnapshot=dict(steps=steps,path=str(path),sha256=digest))
        save(phase='training',interactions=steps,trainingSeconds=saved['seconds'],
             currentPolicyDecisions=saved['currentPolicyDecisions'],activePolicySamples=saved['activePolicySamples'],
             lastUpdate=saved['log'][-1])
        if not stopped and (steps==65536 or steps%5242880==0 or steps>=options.target):
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
        save(phase='paused' if stopped else 'completed-awaiting-review')
    except BaseException as error:
        save(phase='failed',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--target',type=int,default=104857600)
    a=p.parse_args()
    if a.target<65536 or a.target%65536:p.error('Use complete 65,536-interaction PPO batches')
    main(a)
