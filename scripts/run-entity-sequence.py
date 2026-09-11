"""Finish an already approved native encoder comparison without an active agent.

Wait for the existing legacy process, validate its completed checkpoint, then
run the entity arm and the two declared read-only milestone evaluations. This
never selects a policy or writes public model assets.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
from persistent_train import file_hash


def wait_for_exit(pid):
    # The local desktop is macOS. A kernel process-exit event needs no periodic
    # sampling, status requests, or active assistant turn.
    with closing(select.kqueue()) as events:
        event = select.kevent(pid, filter=select.KQ_FILTER_PROC,
                              flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                              fflags=select.KQ_NOTE_EXIT)
        try:
            events.control([event], 0, 0)
        except ProcessLookupError:
            return
        events.control(None, 1)


def main(args):
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    run = protocol_path.parent
    status_path = run / 'sequence-state.json'
    state = dict(launcherPID=os.getpid(), legacyPID=args.legacy_pid,
                 protocolSHA256=file_hash(protocol_path), events=[])

    def record(event, **details):
        row = dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **details)
        state['events'].append(row)
        state['status'] = event
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(status_path)
        print(json.dumps(row), flush=True)

    def run_child(name, command, log):
        with open(log, 'w') as output:
            child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            state[name + 'PID'] = child.pid
            record(name + '-running', pid=child.pid, command=command, log=str(log))
            code = child.wait()
        if code:
            raise RuntimeError(f'{name} exited with status {code}; inspect {log}')
        record(name + '-complete', exitCode=code)

    def validate_checkpoint(arm):
        path = run / arm / 'checkpoint-15990784.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved['totalPolicyInteractions'] != protocol['training']['target_interactions']
                or saved['arguments']['encoder'] != arm
                or saved['provenance']['protocolSHA256'] != file_hash(protocol_path)):
            raise ValueError(f'{arm} did not complete the declared phase')
        record(arm + '-checkpoint-verified', sha256=file_hash(path), file=str(path))

    try:
        record('waiting-for-legacy-exit')
        wait_for_exit(args.legacy_pid)
        validate_checkpoint('legacy')
        for entry in protocol['assets'].values():
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('A frozen native dependency changed')
        if file_hash(ROOT / 'training/physics.py') != protocol['physicsSHA256']:
            raise ValueError('Frozen native physics changed')
        command = [sys.executable, str(ROOT / 'training/train_entity.py'),
            '--parent', protocol['assets']['entity']['path'],
            '--critic', protocol['assets']['critic']['path'],
            '--initial', protocol['assets']['initial']['path'],
            '--history', *[protocol['assets'][name]['path'] for name in protocol['history']],
            '--encoder', 'entity', '--protocol', str(protocol_path), '--output', str(run / 'entity')]
        for field, value in protocol['training'].items():
            command.extend(['--' + field.replace('_', '-'), str(value)])
        run_child('entity', command, run / 'entity.log')
        validate_checkpoint('entity')
        cache = run / 'evaluation-zero/evaluation.json'
        for milestone in protocol['milestones'][1:]:
            destination = run / f'evaluation-{milestone}'
            command = [sys.executable, str(ROOT / 'training/evaluate_entity.py'),
                '--protocol', str(protocol_path), '--milestone', str(milestone),
                '--legacy', str(run / 'legacy' / f'checkpoint-{milestone}.pt'),
                '--entity', str(run / 'entity' / f'checkpoint-{milestone}.pt'),
                '--cache', str(cache), '--workers', '2', '--output', str(destination)]
            run_child(f'evaluation-{milestone}', command, run / f'evaluation-{milestone}.log')
            cache = destination / 'evaluation.json'
        record('complete-awaiting-human-and-agent-review', publicAssetsChanged=False)
    except BaseException as error:
        record('stopped-with-error', error=repr(error), publicAssetsChanged=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--legacy-pid', required=True, type=int)
    main(parser.parse_args())
