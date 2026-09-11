"""Matched training ablations, followed by exact continuation of the full system.

No browser asset promotion, tool-use reward or agent monitoring is involved.
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from checkpoint_store import atomic_json, utc_now
from protocol import archive_exercised
from train_scaled import ARCHIVE_LIMIT, SNAPSHOT_EVERY, VARIANTS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BATCH = 65536
ATTEMPTS = 3
RETRY_DELAY = 30
INTERPRETATION = ('Matched ablations across seeds; fixed opponents, identical cohorts. Full-system continuation; '
                  'no claim of qualified tool use.')


class Suite:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / '.suite.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.file = self.out / 'SUITE.json'
        if self.file.exists():
            self.state = json.loads(self.file.read_text())
        else:
            self.state = dict(config=vars(args), trials=[], publication='No automatic browser promotion')
        if self.state['config'] != vars(args):
            raise ValueError('Resume with identical suite configuration')
        if not archive_exercised(args.pilot_updates, SNAPSHOT_EVERY, 1, ARCHIVE_LIMIT):
            raise ValueError('Pilot length must exceed the opponent archive limit so the archive ablation is exercised')
        self.stop = False
        self.child = None

    def save(self, **changes):
        self.state.update(changes, pid=os.getpid(), updatedUTC=utc_now())
        atomic_json(self.file, self.state)

    def halt(self, sig, frame):
        self.stop = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(sig)

    # -------------------------------------------------------------- one trial
    def finished(self, name, pilot):
        status = self.out / name / 'STATUS.json'
        if not status.exists():
            return False
        phase = json.loads(status.read_text())['phase']
        return phase == 'completed-awaiting-review' or (pilot and phase == 'pilot-complete')

    def command(self, variant, seed, name, pilot):
        args = self.args
        command = [sys.executable, str(HERE / 'train_scaled.py'), '--source', args.source, '--cohort', args.cohort,
                   '--output', str(self.out / name), '--seed', str(seed), '--variant', variant, '--target', str(args.total_steps)]
        if args.history is not None:
            command += ['--history', *args.history]
        if args.heldout:
            command += ['--heldout', args.heldout]
        if pilot:
            command += ['--stop-after-updates', str(args.pilot_updates)]
        return command

    def launch(self, name, command, attempt):
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        with (self.out / (name + '.log')).open('a') as log:
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
            self.save(childPID=self.child.pid, attempt=attempt)
            code = self.child.wait()
            self.child = None
        return code

    def run(self, variant, seed, pilot):
        """Train one trial to completion; the controller resumes from its own latest checkpoint."""
        name = f'{variant}-seed-{seed}'
        if self.finished(name, pilot):
            return
        command = self.command(variant, seed, name, pilot)
        self.save(phase='training', current=name, stage='pilot' if pilot else 'continuation')
        for attempt in range(ATTEMPTS):
            code = self.launch(name, command, attempt)
            if not code or self.stop:
                return
            # A repeatable failure stops the suite.
            self.save(phase='retrying', lastExitCode=code)
            time.sleep(RETRY_DELAY)
        raise RuntimeError(f'{name} failed {ATTEMPTS} times; see log')

    # ---------------------------------------------------------------- pilots
    def record_pilot(self, variant, seed):
        name = f'{variant}-seed-{seed}'
        report_path = self.out / name / 'evaluations' / str(self.args.pilot_updates * BATCH) / 'evaluation.json'
        report = json.loads(report_path.read_text())
        hider = report['summary']['candidate-hider']['hiddenFraction']
        seeker = 1 - report['summary']['candidate-seeker']['hiddenFraction']
        row = dict(name=name, variant=variant, seed=seed, hiderUtility=hider, seekerUtility=seeker,
                   score=(hider + seeker) / 2, contrasts=report['contrasts'])
        self.state['trials'] = [x for x in self.state['trials'] if x['name'] != name] + [row]
        self.save(phase='pilot-evaluated')

    def pilots(self):
        """Every variant on every seed for the pilot budget; False when interrupted."""
        for variant in VARIANTS:
            for seed in self.args.seeds:
                self.run(variant, seed, True)
                if self.stop:
                    return False
                self.record_pilot(variant, seed)
        return True

    def select(self):
        """Continue the median full-system seed.

        Ablations are evidence for later review, not a license to discard
        changes on a noisy early development score.
        """
        trials = self.state['trials']
        comparison = {}
        for variant in VARIANTS:
            scores = [r['score'] for r in trials if r['variant'] == variant]
            comparison[variant] = dict(meanUtility=float(np.mean(scores)), medianUtility=float(np.median(scores)))
        candidates = sorted([r for r in trials if r['variant'] == 'full'], key=lambda r: r['score'])
        chosen = candidates[len(candidates) // 2]
        self.save(phase='comparison-complete', comparison=comparison, selected=chosen, interpretation=INTERPRETATION)
        return chosen

    def main(self):
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self.halt)
        try:
            if not self.pilots():
                self.save(phase='paused', childPID=None)
                return
            chosen = self.select()
            self.run('full', chosen['seed'], False)
            self.save(phase='paused' if self.stop else 'finished-awaiting-review', childPID=None)
        except BaseException as error:
            self.save(phase='failed', error=repr(error), childPID=None)
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--cohort', required=True)
    p.add_argument('--history', nargs='*', default=None)
    p.add_argument('--heldout', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=[109310, 109311, 109312])
    p.add_argument('--pilot-updates', type=int, default=160)
    p.add_argument('--total-steps', type=int, default=1048576000)
    return p


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.pilot_updates < 1 or arguments.total_steps <= arguments.pilot_updates * BATCH or arguments.total_steps % BATCH:
        parser().error('Use complete PPO batches and a larger continuation budget')
    Suite(arguments).main()
