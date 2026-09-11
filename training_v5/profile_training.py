"""Bounded throughput comparison using the real trainer, never a training run.

Writes only an isolated temporary directory. The trainer source and running
checkpoint are left untouched; a resumed model is loaded into separate memory.
"""
import argparse
import contextlib
import io
import json
import tempfile
import time
from types import SimpleNamespace

import train_physics as trainer


def benchmark(envs, workers, sequence_batch, updates, horizon=64, sequence_length=16, resume=None):
    original_update = trainer.recurrent_update
    timings = dict(rollout_physics=0., rollout_policy=0., ppo_update=0., reset=0.)
    original_step = trainer.PhysicsEnvPool.step
    original_reset = trainer.PhysicsEnvPool.reset_at
    original_act = trainer.PhysicalActor.act
    original_run = trainer.run_training

    def measured(key, call):
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            result = call(*args, **kwargs)
            timings[key] = timings.get(key, 0.) + time.perf_counter() - started
            return result
        return wrapper

    trainer.recurrent_update = measured('ppo_update', original_update)
    trainer.PhysicsEnvPool.step = measured('rollout_physics', original_step)
    trainer.PhysicsEnvPool.reset_at = measured('reset', original_reset)
    trainer.PhysicalActor.act = measured('rollout_policy', original_act)
    trainer.run_training = measured('training_total', original_run)
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix='hide-seek-profile-') as output:
            args = SimpleNamespace(envs=envs, workers=workers, horizon=horizon,
                sequence_length=sequence_length, sequence_batch=sequence_batch,
                epochs=3, entropy=.005, learning_rate=3e-4,
                seed=632201, output=output, resume=resume, updates=updates,
                log_every=1, save_every=updates, fixed_arena=False)
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                trainer.train(args)
            rows = [json.loads(line) for line in captured.getvalue().splitlines() if line.startswith('{')]
    finally:
        trainer.recurrent_update = original_update
        trainer.PhysicsEnvPool.step = original_step
        trainer.PhysicsEnvPool.reset_at = original_reset
        trainer.PhysicalActor.act = original_act
        trainer.run_training = original_run
    wall = time.perf_counter() - started
    # Saved counters include previous runs on resume; time this process's actual
    # training loop, including bookkeeping, instead of subtracting saved counters.
    seconds = timings.pop('training_total')
    decisions = updates * horizon * envs * 2
    return dict(envs=envs, workers=workers, sequenceBatch=sequence_batch,
        updates=updates, horizon=horizon, sequenceLength=sequence_length,
        measuredSeconds=round(seconds,3), wallSeconds=round(wall,3),
        decisions=decisions, actorDecisionsPerSecond=round(decisions/seconds),
        timings={key:round(value,3) for key,value in timings.items()},
        fractions={key:round(value/seconds,3) for key,value in timings.items()},
        finalKL=[rows[-1][role]['approximateKL'] for role in ['hider','seeker']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs',type=int,default=16)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--sequence-batch',type=int,default=32)
    parser.add_argument('--updates',type=int,default=3)
    parser.add_argument('--horizon',type=int,default=64)
    parser.add_argument('--sequence-length',type=int,default=16)
    parser.add_argument('--resume')
    args = parser.parse_args()
    print(json.dumps(benchmark(args.envs,args.workers,args.sequence_batch,args.updates,
                              args.horizon,args.sequence_length,args.resume)),flush=True)
