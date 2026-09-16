"""Synchronous MuJoCo process pool. No auto-reset, reward edits or policies.

``PhysicsEnvPool(configs, workers=4)`` partitions independent environments
across persistent spawn workers. ``step(actions)`` returns terminal
observations unchanged. ``reset_at(indices, seeds, overrides=None)`` performs
only caller-requested resets. ``observation='game'`` returns the tag-round
actor observation (physics plus the game.py extras) instead of bare physics. Every request has an ID; worker failures close
the pool instead of returning a partial batch. Instances must be used from a
guarded Python main entry point.
"""
from __future__ import annotations

import multiprocessing as mp
import signal
import time
import traceback
from multiprocessing.connection import wait

import numpy as np

try:
    from .capture import resolve_capture
except ImportError:
    from capture import resolve_capture

UNSTABLE_WARNINGS = ('mjWARN_BADQPOS', 'mjWARN_BADQVEL', 'mjWARN_BADQACC')
OBSERVATION_MODES = ('physics', 'game')
MAX_WORKERS = 16
TIMEOUT_RANGE = (1, 300)
POLL_SECONDS = 1.0
CLOSE_REQUEST = -1


class PhysicsPoolError(RuntimeError):
    pass


def unstable(env):
    """True once MuJoCo has silently reset this world to its XML pose.

    MuJoCo only leaves a warning counter behind; the physical contract stays
    byte-identical, so the check lives here, next to the recovery, instead of
    inside physics.py.
    """
    import mujoco
    return any(env.data.warning[getattr(mujoco.mjtWarning, name)].number for name in UNSTABLE_WARNINGS)


def pressed_tools(env):
    return (env.actions[:, 3:5] > .5).astype(np.float32)


def _import_physics(with_central_state):
    if __package__:
        from .physics import PhysicsEnv
    else:
        from physics import PhysicsEnv
    central_state = None
    if with_central_state:
        if __package__:
            from .central_critic import central_state
        else:
            from central_critic import central_state
    return PhysicsEnv, central_state


def _import_observer(observation):
    """The worker-side observation function: bare physics or the tag-round actor view."""
    if observation == 'physics':
        return lambda env: env.observe()
    if __package__:
        from .game import observe
    else:
        from game import observe
    return observe


def _step_one(env, action, observe):
    """One environment step; a blown-up world ends its episode instead of crashing.

    The episode ends with zero reward for both roles (still zero-sum) and the
    world is rebuilt on the same layout so the trainer's ordinary reset path
    replaces it; the event is reported in the info dictionary.
    """
    try:
        row = env.step(action)
        if unstable(env):
            raise RuntimeError('Physics diverged: MuJoCo instability reset')
        from motion_diagnostics import blocked_roles
        _, reward, done, info = resolve_capture(env, *row)
        info = dict(info, blockedMotion=blocked_roles(env))
        return observe(env), reward, done, info
    except RuntimeError as error:
        if 'diverged' not in str(error):
            raise
        info = dict(env.info(), diverged=True)
        env.reset(seed=env.arena['seed'])
        return np.zeros_like(observe(env)), np.zeros(2), True, info


def _step_all(envs, actions, central_state, observe):
    rows = [_step_one(env, action, observe) for env, action in zip(envs, actions)]
    result = (np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows]),
              np.asarray([row[2] for row in rows], dtype=bool), [row[3] for row in rows])
    if central_state:
        result = (*result, [central_state(env, pressed_tools(env)) for env in envs])
    return result


def _reset_some(envs, configs, payload, PhysicsEnv, central_state, observe):
    result = []
    for local, seed, overrides in payload:
        if overrides:
            configs[local] = {**configs[local], **overrides, 'seed': int(seed)}
            envs[local].close()
            envs[local] = PhysicsEnv(**configs[local])
        else:
            envs[local].reset(seed=int(seed))
        observation = observe(envs[local])
        result.append((local, observation, central_state(envs[local])) if central_state else (local, observation))
    return result


def _restore_all(envs, payload, central_state, observe):
    from snapshots import restore
    for env in envs:
        env.close()
    envs = [restore(record) for record in payload]
    if central_state:
        result = [(observe(env), central_state(env, pressed_tools(env))) for env in envs]
    else:
        result = [(observe(env), None) for env in envs]
    return envs, result


def _worker(connection, configs, with_central_state=False, ignore_parent_signals=False, observation='physics'):
    if ignore_parent_signals:
        # The continuous-training parent owns Ctrl-C and saves after a complete
        # update. Workers must finish the in-flight batch rather than die first.
        for stop_signal in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(stop_signal, signal.SIG_IGN)
    envs = []
    request = 0
    try:
        PhysicsEnv, central_state = _import_physics(with_central_state)
        observe = _import_observer(observation)
        envs = [PhysicsEnv(**config) for config in configs]
        initial = np.stack([observe(env) for env in envs])
        if with_central_state:
            initial = (initial, [central_state(env) for env in envs])
        connection.send((0, 'ok', initial))
        while True:
            request, operation, payload = connection.recv()
            if operation == 'close':
                break
            if operation == 'step':
                result = _step_all(envs, payload, central_state, observe)
            elif operation == 'reset':
                result = _reset_some(envs, configs, payload, PhysicsEnv, central_state, observe)
            elif operation == 'snapshot':
                from snapshots import dump
                result = [dump(env) for env in envs]
            elif operation == 'restore':
                envs, result = _restore_all(envs, payload, central_state, observe)
            elif operation == 'trace':
                result = [env.trace() for env in envs]
            else:
                raise ValueError(f'Unknown pool operation {operation}')
            connection.send((request, 'ok', result))
    except EOFError:
        pass
    except BaseException as error:
        report = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        try:
            connection.send((request, 'error', report))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        connection.close()


class PhysicsEnvPool:
    def __init__(self, configs, workers=4, timeout=60.0, with_central_state=False, ignore_parent_signals=False,
                 observation='physics'):
        if not configs or not isinstance(configs, (list, tuple)):
            raise ValueError('Provide a non-empty sequence of environment configuration dictionaries')
        if observation not in OBSERVATION_MODES:
            raise ValueError(f'Observation mode must be one of {OBSERVATION_MODES}')
        if not isinstance(workers, int) or workers < 1 or workers > MAX_WORKERS:
            raise ValueError('Use 1–16 workers')
        if not TIMEOUT_RANGE[0] <= timeout <= TIMEOUT_RANGE[1]:
            raise ValueError('Pool timeout must be 1–300 seconds')
        self.configs = [dict(c) for c in configs]
        self.count = len(configs)
        self.worker_count = min(workers, self.count)
        self.timeout = float(timeout)
        self.request = 0
        self.closed = False
        self.connections = []
        self.processes = []
        self.groups = []
        self.locations = {}
        self.with_central_state = bool(with_central_state)
        self.observation = observation
        self.central_states = None
        try:
            self._start_workers(ignore_parent_signals)
            initial = self._receive(list(range(self.worker_count)), 0)
            self.observations = np.concatenate(
                [initial[i][0] if self.with_central_state else initial[i] for i in range(self.worker_count)], axis=0)
            if self.with_central_state:
                self.central_states = tuple(state for i in range(self.worker_count) for state in initial[i][1])
        except BaseException:
            self.close()
            raise

    def _start_workers(self, ignore_parent_signals):
        ctx = mp.get_context('spawn')
        for worker, indices in enumerate(np.array_split(np.arange(self.count), self.worker_count)):
            group = [int(i) for i in indices]
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_worker,
                                  args=(child, [self.configs[i] for i in group], self.with_central_state, ignore_parent_signals,
                                        self.observation),
                                  name=f'hide-seek-physics-{worker}', daemon=True)
            self.connections.append(parent)
            self.processes.append(process)
            self.groups.append(group)
            for local, index in enumerate(group):
                self.locations[index] = (worker, local)
            process.start()
            child.close()

    # ------------------------------------------------------------- transport
    def _receive(self, workers, request):
        pending = {self.connections[w]: w for w in workers}
        results = {}
        deadline = time.monotonic() + self.timeout
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PhysicsPoolError(f'Physics request {request} timed out after {self.timeout:g} seconds')
            ready = wait(list(pending), timeout=min(remaining, POLL_SECONDS))
            if not ready:
                dead = [w for w in pending.values() if not self.processes[w].is_alive()]
                if dead:
                    raise PhysicsPoolError(f'Physics workers exited before replying: {dead}')
                continue
            for connection in ready:
                worker = pending.pop(connection)
                results[worker] = self._read_reply(connection, worker, request)
        return results

    def _read_reply(self, connection, worker, request):
        try:
            received, status, result = connection.recv()
        except (EOFError, OSError) as error:
            raise PhysicsPoolError(f'Physics worker {worker} closed its connection') from error
        if received != request:
            raise PhysicsPoolError(f'Stale physics reply {received}, expected {request}')
        if status != 'ok':
            raise PhysicsPoolError(f'Physics worker {worker}: {result["type"]}: {result["message"]}\n{result["traceback"]}')
        return result

    def _call(self, operation, payloads):
        if self.closed:
            raise PhysicsPoolError('Physics pool is closed')
        self.request += 1
        try:
            for worker, payload in payloads.items():
                self.connections[worker].send((self.request, operation, payload))
            return self._receive(list(payloads), self.request)
        except (BrokenPipeError, EOFError, OSError) as error:
            self.close()
            raise PhysicsPoolError(f'Physics transport failed: {error}') from error
        except BaseException:
            self.close()
            raise

    def _broadcast(self, operation):
        results = self._call(operation, {w: None for w in range(self.worker_count)})
        return sum([results[w] for w in range(self.worker_count)], [])

    # ------------------------------------------------------------ operations
    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.count, 2, 6) or not np.isfinite(actions).all():
            raise ValueError(f'Expected finite actions [{self.count},2,6]')
        results = self._call('step', {w: actions[group] for w, group in enumerate(self.groups)})
        ordered = [results[w] for w in range(self.worker_count)]
        self.observations = np.concatenate([row[0] for row in ordered], axis=0)
        if self.with_central_state:
            self.central_states = tuple(state for row in ordered for state in row[4])
        rewards = np.concatenate([row[1] for row in ordered])
        dones = np.concatenate([row[2] for row in ordered])
        infos = sum([row[3] for row in ordered], [])
        return self.observations.copy(), rewards, dones, infos

    def reset_at(self, indices, seeds, overrides=None):
        indices = [int(i) for i in indices]
        seeds = [int(seed) for seed in seeds]
        if len(indices) != len(seeds) or len(set(indices)) != len(indices) or any(i < 0 or i >= self.count for i in indices):
            raise ValueError('Reset indices must be unique, in range, and match the supplied seeds')
        if overrides is None:
            overrides = [None] * len(indices)
        if len(overrides) != len(indices):
            raise ValueError('Provide one configuration override per reset index')
        if not indices:
            return np.empty((0, *self.observations.shape[1:]), dtype=self.observations.dtype)
        payloads = {}
        for index, seed, override in zip(indices, seeds, overrides):
            worker, local = self.locations[index]
            payloads.setdefault(worker, []).append((local, seed, override))
            if override:
                self.configs[index].update(override)
            self.configs[index]['seed'] = seed
        results = self._call('reset', payloads)
        central = list(self.central_states) if self.with_central_state else None
        for worker, rows in results.items():
            for row in rows:
                local, observation = row[:2]
                index = self.groups[worker][local]
                self.observations[index] = observation
                if self.with_central_state:
                    central[index] = row[2]
        if self.with_central_state:
            self.central_states = tuple(central)
        return self.observations[indices].copy()

    def reset(self, seeds=None):
        if seeds is None:
            seeds = [config.get('seed', 1) for config in self.configs]
        return self.reset_at(range(self.count), seeds)

    def snapshot(self):
        return {'configs': self.configs, 'worlds': self._broadcast('snapshot')}

    def restore(self, saved):
        if len(saved['worlds']) != self.count:
            raise ValueError('Snapshot environment count differs')
        self.configs = saved['configs']
        results = self._call('restore', {w: [saved['worlds'][i] for i in group] for w, group in enumerate(self.groups)})
        ordered = sum([results[w] for w in range(self.worker_count)], [])
        self.observations = np.stack([x[0] for x in ordered])
        if self.with_central_state:
            self.central_states = tuple(x[1] for x in ordered)
        return self.observations.copy()

    def traces(self):
        return self._broadcast('trace')

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send((CLOSE_REQUEST, 'close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            if process.pid is None:
                continue
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            process.close()
        for connection in self.connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
