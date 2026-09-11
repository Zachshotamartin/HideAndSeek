"""Original synchronous MuJoCo process pool. No auto-reset, reward edits or policies.

PhysicsEnvPool(configs, workers=4) partitions independent environments across
persistent spawn workers. step(actions) returns terminal observations unchanged.
reset_at(indices, seeds, overrides=None) performs only caller-requested resets.
Every request has an ID; worker failures close the pool instead of returning a
partial batch. Instances must be used from a guarded Python main entry point.
"""
from __future__ import annotations
import multiprocessing as mp
from multiprocessing.connection import wait
import signal
import time
import traceback
import numpy as np

class PhysicsPoolError(RuntimeError):
    pass

def _worker(connection, configs, with_central_state=False, ignore_parent_signals=False):
    if ignore_parent_signals:
        # The continuous-training parent owns Ctrl-C and saves after a complete
        # update. Workers must finish the in-flight batch rather than die first.
        for stop_signal in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(stop_signal, signal.SIG_IGN)
    envs=[]
    try:
        if __package__:
            from .physics import PhysicsEnv
        else:
            from physics import PhysicsEnv
        if with_central_state:
            if __package__:
                from .central_critic import central_state
            else:
                from central_critic import central_state
        envs=[PhysicsEnv(**config) for config in configs]
        initial=np.stack([env.observe() for env in envs])
        if with_central_state:initial=(initial,[central_state(env) for env in envs])
        connection.send((0,'ok',initial))
        while True:
            request,operation,payload=connection.recv()
            if operation=='close':break
            if operation=='step':
                rows=[env.step(action) for env,action in zip(envs,payload)]
                result=(np.stack([row[0] for row in rows]),np.stack([row[1] for row in rows]),np.asarray([row[2] for row in rows],dtype=bool),[row[3] for row in rows])
                if with_central_state:
                    result=(*result,[central_state(env,(env.actions[:,3:5]>.5).astype(np.float32)) for env in envs])
            elif operation=='reset':
                result=[]
                for local,seed,overrides in payload:
                    if overrides:
                        configs[local]={**configs[local],**overrides,'seed':int(seed)}
                        envs[local].close();envs[local]=PhysicsEnv(**configs[local]);obs=envs[local].observe()
                    else:obs=envs[local].reset(seed=int(seed))
                    result.append((local,obs,central_state(envs[local])) if with_central_state else (local,obs))
            elif operation=='snapshot':
                from snapshots import dump
                result=[dump(env) for env in envs]
            elif operation=='restore':
                from snapshots import restore
                for env in envs:env.close()
                envs=[restore(record) for record in payload]
                result=[(env.observe(),central_state(env,(env.actions[:,3:5]>.5).astype(np.float32))) if with_central_state else (env.observe(),None) for env in envs]
            elif operation=='trace':result=[env.trace() for env in envs]
            else:raise ValueError(f'Unknown pool operation {operation}')
            connection.send((request,'ok',result))
    except EOFError:
        pass
    except BaseException as error:
        try:connection.send((locals().get('request',0),'error',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:
        for env in envs:
            try:env.close()
            except Exception:pass
        connection.close()

class PhysicsEnvPool:
    def __init__(self, configs, workers=4, timeout=60.0, with_central_state=False, ignore_parent_signals=False):
        if not configs or not isinstance(configs,(list,tuple)):raise ValueError('Provide a non-empty sequence of environment configuration dictionaries')
        if not isinstance(workers,int) or workers<1 or workers>16:raise ValueError('Use 1–16 workers')
        if not 1<=timeout<=300:raise ValueError('Pool timeout must be 1–300 seconds')
        self.configs=[dict(c) for c in configs];self.count=len(configs);self.worker_count=min(workers,self.count);self.timeout=float(timeout);self.request=0;self.closed=False
        self.connections=[];self.processes=[];self.groups=[];self.locations={}
        self.with_central_state=bool(with_central_state);self.central_states=None
        ctx=mp.get_context('spawn')
        try:
            for worker,indices in enumerate(np.array_split(np.arange(self.count),self.worker_count)):
                group=[int(i) for i in indices];parent,child=ctx.Pipe();process=ctx.Process(target=_worker,args=(child,[self.configs[i] for i in group],self.with_central_state,ignore_parent_signals),name=f'hide-seek-physics-{worker}',daemon=True)
                self.connections.append(parent);self.processes.append(process);self.groups.append(group)
                for local,index in enumerate(group):self.locations[index]=(worker,local)
                process.start();child.close()
            initial=self._receive(list(range(self.worker_count)),0)
            self.observations=np.concatenate([initial[i][0] if self.with_central_state else initial[i] for i in range(self.worker_count)],axis=0)
            if self.with_central_state:self.central_states=tuple(state for i in range(self.worker_count) for state in initial[i][1])
        except BaseException:
            self.close();raise
    def _receive(self, workers, request):
        pending={self.connections[w]:w for w in workers};results={};deadline=time.monotonic()+self.timeout
        while pending:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise PhysicsPoolError(f'Physics request {request} timed out after {self.timeout:g} seconds')
            ready=wait(list(pending),timeout=min(remaining,1.0))
            if not ready:
                dead=[w for w in pending.values() if not self.processes[w].is_alive()]
                if dead:raise PhysicsPoolError(f'Physics workers exited before replying: {dead}')
                continue
            for connection in ready:
                worker=pending.pop(connection)
                try:received,status,result=connection.recv()
                except (EOFError,OSError) as error:raise PhysicsPoolError(f'Physics worker {worker} closed its connection') from error
                if received!=request:raise PhysicsPoolError(f'Stale physics reply {received}, expected {request}')
                if status!='ok':raise PhysicsPoolError(f'Physics worker {worker}: {result["type"]}: {result["message"]}\n{result["traceback"]}')
                results[worker]=result
        return results
    def _call(self, operation, payloads):
        if self.closed:raise PhysicsPoolError('Physics pool is closed')
        self.request+=1
        try:
            for worker,payload in payloads.items():self.connections[worker].send((self.request,operation,payload))
            return self._receive(list(payloads),self.request)
        except (BrokenPipeError,EOFError,OSError) as error:
            self.close();raise PhysicsPoolError(f'Physics transport failed: {error}') from error
        except BaseException:
            self.close();raise
    def step(self, actions):
        actions=np.asarray(actions,dtype=float)
        if actions.shape!=(self.count,2,6) or not np.isfinite(actions).all():raise ValueError(f'Expected finite actions [{self.count},2,6]')
        results=self._call('step',{w:actions[group] for w,group in enumerate(self.groups)})
        ordered=[results[w] for w in range(self.worker_count)]
        self.observations=np.concatenate([row[0] for row in ordered],axis=0)
        if self.with_central_state:self.central_states=tuple(state for row in ordered for state in row[4])
        return self.observations.copy(),np.concatenate([row[1] for row in ordered]),np.concatenate([row[2] for row in ordered]),sum([row[3] for row in ordered],[])
    def reset_at(self, indices, seeds, overrides=None):
        indices=[int(i) for i in indices];seeds=[int(seed) for seed in seeds]
        if len(indices)!=len(seeds) or len(set(indices))!=len(indices) or any(i<0 or i>=self.count for i in indices):raise ValueError('Reset indices must be unique, in range, and match the supplied seeds')
        if overrides is None:overrides=[None]*len(indices)
        if len(overrides)!=len(indices):raise ValueError('Provide one configuration override per reset index')
        if not indices:return np.empty((0,*self.observations.shape[1:]),dtype=self.observations.dtype)
        payloads={}
        for index,seed,override in zip(indices,seeds,overrides):
            worker,local=self.locations[index];payloads.setdefault(worker,[]).append((local,seed,override))
            if override:self.configs[index].update(override)
            self.configs[index]['seed']=seed
        results=self._call('reset',payloads)
        central=list(self.central_states) if self.with_central_state else None
        for worker,rows in results.items():
            for row in rows:
                local,observation=row[:2];index=self.groups[worker][local]
                self.observations[index]=observation
                if self.with_central_state:central[index]=row[2]
        if self.with_central_state:self.central_states=tuple(central)
        return self.observations[indices].copy()
    def reset(self,seeds=None):
        if seeds is None:seeds=[config.get('seed',1) for config in self.configs]
        return self.reset_at(range(self.count),seeds)
    def snapshot(self):
        results=self._call('snapshot',{w:None for w in range(self.worker_count)})
        return {'configs':self.configs,'worlds':sum([results[w] for w in range(self.worker_count)],[])}
    def restore(self,saved):
        if len(saved['worlds'])!=self.count:raise ValueError('Snapshot environment count differs')
        self.configs=saved['configs']
        results=self._call('restore',{w:[saved['worlds'][i] for i in group] for w,group in enumerate(self.groups)})
        ordered=sum([results[w] for w in range(self.worker_count)],[])
        self.observations=np.stack([x[0] for x in ordered])
        if self.with_central_state:self.central_states=tuple(x[1] for x in ordered)
        return self.observations.copy()
    def traces(self):
        results=self._call('trace',{w:None for w in range(self.worker_count)})
        return sum([results[w] for w in range(self.worker_count)],[])
    def close(self):
        if self.closed:return
        self.closed=True
        for connection in self.connections:
            try:connection.send((-1,'close',None))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            if process.pid is None:continue
            process.join(timeout=1)
            if process.is_alive():process.terminate();process.join(timeout=1)
            if process.is_alive():process.kill();process.join(timeout=1)
            process.close()
        for connection in self.connections:connection.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
