import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import load from '../src/vendor/mujoco-csp.js';
import { PhysicsSimulation, generateArena, KNOWN_POSITION_SCHEMA } from '../src/core/physics.js';
import { RelationalPolicy } from '../src/core/relationalPolicy.js';
const mj = await load({wasmBinary:fs.readFileSync(new URL('../node_modules/@mujoco/mujoco/mujoco.wasm',import.meta.url))});

test('new schema exposes hidden position; original checkpoints keep old observations', () => {
  const arena=generateArena(711,'open',8,0,0);
  const old=new PhysicsSimulation(mj,arena);
  const updated=new PhysicsSimulation(mj,arena,{observationSchema:KNOWN_POSITION_SCHEMA});
  try {
    for(const s of [old,updated]) {
      s.data.qpos.set([3,4,.25,Math.PI,5,4,.25,0],0);
      mj.mj_forward(s.model,s.data);
      s.seen=[[false,false],[false,false]];
    }
    for(let role=0;role<2;role++){
      const original=old.observe(role),changed=updated.observe(role);
      assert.equal(changed[10],0);
      assert.ok(Math.abs(changed[11])>.1);
      assert.deepEqual(Array.from(changed.slice(14,18)),[0,0,0,0]);
      assert.deepEqual(Array.from(original.slice(10,18)),Array(8).fill(0));
      assert.deepEqual(Array.from(changed.slice(18)),Array.from(original.slice(18)));
    }
  } finally {old.dispose();updated.dispose();}
});
test('unknown observation semantics are rejected',()=>{
  assert.throws(()=>new PhysicsSimulation(mj,generateArena(1),{observationSchema:'unknown'}),/Unknown/);
  assert.throws(()=>new RelationalPolicy({},'unknown'),/Unknown/);
});
