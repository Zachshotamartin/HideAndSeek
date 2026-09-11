import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import load from '../src/vendor/mujoco-csp.js';
import { PhysicsSimulation, generateArena } from '../src/core/physics.js';

const mj = await load({
  wasmBinary: fs.readFileSync(new URL('../node_modules/@mujoco/mujoco/mujoco.wasm', import.meta.url)),
});

test('48 complete rounds and repeated disposal reuse bounded WASM memory', () => {
  const heapSizes = [];
  let peakArena = 0;
  for (let round = 0; round < 48; round++) {
    const simulation = new PhysicsSimulation(mj, generateArena(7813 + round,
      ['shelter', 'rooms', 'open'][round % 3], 6 + round % 7, 8, 2),
      { immovable: round % 2 === 0 });
    try {
      for (let tick = 0; tick < 240; tick++) {
        simulation.step([[0.6, 0.1, 0.1, 1, 1], [0.4, 0, -0.2, 0, 0]]);
      }
      assert.equal(simulation.done, true);
      heapSizes.push(simulation.data.qpos.buffer.byteLength);
      peakArena = Math.max(peakArena, simulation.data.maxuse_arena);
    } finally {
      simulation.dispose();
      simulation.dispose();
    }
  }
  const warmed = Math.max(...heapSizes.slice(0, 3));
  assert.ok(Math.max(...heapSizes.slice(3)) <= warmed + 32 * 1024 * 1024,
    `Native arena memory grew across disposal: ${heapSizes.join(', ')}`);
  assert.ok(Math.max(...heapSizes) <= 64 * 1024 * 1024,
    `Browser physics exceeded its memory budget: ${Math.max(...heapSizes)} bytes`);
  assert.ok(peakArena < 8 * 1024 * 1024,
    `Less than half the solver arena remains free: ${peakArena} bytes used`);
});

test('a failed native data allocation releases the already-created model', () => {
  let model;
  const failingRuntime = {
    MjModel: {
      from_xml_string(xml) {
        model = mj.MjModel.from_xml_string(xml);
        return model;
      },
    },
    MjData: class {
      constructor() {
        throw new Error('Simulated data allocation failure');
      }
    },
  };
  assert.throws(() => new PhysicsSimulation(failingRuntime, generateArena(2)),
    /Simulated data allocation failure/);
  assert.equal(model.isDeleted(), true);
});
