import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import load from '../src/vendor/mujoco-csp.js';
import { PhysicsSimulation, generateArena } from '../src/core/physics.js';
import { createPolicyControllers } from '../src/core/policyController.js';
import {
  TAG_FORMAT, TAG_SCHEMA, CAPTURE_DISTANCE, EXTRAS, NOISE, ROUND_LENGTHS, ROUND_PLAY,
  captured, extras, observationWidth, preparationSteps, roundSeconds,
} from '../src/core/gameRules.js';

const fixture = JSON.parse(fs.readFileSync(new URL('./fixtures/tag-rounds-native.json', import.meta.url)));
const mj = await load({ wasmBinary: fs.readFileSync(new URL('../node_modules/@mujoco/mujoco/mujoco.wasm', import.meta.url)) });
const near = (a, b, tolerance, name) => {
  assert.equal(a.length, b.length, name);
  for (let i = 0; i < a.length; i++)
    assert.ok(Number.isFinite(a[i]) && Math.abs(a[i] - b[i]) <= tolerance, `${name}[${i}]: ${a[i]} versus ${b[i]}`);
};

test('the fixture names the native sources it was generated from', () => {
  assert.equal(fixture.schema, TAG_SCHEMA);
  for (const [file, sha] of Object.entries(fixture.sources))
    assert.equal(createHash('sha256').update(fs.readFileSync(new URL('../training_v5/' + file, import.meta.url))).digest('hex'), sha, `Stale native source: ${file}`);
  assert.equal(createHash('sha256').update(fs.readFileSync(new URL('../scripts/generate-tag-fixtures.py', import.meta.url))).digest('hex'), fixture.generator);
});

test('preparation follows the play length and the observation width follows the schema', () => {
  assert.equal(preparationSteps(144), 96);
  assert.equal(preparationSteps(188), 96);
  assert.equal(preparationSteps(375), 150);
  assert.equal(preparationSteps(750), 300);
  assert.equal(preparationSteps(ROUND_PLAY), 150);
  assert.throws(() => preparationSteps(0));
  assert.equal(observationWidth(undefined), 208);
  assert.equal(observationWidth(TAG_SCHEMA), 208 + EXTRAS);
});

test('the selectable round lengths are the ones the policies played in training', () => {
  assert.deepEqual(ROUND_LENGTHS, [188, 375, 750]);
  assert.ok(ROUND_LENGTHS.includes(ROUND_PLAY));
  assert.deepEqual(ROUND_LENGTHS.map(roundSeconds), [15, 30, 60]);
  // Preparation follows the round: the training rule, not a fixed 7.7 seconds.
  assert.deepEqual(ROUND_LENGTHS.map(preparationSteps), [96, 150, 300]);
});

test('browser extras, last-seen memory and the capture flag match the native rollout', () => {
  const { physics } = fixture;
  const sim = new PhysicsSimulation(mj, physics.arena, { prep: physics.prep, play: physics.play, observationSchema: TAG_SCHEMA, continuous: true });
  try {
    let captures = 0, sightings = 0;
    physics.actions.forEach((actions, i) => {
      sim.step(actions);
      const frame = physics.frames[i];
      assert.equal(sim.t, frame.t);
      near(sim.data.qpos, frame.qpos, 3e-6, 'qpos');
      for (let a = 0; a < 2; a++) {
        const observation = sim.observe(a);
        assert.equal(observation.length, 208 + EXTRAS);
        near(observation, frame.observation[a], 3e-5, `observation ${a}`);
        near(extras(sim, a), frame.extras[a], 3e-5, `extras ${a}`);
        if (frame.lastSeen[a] === null) assert.equal(sim.lastSeen[a], null);
        else {
          assert.equal(sim.lastSeen[a].t, frame.lastSeen[a].t);
          near(sim.lastSeen[a].position, frame.lastSeen[a].position, 3e-6, `lastSeen ${a}`);
          sightings++;
        }
      }
      assert.equal(captured(sim), frame.captured, `captured at ${frame.t}`);
      captures += Number(frame.captured);
    });
    assert.ok(sightings > 0 && captures > 0, 'the rollout covers sightings and a capture');
  } finally {
    sim.dispose();
  }
});

test('a tag needs play, sight and reach; the last-seen offset turns with the agent', () => {
  const arena = generateArena(31, 'open', 8, 0, 0);
  arena.objects = [];
  arena.agents = [{ position: [3, 3, .25], yaw: 0 }, { position: [3.55, 3, .25], yaw: Math.PI }];
  const sim = new PhysicsSimulation(mj, arena, { prep: 2, play: 20, observationSchema: TAG_SCHEMA, continuous: true });
  try {
    sim.step([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]);
    assert.equal(captured(sim), false, 'blind preparation');
    sim.step([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]);
    sim.step([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]);
    assert.ok(sim.seen[1][0]);
    assert.ok(Math.hypot(sim.data.qpos[0] - sim.data.qpos[4], sim.data.qpos[1] - sim.data.qpos[5]) < CAPTURE_DISTANCE);
    assert.equal(captured(sim), true);
    const before = extras(sim, 1);
    assert.equal(before[0], 1);
    assert.ok(before[1] > 0 && Math.abs(before[2]) < 1e-6, 'the hider is straight ahead of the seeker');
    sim.data.qpos[7] += Math.PI / 2;
    mj.mj_forward(sim.model, sim.data);
    const turned = extras(sim, 1);
    near(turned.slice(1, 3), [before[2], -before[1]], 1e-6, 'turned offset');
    sim.data.qpos[4] = 6;
    mj.mj_forward(sim.model, sim.data);
    sim.senses();
    assert.equal(captured(sim), false, 'out of reach');
  } finally {
    sim.dispose();
  }
});

test('the v6 actor pair reproduces native decisions while carrying its exploration noise', () => {
  const policies = createPolicyControllers(fixture.model);
  assert.equal(policies.length, 2);
  for (const policy of policies) {
    assert.equal(policy.observationSchema, TAG_SCHEMA);
    assert.equal(policy.physicsObservationSize, 208 + EXTRAS);
    assert.equal(policy.observationSize, 208 + EXTRAS + 2 + NOISE);
    assert.equal(policy.format, TAG_FORMAT);
    const state = policy.initialState();
    assert.equal(state.noise.length, NOISE);
  }
  let rows = 0, blind = 0, resets = 0, carried = 0;
  for (const rollout of fixture.rollouts) {
    const states = policies.map(p => p.initialState());
    for (const row of rollout.rows) {
      const policy = policies[row.role];
      if (row.reset) { states[row.role] = policy.initialState(); resets++; }
      near(states[row.role].noise, row.noiseIn, 1e-5, 'noiseIn');
      let used = 0;
      const result = policy.act(Float32Array.from(row.observation), states[row.role], {
        deterministic: rollout.deterministic,
        random: () => { assert.ok(used < row.tape.length); return row.tape[used++]; },
      });
      assert.equal(used, rollout.deterministic ? 0 : 10);
      for (const [key, tolerance] of [['action', 1e-5], ['memory', 1e-5], ['mean', 1e-5], ['toolLogits', 1e-5], ['noise', 1e-5]])
        near(result[key], row[key === 'noise' ? 'noiseOut' : key], tolerance, key);
      assert.deepEqual([...result.commands], row.commands);
      assert.deepEqual([...result.buttons], row.buttons);
      if (rollout.deterministic) assert.ok(result.noise.every(v => v === 0), 'deterministic playback carries no noise');
      else if (result.noise.some(v => v !== 0)) carried++;
      if (row.observation[7] < .5 && row.observation[5] < 1) { assert.ok(result.action.every(v => v === 0)); blind++; }
      states[row.role] = result.state;
      rows++;
    }
  }
  assert.deepEqual({ rows, blind, resets }, fixture.counts);
  assert.ok(carried > 200, 'sampled rows carry noise into the next decision');
});

test('the tag-round format is rejected without its schema or with a legacy width', () => {
  const missingSchema = structuredClone(fixture.model); delete missingSchema.observationSchema;
  assert.throws(() => createPolicyControllers(missingSchema), /schema/);
  const narrow = structuredClone(fixture.model); narrow.physicsObservationSize = 208; narrow.observationSize = 214;
  assert.throws(() => createPolicyControllers(narrow));
  const noNoise = structuredClone(fixture.model); noNoise.actors[0].noiseRho = 0;
  assert.throws(() => createPolicyControllers(noNoise));
  const legacyWithSchema = structuredClone(fixture.model); legacyWithSchema.format = 'original-mujoco-relational-jump-policy-pair-v4';
  assert.throws(() => createPolicyControllers(legacyWithSchema));
  assert.throws(() => new PhysicsSimulation(mj, generateArena(1), { observationSchema: 'unknown' }), /Unknown/);
});
