import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { EntityPolicy, createEntityPolicies, ENTITY_PAIR_FORMAT } from '../src/core/entityPolicy.js';
import { createPolicyControllers, BINARY_FORMAT, PERSISTENT_FORMAT } from '../src/core/policyController.js';
import { Random } from '../src/core/physics.js';

const directory = new URL('../public/models/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('MANIFEST.json', directory)));
// Compatibility fixtures must stay on the legacy encoder when the live pair changes.
const reference = JSON.parse(readFileSync(new URL('physical-policy-initial-1e9ef2a1b555.json', directory)));
reference.format = PERSISTENT_FORMAT;

function fixture() {
  const actor = structuredClone(reference.actors[0]);
  actor.encoderType = 'object-set-v1';
  delete actor.weights['encoder.weight']; delete actor.weights['encoder.bias'];
  const random = new Random(57119);
  const matrices = [
    ['fixed', 96, 44], ['embedding.0', 32, 16], ['embedding.2', 32, 32],
    ['query', 32, 44], ['key', 32, 32], ['value', 32, 32], ['residual', 96, 32],
  ];
  for (const [name, rows, columns] of matrices) {
    actor.weights[`encoder.${name}.weight`] = Array.from({ length: rows }, () =>
      Array.from({ length: columns }, () => (random.next() - .5) * .2));
    actor.weights[`encoder.${name}.bias`] = Array.from({ length: rows }, () => (random.next() - .5) * .2);
  }
  actor.weights['encoder.object_linear'] = Array.from({ length: 96 }, () =>
    Array.from({ length: 16 }, () => (random.next() - .5) * .2));
  return actor;
}

function* permutations(items) {
  if (!items.length) { yield []; return; }
  for (let i = 0; i < items.length; i += 1)
    for (const tail of permutations(items.filter((_, index) => index !== i))) yield [items[i], ...tail];
}

function observation() {
  const random = new Random(31317);
  const result = Float32Array.from({ length: 140 }, () => (random.next() - .5) * .4);
  result[5] = 1; result[7] = 1; result[10] = 1; result[138] = 1; result[139] = 0;
  for (let slot = 0; slot < 6; slot += 1) result[18 + slot * 16] = 1;
  return result;
}

function near(a, b, tolerance = 1e-6) {
  assert.equal(a.length, b.length);
  assert(a.every((value, index) => Number.isFinite(value) && Math.abs(value - b[index]) <= tolerance));
}

test('all 720 JavaScript object permutations preserve the active nonlinear encoder', () => {
  const policy = new EntityPolicy(fixture());
  for (const visible of [0, 3, 6]) {
    const row = observation();
    for (let slot = visible; slot < 6; slot += 1) row[18 + slot * 16] = 0;
    const expected = policy.encode(row);
    let count = 0;
    for (const order of permutations([0, 1, 2, 3, 4, 5])) {
      const changed = row.slice();
      order.forEach((slot, index) => changed.set(row.slice(18 + slot * 16, 34 + slot * 16), 18 + index * 16));
      near(policy.encode(changed), expected);
      count += 1;
    }
    assert.equal(count, 720);
  }
});

test('unseen payloads and learned empty-set biases cannot create information', () => {
  const policy = new EntityPolicy(fixture());
  const row = observation(); row[10] = 0;
  for (let slot = 0; slot < 6; slot += 1) row[18 + slot * 16] = 0;
  const modified = row.slice(); modified.fill(1e6, 11, 18);
  for (let slot = 0; slot < 6; slot += 1) modified.fill(-1e6, 19 + slot * 16, 34 + slot * 16);
  assert.deepEqual(policy.encode(row), policy.encode(modified));
  assert.deepEqual(policy.forward(row, new Float32Array(64)), policy.forward(modified, new Float32Array(64)));
});

test('mixed typed actors retain stateful controls and strict actor-only validation', () => {
  const legacy = structuredClone(reference.actors[0]); legacy.encoderType = 'legacy-linear-v1';
  const model = { format: ENTITY_PAIR_FORMAT, observationSize: 140, physicsObservationSize: 138,
    commands: ['keep', 'press', 'release'], actors: [legacy, fixture()] };
  const policies = createEntityPolicies(model);
  const random = new Random(791), repeated = new Random(791);
  const replay = rng => {
    let states = policies.map(policy => policy.initialState());
    const actions = [];
    for (let tick = 0; tick < 96; tick += 1) {
      if (tick === 80) states = policies.map(policy => policy.initialState());
      for (let role = 0; role < 2; role += 1) {
        const physical = observation().slice(0, 138);
        physical[7] = 1-role; physical[5] = tick < 10 ? .5 : 1;
        const out = policies[role].act(physical, states[role], { random: () => rng.next() });
        if (role === 1 && tick < 10) {
          assert(out.action.every(value => value === 0)); assert(out.buttons.every(value => value === 0));
          assert(out.memory.some(value => value !== 0));
        }
        states[role] = out.state; actions.push([...out.action]);
      }
    }
    return actions;
  };
  assert.deepEqual(replay(random), replay(repeated));
  for (const change of [
    value => { value.actors[1].weights['critic.weight'] = [[1]]; },
    value => { value.actors[1].weights['encoder.key.bias'][0] = null; },
    value => { value.actors[1].encoderType = 'unknown'; },
    value => { delete value.actors[0].encoderType; },
  ]) { const bad = structuredClone(model); change(bad); assert.throws(() => createEntityPolicies(bad)); }
});

test('existing persistent and legacy binary imports keep exactly their old inference path', () => {
  const binary = structuredClone(reference);
  binary.format = BINARY_FORMAT; binary.observationSize = 138;
  for (const actor of binary.actors) {
    actor.observationSize = 138;
    actor.weights['encoder.weight'] = actor.weights['encoder.weight'].map(row => row.slice(0, 138));
    actor.weights['tools.weight'] = actor.weights['tools.weight'].slice(0, 2);
    actor.weights['tools.bias'] = actor.weights['tools.bias'].slice(0, 2);
  }
  for (const model of [reference, binary]) {
    const old = createPolicyControllers(model), routed = createEntityPolicies(model);
    const a = new Random(981), b = new Random(981);
    const statesA = old.map(policy => policy.initialState()), statesB = routed.map(policy => policy.initialState());
    for (let tick = 0; tick < 24; tick += 1) for (let role = 0; role < 2; role += 1) {
      const row = observation().slice(0, 138);
      const first = old[role].act(row, statesA[role], { random: () => a.next() });
      const second = routed[role].act(row, statesB[role], { random: () => b.next() });
      assert.deepEqual(first, second);
      statesA[role] = first.state; statesB[role] = second.state;
    }
  }
});

test('live entity pair dispatches through the application controller', () => {
  const model = JSON.parse(readFileSync(new URL(manifest.file, directory)));
  const direct = createEntityPolicies(model), routed = createPolicyControllers(model);
  for (let role = 0; role < 2; role++) {
    assert.equal(routed[role].physicsObservationSize, 138);
    const physical = observation().slice(0,138);
    const a = direct[role].act(physical,direct[role].initialState(),{ deterministic: true });
    const b = routed[role].act(physical,routed[role].initialState(),{ deterministic: true });
    assert.deepEqual(a,b);
  }
});
