import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createEntityPolicies } from '../src/core/entityPolicy.js';

const [modelPath, fixturePath] = process.argv.slice(2);
const model = JSON.parse(readFileSync(modelPath));
const fixtures = JSON.parse(readFileSync(fixturePath));
const policies = createEntityPolicies(model);
let rows = 0, blindRows = 0, resets = 0, maximum = 0;
function compare(actual, expected, name) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const error = Math.abs(actual[i] - expected[i]);
    maximum = Math.max(maximum, error);
    assert(error <= 1e-5, `${name}[${i}]: ${error} > 1e-5`);
  }
}
for (const test of fixtures.rollouts) {
  const states = policies.map(policy => policy.initialState());
  for (const row of test.rows) {
    if (row.reset) { states[row.role] = policies[row.role].initialState(); resets += 1; }
    let consumed = 0;
    const prediction = policies[row.role].act(row.observation, states[row.role], {
      deterministic: test.deterministic,
      random: () => { assert(consumed < row.tape.length); return row.tape[consumed++]; },
    });
    assert.equal(consumed, test.deterministic ? 0 : 8);
    assert.deepEqual([...prediction.commands], row.commands);
    assert.deepEqual([...prediction.buttons], row.buttons);
    for (const key of ['action', 'memory', 'mean', 'toolLogits']) compare(prediction[key], row[key], key);
    if (row.observation[7] < .5 && row.observation[5] < 1) {
      assert(prediction.action.every(value => value === 0));
      assert(prediction.buttons.every(value => value === 0));
      blindRows += 1;
    }
    states[row.role] = prediction.state;
    rows += 1;
  }
}
for (const row of fixtures.sensorCases) {
  const policy = policies[row.role];
  compare(policy.encode(row.observation), row.encoded, 'encoded');
  const result = policy.forward(row.observation, row.previousMemory);
  for (const key of ['mean', 'toolLogits', 'memory']) compare(result[key], row[key], key);
}
assert.equal(rows, 1056); assert.equal(blindRows, 208); assert.equal(resets, 4);
const bad = structuredClone(model);
bad.actors[0].weights['critic.weight'] = [[1]];
assert.throws(() => createEntityPolicies(bad));
console.log(JSON.stringify({ rows, blindRows, resets, sensorCases: fixtures.sensorCases.length,
  maxInferenceError: maximum, tolerance: 1e-5, deterministicAndSeededSampling: true }));
