import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { advanceButtons, createPersistentPolicies } from '../src/core/persistentPolicy.js';

const [modelPath, fixturePath] = process.argv.slice(2);
const model = JSON.parse(readFileSync(modelPath, 'utf8'));
const cases = JSON.parse(readFileSync(fixturePath, 'utf8'));
const policies = createPersistentPolicies(model);
let error = 0;
let rows = 0;
let resets = 0;
let stateChanges = 0;
let blindRows = 0;
const commandCounts = [0, 0, 0];
for (const test of cases) {
  const states = policies.map(policy => policy.initialState());
  for (const row of test.rows) {
    const role = row.role;
    if (row.reset) { states[role] = policies[role].initialState(); resets += 1; }
    let consumed = 0;
    const prediction = policies[role].act(row.observation, states[role], {
      deterministic: test.deterministic,
      random: () => { assert(consumed < row.tape.length, 'Random tape exhausted'); return row.tape[consumed++]; },
    });
    assert.equal(consumed, test.deterministic ? 0 : 8);
    for (const key of ['commands', 'buttons']) assert.deepEqual([...prediction[key]], row[key]);
    for (const key of ['action', 'memory', 'mean', 'toolLogits']) {
      for (let index = 0; index < row[key].length; index += 1) {
        const difference = Math.abs(prediction[key][index] - row[key][index]);
        error = Math.max(error, difference);
        assert(difference <= 1e-5, `${key} differs ${difference} at row ${rows}, index ${index}`);
      }
    }
    if (row.observation[7] < .5 && row.observation[5] < 1) {
      assert(prediction.action.every(value => value === 0));
      assert(prediction.buttons.every(value => value === 0));
      blindRows += 1;
    }
    if (prediction.buttons.some((value, index) => value !== states[role].buttons[index])) stateChanges += 1;
    prediction.commands.forEach(command => { commandCounts[command] += 1; });
    states[role] = prediction.state;
    rows += 1;
  }
}
assert.deepEqual([...advanceButtons([0, 0], [1, 1])], [1, 1]);
assert.deepEqual([...advanceButtons([1, 1], [0, 0])], [1, 1]);
assert.deepEqual([...advanceButtons([1, 1], [2, 2])], [0, 0]);
assert.deepEqual([...advanceButtons([1, 1], [0, 0], true)], [0, 0]);
assert.throws(() => advanceButtons([1, 0], [3, 0]));
assert.throws(() => policies[0].act(new Float32Array(137), policies[0].initialState()));
const invalid = structuredClone(model); invalid.actors[0].weights['tools.bias'][0] = null;
assert.throws(() => createPersistentPolicies(invalid));
assert(rows === 1056 && resets === 4 && blindRows === 208);
// Inference correctness must also pass a policy that never presses a button.
// Command semantics are checked explicitly above; event counts are diagnostics.
console.log(JSON.stringify({ rows, resets, stateChanges, blindRows, commandCounts,
  maxInferenceError: error, tolerance: 1e-5, deterministicAndSeededSampling: true,
  observationSource: 'Native physical episodes, preparation boundaries and terminal resets' }));
