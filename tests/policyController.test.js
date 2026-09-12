import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createPolicyControllers, POLICY_FORMATS, PERSISTENT_FORMAT, BINARY_FORMAT } from '../src/core/policyController.js';
import { advanceButtons } from '../src/core/persistentPolicy.js';
import { Random } from '../src/core/physics.js';

const asset = file => new URL(`../public/models/${file}`, import.meta.url);
const manifest = JSON.parse(readFileSync(asset('MANIFEST.json')));
const candidate = JSON.parse(readFileSync(asset(manifest.file)));
const initial = JSON.parse(readFileSync(asset(manifest.checkpoints[1].file)));
const zero = n => Array(n).fill(0);
const matrix = (rows, columns) => Array.from({ length: rows }, () => zero(columns));
function fixture(format = PERSISTENT_FORMAT) {
  const persistent = format === PERSISTENT_FORMAT, inputs = persistent ? 140 : 138, tools = persistent ? 6 : 2;
  const actor = { observationSize: inputs, physicsObservationSize: 138, hiddenSize: 64, encoderSize: 96,
    weights: { 'encoder.weight': matrix(96, inputs), 'encoder.bias': zero(96),
      'memory.weight_ih': matrix(192, 96), 'memory.weight_hh': matrix(192, 64),
      'memory.bias_ih': zero(192), 'memory.bias_hh': zero(192),
      'movement.weight': matrix(3, 64), 'movement.bias': zero(3),
      'tools.weight': matrix(tools, 64), 'tools.bias': zero(tools), log_std: zero(3) } };
  return { format, observationSize: inputs, physicsObservationSize: 138, commands: ['keep', 'press', 'release'],
    actors: [actor, structuredClone(actor)] };
}

test('shipped v4 actors retain strict integrity and explicit development provenance', () => {
  assert.equal(manifest.format, 'original-mujoco-relational-jump-policy-pair-v4');
  assert.equal(manifest.status, 'DEVELOPMENT');
  assert.equal(manifest.qualified, false);
  assert.equal(manifest.physicsObservationSize, 208);
  assert.equal(manifest.observationSize, 210);
  assert.equal(manifest.actionSize, 6);
  assert.deepEqual(manifest.training, candidate.training);
  for (const entry of manifest.checkpoints) {
    const bytes = readFileSync(asset(entry.file)), model = JSON.parse(bytes);
    assert.equal(bytes.length, entry.bytes);
    assert.equal(createHash('sha256').update(bytes).digest('hex'), entry.sha256);
    assert(entry.file.includes(entry.sha256.slice(0,12)));
    assert.equal(model.physicsObservationSize, 208);
    assert.equal(model.observationSize, 210);
    assert.equal(model.actionSize, 6);
    assert.equal(model.qualified ?? model.localPreview?.qualified, false);
    assert.equal((model.provenance ?? model.localPreview).checkpointSHA256, entry.checkpointSHA256);
    assert.deepEqual(model.training, entry.training);
    assert.equal(createPolicyControllers(model).length, 2);
    assert(model.actors.every(actor => !Object.keys(actor.weights).some(k => /^(value|critic)\./.test(k))));
  }
  assert.equal(initial.training.decisions, 0);
});

test('bundled development evaluation evidence is hash-pinned, names the shipped export as its candidate and reproduces its contrasts', () => {
  const evidence = manifest.evaluation;
  assert(evidence, 'manifest must carry the evaluation evidence entry');
  const bytes = readFileSync(asset(evidence.file)), report = JSON.parse(bytes);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), evidence.sha256);
  assert.equal(evidence.shippedRole, 'candidate');
  assert.equal(report.checkpointSHA256, manifest.checkpoints[0].checkpointSHA256);
  assert.equal(report.referenceSHA256, evidence.referenceSHA256);
  assert.equal(report.checkpointSHA256, evidence.candidateSHA256);
  assert.equal(report.maps.length, evidence.maps);
  assert.equal(report.episodes.length, evidence.episodes);
  assert.equal(report.episodes.length % report.maps.length, 0);
  assert.match(evidence.note, /not qualification/i);
  const intervals=['Hider change','Seeker change'].map(k=>report.contrasts[k].bootstrap95Percent);
  assert(intervals.some(([lo])=>lo>0));assert(intervals.every(([,hi])=>hi>=0));
  const byMap = new Map();
  for (const row of report.episodes) {
    const key = `${row.seed}:${row.scenario}`;
    if (!byMap.has(key)) byMap.set(key, {});
    byMap.get(key)[row.mode] = row.hiddenFraction;
  }
  assert.equal(byMap.size, report.maps.length);
  const definitions = { 'Hider change': ['candidate-hider', 'reference'], 'Seeker change': ['reference', 'candidate-seeker'],
    'Hider grab/lock benefit': ['candidate-pair', 'no-hider-tools'], 'Seeker grab/lock benefit': ['no-seeker-tools', 'candidate-pair'] };
  for (const [label, [left, right]] of Object.entries(definitions)) {
    const result = report.contrasts[label];
    assert(result, label);
    let sum = 0, positive = 0, negative = 0;
    for (const rows of byMap.values()) { const d = rows[left] - rows[right]; sum += d; if (d > 0) positive++; if (d < 0) negative++; }
    assert(Math.abs(sum / byMap.size - result.mean) < 1e-9, label);
    assert.equal(positive, result.positiveCases); assert.equal(negative, result.negativeCases);
    assert(result.bootstrap95Percent[0] <= result.mean && result.mean <= result.bootstrap95Percent[1]);
  }
});

test('persistent controls feed back own requested buttons and reset each actor independently', () => {
  const model = fixture(), w = model.actors[0].weights;
  // A sensor probe through an otherwise zero actor makes the two extra columns observable.
  w['encoder.weight'][0][138] = 1; w['encoder.weight'][1][139] = 1;
  w['memory.weight_ih'][128][0] = 1; w['memory.weight_ih'][129][1] = 1;
  w['movement.weight'][0][0] = 1; w['movement.weight'][1][1] = 1;
  const policies = createPolicyControllers(model), states = policies.map(p => p.initialState());
  states[0].buttons.set([1, 1]);
  const observation = new Float32Array(138); observation[7] = 1;
  const out = policies[0].act(observation, states[0], { deterministic: true });
  assert.deepEqual([...out.commands], [0, 0]);
  assert.deepEqual([...out.action.slice(3)], [1, 1], 'Keep must retain requested state even without a physical grasp');
  assert(out.action[0] > 0 && out.action[1] > 0, 'Both requested states enter the encoder');
  assert(out.state.memory.some(x => x !== 0));
  assert(states[1].memory.every(x => x === 0) && states[1].buttons.every(x => x === 0));
  const reset = policies.map(p => p.initialState());
  assert(reset.every(state => state.memory.every(x => x === 0) && state.buttons.every(x => x === 0)));
  assert.deepEqual([...policies[0].act(observation, reset[0], { deterministic: true }).action], [0, 0, 0, 0, 0]);
  assert.deepEqual([...advanceButtons([1, 0], [0, 1])], [1, 1]);
  assert.deepEqual([...advanceButtons([1, 1], [2, 0])], [0, 1]);
});

test('blind seeker clears requested buttons and actions while recurrent memory advances', () => {
  const [,policy] = createPolicyControllers(candidate);
  const state = policy.initialState(); state.buttons.set([1, 1]);
  const physical = new Float32Array(policy.physicsObservationSize); physical[5] = .5; physical[7] = 0;
  const result = policy.act(physical, state, { deterministic: true });
  assert(result.action.every(x => x === 0));
  assert(result.state.buttons.every(x => x === 0));
  assert(result.state.memory.some(x => x !== 0));
});

test('both persistent actors reproduce seeded sampled and deterministic stateful decisions after reset', () => {
  for (const model of [candidate, initial]) for (const deterministic of [false, true]) {
    const policies = createPolicyControllers(model);
    const run = () => {
      const rng = new Random(89123), states = policies.map(p => p.initialState()), actions = [];
      for (let t = 0; t < 96; t++) {
        actions.push(policies.map((policy, role) => {
          const physical = new Float32Array(policy.physicsObservationSize); physical[7] = Number(role === 0); physical[5] = t / 80;
          const out = policy.act(physical, states[role], { deterministic, random: () => rng.next() });
          states[role] = out.state;
          return [...out.action];
        }));
      }
      return actions;
    };
    assert.deepEqual(run(), run());
  }
});

test('legacy binary imports remain binary; unsupported/mixed schemas and critic parameters are rejected', () => {
  const observation = new Float32Array(138); observation[7] = 1;
  const [binary] = createPolicyControllers(fixture(BINARY_FORMAT));
  const [persistent] = createPolicyControllers(fixture());
  assert.equal(binary.observationSize, 138);
  assert.equal(persistent.observationSize, 140);
  assert.deepEqual([...binary.act(observation, binary.initialState(), { deterministic: true }).action.slice(3)], [1, 1]);
  assert.deepEqual([...persistent.act(observation, persistent.initialState(), { deterministic: true }).action.slice(3)], [0, 0]);
  const invalid = fixture(); invalid.observationSize = 138;
  assert.throws(() => createPolicyControllers(invalid), /138 physical.*two own/);
  invalid.observationSize = 140; invalid.commands = ['press', 'keep', 'release'];
  assert.throws(() => createPolicyControllers(invalid));
  const unknown = fixture(); unknown.actors[0].weights['critic.weight'] = [[0]];
  assert.throws(() => createPolicyControllers(unknown), /actor-only/);
  assert.throws(() => createPolicyControllers({ ...fixture(), format: 'unknown' }), /Unsupported policy/);
  const missing = fixture(); missing.actors[0].weights['tools.bias'][0] = null;
  assert.throws(() => createPolicyControllers(missing));
});

test('relational imports reject unknown parameters, malformed matrices and mixed action contracts', () => {
  for (const mutate of [
    m => { m.actors[0].weights['critic.weight'] = [[1]]; },
    m => { delete m.actors[0].weights['tools.bias']; },
    m => { m.actors[0].weights['memory.weight_hh'][0].pop(); },
    m => { m.actors[0].weights['encoder.relation_scale'] = null; },
    m => { m.actors[0].weights['movement.bias'][0] = Infinity; },
    m => { m.actors[1].encoderType = 'object-relations-v2'; },
    m => { m.physicsObservationSize = 138; },
    m => { m.commands = ['press','keep','release']; },
    m => { m.actionSize = 5; },
  ]) { const bad = structuredClone(candidate); mutate(bad); assert.throws(() => createPolicyControllers(bad)); }
  const [policy] = createPolicyControllers(candidate);
  assert.throws(() => policy.act(new Float32Array(138), policy.initialState()), /Invalid persistent state/);
});
