// Original masked object-set inference with explicit actor-only validation.
import { PersistentPolicy } from './persistentPolicy.js';
import { createPolicyControllers, POLICY_FORMATS } from './policyController.js';

export const ENTITY_PAIR_FORMAT = 'original-mujoco-entity-policy-pair-v1';
const ENTITY = 'object-set-v1';
const LEGACY = 'legacy-linear-v1';
const finite = values => values.every(Number.isFinite);
const sigmoid = value => 1 / (1 + Math.exp(-value));
const baseMatrices = [
  ['memory.weight_ih', 192, 96], ['memory.weight_hh', 192, 64],
  ['movement.weight', 3, 64], ['tools.weight', 6, 64],
];
const baseVectors = [['memory.bias_ih', 192], ['memory.bias_hh', 192],
  ['movement.bias', 3], ['tools.bias', 6], ['log_std', 3]];
const entityMatrices = [
  ['encoder.fixed.weight', 96, 44], ['encoder.object_linear', 96, 16],
  ['encoder.embedding.0.weight', 32, 16], ['encoder.embedding.2.weight', 32, 32],
  ['encoder.query.weight', 32, 44], ['encoder.key.weight', 32, 32],
  ['encoder.value.weight', 32, 32], ['encoder.residual.weight', 96, 32],
];
const entityVectors = [['encoder.fixed.bias', 96], ['encoder.embedding.0.bias', 32],
  ['encoder.embedding.2.bias', 32], ['encoder.query.bias', 32], ['encoder.key.bias', 32],
  ['encoder.value.bias', 32], ['encoder.residual.bias', 96]];

function validate(definition, type) {
  if (definition?.encoderType !== type || definition.observationSize !== 140 ||
      definition.physicsObservationSize !== 138 || definition.hiddenSize !== 64 ||
      definition.encoderSize !== 96 || !definition.weights)
    throw Error('Invalid explicitly typed actor definition');
  const matrices = [...baseMatrices, ...(type === ENTITY ? entityMatrices : [['encoder.weight', 96, 140]])];
  const vectors = [...baseVectors, ...(type === ENTITY ? entityVectors : [['encoder.bias', 96]])];
  const expected = new Set([...matrices, ...vectors].map(([name]) => name));
  if (Object.keys(definition.weights).length !== expected.size ||
      Object.keys(definition.weights).some(name => !expected.has(name)))
    throw Error('Only the exact declared actor weights are permitted; no critic or unknown keys');
  for (const [name, rows, columns] of matrices) {
    const value = definition.weights[name];
    if (!Array.isArray(value) || value.length !== rows || value.some(row =>
      !Array.isArray(row) || row.length !== columns || !finite(row)))
      throw Error(`Invalid entity actor matrix: ${name}`);
  }
  for (const [name, length] of vectors) {
    const value = definition.weights[name];
    if (!Array.isArray(value) || value.length !== length || !finite(value))
      throw Error(`Invalid entity actor vector: ${name}`);
  }
}

function linear(input, weight, bias) {
  return Float32Array.from(weight, (row, index) => {
    let sum = bias ? bias[index] : 0;
    for (let column = 0; column < row.length; column += 1) sum += row[column] * input[column];
    return sum;
  });
}
const tanh = input => Float32Array.from(input, Math.tanh);

export class EntityPolicy {
  constructor(definition) {
    validate(definition, ENTITY);
    this.weights = definition.weights;
  }

  initialState() { return PersistentPolicy.prototype.initialState.call(this); }

  encode(observation) {
    if (observation?.length !== 140 || !finite(observation)) throw Error('Invalid restricted observation');
    const w = this.weights;
    const fixed = Float32Array.from([...observation.slice(0, 18), ...observation.slice(114)]);
    if (fixed[10] <= 0.5) fixed.fill(0, 11, 18);
    const result = linear(fixed, w['encoder.fixed.weight'], w['encoder.fixed.bias']);
    const query = linear(fixed, w['encoder.query.weight'], w['encoder.query.bias']);
    const records = [];
    const objectSum = new Float64Array(96);
    for (let slot = 0; slot < 6; slot += 1) {
      const offset = 18 + slot * 16;
      if (observation[offset] <= 0.5) continue;
      const object = observation.slice(offset, offset + 16);
      const contribution = linear(object, w['encoder.object_linear']);
      for (let i = 0; i < 96; i += 1) objectSum[i] += contribution[i];
      let embedded = tanh(linear(object, w['encoder.embedding.0.weight'], w['encoder.embedding.0.bias']));
      embedded = tanh(linear(embedded, w['encoder.embedding.2.weight'], w['encoder.embedding.2.bias']));
      const key = linear(embedded, w['encoder.key.weight'], w['encoder.key.bias']);
      const value = linear(embedded, w['encoder.value.weight'], w['encoder.value.bias']);
      let dot = 0;
      for (let i = 0; i < 32; i += 1) dot += key[i] * query[i];
      records.push({ logit: Math.fround(dot / Math.sqrt(32)), value });
    }
    for (let i = 0; i < 96; i += 1) result[i] = Math.fround(result[i] + Math.fround(objectSum[i]));
    if (records.length) {
      // The null token has logit zero and a zero vector, including for empty sets.
      const maximum = Math.max(0, ...records.map(record => record.logit));
      let denominator = Math.exp(-maximum);
      for (const record of records) { record.mass = Math.exp(record.logit - maximum); denominator += record.mass; }
      const pooled = new Float64Array(32);
      for (const record of records) {
        const weight = Math.fround(record.mass / denominator);
        for (let i = 0; i < 32; i += 1) pooled[i] += Math.fround(weight * record.value[i]);
      }
      const residual = linear(Float32Array.from(pooled), w['encoder.residual.weight'], w['encoder.residual.bias']);
      for (let i = 0; i < 96; i += 1) result[i] += residual[i];
    }
    return result;
  }

  forward(observation, memory) {
    if (memory?.length !== 64 || !finite(memory)) throw Error('Invalid recurrent memory');
    const w = this.weights;
    const encoded = tanh(this.encode(observation));
    const input = linear(encoded, w['memory.weight_ih'], w['memory.bias_ih']);
    const state = linear(memory, w['memory.weight_hh'], w['memory.bias_hh']);
    const nextMemory = new Float32Array(64);
    for (let i = 0; i < 64; i += 1) {
      const reset = sigmoid(input[i] + state[i]);
      const update = sigmoid(input[64 + i] + state[64 + i]);
      const candidate = Math.tanh(input[128 + i] + reset * state[128 + i]);
      nextMemory[i] = candidate * (1 - update) + update * memory[i];
    }
    return { mean: linear(nextMemory, w['movement.weight'], w['movement.bias']),
      toolLogits: linear(nextMemory, w['tools.weight'], w['tools.bias']), memory: nextMemory };
  }

  act(physical, state, options) {
    return PersistentPolicy.prototype.act.call(this, physical, state, options);
  }
}

export function createEntityPolicies(model) {
  // Older imports retain the existing exact controller path.
  if (model?.format !== ENTITY_PAIR_FORMAT && POLICY_FORMATS.includes(model?.format)) return createPolicyControllers(model);
  if (model?.format !== ENTITY_PAIR_FORMAT || model.observationSize !== 140 ||
      model.physicsObservationSize !== 138 || model.actors?.length !== 2 ||
      JSON.stringify(model.commands) !== JSON.stringify(['keep', 'press', 'release']))
    throw Error('Unsupported explicitly typed policy pair');
  return model.actors.map(definition => {
    if (![ENTITY, LEGACY].includes(definition?.encoderType)) throw Error('Unknown actor encoder');
    validate(definition, definition.encoderType);
    return definition.encoderType === ENTITY ? new EntityPolicy(definition) : new PersistentPolicy(definition);
  });
}
