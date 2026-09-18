// Original masked object-set inference with explicit actor-only validation.
import { PersistentPolicy } from './persistentPolicy.js';
import { NOISE, TAG_FORMAT, TAG_SCHEMA } from './gameRules.js';

export const KNOWN_POSITION_SCHEMA = 'known-opponent-position-210-v1';
export const ENTITY_PAIR_FORMAT = 'original-mujoco-relational-policy-pair-v2';
export const JUMP_PAIR_FORMAT = 'original-mujoco-relational-jump-policy-pair-v4';
export const RELATIONAL_FORMATS = [ENTITY_PAIR_FORMAT, JUMP_PAIR_FORMAT, TAG_FORMAT];
const ENTITY = 'object-relations-v2';
const JUMP_ENTITY = 'object-relations-jump-v4';
const HEADS = 4;
const SLOTS = 10;
const SLOT = 16;
const OBJECT_START = 18;
const OBJECT_END = OBJECT_START + SLOTS * SLOT;
const PHYSICS = 208;
const finite = values => values.every(Number.isFinite);
const sigmoid = value => 1 / (1 + Math.exp(-value));

/** Input layout of one actor: how many physical values, buttons and noise values it reads. */
function layoutOf(d) {
  const physical = d.physicsObservationSize;
  const rho = d.noiseRho ?? 0;
  const noise = rho > 0 ? NOISE : 0;
  if (!Number.isInteger(physical) || physical < PHYSICS || !(rho >= 0 && rho < 1) ||
      d.observationSize !== physical + 2 + noise)
    throw Error('Unsupported relational observation layout');
  return { observationSize: d.observationSize, physical, rho, noise, fixed: d.observationSize - (OBJECT_END - OBJECT_START) };
}

function validate(d) {
  if (!d || ![ENTITY, JUMP_ENTITY].includes(d.encoderType)) throw Error('Unsupported relational model');
  const { hiddenSize: H, encoderSize: E, embeddingSize: D } = d;
  if (![H, E, D].every(n => Number.isInteger(n) && n > 0) || D % HEADS) throw Error('Unsupported relational model');
  const layout = layoutOf(d);
  const movement = d.encoderType === JUMP_ENTITY ? 4 : 3;
  const shapes = {
    'memory.weight_ih': [3 * H, E], 'memory.weight_hh': [3 * H, H],
    'memory.bias_ih': [3 * H], 'memory.bias_hh': [3 * H],
    'movement.weight': [movement, H], 'movement.bias': [movement],
    'tools.weight': [6, H], 'tools.bias': [6], log_std: [movement],
    'encoder.fixed.weight': [E, layout.fixed], 'encoder.fixed.bias': [E],
    'encoder.object_linear': [E, SLOT],
    'encoder.embedding.0.weight': [D, SLOT], 'encoder.embedding.0.bias': [D],
    'encoder.embedding.2.weight': [D, D], 'encoder.embedding.2.bias': [D],
    'encoder.query.weight': [D, layout.fixed], 'encoder.query.bias': [D],
    'encoder.key.weight': [D, D], 'encoder.key.bias': [D],
    'encoder.value.weight': [D, D], 'encoder.value.bias': [D],
    'encoder.residual.weight': [E, D], 'encoder.residual.bias': [E],
    'encoder.relations.in_proj_weight': [3 * D, D], 'encoder.relations.in_proj_bias': [3 * D],
    'encoder.relations.out_proj.weight': [D, D], 'encoder.relations.out_proj.bias': [D],
    'encoder.relation_scale': [],
  };
  if (!d.weights || Object.keys(d.weights).some(key => !Object.hasOwn(shapes, key)))
    throw Error('Expected actor-only relational weights; unknown or critic parameter');
  const matches = (value, shape) => shape.length === 0 ? Number.isFinite(value) :
    Array.isArray(value) && value.length === shape[0] &&
      Array.from(value).every(row => matches(row, shape.slice(1)));
  for (const [key, shape] of Object.entries(shapes))
    if (!matches(d.weights[key], shape)) throw Error('Invalid relational weights: ' + key);
  return layout;
}

function linear(input, weight, bias) {
  return Float32Array.from(weight, (row, index) => {
    let sum = bias ? bias[index] : 0;
    for (let column = 0; column < row.length; column += 1) sum += row[column] * input[column];
    return sum;
  });
}
const tanh = input => Float32Array.from(input, Math.tanh);

export class RelationalPolicy {
  constructor(definition, observationSchema = undefined) {
    if (observationSchema !== undefined && ![KNOWN_POSITION_SCHEMA, TAG_SCHEMA].includes(observationSchema))
      throw Error('Unknown opponent observation schema');
    const layout = validate(definition);
    if ((observationSchema === TAG_SCHEMA) !== (layout.physical !== PHYSICS))
      throw Error('The observation schema and the actor input width disagree');
    this.observationSchema = observationSchema;
    this.weights = definition.weights;
    this.hiddenSize = definition.hiddenSize;
    this.encoderSize = definition.encoderSize;
    this.embeddingSize = definition.embeddingSize;
    this.physicalSize = layout.physical;
    this.physicsObservationSize = layout.physical;
    this.observationSize = layout.observationSize;
    this.noiseRho = layout.rho;
    this.format = definition.encoderType === JUMP_ENTITY ? (layout.noise ? TAG_FORMAT : JUMP_PAIR_FORMAT) : ENTITY_PAIR_FORMAT;
  }

  initialState() {
    const state = { memory: new Float32Array(this.hiddenSize), buttons: new Float32Array(2) };
    if (this.noiseRho > 0) state.noise = new Float32Array(NOISE);
    return state;
  }

  encode(observation) {
    if (observation.length !== this.observationSize || !finite(observation)) throw Error('Invalid relational observation');
    const w = this.weights, E = this.encoderSize, D = this.embeddingSize, perHead = D / HEADS;
    const fixed = Float32Array.from([...observation.slice(0, OBJECT_START), ...observation.slice(OBJECT_END)]);
    if (fixed[10] <= .5) fixed.fill(0, this.observationSchema === KNOWN_POSITION_SCHEMA ? 14 : 11, 18);
    const result = linear(fixed, w['encoder.fixed.weight'], w['encoder.fixed.bias']);
    const objects = []; const sum = new Float64Array(E);
    for (let slot = 0; slot < SLOTS; slot++) {
      const o = observation.slice(OBJECT_START + slot * SLOT, OBJECT_START + (slot + 1) * SLOT); if (o[0] <= .5) continue;
      const v = linear(o, w['encoder.object_linear']); for (let i = 0; i < E; i++) sum[i] += v[i];
      objects.push(tanh(linear(tanh(linear(o, w['encoder.embedding.0.weight'], w['encoder.embedding.0.bias'])), w['encoder.embedding.2.weight'], w['encoder.embedding.2.bias'])));
    }
    for (let i = 0; i < E; i++) result[i] = Math.fround(result[i] + Math.fround(sum[i]));
    if (!objects.length) return result;
    const tokens = [...objects, new Float32Array(D)];
    const qkv = tokens.map(t => linear(t, w['encoder.relations.in_proj_weight'], w['encoder.relations.in_proj_bias']));
    const related = objects.map((o, index) => {
      const joined = new Float32Array(D);
      for (let head = 0; head < HEADS; head++) {
        const logits = qkv.map(t => { let z = 0; for (let k = 0; k < perHead; k++) z += qkv[index][head * perHead + k] * t[D + head * perHead + k]; return z / Math.sqrt(perHead); });
        const max = Math.max(...logits), masses = logits.map(x => Math.exp(x - max)), den = masses.reduce((a, b) => a + b, 0);
        for (let k = 0; k < perHead; k++) { let z = 0; for (let j = 0; j < tokens.length; j++) z += masses[j] / den * qkv[j][2 * D + head * perHead + k]; joined[head * perHead + k] = z; }
      }
      const v = linear(joined, w['encoder.relations.out_proj.weight'], w['encoder.relations.out_proj.bias']);
      return Float32Array.from(o, (x, i) => x + w['encoder.relation_scale'] * v[i]);
    });
    const query = linear(fixed, w['encoder.query.weight'], w['encoder.query.bias']);
    const records = related.map(t => { const key = linear(t, w['encoder.key.weight'], w['encoder.key.bias']); let z = 0; for (let i = 0; i < D; i++) z += key[i] * query[i]; return { logit: Math.fround(z / Math.sqrt(D)), value: linear(t, w['encoder.value.weight'], w['encoder.value.bias']) }; });
    const max = Math.max(0, ...records.map(r => r.logit)); let den = Math.exp(-max); for (const r of records) { r.mass = Math.exp(r.logit - max); den += r.mass; }
    const pooled = new Float64Array(D); for (const r of records) for (let i = 0; i < D; i++) pooled[i] += Math.fround(Math.fround(r.mass / den) * r.value[i]);
    const residual = linear(Float32Array.from(pooled), w['encoder.residual.weight'], w['encoder.residual.bias']); for (let i = 0; i < E; i++) result[i] += residual[i];
    return result;
  }

  forward(observation, memory) {
    const H = this.hiddenSize;
    if (memory?.length !== H || !finite(memory)) throw Error('Invalid recurrent memory');
    const w = this.weights;
    const encoded = tanh(this.encode(observation));
    const input = linear(encoded, w['memory.weight_ih'], w['memory.bias_ih']);
    const state = linear(memory, w['memory.weight_hh'], w['memory.bias_hh']);
    const nextMemory = new Float32Array(H);
    for (let i = 0; i < H; i += 1) {
      const reset = sigmoid(input[i] + state[i]);
      const update = sigmoid(input[H + i] + state[H + i]);
      const candidate = Math.tanh(input[2 * H + i] + reset * state[2 * H + i]);
      nextMemory[i] = candidate * (1 - update) + update * memory[i];
    }
    return { mean: linear(nextMemory, w['movement.weight'], w['movement.bias']),
      toolLogits: linear(nextMemory, w['tools.weight'], w['tools.bias']), memory: nextMemory };
  }

  act(physical, state, options) {
    return PersistentPolicy.prototype.act.call(this, physical, state, options);
  }
}

export function createRelationalPolicies(model) {
  if (!model || !RELATIONAL_FORMATS.includes(model.format) || model.actors?.length !== 2 ||
      !Number.isInteger(model.observationSize) || !Number.isInteger(model.physicsObservationSize) ||
      JSON.stringify(model.commands) !== JSON.stringify(['keep', 'press', 'release']) ||
      (model.actionSize !== undefined && model.actionSize !== (model.format === ENTITY_PAIR_FORMAT ? 5 : 6)))
    throw Error('Invalid relational policy pair');
  const tag = model.format === TAG_FORMAT;
  if (tag ? model.observationSchema !== TAG_SCHEMA : model.observationSchema === TAG_SCHEMA)
    throw Error('The tag-round pair format requires its observation schema');
  if (!tag && (model.observationSize !== 210 || model.physicsObservationSize !== 208))
    throw Error('Invalid relational policy pair');
  const expected = model.format === ENTITY_PAIR_FORMAT ? ENTITY : JUMP_ENTITY;
  if (model.actors.some(actor => actor?.encoderType !== expected)) throw Error('Mixed relational action schemas');
  const policies = model.actors.map(d => new RelationalPolicy(d, model.observationSchema));
  if (policies.some(p => p.physicsObservationSize !== model.physicsObservationSize || p.observationSize !== model.observationSize))
    throw Error('Actor input widths disagree with the pair record');
  return policies;
}
