// Original masked object-set inference with explicit actor-only validation.
import { PersistentPolicy } from './persistentPolicy.js';

export const KNOWN_POSITION_SCHEMA = 'known-opponent-position-210-v1';
export const ENTITY_PAIR_FORMAT = 'original-mujoco-relational-policy-pair-v2';
const ENTITY = 'object-relations-v2';
const finite = values => values.every(Number.isFinite);
const sigmoid = value => 1 / (1 + Math.exp(-value));

function validate(d) {
  if (!d || !['object-relations-v2', 'object-relations-jump-v4'].includes(d.encoderType) ||
      d.observationSize !== 210 || d.physicsObservationSize !== 208 ||
      d.hiddenSize !== 256 || d.encoderSize !== 256 || d.embeddingSize !== 128)
    throw Error('Unsupported relational model');
  const movement = d.encoderType === 'object-relations-jump-v4' ? 4 : 3;
  const shapes = {
    'memory.weight_ih': [768,256], 'memory.weight_hh': [768,256],
    'memory.bias_ih': [768], 'memory.bias_hh': [768],
    'movement.weight': [movement,256], 'movement.bias': [movement],
    'tools.weight': [6,256], 'tools.bias': [6], log_std: [movement],
    'encoder.fixed.weight': [256,50], 'encoder.fixed.bias': [256],
    'encoder.object_linear': [256,16],
    'encoder.embedding.0.weight': [128,16], 'encoder.embedding.0.bias': [128],
    'encoder.embedding.2.weight': [128,128], 'encoder.embedding.2.bias': [128],
    'encoder.query.weight': [128,50], 'encoder.query.bias': [128],
    'encoder.key.weight': [128,128], 'encoder.key.bias': [128],
    'encoder.value.weight': [128,128], 'encoder.value.bias': [128],
    'encoder.residual.weight': [256,128], 'encoder.residual.bias': [256],
    'encoder.relations.in_proj_weight': [384,128], 'encoder.relations.in_proj_bias': [384],
    'encoder.relations.out_proj.weight': [128,128], 'encoder.relations.out_proj.bias': [128],
    'encoder.relation_scale': [],
  };
  if (!d.weights || Object.keys(d.weights).some(key => !Object.hasOwn(shapes, key)))
    throw Error('Expected actor-only relational weights; unknown or critic parameter');
  const matches = (value, shape) => shape.length === 0 ? Number.isFinite(value) :
    Array.isArray(value) && value.length === shape[0] &&
      Array.from(value).every(row => matches(row, shape.slice(1)));
  for (const [key, shape] of Object.entries(shapes))
    if (!matches(d.weights[key], shape)) throw Error('Invalid relational weights: ' + key);
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
    if (observationSchema !== undefined && observationSchema !== KNOWN_POSITION_SCHEMA) throw Error('Unknown opponent observation schema');
    this.observationSchema = observationSchema;
    validate(definition, ENTITY);
    this.weights = definition.weights;this.physicalSize=208;this.physicsObservationSize=208;this.observationSize=210;this.format=definition.encoderType==='object-relations-jump-v4'?'original-mujoco-relational-jump-policy-pair-v4':ENTITY_PAIR_FORMAT;
  }

  initialState() { return {memory:new Float32Array(256),buttons:new Float32Array(2)}; }

  encode(observation) {
    if(observation.length!==210||!finite(observation))throw Error('Invalid relational observation');
    const w=this.weights,fixed=Float32Array.from([...observation.slice(0,18),...observation.slice(178)]);
    if(fixed[10]<=.5)fixed.fill(0,this.observationSchema === KNOWN_POSITION_SCHEMA ? 14 : 11,18);
    const result=linear(fixed,w['encoder.fixed.weight'],w['encoder.fixed.bias']);
    const objects=[];const sum=new Float64Array(256);
    for(let slot=0;slot<10;slot++){
      const o=observation.slice(18+slot*16,34+slot*16);if(o[0]<=.5)continue;
      const v=linear(o,w['encoder.object_linear']);for(let i=0;i<256;i++)sum[i]+=v[i];
      objects.push(tanh(linear(tanh(linear(o,w['encoder.embedding.0.weight'],w['encoder.embedding.0.bias'])),w['encoder.embedding.2.weight'],w['encoder.embedding.2.bias'])));
    }
    for(let i=0;i<256;i++)result[i]=Math.fround(result[i]+Math.fround(sum[i]));
    if(!objects.length)return result;
    const tokens=[...objects,new Float32Array(128)];
    const qkv=tokens.map(t=>linear(t,w['encoder.relations.in_proj_weight'],w['encoder.relations.in_proj_bias']));
    const related=objects.map((o,index)=>{
      const joined=new Float32Array(128);
      for(let head=0;head<4;head++){
        const logits=qkv.map(t=>{let z=0;for(let k=0;k<32;k++)z+=qkv[index][head*32+k]*t[128+head*32+k];return z/Math.sqrt(32);});
        const max=Math.max(...logits),masses=logits.map(x=>Math.exp(x-max)),den=masses.reduce((a,b)=>a+b,0);
        for(let k=0;k<32;k++){let z=0;for(let j=0;j<tokens.length;j++)z+=masses[j]/den*qkv[j][256+head*32+k];joined[head*32+k]=z;}
      }
      const v=linear(joined,w['encoder.relations.out_proj.weight'],w['encoder.relations.out_proj.bias']);
      return Float32Array.from(o,(x,i)=>x+w['encoder.relation_scale']*v[i]);
    });
    const query=linear(fixed,w['encoder.query.weight'],w['encoder.query.bias']);
    const records=related.map(t=>{const key=linear(t,w['encoder.key.weight'],w['encoder.key.bias']);let z=0;for(let i=0;i<128;i++)z+=key[i]*query[i];return {logit:Math.fround(z/Math.sqrt(128)),value:linear(t,w['encoder.value.weight'],w['encoder.value.bias'])};});
    const max=Math.max(0,...records.map(r=>r.logit));let den=Math.exp(-max);for(const r of records){r.mass=Math.exp(r.logit-max);den+=r.mass;}
    const pooled=new Float64Array(128);for(const r of records)for(let i=0;i<128;i++)pooled[i]+=Math.fround(Math.fround(r.mass/den)*r.value[i]);
    const residual=linear(Float32Array.from(pooled),w['encoder.residual.weight'],w['encoder.residual.bias']);for(let i=0;i<256;i++)result[i]+=residual[i];
    return result;
  }

  forward(observation, memory) {
    if (memory?.length !== 256 || !finite(memory)) throw Error('Invalid recurrent memory');
    const w = this.weights;
    const encoded = tanh(this.encode(observation));
    const input = linear(encoded, w['memory.weight_ih'], w['memory.bias_ih']);
    const state = linear(memory, w['memory.weight_hh'], w['memory.bias_hh']);
    const nextMemory = new Float32Array(256);
    for (let i = 0; i < 256; i += 1) {
      const reset = sigmoid(input[i] + state[i]);
      const update = sigmoid(input[256 + i] + state[256 + i]);
      const candidate = Math.tanh(input[512 + i] + reset * state[512 + i]);
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
  const jumpFormat = 'original-mujoco-relational-jump-policy-pair-v4';
  if (!model || ![ENTITY_PAIR_FORMAT,jumpFormat].includes(model.format) || model.actors?.length !== 2 ||
      model.observationSize !== 210 || model.physicsObservationSize !== 208 ||
      JSON.stringify(model.commands) !== JSON.stringify(['keep','press','release']) ||
      (model.actionSize !== undefined && model.actionSize !== (model.format === jumpFormat ? 6 : 5)))
    throw Error('Invalid relational policy pair');
  const expected = model.format === jumpFormat ? 'object-relations-jump-v4' : ENTITY;
  if (model.actors.some(actor => actor?.encoderType !== expected)) throw Error('Mixed relational action schemas');
  return model.actors.map(d => new RelationalPolicy(d, model.observationSchema));
}
