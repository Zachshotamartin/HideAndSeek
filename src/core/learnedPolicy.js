// Original recurrent policy inference. Matches training/actor.py; no planner,
// external model, hidden world state or scripted object-use actions enter here.
const sigmoid = value => 1 / (1 + Math.exp(-value));

function linear(input, weight, bias, output = new Float32Array(weight.length)) {
  for (let row = 0; row < weight.length; row += 1) {
    let total = bias[row];
    const coefficients = weight[row];
    for (let column = 0; column < input.length; column += 1) total += coefficients[column] * input[column];
    output[row] = total;
  }
  return output;
}

function gaussian(random) {
  return Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))) * Math.cos(2 * Math.PI * random());
}

export class PhysicalPolicy {
  constructor(definition) {
    if (definition?.observationSize !== 138 || definition?.hiddenSize !== 64 || definition?.encoderSize !== 96 ||
        !definition.weights || typeof definition.weights !== 'object') {
      throw new Error('Unsupported physical policy architecture');
    }
    this.observationSize = definition.observationSize;
    this.hiddenSize = definition.hiddenSize;
    this.weights = definition.weights;
    const w = this.weights;
    const dimensions = [
      ['encoder.weight', 96, this.observationSize], ['memory.weight_ih', this.hiddenSize * 3, 96],
      ['memory.weight_hh', this.hiddenSize * 3, this.hiddenSize],
      ['movement.weight', 3, this.hiddenSize], ['tools.weight', 2, this.hiddenSize],
    ];
    for (const [key, rows, columns] of dimensions) {
      if (!Array.isArray(w[key]) || w[key].length !== rows ||
          w[key].some(row => row.length !== columns || row.some(value => !Number.isFinite(value)))) {
        throw new Error(`Invalid physical policy matrix: ${key}`);
      }
    }
    for (const [key, length] of [['encoder.bias', 96], ['memory.bias_ih', this.hiddenSize * 3],
      ['memory.bias_hh', this.hiddenSize * 3], ['movement.bias', 3], ['tools.bias', 2], ['log_std', 3]]) {
      if (w[key]?.length !== length || w[key].some(value => !Number.isFinite(value))) {
        throw new Error(`Invalid physical policy vector: ${key}`);
      }
    }
  }

  initialMemory() { return new Float32Array(this.hiddenSize); }

  forward(observation, memory) {
    if (observation.length !== this.observationSize || memory.length !== this.hiddenSize ||
        observation.some(value => !Number.isFinite(value))) throw new Error('Invalid physical policy observation');
    const w = this.weights;
    const encoded = linear(observation, w['encoder.weight'], w['encoder.bias']);
    for (let i = 0; i < encoded.length; i += 1) encoded[i] = Math.tanh(encoded[i]);
    const inputGates = linear(encoded, w['memory.weight_ih'], w['memory.bias_ih']);
    const stateGates = linear(memory, w['memory.weight_hh'], w['memory.bias_hh']);
    const nextMemory = new Float32Array(this.hiddenSize);
    // PyTorch GRU gate order is reset, update, candidate. Reset multiplies
    // the hidden projection *including* its candidate bias.
    for (let i = 0; i < this.hiddenSize; i += 1) {
      const reset = sigmoid(inputGates[i] + stateGates[i]);
      const update = sigmoid(inputGates[this.hiddenSize + i] + stateGates[this.hiddenSize + i]);
      const candidate = Math.tanh(inputGates[this.hiddenSize * 2 + i] + reset * stateGates[this.hiddenSize * 2 + i]);
      nextMemory[i] = candidate * (1 - update) + update * memory[i];
    }
    return { mean: linear(nextMemory, w['movement.weight'], w['movement.bias']),
      toolLogits: linear(nextMemory, w['tools.weight'], w['tools.bias']), memory: nextMemory };
  }

  act(observation, memory, { deterministic = true, random = Math.random } = {}) {
    const prediction = this.forward(observation, memory);
    const action = new Float32Array(5);
    for (let axis = 0; axis < 3; axis += 1) {
      const deviation = Math.exp(Math.max(-2.5, Math.min(.3, this.weights.log_std[axis])));
      action[axis] = Math.tanh(prediction.mean[axis] + (deterministic ? 0 : deviation * gaussian(random)));
    }
    for (let tool = 0; tool < 2; tool += 1) {
      const probability = sigmoid(prediction.toolLogits[tool]);
      action[3 + tool] = deterministic ? Number(probability >= .5) : Number(random() < probability);
    }
    return { ...prediction, action };
  }
}

export function createPhysicalPolicies(model) {
  if (model.format !== 'original-mujoco-recurrent-ppo-v1' || model.actors?.length !== 2) {
    throw new Error('Unsupported physical hide-and-seek model');
  }
  return model.actors.map(actor => new PhysicalPolicy(actor));
}
