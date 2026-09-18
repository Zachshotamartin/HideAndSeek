// Learned Keep/Press/Release commands drive ordinary physical buttons.
// Requested button state is part of each actor's own next observation.
import { NOISE } from './gameRules.js';

const FORMAT = 'original-mujoco-persistent-buttons-ppo-v1';
const sigmoid = (value) => 1 / (1 + Math.exp(-value));
const finite = (values) => values.every(Number.isFinite);

function linear(input, weight, bias) {
  return Float32Array.from(weight, (row, index) => {
    let total = bias[index];
    for (let column = 0; column < input.length; column += 1) total += row[column] * input[column];
    return total;
  });
}

function sampleCommand(logits, deterministic, random) {
  const maximum = Math.max(...logits);
  if (deterministic) return logits.indexOf(maximum);
  const weights = logits.map((value) => Math.exp(value - maximum));
  const threshold = random() * weights.reduce((sum, value) => sum + value, 0);
  let sum = 0;
  for (let command = 0; command < 2; command += 1) {
    sum += weights[command];
    if (threshold < sum) return command;
  }
  return 2;
}

export function advanceButtons(previous, commands, blind = false) {
  if (
    previous?.length !== 2 ||
    commands?.length !== 2 ||
    !previous.every((value) => value === 0 || value === 1) ||
    !commands.every((value) => Number.isInteger(value) && value >= 0 && value <= 2)
  ) {
    throw new Error('Expected two binary states and Keep/Press/Release commands');
  }
  return Float32Array.from(commands, (command, index) =>
    blind ? 0 : command === 1 ? 1 : command === 2 ? 0 : previous[index],
  );
}

export class PersistentPolicy {
  constructor(definition) {
    if (
      definition?.observationSize !== 140 ||
      definition?.physicsObservationSize !== 138 ||
      definition?.hiddenSize !== 64 ||
      definition?.encoderSize !== 96 ||
      !definition.weights
    ) {
      throw new Error('Unsupported persistent-button architecture');
    }
    this.weights = definition.weights;
    for (const [key, rows, columns] of [
      ['encoder.weight', 96, 140],
      ['memory.weight_ih', 192, 96],
      ['memory.weight_hh', 192, 64],
      ['movement.weight', 3, 64],
      ['tools.weight', 6, 64],
    ]) {
      const matrix = this.weights[key];
      if (
        !Array.isArray(matrix) ||
        matrix.length !== rows ||
        matrix.some((row) => !Array.isArray(row) || row.length !== columns || !finite(row))
      ) {
        throw new Error(`Invalid persistent matrix: ${key}`);
      }
    }
    for (const [key, length] of [
      ['encoder.bias', 96],
      ['memory.bias_ih', 192],
      ['memory.bias_hh', 192],
      ['movement.bias', 3],
      ['tools.bias', 6],
      ['log_std', 3],
    ]) {
      if (
        !Array.isArray(this.weights[key]) ||
        this.weights[key].length !== length ||
        !finite(this.weights[key])
      ) {
        throw new Error(`Invalid persistent vector: ${key}`);
      }
    }
  }

  initialState() {
    return { memory: new Float32Array(64), buttons: new Float32Array(2) };
  }

  forward(observation, memory) {
    if (
      observation?.length !== 140 ||
      memory?.length !== 64 ||
      !finite(observation) ||
      !finite(memory)
    ) {
      throw new Error('Invalid persistent observation or memory');
    }
    const w = this.weights;
    const encoded = linear(observation, w['encoder.weight'], w['encoder.bias']);
    for (let i = 0; i < 96; i += 1) encoded[i] = Math.tanh(encoded[i]);
    const input = linear(encoded, w['memory.weight_ih'], w['memory.bias_ih']);
    const state = linear(memory, w['memory.weight_hh'], w['memory.bias_hh']);
    const nextMemory = new Float32Array(64);
    for (let i = 0; i < 64; i += 1) {
      const reset = sigmoid(input[i] + state[i]);
      const update = sigmoid(input[64 + i] + state[64 + i]);
      const candidate = Math.tanh(input[128 + i] + reset * state[128 + i]);
      nextMemory[i] = candidate * (1 - update) + update * memory[i];
    }
    return {
      mean: linear(nextMemory, w['movement.weight'], w['movement.bias']),
      toolLogits: linear(nextMemory, w['tools.weight'], w['tools.bias']),
      memory: nextMemory,
    };
  }

  act(physical, state, { deterministic = false, random = Math.random } = {}) {
    // Sizes come from the actor's own layout; the original persistent actor
    // reads 138 physical values and no exploration noise.
    const physicalSize = this.physicalSize ?? 138;
    const rho = this.noiseRho ?? 0;
    const noiseSize = rho > 0 ? NOISE : 0;
    if (
      physical?.length !== physicalSize ||
      !finite(physical) ||
      !state ||
      state.buttons?.length !== 2 ||
      !state.buttons.every((value) => value === 0 || value === 1) ||
      (noiseSize > 0 && (state.noise?.length !== noiseSize || !finite(state.noise)))
    )
      throw new Error('Invalid persistent state');
    // Seeker preparation commands are ignored, including requested button state.
    // Recurrence advances normally, matching the training actor and critic.
    const blind = physical[7] < 0.5 && physical[5] < 1;
    const previous = blind ? new Float32Array(2) : state.buttons;
    const observation = new Float32Array(physicalSize + 2 + noiseSize);
    observation.set(physical);
    observation.set(previous, physicalSize);
    if (noiseSize > 0) observation.set(state.noise, physicalSize + 2);
    const prediction = this.forward(observation, state.memory);
    const dimensions = prediction.mean.length;
    const action = new Float32Array(dimensions + 2);
    const noise = new Float32Array(noiseSize);
    const spread = Math.sqrt(1 - rho * rho);
    for (let axis = 0; axis < dimensions; axis += 1) {
      const gaussian = deterministic
        ? 0
        : Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))) * Math.cos(2 * Math.PI * random());
      const deviation = Math.exp(Math.max(-2.5, Math.min(0.3, this.weights.log_std[axis]))) * spread;
      // The head's own mean is the deterministic action; sampling adds the
      // carried noise and a fresh innovation, and the difference is carried on.
      const base = prediction.mean[axis];
      const carried = noiseSize > 0 ? rho * state.noise[axis] : 0;
      const movement = deterministic ? base : base + carried + deviation * gaussian;
      if (noiseSize > 0) noise[axis] = movement - base;
      action[axis < 3 ? axis : 5] = Math.tanh(movement);
    }
    const commands = Int32Array.from([0, 1], (tool) =>
      sampleCommand(
        Array.from(prediction.toolLogits.slice(tool * 3, tool * 3 + 3)),
        deterministic,
        random,
      ),
    );
    const buttons = advanceButtons(previous, commands, blind);
    action.set(buttons, 3);
    if (blind) action.fill(0);
    const next = { memory: prediction.memory, buttons };
    if (noiseSize > 0) next.noise = noise;
    return {
      ...prediction,
      action,
      commands,
      buttons,
      noise,
      state: next,
    };
  }
}

export function createPersistentPolicies(model) {
  if (
    model?.format !== FORMAT ||
    model?.actors?.length !== 2 ||
    JSON.stringify(model.commands) !== JSON.stringify(['keep', 'press', 'release'])
  ) {
    throw new Error('Unsupported persistent-button model');
  }
  return model.actors.map((definition) => new PersistentPolicy(definition));
}
