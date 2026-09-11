import {createRelationalPolicies} from './relationalPolicy.js';
import { createEntityPolicies } from './entityPolicy.js';
import { createPhysicalPolicies } from './learnedPolicy.js';
import { createPersistentPolicies } from './persistentPolicy.js';

export const PERSISTENT_FORMAT = 'original-mujoco-persistent-buttons-ppo-v1';
export const BINARY_FORMAT = 'original-mujoco-recurrent-ppo-v1';
export const POLICY_FORMATS = Object.freeze(['original-mujoco-relational-jump-policy-pair-v4','original-mujoco-relational-policy-pair-v2',PERSISTENT_FORMAT, BINARY_FORMAT, 'original-mujoco-entity-policy-pair-v1']);
const ACTOR_WEIGHTS = new Set([
  'encoder.weight', 'encoder.bias', 'memory.weight_ih', 'memory.weight_hh',
  'memory.bias_ih', 'memory.bias_hh', 'movement.weight', 'movement.bias',
  'tools.weight', 'tools.bias', 'log_std',
]);

// Explicit dispatch: a binary tool action is never reinterpreted as a command.
// Only an actor's own recurrent state is passed here; a training critic has no
// place in either browser format or its inputs.
export function createPolicyControllers(model) {
 if(['original-mujoco-relational-jump-policy-pair-v4','original-mujoco-relational-policy-pair-v2'].includes(model?.format))return createRelationalPolicies(model);
  if (model?.format === 'original-mujoco-entity-policy-pair-v1')
    return createEntityPolicies(model).map(policy => ({
      format: model.format, observationSize: 140, physicsObservationSize: 138,
      initialState: () => policy.initialState(),
      act: (physical, state, options) => policy.act(physical, state, options),
    }));
  if (!POLICY_FORMATS.includes(model?.format) || !Array.isArray(model.actors) || model.actors.length !== 2)
    throw Error('Unsupported policy format. Use a persistent-button or legacy binary actor export.');
  for (const actor of model.actors) {
    if (!actor?.weights || Object.keys(actor.weights).some(key => !ACTOR_WEIGHTS.has(key)))
      throw Error('Expected actor-only weights; critic or unknown parameters are not supported.');
  }
  if (model.format === PERSISTENT_FORMAT) {
    if (model.observationSize !== 140 || model.physicsObservationSize !== 138)
      throw Error('Persistent policies require 138 physical values plus two own button states.');
    return createPersistentPolicies(model).map(policy => ({
      format: PERSISTENT_FORMAT,
      observationSize: 140,
      physicsObservationSize: 138,
      initialState: () => policy.initialState(),
      act: (physical, state, options) => policy.act(physical, state, options),
    }));
  }
  if (model.observationSize !== undefined && model.observationSize !== 138)
    throw Error('Legacy binary policies require the 138-value physical observation.');
  return createPhysicalPolicies(model).map(policy => ({
    format: BINARY_FORMAT,
    observationSize: 138,
    physicsObservationSize: 138,
    initialState: () => ({ memory: policy.initialMemory() }),
    act(physical, state, options) {
      if (state?.memory?.length !== 64 || !state.memory.every(Number.isFinite))
        throw Error('Invalid legacy recurrent state.');
      const out = policy.act(physical, state.memory, options);
      if (physical[7] < .5 && physical[5] < 1) out.action.fill(0);
      return { ...out, state: { memory: out.memory } };
    },
  }));
}
