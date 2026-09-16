// Tag-round rules and the v6 actor observation, kept beside the physics mirror.
// Nothing here changes the physical simulation; index.js applies the round
// rules the way training_v5/capture.py and game.py do beside physics.py.
export const TAG_SCHEMA = 'tag-rounds-last-seen-214-v6';
export const TAG_FORMAT = 'original-mujoco-relational-tag-rounds-pair-v6';
export const NOISE = 4; // observed AR(1) exploration-noise values of the four movement outputs
export const EXTRAS = 6; // last-seen flag, relative x, y, z, age, time remaining
export const CAPTURE_DISTANCE = 0.7; // metres between agent centres
export const MEMORY_AGE = 10; // seconds at which the last-seen age saturates
export const CLOCK = 60; // seconds that scale the remaining play time
export const PREP_FRACTION = 0.4;
export const MINIMUM_PREP = 96;
export const ROUND_PLAY = 375; // 30 s browser rounds
const POSITION_SCALE = 6;
const HEIGHT_SCALE = 2;

export function preparationSteps(play) {
  if (!Number.isInteger(play) || play < 1) throw Error('The play length must be a positive step count.');
  return Math.max(MINIMUM_PREP, Math.round(PREP_FRACTION * play));
}

export function observationWidth(schema) {
  return 208 + (schema === TAG_SCHEMA ? EXTRAS : 0);
}

export function remainingPlaySeconds(sim) {
  return Math.max(0, sim.prep + sim.play - Math.max(sim.t, sim.prep)) * sim.dt;
}

/** One agent's six extra measurements: last-seen opponent offset (own frame), its age, the clock. */
export function extras(sim, a) {
  const clock = Math.min(1, remainingPlaySeconds(sim) / CLOCK);
  const seen = sim.lastSeen[a];
  if (!seen) return [0, 0, 0, 0, 1, clock];
  const q = sim.data.qpos;
  const dx = seen.position[0] - q[a * 4];
  const dy = seen.position[1] - q[a * 4 + 1];
  const dz = seen.position[2] - q[a * 4 + 2];
  const yaw = q[a * 4 + 3];
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const age = Math.min(MEMORY_AGE, (sim.t - seen.t) * sim.dt) / MEMORY_AGE;
  return [1, (c * dx + s * dy) / POSITION_SCALE, (-s * dx + c * dy) / POSITION_SCALE, dz / HEIGHT_SCALE, age, clock];
}

/** True during play when the seeker sees the hider from within reach. */
export function captured(sim) {
  if (sim.t <= sim.prep || !sim.seen[1][0]) return false;
  const q = sim.data.qpos;
  return Math.hypot(q[0] - q[4], q[1] - q[5]) < CAPTURE_DISTANCE;
}
