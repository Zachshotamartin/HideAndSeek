import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Simulation, generateArena, validateArena, editBlock, lineClear, gridFor, Random, connected } from '../src/core/arena.js';
import { logits, chooseAction, validateModel } from '../src/core/policy.js';
const arena = { version: 1, seed: 91, width: 12, height: 12, blocks: [{ x: 5, y: 3, width: 1, height: 5 }], spawns: [[3.5, 5.5], [7.5, 5.5]] };
// Tests use legal editor dimensions; adjacent pieces form a longer wall.
const world = () => ({ ...structuredClone(arena), blocks: [{ x: 5, y: 3, width: 1, height: 3 }, { x: 5, y: 6, width: 1, height: 2 }] });
const near = (a, b, tolerance = 2e-5) => { assert.equal(a.length, b.length); a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= tolerance, `${i}: ${v} != ${b[i]}`)); };

test('seeded bounded generator creates connected valid layouts and deterministic spawns', () => {
  for (const size of [10, 14, 20]) for (let seed = 1; seed <= 20; seed++) { const a = generateArena(seed, size, size, 12); assert.deepEqual(a, generateArena(seed, size, size, 12)); assert.ok(connected(a)); validateArena(a); }
  assert.throws(() => generateArena(1, 21, 12, 3)); assert.throws(() => generateArena(1, 12, 12, 13));
});
test('exact line of sight handles parallel slabs, corner grazing and symmetry', () => {
  const a = world(), g = gridFor(a);
  assert.equal(lineClear(a, g, [3, 5], [8, 5]), false);
  assert.equal(lineClear(a, g, [3, 2], [8, 2]), true);
  assert.equal(lineClear(a, g, [4, 4], [6, 2]), false); // exactly touches (5,3)
  assert.equal(lineClear(a, g, [4, 3.99], [6, 1.99]), true);
  for (const [p, q] of [[[2.5, 2.5], [9.5, 9.5]], [[4, 4], [6, 2]], [[5, 1.5], [5, 9.5]]]) assert.equal(lineClear(a, g, p, q), lineClear(a, g, q, p));
});
test('walls stop motion and a tag cannot happen through cover', () => {
  const s = new Simulation(world()); s.t = 24;
  for (let i = 0; i < 30; i++) { s.step([2, 4]); assert.ok(s.pos[0][0] < 5 - .24); assert.ok(s.pos[1][0] > 6 + .24); assert.equal(s.capture, false); }
  assert.ok(s.collisions[0] > 0); assert.ok(s.collisions[1] > 0);
});
test('seeker preparation masks sight and memory, and freezes seeker movement', () => {
  const s = new Simulation({ ...world(), blocks: [], spawns: [[3.5, 3.5], [5.5, 3.5]] });
  const start = s.pos[1].slice();
  for (let i = 0; i < 24; i++) { const o = s.observe(1); assert.deepEqual(Array.from(o.slice(8, 14)), [0, 0, 0, 0, 0, 0]); assert.equal(s.known[1], false); s.step([0, 2]); assert.deepEqual(s.pos[1], start); }
  assert.equal(s.observe(1)[8], 1); assert.equal(s.observe(1)[11], 1);
});
test('hidden opponent changes do not leak into observation or actor logits', () => {
  const s = new Simulation(world()); s.t = 30; s.memory[1] = [3.5, 4.5]; s.known[1] = true; s.age[1] = 8; s.updateSight(false);
  const before = s.observe(1); s.pos[0] = [3.5, 6.5]; s.updateSight(false); const after = s.observe(1); assert.deepEqual(after, before);
  const models = JSON.parse(fs.readFileSync(new URL('../public/models/hide-seek.json', import.meta.url)));
  assert.deepEqual(logits(models.trained.policies[1], after), logits(models.trained.policies[1], before));
  s.pos[1][1] += 1; const moved = s.observe(1); assert.notEqual(moved[13], before[13]); assert.deepEqual(s.memory[1], [3.5, 4.5]);
});
test('clear-sight tag and timeout terminate with correct dominant rewards', () => {
  const s = new Simulation({ ...world(), blocks: [], spawns: [[3.5, 3.5], [5.5, 3.5]] }); s.t = 24;
  while (!s.done) s.step([2, 4]); assert.equal(s.capture, true); assert.ok(s.returns[1] > 1.9); const p = structuredClone(s.pos); s.step([4, 2]); assert.deepEqual(s.pos, p);
  const hidden = new Simulation(world()); for (let i = 0; i < 204; i++) hidden.step([0, 0]); assert.equal(hidden.done, true); assert.equal(hidden.capture, false); assert.equal(hidden.hidden, 180); assert.ok(hidden.returns[0] > 2);
});
test('validated edits preserve models while preventing overlap, trapping and spawn obstruction', () => {
  const a = generateArena(10, 14, 12, 0), before = structuredClone(a);
  const b = editBlock(a, -1, { x: 2, y: 2, width: 1, height: 1 }); assert.equal(b.blocks.length, 1); assert.deepEqual(a, before);
  assert.throws(() => editBlock(b, -1, { x: 2, y: 2, width: 1, height: 1 }), /overlap/);
  assert.throws(() => validateArena({ ...a, width: 1 })); assert.throws(() => validateArena({ ...a, spawns: [[0, 0], [4, 4]] }));
  assert.equal(editBlock(b, 0, null).blocks.length, 0);
});
test('model validation rejects malformed or nonfinite weights and seeded inference repeats', () => {
  const bundle = JSON.parse(fs.readFileSync(new URL('../public/models/hide-seek.json', import.meta.url))), model = validateModel(bundle.trained);
  const bad = structuredClone(model); bad.policies[0].layers[0].weights[0] = Infinity; assert.throws(() => validateModel(bad));
  assert.throws(() => validateModel({ format: 'hide-seek-ppo-v1', policies: [] }));
  const obs = new Simulation(generateArena()).observe(0), a = new Random(25), b = new Random(25);
  assert.deepEqual(Array.from({ length: 50 }, () => chooseAction(model.policies[0], obs, a)), Array.from({ length: 50 }, () => chooseAction(model.policies[0], obs, b)));
});
test('Python / JavaScript generator, trajectories, observations, rewards and logits agree', () => {
  const fixture = JSON.parse(fs.readFileSync(new URL('./fixtures/python-parity.json', import.meta.url)));
  for (const a of fixture.arenas) { const js = generateArena(a.seed, a.width, a.height, 6); assert.deepEqual(js.blocks, a.blocks); assert.deepEqual(js.spawns, a.spawns); }
  const s = new Simulation(fixture.arenas[1]);
  for (const state of fixture.states) { const rewards = s.step(state.actions); near(rewards, state.rewards); for (let r = 0; r < 2; r++) { near(s.pos[r], state.pos[r]); near(Array.from(s.observe(r)), state.obs[r]); } assert.equal(s.visible, state.visible); }
  const model = JSON.parse(fs.readFileSync(new URL('../public/models/hide-seek.json', import.meta.url))).trained;
  for (const x of fixture.logits) near(logits(model.policies[x.role], x.observation), x.logits, 2e-4);
});
test('frozen evaluation records disjoint held-out geometry and measured learning improvement', () => {
  const report = JSON.parse(fs.readFileSync(new URL('../evaluation.json', import.meta.url)));
  assert.equal(report.format, 'hide-seek-evaluation-v1'); assert.equal(report.protocol.episodesPerMatchup, 600); assert.equal(new Set(report.protocol.geometryHashes).size, 200); assert.equal(report.protocol.geometryDisjointFromTraining, true);
  assert.ok(report.protocol.mapSeedRange[0] > report.training.arenaSeedRange[1]);
  for (const role of ['hider', 'seeker']) assert.ok(report.improvement[role].pairedMapBootstrap95CI[0] > 0);
  assert.ok(report.results.trainedSeeker.captureRate > .9); assert.ok(report.results.trainedHider.hiderSurvivalRate > .7);
});
