// Validate the frozen role selection and physical evidence before public writes.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, readdir, unlink } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { createPolicyControllers, PERSISTENT_FORMAT } from '../src/core/policyController.js';

const [trainedPath, initialPath, evidencePath] = process.argv.slice(2);
assert(trainedPath && initialPath && evidencePath,
  'Usage: node scripts/package-selected-assets.mjs TRAINED.json INITIAL.json EVIDENCE.json');
const root = new URL('../', import.meta.url);
const target = new URL('public/models/', root);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const staged = [], checkpoints = [], models = [];
for (const [id, path] of [['trained', trainedPath], ['initial', initialPath]]) {
  const bytes = await readFile(resolve(path));
  const model = JSON.parse(bytes);
  assert.equal(model.format, PERSISTENT_FORMAT);
  createPolicyControllers(model);
  const parityBytes = await readFile(resolve(path.replace(/\.json$/, '.parity.json')));
  const parity = JSON.parse(parityBytes);
  const hash = sha(bytes);
  assert.equal(hash, parity.exportSHA256);
  assert.equal(bytes.length, parity.bytes);
  assert.equal(parity.rows, 1056);
  assert.equal(parity.blindRows, 208);
  assert.equal(parity.resets, 4);
  assert(parity.maxInferenceError <= 1e-5 && parity.deterministicAndSeededSampling);
  if (id === 'initial') assert.equal(model.training.decisions, 0);
  const file = `physical-policy-${id === 'initial' ? 'initial-' : ''}${hash.slice(0, 12)}.json`;
  const parityFile = file.replace('.json', '.parity.json');
  staged.push([file, bytes], [parityFile, parityBytes]);
  checkpoints.push({ id, label: id === 'initial' ? 'Before game experience' : 'Selected development pair',
    status: id === 'initial' ? 'UNTRAINED' : 'DEVELOPMENT', file, bytes: bytes.length, sha256: hash,
    checkpointSHA256: parity.checkpointSHA256, parityFile, training: model.training });
  models.push(model);
}
const evidenceBytes = await readFile(resolve(evidencePath));
const evidence = JSON.parse(evidenceBytes);
assert.equal(evidence.format, 'selected-role-development-evidence-v1');
assert.equal(evidence.status, 'DEVELOPMENT');
assert.equal(evidence.checkpointSHA256, checkpoints[0].checkpointSHA256);
assert.deepEqual(evidence.roleSources, models[0].provenance.roleSources);
assert.equal(evidence.maps, 96);
assert.equal(evidence.pairedMaps.length, 96);
assert.equal(new Set(evidence.pairedMaps.map(row => `${row.scenario}:${row.seed}`)).size, 96);
for (const [label, effect] of Object.entries(evidence.contrasts)) {
  const left = evidence.conditionOrder.indexOf(effect.leftCondition);
  const right = evidence.conditionOrder.indexOf(effect.rightCondition);
  assert(left >= 0 && right >= 0);
  let total = 0;
  for (const row of evidence.pairedMaps) {
    assert.equal(row.hiddenFractions.length, evidence.conditionOrder.length);
    assert(row.hiddenFractions.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
    total += row.hiddenFractions[left] - row.hiddenFractions[right];
  }
  assert(Math.abs(total / 96 - effect.mean) < 1e-12, `${label} must match saved physical counts`);
  assert(effect.bootstrap95Percent.length === 2 && effect.bootstrap95Percent.every(Number.isFinite));
}
const evidenceHash = sha(evidenceBytes);
const evidenceFile = `development-evaluation-${evidenceHash.slice(0, 12)}.json`;
staged.push([evidenceFile, evidenceBytes]);
const number = value => Number(value).toLocaleString('en-US');
const effect = label => {
  const result = evidence.contrasts[label];
  const pp = value => `${value >= 0 ? '+' : '−'}${Math.abs(value * 100).toFixed(2)}`;
  return `${pp(result.mean)} percentage points (95% interval ${result.bootstrap95Percent.map(pp).join(' to ')})`;
};
const hider = evidence.roleSources[0].sourceRunCounters;
const seeker = evidence.roleSources[1].sourceRunCounters;
const details = {
  training: `Hider retained from the ${number(hider.decisions)}-interaction reference. Seeker selected after ${number(seeker.totalPolicyInteractions)} fresh league interactions, including ${number(seeker.currentPolicyDecisions[1])} current-seeker decisions and ${number(seeker.activePolicySamples[1])} active play samples. Shared ancestry is counted once; selection adds no updates. Visibility remains the only game reward.`,
  evaluation: `On 96 maps reused for selection, the selected seeker missed ${evidence.modes.candidate_pair_sample.completeSearchMisses} complete searches versus ${evidence.modes.browser_pair_sample.completeSearchMisses} for the original seeker with the same hider. Visibility change: ${effect('Selected seeker vs original browser seeker')}. Sampled grab/lock effects: hider ${effect('Sampled hider grab/lock benefit')}; seeker ${effect('Sampled seeker grab/lock benefit')}. Reliable useful tool strategies remain unproven. These are development comparisons, not untouched final testing.`,
  modes: `Mean actions versus sampling, with the same sampled browser hider: ${effect('Mean seeker vs sampled, same sampled browser hider')}; ${evidence.modes.fixed_browser_hider_candidate_seeker_mean.completeSearchMisses} versus ${evidence.modes.fixed_browser_hider_candidate_seeker_sample.completeSearchMisses} complete misses. Longer holds did not establish useful tool behavior. Seeded sampling remains the default.`,
};
const { id, label, checkpointSHA256, parityFile, ...primary } = checkpoints[0];
const manifest = {
  format: PERSISTENT_FORMAT, observationSize: 140, physicsObservationSize: 138,
  commands: ['keep', 'press', 'release'], ...primary,
  provenance: 'Original frozen role selection. Hider and seeker retain separate checkpoint hashes and actual source counters. No borrowed weights, demonstrations, pursuit bonuses, or tool rewards.',
  roleSources: evidence.roleSources, qualification: evidence.qualification, checkpoints,
  evaluation: { file: evidenceFile, bytes: evidenceBytes.length, sha256: evidenceHash, status: 'DEVELOPMENT', maps: 96 },
};
staged.push(['MANIFEST.json', JSON.stringify(manifest, null, 2) + '\n']);
await mkdir(target, { recursive: true });
for (const [file, bytes] of staged) await writeFile(new URL(file, target), bytes);
const names = new Set(staged.map(([file]) => file));
for (const file of await readdir(target)) {
  if (/^(physical-policy-(initial-)?[a-f0-9]{12}(\.parity)?|development-evaluation-[a-f0-9]{12})\.json$/.test(file) && !names.has(file))
    await unlink(new URL(file, target));
}
await writeFile(new URL('src/core/policyAsset.js', root),
  `export const PHYSICAL_POLICY_FILE = ${JSON.stringify(`models/${checkpoints[0].file}`)};\n` +
  `export const PHYSICAL_POLICY_DETAILS = ${JSON.stringify(details, null, 2)};\n`);
console.log(JSON.stringify(manifest, null, 2));
