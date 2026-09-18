// Export to an ignored directory first; publish only actor JSON and compact evidence.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, readdir, unlink } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, basename } from 'node:path';
import { createPolicyControllers, PERSISTENT_FORMAT } from '../src/core/policyController.js';

const [trainedPath, initialPath, evaluationPath] = process.argv.slice(2);
assert(trainedPath && initialPath && evaluationPath,
  'Usage: node scripts/package-persistent-assets.mjs TRAINED.json INITIAL.json EVALUATION.json');
const root = new URL('../', import.meta.url);
const target = new URL('public/models/', root);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const staged = [];
const checkpoints = [];
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
  assert.equal(model.training.decisions, model.training.parentDecisions + model.training.pilotDecisions);
  if (id === 'initial') assert.equal(model.training.decisions, 0);
  const file = `physical-policy-${id === 'initial' ? 'initial-' : ''}${hash.slice(0, 12)}.json`;
  const parityFile = file.replace('.json', '.parity.json');
  staged.push([file, bytes], [parityFile, parityBytes]);
  checkpoints.push({ id, label: id === 'initial' ? 'Before game experience' : 'Development pair · 37.0M decisions',
    status: id === 'initial' ? 'UNTRAINED' : 'DEVELOPMENT', file, bytes: bytes.length, sha256: hash,
    checkpointSHA256: parity.checkpointSHA256, parityFile, training: model.training });
}

const evaluationBytes = await readFile(resolve(evaluationPath));
const evaluation = JSON.parse(evaluationBytes);
assert.equal(evaluation.checkpointSHA256, checkpoints[0].checkpointSHA256);
const comparisons = [
  ['Pilot hider vs frozen trained binary seeker', 'pilot-hider-v-binary', 'binary-pair'],
  ['Pilot seeker vs frozen trained binary hider', 'binary-pair', 'pilot-seeker-v-binary'],
  ['Hider tools benefit', 'learned', 'no-hider-tools'],
  ['Seeker tools benefit', 'no-seeker-tools', 'learned'],
];
const byMap = new Map();
for (const episode of evaluation.episodes) {
  const key = `${episode.scenario}:${episode.seed}`;
  if (!byMap.has(key)) byMap.set(key, { seed: episode.seed, scenario: episode.scenario, hiddenFractions: {} });
  assert(Number.isFinite(episode.hiddenFraction) && episode.hiddenFraction >= 0 && episode.hiddenFraction <= 1);
  assert.equal(episode.hiddenFraction, episode.info.hidden / episode.info.play_steps);
  if (comparisons.some(([, left, right]) => episode.mode === left || episode.mode === right)) {
    const values = byMap.get(key).hiddenFractions;
    assert.equal(values[episode.mode], undefined, 'Each paired mode must occur once per map');
    values[episode.mode] = episode.hiddenFraction;
  }
}
assert.equal(byMap.size, 36);
for (const [label, left, right] of comparisons) {
  const differences = [...byMap.values()].map(row => row.hiddenFractions[left] - row.hiddenFractions[right]);
  assert(differences.every(Number.isFinite));
  const mean = differences.reduce((a, b) => a + b, 0) / differences.length;
  assert(Math.abs(mean - evaluation.contrasts[label].mean) < 1e-12, `${label} must match saved episodes`);
}
const evidence = {
  status: 'DEVELOPMENT', checkpointSHA256: evaluation.checkpointSHA256,
  sourceEvaluationSHA256: sha(evaluationBytes), source: basename(evaluationPath),
  maps: 36, episodes: evaluation.episodes.length, seedStart: evaluation.seedStart,
  seedUsage: evaluation.seedUsage, sampling: evaluation.sampling,
  scope: 'Eight-metre maps with three boxes and one ramp across shelter, rooms and open layouts. Reused validation, not untouched final testing.',
  qualification: 'General role strength improved against a fixed trained binary opponent. Reliable useful grab/lock strategies are not established; paired tool effects include zero.',
  frozenBinaryBaseline: evaluation.additionalFrozenTrainedBaseline,
  effectUnits: 'Difference in role objective fraction; multiply by 100 for percentage points. Map-paired bootstrap 95% intervals copied from the source evaluation.',
  contrasts: Object.fromEntries(comparisons.map(([label]) => [label, evaluation.contrasts[label]])),
  pairedMaps: [...byMap.values()],
};
const evidenceBytes = Buffer.from(JSON.stringify(evidence, null, 2) + '\n');
const evidenceHash = sha(evidenceBytes);
const evidenceFile = `development-evaluation-${evidenceHash.slice(0, 12)}.json`;
staged.push([evidenceFile, evidenceBytes]);
const { id, label, checkpointSHA256, parityFile, ...primary } = checkpoints[0];
const manifest = {
  format: PERSISTENT_FORMAT, observationSize: 140, physicsObservationSize: 138,
  commands: ['keep', 'press', 'release'], ...primary,
  provenance: 'Original actor policies, without external checkpoints or demonstrations. The persistent pilot continued an original 22,937,600-decision binary backbone. Its zero-experience reference is separately exported from the saved random backbone; it is not the pilot warm start.',
  qualification: evidence.qualification, checkpoints,
  evaluation: { file: evidenceFile, bytes: evidenceBytes.length, sha256: evidenceHash, status: 'DEVELOPMENT', maps: 36 },
};
staged.push(['MANIFEST.json', JSON.stringify(manifest, null, 2) + '\n']);
// All validation precedes writes. Remove only superseded generated public files.
await mkdir(target, { recursive: true });
for (const [file, bytes] of staged) await writeFile(new URL(file, target), bytes);
const names = new Set(staged.map(([file]) => file));
for (const file of await readdir(target)) {
  if (/^(physical-policy-(initial-)?[a-f0-9]{12}(\.parity)?|development-evaluation-[a-f0-9]{12})\.json$/.test(file) && !names.has(file))
    await unlink(new URL(file, target));
}
await writeFile(new URL('src/core/policyAsset.js', root),
  `export const PHYSICAL_POLICY_FILE = ${JSON.stringify(`models/${checkpoints[0].file}`)};\n` +
  `export const PHYSICAL_POLICY_DETAILS = ${JSON.stringify({
    training: `Original persistent pilot: ${primary.training.decisions.toLocaleString('en-US')} total lineage interactions. Visibility is the only game reward.`,
    evaluation: 'General play improved against fixed opponents on 36 reused validation maps. Reliable useful grab/lock strategies remain unproven; this is development evidence, not untouched final testing.',
    modes: 'Seeded sampling remains the default; deterministic actions are an explicit comparison.',
  }, null, 2)};\n`);
console.log(JSON.stringify(manifest, null, 2));
