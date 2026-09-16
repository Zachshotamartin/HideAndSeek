import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';

const candidates = [
  process.env.PYTHON_BIN,
  '.venv-physics/bin/python',
  '/opt/homebrew/bin/python3',
  'python3',
].filter(Boolean);

const python = candidates.find((candidate) => {
  if (candidate.includes('/') && !existsSync(candidate)) return false;
  const result = spawnSync(
    candidate,
    ['-c', 'import mujoco,torch,numpy; print("physical-training-ready")'],
    { encoding: 'utf8' },
  );
  return result.status === 0 && result.stdout.includes('physical-training-ready');
});

if (!python) {
  console.error(
    'Install training/requirements.txt in a Python environment and set PYTHON_BIN to its executable.',
  );
  process.exit(1);
}

// The legacy launcher stays the default; versioned native checkpoints require
// an explicit trainer so that incompatible schemas never get mixed silently.
const args = process.argv.slice(2);
const versionIndex = args.indexOf('--trainer-version');
let directory = 'training';
if (versionIndex !== -1) {
  const version = args[versionIndex + 1];
  if (version !== 'v5' && version !== 'legacy') {
    console.error('--trainer-version must be v5 or legacy');
    process.exit(1);
  }
  directory = version === 'v5' ? 'training_v5' : 'training';
  args.splice(versionIndex, 2);
}

const child = spawn(
  python,
  [`${directory}/train_continuous.py`, ...args],
  { stdio: 'inherit' },
);

// Stay alive while Python completes the current update and saves. Terminal
// signals also reach Python directly; its stop request is idempotent.
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}
child.on('error', (error) => {
  console.error(error.message);
  process.exitCode = 1;
});
child.on('exit', (code, signal) => {
  if (signal) console.error(`Training ended with ${signal}.`);
  process.exitCode = code ?? 1;
});
