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

const child = spawn(
  python,
  ['training/train_saved.py', ...process.argv.slice(2)],
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
