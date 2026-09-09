import { spawnSync } from 'node:child_process';
const python = process.env.PYTHON_BIN || 'python3';
for (const args of [
  ['training/train.py', '--updates', '400', '--output', 'training/runs/final'],
  ['training/train.py', '--updates', '2000', '--output', 'training/runs/final', '--resume', 'training/runs/final/latest.pt'],
  ['training/evaluate.py', 'training/runs/final/latest.pt', '--output', 'training/validation.json']
]) { const result = spawnSync(python, args, { stdio: 'inherit' }); if (result.status !== 0 || result.error) { console.error(result.error || `Training command exited ${result.status}`); process.exit(result.status || 1); } }
console.log('Validation complete. Run report.py on a frozen checkpoint, then export.py and parity.py to prepare a release.');
