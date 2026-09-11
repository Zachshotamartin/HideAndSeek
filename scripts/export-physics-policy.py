"""Export our trained actors; verify JavaScript recurrence against PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from actor import PhysicalActor, export_actor


def export_checkpoint(checkpoint_path, destination, prefix='physical-policy'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    models = []
    for state in checkpoint['models']:
        model = PhysicalActor(checkpoint['observationSize'])
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    result = {
        'format': 'original-mujoco-recurrent-ppo-v1',
        'provenance': 'Original policies trained from random initialization in this repository. No external checkpoints or expert demonstrations.',
        'training': {key: checkpoint.get(key, 0) for key in ('seed', 'decisions', 'episodes', 'updates', 'seconds')},
        'actors': [export_actor(model) for model in models],
    }
    output = ROOT / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, separators=(',', ':')) + '\n').encode()
    if destination == 'public/models/physical-policy.json':
        digest = hashlib.sha256(encoded).hexdigest()
        output = output.with_name(f'{prefix}-{digest[:12]}.json')
    output.write_bytes(encoded)
    # Multiple recurrent steps test gate ordering and state retention, not only zero-state output.
    rng = np.random.default_rng(74263)
    observations = rng.uniform(-1, 1, (2, 40, checkpoint['observationSize'])).astype(np.float32)
    expected = []
    for role, model in enumerate(models):
        memory = torch.zeros(1, model.hidden_size)
        sequence = []
        with torch.no_grad():
            for observation in observations[role]:
                action, _, _, _, memory = model.act(torch.from_numpy(observation[None]), memory, deterministic=True)
                sequence.append({'action': action[0].numpy().tolist(), 'memory': memory[0].numpy().tolist()})
        expected.append(sequence)
    script = """
import {readFileSync} from 'node:fs';
import {createPhysicalPolicies} from './src/core/learnedPolicy.js';
const input=JSON.parse(readFileSync(0,'utf8'));
const policies=createPhysicalPolicies(JSON.parse(readFileSync(input.path,'utf8')));
const output=policies.map((policy,role)=>{let memory=policy.initialMemory();return input.observations[role].map(obs=>{const result=policy.act(obs,memory);memory=result.memory;return {action:[...result.action],memory:[...memory]};});});
process.stdout.write(JSON.stringify(output));
"""
    run = subprocess.run(['node', '--input-type=module', '-e', script], cwd=ROOT, check=True,
        input=json.dumps({'path': str(output), 'observations': observations.tolist()}), text=True, capture_output=True)
    actual = json.loads(run.stdout)
    error = 0
    for role in range(2):
        for step in range(40):
            for key in ('action', 'memory'):
                error = max(error, float(np.max(np.abs(np.asarray(actual[role][step][key]) - expected[role][step][key]))))
    if error > 1e-5:
        raise AssertionError(f'JS/Torch recurrent inference differs by {error}')
    report = {'checkpointSHA256': hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        'exportSHA256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'bytes': output.stat().st_size, 'recurrentStepsPerActor': 40,
        'maxInferenceError': error, 'tolerance': 1e-5}
    output.with_suffix('.parity.json').write_text(json.dumps(report, indent=2) + '\n')
    return result, output, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--output', default='public/models/physical-policy.json')
    p.add_argument('--initial-checkpoint', help='Original random initialization saved by this training run')
    args = p.parse_args()
    result, output, report = export_checkpoint(args.checkpoint, args.output)
    if args.output == 'public/models/physical-policy.json':
        manifest = {'format': result['format'], 'file': output.name, 'bytes': report['bytes'],
                    'sha256': report['exportSHA256'], 'provenance': result['provenance'],
                    'training': result['training']}
        manifest['checkpoints'] = [{'id': 'trained', 'label': 'Trained policies',
            'file': output.name, 'bytes': report['bytes'], 'sha256': report['exportSHA256'],
            'training': result['training']}]
        retained = {output, output.with_suffix('.parity.json')}
        if args.initial_checkpoint:
            initial, initial_path, initial_report = export_checkpoint(
                args.initial_checkpoint, args.output, prefix='physical-policy-initial')
            if initial['training']['decisions'] != 0:
                raise ValueError('Initial comparison must contain zero training decisions')
            manifest['checkpoints'].append({'id': 'initial', 'label': 'Before training',
                'file': initial_path.name, 'bytes': initial_report['bytes'],
                'sha256': initial_report['exportSHA256'], 'training': initial['training']})
            retained.update((initial_path, initial_path.with_suffix('.parity.json')))
        (output.parent / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (ROOT / 'src/core/policyAsset.js').write_text(
            '// Generated from our own training checkpoint. See models/MANIFEST.json.\n'
            f'export const PHYSICAL_POLICY_FILE = "models/{output.name}";\n')
        for obsolete in output.parent.glob('physical-policy*.json'):
            if obsolete not in retained:
                obsolete.unlink()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
