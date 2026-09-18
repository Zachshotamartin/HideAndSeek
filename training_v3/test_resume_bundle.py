"""A fresh-checkout smoke uses the actual pretrained portable native bundle."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch
from persistent_train import file_hash


class ResumeBundleTests(unittest.TestCase):
    def test_actual_bundle_trains_without_original_workspace_assets(self):
        root = Path(__file__).resolve().parent.parent
        bundle = root / 'training/resume-bundle'
        manifest = json.loads((bundle / 'MANIFEST.json').read_text())
        for name, record in manifest['files'].items():
            self.assertEqual(file_hash(bundle / name), record['sha256'])
        with tempfile.TemporaryDirectory(prefix='hide-seek-fresh-clone-') as temporary:
            checkout = Path(temporary)
            (checkout / 'training').mkdir()
            (checkout / 'scripts').mkdir()
            for source in (root / 'training').glob('*.py'):
                shutil.copyfile(source, checkout / 'training' / source.name)
            shutil.copytree(bundle, checkout / 'training/resume-bundle')
            shutil.copyfile(root / 'scripts/train-continuous.mjs', checkout / 'scripts/train-continuous.mjs')
            shutil.copyfile(root / 'package.json', checkout / 'package.json')
            result = subprocess.run(['npm', 'run', 'train:continuous', '--', '--resume',
                'training/resume-bundle/resume.pt', '--output', 'output/fresh-run', '--steps', '65536'],
                cwd=checkout, env={**os.environ, 'PYTHON_BIN': sys.executable},
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            destination = checkout / 'output/fresh-run'
            saved = torch.load(destination / 'latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(saved['totalPolicyInteractions'], 65536)
            self.assertEqual(saved['parentDecisions'], 37036032)
            self.assertEqual(saved['pilotUpdates'], 1)
            for name in ['parent', 'critic', 'initial']:
                relative = Path(saved['arguments'][name])
                self.assertFalse(relative.is_absolute())
                self.assertTrue((destination / relative).is_file())
            self.assertEqual(file_hash(destination / 'best-reference.pt'), manifest['files']['assets/parent.pt']['sha256'])
            self.assertTrue((destination / 'checkpoints/65536.pt').exists())


if __name__ == '__main__':
    unittest.main()
