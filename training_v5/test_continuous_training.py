"""Actual process-group Ctrl-C and resume test using synthetic lifecycle fixtures."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import torch
from persistent_actor import PersistentActor, FORMAT
from persistent_train import file_hash
from residual_critic import ResidualCentralCritic, SCHEMA


class ContinuousTrainingTests(unittest.TestCase):
    def test_group_interrupt_archives_complete_state_and_resume_adds_fresh_rollout(self):
        self.exercise_launcher([sys.executable, 'training_v5/train_continuous.py'])

    def test_node_launcher_waits_for_safe_group_interrupt_and_resumes(self):
        self.exercise_launcher(['node', 'scripts/train-continuous.mjs', '--trainer-version', 'v5'])

    def test_group_sigterm_saves_before_workers_exit(self):
        self.exercise_launcher([sys.executable, 'training_v5/train_continuous.py'], signal.SIGTERM)

    def test_node_launcher_handles_group_sigterm(self):
        self.exercise_launcher(['node', 'scripts/train-continuous.mjs', '--trainer-version', 'v5'], signal.SIGTERM)

    def exercise_launcher(self, launcher, stop_signal=signal.SIGINT):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            parent_path, critic_path = folder / 'parent.pt', folder / 'critic.pt'
            models = [PersistentActor(), PersistentActor()]
            optimizers = [torch.optim.Adam(model.parameters(), lr=.0001, eps=1e-5) for model in models]
            torch.save(dict(format=FORMAT, decisions=0,
                models=[model.state_dict() for model in models],
                optimizers=[optimizer.state_dict() for optimizer in optimizers],
                torchRNG=torch.get_rng_state(),
                provenance=dict(physicsSHA256=file_hash(root / 'training_v5/physics.py'))), parent_path)
            critic = ResidualCentralCritic(.1)
            optimizer = torch.optim.AdamW(critic.parameters(), lr=.0001)
            torch.save(dict(format=SCHEMA, parentSHA256=file_hash(parent_path), dropout=.1,
                critic=critic.state_dict(), criticOptimizer=optimizer.state_dict(),
                report=dict(environmentActorDecisions=dict(train=0))), critic_path)
            output = folder / 'run'
            command = [*launcher,
                '--parent', str(parent_path), '--critic', str(critic_path),
                '--initial', str(parent_path), '--history', str(parent_path),
                '--output', str(output), '--envs', '4', '--workers', '2',
                '--horizon', '64', '--sequence-length', '16', '--sequence-batch', '16',
                '--critic-batch-size', '64', '--epochs', '1', '--archive-every', '1', '--until-stop']
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
                env={**os.environ, 'PYTHON_BIN': sys.executable})
            try:
                deadline = time.monotonic() + 30
                while not (output / 'latest.pt').exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue((output / 'latest.pt').exists(), process.communicate()[0] if process.poll() is not None else 'First durable checkpoint was not created')
                # This reaches every worker as real terminal Ctrl-C does.
                os.killpg(process.pid, stop_signal)
                text, _ = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, text)
                self.assertIn('Stop requested.', text)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            stopped = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(stopped['arguments']['continuation']['stopSignal'], signal.Signals(stop_signal).name)
            self.assertEqual(stopped['totalPolicyInteractions'] % 512, 0)
            self.assertEqual(file_hash(parent_path), file_hash(output / 'best-reference.pt'))
            archive = output / 'checkpoints' / f'{stopped["totalPolicyInteractions"]}.pt'
            self.assertEqual(file_hash(archive), file_hash(output / 'latest.pt'))
            reference_hash, archive_hash = file_hash(parent_path), file_hash(archive)
            self.assertFalse(Path(stopped['arguments']['parent']).is_absolute())
            # Prove independence from the former workspace and original inputs.
            parent_path.unlink()
            critic_path.unlink()
            moved = folder / 'relocated-run'
            output.rename(moved)
            output = moved
            result = subprocess.run([*launcher, '--resume',
                str(output / 'latest.pt'), '--steps', '512', '--archive-every', '1'], cwd=root,
                capture_output=True, text=True, timeout=30, env={**os.environ, 'PYTHON_BIN': sys.executable})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            resumed = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(resumed['totalPolicyInteractions'], stopped['totalPolicyInteractions'] + 512)
            self.assertEqual(resumed['pilotUpdates'], stopped['pilotUpdates'] + 1)
            self.assertEqual(resumed['provenance']['resumes'][-1]['worldsRestarted'], 4)
            self.assertEqual(resumed['arguments']['continuation']['stopSignal'], None)
            self.assertFalse(torch.equal(stopped['torchRNG'], resumed['torchRNG']))
            for key, state in stopped['optimizers'][0]['state'].items():
                self.assertEqual(float(resumed['optimizers'][0]['state'][key]['step']), float(state['step']) + 1)
            self.assertEqual(archive_hash, file_hash(output / 'checkpoints' / archive.name))
            self.assertEqual(reference_hash, file_hash(output / 'best-reference.pt'))


if __name__ == '__main__':
    unittest.main()
