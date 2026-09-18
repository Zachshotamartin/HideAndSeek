"""Lifecycle/registry fixtures below are not learned-performance evidence."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import torch

from checkpoint_store import load_native
from entity_actor import prepare_pair
from evaluated_models import register
from persistent_actor import PersistentActor, FORMAT
from persistent_train import file_hash
from residual_critic import ResidualCentralCritic, SCHEMA
from train_entity import parser, train


ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(1)


def fixture(folder):
    folder.mkdir(parents=True)
    torch.manual_seed(761511)
    physics = file_hash(ROOT / 'training_v5/physics.py')
    actors = [PersistentActor(), PersistentActor()]
    initial = dict(format=FORMAT, models=[actor.state_dict() for actor in actors],
        decisions=0, torchRNG=torch.get_rng_state(),
        provenance=dict(physicsSHA256=physics, roleSources=[], sourceSelectedPairSHA256='synthetic-fixture'))
    torch.save(initial, folder / 'initial.pt')
    torch.save(prepare_pair(initial, projected=True), folder / 'entity.pt')
    torch.save(dict(centralCriticSchema=SCHEMA, centralCritic=ResidualCentralCritic(.1).state_dict(),
        provenance=dict(physicsSHA256=physics)), folder / 'critic.pt')
    args = parser().parse_args([
        '--parent', str(folder / 'entity.pt'), '--critic', str(folder / 'critic.pt'),
        '--initial', str(folder / 'initial.pt'), '--history', str(folder / 'initial.pt'),
        '--output', str(folder / 'original'), '--protocol', str(folder / 'protocol.json'),
        '--encoder', 'entity', '--envs', '2', '--workers', '1', '--horizon', '256',
        '--sequence-batch', '16', '--critic-batch-size', '256', '--epochs', '1',
        '--target-interactions', '1024', '--retain-updates', '1'])
    fields = ['seed', 'arm', 'envs', 'workers', 'horizon', 'sequence_length', 'sequence_batch',
              'critic_batch_size', 'epochs', 'learning_rate', 'critic_learning_rate',
              'entropy', 'kl_limit', 'target_interactions']
    protocol = dict(reward='Zero-sum visibility with capture credit; no tool bonuses.', training={name: getattr(args, name) for name in fields},
        physicsSHA256=physics, history=['initial'],
        assets={name: dict(path=str(folder / (name + '.pt')), sha256=file_hash(folder / (name + '.pt')))
                for name in ['entity', 'initial', 'critic']})
    (folder / 'protocol.json').write_text(json.dumps(protocol))
    with contextlib.redirect_stdout(io.StringIO()):
        train(args)
    return folder / 'original/latest.pt'


def command(*arguments, version="v5"):
    return ['node', str(ROOT / 'scripts/train-saved.mjs'), '--trainer-version', version, *map(str, arguments)]


def execute(arguments, timeout=90):
    result = subprocess.run(arguments, cwd=ROOT, env={**os.environ, 'PYTHON_BIN': sys.executable},
                            text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result


class SavedWorkflowTests(unittest.TestCase):
    def test_entity_pause_snapshots_relocated_resume_and_extension_preserve_original_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root / 'source')
            source_hash = file_hash(source)
            protocol_hash = file_hash(root / 'source/protocol.json')
            before = load_native(source)
            output = root / 'managed'
            process = subprocess.Popen(command('--resume', source, '--output', output,
                '--until-stop', '--snapshot-every', '1024'), cwd=ROOT, start_new_session=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env={**os.environ, 'PYTHON_BIN': sys.executable})
            try:
                deadline = time.monotonic() + 40
                while not (output / 'latest.pt').is_file() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue((output / 'latest.pt').is_file())
                duplicate = subprocess.run(command('--resume', output / 'latest.pt', '--steps', '1024'),
                    cwd=ROOT, env={**os.environ, 'PYTHON_BIN': sys.executable},
                    capture_output=True, text=True, timeout=30)
                self.assertNotEqual(duplicate.returncode, 0)
                self.assertIn('Another trainer', duplicate.stderr)
                os.killpg(process.pid, signal.SIGTERM)
                text, _ = process.communicate(timeout=40)
                self.assertEqual(process.returncode, 0, text)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            paused = load_native(output / 'latest.pt')
            record = json.loads((output / 'RUN.json').read_text())
            self.assertEqual(record['status'], 'paused')
            self.assertEqual(record['sessions'][-1]['stopSignal'], 'SIGTERM')
            self.assertGreater(paused['totalPolicyInteractions'], 1024)
            self.assertGreaterEqual(paused['seconds'], before['seconds'])
            self.assertGreater(record['sessions'][-1]['activeSessionWallSeconds'], 0)
            self.assertGreater(record['sessions'][-1]['rolloutSecondsThisSession'], 0)
            self.assertGreater(record['sessions'][-1]['optimizerSecondsThisSession'], 0)
            self.assertEqual(file_hash(source), source_hash)
            self.assertEqual(file_hash(root / 'source/protocol.json'), protocol_hash)
            snapshot = output / f'checkpoints/{paused["totalPolicyInteractions"]}.pt'
            snapshot_hash = file_hash(snapshot)
            self.assertEqual(snapshot_hash, file_hash(output / 'latest.pt'))
            # The administrative adaptation copied every numerical training field.
            session_id = record['sessions'][0]['id']
            adapted = load_native(output / f'resume-inputs/{session_id}-adapted.pt')
            for old, new in zip(before['models'], adapted['models']):
                for name in old:
                    torch.testing.assert_close(old[name], new[name], atol=0, rtol=0)
            torch.testing.assert_close(before['torchRNG'], adapted['torchRNG'], atol=0, rtol=0)
            self.assertEqual(before['worldRNG'], adapted['worldRNG'])
            for old, new in zip(before['optimizers'], adapted['optimizers']):
                for key, state in old['state'].items():
                    for name, value in state.items():
                        torch.testing.assert_close(value, new['state'][key][name], atol=0, rtol=0)
            moved = root / 'moved'
            output.rename(moved)
            shutil.rmtree(root / 'source')
            archived = moved / f'checkpoints/{paused["totalPolicyInteractions"]}.pt'
            execute(command('--resume', archived, '--steps', '2048', '--snapshot-every', '1024'))
            after = load_native(moved / 'latest.pt')
            self.assertEqual(after['totalPolicyInteractions'], paused['totalPolicyInteractions'] + 2048)
            self.assertEqual(file_hash(archived), snapshot_hash)
            self.assertEqual(len(json.loads((moved / 'RUN.json').read_text())['sessions']), 2)
            self.assertEqual(after['provenance']['resumes'][-1]['worldsRestarted'], 0)
            self.assertFalse(Path(after['arguments']['parent']).is_absolute())
            for old, new in zip(paused['optimizers'], after['optimizers']):
                for key, state in old['state'].items():
                    self.assertGreater(float(new['state'][key]['step']), float(state['step']))
            rejected = subprocess.run(command('--resume', archived, '--steps', '1024'), cwd=ROOT,
                env={**os.environ, 'PYTHON_BIN': sys.executable}, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('new --output branch', rejected.stderr)

    def test_actual_portable_37m_bundle_supports_the_new_saved_workflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / 'bundle'
            shutil.copytree(ROOT / 'training/resume-bundle', bundle)
            output = root / 'run'
            execute(command('--resume', bundle / 'resume.pt', '--output', output,
                            '--steps', '65536', '--snapshot-every', '65536', version='legacy'))
            saved = load_native(output / 'latest.pt')
            self.assertEqual(saved['totalPolicyInteractions'], 65536)
            self.assertEqual(saved['parentDecisions'], 37036032)
            self.assertTrue((output / 'checkpoints/65536.pt').is_file())
            self.assertEqual(json.loads((output / 'RUN.json').read_text())['status'], 'completed-requested-extension')

    def test_explicit_evaluation_registry_keeps_best_pair_and_rejects_training_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = fixture(root / 'source')
            cohort = root / 'maps.json'
            cohort.write_text(json.dumps(dict(maps=[dict(seed=1600200001, scenario='open',
                arenaConfig=dict(size=6, n_boxes=1, n_ramps=0))])))
            registry = root / 'registry'
            output = root / 'evaluation'
            execute([sys.executable, str(ROOT / 'training_v5/evaluate_saved.py'),
                '--checkpoint', str(reference), '--reference', str(reference), '--cohort', str(cohort),
                '--output', str(output), '--workers', '1', '--registry', str(registry)])
            actual = json.loads((output / 'evaluation.json').read_text())
            first = json.loads((registry / 'best-evaluated.json').read_text())
            self.assertEqual(first['sha256'], file_hash(reference))
            self.assertEqual(first['score'], .5)
            # Deliberately synthetic numerical registry fixtures exercise ranking;
            # these are never reported as measured learning outcomes.
            original = load_native(reference)
            for index, (utility, upper_loss) in enumerate([(0.7, 0.1), (0.65, 0.1), (0.8, -0.1)]):
                candidate = copy.deepcopy(original)
                candidate['models'][0]['movement.bias'][0] += (index + 1) * .001
                path = root / f'candidate-{index}.pt'
                torch.save(candidate, path)
                report = copy.deepcopy(actual)
                report['checkpointSHA256'] = file_hash(path)
                report['summary']['candidate-hider']['hiddenFraction'] = utility
                report['summary']['candidate-seeker']['hiddenFraction'] = 1 - utility
                report['contrasts']['Hider change']['bootstrap95Percent'] = [-.2, upper_loss]
                report['contrasts']['Seeker change']['bootstrap95Percent'] = [.02, .2]
                for block in report['byPlayLength'].values():
                    block['summary'] = copy.deepcopy(report['summary'])
                    block['contrasts'] = copy.deepcopy(report['contrasts'])
                report['syntheticRegistryTestOnly'] = True
                report_path = root / f'report-{index}.json'
                report_path.write_text(json.dumps(report))
                result = register(path, reference, report_path, registry)
                if index == 0:
                    best_hash = file_hash(path)
                self.assertEqual(result['bestSHA256'], best_hash)
            selected = json.loads((registry / 'best-evaluated.json').read_text())
            self.assertEqual(file_hash(registry / selected['file']), best_hash)
            self.assertEqual(file_hash(registry / 'best-evaluated.pt'), best_hash)
            self.assertTrue(selected['entirePairPreserved'])
            self.assertEqual(selected['sourceProvenance'], original['provenance'])
            log_path = root / 'training-log.json'
            log_path.write_text(json.dumps(dict(meanHiddenFraction=1)))
            with self.assertRaisesRegex(ValueError, 'not a training log'):
                register(reference, reference, log_path, registry)
            changed = copy.deepcopy(actual)
            changed['maps'][0]['seed'] += 1
            log_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'different maps'):
                register(reference, reference, log_path, registry)
            # A best native checkpoint remains resumable using registry assets.
            shutil.rmtree(root / 'source')
            execute(command('--resume', registry / 'best-evaluated.pt', '--output', root / 'from-best',
                            '--steps', '1024', '--snapshot-every', '1024'))
            self.assertEqual(load_native(root / 'from-best/latest.pt')['totalPolicyInteractions'], 2048)


if __name__ == '__main__':
    unittest.main()
