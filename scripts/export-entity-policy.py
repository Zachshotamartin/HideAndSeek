"""Export isolated typed actors and audit original native/JS encoder/controller parity."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
from entity_actor import EntityActor, FORMAT, load_pair, export_actor
from evaluate_policy_modes import deterministic
from persistent_evaluate import sample
from persistent_actor import augment
from persistent_train import file_hash
from physics import PhysicsEnv

torch.set_num_threads(1)


@torch.no_grad()
def fixtures(models):
    cases = []
    for mean in [True, False]:
        memory = [torch.zeros(1, 64), torch.zeros(1, 64)]
        buttons = np.zeros((2, 2), np.float32)
        env = PhysicsEnv(seed=1500273017, scenario='shelter', size=8, n_boxes=4, n_ramps=1)
        physical = env.observe()
        rows = []
        generator = np.random.default_rng(71391)
        for tick in range(264):
            if tick == 240:
                env.close()
                env = PhysicsEnv(seed=1500273018, scenario='rooms', size=10, n_boxes=5, n_ramps=2)
                physical = env.observe(); memory = [torch.zeros(1, 64), torch.zeros(1, 64)]; buttons.fill(0)
            actions = np.zeros((2, 5), np.float32)
            for role, model in enumerate(models):
                prior = buttons[role].copy()
                if physical[role, 7] < .5 and physical[role, 5] < 1:
                    prior.fill(0)
                prediction = model(torch.from_numpy(augment(physical[role], prior)[None]), memory[role])
                tape = [] if mean else generator.random(8).tolist()
                class Tape:
                    def random(self, count):
                        assert count == 8
                        return np.asarray(tape)
                actions[role], memory[role], buttons[role], commands = (
                    deterministic(model, physical[role], memory[role], buttons[role]) if mean else
                    sample(model, physical[role], memory[role], buttons[role], Tape()))
                rows.append(dict(role=role, reset=tick == 240, observation=physical[role].tolist(),
                    tape=tape, mean=prediction[0].mean[0].tolist(), toolLogits=model.tools(memory[role])[0].tolist(),
                    memory=memory[role][0].tolist(), commands=commands.tolist(), buttons=buttons[role].tolist(),
                    action=actions[role].tolist()))
            physical, _, _, _ = env.step(actions)
        env.close()
        cases.append(dict(deterministic=mean, rows=rows))
    sensor_cases = []
    generator = torch.Generator().manual_seed(7127)
    for role, model in enumerate(models):
        if not isinstance(model, EntityActor):
            continue
        row = torch.randn(140, generator=generator) * .2
        row[5], row[7], row[10], row[138], row[139] = 1, 1-role, 1, 1, 0
        objects = row[18:114].reshape(6, 16)
        objects[:, 0] = 1
        memory = torch.randn(1, 64, generator=generator) * .2
        for visible in [6, 3, 0]:
            test = row.clone(); test[18:114].reshape(6, 16)[visible:, 0] = 0
            for permutation in itertools.permutations(range(6)):
                changed = test.clone()
                changed[18:114] = test[18:114].reshape(6, 16)[list(permutation)].flatten()
                normal, _, _, next_memory = model(changed[None], memory)
                sensor_cases.append(dict(role=role, observation=changed.tolist(), previousMemory=memory[0].tolist(),
                    encoded=model.encoder(changed[None])[0].tolist(), mean=normal.mean[0].tolist(),
                    toolLogits=model.tools(next_memory)[0].tolist(), memory=next_memory[0].tolist()))
    return dict(rollouts=cases, sensorCases=sensor_cases,
                sensorCaseScope='Synthetic numeric permutation vectors, separate from actual physical episode rows')


def export(models, record, destination, label):
    model = dict(format=FORMAT, observationSize=140, physicsObservationSize=138,
        commands=['keep', 'press', 'release'], provenance=record['provenance'],
        auditLabel=label, actors=[export_actor(actor) for actor in models])
    destination.write_text(json.dumps(model, separators=(',', ':')) + '\n')
    fixture = destination.with_suffix('.fixture.json')
    fixture.write_text(json.dumps(fixtures(models), separators=(',', ':')) + '\n')
    result = subprocess.run(['node', 'scripts/test-entity-parity.mjs', str(destination), str(fixture)],
                            check=True, cwd=ROOT, text=True, capture_output=True)
    report = json.loads(result.stdout)
    report.update(exportSHA256=file_hash(destination), bytes=destination.stat().st_size,
                  auditLabel=label, sourceSHA256=file_hash(__file__))
    destination.with_suffix('.parity.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    models, record = load_pair(args.entity)
    records = [export(models, record, output / 'entity.json', 'Zero-update own projected actor')]
    probe = copy.deepcopy(models)
    with torch.random.fork_rng():
        torch.manual_seed(931772)
        for actor in probe:
            with torch.no_grad():
                actor.encoder.residual.weight.normal_(0, .04)
                actor.encoder.residual.bias.normal_(0, .03)
    records.append(export(probe, record, output / 'attention-probe.json',
                          'Numerical audit fixture with nonzero attention residual; not trained or selected'))
    legacy = load_pair(args.legacy)[0]
    records.append(export([legacy[0], probe[1]], record, output / 'mixed-probe.json',
                          'Mixed-encoder controller audit fixture; not selected'))
    result = dict(entityCheckpointSHA256=file_hash(args.entity), legacyCheckpointSHA256=file_hash(args.legacy),
                  reports=records, publicAssetsChanged=False)
    (output / 'parity-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['entity', 'legacy', 'output']:
        parser.add_argument('--' + name, required=True)
    main(parser.parse_args())
