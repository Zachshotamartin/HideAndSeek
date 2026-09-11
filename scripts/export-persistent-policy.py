"""Export the isolated original pilot and test its stateful JS controller.

Stochastic parity uses a shared uniform tape and the same Box–Muller transform;
Torch and JavaScript PRNG seeds are not assumed to identify the same stream.
Physics observations come from real native episodes, including preparation and
terminal resets. Neither production assets nor the active manifest are changed.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from persistent_actor import PersistentActor, advance_buttons, augment, export_actor, FORMAT, COMMANDS
from physics import PhysicsEnv

torch.set_num_threads(1)


def parity(models, destination):
    rng = np.random.default_rng(971330)
    cases = []
    for deterministic in [True, False]:
        env = PhysicsEnv(seed=1500030017, scenario='shelter', size=8, n_boxes=4, n_ramps=1)
        physical = env.observe()
        memory = [torch.zeros(1, 64), torch.zeros(1, 64)]
        buttons = np.zeros((2, 2), np.float32)
        rows = []
        with torch.no_grad():
            for step in range(264):
                reset = step == 240
                if reset:
                    env.close()
                    env = PhysicsEnv(seed=1500030018, scenario='rooms', size=10, n_boxes=5, n_ramps=2)
                    physical = env.observe()
                    memory = [torch.zeros(1, 64), torch.zeros(1, 64)]
                    buttons.fill(0)
                actions = np.zeros((2, 5), np.float32)
                for role, model in enumerate(models):
                    tape = [] if deterministic else rng.random(8).tolist()
                    observation = augment(physical[role], buttons[role])
                    normal, tools, _, memory[role] = model(torch.from_numpy(observation[None]), memory[role])
                    mean = normal.mean[0].numpy()
                    logits = model.tools(memory[role])[0].numpy()
                    movement = mean.copy()
                    commands = np.zeros(2, np.int64)
                    for axis in range(3):
                        if not deterministic:
                            noise = math.sqrt(-2 * math.log(max(1e-12, tape[axis * 2]))) * math.cos(2 * math.pi * tape[axis * 2 + 1])
                            movement[axis] = mean[axis] + float(normal.scale[0, axis]) * noise
                        actions[role, axis] = math.tanh(float(movement[axis]))
                    probabilities = tools.probs[0].numpy()
                    for tool in range(2):
                        commands[tool] = np.argmax(logits[tool * 3:tool * 3 + 3]) if deterministic else \
                            min(2, np.searchsorted(np.cumsum(probabilities[tool]), tape[6 + tool], side='right'))
                    blind = role == 1 and physical[role, 5] < 1
                    buttons[role] = advance_buttons(buttons[role], commands, blind)
                    actions[role, 3:] = buttons[role]
                    if blind:
                        actions[role] = 0
                    rows.append(dict(role=role, reset=reset, observation=physical[role].tolist(), tape=tape,
                        mean=mean.tolist(), toolLogits=logits.tolist(), memory=memory[role][0].numpy().tolist(),
                        commands=commands.tolist(), buttons=buttons[role].tolist(), action=actions[role].tolist()))
                physical, _, done, _ = env.step(actions)
        env.close()
        cases.append(dict(deterministic=deterministic, rows=rows))
    fixture = destination.with_suffix('.fixture.json')
    fixture.write_text(json.dumps(cases, separators=(',', ':')) + '\n')
    run = subprocess.run(['node', str(ROOT / 'scripts/test-persistent-parity.mjs'), str(destination), str(fixture)],
                         cwd=ROOT, check=True, text=True, capture_output=True)
    return json.loads(run.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.checkpoint)
    saved = torch.load(source, map_location='cpu', weights_only=False)
    if saved['format'] != FORMAT:
        raise ValueError('Only an isolated persistent-button checkpoint can be exported')
    models = [PersistentActor().eval() for _ in range(2)]
    for model, weights in zip(models, saved['models']):
        model.load_state_dict(weights)
    result = dict(format=FORMAT, commands=list(COMMANDS), observationSize=140, physicsObservationSize=138,
        provenance=saved['provenance'], training={key:saved.get(key, 0) for key in
        ['parentDecisions', 'pilotDecisions', 'decisions', 'pilotUpdates', 'pilotEpisodes', 'seconds']},
        actors=[export_actor(model) for model in models])
    # Value-only experience is not silently presented as actor optimization.
    # These keys are absent from earlier persistent checkpoints, preserving
    # their existing portable payloads exactly.
    result['training'].update({key: saved[key] for key in
        ['criticWarmupDecisions', 'actorUpdateDecisions', 'trainingMethod'] if key in saved})
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    report = parity(models, destination)
    report.update(checkpointSHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
        exportSHA256=hashlib.sha256(destination.read_bytes()).hexdigest(), bytes=destination.stat().st_size,
        training=result['training'])
    destination.with_suffix('.parity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
