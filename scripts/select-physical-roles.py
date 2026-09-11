"""Freeze an explicitly selected pair of our own role weights without training.

Per-role checkpoint identities and counters remain authoritative. This creates
an unpromoted native candidate, not browser assets or a full in-flight resume.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
from persistent_actor import FORMAT
from persistent_train import file_hash


def source_ancestors(record):
    provenance = record.get('provenance', {})
    result = {provenance.get('parentSHA256')}
    for field in ['roleSources', 'inheritedRoleSources']:
        for role in provenance.get(field, []):
            result.add(role.get('sourceCheckpointSHA256'))
    return result


def select(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError('Keep each selected candidate immutable')
    paths = [Path(args.hider), Path(args.seeker)]
    records = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
    hashes = [file_hash(path) for path in paths]
    physics_hash = file_hash(ROOT / 'training/physics.py')
    if any(record['format'] != FORMAT or record['provenance']['physicsSHA256'] != physics_hash for record in records):
        raise ValueError('Selected roles must share the same unchanged physical game and controller')
    related = hashes[0] == hashes[1] or hashes[0] in source_ancestors(records[1]) or hashes[1] in source_ancestors(records[0])
    same_phase = (paths[0].resolve().parent == paths[1].resolve().parent
                  and records[0].get('provenance', {}).get('parentSHA256') == records[1].get('provenance', {}).get('parentSHA256')
                  and records[0].get('arguments', {}).get('seed') == records[1].get('arguments', {}).get('seed'))
    if not (related or same_phase):
        raise ValueError('Cannot infer a union budget for unrelated training branches; provide a separately reviewed lineage record')
    union_budget = max(record['decisions'] for record in records)
    lineage = []
    for role, source, digest, path in zip(['hider', 'seeker'], records, hashes, paths):
        lineage.append(dict(role=role, sourceCheckpointSHA256=digest, sourceFile=path.name,
            sourceRunCounters={key: source.get(key) for key in ['decisions', 'parentDecisions', 'pilotDecisions',
                'totalPolicyInteractions', 'criticWarmupDecisions', 'currentPolicyDecisions', 'historicalPolicyDecisions', 'activePolicySamples']},
            inheritedRoleSources=source['provenance'].get('inheritedRoleSources', source['provenance'].get('roleSources', []))))
    selected = dict(format=FORMAT, observationSize=140, physicsObservationSize=138,
        models=[copy.deepcopy(record['models'][role]) for role, record in enumerate(records)],
        optimizers=[copy.deepcopy(record['optimizers'][role]) for role, record in enumerate(records)],
        torchRNG=records[1]['torchRNG'].clone(),
        decisions=union_budget, parentDecisions=union_budget, pilotDecisions=0,
        pilotUpdates=0, pilotEpisodes=0, seconds=0, newActorUpdates=0,
        trainingMethod='Frozen role selection only; no new optimization or automatic promotion',
        decisionsDefinition='Union source-run resource budget with shared ancestry counted once. It includes critic-only and frozen-opponent interactions; the per-role source counters describe actual training lineage.',
        provenance=dict(physicsSHA256=physics_hash, roleSources=lineage,
            operation='Exact selected role weights and optimizer states; no averaging, head changes, or policy update',
            torchRNGSourceSHA256=hashes[1], selectionReason=args.reason),
        status='Unpromoted selected candidate; requires its own measured pair and mode assessment')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected, output)
    for role, source in enumerate(records):
        for name, value in source['models'][role].items():
            torch.testing.assert_close(value, selected['models'][role][name], atol=0, rtol=0)
    report = dict(candidateSHA256=file_hash(output), candidateFile=output.name, roleSources=lineage,
                  unionSourceBudget=union_budget, newActorUpdates=0, sourceSHA256=file_hash(__file__), reason=args.reason)
    output.with_suffix('.selection.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['hider', 'seeker', 'output', 'reason']:
        parser.add_argument('--' + name, required=True)
    select(parser.parse_args())
