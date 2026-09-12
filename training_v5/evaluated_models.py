"""Retain evaluated native pairs without ever selecting from training return."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from checkpoint_store import atomic_json, copy_immutable, load_native, replace_from_immutable, stage_dependencies, utc_now
from persistent_train import file_hash


REPORT_FORMAT = 'hide-seek-saved-pair-fixed-opponent-evaluation-v1'


def evaluation_contract(report):
    if report.get('format') != REPORT_FORMAT:
        raise ValueError('Use an explicit fixed-opponent evaluation, not a training log')
    body = {name: report[name] for name in ['maps', 'referenceSHA256', 'sampling', 'sources']}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def interval_of(contrasts, role):
    interval = contrasts[role]['bootstrap95Percent']
    if len(interval) != 2 or not all(math.isfinite(value) for value in interval) or not -1 <= interval[0] <= interval[1] <= 1:
        raise ValueError('Role evaluation intervals must be finite valid utility differences')
    return interval


def utility(report):
    """Candidate utility balanced over episode lengths, plus clear improvements and regressions.

    The cohort holds 144 short maps and 24 long ones. Averaging over maps let a
    policy trained on 11.5 s episodes win while losing every 30 s and 60 s map,
    which is exactly what the browser then showed. Each play length now counts
    once, and a clear regression at any length blocks promotion.
    """
    by_length = {play: block for play, block in (report.get('byPlayLength') or {}).items() if block.get('maps')}
    if not by_length:
        by_length = {'all': dict(summary=report['summary'], contrasts=report['contrasts'])}
    per_length = {}
    for play, block in by_length.items():
        hider = float(block['summary']['candidate-hider']['hiddenFraction'])
        seeker = 1 - float(block['summary']['candidate-seeker']['hiddenFraction'])
        if not (0 <= hider <= 1 and 0 <= seeker <= 1):
            raise ValueError('Evaluation utilities must be finite fractions')
        per_length[play] = dict(hiderUtility=hider, seekerUtility=seeker, score=.5 * (hider + seeker))
    hider = float(np.mean([v['hiderUtility'] for v in per_length.values()]))
    seeker = float(np.mean([v['seekerUtility'] for v in per_length.values()]))
    regressions, improvements = [], []
    for role in ['Hider change', 'Seeker change']:
        interval = interval_of(report['contrasts'], role)
        if interval[1] < 0:
            regressions.append(role)
        if interval[0] > 0:
            improvements.append(role)
        for play, block in by_length.items():
            if play != 'all' and interval_of(block['contrasts'], role)[1] < 0:
                regressions.append(f'{role} at play {play}')
    return dict(score=.5 * (hider + seeker), hiderUtility=hider, seekerUtility=seeker, byPlayLength=per_length,
                roleRegressions=regressions, clearImprovements=improvements)


def register(candidate_path, reference_path, report_path, directory):
    report = json.loads(Path(report_path).read_text())
    contract = evaluation_contract(report)
    candidate_hash, reference_hash = file_hash(candidate_path), file_hash(reference_path)
    if candidate_hash != report['checkpointSHA256'] or reference_hash != report['referenceSHA256']:
        raise ValueError('Evaluation hashes must match the exact native candidate and reference pairs')
    candidate, reference = load_native(candidate_path), load_native(reference_path)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'EVALUATED.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        format='hide-seek-evaluated-native-pairs-v1', evaluationContractSHA256=contract,
        rule='Highest length-balanced utility against the same fixed opponents, promoted only with a statistically clear improvement in at least one role and no statistically clear regression versus reference overall or at any episode length. This is measured development performance, not browser promotion or proof of useful tools.',
        records=[])
    if manifest['evaluationContractSHA256'] != contract:
        raise ValueError('Do not rank evaluations with different maps, opponents, physics or inference code')
    evidence_hash = file_hash(report_path)
    relative_report = f'evaluations/{evidence_hash}.json'
    copy_immutable(report_path, directory / relative_report)

    def preserve(path, saved, digest):
        relative = f'evaluated/{digest}.pt'
        if copy_immutable(path, directory / relative) != digest:
            raise ValueError('Candidate changed after evaluation; select its immutable checkpoint')
        stage_dependencies(path, saved, directory)
        return dict(file=relative, sha256=digest, sourceProvenance=saved['provenance'],
            encoderTypes=saved.get('encoderTypes', ['legacy-linear-v1'] * 2),
            trainingCounters={name: saved.get(name) for name in ['totalPolicyInteractions',
                'currentPolicyDecisions', 'historicalPolicyDecisions', 'activePolicySamples', 'seconds']},
            evidence=relative_report, recordedUTC=utc_now(), entirePairPreserved=True)

    if 'best' not in manifest:
        manifest['best'] = dict(preserve(reference_path, reference, reference_hash),
            score=.5, selection='Evaluated fixed reference; paired self-reference utility is exactly one half.')
    measured = utility(report)
    score = measured['score']
    regressions, improvements = measured['roleRegressions'], measured['clearImprovements']
    # A promotion needs evidence, not a coin flip: some role must have improved
    # with a bootstrap interval excluding zero, and no role may have clearly
    # regressed overall or at any episode length.
    eligible = not regressions and bool(improvements)
    entry = dict(preserve(candidate_path, candidate, candidate_hash), **measured, eligible=eligible, reportSHA256=evidence_hash)
    if not any(row['sha256'] == candidate_hash and row['reportSHA256'] == evidence_hash
               for row in manifest['records']):
        manifest['records'].append(entry)
    if eligible and score > manifest['best']['score']:
        manifest['best'] = dict(entry, selection='Higher fixed-opponent utility with a statistically clear role improvement; no clear regression.')
    manifest['updatedUTC'] = utc_now()
    replace_from_immutable(directory / manifest['best']['file'], directory / 'best-evaluated.pt')
    atomic_json(manifest_path, manifest)
    atomic_json(directory / 'best-evaluated.json', manifest['best'])
    return dict(candidateSHA256=candidate_hash, score=score, eligible=eligible,
                bestSHA256=manifest['best']['sha256'], bestFile=manifest['best']['file'], publicAssetsChanged=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'reference', 'report', 'registry']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.checkpoint, args.reference, args.report, args.registry), indent=2))
