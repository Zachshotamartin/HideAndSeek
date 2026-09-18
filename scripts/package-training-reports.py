"""Copy compact measured development results without raw rollout storage.

All reported metrics are copied from complete frozen reports. No evaluation or
training is performed, and this does not modify browser model assets.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPORTS = {
    'league-4m': 'validation-4m/evaluation.json',
    'league-8m': 'validation-8m/evaluation.json',
    'mixed-broad': 'mixed-validation/evaluation.json',
    'gradients-parent': 'diagnostics-parent/diagnostics.json',
    'gradients-A': 'diagnostics-A/diagnostics.json',
    'gradients-B': 'diagnostics-B/diagnostics.json',
}

SUSTAINED_REPORTS = {
    'sustained-8m': 'validation-7995392/evaluation.json',
    'sustained-16m': 'validation-15990784/evaluation.json',
    'sustained-32m': 'validation-31981568/evaluation.json',
    'sustained-8m-browser': 'validation-7995392/browser-reference-comparison.json',
    'sustained-16m-browser': 'validation-15990784/browser-reference-comparison.json',
    'sustained-32m-browser': 'validation-31981568/browser-reference-comparison.json',
    'search-memory-diagnostic': 'search-audit/report.json',
    'entity-search-diagnostic': 'entity-failure-audit/report.json',
    'sustained-completion': 'completion.json',
    'sustained-16m-vs-32m': 'comparison-16m-32m.json',
    'selected-action-modes': 'policy-modes/evaluation.json',
    'selected-tool-effects': 'selected-tools/evaluation.json',
    'selected-role-lineage': 'selected-candidate/candidate.selection.json',
}


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()
                if key not in ['episodes', 'newEpisodes', 'frames'] or not isinstance(item, list)}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.startswith('/Users/'):
        return 'local-artifact/' + Path(value).name
    return value


def main(args):
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    inputs = [(name, relative, Path(args.input)) for name, relative in REPORTS.items()]
    if args.sustained_input:
        inputs += [(name, relative, Path(args.sustained_input))
                   for name, relative in SUSTAINED_REPORTS.items()]
    for name, relative, base in inputs:
        source = base / relative
        data = json.loads(source.read_text())
        payload = dict(format='hide-seek-compact-development-report-v1', name=name,
            sourceReportSHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
            sourceReportRelativePath=relative,
            packaging='Metrics copied unchanged from the complete frozen report. Raw episode arrays are omitted; absolute local artifact paths are replaced with display-only basenames.',
            acceptedBrowserModelChanged=False, report=portable(data))
        target = destination / f'{name}.json'
        target.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'{target}: {target.stat().st_size} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/league-trial')
    parser.add_argument('--output', default='training/reports')
    parser.add_argument('--sustained-input', help='Include all completed sustained milestones and read-only diagnostics')
    main(parser.parse_args())
