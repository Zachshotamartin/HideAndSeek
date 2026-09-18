"""Reproduce the largest positive/negative validation tool effects as raw traces.

These deliberately selected extremes are diagnostics, not representative success
rates. Aggregate paired validation intervals remain the performance evidence.
"""
import argparse
import json
import math
from pathlib import Path

from persistent_evaluate import episode, load, plain
from persistent_train import file_hash


def figure(left, right, heading, ablation):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="610">',
           '<rect width="1100" height="610" fill="#f7f7f2"/>',
           '<style>text{font-family:Arial;fill:#26323c}</style>',
           f'<text x="22" y="28" font-size="19">{heading}</text>',
           '<text x="22" y="51" font-size="13">Blue hider / red seeker; dashed preparation. Orange props initially; dark outlines finally.</text>']
    for column, result in enumerate([left, right]):
        x0, y0, scale = 25 + column * 545, 110, 48

        def point(x, y):
            return x0 + x * scale, y0 + (8 - y) * scale

        def poly(position, size, yaw, style):
            vertices = []
            for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                vertices.append(point(position[0] + math.cos(yaw) * dx * size[0] / 2 - math.sin(yaw) * dy * size[1] / 2,
                                      position[1] + math.sin(yaw) * dx * size[0] / 2 + math.cos(yaw) * dy * size[1] / 2))
            return '<polygon points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in vertices) + '" ' + style + '/>'

        def path(positions, style):
            return '<polyline points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in [point(*p) for p in positions]) + '" ' + style + '/>'

        label = 'Actual tools' if column == 0 else ablation
        svg.append(f'<text x="{x0}" y="86" font-size="17">{label}: hidden {result["hiddenFraction"]:.1%}</text>')
        for wall in result['arena']['walls']:
            svg.append(poly(wall['position'], wall['size'], wall['yaw'], 'fill="#8a979d"'))
        for index, obj in enumerate(result['arena']['objects']):
            svg.append(poly(obj['position'], obj['size'], obj['yaw'], 'fill="#e8aa4f" fill-opacity=".5"'))
            qpos = result['frames'][-1]['qpos']
            svg.append(poly(qpos[8 + 4 * index:11 + 4 * index], obj['size'], qpos[11 + 4 * index],
                            'fill="none" stroke="#bd6800" stroke-width="2"'))
            svg.append(path([frame['qpos'][8 + 4 * index:10 + 4 * index] for frame in result['frames']],
                            'fill="none" stroke="#bd6800" stroke-width="1.2"'))
        for role, color in enumerate(['#2266a5', '#bc3d44']):
            positions = [frame['qpos'][4 * role:4 * role + 2] for frame in result['frames']]
            svg.append(path(positions[:81], f'fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="4 3"'))
            svg.append(path(positions[80:], f'fill="none" stroke="{color}" stroke-width="2"'))
            for tick in [0, 240]:
                x, y = point(*positions[tick])
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        grips = [max(role['gripDurationsTicks'], default=0) * .08 for role in result['roles']]
        pressing = [role['playWallPressFrames'] for role in result['roles']]
        svg.append(f'<text x="{x0}" y="535" font-size="13">Longest actual grip H/S: {grips[0]:.2f}s / {grips[1]:.2f}s</text>')
        svg.append(f'<text x="{x0}" y="557" font-size="13">Slow wall-press frames H/S: {pressing[0]} / {pressing[1]} of160 play ticks</text>')
    svg.append('<text x="22" y="594" font-size="12">Extreme paired effects selected for inspection. Changed outcomes alone do not establish construction or intent.</text></svg>')
    return ''.join(svg)


def main(args):
    report = json.loads(Path(args.report).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = {(row['mode'], row['seed']): row for row in report['episodes']}
    cases = []
    for arm, path in [('A', args.a), ('B', args.b)]:
        if file_hash(path) != report['checkpoints'][arm]['sha256']:
            raise ValueError('Trace actors must exactly reproduce the assessed checkpoint')
        models, _ = load(path)
        learned = [row for row in report['episodes'] if row['mode'] == f'{arm}/learned']
        for role, physical_mode in [('hider', 'no-hider-tools'), ('seeker', 'no-seeker-tools')]:
            def effect(row):
                difference = row['hiddenFraction'] - rows[(f'{arm}/{physical_mode}', row['seed'])]['hiddenFraction']
                return difference if role == 'hider' else -difference
            for label, selected in [('best', max(learned, key=effect)), ('worst', min(learned, key=effect))]:
                name = f'{arm}-{role}-{label}-{selected["seed"]}'
                traces = []
                for mode in ['learned', physical_mode]:
                    result = episode(models, selected['seed'], selected['scenario'], mode, trace=True)
                    expected = rows[(f'{arm}/{mode}', selected['seed'])]
                    if result['hiddenFraction'] != expected['hiddenFraction'] or result['info'] != expected['info']:
                        raise AssertionError('Actual replay does not match the evaluated trajectory outcomes')
                    (output / f'{name}-{mode}.json').write_text(json.dumps(result, default=plain) + '\n')
                    traces.append(result)
                title = f'Arm {arm}, {role} {label} tool effect: {effect(selected):+.1%}; {selected["scenario"]}, seed {selected["seed"]}'
                (output / f'{name}.svg').write_text(figure(*traces, title, physical_mode))
                cases.append(dict(name=name, arm=arm, role=role, selection=label, seed=selected['seed'],
                                  scenario=selected['scenario'], toolBenefit=effect(selected)))
    (output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    print(json.dumps(cases, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ['a', 'b', 'report', 'output']:
        parser.add_argument('--' + argument, required=True)
    main(parser.parse_args())
