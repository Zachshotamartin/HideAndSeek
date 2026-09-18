"""One-update training smoke test on a synthetic pair in a temporary directory."""
import argparse
import json
import tempfile
from pathlib import Path

from synthetic import prepared_setup
from train_entity import train

SMALL_RUN = dict(envs=4, workers=2, horizon=64, sequence_length=32, burn_in=8, snapshot_every=1, archive_limit=13,
                 sequence_batch=4, target_interactions=1024, stop_after_updates=1)


def small_arguments(setup, **overrides):
    """The prepared arguments shrunk to a laptop-sized run, with the protocol rewritten to match."""
    args = argparse.Namespace(**setup['arguments'])
    for name, value in overrides.items():
        setattr(args, name, value)
    protocol = json.loads(Path(args.protocol).read_text())
    protocol['training'] = {k: getattr(args, k) for k in protocol['training']}
    Path(args.protocol).write_text(json.dumps(protocol))
    return args


def main():
    with tempfile.TemporaryDirectory() as folder:
        setup = prepared_setup(folder, target=1024)
        train(small_arguments(setup, **SMALL_RUN))
        print('smoke: one synthetic update completed')


if __name__ == '__main__':
    main()
