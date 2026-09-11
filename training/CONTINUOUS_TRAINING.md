# Keep training on fresh games

`npm run train:continuous` runs original recurrent PPO on fresh randomized physical games. It can stop at an additional interaction budget or keep running until interrupted. The existing browser actor remains a separate frozen reference. Training does not automatically replace it.

Install `training/requirements.txt` in a virtual environment first. The launcher checks for a working MuJoCo/Torch/NumPy interpreter. Set `PYTHON_BIN` to use another interpreter. Native `.pt` checkpoints include both actors, optimizer states, the separate critic and optimizer, RNG states, progress counters, and source provenance. They remain local training artifacts, separate from the small browser actor JSONs.

## Continue the pretrained model from a fresh clone

The repository includes `training/resume-bundle/`. It contains the exact 37M development actor, its native optimizer moments/Torch RNG, the genuine initial reference, our own historical opponents, and the compatible fitted critic. `MANIFEST.json` records every asset's full SHA-256; the driver verifies the manifest before loading the bundle. The prepared checkpoint contains **zero new actor updates**. It starts a documented fresh training phase with learning rate 0.0001 and new-world RNG seed 773119; it is not a claim of seamless continuation of the original pilot's in-flight games.

After installing the Python requirements, one command starts collecting fresh games:

```sh
npm run train:continuous -- \
  --resume training/resume-bundle/resume.pt \
  --output output/continuous-run \
  --until-stop
```

The run copies every required native asset into its own `assets/` folder and saves relative paths. Copying the whole run directory to another machine preserves its dependencies; no `/Users/...` paths or files in the original ignored output directory are required. Its configured Python dependencies still need to be installed on that machine. Browser downloads do not include this native bundle.

## Resume indefinitely

For an existing continuous run:

```sh
npm run train:continuous -- \
  --resume output/continuous-run/latest.pt \
  --until-stop
```

Press **Ctrl-C once** to finish the current rollout and optimization update, then save. Sending `SIGTERM` to the trainer or its process group also requests a safe stop. Workers ignore those stop signals while the parent completes that operation; they are then shut down normally. A complete update usually takes seconds on the development machine, rather than stopping partway through an optimizer step.

To add a finite amount of fresh game experience:

```sh
npm run train:continuous -- \
  --resume output/continuous-run/latest.pt \
  --steps 10000000
```

`--steps` counts both agents' physical decisions, including frozen opponents when using the historical-opponent arm. It rounds **down** to complete rollouts and prints the actual requested extension. With 128 worlds and 256 control ticks, a rollout contains 65,536 actor decisions. Current-policy decisions, historical-opponent decisions, and active policy-loss samples are reported separately for each role.

## Start a separate continuation from a saved experiment

Fixed experiments are preserved. Continue one in a new output directory:

```sh
npm run train:continuous -- \
  --resume output/league-trial/B/checkpoint-7995392.pt \
  --output output/continuous-run \
  --until-stop
```

This command illustrates resumability, **not a recommendation to deploy that pair**. The B arm’s 8M hider regressed in the recorded development comparison; its improved seeker is retained separately for further assessment. A training checkpoint's existence does not qualify its behavior.

To initialize from the original persistent parent and its compatible fitted value baseline:

```sh
npm run train:continuous -- \
  --parent output/persistent-button-pilot/run/checkpoint-14098432.pt \
  --critic output/central-value-pilot/residual-fit/fitted-critic.pt \
  --initial output/persistent-button-pilot/run/initial.pt \
  --history output/persistent-button-pilot/run/warm-start.pt \
            output/persistent-button-pilot/run/checkpoint-6291456.pt \
            output/persistent-button-pilot/run/checkpoint-14098432.pt \
  --arm A --output output/continuous-run --until-stop
```

Use the actual retained native files at those paths, or substitute compatible files. Browser-only JSONs omit optimizer/value/RNG training state and are not exact training-resume checkpoints. Arm A uses current/current self-play. Arm B uses the documented 50% current/current and 25% each current/historical role mix. Both retain the same visibility reward; neither uses scripted pursuit, tool-use bonuses, demonstrations, or enforced hiding behavior.

## Start with random actors

To train the original binary-button architecture from a random initialization in a separate directory:

```sh
npm run train -- --output training/runs/from-scratch \
  --envs 64 --workers 8 --horizon 128 --updates 1000
```

This older trainer is a reproducible starting point, not the pretrained persistent-button continuation described above. It saves native checkpoints periodically; `--updates` is its absolute update target. The continuous driver is the primary workflow for extending the supplied persistent model while retaining its learned weights and complete training state.

## Latest, archived, and accepted models

- `latest.pt` is atomically replaced after every completed update. It is resumable progress, not a performance-selected model.
- `checkpoints/<interactions>.pt` contains immutable periodic archives, plus the final or interrupted checkpoint. `--archive-every 16` controls their interval in complete rollouts.
- `best-reference.pt` is a byte-identical copy of the frozen parent comparison. Training never updates or silently reselects it.
- The browser manifest remains unchanged. Promotion requires a separate fixed-opponent and tool-ablation evaluation, with the exact checkpoint hash, before an actor-only export.

The saved optimizer and RNG state resume exactly. In-flight physical worlds, recurrent memories, and requested buttons deliberately restart at process boundaries; that restart is recorded in provenance. This is **not** a claim that an interrupted run follows the exact same future trajectory as an uninterrupted process. Only complete updates count toward progress; a hard process kill can lose work since the last completed update. Both source snapshots and immutable archives remain in the run directory.

Longer training means more fresh rollout/update cycles. Increasing PPO `--epochs` instead repeatedly optimizes the same rollout and can increase policy drift. Resume locks the optimization settings to the recorded run. Start a separately documented experiment for an intentional optimization change. This distinction keeps training extendable without quietly invalidating a matched comparison.

Run the lifecycle regression with:

```sh
.venv-physics/bin/python -m unittest discover -s training -p test_continuous_training.py -v
```

It sends real process-group SIGINT and SIGTERM through both Python and Node launchers, checks a complete archived checkpoint, moves the run and deletes its original input files, then resumes for exactly one fresh rollout. It verifies optimizer/RNG progression and confirms the frozen reference stayed unchanged. Synthetic lifecycle fixtures are not learned-performance evidence.
