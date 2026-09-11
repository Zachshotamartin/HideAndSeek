# Saved training, timing and evaluated models

`npm run train:saved` continues a full native persistent or entity checkpoint. It collects additional fresh games, keeps periodic immutable snapshots, and supports an orderly stop and later resume. It leaves fixed experiments, their protocols, and browser model assets unchanged.

Install `training/requirements.txt` in a Python environment. The Node launcher checks for MuJoCo, PyTorch and NumPy; set `PYTHON_BIN` if needed. The original portable 37M bundle remains a supported starting point:

```sh
npm run train:saved -- \
  --resume training/resume-bundle/resume.pt \
  --output output/saved-run \
  --steps 10000000 \
  --snapshot-every 1000000
```

`--steps` means **additional** actor interactions, including frozen-opponent decisions and preparation. It rounds down to complete fresh rollouts. Snapshot spacing rounds up. For 128 worlds and a 256-tick rollout, the basic unit is 65,536 interactions: the command above adds 9,961,472 and archives every 1,048,576. Active policy-gradient samples and each role's current/frozen decisions remain separate counters. This is not a request to increase PPO epochs over old data.

## Pause and resume

Use `--until-stop` instead of `--steps` for an open-ended run:

```sh
npm run train:saved -- \
  --resume output/saved-run/latest.pt \
  --until-stop
```

Press **Ctrl-C once**, or send `SIGTERM` to the launcher or its process group. The parent finishes the current rollout and optimizer update, saves it, and exits. Worker stop handling is unchanged. A kernel lock prevents a second trainer from writing the same managed run.

Resume `latest.pt` or its identical final snapshot. To extend beyond an earlier target:

```sh
npm run train:saved -- \
  --resume output/saved-run/latest.pt \
  --steps 20000000
```

To branch from an older snapshot or evaluated-best model, choose a new output directory. This preserves the later snapshots already in the original run:

```sh
npm run train:saved -- \
  --resume output/saved-run/checkpoints/1048576.pt \
  --output output/alternate-continuation \
  --steps 10000000
```

Replace the example snapshot filename with a retained checkpoint from the run. Whole hider/seeker weights, both actor optimizers, central critic and optimizer, RNG states, counters, and source provenance are retained. Physical worlds, recurrent memories and requested buttons **restart together at process boundaries**. Resuming does not mean an interrupted process follows the same future trajectory as an uninterrupted one. Nothing is retrained from zero unless a zero-experience checkpoint was explicitly selected.

For a fixed entity experiment, `--output` is mandatory on the first continuation. The wrapper creates a separate protocol linked by hash to the original checkpoint and protocol, recording the new target. Only administrative metadata is adapted before the next real update; actor weights, optimizer moments, critic and RNG are unchanged. Existing optimization settings remain locked. The original fixed comparison is never reinterpreted as a longer matched experiment.

## Files and timing

| File | Meaning |
| --- | --- |
| `latest.pt` | Atomic full native checkpoint after every completed update. |
| `checkpoints/<interactions>.pt` | Immutable step snapshots, plus final/interrupted snapshots. |
| `RUN.json` | Snapshot hashes, current progress and per-invocation timing. |
| `assets/` | Exact required native dependencies stored by hash. |
| `resume-inputs/` | Original checkpoint bytes and explicitly derived administrative resume inputs. |
| `protocols/`, `source/`, `managed-source/` | Extension protocol and exact training/driver source history. |
| `best-reference.pt` | Frozen starting pair, not automatically a best evaluated model. |

`RUN.json` records UTC start/end times, active session wall seconds, cumulative native training seconds, and rollout/optimizer seconds for each invocation. Session wall time includes setup and saving; native training time describes the trainer loop. Time between stopped invocations is excluded. This does not treat an operating-system suspension as a separately measured pause.

Copy the **whole run directory** when moving machines. Relative and hash-addressed dependencies make the copied run resumable, including snapshots nested in `checkpoints/`. The native bundle manifest is checked before loading. Actor-only browser JSON files omit optimizer, critic and RNG state and cannot replace full native checkpoints. A hard kill can lose an incomplete update; completed snapshots remain intact.

## Explicit evaluated-best selection

Training return never updates the best-model pointer. Run a separate fixed-opponent evaluation using a declared development cohort. The cohort JSON contains a `maps` array with `seed`, `scenario`, and `arenaConfig` (`size`, `n_boxes`, `n_ramps`). The existing encoder protocol is one such cohort; it is development data, not a final-test claim.

```sh
python training/evaluate_saved.py \
  --checkpoint output/saved-run/checkpoints/1048576.pt \
  --reference output/saved-run/best-reference.pt \
  --cohort output/entity-encoder/comparison/PROTOCOL.json \
  --output output/saved-run/evaluation-1048576 \
  --registry output/saved-run
```

Use an existing immutable snapshot and cohort. The evaluator first freezes the exact candidate/reference inputs. It measures each role against the same frozen counterpart, the current pair, and physical grab/lock ablations that preserve pushing. Its common seeded action tapes match actual stochastic policy sampling. It records complete misses, physical interaction statistics, and paired bootstrap intervals.

The registry ranks the average of hider hidden fraction and seeker visible fraction against those **fixed opponents**. The fixed reference scores 0.5 by construction. A candidate with a statistically clear role regression versus that reference is ineligible. An eligible candidate must exceed the previous best score to replace the local pointer. All candidates and reports remain retained, including rejected ones. Different maps, reference opponents, physics or inference code require a separate registry; incompatible scores cannot silently compete.

This is “best under this declared development criterion,” not proof of a globally strongest policy or useful tool strategy. Repeated selection on the same cohort is not held-out evidence. The guard uses unadjusted 95% paired intervals. Browser deployment still requires a separate review and explicit export.

| File | Meaning |
| --- | --- |
| `EVALUATED.json` | Every registered candidate, evaluation, eligibility decision and best selection. |
| `evaluated/<sha256>.pt` | Exact immutable native role pair with optimizers and provenance. |
| `evaluations/<sha256>.json` | Immutable evidence tied to checkpoint hashes. |
| `best-evaluated.json` | Selected native pair, score, evidence and source lineage. |
| `best-evaluated.pt` | Atomic copy of the selected full native checkpoint. |

Continue the selected pair in a separate branch:

```sh
npm run train:saved -- \
  --resume output/saved-run/best-evaluated.pt \
  --output output/from-evaluated-best \
  --until-stop
```

The registry retains the full pair; it never silently takes a hider from one checkpoint and a seeker from another. Dependencies are preserved so the best checkpoint can resume even after the former source directory is unavailable.

## Existing workflow audit

The frozen entity comparison saves `latest.pt` every eight rollouts and retains its declared 8M/16M milestones. It records cumulative training and rollout/optimizer timing, and supports resuming within its fixed protocol. Its automatic launcher evaluates the retained checkpoints but does not select best. These files were not changed to add the managed continuation workflow.

The established `train:continuous` workflow still supports the portable 37M bundle, relative dependencies, periodic archives, graceful stops, and additional steps/until-stop. Its `best-reference.pt` is explicitly the frozen starting reference. The new `train:saved` and evaluation registry add the uniform checkpoint/timing/evaluated-best workflow for subsequent persistent and entity runs.

## Regression checks

```sh
python -m unittest discover -s training -p test_saved_workflow.py -v
```

Three regression cases exercise actual native training: process-group SIGTERM and duplicate-writer rejection; immutable snapshot retention, relocation after deleting original inputs, optimizer/RNG preservation and extension beyond the old target; the actual portable 37M bundle; and explicit evaluation/registry validation with resume from the best checkpoint after source dependencies disappear. Synthetic ranking numbers in the registry test are test fixtures, not learned-performance evidence. No extra performance-training phase is started by these checks.
