# Persistent-button PPO pilot

This is a separate controller experiment. It does not replace the released
actor, training run, renderer, physical simulator, assets, or interface.

Each tool head chooses **Keep, Press, or Release** at every 0.08-second control
step. Keep retains that tool's requested button state. Press sets it to one;
Release immediately sets it to zero. There is no minimum hold time. The two
requested states are appended to the 138 ordinary physical observations, so
both actor and critic receive 140 measurements. These are the agent's own
controller states, not hidden object or opponent information.

Movement still uses the original bounded Gaussian force/torque distribution.
The physical simulator receives exactly the original three continuous controls
and two binary tool controls. No physics or reward changes are involved. The
only reward is the existing zero-sum visibility outcome, uniformly scaled by
0.05 for value optimization. No holding, grabbing, placement, pursuit, or
construction reward was introduced.

The new categorical head starts with six exactly zero logits. Every command
therefore has probability one third, with no preference for any command.
Deterministic argmax ties resolve to Keep; evaluation uses seeded sampling.
Uniform commands give both pressed and released states temporal continuity,
without requiring a duration or prescribing what to do with an object.

The seeker still advances its GRU during preparation. Its applied controls and
requested button states remain zero until preparation finishes. Discarded
preparation commands have no actor or entropy gradient; the critic and memory
still learn. Terminal resets clear both the GRU and requested states.

## Initialization and provenance

The parent checkpoint is an immutable copy of our own ordinary PPO actor after
22,937,600 decisions. The pilot copies encoder columns 0–137 and its bias, GRU,
movement head, movement deviations, and value head. New encoder columns 138–139
and the new tool head are zero. Adam starts fresh; no optimizer moments are
copied from the parent. The checkpoint records the exact parent and source
hashes, the optimizer reset, and parent versus additional pilot experience.

The initial comparator copies the original **zero-experience** checkpoint,
with the same new zero categorical head. It has zero parent and pilot
decisions. The zero-update warm start is a different checkpoint and is never
called an initial or untrained model.

The bounded run used 32 physical environments in four workers, 128-step
rollouts, 32-step recurrent sequences, sequence minibatches of 64, three PPO
epochs, learning rate 0.0003, entropy coefficient 0.005, and seed 940313. It ran
256 updates: 2,097,152 additional actor decisions and 4,352 completed games in
384.9 seconds. The total inherited plus new experience is 25,034,752 decisions.
Checkpoints retain actor/critic weights, optimizer state, RNG state, counters,
arguments, and provenance. They do not serialize live MuJoCo rollout states;
an exact interrupted-rollout continuation is not claimed.

## Files and reproduction

- `persistent_actor.py`: original 140-input actor and categorical likelihoods.
- `persistent_train.py`: isolated pooled recurrent PPO training loop.
- `persistent_evaluate.py`: paired validation, fixed opponents, physical
  ablations, actual grip durations and contact diagnostics.
- `test_persistent.py`: controller, warm-start, density, and RNG tests.
- `../src/core/persistentPolicy.js`: separate browser inference and controller.
- `../scripts/export-persistent-policy.py`: portable export and native/JS parity.
- `../scripts/test-persistent-parity.mjs`: stateful JavaScript regression.

From the repository root, using a Python environment with Torch, NumPy and
MuJoCo 3.13:

```sh
python training/test_persistent.py
python training/persistent_train.py \
  --parent output/persistent-button-pilot/parent.pt \
  --initial output/persistent-button-pilot/original-initial.pt \
  --output output/persistent-button-pilot/run --updates 256
python scripts/export-persistent-policy.py \
  output/persistent-button-pilot/run/latest.pt \
  --output output/persistent-button-pilot/pilot.json
python training/persistent_evaluate.py \
  --checkpoint output/persistent-button-pilot/run/latest.pt \
  --warm-start output/persistent-button-pilot/run/warm-start.pt \
  --initial output/persistent-button-pilot/run/initial.pt \
  --parent output/persistent-button-pilot/parent.pt \
  --output output/persistent-button-pilot/validation \
  --episodes-per-scenario 12 --seed 1500010000
```

Export parity covers 1,056 decisions across both roles and deterministic plus
seeded sampled inference, native observations through preparation and terminal
boundaries, four actor resets, categorical commands, requested states, logits,
movement and recurrent memory. Sampling uses an explicit common uniform tape
and Box–Muller transform; it does not assume matching Torch/JavaScript PRNG
seeds. The final pilot's maximum numerical error is 1.76e-6 at tolerance 1e-5.
The warm start and actual initial models also pass independently.

## Evaluation protocol

The 432-game comparison uses 36 validation maps: 12 each of shelter, rooms and
open arenas, eight metres wide with three boxes and one ramp. Seeds begin at
1,500,010,000. They are outside training's seed range and repeat prior
validation maps; they are not the untouched final test set beginning at 1.9B.

Twelve modes compare learned self-play, each role's tool ablation, both tools
disabled, the zero-update persistent warm start, the actual initial pair, and
each trained/warm-start/original-parent role against the **same fixed initial
opponent**. Role-specific uniform tapes are shared across modes. The parent
comparisons separate effects of changing the action representation from those
of the additional pilot learning.

Tool ablations disable physical grab/lock effects while leaving physical
pushing and the actor's requested controller state intact. A changed result
can arise from a changed trajectory; it is not automatically evidence of
barrier construction. Reports therefore retain contact and real grip-duration
diagnostics and use map-paired bootstrap confidence intervals. Longer grip
duration alone is not an acceptance criterion.

Frozen models, exported JSON, parity fixtures, full logs and the validation
report are under `output/persistent-button-pilot/`, outside production assets.
See that directory's review for the measured outcome and recommendation.
