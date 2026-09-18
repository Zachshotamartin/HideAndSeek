# Larger recurrent entity continuation

The previous entity comparison used 96 encoder units, 32 embedding units and
64 recurrent units, with 32-step sequences (2.56 seconds). Its actor had 45,868
parameters. Changing representation alone did not establish better search or
useful tool use in the 16M-interaction comparison.

This continuation uses 256 encoder units, 128 embedding units, 256 GRU units,
and 128-step sequences (10.24 seconds): 503,180 actor parameters per role. Full
rollouts are 256 steps across 128 environments. A separate central critic sees
the complete scene only during training. The deployed actor receives the same
restricted 140 observations as before.

```sh
python training/train_scaled.py \
  --source output/entity-encoder/comparison/entity/checkpoint-15990784.pt \
  --output output/scaled-memory-256
```

The default target is 104,857,600 new combined actor interactions. Counts include
both roles and historical opponents. Current-policy and active-play samples are
recorded separately. The first training update and roughly every 1M interactions
retain an immutable complete checkpoint; the atomic latest checkpoint updates
every PPO update. SIGTERM requests a safe save and pause. The identical command
continues the run. To extend after the target, use `train_saved.py` with its latest
or any archived/best full checkpoint and a new output directory.

`widen_policy.py` embeds the learned smaller actor into the larger network. It
preserves existing recurrent trajectories, attention normalization, action heads
and value predictions initially; additional trainable units start disconnected
from the old outputs. The larger critic also initially preserves its predictions.
Tests cover 160-step recurrent equivalence, new-unit gradients, real physics
updates, and resume. Optimizer moments are explicitly reset for this migration;
it is not an exact continuation in an unchanged parameter space.

The first update and each 5,242,880 interactions receive a fixed 96-map
development evaluation. Each role plays against the same pre-expansion reference
opponent. Paired tests remove grab/lock while retaining physical pushing to check
whether those tools help. An evaluated best registry retains entire native
actor/critic/optimizer checkpoints. A statistically clear regression in either
role blocks best replacement. Training return does not select the best.

Half the training episodes are current versus current. The rest use frozen own
historical opponents, including the completed 16M entity pair. No hand-authored
pursuit, hiding targets, barricade instructions, demonstrations, or tool bonuses
are added. The reward remains zero-sum visibility.

This changes capacity, sequence length and training duration together. Improvement
over the fixed source measures their combined effect; it does not isolate which
change caused it. Reused development maps and training return cannot establish
generalization. Final held-out maps, browser inference checks, and actual visual
play review are required before promotion. The run never publishes automatically.

Read `output/scaled-memory-256/STATUS.json` for phase, counters, timing, retained
snapshots and completed evaluations. Training is an independent native process;
no assistant or subagent needs to stay active for it to continue.
