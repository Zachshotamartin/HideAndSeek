# Sustained self-play from retained roles

This phase preserves the exact 37,036,032-interaction development hider and the seeker from the completed historical-opponent arm after 7,995,392 additional interactions. The original whole league pair was rejected because its hider regressed. The separately assessed mixed pair remains a development initialization: 22 of 96 broad validation games were complete search misses, and useful tool effects were not established.

The assessed mixed file is immutable. `scripts/prepare-mixed-continuation.py` creates a new native training checkpoint by copying the selected role weights and Adam moments exactly. It records each source checkpoint’s SHA-256 and counters. Their shared ancestry, critic-only warm-up, and historical-opponent interactions are counted once in the union source budget; the roles are never described as having identical training experience. Preparation itself makes zero actor updates.

The central critic and its optimizer are inherited from the same league seeker checkpoint. This is an existing training initialization, **not a claim that the critic was newly fitted or calibrated on the mixed pair**. No new critic warm-up decisions are added. The checkpoint records that distinction explicitly. The critic is never supplied to browser inference or actor observations. Torch RNG is inherited from the seeker phase; new world and opponent streams start from seed 774113. Physical worlds, recurrent memory, and requested buttons start fresh.

## Fixed phase

Collect 31,981,568 fresh actor interactions: 488 complete rollouts of 128 worlds × 256 ticks × two roles. This is the complete-rollout value below a nominal 32 million budget. It is an assessment phase, not a permanent training limit; the same native checkpoint can continue with `--until-stop` or further `--steps` afterward.

Keep the reviewed league unchanged: 50% current/current, 25% current hider/historical seeker, and 25% historical hider/current seeker. The frozen pool remains our original persistent warm start, intermediate pilot, and 37M pair. Settings remain two PPO epochs, recurrent chunks of 32, minibatches of 256 chunks, actor/critic learning rates 0.0001, entropy coefficient 0.005, and active-row KL cutoff 0.008. Physics, rewards, restricted actor observations, and persistent button semantics remain unchanged. There are no demonstrations, pursuit or tool bonuses, mandatory hiding locations, or scripted strategies.

Record total physical interactions, current-policy decisions, frozen-opponent decisions, and active loss samples separately by role. Save `latest.pt` after every update and immutable milestone archives after every 122 rollouts. Preserve the selected parent and earlier references throughout the run. Training reward cannot select an accepted model automatically.

The reproducible preparation and run commands, using the retained local development artifacts, are:

```sh
.venv-physics/bin/python scripts/prepare-mixed-continuation.py \
  --hider output/persistent-button-pilot/run/checkpoint-14098432.pt \
  --seeker output/league-trial/B/checkpoint-7995392.pt \
  --mixed output/league-trial/mixed-validation/mixed-role-candidate.pt \
  --critic output/central-value-pilot/residual-fit/fitted-critic.pt \
  --initial output/persistent-button-pilot/run/initial.pt \
  --history output/persistent-button-pilot/run/warm-start.pt \
            output/persistent-button-pilot/run/checkpoint-6291456.pt \
            output/persistent-button-pilot/run/checkpoint-14098432.pt \
  --output output/sustained-mixed/prepared

npm run train:continuous -- \
  --resume output/sustained-mixed/prepared/resume.pt \
  --output output/sustained-mixed/run \
  --steps 31981568 --archive-every 122
```

For a fresh repository clone, the separately documented `training/resume-bundle/` provides the native 37M reference and every dependency needed to start its own continuation. The mixed phase’s local preparation is a different, explicitly named development lineage.

## Assessment at milestones

Assess checkpoints at 7,995,392, 15,990,784, and 31,981,568 new interactions on the same 96 broad development maps already declared for the mixed-role assessment: three scenarios, sizes 6–12 m, zero to eight boxes, and zero to two ramps. The same role-specific action tapes apply. These maps now serve as a repeated validation cohort; they are not a final untouched test. Final seeds at 1.9 billion remain reserved.

Reuse the exact cached outcomes of unchanged frozen actors instead of collecting identical games again. Each new checkpoint adds 11 conditions per map: its own pair; each new role against the frozen mixed counterpart; its hider against the original seeker; both roles against genuine initial opponents; its seeker against warm-start and intermediate hiders; and three grab/lock ablations of the new pair. Physical pushing remains enabled in the ablations. Log the original baseline report hash and new checkpoint hash.

Report paired whole-map 95% intervals, complete search misses, visibility, collisions/wall pressing, grip duration, prop displacement and occlusion, and scenario/size/count breakdowns. A credible fixed-opponent role regression prevents replacing that retained role. Inconclusive results do not establish noninferiority or useful tools. Inspect actual traces when a measured change merits explanation; counts of grabs or moved objects alone do not establish intentional constructive use. Preserve promising individual roles with their own lineage rather than automatically promoting a full pair. The browser stays on its existing frozen development reference until separately approved evaluation supports a change.

## Completed outcome and development selection

The declared phase completed all 488 updates in 4,121.8 seconds. It collected 12,006,016 current-hider and 11,917,840 current-seeker decisions, plus 3,984,768 historical-hider and 4,072,944 historical-seeker decisions. Active loss samples were 12,006,016 for the hider and 7,941,680 for the seeker. The final native checkpoint remains immutable at SHA-256 `447c4a2661e35b96d26aa1fba3c25998299aa4ea02a2c305960ce20602469adc`, with optimizer and RNG state preserved for continuation.

The 16M seeker was retained over the final seeker. Against the unchanged 37M hider, complete misses were 16/96 at 16M and 22/96 at 32M. The final seeker’s paired visibility change relative to 16M was +0.25 percentage points [−5.36, +5.70]; against warm-start and intermediate hiders it was −3.93 [−8.86, +0.93] and −4.55 [−9.38, +0.45]. No credible hider improvement appeared. This supports retaining the earlier role for development review, not claiming that latest weights are always stronger.

The selected pair is the exact unchanged 37M hider and 16M seeker, native SHA-256 `ca9462e9cd3f084a2df687defc4de23513f15df700826a3a4524d0676bc74117`. Its own 96-map outcomes, action-mode comparison, and paired tool interventions are packaged in `training/reports/` and the public manifest’s development evidence. Sampled grab/lock effects remained inconclusive for both roles. Mean mode lengthened some holds but did not establish useful tool behavior. Seeded sampling stays the default. The prepared browser package labels this pair DEVELOPMENT and records each role’s actual lineage; it does not present the union resource budget as equal training for both agents.
