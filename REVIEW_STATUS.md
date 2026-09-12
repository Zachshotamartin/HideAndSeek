# Review snapshot: Hide-and-seek

This branch preserves the current source and development browser assets. It is not a model-quality or production-readiness claim. No GitHub Actions were added.

## Checks run before push

- `npm run build`: passed.
- `npm run test:runtime`: passed for trained and initial models in an actual browser, without page errors.
- `npm test`: 44 passed / 0 failed. Updated v4 observation/action fixtures, restored manifest integrity metadata, enforced relational actor-only shapes, and added native policy/jump parity.
- The preceding native training verification passed 9 diving tests and 17 focused hide-and-seek tests, plus a bounded-torque/momentum audit and pure browser-policy inference parity. Native training verification remains separate from browser/model quality.

## Training boundary

Current runs execute from frozen source copies on the owner's SSD. This Git push does not restart them or replace the site's model. Native optimizer/checkpoint archives and transient output directories are not committed. Public model files are the existing development exports. Read the version-specific trainer README for the new training path; the root README describes the existing browser runtime.

New system: `training_v5`, retaining `training` through `training_v4` for provenance. At the status check, the first full-system pilot completed 10,485,760 new interactions; the second had 2,949,120. The first pilot mean fixed-opponent utility was 0.50041. Both role-change confidence intervals included zero. Hider grab/lock benefit was +0.10378 (95% bootstrap interval +0.05122 to +0.15741); seeker grab/lock benefit was -0.00661 (-0.03371 to +0.02035). These are development counterfactual measurements, not proof of an effective general tool-use strategy.

## Review repairs (v5.1, 2026-09-11)

An independent review of the pilots found: the evaluation cohort had silently dropped all 24 long-play maps (seed collision with the v4 cohort), the declared 32-step burn-in never executed (sequence length equalled the horizon), the behaviour-diverse archive was never exercised within a 160-update pilot, the evaluation reference was also a league anchor, promotion needed no statistically clear improvement, preparation depended on absolute paths remembered inside the source checkpoint, and two v5 tests still sent five actions. All of these are repaired in `training_v5` (see its README), the runtime parity fixture was widened from 64 to 528 rows with a mid-rollout reset and blind rows, and the bundled development evaluation evidence is hash-pinned and recomputed in `npm test`. The v5 pilots (two full-system seeds) are invalid as ablation evidence and the suite was stopped and relaunched from the same frozen v4 source with v5.1. Nothing here is a model-quality claim.

## Source restyle (2026-09-11)

The v5 trainer files touched by the review repairs (`train_entity.py`, `train_scaled.py`, `env_pool.py`, `persistent_evaluate.py`, `evaluate_saved.py`, `run_suite.py`, `protocol.py`, `snapshots.py`, `smoke.py` and the focused tests) were rewritten from the dense one-statement-per-line convention into ordinary readable Python without changing behaviour: the 290-line `train()` became a `Trainer` class with one method per phase, the controller became a `Controller` class, and the pool worker, evaluator and suite were split into named functions. `physics.py`, the legacy v1 to v4 modules and `scripts/generate-runtime-fixtures.py` (whose own hash is embedded in the shipped parity fixtures) are untouched. Behaviour was verified bit-exactly against references captured before the rewrite: a two-update synthetic training run (both actors, both optimizers, the central critic, the training log excluding wall-clock timings, the full rollout state, league descriptors and decision counters identical), twelve evaluation episodes across four counterfactual modes with their bootstrap contrasts, and the protocol functions (JSON identical). Training sources are hashed into every checkpoint's provenance, so checkpoints written by the earlier dense source (including the live SSD run, which executes from its own frozen copy) will not exact-resume under this source; that is the contract working as intended.

## v5.3 capture rule, promotion repair and relaunch (2026-09-12)

Fifteen v5.2 pilots finished. The full system scored 0.50 to 0.51 against the migrated reference at 10.5M interactions; the short-episode ablation scored best on the map-averaged utility and its seed-109310 pair was published to the site on 2026-09-11. Broken down by play length that pair is worse than the reference seeker on every 30 s and 60 s map (seeker change -0.16 and -0.21) and loses sight without recovering in 62% of long episodes; the map average hid this because 144 of 168 maps are 11.5 s episodes. Two repairs: promotion and pilot ranking now use a length-balanced utility with a per-length regression guard, and a capture rule (`capture.py`, physics untouched) ends play when the seeker reaches the hider in sight, crediting the remaining steps, so the seeker is rewarded for closing rather than for keeping a distant line of sight. The suite was stopped and relaunched on the full system only, three seeds, then continuation. The published pair is a development export and is not qualified.
