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

## Handoff verification (2026-09-12, 17:42 UTC)

The v5.3 full-system continuation is alive at 53,805,056 interactions. Latest completed evidence: `full-seed-109312/evaluations/52428800/evaluation.json` under `2026-09-12-v12-v5.3`. Relative visibility utility changes are hider +0.0596 (95% interval -0.0074 to +0.1265) and seeker -0.0405 (-0.0924 to +0.0104); neither establishes an improvement. Hider grab/lock benefit is +0.0791 (+0.0277 to +0.1297); seeker benefit remains inconclusive.

The hider's mean stuck frames are 20.36 versus the reference's 14.95 for 144-step play, and 149.25 versus 56.25 for 750-step play. Candidate seeker capture rates are 53.5% versus 43.8% on 144-step maps, equal at 87.5% on 375-step maps, and 87.5% versus 75% on 750-step maps. Thus this latest evaluation differs from the earlier handoff's long-map capture report; better capture in some cohorts does not establish better overall visibility utility or navigation. Any proposed stuck cost should distinguish attempted movement blocked by contact from deliberately staying still in useful cover. No learning rules, physics, checkpoints, running processes or deployment were changed.

## Controlled training improvements (2026-09-12)

35 focused native regression tests passed, including exact split/resumed training with curriculum sampling, blocked cost and repeated episode resets. Physics was not edited. Alternative learning settings are separate comparison arms; defaults preserve the preceding objective. Existing frozen training runs and published assets were left untouched. These checks establish implementation behavior, not model-quality improvement. Read the versioned trainer README for comparison commands and limitations.

Controlled comparisons launched in `/Volumes/Zach's SSD/PortfolioTraining/2026-09-12-controlled-comparisons`; immutable source/input hashes and controller commands are recorded there. Three paired seeds, no automatic promotion or monitoring. The first active arms are continuous-entry feedback (diving) and outcome-based opponent sampling (hide-and-seek). The original v12/v5.3 controllers remain untouched. Current test evidence: 29 diving native tests, 35 hide-and-seek native tests; the hide-and-seek evaluator smoke completed 48 episodes across four play lengths. These are implementation checks, not evidence that the new models are better.

## Additional research changes (2026-09-12)

PopArt residual-value normalization, larger fresh batches, outcome-progress map replay, and discounted-equivalent capture credit are implemented. Defaults retain the previous behavior; each new setting is explicit, checkpointed, and included in the comparison contract. See the versioned README for mechanisms, evidence and limitations. No new model is qualified and no browser assets were replaced. Existing frozen runs remain untouched.

The research suites are running from `/Volumes/Zach's SSD/PortfolioTraining/2026-09-12-research-training-options` with three matched seeds, six arms per project and source/input hash manifests. Verified implementation evidence is in `review-evidence/2026-09-12/research-options/`; 74 native tests passed across both projects. No result has been selected for publication.

Additional assessment changes (2026-09-12): current six-action mean/sample comparisons, controlled-target pursuit/obstacle/reacquisition diagnostics, a matched fresh-versus-inherited actor suite with fresh central critics, and paired training-seed confidence/noninferiority reporting. All are local changes; running frozen suites and browser assets are unchanged. No additional long comparison was launched.

Final assessment verification: 70 tests passed. Evidence: `review-evidence/2026-09-12/assessment-and-selection/`. Saved-model native assessments completed; these verify mechanics and report behavior, not improved trained performance.

## v6 tag rounds (2026-09-15)

The v5.3 game was replaced by tag rounds with a clock (see `training_v5/README.md`, "v6 tag rounds"): a seeker within reach and in sight ends the round, the hider wins by surviving the clock, preparation grows with the play length, and actors observe the remembered last sighting, its age, the time remaining and their own coherent exploration noise (input width 220, schema `tag-rounds-last-seen-214-v6`). The source pair is widened with zero weights so training starts from its exact behaviour. Sequences are 256 steps with 64-step burn-in; seeker matchmaking, PopArt, progress map replay and the discounted capture credit are on; a staged arena curriculum is gated on the measured seeker find rate; a gate at 52.4M interactions stops the run for review if the seeker does not find the hider. `physics.py` is untouched. The browser runtime plays timed rounds for a v6 pair and keeps the earlier formats unchanged; the shipped model file is unchanged until a v6 pair has been evaluated and exported.

Checks run: 22 focused Python test modules (`test_game`, `test_stages`, `test_widen_schema`, `test_coherent_noise` added; `test_saved_workflow` included), `npm test` 53 passed / 0 failed (`tests/tagRounds.test.js` added with a native fixture of 528 policy rows and a 64-step physics rollout with sightings and a tag). Codex's known-position run (agents told the opponent's position) was stopped and is not continued; the relaunch is recorded in the run folder's `LAUNCH.json`.
