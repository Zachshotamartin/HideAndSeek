# Hide-and-seek training v5.1

This fork retains the v4 physical game, jump limits and restricted entity/ray observations. Reward is visibility-only, zero-sum, with zero preparation reward. Grabbing, locking, jumping and tool positioning are not rewarded or scripted.

v5.3 adds a capture rule outside the physics (`capture.py`): during play, a seeker that sees the hider from within 0.7 m ends the episode and is credited with every remaining step as seen, the hider with the same steps as found. The reward is still the zero-sum visibility reward and `physics.py` is byte-identical; the rule exists because a visibility-only seeker was right to keep its distance as long as it could see the hider, which is not seeking. Promotion now uses a length-balanced utility (each play length counts once, so 144 short maps cannot outvote 24 long ones) and any clear regression at a play length blocks it; the pair published on 2026-09-11 came from the short-episode ablation and lost every 30 s and 60 s map for that reason. Evaluations also report capture rate and stuck frames per role.

Changes versus v4: connected randomized rooms/corridors/multi-exit layouts, varied boxes and ramps, 15/30/60-second play episodes, 70% current self-play with active-sample-balanced historical opponents, fixed anchor/recent/behavior-diverse archive, 128-step recurrent sequences with a real 32-step burn-in that re-runs the recurrence through the previous rollout tail (v5 declared burn-in but never executed it because its sequence length equalled the horizon). The existing 256-unit recurrent entity policy is retained; a larger network is not assumed to fix strategy learning.

```
python training_v5/run_suite.py --source /absolute/best-evaluated.pt --cohort /absolute/cohort.json \
    --history --heldout /absolute/earlier-parent.pt --output /mounted/ssd/hide-seek-run
```

Every external input is explicit. `--cohort` is the fixed development map list; the controller extends it with 24 long-play construction maps on fresh seeds and refuses seed collisions instead of silently dropping them (the v5 pilots evaluated only 11.5 s episodes for that reason). `--history` lists frozen league anchors (an empty list keeps only the migrated reference); `--heldout` is an evaluation opponent that never enters the league, so improvement is measured against something the learner did not train against. Missing files fail loudly; nothing is read from paths remembered inside a checkpoint.

The suite runs matched seeds for full, baseline, short episodes, unbalanced roles, recent-only history and short memory. Each pilot runs 160 PPO updates (10,485,760 interactions); opponents are snapshotted every 4 updates against a 16-entry archive limit, so every pilot exercises archive truncation and diversity selection (the controller refuses pilots too short to do so). It then exactly continues the median full-system seed to a default budget of 1,048,576,000 new interactions. All arms inherit identical source actor/critic weights and start fresh optimizers; inherited experience is not relabeled new training.

Full checkpoints save simulator integration state, mutable equality/weld data, object/button states, each environment RNG, both actor GRUs, the burn-in prefixes, optimizer state and opponent archive. SIGTERM the suite PID to pause, then repeat the same command to resume. Changed training source is rejected during exact resume. Full snapshots are retained around every million interactions; latest is saved each update.

Evaluation uses fixed disjoint seeds, both role matchups, the held-out opponent, tool-disabled, stateless (memory zeroed every step) and observation-blackout counterfactuals, visibility, pursuit and reacquisition diagnostics, and per-play-length breakdowns. A tool-disabled difference is evidence to inspect, not proof of intelligent tool use; the stateless conditions measure how much the policy breaks without its recurrence, not what it remembers. A checkpoint is promoted to `best-evaluated.pt` only with a statistically clear improvement in at least one role and no clear regression. Browser playback remains indefinite; finite training episodes do not change the browser time limit.

Scale context: OpenAI's emergent tool use paper reports fort building after roughly 25 million episodes; a pilot here holds about ten thousand episodes and the full budget on the order of a million, so paper-like phases are not expected from these runs.

No automatic model publication and no GitHub Actions are added. The native controller schedules training/evaluation without any agent monitoring process.

Module map: `protocol.py` (training distribution and archive selection), `league_ppo.py` (grouped acting against frozen opponents and the recurrent PPO update), `env_pool.py` (the spawn-worker physics pool), `snapshots.py` (exact world snapshots), `train_entity.py` (the `Trainer` class: validated inputs, rollout, updates, league upkeep and checkpoints), `train_scaled.py` (input preparation and the milestone `Controller`), `evaluate_saved.py` and `persistent_evaluate.py` (fixed-opponent evaluation and its episode measurements), `evaluated_models.py` (the promotion registry) and `run_suite.py` (the matched-seed suite). `physics.py` is the byte-identical v4 physical contract and is never edited.

Focused tests (fresh-clone reproducible, synthetic pairs in temporary directories): `test_protocol.py`, `test_tool_jump.py`, `test_league.py`, `test_snapshots.py`, `test_exact_training.py`, `test_evaluation.py`, `test_env_pool.py`, `test_layouts.py`; `smoke.py` runs one synthetic update. Run them with `python -m unittest discover -s training_v5 -p 'test_<name>.py'`. Older `test_*` modules inherited from the v1 to v4 trainers still assume five actions or three movement outputs and are not part of the v5.1 verification claim.

## Controlled seeker and contact comparisons (2026-09-12)

`--matchmaking seeker-curriculum` changes only the historical hider chosen for episodes in which the current seeker faces a frozen opponent. Current/current and role-balancing probabilities remain unchanged. Historical opponents are weighted toward intermediate observed capture success, with a 25% uniform mixture preserving exposure to hard and easy opponents. Unknown opponents start at 50%; each completed match updates an exponential average. Counts, averages, opponent identities and RNG state survive exact checkpoints and archive pruning. No evaluation results are used to choose training opponents.

`--blocked-cost 0.02` is a separate, optional zero-sum effort-cost ablation. After preparation, futile requested motion into a wall or stationary prop transfers 0.02 reward to the other role per decision. Stationary hiding, movement along a wall, and pushing an object that moves are excluded. Both actors blocked together yields zero net transfer. This changes training incentives and is deliberately not the default or silently included in reported visibility/capture evaluation. Physics and observations remain byte-identical.

The evaluator now aggregates reacquisition time, unresolved losses of sight, distance progress after visibility, retreat-distance frames, capture time and contact-aware blocked frames. Distance changes also reflect opponent motion; these measurements do not prove what an actor intended.

`compare_curriculum.py --output NEW_FOLDER --source CHECKPOINT --cohort COHORT --heldout OPPONENT --history ANCHORS...` runs baseline, outcome sampling, and blocked effort separately on the same three seeds and parent actor. Each arm uses 32 environments, one physics worker, 5,242,880 interactions and unchanged architecture. All arms use the same fixed evaluation maps, opponents, capture rules and length breakdowns. Source/input hashes are recorded; SIGTERM saves the active training child, and the identical command resumes it. Compare all seeds and both roles; no automatic selection, deployment or longer continuation occurs.

## Research training options (2026-09-12)

| Option | Behavior | Default |
| --- | --- | --- |
| `--value-normalization popart` | Running target normalization, affine preservation of the residual value prediction, and rescaled optimizer moments. GAE always receives raw-unit values. | `none` |
| `--map-replay progress` | Revisit training maps by outcome progress, uncertainty and staleness; 50% fresh maps and balanced family/play-length sampling among archived maps. | `random` |
| `--capture-credit discounted-equivalent` | Replace only the terminal future-credit lump with the discounted sum of remaining visibility rewards. | `immediate` |

Value normalization has a distinct critic schema. Ordinary residual critics can initialize normalized critics without changing predictions; exact resumes restore normalizer and optimizer state. Metrics report value MSE and explained variance separately for each play length. No privileged critic inputs enter the actor.

Map replay collects fresh on-policy actions in each revisited world. Its bounded 384-map archive uses only training seeds below 2^30, keeps diverse family/play strata, and is saved in full alongside RNG state. This is an outcome-progress sampler inspired by curriculum research, **not** PLR's exact TD-error estimator. Changes in opponents also affect its outcome estimates; paired-seed evaluation is necessary.

Capture remains an observed in-reach, visible event under the existing game rule. The training-only discounted option credits sum(gamma^k, k=1..remaining), not the undiscounted number of remaining steps. At gamma .998 and 750 future steps this is 387.825 rather than 750. The current-step reward is retained, both roles remain zero-sum, and evaluation continues to report original capture/visibility outcomes. The immediate option is still available as an explicit stronger capture objective.

`compare_curriculum.py --suite research --output NEW_FOLDER --source PARENT --cohort COHORT --heldout OPPONENT --history ANCHORS...` runs combined, baseline, PopArt-only, map-only, capture-only and large-batch arms across three matched seeds, at 5,242,880 role interactions per arm. The combined arm also uses the earlier outcome-based opponent sampler; standalone arms isolate the new mechanisms.

The large-batch arms collect 128 x 256 = 32,768 environment transitions (65,536 role interactions), four times the baseline. Their optimizer uses 128 recurrent sequences per batch rather than 32. Opponent snapshots remain separated by the same amount of experience: every large update versus every four baseline updates. Budgets count interactions, not updates. No off-policy trajectory replay, extra reward for tool use, or scripted strategy is added.

The launcher records batch units and all objective settings, preserves protocols across resume, checkpoints every 1,048,576 interactions, and terminates its own evaluator process group on pause. Evaluation can resume from its immutable completed-map records. It never changes older frozen runs or publishes models.

Verification: 41 focused native tests pass, including combined-option exact resume after real world resets, normalizer prediction invariance, capture-credit arithmetic, map/RNG restoration and protocol idempotence. The combined 128-world smoke run completed 65,536 interactions with finite optimization statistics and zero physics divergences. This does not establish policy improvement.

Research foundations: [MAPPO](https://arxiv.org/abs/2103.01955), [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html), [original hide-and-seek study](https://arxiv.org/html/1909.07528v1). The paper's batch units are ten-transition chunks and must not be confused with this trainer's transition or role-interaction counts.

### Behavioral assessment and initialization comparisons (2026-09-12)

`python training_v5/assess_behavior.py --checkpoint IMMUTABLE.pt --output NEW_REPORT.json` evaluates current six-action entity policies, including jump and persistent grab/lock commands. Mean/sample role comparisons keep the other actor sampled; the pair-mean condition assesses deterministic playback. Conditions share independent role random tapes on the same maps. The tool records checkpoint/source hashes and never trains or publishes a model. This supersedes the old five-action `evaluate_policy_modes.py` for current checkpoints.

Three evaluation-only scenes isolate pursuit of an initially visible stationary target, navigation around a wall to an occluded target, and reacquisition after a target physically moves behind a wall. Target movement uses the normal bounded action/physics interface, not teleportation. Only the seeker is assessed in these controlled-target scenes. They are not training curricula or evidence of learned hider quality. If sight loss never occurs, reacquisition is marked unexercised rather than a success. Ordinary learned-pair games remain separate. The diagnostic seed range is reserved from training and development-ranking maps; once used for debugging, these seeds are not untouched final-test evidence.

`python training_v5/compare_curriculum.py --suite initialization --source PARENT.pt --cohort COHORT.json --heldout HELDOUT.pt --history ANCHORS... --output NEW_FOLDER` compares fresh versus inherited actors under the same combined training settings, equal new-interaction budgets, seeds, freshly initialized central critics/optimizers, and identical experienced frozen league anchors. This isolates actor initialization; the fresh arm is not learning in a world without experienced opponents. Actual inherited resource lineage differs. Prepared assets are immutable, hashed, and resume-checked.

New comparison runs default to five independent training seeds. Results bootstrap paired differences across training seeds, never pool their episodes as independent training runs. Confidence intervals adjust for the declared arm/metric comparisons. Eligibility requires a meaningful score improvement and noninferiority for both roles. Fewer than five seeds are preliminary; five is a minimum screening rule, not a power guarantee. Existing length-specific fixed-opponent reports and promotion safeguards remain available. `comparison_summary.py --results RESULTS.json --baseline BASELINE_ARM --output NEW_REPORT.json` supports both original dictionary and new list result formats. No tool automatically deploys weights; final unused holdout evaluation is still required.

Existing frozen suites keep their original code and three-seed plans. The new initialization suite has not been launched alongside them.

The Node continuation launchers now accept `--trainer-version v5` explicitly; their legacy default remains available for the old physical schema. Lifecycle regression fixes cover six-action buffers, mixed-encoder opponent archives, and exact world restoration. New checkpoints store encoder identities for every saved league pair; older full-state checkpoints can recover the two known types from their native parameter keys.

## v6 tag rounds (2026-09-15)

The audit of the v5.3 runs found two stalemates that the visibility-only game cannot resolve: a seeker that has cornered the hider keeps a sight line and neither agent can change the outcome, and a hider that has escaped once is chased at equal speed forever. The v6 game is hide-and-seek with a tag and a clock, and the actors are given the memory a real player has. Physics (`physics.py`) is byte-identical; every rule lives beside it.

- **Game (`game.py`, `capture.py`, `protocol.py`)**: a seeker within 0.7 m and in sight of the hider during play tags it; the round ends and the remaining play is credited at its discounted equivalent (`--capture-credit discounted-equivalent`). The hider wins by surviving the clock. Preparation is `max(96, 0.4 × play)` steps (7.7 s for 15 s rounds, 12 s for 30 s, 24 s for 60 s). The per-step visibility reward is unchanged.
- **Observation schema `tag-rounds-last-seen-214-v6`**: the 208 physical measurements plus six extras per agent: a last-seen flag, the remembered opponent offset in the agent's own frame (x, y over 6 m, z over 2 m), the age of that memory (saturating at 10 s) and the play time remaining (over 60 s). Nothing tells an agent where an unseen opponent is now. Workers return this 214-value observation (`PhysicsEnvPool(observation='game')`); every actor, current or frozen, takes the prefix its own schema defines, so the 210-input v4/v5 pairs still play in the league and in evaluation.
- **Coherent exploration**: the actor observes its own AR(1) exploration noise (`--noise-rho 0.7`, four values after the two buttons, input width 220). The movement distribution is `N(head + ρ·noise, σ·sqrt(1−ρ²))`, so the recorded likelihood stays exact for PPO while exploration is correlated over about a quarter of a second instead of being white at 12.5 Hz. Deterministic playback is the head's own mean.
- **Schema migration (`widen_schema.py`)**: the source pair is widened with zero weights on the new inputs and the button columns moved, so training starts from exactly the source's behaviour (`test_widen_schema.py` checks bit-exact outputs).
- **Sequences**: 256-step sequences with a 64-step burn-in through the previous rollout (was 128/32).
- **Enabled options**: seeker matchmaking, PopArt value normalisation, progress-based map replay and the discounted capture credit, all previously implemented behind flags and now on in `train_scaled.py`.
- **Curriculum (`stages.py`)**: three arena stages (8 to 9 m with 15 s rounds; up to 10 m with 15/30 s; the full 8 to 12 m, 15/30/60 s distribution). A stage opens once the seeker has seen the hider at least once in 80% of the last 256 rounds after at least 2,048 rounds at the stage. The stage and its evidence are in every checkpoint and resume exactly.
- **Gate**: at 52,428,800 interactions the candidate seeker must see the hider in at least half of the evaluation rounds and neither role may show a statistically clear regression, or the controller stops with phase `gate-failed-awaiting-review` (the suite records `gateFailedTrial`).
- **Variants**: `slow-seeker` scales the seeker's movement force by 0.85 outside the physics as an asymmetry ablation for evidence only; it cannot be shipped because the browser physics has no such scale.
- **Evaluation**: episodes report `seekerFound`, `hiderEscaped`, `foundRate`, `escapeRate` and `meanFirstSightSeconds`; `evaluated_models.utility` reports `seekerFoundRate` alongside the unchanged length-balanced score.
- **Browser**: `src/core/gameRules.js` mirrors the rules and extras; `relationalPolicy.js` reads any actor width and carries the noise state; `index.js` plays timed 30 s rounds for a v6 pair (a tag or the clock ends the round and the next starts on its own). `scripts/export-tag-rounds-preview.py` exports an eligible registry pair; `scripts/generate-tag-fixtures.py` writes `tests/fixtures/tag-rounds-native.json` for `tests/tagRounds.test.js`.

Run (three seeds of the full system for 240 updates, then the median seed continues):

```
python training_v5/run_suite.py --output <folder> --source <v4 or v5 pair> --cohort <cohort.json> --history --heldout <parent.pt> --variants full --seeds 109310 109311 109312 --pilot-updates 240 --total-steps 1048576000
```

Tests, from `training_v5/`: `python -m unittest test_game test_stages test_widen_schema test_coherent_noise test_capture test_registry test_protocol test_tool_jump test_league test_snapshots test_exact_training test_evaluation test_env_pool test_layouts test_entity_actor test_entity_training test_learning_options test_widen_policy test_saved_workflow test_comparison_controller test_curriculum test_assessment`; `npm test` (53). `test_widen_policy.test_long_sequence_updates_are_finite` predates the jump action (five raw values) and fails at HEAD as before.

Nothing here is a model-quality claim; the run's evaluations decide that.
