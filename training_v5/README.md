# Hide-and-seek training v5.1

This fork retains the v4 physical game, jump limits and restricted entity/ray observations. Reward is visibility-only, zero-sum, with zero preparation reward. Grabbing, locking, jumping and tool positioning are not rewarded or scripted.

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

Focused tests (fresh-clone reproducible, synthetic pairs in temporary directories): `test_protocol.py`, `test_tool_jump.py`, `test_league.py`, `test_snapshots.py`, `test_exact_training.py`, `test_evaluation.py`, `test_env_pool.py`, `test_layouts.py`; `smoke.py` runs one synthetic update. Run them with `python -m unittest discover -s training_v5 -p 'test_<name>.py'`. Older `test_*` modules inherited from the v1 to v4 trainers still assume five actions or three movement outputs and are not part of the v5.1 verification claim.
