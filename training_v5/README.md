# Hide-and-seek training v5

This fork retains the v4 physical game, jump limits and restricted entity/ray observations. Reward is visibility-only, zero-sum, with zero preparation reward. Grabbing, locking, jumping and tool positioning are not rewarded or scripted.

Changes: connected randomized rooms/corridors/multi-exit layouts, varied boxes and ramps, 15/30/60-second play episodes, 70% current self-play with active-sample-balanced historical opponents, fixed anchors/recent opponents/behavior-diverse archive, 256-step recurrent sequences and 32-step burn-in. The existing 256-unit recurrent entity policy is retained; a larger network is not assumed to fix strategy learning.

`python training_v5/run_suite.py --source /absolute/best-evaluated.pt --output /mounted/ssd/hide-seek-run`

The suite runs matched seeds for full, baseline, short episodes, unbalanced roles, recent-only history and short memory. Each pilot runs 160 PPO updates (10,485,760 interactions), long enough to exercise archive retention. It then exactly continues the median full-system seed to a default budget of 1,048,576,000 new interactions. This is a comparison of training components, not a new architecture claim. All arms inherit identical source actor/critic weights and start fresh optimizers; inherited experience is not relabeled new training.

Full checkpoints save simulator integration state, mutable equality/weld data, object/button states, each environment RNG, both actor GRUs, optimizer state and opponent archive. SIGTERM the suite PID to pause, then repeat the same command to resume. Changed training source is rejected during exact resume. Full snapshots are retained around every million interactions; latest is saved each update. Evaluated best pairs use fixed-opponent utility with role-regression guards.

Evaluation uses fixed disjoint seeds, both role matchups, tool-disabled and memory-reset counterfactuals, visibility, pursuit and reacquisition diagnostics. A tool-disabled difference is evidence to inspect, not proof of intelligent tool use. Additional scripted occlusion probes and visual review remain qualification work. Browser playback remains indefinite; finite training episodes do not change the browser time limit.

No automatic model publication and no GitHub Actions are added. The native controller schedules training/evaluation without any agent monitoring process.

Focused tests: `test_protocol.py`, `test_tool_jump.py`, `test_league.py`, `test_snapshots.py`, `test_exact_training.py`, `test_evaluation.py`. Some inherited legacy diagnostic scripts require their original fixtures and are not part of the v5 verification claim.
