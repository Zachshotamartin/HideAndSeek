# Relational environment v2

Isolated new training contract; accepted browser policies and original `training/` remain unchanged. Native old run stopped at 94,371,840 interactions. Migration retains learned weights with explicitly remapped sensors and fresh optimizers, not a function-preserving or exact continuation claim.

- 135-degree occluded sight with no artificial six-meter cutoff. Ten visible object slots, thirty lidar rays. Hidden entities remain masked; central critic information does not enter the actor.
- Object-to-object multihead self-attention followed by query pooling and 256-unit recurrent memory. Newly introduced attention starts zero-gated during weight migration.
- 96-step preparation / 144-step search. Zero prep reward; otherwise visibility-only zero-sum +/-1. No tool, construction, pursuit or distance incentives.
- Varied shelter/rooms/open arenas, 8–12m during new training, 5–8 boxes including elongated planks and two ramps. All 240 sampled diagnostic layouts had at least three planks and two ramps. Old smaller fixed evaluation layouts are retained as well.
- 80% current/current self-play; 20% history. Fresh frozen pairs every 16 updates. Retain the latest eight plus opponents still serving an episode; saved league states survive resume.
- Full checkpoint every PPO update; immutable milestones about every million interactions. Fixed-reference and no-hider-tools/no-seeker-tools evaluations about every five million. Best-model registry remains separate from latest. One billion new interactions requested, not a claim that this guarantees emergent behavior.
- Resume retains both actor optimizers, critic optimizer, world/opponent/Torch RNG and league. Current worlds restart explicitly at a process boundary, so it does not claim identical uninterrupted physics trajectories.
- Tests cover masked/permutation-invariant observations, long sight, visibility-only rewards, real grabbing/locking, movable barriers changing visibility, ramp climbing, migrated updates, league refresh and checkpoint resume.

Reference: https://arxiv.org/html/1909.07528v2 and https://openai.com/index/emergent-tool-use/ . This is an original smaller 1v1 implementation, not a reproduction of OpenAI's compute scale or learned strategies.

Active run: `output/relational-environment-v2/STATUS.json`. Exact command/PID in `launch.json`. SIGTERM gracefully pauses; rerun recorded command to resume. Additional budgets require a new declared continuation protocol; do not edit an active protocol in place. Training does not automatically replace browser models, because their sensor/encoder contract must be upgraded and validated first.
