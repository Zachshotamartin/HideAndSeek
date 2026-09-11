# Claude review prompt

Review both repositories together at branch `codex/training-systems-review`:
- https://github.com/Zachshotamartin/LearnToDive/tree/codex/training-systems-review
- https://github.com/Zachshotamartin/HideAndSeek/tree/codex/training-systems-review

This is an independent correctness review of an in-progress research implementation, not a request to endorse it. Read the code, reproduce failures, and challenge the design and evaluation. Treat README claims and previous verification reports as claims to verify. Start with REVIEW_STATUS.md in each repository.

Do not stop, restart, modify, or attach debuggers to running training jobs. Do not alter frozen sources/checkpoints on the SSD, publish models, deploy the website, push changes, or add GitHub Actions. Run bounded tests in isolated output directories. Review first; provide concrete findings and a prioritized repair plan before making changes. Never loosen tolerances, replace golden fixtures, bypass schema checks, or change rewards merely to make tests pass without establishing the correct behavior independently.

## Intended behavior

Diving: one shared policy receives category, apparatus, height, physical state and routine history. It chooses its own legal dive declaration before takeoff and then controls bounded physical joints throughout the dive. It is not assigned a particular somersault/twist target by the trainer. Respect forward/backward/reverse/inward/twisting/armstand categories, legal apparatus, body positions and nonrepeat rules. Difficulty and execution are separate; extra rotations do not earn points if the declared dive fails. Judge the entire water crossing, not just favorable hand contact. No root steering, animation substituted for learned control, unlimited torque, or hidden launch assistance. Automated execution/water approximations must be clearly distinguished from official rules.

Hide-and-seek: two recurrent policies learn continuous movement, grabbing, locking, pushing and bounded jumping in randomized physical environments. Reward remains visibility-only and zero-sum; tool use must emerge when useful, not from bonuses or forced strategies. Seeker is blind during preparation and neither actor receives privileged critic state. Jumping permits mounting objects but must not bypass walls. Finite training episodes are separate from indefinite browser playback.

## Inspect in detail

1. **Diving training_v3 (v10):** environment.py, engine.py, diver.xml, rules.py, difficulty.json, judge.py, policy.py, train.py and run_suite.py. Check official table extraction and category decoding, declaration legality and timing, delayed credit assignment, categorical/motor PPO losses and masks, tanh Jacobian and autoregressive exploration likelihood, value normalization/optimizer rescaling, reward scales and saturation. Look for learned shortcuts, impossible thresholds, state leakage and why execution remains mostly zero. Check armstand support, board preload, joint forces, angular momentum, toe/leg/hand metrics, water-entry ordering and whole-entry completion. Determine whether recovery snapshots are physically reachable and sampled without outcome leakage or target imposition. Check routine/history bookkeeping, legal masks and curriculum distribution.
2. **H&S training_v5:** protocol.py, train_scaled.py, train_entity.py, league_ppo.py, entity_actor.py, env_pool.py, snapshots.py, physics.py, persistent_evaluate.py, evaluate_saved.py and run_suite.py. Check GRU resets/burn-in, active sample masks, opponent sampling, archive diversity and indexing, zero-sum critic/advantage math, PPO likelihoods for tools/jump, visibility/occlusion and partial observability. Inspect grabbing/releasing incentives and whether the environment actually enables useful tools. Audit connected layout generation, barriers and jump limits. Check pursuit/reacquisition diagnostics and memory/tool counterfactual validity.
3. **Evaluation and experiment design:** are seed splits, matched ablations, opponent strength, sample counts, confidence intervals, architecture selection, plateau logic and failure denominators defensible? Distinguish development selection from untouched held-out qualification. Investigate leakage, early-pilot ranking, seed selection, misleading aggregate metrics and evidence for each role separately. A tool-disabled performance difference alone is not proof of intentional tool strategy.
4. **Exact resumability:** independently compare uninterrupted and split runs, including actor/critic/optimizer/RNG, simulator integration, welded objects, button state, recurrent memory, opponent archive, curriculum and routine history. Check snapshot consistency, source/version checks, interruption during evaluation, pilot-to-continuation transfer, completed-budget extension and source dependency portability. Some tests reference local output fixtures; identify what prevents fresh-clone reproduction rather than silently skipping them.
5. **Browser/runtime integration:** inspect src/core, workers/controllers, renderer and asset manifests against their actual exported observation/action/physics contracts. The new v10 native diver is NOT yet wired into the existing browser runtime. Do not confuse a pure inference parity test with complete closed-loop native/WASM parity. Inspect the shared character's world-space planted feet, direction changes, attachment and visual/physical agreement.

## Reproduce known failures first

Both `npm run build` commands passed. `npm test` currently reports:
- LearnToDive: 47/49 pass; native/WASM trajectory parity and first-contact timing fail.
- HideAndSeek: 35/40 pass; live entity dispatch, native/browser physics parity, shipped model validation, blind-seeker recurrent controls and deterministic/sampled reset tests fail.

Read the actual assertions and schemas before deciding whether these are implementation defects, stale fixtures, or both. Do not simply regenerate expected outputs from the implementation under test. Native training's focused tests passed separately, but that does not resolve these browser failures or prove model quality.

For research claims, verify against primary sources, including the World Aquatics rules linked in difficulty.json and OpenAI's Emergent Tool Use paper. This is original project code: do not replace it with OpenAI's implementation.

## Deliverable

Return findings first, sorted by severity. For each: repository and file:line, concrete defect, minimal reproduction/evidence, impact on learning/physics/evaluation/UI, and proposed fix with a regression test. Separate confirmed bugs from hypotheses requiring an experiment. Include commands run, pass/fail counts, missing prerequisites and areas not verified. Finish with an ordered repair plan and a judgment on whether the current runs should continue, be paused for a fix, or be treated as invalid evidence—with specific reasons. Do not claim the agents are well trained based on a screenshot, training return, or a successful build.
