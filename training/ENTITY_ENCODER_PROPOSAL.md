# Proposed object-set encoder comparison

The isolated implementation has passed its initial checks and the matched learning comparison is now running. The proposal below records the design and acceptance boundaries; [ENTITY_ENCODER_RESULTS.md](ENTITY_ENCODER_RESULTS.md) records the measured starting behavior, critic check, throughput, and declared learning protocol. Public assets remain the selected development pair, native selection SHA-256 `ca9462e9cd3f084a2df687defc4de23513f15df700826a3a4524d0676bc74117`.

## Evidence and the limited hypothesis

The retained seeker reduced complete misses from 25 to 16 on the repeated 96-map cohort. By scenario, the original/selected misses were 3/1 in open arenas, 5/3 in shelters, and 17/12 in rooms. All 16 selected misses occurred on maps with at least two props, but 84 of the 96 maps were in that category; the four zero-prop and eight one-prop maps are too few to establish a clutter effect.

The earlier read-only probes found sensitivity to swapping the same visible object records: mean seeker distribution KL 0.144, with mean movement-vector RMS change 0.095. However, most frames in the outcome-matched room audit exposed zero or one object. Four of its 20 failed searches had no persistent-object slot changes, and pure rank swaps occurred on only 0.34% of failed frames. These were earlier diagnostic checkpoints, not a causal intervention on the selected seeker. Slot ordering cannot explain every failure.

Recurrent reconstruction matched stored states exactly. Memory survives chunk boundaries; measured update-to-update staleness was small. Useful preparation still has nonzero delayed credit, though its variance may be high. The 32M continuation did not reliably improve on the 16M seeker. Mean actions lengthened a small number of grips but failed to establish useful tool benefit. This supports testing a representation change, while preserving the physical game and scoring. It does not establish that this particular change will solve room search or produce construction.

The testable hypothesis is narrower: sharing object processing can reduce dependence on transient slot assignment and transfer learned object relationships across arrangements. The study’s entity-aware processing is a reference, not an imported implementation. [Baker et al., Appendix B](https://arxiv.org/html/1909.07528v2#A2)

## Exact unchanged input boundary

The actor still receives exactly 140 values. The present native schema is:

| Indices | Permitted measurements |
| --- | --- |
| 0–9 | Own local velocity, yaw rate, height, phase/time, role, and grip/lock indicators |
| 10–17 | One opponent’s visible measurements, with visibility mask |
| 18–113 | Six visible object records of 16 values each |
| 114–137 | Twenty-four body-relative solid-geometry range rays |
| 138–139 | Own requested grab/lock button states |

Object records contain visibility, local relative position/velocity, relative yaw sine/cosine, dimensions, ramp indicator, observed lock ownership and own grip association. They do not contain an instruction about what action to take. The native sensor continues selecting at most six currently visible objects. No world-coordinate map, absolute horizontal agent position, arena-size input, unseen object, hidden opponent coordinate, subgoal, or privileged critic feature is added. The network itself becomes insensitive to permutation of those six permitted records.

## Original encoder design

Keep the 64-unit GRU, 96-value recurrent input, Gaussian movement distribution, categorical Keep/Press/Release commands, button feedback, and exact action-density/entropy calculation.

Replace only the 140→96 linear encoder before its existing tanh:

1. A 44→96 linear branch processes the existing self, opponent, lidar, and button features in their original order.
2. A single shared 16→96 linear kernel processes each visible object. Sum its six masked contributions. It has no slot-specific weights or slot embeddings.
3. A shared 16→32→32 nonlinear object network produces entity features. A query from the same 44 permitted fixed features attends to shared 32-value keys/values. A zero-value null token makes empty sets finite. A 32→96 residual projection adds the pooled result before the original tanh.

Hidden payloads are masked before embedding, and their embeddings/attention entries are masked again so learned biases cannot create phantom objects. The opponent branch uses its existing visibility mask as well; valid native observations are unchanged. The null token contributes zero when no objects are visible. Counts arise only from the already permitted masks. No object ID, transport slot index, positional embedding, semantic target, or strategy prior enters the network.

The proposed encoder has approximately 14,176 parameters versus 13,536 in the current encoder. That estimate includes the 96-value bias and all query/key/value/residual projections. Model capacity is therefore close; the actual full-rollout CPU benchmark still determines whether the implementation is practical. GRU and controller dimensions do not grow.

## Initialization and comparison control

Both arms start from the exact selected hider and seeker, not from the weaker latest pair or a new random locomotion model.

- **Control:** original encoder and exact selected actor parameters.
- **Set encoder:** copy the original fixed-feature columns and bias; initialize the shared object kernel to the average of the six old 16-column kernels; retain GRU, movement/tool heads, log standard deviation, and old partial-value head. Initialize the new residual output to zero.

This is a projection of our own parameters, not imitation or demonstrations. It necessarily changes the initial policy when objects are visible. Histories with no visible objects and valid masked opponent inputs must reproduce the old policy within numerical precision. The zero-update projected actor must be measured and named as a warm start, never as an untrained reference.

Reset actor Adam state in **both** arms, recording that decision. This prevents comparing a fresh optimizer on new shared parameters with inherited moments on the control. Save both zero-update states and RNGs. Also save a genuine zero-experience version of the new architecture for reference, but do not spend a matched learning budget on a separate from-scratch arm unless the warm start shows a specific harmful transfer problem.

Use the same inherited training-only critic initialization and its documented architecture in both arms. The partial-value baseline and detached pre-action actor memory must come from the actual actor selected for that row. The existing masked policy objective and no-gradient boundary remain unchanged. Do not silently declare an inherited critic calibrated to the projected actor. First assess complete-game value errors against the partial-pair and time-conditioned baselines within each arm. If recalibration is necessary, use the established protocol with thousands of distinct whole training maps and separate map-level selection/assessment, count those decisions separately, and give both arms matched treatment. Do not fit many epochs on another small collection of reused maps.

## Pre-training gates

Before allocating a learning phase:

- Test all 720 permutations of six object slots on full, partial, empty, near-tie, and repeated-value cases. Logits, recurrent states and values must agree up to explicitly bounded float32 reduction error. Check gradient permutation equivariance too; compare shared-parameter gradients rather than gradients associated with a particular input slot.
- Perturb finite payloads behind object/opponent visibility masks and require unchanged outputs. Exercise actual environment hidden-position perturbations and blind preparation. Unseen entities must not affect pooling through biases or the null token.
- Verify the no-visible-object history warm-start equivalence, multi-step requested buttons, terminal reset, deterministic decisions and seeded uniform-tape decisions. Near an argmax/CDF boundary, report numerical ambiguity rather than concealing it with a large action tolerance.
- Reuse the existing mixed historical/current row and all-frozen-minibatch tests. Historical legacy actors and new actors must each advance their own memory and requested state, with identities changing only at terminal reset.
- Benchmark the full 128-world, eight-worker rollout/update path. Report actor time, environment time, optimizer time and interactions per second for both arms. The completed legacy phase averaged about 7,760 interactions/second under shared-machine load; do not assume a forward-only benchmark predicts training throughput.

No physics file, observation builder, reward, tool action, episode length, gamma/lambda, BPTT length, or browser asset changes are needed for these gates.

## Proposed bounded learning phase

After those gates and review, the approved phase is **16M fresh total interactions per arm**, 32M combined, using the reviewed historical-opponent league and unchanged optimization settings. The exact rounded budget is 15,990,784 per arm. This is a comparison phase, not a permanent training cap or a promised amount of experience sufficient for emergence. The full benchmark predicts roughly 38 minutes for legacy and 46 minutes for the entity arm, plus evaluation. Actual concurrent work and training throughput are reported separately.

Use one explicitly paired world/opponent seed for this first bounded comparison, and state that it does not establish robustness across training seeds. Both arms use the same scenario, size/count distribution and opponent mixture. Keep current, historical, preparation and active samples separate. Save latest atomically, retain immutable milestones, and preserve the exact selected browser pair throughout. Rewards remain solely the original visibility outcome. No pursuit reward, tool bonus, forced hold, hand-selected shelter, expert action or scripted opponent is introduced.

Assess zero-update, 8M and 16M checkpoints on a fixed development cohort declared before training. Continue reporting the original 96 maps for continuity, but use a new 96-map development cohort for the principal architecture comparison so the old selection set is not mistaken for fresh evidence. Keep final-test seeds at 1.9 billion untouched. Compare each role with the same selected frozen counterpart, plus the existing fixed initial/warm/intermediate references. Report paired objective changes, complete misses and scenario/count breakdowns. Use the actual sampled mode; retain a separate mean-mode comparison for any selected candidate rather than changing both architecture and runtime default at once.

A credible role regression blocks replacement of that role. If neither arm beats the selected reference, preserve the reference and inspect the new representation/trajectory evidence before deciding on further training. If the new encoder appears better, repeat with an independent paired training seed before attributing the gain broadly to architecture. Inconclusive small differences do not establish equivalence. Useful tool claims still require the chosen pair’s own paired physical interventions and recorded beneficial effects, not attention weights, grip length or counts. No automatic promotion occurs.

## Export and browser path, only for an assessed selection

Keep new training files isolated, for example `entity_actor.py` and `train_entity.py`. Export actor parameters only. Add a new explicit format with an encoder tag per role so an improved entity seeker can be paired with the exact retained legacy hider without converting or retraining that hider. A type-specific parameter whitelist must reject critic keys and malformed shapes. Existing persistent and binary imports remain explicit and unchanged.

Implement the small shared embedding, masks, pooling, attention and recurrent operations independently in JavaScript. Validate actual native/JavaScript multi-episode trajectories with common uniform tapes, including permutations, empty sets, preparation and resets. Model weights remain content-addressed lazy assets; the mount/dispose API, physical sensors, actual controls and replay format remain stable. Any new pair receives its own measured report and localhost review before public model selection. No current asset or runtime change is part of this proposal.
