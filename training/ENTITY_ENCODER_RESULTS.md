# Isolated object-encoder comparison

The current browser pair is unchanged. This work tests one representation change against an equally optimizer-reset reference; it does not establish better agents or useful construction. Both arms are warm starts from our own selected actors. No research implementation, released research weights, demonstrations, strategy rewards, or scripted opponent is used.

## Implementation and numerical checks

The object encoder uses one shared linear kernel and masked attention over the same six visible object records. The 140 restricted measurements, GRU, action distributions, persistent buttons, physical game and visibility-only reward are unchanged. `training/entity_actor.py` implements the actor; `src/core/entityPolicy.js` is an isolated inference module that the current browser entry does not import. It also dispatches existing persistent and binary formats through their unchanged controllers. The new typed pair format permits a retained legacy role beside an entity role.

- Seven native encoder tests cover all 720 object permutations, live nonlinear attention, empty/partial/full sets, duplicate/near-tied inputs, shared-gradient equivariance, hidden-payload masking, exact compatible-weight copying, no-object recurrent histories, frozen historical rows, and blind preparation. An actual MuJoCo hidden-opponent position perturbation leaves both permitted sensors and actor outputs unchanged.
- Two native trainer tests exercise actual MuJoCo terminal boundaries, atomic checkpointing, restored optimizer progress, explicit world/memory restart semantics, immutable protocol guards, and equal fresh critic optimizers.
- Four JavaScript tests cover permutations, masking, stateful mixed-role actions and resets, strict model validation, and unchanged legacy imports.
- Three native/browser parity fixtures cover 3,168 actual physical actor calls, including preparation and four terminal resets per fixture. Another 10,800 explicitly synthetic permutation vectors exercise both zero and nonzero attention. Sampled commands and buttons match exactly; the largest numeric error is `1.26e-6`. Nonzero-attention probes are numerical tests, not trained policies.

Commands:

```sh
python -m unittest discover -s training -p 'test_entity*.py' -v
node --test tests/entityPolicy.test.js
python scripts/export-entity-policy.py \
  --entity output/entity-encoder/zero/entity.pt \
  --legacy output/entity-encoder/zero/legacy.pt \
  --output output/entity-encoder/parity
```

Use the repository's native training environment with PyTorch, NumPy and MuJoCo. Export/parity outputs remain isolated from `public/models`.

## Measured cost of the parameter projection

The exact selected native parent has SHA-256 `ca9462e9cd3f084a2df687defc4de23513f15df700826a3a4524d0676bc74117`. Its prepared legacy reference has SHA-256 `40ce6061879a3272a5128d738803ec9706148065b442cd61c9d57e5f8efca5cf`; the projected entity pair has SHA-256 `9a52bf5ffa26a098a674a105856e2f371217cbbbfdbb58d0c98aa9a23d3f521f`. Both actor Adam states are empty and both have zero new actor updates. A separate genuinely zero-experience entity pair is retained in the preparation directory.

On the earlier reused 96-map development cohort, the projection changed the policy immediately. These are percentage-point changes in the relevant role's visibility objective, with paired map-bootstrap 95% intervals:

| Projected role | Same fixed opponent | Change | Interval |
| --- | --- | ---: | ---: |
| Hider | Selected seeker | −1.04 | [−6.52, +4.34] |
| Seeker | Selected hider | −2.73 | [−7.85, +2.24] |
| Hider | Genuine initial seeker | −5.69 | [−10.06, −1.60] |
| Seeker | Genuine initial hider | −1.93 | [−6.47, +2.53] |

The hider regression against the initial opponent is credible in this assessment. The projected pair also had 27 complete search misses versus 16 for the selected pair. Projection is not a free improvement, and an inconclusive difference does not prove equivalence. Learning results must recover or exceed these starting policies before any replacement is considered.

Evidence: `output/entity-encoder/zero/`, `output/entity-encoder/zero-evaluation/evaluation.json`, and `output/entity-encoder/parity/parity-summary.json`. The behavior check simulated 480 new games and reused 288 exact fixed-reference games; none used a final-test map.

The predeclared new 96-map cohort also shows a cost before learning. Against the same selected hider, the projected seeker misses 25 games versus 20 for the legacy reference; its objective change is −3.54 percentage points [−9.30, +1.92]. Projected seeker changes against fixed initial, warm-start and intermediate hiders are −5.07 [−9.19, −1.23], −10.27 [−15.59, −5.08], and −6.76 [−12.94, −0.90]. The projected hider changes by +1.63 [−2.35, +6.06] against the selected seeker, but regresses against the intermediate seeker by −6.55 [−11.54, −1.78]. Neither role's tool-benefit interval establishes utility. These are starting-policy measurements; improvement from this projection is distinct from improvement over the exact selected reference.

Evidence: `output/entity-encoder/comparison/evaluation-zero/evaluation.json`, SHA-256 `a59b93f65ee1cf5cc9a3c215b817515a1d5fbac09d8d41992ecfd06002cb8b6b`. The 32-condition table contains 2,208 physically simulated games and 864 exact aliases of identical legacy pairings, costing 1,059,840 evaluation interactions. All 864 alias comparisons match exactly and are not treated as extra independent samples.

## Full rollout/update benchmark

Both arms ran four full cycles with 128 worlds, eight native workers, a 256-step horizon, 32-step recurrent sequences, sequence batches of 256, and two PPO epochs. The first cycle was excluded from the averages. Updates were real PPO updates on disposable copies; every later rollout used the original frozen source actors, and no updated checkpoint was retained.

| Measurement | Legacy | Entity |
| --- | ---: | ---: |
| Interactions per second | 7,105 | 5,802 |
| Rollout seconds per cycle | 7.29 | 8.39 |
| Actor inference seconds | 0.93 | 1.23 |
| Physics/transport seconds | 5.58 | 6.32 |
| Actor optimization seconds | 0.67 | 1.59 |
| Critic optimization seconds | 1.27 | 1.31 |

Entity throughput was 81.7% of the reference in this sequential shared-machine benchmark. Different physical trajectories and concurrent load affect the total; this is not a claim that every timing difference is caused by the encoder. Startup and snapshot-copy time are outside the cycle averages. Early disposable optimizer updates are a cost estimate, not a training result. Evidence: `output/entity-encoder/benchmark/report.json`.

## Zero-update complete-game value check

Before the first training update, each arm played 96 complete games under the actual historical-opponent league. The inherited central critic and same-pre-action partial-pair values were compared with exact discounted complete-game returns. The time-only baseline was fitted on 47 games from 46 static geometry groups; assessment used 49 games from 48 different geometry groups. No actor or critic fitting occurred. Optimization reward scale is 0.05 and discount is 0.998.

| Assessment mean squared error | Legacy | Entity |
| --- | ---: | ---: |
| Inherited central critic | 8.29 | 8.37 |
| Detached partial-pair value | 10.54 | 9.87 |
| Time-conditioned mean return | 15.93 | 12.71 |

For central minus partial MSE, whole-geometry bootstrap intervals were [−4.03, −0.18] in the legacy arm and [−3.15, +0.39] in the entity arm. Central minus time-baseline intervals were negative in both. These noisy Monte Carlo errors did not show a projected-arm value collapse, but do not certify calibrated advantages. Preparation errors remain larger than play errors. The planned continuation therefore retains the same critic weights with new AdamW state in both arms, without an extra fitting phase.

Evidence: `output/entity-encoder/comparison/value-zero/` includes the report, source snapshot, complete per-timestep prediction/return arrays, actual opponent assignments, and geometry split. This check cost 92,160 evaluation interactions, separately from learning.

## Predeclared learning phase

`output/entity-encoder/comparison/PROTOCOL.json` was written before training; SHA-256 `41ca508cc34aea39dc7f1f8e02e86c7938e07a90a2a3fdeb2339ce46fd9f378a`.

- Principal development cohort: 96 maps beginning at seed 1,600,100,000, with 32 maps per scenario and declared sizes 6–12, requested boxes 0–8 and ramps 0–2. Actual placed counts are reported. These maps become reused development evidence after their zero-update assessment; final-test seeds at 1.9 billion stay untouched.
- One paired training seed, 885713. Both arms use the same reviewed league B: half current/current, one quarter each current role against an own frozen historical counterpart. No opponent identity enters an actor.
- Exact milestones are 0, 7,995,392 and 15,990,784 fresh interactions per arm. Count current, historical and active policy samples separately. The total includes blind preparation; it is not a count of gradient-bearing samples or independent episodes.
- Both actor Adam and central-critic AdamW states start empty. Recurrent PPO, entropy, KL limit, sampling and the entire physical/reward contract are unchanged. Source snapshots accompany native checkpoints.
- Every role faces the same frozen selected, genuine initial, warm-start and intermediate opponents. Paired physical tool ablations retain pushing. Search misses, scenario/count results, wall pressing, displacement, occlusion, grip behavior and objective intervals are recorded. Identical policy-pair cases can be reused only with matching tensors, controller code, physics mode and map configuration.
- There is no automatic promotion. A credible role regression blocks that role. Inconclusive intervals cannot establish useful tools, and longer holds alone do not establish purposeful manipulation. Any promising role combination needs its own pair assessment and actual visual review. One training seed cannot establish a general architecture advantage.

`training/train_entity.py` is isolated from the earlier trainer and validates the protocol and frozen assets. It writes `latest.pt` atomically and retains milestone checkpoints. `SIGINT` or `SIGTERM` finishes the current rollout/update and saves actor weights, both optimizer types, central critic, counters and RNG state. Resume with `--resume` plus the same protocol and dependency arguments. On explicit process restart, worlds, recurrent states and requested buttons restart together; it does not claim byte-identical uninterrupted trajectories. All native assets remain local and the established portable training workflow remains available separately.

Public model SHA-256 remains `6589ba0ecff6491d0b6c49f3c9b418b175597e87e6059ae1ea09f5277de99d90`. Native physics remains `1a0b54cdee8f7fc417f3b32bd901398501425f20e47aa17f06930745ca22faf5`.
