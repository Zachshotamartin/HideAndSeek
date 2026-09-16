# Hide and Seek

**Current browser release (September 15):** the [best evaluated known-position pair](PUBLICATION_2026_09_15.md), at 471.86M cumulative interactions and 0.8384 fixed-opponent development utility. Both agents know the opponent's position behind walls; actual visibility remains occlusion-dependent. The historical restricted-vision training account below describes an earlier release. The current model, task contract, native/browser parity and evidence are identified in the linked publication record.

Current training implementation: [training_v5/README.md](training_v5/README.md). Browser runtime and training versions are intentionally separate while new models are evaluated. See [review status](REVIEW_STATUS.md) for known test failures and [the Claude review prompt](CLAUDE_REVIEW_PROMPT.md).

An original physical learning experiment by Zachary Martin. Two recurrent neural policies play hide and seek in an editable 3D arena. The agents can accelerate in any direction, turn, push, grab, release, and lock physical props. Boxes, long planks, and ramps have real mass, inertia, friction, and collision geometry.

**Development checkpoint:** general play has improved in paired validation, but reliable useful tool strategies have not been established. This is a local development candidate, not a qualified final model or a claim of emergent construction.

The [OpenAI hide-and-seek study](https://openai.com/index/emergent-tool-use/) is a research reference for the mechanics. This project uses its own environment, policy implementation, and learned weights. It does not import the study's implementation or checkpoints, and it does not claim to reproduce its scale or stages of emergent tool use.

## Run

```sh
npm install
npm run dev
npm run build
npm test
npm run test:browser
npm run test:csp
```

Node 22 or newer is required. The development server uses port 5181. Browser regression tests use a separate port, 5193; override it with `HS_TEST_PORT`. The browser lazily loads MuJoCo WebAssembly and the frozen policy asset. No account, server, inference API, or browser training is involved.

## The physical game

The blue hider has 6.4 seconds of preparation. The red seeker has no visual observations, its applied controls and requested grab/lock button states are zero during that phase. Both recurrent states advance on their own permitted observations, matching training.

The next 12.8 seconds are scored by visibility. Each physical decision gives the hider +1 when unseen and −1 when seen. The seeker receives the opposite. There is no touch-to-tag rule, prescribed shelter location, tool-use bonus, expert action, path planner, or pursuit bonus. Grabbing a prop is useful only if it improves the learned game outcome.

Both agents have a forward 135° field of view with a 6 m range. Exact MuJoCo ray intersections determine whether objects or the opponent are visible. Twenty-four body-relative rays sense walls and props. Opponents are excluded from these all-around range rays. A hidden opponent's current coordinates never appear in the policy observation. The GRU retains information from earlier observations.

A grab attaches an eligible nearby visible prop through a physical weld. The grasp remains until release, including when the object moves out of the visual cone. A lock constrains the prop to the world; only its owner can release it. Solid ramps can support agents above the floor. Object motion and contact are resolved by the same MuJoCo 3.13.0 engine in native Python and browser WebAssembly. Agents use upright sphere collision bodies. Props have three translation axes and a vertical rotation axis, so they can lift and turn but do not pitch or roll freely. The articulated-looking character meshes animate from the physical body and contact state.

The application permits 6–12 m arenas and up to ten props. Generated layouts include shelters, rooms, and open space, with seeded rotations, reflections, object placement, and independent start jitter. Imported or edited geometry must remain within size limits and start clear of agents, walls, and other props. Editing resets the physical world and policy memory.

## Learning and evaluation

`training/physics.py` is the original native environment. The development browser uses the `original-mujoco-persistent-buttons-ppo-v1` actor from `training/persistent_actor.py` and `training/persistent_train.py`. Each actor receives **140 values: 138 permitted physical sensor values plus its own two previous requested button states**, followed by a 96-unit encoder and 64-unit GRU. It outputs continuous movement/turning distributions and separate categorical Keep / Press / Release commands for grabbing and locking. Keep preserves the requested button; there is no forced hold duration, release schedule, scripted maneuver, or extra tool reward. Requested button state is separate from whether a physical grasp actually succeeds.

The supplied development pair retains the **hider from the 37,036,032-interaction reference** and the **seeker selected after 15,990,784 further league interactions**. That seeker phase includes 5,957,888 decisions by the current seeker, of which 3,968,000 are active play samples; other decisions belong to its opponent or blind preparation. Earlier hider/seeker ancestry overlaps. The union source budget is 62,988,288 interactions, including prior value-only and historical-opponent experience, **not an equal per-agent training count**. The actor export records each role’s exact checkpoint hash and inherited counters. Selecting the two unchanged roles made no new optimizer updates.

The completed continuation collected 31,981,568 fresh interactions in 68.7 minutes. Its latest checkpoint remains available for native continuation, but the earlier seeker was retained: on the same 96 maps, complete search misses increased from 16 at the selected checkpoint to 22 at the final checkpoint. The original browser seeker missed 25. The hider showed no credible fixed-opponent improvement, so its earlier weights were retained. See [the sustained protocol and outcome](training/SUSTAINED_SELF_PLAY.md).

The separate **Before game experience** checkpoint is the genuine saved zero-experience original backbone converted to the persistent architecture with a uniform categorical head. It is not the pilot warm start. Native optimizer and privileged value-model state are excluded from browser assets.

The exact selected pair was assessed on **96 maps across shelter, rooms and open layouts**, sizes 6–12 m, zero to eight boxes and zero to two ramps. These maps were reused for development selection; they are **not untouched final-test data**. Role comparisons keep the named opponent fixed. Grab/lock interventions preserve physical pushing and the actor’s own requested button state.

| Paired comparison | Role objective change | Map-paired bootstrap 95% interval |
| --- | ---: | ---: |
| Selected seeker vs original browser seeker, same original hider | +5.26 percentage points visible | −0.36 to +10.68 |
| Selected sampled hider vs genuine initial hider, same browser seeker | +19.56 percentage points hidden | +13.77 to +25.46 |
| Selected sampled seeker vs genuine initial seeker, same browser hider | +41.47 percentage points visible | +35.17 to +47.87 |
| Sampled hider grab/lock enabled vs disabled | −0.24 percentage points hidden | −1.86 to +1.40 |
| Sampled seeker grab/lock enabled vs disabled | −0.86 percentage points visible | −3.54 to +1.44 |

Learning relative to zero experience is clear in this cohort. The improvement over the original trained seeker is descriptive, with an interval crossing zero. **Reliable useful tool strategies remain unproven.** The compact, content-addressed evidence file referenced by `public/models/MANIFEST.json` retains exact role provenance, source report hashes, and all 96 paired-map objective counts. Packaging recomputes contrast means from those physical episode counts before writing public assets.

The separate action-mode assessment uses the runtime’s actual tanh Gaussian mean and categorical argmax. Against the same sampled browser hider, mean seeker actions changed visibility by +4.32 percentage points [−0.42, +9.24], with 17 complete misses versus 16 for sampling. Median actual seeker grip length increased from 0.16 seconds across 134 grips to 1.84 seconds across 11 grips. These longer holds did not establish useful behavior: whole-pair mean-mode tool effects were −1.70 points for the hider [−3.32, −0.19] and −0.12 for the seeker [−1.22, +0.90]. Seeded sampling therefore remains the default; mean mode stays an explicit comparison.

`src/core/policyController.js` explicitly dispatches persistent models and legacy `original-mujoco-recurrent-ppo-v1` imports. A legacy import retains its 138-input binary tool behavior; it is never silently reinterpreted as Keep / Press / Release. Unsupported formats and non-actor weight keys are rejected. Imported training or performance claims are shown as unverified. Both agents' memories and requested buttons reset when the arena, checkpoint, sampling, or tool settings change, on replay reset, and at a new round. Actors receive no central-critic input or weights. The selected seeker was trained with a separate privileged critic; that value model was used only during optimization and is absent from inference.

The default seeded sampling mode samples the learned action distribution repeatably. Deterministic mode is available for comparison. Browser sampling uses its own seeded PRNG; a native Torch random seed is not an identical random-number stream. Exported replay files store the actual applied actions, so their physical replay does not depend on this distinction.

```sh
python3 -m venv .venv-physics
.venv-physics/bin/python -m pip install -r training/requirements.txt
npm run train:continuous -- --help
```

Set `PYTHON_BIN` when using another Python environment. The training command verifies that its interpreter can import the required libraries. [Continuous training](training/CONTINUOUS_TRAINING.md) supports additional fresh-game budgets or `--until-stop`, safe Ctrl-C checkpointing, optimizer/RNG resume, immutable archives, and a separate frozen reference. Training progress never automatically replaces the browser model. The older binary-policy trainer remains available as `npm run train`.

[Saved training and evaluated models](training/SAVED_TRAINING.md) adds `npm run train:saved` for both persistent and entity checkpoints: configurable step snapshots, recorded training/session timing, portable resume, extensions beyond fixed experiment targets, and a separate evaluated-best registry. The registry retains the full native pair and evidence; training return never selects best. In-flight worlds and recurrent state restart on process resume.

To reproduce a selected browser package from its saved native development artifacts:

```sh
.venv-physics/bin/python scripts/export-persistent-policy.py output/sustained-mixed/selected-candidate/candidate.pt --output output/sustained-mixed/selected-candidate/candidate.json
node scripts/package-selected-assets.mjs output/sustained-mixed/selected-candidate/candidate.json public/models/physical-policy-initial-1e9ef2a1b555.json output/sustained-mixed/selected-candidate/development-evidence.json
```

`scripts/select-physical-roles.py` records an exact unmodified role selection; `scripts/build-selected-evidence.py` resolves the corresponding cached physical outcomes. These export commands require those saved development artifacts. Fresh-clone native continuation instead uses the self-contained `training/resume-bundle/` command documented above; its starting reference is the original 37M pair, not a reconstruction of this mixed selection.

Each actor export automatically checks 1,056 actual native policy decisions against JavaScript, including deterministic and sampled actions, 208 blind-preparation actor calls, and four terminal resets. Maximum inference error was 1.21e−6 for the selected candidate and 1.20e−7 for initialization, below the 1e−5 tolerance. Packaging verifies the hashes and evidence before copying two actor JSONs, compact parity reports, the 96-map evidence and manifest into `public/models`. Large fixtures, native checkpoints and optimizer state remain outside public assets.

## Embedding

```js
import { mountExperiment, metadata } from '@zachshotamartin/hide-and-seek';
import '@zachshotamartin/hide-and-seek/style.css';
const instance = mountExperiment(element, {
  embedded: true,
  assetBase: '/experiments/hide-and-seek/'
});
// Later:
instance.dispose();
```

`mountExperiment` returns synchronously. Physics, renderer, and model loading run internally. Copy `public/models/` under `assetBase`; the generated `src/core/policyAsset.js` names the content-addressed default model. Vite emits the MuJoCo WASM dependency as a separate lazy asset. The live scene canvas is transparent. All library CSS is scoped to `.hide-seek`.

The browser loader supports `script-src 'self' 'wasm-unsafe-eval'` without JavaScript
`unsafe-eval`. Its generated binding glue uses Emscripten's non-dynamic invocation
approach; the pinned MuJoCo WASM binary is unchanged. Source hashes, full licenses,
and reproduction details are in `src/vendor/README.md`. Builds verify that the
generated glue matches its source. `npm run test:csp` serves the production build
under the complete portfolio security header and exercises both model checkpoints,
a full game, and a separate module worker with real physics.

Playback pauses while the arena is offscreen or its document is hidden. Disposing a mount releases its MuJoCo model/data, renderer resources, animation frame, listeners, and observers. Late file reads cannot replace a newer world or touch disposed DOM.

## Verification

`tests/physics.test.js` checks a 64-frame native/browser trajectory containing real acceleration, collisions, grabbing, and locking. It compares positions, velocities, both 138-value observations, grip state, and lock state. Additional cases exercise hidden-opponent privacy, preparation blindness, diagonal motion, wall nonpenetration, ramp support, lock ownership, blocked grabs, deterministic generation, import validation, and the physical push-only intervention.

`tests/policyController.test.js` verifies explicit format dispatch, actor-only export hashes, both state resets, persistent button feedback, preparation zeroing, seeded sampling and invalid imports. Browser tests additionally compare a 96-decision rollout across resets and checkpoint/replay changes, against actual standalone policy inference on the same physical observations.

Regenerate the native fixture with:

```sh
.venv-physics/bin/python training/export_parity.py
```

Browser tests operate the actual learned policies, complete a round, manipulate props through controls and the 3D picker, export/import actual files and replays, exercise delayed-file/disposal races, and check 320 px and 390 px phone layouts.

## Runtime regression checks

`npm test` checks the committed native reference trajectories and current policy schema. `npm run test:runtime` runs focused browser loading/playback/export checks for trained and initial models. These are correctness checks, not proof of model quality; older browser qualification scenarios are separate.

Regenerate native references only after investigating a contract change, using a Python environment with the pinned training dependencies: `python scripts/generate-runtime-fixtures.py`. References carry source/model hashes so drift is reported explicitly. Existing numerical parity tolerances remain unchanged.

## September 13 portfolio checkpoints

The default is the best eligible saved pair at 209,715,200 self-play interactions (fixed-opponent mean role utility 0.65865). The Latest comparison is a frozen 244,580,352-interaction snapshot; it is not asserted to be better. Both exports preserve the hider/seeker pair, use the existing 210-observation/6-action interface, and match native recurrent inference within 1e-5. This is a development model, with reliable useful tool behavior still unproven. Training runs separately from the browser. The manifest includes the selection checkpoint hashes and the fixed-opponent evaluation report.
