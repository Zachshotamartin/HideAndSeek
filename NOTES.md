# Hide and Seek handoff

Status: FROZEN. Implementation, training, untouched evaluation, tests, build and final captures are complete. No Git operations performed by this agent. Root should publish a **new HideAndSeek repository**, retaining the old TeachThePet history separately.

## Clean source to publish

Copy these paths from the current TeachThePet working directory into the new repository:

- `src/` (only `index.js`, `main.js`, `style.css`, `renderer.js`, `core/arena.js`, `core/policy.js`; `src/data/` is empty and unnecessary)
- `public/models/hide-seek.json`
- `scripts/` (all remaining scripts are H&S: browser helper/test, capture, train orchestrator)
- `tests/` including the Python parity fixture
- `training/*.py`, `training/requirements.txt`, `training/training-log.json`, `training/validation.json`, `training/test-episodes.json`
- `examples/hide-and-seek-arena.png`, `examples/hide-and-seek-vision.png`
- `index.html`, `package.json`, `package-lock.json`, `.gitignore`, `README.md`, `NOTES.md`, `LICENSE`, `evaluation.json`

Do not copy `.git`, `node_modules`, `dist`, `test-results`, `training/runs`, or `training/__pycache__`. All prior maze algorithms/worker/tests/weights/showcase images were removed from the working source. The local folder remains named TeachThePet only for coordination.

## Integration

- Package: `@zachshotamartin/hide-and-seek`.
- Metadata id/title: `hide-and-seek` / `Hide and Seek`.
- Named synchronous `mountExperiment(element, { embedded: true, assetBase: '/your/copied/public/path/' }) → { dispose() }`.
- `metadata.instructions` and `.limitations` are string arrays. `metadata.technique` is a string.
- Scope/root selector: `.hide-seek`. Ready marker: `[data-ready="true"]`; current policy in `data-model`; ticks in `data-steps`; phase in `data-phase`.
- Embedded mode removes standalone H1 and uses a truly transparent WebGL clear. `.hs-board-shell` and `.hs-viewport` are transparent in embedded mode. The actual arena floor remains opaque geometry. Root page texture/background shows around it.
- Copy the public directory; `assetBase` resolves `models/hide-seek.json`. Default serves this relative to the standalone document. Browser needs WebGL2, no workers/server/network beyond the static bundle and model file.
- Three dependency aligned to `^0.180.0` to share Portfolio's renderer dependency.

## Model / report schema

`public/models/hide-seek.json` has `{ version:1, trained, initial, training, evaluation }`. Each portable actor-pair object is `{ format:'hide-seek-ppo-v1', label, policies:[{layers}, {layers}] }`. Three layers declare input/output and flat row-major weights+bias arrays:23→48→48→5. There are 3,749 parameters per actor. Runtime validates dimensions and finite bounded numbers. The critic/optimizer never ships. Imported model scores are explicitly unverified.

`evaluation.json` has format `hide-seek-evaluation-v1`, training/protocol/summary/improvement/results. `summary` supplies `episodesPerMatchup:600`, `arenas:200`, `seekerCapture:0.975`, `hiderSurvival:0.7733`. See the report for fixed-opponent definitions, exact test seeds, geometry hashes, visibility, collisions, distance and paired bootstrap intervals. `training/test-episodes.json` records the 4,200 actual scored episodes across 7 matchups.

## Learning result

Final reproducible training: 14,745,600 arena steps, 138,563 episodes, 2,400 PPO updates, 468.46 CPU seconds. Seed 2709. 121,647 recorded unique training geometry hashes. Exact LoS was used throughout this final run. The earlier exploratory run is ignored under training/runs/ppo.

Selected final checkpoint using 100 validation arenas, then one untouched evaluation on 200 geometry seeds × 3 independently seeded starts, excluding every training geometry hash. Same fixed actors on new layouts, no per-map training or route planner.

- Seeker vs reactive flee: 2.17% initial → 97.50% trained capture.
- Hider vs reactive chase/search: 6.00% initial → 77.33% trained survival.
- Trained pair against each other: 72.17% seeker / 27.83% hider wins.
- Paired map-bootstrap 95% improvement intervals: seeker +93.50–97.00 percentage points; hider +66.67–75.83 points.

Do not turn these separate-opponent scores into a claimed 97.5% win rate against the trained hider. Do not claim emergent tool use, optimal play, general-purpose AI, or guaranteed generalization to arbitrary edits.

## Checks and captures

- `npm test`: 10 passing Node tests: seeded connected generation/limits, exact LoS corners/symmetry, walls/tagging, prep blindness, hidden-position invariance, stale memory, termination/reward, editing validation, model validation/replay, Python/JS trajectories/observations/rewards/logits, held-out metrics.
- `/tmp/teach-pet-training-venv/bin/python training/test_sim.py`: 3 passing Python privacy/lifecycle tests.
- `npm run test:browser`:actual trained/initial movement, complete learned rounds, playback and keyboard camera controls, 3D picker + numeric cover edit, export/import, invalid JSON, delayed-read races/dispose, missing-model error, batched false/true IntersectionObserver, 320/390px and full speed text, alpha 0 embedded vs alpha 255 standalone pixel check.
- `npm run build`:passes with lazy entry 10.02 KB gzip and Three renderer 138.04 KB gzip. Vite reports the normal 500 KB raw chunk warning for the Three renderer; runtime remains lazy. CSS 2.18 KB gzip.
- Static model bundle 171,910 bytes / 72,788 bytes gzip (includes both learned and initial weights plus aggregate evaluation). Full episode records are source-only, never part of browser startup.
- `npm run capture`:two real canvas exports, each 1850×1100 RGBA PNG. Files are `examples/hide-and-seek-arena.png` (seed 2709, 42 decisions) and `examples/hide-and-seek-vision.png` (seed 98231, 48 decisions, seeker sight inspection). No UI edges, labels, or solid canvas background; both use alpha 0 around arena geometry. This is a renderer export, not a generated illustration or composite.

The final browser reproduction server is port 5181. Source was built/tested with Node 22. Training runtime was `/tmp/teach-pet-training-venv/bin/python` (PyTorch 2.14.0, NumPy 2.5.3); standard Python instructions are in README.
