# Explicit Hide and Seek delivery list

Root owns Git staging and publication. This file distinguishes source, reproducible native continuation, measured evidence, and local working artifacts. Keep native training/physics and browser model selection separate. No GitHub Actions are needed.

## Training source to include

The repository should retain these authored modules and their documented experiments:

- Native environment and transport: `training/physics.py`, `training/env_pool.py`, `training/export_parity.py`, `training/fixtures/native-physics.json`, `training/requirements.txt`.
- Original actor/trainer/evaluator: `training/actor.py`, `training/train_physics.py`, `training/evaluate_physics.py`, `training/profile_training.py`.
- Persistent controller: `training/persistent_actor.py`, `training/persistent_train.py`, `training/persistent_evaluate.py`.
- Separate training-only value models: `training/central_critic.py`, `training/residual_critic.py`, `training/fit_central_value.py`, `training/fit_residual_value.py`, `training/central_persistent_train.py`, `training/evaluate_central_pilot.py`.
- Completed league and diagnostic drivers: `training/league_ppo.py`, `training/train_league.py`, `training/evaluate_league.py`, `training/evaluate_mixed_roles.py`, `training/evaluate_object_mobility.py`, `training/review_league_traces.py`, `training/diagnose_league_gradients.py`, `training/diagnose_search_memory.py`, `training/diagnose_entity_failures.py`.
- Extendable and sustained workflows: `training/train_continuous.py`, `training/evaluate_sustained.py`, `training/evaluate_policy_modes.py`, `training/evaluate_selected_tools.py`, `scripts/train-continuous.mjs`, `scripts/build-training-bundle.py`, `scripts/prepare-mixed-continuation.py`, `scripts/select-physical-roles.py`, `scripts/build-selected-evidence.py`, `scripts/package-selected-assets.mjs`, `scripts/package-training-reports.py`. Include the `train:continuous` script addition in `package.json`, preserving root’s dependency pins and lockfile.
- Isolated, still experimental object encoder: `training/entity_actor.py`, `training/train_entity.py`, `training/evaluate_entity.py`, `training/evaluate_entity_zero.py`, `training/check_entity_value.py`, `training/benchmark_entity.py`, `scripts/prepare-entity-pilot.py`, `scripts/export-entity-policy.py`, and `scripts/test-entity-parity.mjs`. These do not replace the accepted browser controller or earlier trainers. Keep `training/ENTITY_ENCODER_PROPOSAL.md` and `training/ENTITY_ENCODER_RESULTS.md` with them; the latter distinguishes measured initialization from the ongoing comparison.
- `scripts/run-entity-sequence.py` is the local macOS process-exit launcher for finishing this approved native comparison unattended. It validates the completed control, starts the declared entity arm, then evaluates the retained 8M/16M milestones. It does not require an active assistant and cannot promote a model.
- Saved-run management for subsequent runs: `training/train_saved.py`, `training/checkpoint_store.py`, `training/evaluate_saved.py`, `training/evaluated_models.py`, `training/test_saved_workflow.py`, `training/SAVED_TRAINING.md`, `scripts/train-saved.mjs`, and the `train:saved` package script. Preserve existing dependency pins and lockfile. These additions do not edit the fixed comparison's trainer, protocol or automatic evaluator.
- Existing actor exporters and parity tools: `scripts/train.mjs`, `scripts/export-physics-policy.py`, `scripts/export-persistent-policy.py`, `scripts/package-persistent-assets.mjs`, `scripts/test-persistent-parity.mjs`.

## Native tests to include

`training/test_actor.py`, `training/test_persistent.py`, `training/test_env_pool.py`, `training/test_central_pool.py`, `training/test_central_critic.py`, `training/test_central_training.py`, `training/test_residual_value.py`, `training/test_object_mobility.py`, `training/test_league.py`, `training/test_continuous_training.py`, and `training/test_resume_bundle.py`.

For the isolated object encoder, include `training/test_entity_actor.py`, `training/test_entity_training.py`, `tests/entityPolicy.test.js`, and `src/core/entityPolicy.js`. The new inference module is inactive in the current production entry; existing binary and persistent imports remain supported. Experimental weights and optimizer artifacts under `output/entity-encoder/` remain excluded until separately reviewed.

## Portable native assets to include

Keep the entire **6.02 MB** `training/resume-bundle/` directory: `README.md`, `MANIFEST.json`, `resume.pt`, and its five `assets/*.pt` files. These are actual own native weights, optimizers, critic, and references with full hashes. They are intentionally outside package.json’s npm `files` list and must remain absent from browser downloads. No other `.pt`, `.npy`, or optimizer files are required for the documented fresh-clone 37M continuation.

## Documentation and measured evidence to include

`README.md`, `NOTES.md`, this file, `training/CONTINUOUS_TRAINING.md`, `training/PERSISTENT_BUTTON_PILOT.md`, `training/CTDE_EXPERIMENT.md`, `training/LEAGUE_EXPERIMENT.md`, `training/MIXED_ROLE_ASSESSMENT.md`, `training/SUSTAINED_SELF_PLAY.md`, `training/SEARCH_DIAGNOSIS.md`, and all `training/reports/*.json` plus its README. The compact reports retain unchanged measured values and hashes without shipping raw rollout caches. The supplied browser reference remains explicitly a development model.

## Runtime and accepted browser assets

Root and the visual owner should stage their reviewed runtime changes, including `src/core/physics.js`, the MuJoCo loader/vendor glue and notices, `tests/physics-resources.test.js`, the final shared-stick renderer/character helpers and `tests/character.test.js`, current `src/index.js`, `src/style.css`, policy controller/inference modules, and corresponding numerical/browser/CSP tests. This agent did not change runtime behavior during the league and sustained runs.

Keep only the selected development pair and genuine initial actor files named by `public/models/MANIFEST.json`, their compact parity/evaluation artifacts, and generated `src/core/policyAsset.js`. The selected actor file is `physical-policy-6589ba0ecff6.json`: exact retained 37M hider plus the seeker from the sustained 16M milestone. Its source SHA and separate lineage are in the manifest; it is not the latest 32M pair. Include the small disclosure/copy changes in `src/index.js` and `src/style.css` and updated asset/browser assertions in `tests/policyController.test.js` and `scripts/browser-test.mjs`. Native latest, optimizer state and the other assessed policies stay separate. Keep root’s exact shared StickFigure package/lock pin. Real seed-2709 review images, videos and replay JSON are in `output/sustained-mixed/browser-selected/`; diagnostic SVG paths are not game screenshots.

## Keep local and ignored

Do **not** stage `output/`, `training/runs/`, `.venv-physics/`, `.venv/`, `node_modules/`, `dist/`, `test-results/`, `__pycache__/`, `training/pure-self-play-progress.log`, old unused artwork, raw training archives, optimizer caches outside the deliberate resume bundle, profiler dumps, or the external OpenAI reference checkout. Preserve these local files; no deletion is requested. Native full reports/checkpoints may be attached later as explicitly chosen release artifacts with hashes, rather than accidentally committed from a working directory.
