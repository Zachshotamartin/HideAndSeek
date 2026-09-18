# Hide and Seek — training and runtime handoff

The original MuJoCo game and browser interface are implemented. The prepared development package selects the unchanged 37M hider and the seeker from the 15,990,784-interaction sustained milestone, with exact separate role lineage. **Reliable constructive tool use has not been established.** The declared 31,981,568-interaction continuation is complete; the latest native checkpoint is preserved separately from the selected earlier role. Root owns final integration, model selection, publication, and the final runtime/visual review. No Git operations were performed by this agent.

## Extendable training delivered

`npm run train:continuous` supports an additional fresh-interaction budget or `--until-stop`. A graceful Ctrl-C or SIGTERM, including a process-group signal, finishes the current rollout/update, atomically saves `latest.pt`, creates an immutable archive, and closes workers. Optimizer moments, central critic, RNG streams, counters, and source hashes are retained. In-flight worlds, recurrent memory, and requested buttons restart at process boundaries; the documentation explicitly avoids claiming an identical uninterrupted future trajectory.

The tracked `training/resume-bundle/` is approximately 6.02 MB and includes the actual native 37M actors, optimizer/RNG state, fitted critic, genuine zero-experience reference, and own historical opponents. Its full SHA-256 manifest is verified before loading. A copied fresh checkout can train using only that bundle; no original machine paths or ignored output assets are required. Native files are excluded from the npm package and browser downloads. See `training/CONTINUOUS_TRAINING.md` for one-command continuation and the separate random-initialization workflow.

The driver keeps `latest.pt`, immutable archives, and `best-reference.pt` distinct. It never chooses a model by training reward or rewrites the browser manifest. A complete copied run remains resumable using relative asset paths.

The subsequent `train:saved` workflow supports both persistent and entity native checkpoints, configurable step archives, atomic latest state, UTC/session/training/rollout/optimizer timing, and a single-writer lock. Entity extensions create a separate linked protocol; the fixed comparison remains unchanged. `evaluate_saved.py` and `evaluated_models.py` retain an explicitly evaluated best whole pair, its native dependencies, all previous candidates, and evidence under one fixed cohort/opponent criterion. See `training/SAVED_TRAINING.md`. Three native regression cases verify real SIGTERM, duplicate-writer rejection, moved snapshot resume after original inputs are removed, extension beyond the old limit, the actual 37M bundle, and evaluated-best resume. Synthetic registry ranking fixtures are not performance evidence.

## Completed comparisons and selected development pair

The matched larger-rollout A/B experiment completed 7,995,392 fresh physical actor decisions per arm. All 4M/8M checkpoints, source snapshots, counts, negative evaluations, native trace pairs, and read-only gradient diagnostics are retained in `output/league-trial/`. The full protocol and outcome are in `training/LEAGUE_EXPERIMENT.md`.

The historical-opponent seeker improved in the narrow 36-map comparison, while its hider regressed. That whole pair was rejected. The unchanged 37M hider and B seeker were then assessed on 96 varied maps. The broader seeker gain was inconclusive, with 22 complete search misses. Tool-ablation effects also remained inconclusive. `training/MIXED_ROLE_ASSESSMENT.md` records the scope and measured outcome.

The authorized sustained phase completed 31,981,568 fresh interactions in 68.7 minutes, with immutable assessments at approximately 8M, 16M, and 32M. It preserves the original visibility objective, physical semantics, restricted observations, and historical-opponent league. Each role’s lineage is explicit. The inherited central critic is an initialization, not falsely described as a new fit on the mixed pair. The phase can be extended after assessment with the same continuous workflow. See `training/SUSTAINED_SELF_PLAY.md` and `output/sustained-mixed/`.

## Verification for these deliverables

Run with a working interpreter from `training/requirements.txt`:

```sh
.venv-physics/bin/python -m unittest discover -s training -p test_continuous_training.py -v
.venv-physics/bin/python -m unittest discover -s training -p test_resume_bundle.py -v
.venv-physics/bin/python -m unittest discover -s training -p test_league.py -v
.venv-physics/bin/python -m unittest discover -s training -p test_env_pool.py -v
```

All pass: four real Python/Node process-group SIGINT/SIGTERM lifecycle cases, one actual copied-checkout pretrained bundle rollout, four league masking/reset/accounting tests, and three native pool parity/failure-cleanup tests. The lifecycle cases delete original input assets, relocate the saved run, resume fresh games, and verify optimizer/RNG progression and unchanged references. The copied-checkout smoke is explicitly separate from training/evaluation budgets and is not performance evidence.

An independent actual eight-world, 256-tick league audit verified terminal resets, selected actor/memory identity, frozen/current assignments, exact counters, and bootstrap behavior. Its evidence is in Portfolio’s `output/ai-rebuild/league-review/`. No material gradient-mask, actor-coordinate, or likelihood defect was found in the completed audits.

## Runtime boundaries and current visuals

Native `training/physics.py` remains frozen at SHA-256 `1a0b54cdee8f7fc417f3b32bd901398501425f20e47aa17f06930745ca22faf5`. Browser MuJoCo accessor ownership was fixed and repeated-world memory plateaued. Root’s subsequent browser-only 16 MB arena-capacity allocation reduced the steady heap to approximately 21.1 MB with exact trajectory/sensor parity across 36 full rounds. Neither change alters native learning dynamics.

The renderer and character presentation belong to the visual owner. Directional footfalls and moving-support presentation were reviewed separately. The exact selected pair completed two actual localhost 5173 seed-2709 follow-seeker rounds, sampled and mean, with 240 decisions, no page errors and no failed requests. Neither diagnostic round used grab/lock. Videos, real scene images and replay actions are in `output/sustained-mixed/browser-selected/`. Current real scene captures are in Portfolio’s `output/ai-rebuild/stick-figures/` and later capture-review directories. Do not use the training diagnostic SVGs as screenshots or present incidental tool contacts as a learned construction strategy.

Package API remains synchronous `mountExperiment(element, { embedded: true, assetBase }) → { dispose() }`, with `.hide-seek` scoped styles. Native training files, optimizer checkpoints, and privileged critic features never enter browser inference. Root performs the final build, CSP, integration, visual, and public-claims review against the exact packaged runtime.
