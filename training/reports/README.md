# Development evidence

These compact JSON files copy measured values from complete frozen reports. Each includes its source report’s SHA-256. Raw episode arrays and large trajectory caches remain local; this package contains the unchanged summaries, paired confidence intervals, checkpoint identities, scope, and diagnostic results. `scripts/package-training-reports.py` performs this extraction without training or evaluation.

- `league-4m.json` and `league-8m.json` record the completed matched A/B comparison. Neither complete pair qualified for promotion.
- `mixed-broad.json` records the separate 96-map varied-size/count assessment of the unchanged 37M hider and league seeker. Broad improvement and reliable useful-tool behavior remain unproven.
- `gradients-{parent,A,B}.json` record read-only likelihood, gradient, and value-calibration diagnostics. They made no optimizer updates and are not held-out performance tests.

The completed `sustained-{8m,16m,32m}.json` reports preserve all three predeclared milestones. Their `-browser` counterparts compare directly with the original browser seeker on the same fixed opponents. `sustained-completion.json` records exact current/historical/active counts; `sustained-16m-vs-32m.json` explains why the earlier seeker was retained. `selected-role-lineage.json`, `selected-action-modes.json`, and `selected-tool-effects.json` describe that exact frozen selection and its own additional read-only assessments. `search-memory-diagnostic.json` and `entity-search-diagnostic.json` preserve the bounded architecture probes; neither proves the cause of every missed search.

These are repeated development comparisons against fixed own opponents, not a final test or OpenAI-scale reproduction. The selected browser model’s manifest references its own compact paired-map evidence. No training reward automatically promotes a model, and the final 1.9-billion seed range remains untouched.
