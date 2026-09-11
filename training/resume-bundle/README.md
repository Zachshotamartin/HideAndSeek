# Portable native training reference

These six `.pt` files contain our frozen 37M development actors, their native Adam moments and RNG, the compatible fitted critic, the genuine zero-experience reference, and our own historical opponents. The prepared `resume.pt` adds **zero new actor updates**. Every required file is included; no original machine paths are needed.

Install `training/requirements.txt` in a Python virtual environment, then run from the repository root:

```sh
npm run train:continuous -- \
  --resume training/resume-bundle/resume.pt \
  --output output/continuous-run --until-stop
```

`MANIFEST.json` contains full SHA-256 hashes verified before loading. Keep this directory immutable; the driver copies dependencies into the chosen run folder. After Ctrl-C or SIGTERM, resume `output/continuous-run/latest.pt`. The complete workflow and exact restart semantics are in [`../CONTINUOUS_TRAINING.md`](../CONTINUOUS_TRAINING.md).

The reference remains a development model with no reliable useful-tool claim. This native bundle is excluded from the npm package and browser downloads. Actor-only browser JSONs cannot replace its optimizer, value, and RNG state.
