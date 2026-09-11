# Review snapshot: Hide-and-seek

This branch preserves the current source and development browser assets. It is not a model-quality or production-readiness claim. No GitHub Actions were added.

## Checks run before push

- `npm run build`: passed.
- `npm run test:runtime`: passed for trained and initial models in an actual browser, without page errors.
- `npm test`: 44 passed / 0 failed. Updated v4 observation/action fixtures, restored manifest integrity metadata, enforced relational actor-only shapes, and added native policy/jump parity.
- The preceding native training verification passed 9 diving tests and 17 focused hide-and-seek tests, plus a bounded-torque/momentum audit and pure browser-policy inference parity. Native training verification remains separate from browser/model quality.

## Training boundary

Current runs execute from frozen source copies on the owner's SSD. This Git push does not restart them or replace the site's model. Native optimizer/checkpoint archives and transient output directories are not committed. Public model files are the existing development exports. Read the version-specific trainer README for the new training path; the root README describes the existing browser runtime.

New system: `training_v5`, retaining `training` through `training_v4` for provenance. At the status check, the first full-system pilot completed 10,485,760 new interactions; the second had 2,949,120. The first pilot mean fixed-opponent utility was 0.50041. Both role-change confidence intervals included zero. Hider grab/lock benefit was +0.10378 (95% bootstrap interval +0.05122 to +0.15741); seeker grab/lock benefit was -0.00661 (-0.03371 to +0.02035). These are development counterfactual measurements, not proof of an effective general tool-use strategy.
