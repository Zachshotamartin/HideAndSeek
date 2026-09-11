# Training-only centralized value experiment

This is an isolated development experiment. The production policy is not selected by these files. The physical visibility reward, partial actor observations, persistent controller commands, and physical dynamics are unchanged. A privileged critic predicts discounted return during training; it never chooses actions and is not exported to the browser.

The frozen actor parent is the persistent-button checkpoint after 14,098,432 new decisions, with 37,036,032 ancestral training decisions in total:

- File: `output/persistent-button-pilot/run/checkpoint-14098432.pt`
- SHA256: `375fd147ae2af5c284b15e4c898342a5256db5295907e302b530da0b61763075`

The parent improved both roles against the frozen binary opponent pair in the recorded 36-map validation, but paired tool ablations did not establish beneficial tool use. The centralized baseline is an optimization experiment, not evidence of a tool strategy.

## First fit: rejected

`fit_central_value.py` fitted a separate full-scene critic using 384 complete training games, selected the epoch on 64 maps, and assessed once on 96 more maps. Selected epoch 1 had held-out MSE 9.09848. The average of the existing partial predictions, `(hider V - seeker V) / 2`, had MSE 7.43727. The paired whole-map difference was +1.66121, 95% bootstrap CI [+0.37249, +3.05297], so the new critic was worse.

The original report's `oldPartialPair` field averaged the two individual squared errors. It was not the squared error of the averaged predictions. The original report is preserved; `output/central-value-pilot/value-fit/split-and-baseline-audit.json` records the corrected comparison. The revised canonical map hash ignores agent spawns, hashes the outer boundary by size, and identifies interior geometry modulo square rotations/reflections. Re-auditing all 544 maps found no repeated maps within or between the three splits.

Only four implementation-smoke PPO updates used this critic. No substantial continuation or production promotion used the rejected fit.

## Residual revision: predeclared protocol

`residual_critic.py` predicts the detached current partial-pair estimate plus a learned central correction. The correction's final output layer starts exactly zero, preserving the previous estimates. Both actor memories and partial values are detached. Full-scene features remain confined to the separate critic.

`fit_residual_value.py` collects 4096 complete training maps, 256 selection maps, and 256 untouched assessment maps. Prior development and smoke maps are excluded under the revised canonical hash. Selection seeds occupy 1.47B, assessment seeds 1.48B; final game-test seeds at 1.9B remain untouched. Rewards are uniformly scaled by .05 and discounted at .998, matching existing PPO values.

The settings are fixed before assessment: AdamW learning rate .0001, weight decay .001, dropout .1, squared correction penalty .1, batch 2048, at most 12 epochs, and selection-based patience 3. Epoch zero is an eligible candidate. Selection must improve by at least .001 MSE to replace the selected epoch. The report compares actual MSE of the averaged partial predictions, each individual partial critic, mean individual critic MSE, and a per-timestep mean-return baseline fitted on training maps. Confidence intervals bootstrap complete paired maps, not individual correlated timesteps.

Assessment games are never used for fitting or epoch selection. Improved value prediction alone does not establish improved play or useful tools. A bounded actor continuation requires an acceptable value-fit assessment, followed by separate gameplay and tool-ablation validation.

The completed residual fit selected epoch 8. On 256 fresh assessment maps, residual MSE was 7.48998 versus 8.25668 for the averaged partial predictions: difference −0.76669, paired-map 95% bootstrap CI [−1.26656, −0.27251]. The time-only baseline was 11.62966. Every actor parameter remained unchanged. The detailed result is `output/central-value-pilot/residual-fit/assessment.json`.

That result qualified the value baseline for a bounded 2,097,152-decision actor continuation, with the critic learning rate retained at .0001. Gameplay validation uses fresh seeds beginning 1,500,080,000, the exact frozen parent opponents, the genuine initial opponents, and physical tool ablations. Any resulting gameplay change measures the new checkpoint; it does not isolate a causal effect of centralized values alone because this continuation also corrects preparation-mask normalization and adds training experience.

The completed continuation did not qualify for promotion. On 36 validation maps, hider change against the same parent seeker was +6.27pp hidden (95% CI −3.33 to +16.25); seeker change against the same parent hider was −9.51pp visible (−22.07 to +2.69). Seeker performance against the same initial hider regressed −8.82pp (−17.00 to −0.49). Grab/lock tool benefits remained inconclusive: hider +0.87pp and seeker −1.79pp, both intervals including zero. This ablation preserves pushing, so it cannot establish that object movement is useless. The frozen 37M development actor remains unchanged. Full evidence: `output/central-value-pilot/residual-validation/README.md` and `evaluation.json`.

## PPO integration

`central_persistent_train.py` preserves the restricted 140-input recurrent actors and restores their own parent optimizer state. It records the partial-pair baseline from the same pre-action forward call used to sample the action. Bootstrap partial values use the current pre-action state without advancing the action RNG. The critic's AdamW state is separate, and both its input memories and baseline predictions are detached.

Seeker recurrence advances through preparation, while actor loss, entropy, advantage normalization, and approximate KL use active play samples only. All-blind batches skip the optimizer, including Adam momentum. The critic learns from the whole episode, including preparation, without changing game rewards.

Saved training checkpoints distinguish parent decisions, critic warm-up environment decisions, and new actor-update decisions. Selection and assessment decisions are reported separately. Central critic weights and optimizers remain training artifacts; portable actor exports contain no privileged inputs or critic tensors.

## Commands

```sh
/tmp/teach-pet-training-venv/bin/python -m unittest discover -s training -p 'test_central*.py' -v
/tmp/teach-pet-training-venv/bin/python -m unittest discover -s training -p test_residual_value.py -v

/tmp/teach-pet-training-venv/bin/python training/fit_residual_value.py \
  --parent output/persistent-button-pilot/run/checkpoint-14098432.pt \
  --output output/central-value-pilot/residual-fit \
  --exclude-dataset output/central-value-pilot/residual-excluded-maps.pt
```

Do not rerun an assessed fit into the same output directory. Preserve checkpoint hashes, source hashes, split identities, and negative findings when continuing development.
