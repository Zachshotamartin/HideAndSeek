# Tool control and grounded jumping — development continuation

Restarted from the paused `output/expanded-layouts-v3/run/latest.pt` (5,308,416 interactions in that branch). Original v3 files and checkpoints remain intact. Diving is a separate, uninterrupted process.

- Grabs are exclusive. Locked props cannot be grabbed, and locking releases a carrier instead of welding the agent to the world.
- Visibility-only, zero-sum reward is unchanged. No reward for grabs, jumping, tool movement, or prescribed tactics. Tool entropy contributes 0.1 of its previous strength; movement exploration remains unchanged. This reduces exploration pressure on button chatter but does not guarantee useful behavior.
- An additional learned continuous output requests a jump above 0.5. The actor's normal head grows from 3 to 4 outputs. Physical actions are `[forward, sideways, yaw, grab, lock, jump]`; PPO stores four raw motion outputs followed by two categorical tool commands.
- Jumping requires nearby support and low vertical speed, and is forbidden on wall tops and in midair. Maximum rise is 0.8m under gravity 15m/s². Standard props are 0.7m tall; generated walls are 2.2m tall. The apex is additionally capped 0.15m below the shortest wall, including launches from props. Carrying a prop reduces the impulse per combined mass.
- Actor inputs remain 208 physical observations plus two button states. Vertical position/velocity are already observed. The action/export contract is explicitly versioned `original-mujoco-relational-jump-policy-pair-v4`.
- Migration preserves compatible actor/critic tensors, creates the new jump output, and restarts optimizers. It is not exact continuation of an unchanged MDP. Full new-run checkpoints, milestone snapshots, fixed-cohort evaluations and best evaluated model retention continue through the controller.
- Training episodes remain finite. The browser uses continuous mode and clamps the observed phase clock after the training horizon; it does not reset agent memory or arena positions at that horizon. Behavior far beyond the trained episode length is unvalidated.

Run/resume:

```sh
/tmp/teach-pet-training-venv/bin/python training_v4/train_scaled.py --source output/expanded-layouts-v3/run/latest.pt --output output/tool-control-jump-v4 --target 1048576000
```

Do not edit hashed training sources while this run is active. Stop with SIGTERM for a checkpointed pause before changing training logic. Repeating this command resumes its saved full training state. Browser model updates are explicit; a training update alone does not promote a model.

Validation: seven targeted native tests cover exclusive grips, lock/release, grounded-only jumping, prop landing, wall collision, preparation, weight migration and a PPO update. Browser/native trajectories for jumping and grabbing agree within 1e-12; actor inference within 3e-7. The browser was exercised beyond 600 decisions without resetting, then paused and reset. No learned-strategy improvement is claimed from these implementation tests.
