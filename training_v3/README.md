# Expanded wall layouts v3

Training-only successor to relational environment v2. Learned actor/attention/critic tensors are preserved. Fresh optimizers are declared because the environment distribution changed; old checkpoints and sources remain in training_v2. Sensors, actions, physics forces, and visibility-only reward are unchanged.

Six balanced layout families: shelter, rooms, open, connected-rooms, corridors, multi-exit. New partitions have 1.10–1.35m clear openings, central corridors have 1.5–2m width, and multi-exit shelters have a door on every side. Partition positions, openings, starts, rotations/reflections, sizes and props vary. All layouts fit the critic's 16-wall capacity.

Validation: 120 seeded layouts checked for connected wall-only walkable space with .27m clearance; spawn clearance including props; 9 complete native physics episodes; inherited policy migration, training updates, league refresh and resume; 6 sensor/reward/tool regressions. Movable props may obstruct routes and are intentionally manipulable.

Evaluation retains prior fixed maps and adds 24 new-family cases. Reports break down all six environment types with tools-disabled counterfactuals. No strategy or tool-use rewards.

Active command and PID: output/expanded-layouts-v3/launch.json. Status: output/expanded-layouts-v3/STATUS.json. SIGTERM pauses at an update boundary; the same command resumes. No browser model is automatically replaced.
