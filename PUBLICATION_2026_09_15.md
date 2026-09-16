# September 15 evaluated browser snapshot

The default browser pair is the eligible champion from the known-position,
sustained-visibility continuation, checkpoint
`3f465cf2a5f42d4dfa820f1ceeebdd1f5cf8f930f324d863ab771623586832cc`.
It has 471,859,200 cumulative self-play interactions, including inherited
experience. Its mean fixed-opponent development utility is 0.838415 across 168
maps and 2,016 evaluation episodes. This is selection on development data, not an
untouched final test. Newer training checkpoints are not automatically better.

Both actors know the opponent's current relative position, even when occluded.
Velocity and heading remain visible-only. The only reward is sustained visibility:
the seeker receives +1 when it can see the hider and -1 otherwise; the hider gets
the opposite. There is no capture termination or future visibility credit.
Physical movement, grounded jumps and persistent grab/lock commands are unchanged.
The browser still plays until paused or reset. These task changes mean the old
restricted-vision development scores cannot be compared directly with this score.

The before-training comparison retains the original saved zero-experience actor
tensors. Only its explicit observation contract is migrated to the same current
position sensors; no optimization takes place. The old latest-training option is
retired from this release because it used a different task.

`scripts/publish-known-position.py REGISTRY` verifies the immutable selected
checkpoint and evidence hashes, exports actor-only weights, and writes matching
manifest metadata. Absolute local paths in public evidence become basenames;
all episode measurements, source hashes and checkpoint identities are retained.
The manifest separately retains the original evidence hash. The exporter never
modifies a running checkpoint or trainer.

`scripts/generate-publication-fixtures.py` generates independent native truth for
the current sensor contract, using frozen source under
`tests/fixtures/native-known/`. Each of the two exported models is checked on 528
recurrent decisions, including sampled/deterministic movement and jump, persistent
buttons, preparation and memory resets. JS/native tolerance is 1e-5. The original
physical parity tests and backward compatibility tests remain in place.

The detailed historical training discussion in README describes the earlier
restricted-vision release. This document and `public/models/MANIFEST.json` identify
the current browser model and its associated evidence.
