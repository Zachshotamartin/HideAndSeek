# Shared stick-figure renderer

The visible figure is defined once in `@zachshotamartin/stick-figure`, the same
original package consumed by Learn to Dive. It draws eighteen thin straight
sausage segments and twenty round joint markers, including a small sphere head.
There is no sculpted mesh, GLB, face, skinning, clothing or animation clip. The
same dimensions and code serve both apps; blue and red distinguish the teams.

The policy controls a physical capsule's forces, turning, grabbing and locking.
Visual gait IK follows measured displacement and actual support normals. Hands
reach the measured grip/contact anchor. This presentation is not additional
learned joint control. Props and visibility come from actual MuJoCo state.

`node scripts/visuals/audit-contacts.mjs` checks grip anchors, airborne-to-grounded
transitions, browser errors and mobile rendering using real physics fixtures.
Its actions are deliberately scripted to exercise contacts; these images are
renderer checks, not learned-policy performance demonstrations.

`node scripts/visuals/capture-figure.mjs` creates an art-review closeup and arena
view. Final policy examples come from the qualified training/capture workflow.

All character code is original project work, MIT © Zachary Martin, 2026.
