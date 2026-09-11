import test from "node:test";
import assert from "node:assert/strict";
import { createCharacter } from "../src/visuals/character.js";

function frame(character, gripId, index, stopAt = 12) {
  const moving = index < stopAt;
  character.update({
    position: [Math.min(index, stopAt - 1) * .007, .25, 0],
    velocity: moving ? [.35, 0, 0] : [0, 0, 0],
    forward: [1, 0, 0],
    groundHeight: 0,
    grounded: true,
    colliderHeight: .5,
    gripId,
    handTarget: [.3, .24, 0],
    contacts: [],
  }, index * .02);
  return structuredClone(character.root.userData.pose);
}

test("holding the first prop uses the same stance as every other prop", () => {
  const first = createCharacter(0), other = createCharacter(0);
  try {
    for (let i = 0; i < 50; i++) {
      const a = frame(first, 0, i), b = frame(other, 1, i);
      assert.equal(a.gripping, true);
      assert.deepEqual(a.foot0, b.foot0);
      assert.deepEqual(a.foot1, b.foot1);
      assert.deepEqual(a.landmarks, b.landmarks);
    }
  } finally {
    first.dispose(); other.dispose();
  }
});

test("stopping during a step plants the foot while retaining a grip", () => {
  // Different stopping distances cover both swing legs and planted phases.
  for (const stopAt of [8, 16, 23, 31, 39]) {
    const character = createCharacter(0);
    try {
      let stopped;
      for (let i = 0; i < stopAt + 10; i++) {
        const pose = frame(character, 0, i, stopAt);
        if (i >= stopAt) {
          assert.equal(pose.foot0[1], 0);
          assert.equal(pose.foot1[1], 0);
          if (stopped) {
            assert.deepEqual(pose.foot0, stopped.foot0);
            assert.deepEqual(pose.foot1, stopped.foot1);
          }
          stopped = pose;
        }
      }
    } finally { character.dispose(); }
  }
});

const distance = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));
const scale = .70 / 1.743;
function updateMotion(character, t, position, forward = [0, 0, 1], extras = {}) {
  character.update({ position, forward, grounded: true, colliderHeight: .5,
    groundHeight: 0, gripId: null, contacts: [], ...extras }, t);
  return structuredClone(character.root.userData.pose);
}
function assertLengths(pose) {
  for (const side of ['L', 'R']) {
    for (const [a, b, length] of [['hip','knee',.42],['knee','ankle',.42],['shoulder','elbow',.30],['elbow','wrist',.28],['wrist','hand',.13]]) {
      assert.ok(Math.abs(distance(pose.landmarks[a + side], pose.landmarks[b + side]) - length * scale) < 2e-6, `${a}–${b} ${side} stretched`);
    }
  }
}
function assertPlanted(previous, pose, i) {
  if (!previous || previous.feet[i].swing || pose.feet[i].swing || previous.feet[i].plantId !== pose.feet[i].plantId) return;
  const side = i ? 'L' : 'R';
  for (const name of ['ankle', 'toe']) assert.ok(distance(previous.landmarks[name + side], pose.landmarks[name + side]) < 1e-7, `planted ${name} ${side} drifted`);
  assert.equal(pose.feet[i].yaw, previous.feet[i].yaw, 'planted foot rotated');
}

test('forward, backward and both strafes alternate clear lifts and plant in their travel direction', () => {
  for (const velocity of [[0, 0, .6], [0, 0, -.6], [.6, 0, 0], [-.6, 0, 0]]) {
    const character = createCharacter(0), lifts = [0, 0], landings = [0, 0], maxLift = [0, 0];
    let previous, lastLanded = [null, null];
    try {
      for (let frame = 0; frame < 180; frame++) {
        const t = frame / 60, pose = updateMotion(character, t, [velocity[0] * t, .25, velocity[2] * t]);
        assertLengths(pose);
        assert.ok(!(pose.feet[0].swing && pose.feet[1].swing), 'both feet swing together');
        for (let i = 0; i < 2; i++) {
          assertPlanted(previous, pose, i);
          const foot = pose['foot' + i]; maxLift[i] = Math.max(maxLift[i], foot[1]);
          if (previous && pose.feet[i].swing && !previous.feet[i].swing) lifts[i]++;
          if (previous && !pose.feet[i].swing && previous.feet[i].swing) {
            landings[i]++;
            if (lastLanded[i]) {
              const along = (foot[0] - lastLanded[i][0]) * velocity[0] + (foot[2] - lastLanded[i][2]) * velocity[2];
              assert.ok(along > .01, 'a landing went against travel direction');
            }
            lastLanded[i] = foot;
          }
          if (previous && !pose.feet[i].swing && !previous.feet[i].swing) {
            assert.ok(Math.abs(foot[0] - previous['foot' + i][0]) < 1e-10);
            assert.ok(Math.abs(foot[2] - previous['foot' + i][2]) < 1e-10, 'stance foot slides');
          }
          assert.ok(foot[1] >= 0);
        }
        previous = pose;
      }
      assert.ok(lifts.every(n => n >= 5) && landings.every(n => n >= 4));
      assert.ok(maxLift.every(y => y > .045), 'swing is not visibly lifted');
    } finally { character.dispose(); }
  }
});

test('a sideways step is expressed relative to facing, without turning the physical root toward travel', () => {
  const character = createCharacter(0);
  try {
    updateMotion(character, 0, [0, .25, 0], [1, 0, 0]);
    const pose = updateMotion(character, .02, [0, .25, -.012], [1, 0, 0]);
    assert.ok(pose.localVelocity[0] > 1 && Math.abs(pose.localVelocity[2]) < 1e-6);
    assert.ok(Math.abs(character.root.rotation.y - Math.PI / 2) < 1e-9);
    const step = pose.feet.find(f => f.swing);
    assert.ok(step.target[2] < -.03);
    assert.ok(Math.abs(step.target[0]) < .001, 'strafe target was placed forward');
  } finally { character.dispose(); }
});

test('turning in place replants alternating feet without translating the root or sliding the stance', () => {
  const character = createCharacter(1), lifted = new Set();
  let previous;
  try {
    for (let frame = 0; frame < 210; frame++) {
      const t = frame / 60, angle = t * 1.6;
      const pose = updateMotion(character, t, [2, .25, 3], [Math.sin(angle), 0, Math.cos(angle)]);
      assertLengths(pose);
      assert.deepEqual(character.root.position.toArray(), [2, 0, 3]);
      for (let i = 0; i < 2; i++) {
        assertPlanted(previous, pose, i);
        if (pose['foot' + i][1] > .02) lifted.add(i);
        if (previous && !pose.feet[i].swing && !previous.feet[i].swing)
          assert.ok(distance(pose['foot' + i], previous['foot' + i]) < 1e-9);
      }
      previous = pose;
    }
    assert.equal(lifted.size, 2);
  } finally { character.dispose(); }
});

test('running, reversals and moving grips preserve limb lengths and settle without foot drift', () => {
  for (const gripId of [null, 0]) {
    const character = createCharacter(0);
    let x = 0, z = 0, previous;
    try {
      for (let frame = 0; frame < 240; frame++) {
        const t = frame / 60;
        const vx = frame < 60 ? 1.6 : frame < 120 ? -1.6 : frame < 180 ? 0 : 0;
        const vz = frame >= 120 && frame < 180 ? .8 : 0;
        if (frame) { x += vx / 60; z += vz / 60; }
        const pose = updateMotion(character, t, [x, .25, z], [0, 0, 1], { gripId, handTarget: [x + .16, .48, z + .19] });
        assertLengths(pose);
        for (let i = 0; i < 2; i++) assertPlanted(previous, pose, i);
        assert.ok(pose.landmarks.head.every(Number.isFinite));
        if (frame >= 180) {
          assert.equal(pose.foot0[1], 0); assert.equal(pose.foot1[1], 0);
          if (frame > 180) {
            assert.deepEqual(pose.foot0, previous.foot0); assert.deepEqual(pose.foot1, previous.foot1);
          }
        }
        previous = pose;
      }
    } finally { character.dispose(); }
  }
});

test('pause redraws are stable, and rewinds, teleport and landing reset safely', () => {
  const character = createCharacter(0);
  try {
    let pose;
    for (let i = 0; i < 18; i++) pose = updateMotion(character, i / 60, [i * .009, .25, 0]);
    const paused = updateMotion(character, 17 / 60, [17 * .009, .25, 0]);
    assert.deepEqual(paused.landmarks, pose.landmarks);
    updateMotion(character, .4, [.19, .6, 0], undefined, { grounded: false });
    const landed = updateMotion(character, .42, [.2, .25, 0]);
    assert.equal(landed.foot0[1], 0); assert.equal(landed.foot1[1], 0); assertLengths(landed);
    const reset = updateMotion(character, 0, [6, .25, 7]);
    assert.equal(reset.foot0[1], 0); assert.equal(reset.foot1[1], 0); assertLengths(reset);
    assert.ok(distance(reset.landmarks.pelvis, [6, .959 * scale, 7]) < .001);
  } finally { character.dispose(); }
});

test('high-speed travel and sparse physics-step renders keep both legs connected', () => {
  // The actual learned replay peaks at 1.956 m/s. Sustained 4 m/s is a stress
  // case, including 12.5 Hz single-step playback and sudden reversals.
  // 7.5 / 3.75 simulation frames per second also cover 4× playback rendered
  // at 30 / 15 display FPS; physics time remains authoritative.
  for (const fps of [60, 30, 12.5, 7.5, 3.75]) for (const speed of [1.956, 4]) {
    const character = createCharacter(0);
    let x = 0, z = 0;
    try {
      for (let frame = 0; frame < fps * 4; frame++) {
        const t = frame / fps, sign = t < 1 ? 1 : t < 2 ? -1 : 0;
        if (frame) { x += sign * speed / fps; z += t >= 2 && t < 3 ? speed / fps : 0; }
        const pose = updateMotion(character, t, [x, .25, z]);
        assertLengths(pose);
        assert.deepEqual(character.root.position.toArray(), [x, 0, z]);
        const paused = updateMotion(character, t, [x, .25, z]);
        assert.deepEqual(paused.landmarks, pose.landmarks);
      }
    } finally { character.dispose(); }
  }
});

test('short start-stop bursts continue alternating rather than restarting the same leg', () => {
  const character = createCharacter(0), lifts = [0, 0];
  let x = 0, previous;
  try {
    for (let frame = 0; frame < 180; frame++) {
      const moving = frame % 18 < 12;
      if (frame && moving) x += .004;
      const pose = updateMotion(character, frame / 60, [x, .25, 0]);
      for (let i = 0; i < 2; i++) if (previous && !previous.feet[i].swing && pose.feet[i].swing) lifts[i]++;
      previous = pose;
    }
    assert.ok(lifts.every(n => n >= 4), `uneven restart footfalls: ${lifts}`);
  } finally { character.dispose(); }
});

test('brief grip changes blend the spectator reach while reporting physical grip immediately', () => {
  const character = createCharacter(0);
  try {
    let previous = updateMotion(character, 0, [0, .25, 0]);
    for (let frame = 1; frame <= 100; frame++) {
      const grip = frame < 50;
      const pose = updateMotion(character, frame / 60, [0, .25, 0], undefined,
        { gripId: grip ? 0 : null, handTarget: grip ? [.05, .45, .23] : null });
      assert.equal(pose.gripping, grip);
      assertLengths(pose);
      for (const side of ['L','R']) assert.ok(distance(pose.landmarks['hand' + side], previous.landmarks['hand' + side]) < .075);
      if (frame === 1) assert.ok(pose.reachWeight > 0 && pose.reachWeight < 1);
      if (frame === 48) assert.equal(pose.reachWeight, 1);
      if (frame === 96) assert.equal(pose.reachWeight, 0);
      previous = pose;
    }
  } finally { character.dispose(); }
});

test('alternating button decisions do not flap stationary arms through full reach', () => {
  const character = createCharacter(0), samples = [];
  try {
    updateMotion(character, 0, [0, .25, 0]);
    for (let frame = 1; frame < 120; frame++) {
      const time = frame / 60, grip = Math.floor(time / .08) % 2 === 0;
      const pose = updateMotion(character, time, [0, .25, 0], undefined,
        { gripId: grip ? 0 : null, handTarget: grip ? [.05, .45, .23] : null });
      assert.equal(pose.gripping, grip); assertLengths(pose);
      assert.equal(pose.feet.some(f => f.swing), false);
      if (time > .4) samples.push(pose);
    }
    assert.ok(Math.max(...samples.map(p => p.reachWeight)) - Math.min(...samples.map(p => p.reachWeight)) < .4);
  } finally { character.dispose(); }
});

test('idle soles respect soft floor contacts and planted feet follow the support plane', () => {
  const character = createCharacter(0);
  try {
    const contacts = [{ point: [0, -.0001, 0], normal: [0, 1, 0] }];
    const idle = updateMotion(character, 0, [0, .2499, 0], undefined, { contacts });
    assert.equal(idle.foot0[1], 0); assert.equal(idle.foot1[1], 0);
    const normal = [-.3, Math.sqrt(.91), 0], point = [2, .3, 0];
    const pose = updateMotion(character, .1, [2, .55, 0], [0, 0, 1], { contacts: [{ point, normal }] });
    for (const side of ['L', 'R']) {
      const toe = pose.landmarks['toe' + side];
      const clearance = toe.reduce((sum, x, i) => sum + (x - point[i]) * normal[i], 0) - .038 * scale;
      assert.ok(clearance >= -1e-7, 'toe intersects the ramp');
    }
    assertLengths(pose);
  } finally { character.dispose(); }
});

test('a planted ankle and toe retain their world position and orientation when the support normal changes', () => {
  const character = createCharacter(0);
  try {
    let previous = updateMotion(character, 0, [0, .25, 0], undefined,
      { support: { point: [0, 0, 0], normal: [0, 1, 0] } });
    for (let frame = 1; frame <= 24; frame++) {
      const pose = updateMotion(character, frame / 60, [0, .25, 0], undefined,
        { support: { point: [0, 0, 0], normal: [-.3, Math.sqrt(.91), 0] } });
      for (let i = 0; i < 2; i++) assertPlanted(previous, pose, i);
      assertLengths(pose); previous = pose;
    }
  } finally { character.dispose(); }
});
