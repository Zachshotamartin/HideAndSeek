import * as THREE from "three";
import { createStickFigure, MODEL, NEUTRAL_POSE } from "@zachshotamartin/stick-figure";
import { supportFrames, frameById, toLocal, toWorld, toLocalDirection, toWorldDirection, interpolateFrames, interpolateSupport, findFoothold } from './support.js';

const UP = new THREE.Vector3(0, 1, 0);
const v = (a) => new THREE.Vector3(...a);
const clamp = THREE.MathUtils.clamp;
const angleDelta = (a, b) => Math.atan2(Math.sin(a - b), Math.cos(a - b));

/** Procedural footfall/contact presentation for the physical sphere agents.
 * Only the physical root and facing are authoritative; this gait is not learned. */
export function createCharacter(role, onReady) {
  const skin = createStickFigure({ color: role ? "#d9514c" : "#348cdd", onReady });
  const root = skin.root;
  const feet = [0, 1].map(() => ({
    anchor: null, start: null, target: null, swing: false,
    progress: 0, duration: 0, yaw: 0, startYaw: 0, targetYaw: 0,
    normal: UP.clone(), startNormal: UP.clone(), targetNormal: UP.clone(), plantId: 0,
    direction: new THREE.Vector3(0, 0, 1), targetDirection: new THREE.Vector3(0, 0, 1), searching: false, attachment: null, landingAttachment: null,
  }));
  let lastPosition = null, lastTime = null, lastHeading = 0;
  let nextFoot = 0, stepClock = 1, wasGrounded = true;
  let lastVelocity = new THREE.Vector3(), lastTurnRate = 0;
  let reachWeight = 0, lastHandTarget = null;
  const armOffsets = [new THREE.Vector3(), new THREE.Vector3()];
  const presentationVelocity = new THREE.Vector3();
  let lastFrames = [], lastSupport = null;
  const facing = (yaw, normal) => {
    const direction = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
    return direction.addScaledVector(normal, -direction.dot(normal)).normalize();
  };
  const attachment = (point, normal, direction, frame) => frame ? {
    objectId: frame.objectId, point: toLocal(point, frame),
    normal: toLocalDirection(normal, frame), direction: toLocalDirection(direction, frame),
  } : null;
  const resolveAttachment = (saved, frames) => {
    const frame = saved && frameById(frames, saved.objectId);
    return frame ? { point: toWorld(saved.point, frame), normal: toWorldDirection(saved.normal, frame).normalize(),
      direction: toWorldDirection(saved.direction, frame).normalize(), frame } : null;
  };

  function solve(a, b, lengthA, lengthB, pole) {
    const delta = b.clone().sub(a);
    const d = clamp(delta.length(), .0001, lengthA + lengthB - .0001);
    delta.normalize();
    const along = (lengthA ** 2 - lengthB ** 2 + d ** 2) / (2 * d);
    const bend = Math.sqrt(Math.max(0, lengthA ** 2 - along ** 2));
    const plane = pole.clone().addScaledVector(delta, -pole.dot(delta));
    if (plane.lengthSq() < 1e-8) plane.copy(UP).addScaledVector(delta, -delta.y);
    return a.clone().addScaledVector(delta, along).addScaledVector(plane.normalize(), bend);
  }

  function updateFrame(agent, time) {
    const position = v(agent.position);
    const scale = (agent.visualHeight ?? .70) / MODEL.height;
    const grounded = agent.grounded !== false;
    const frames = supportFrames(agent);
    const gripping = agent.gripId !== null && agent.gripId !== undefined;
    const support = agent.support ?? agent.contacts?.find(c => c.normal?.[1] > .45);
    const normal = support ? v(support.normal).normalize() : UP.clone();
    const surface = (point) => Math.max(0, support
      ? support.point[1] - (normal.x * (point.x - support.point[0]) + normal.z * (point.z - support.point[2])) / normal.y
      : (agent.groundHeight ?? position.y - (agent.colliderHeight ?? .5) / 2));
    const forward = agent.forward ? v(agent.forward) : new THREE.Vector3(Math.cos(agent.heading || 0), 0, -Math.sin(agent.heading || 0));
    forward.y = 0;
    if (forward.lengthSq() < 1e-8) forward.set(0, 0, 1);
    forward.normalize();
    const heading = Math.atan2(forward.x, forward.z);
    root.scale.setScalar(scale);
    root.position.set(position.x, surface(position), position.z);
    root.rotation.y = heading;
    const local = world => world.clone().sub(root.position).applyAxisAngle(UP, -heading).divideScalar(scale);
    const world = point => point.clone().multiplyScalar(scale).applyAxisAngle(UP, heading).add(root.position);
    const neutral = i => world(new THREE.Vector3(NEUTRAL_POSE[i ? 'ankleL' : 'ankleR'][0], 0, 0));
    const rawDt = lastTime === null ? 0 : time - lastTime;
    let externalFootSpeed = 0;
    for (const f of feet) {
      if (!grounded) { f.attachment = f.landingAttachment = null; continue; }
      const carried = !f.swing && resolveAttachment(f.attachment, frames);
      if (carried) {
        if (rawDt > 0 && lastPosition) {
          const change = carried.point.clone().sub(f.anchor).sub(position.clone().sub(lastPosition)); change.y = 0;
          externalFootSpeed = Math.max(externalFootSpeed, change.length() / (rawDt * scale));
        }
        f.anchor.copy(carried.point); f.normal.copy(carried.normal); f.direction.copy(carried.direction);
        f.yaw = Math.atan2(f.direction.x, f.direction.z);
      }
    }
    let carriedPosition = lastPosition, carriedHeading = lastHeading;
    if (lastPosition && grounded && wasGrounded && support?.objectId !== undefined && lastSupport?.objectId === support.objectId) {
      const before = frameById(lastFrames, support.objectId), after = frameById(frames, support.objectId);
      if (before && after) {
        carriedPosition = toWorld(toLocal(lastPosition, before), after);
        const carriedForward = toWorldDirection(toLocalDirection(new THREE.Vector3(Math.sin(lastHeading), 0, Math.cos(lastHeading)), before), after);
        carriedHeading = Math.atan2(carriedForward.x, carriedForward.z);
      }
    }
    const delta = carriedPosition ? position.clone().sub(carriedPosition) : new THREE.Vector3();
    delta.y = 0;
    const reset = !lastPosition || rawDt < -1e-6 || delta.length() > 1 || (grounded && !wasGrounded);
    const dt = reset ? 0 : clamp(rawDt, 0, .1);
    const velocity = dt > 1e-6 ? delta.clone().divideScalar(dt * scale) : reset ? new THREE.Vector3() : lastVelocity.clone();
    const localVelocity = velocity.clone().applyAxisAngle(UP, -heading);
    const speed = velocity.length();
    const turnRate = dt > 1e-6 ? angleDelta(heading, carriedHeading) / dt : reset ? 0 : lastTurnRate;
    const moving = speed > .08 && delta.length() > .00001;
    const turning = Math.abs(turnRate) > .12;
    const displaced = feet.some((f, i) => f.attachment && f.anchor?.distanceTo(neutral(i)) / scale > .36);
    const active = grounded && (moving || turning || displaced);
    const gaitSpeed = Math.max(speed, displaced ? externalFootSpeed : 0);
    const cadence = clamp(2.2 + gaitSpeed * 1.05 + Math.abs(turnRate) * .3, 2.2, 24);
    const interval = 1 / cadence;
    const foothold = (desired, yaw = heading) => {
      if (frames.some(frame => frame.size) || agent.supportSurfaces) return findFoothold(desired, {
        frames, surfaces: agent.supportSurfaces, heading: yaw, scale,
        minHeight: root.position.y - .5 * scale, maxHeight: root.position.y + .35 * scale,
      });
      const point = desired.clone(); point.y = surface(point);
      return { point, normal: normal.clone(), frame: frameById(frames, support?.objectId) ?? null };
    };
    const landingTarget = (i, duration, yaw) => {
      const lead = velocity.clone().multiplyScalar(duration + .70 * interval);
      if (lead.length() > .90) lead.setLength(.90);
      const target = neutral(i).addScaledVector(lead, scale);
      return foothold(target, yaw) ?? foothold(neutral(i), yaw) ?? foothold(feet[i].anchor, yaw);
    };
    const setLanding = (f, hit, yaw) => {
      f.target = hit.point; f.targetNormal.copy(hit.normal); f.targetYaw = yaw;
      f.targetDirection.copy(facing(yaw, hit.normal)); f.searching = false;
      f.landingAttachment = attachment(hit.point, hit.normal, facing(yaw, hit.normal), hit.frame);
    };
    const plant = (f, hit) => {
      f.anchor.copy(hit.point); f.normal.copy(hit.normal); f.direction.copy(facing(f.yaw, f.normal));
      f.attachment = attachment(f.anchor, f.normal, f.direction, hit.frame); f.plantId++;
    };

    if (reset) {
      feet.forEach((f, i) => {
        f.anchor = neutral(i);
        f.start = f.target = null; f.swing = false; f.progress = 0; f.yaw = heading;
        f.landingAttachment = null;
        const hit = foothold(f.anchor);
        if (hit) { plant(f, hit); f.searching = false; }
        else {
          // No finite reachable surface: keep the foot unplanted. Never invent
          // a foothold on the infinite extension of the body support plane.
          f.anchor.y = surface(position); f.normal.copy(normal); f.direction.copy(facing(heading, normal));
          f.attachment = null; f.swing = f.searching = true;
          f.start = f.target = f.anchor.clone(); f.duration = interval; f.startNormal.copy(normal);
        }
      });
      nextFoot = localVelocity.x > 0 ? 1 : 0;
      stepClock = interval;
      presentationVelocity.set(0, 0, 0); armOffsets.forEach(offset => offset.set(0, 0, 0));
    }

    if (dt > 0 && active) {
      stepClock += dt;
      for (const [i, f] of feet.entries()) {
        if (!f.swing) continue;
        if (f.searching) {
          const hit = landingTarget(i, .7 * interval, heading);
          if (!hit) continue;
          f.start = f.anchor.clone(); setLanding(f, hit, heading);
          f.progress = 0; f.duration = .7 * interval; f.startYaw = f.yaw; f.startNormal.copy(f.normal);
        }
        // A sudden reversal must not finish a now-unreachable old landing.
        // Continue from the current lifted position into the new direction.
        if (speed > .3 && lastVelocity.length() > .3 && velocity.dot(lastVelocity) < .5 * speed * lastVelocity.length()) {
          const hit = landingTarget(i, .7 * interval, heading);
          if (!hit) { f.searching = true; continue; }
          f.start = f.anchor.clone(); setLanding(f, hit, heading);
          f.progress = 0; f.duration = .7 * interval; f.startYaw = f.yaw;
          f.startNormal.copy(f.normal);
        }
        const carriedTarget = resolveAttachment(f.landingAttachment, frames);
        if (carriedTarget) {
          f.target.copy(carriedTarget.point); f.targetNormal.copy(carriedTarget.normal); f.targetDirection.copy(carriedTarget.direction);
          f.targetYaw = Math.atan2(carriedTarget.direction.x, carriedTarget.direction.z);
        }
        const other = feet[1 - i];
        const stanceReach = other.anchor.distanceTo(neutral(1 - i)) / scale;
        const timeBeforeLimit = Math.max(dt, (.74 - stanceReach) / Math.max(.1, gaitSpeed));
        f.progress = Math.min(1, f.progress + Math.max(dt / f.duration, (1 - f.progress) * dt / timeBeforeLimit));
        const t = f.progress, ease = t * t * (3 - 2 * t);
        f.anchor.copy(f.start).lerp(f.target, ease);
        // A clear swing arc; stance anchors do not follow the body's motion.
        f.anchor.y = THREE.MathUtils.lerp(f.start.y, f.target.y, ease) + Math.sin(Math.PI * t) * (moving ? .15 : .09) * scale;
        f.yaw = f.startYaw + angleDelta(f.targetYaw, f.startYaw) * ease;
        f.normal.copy(f.startNormal).lerp(f.targetNormal, ease).normalize();
        f.direction.copy(facing(f.yaw, f.normal));
        if (t === 1) {
          f.anchor.copy(f.target); f.direction.copy(f.targetDirection); f.swing = false; f.plantId++;
          f.attachment = f.landingAttachment; f.landingAttachment = null;
        }
      }
      const reachRisk = feet.map((f, i) => f.anchor.distanceTo(neutral(i).addScaledVector(velocity, .7 * interval * scale)) / scale);
      const urgentFoot = reachRisk[0] > reachRisk[1] ? 0 : 1;
      const urgent = reachRisk[urgentFoot] > .74;
      if (!feet.some(f => f.swing) && (stepClock >= interval || urgent)) {
        // Strafe with the leading foot first; then alternate. The target predicts
        // root travel during swing, regardless of the character's facing.
        if (feet.every((f, i) => f.anchor.distanceTo(neutral(i)) < .015 * scale))
          nextFoot = localVelocity.x > .1 ? 1 : 0;
        if (urgent) nextFoot = urgentFoot;
        const f = feet[nextFoot], duration = .7 * interval;
        f.start = f.anchor.clone();
        f.startYaw = f.yaw;
        const yaw = heading + clamp(turnRate * duration, -.3, .3);
        const hit = landingTarget(nextFoot, duration, yaw);
        if (hit) setLanding(f, hit, yaw);
        else { f.searching = true; f.target = f.anchor.clone(); }
        f.startNormal.copy(f.normal); f.attachment = null;
        f.progress = 0; f.duration = duration; f.swing = true;
        nextFoot = 1 - nextFoot; stepClock = Math.max(0, stepClock - interval);
      }
    } else if (dt > 0 || reset) {
      // Stop immediately, including midway through a swing and while gripping
      // prop 0. Retain the world anchors instead of sliding toward neutral.
      for (const f of feet) {
        if (f.swing) {
          const hit = foothold(f.anchor, f.yaw) ?? foothold(neutral(feet.indexOf(f)), f.yaw);
          if (!hit) { f.searching = true; continue; }
          plant(f, hit); f.searching = false;
          f.landingAttachment = null;
        }
        f.swing = false; f.progress = 0;
      }
      stepClock = interval;
    }
    if (!grounded) {
      feet.forEach((f, i) => {
        f.anchor.copy(world(new THREE.Vector3(i ? .105 : -.105, .10, -.06)));
        f.swing = false; f.yaw = heading; f.normal.copy(UP); f.direction.copy(facing(heading, UP));
      });
    }
    // Teleport/reset recovery is above; normal motion never snaps a foot to the
    // root. Inactive render calls (camera movement, pause) do not advance gait.
    lastPosition = position.clone(); lastTime = time; lastHeading = heading; wasGrounded = grounded;
    lastFrames = frames; lastSupport = support;
    if (dt > 0 || reset) { lastVelocity = velocity.clone(); lastTurnRate = turnRate; }

    const ankleWorld = feet.map(f => f.anchor.clone().addScaledVector(f.normal, NEUTRAL_POSE.ankleL[1] * scale));
    const ankles = ankleWorld.map(local);
    presentationVelocity.lerp(localVelocity, 1 - Math.exp(-dt / .08));
    for (let i = 0; i < 2; i++) {
      // Opposite-arm motion follows the visible step, not a separate oscillator
      // that can shake rapidly while the agent slides or changes direction.
      const target = ankles[i].clone().sub(v(NEUTRAL_POSE[i ? 'ankleL' : 'ankleR']));
      target.y = 0;
      target.multiplyScalar(-.35 * clamp(presentationVelocity.length() * scale / .3, 0, 1));
      if (target.length() > .24) target.setLength(.24);
      armOffsets[i].lerp(target, 1 - Math.exp(-dt / .055));
    }
    const reaching = !!agent.handTarget && (gripping || !!agent.contacts?.length);
    if (reaching) lastHandTarget = v(agent.handTarget);
    // Physical grip state is immediate. A settling spectator blend prevents
    // alternating 80 ms button presses from flapping the arms through full reach.
    reachWeight = reset ? Number(reaching) : THREE.MathUtils.lerp(reachWeight, Number(reaching), 1 - Math.exp(-dt / .12));
    if (reachWeight < .005) reachWeight = 0;
    if (reachWeight > .995) reachWeight = 1;
    const reachTarget = reachWeight > 0 && lastHandTarget ? local(lastHandTarget) : null;
    const reachingPose = !!reachTarget;
    const lean = new THREE.Euler(
      THREE.MathUtils.lerp(clamp(presentationVelocity.z * .025, -.13, .13), reachingPose ? clamp(reachTarget.z * .28, -.30, .30) : 0, reachWeight),
      0,
      THREE.MathUtils.lerp(clamp(-presentationVelocity.x * .025 - turnRate * .01, -.12, .12), reachingPose ? clamp(-reachTarget.x * .25, -.24, .24) : 0, reachWeight),
    );
    const rotation = new THREE.Quaternion().setFromEuler(lean);
    const pivot = v(NEUTRAL_POSE.pelvis);
    const shift = new THREE.Vector3();
    const upperPoint = name => v(NEUTRAL_POSE[name]).sub(pivot).applyQuaternion(rotation).add(pivot).add(shift);
    if (reachingPose) {
      for (const side of [-1, 1]) {
        const shoulder = upperPoint('shoulder' + (side > 0 ? 'L' : 'R'));
        const reach = reachTarget.clone().add(new THREE.Vector3(side * .045, 0, 0)).sub(shoulder);
        const extra = Math.max(0, reach.length() - (MODEL.upperArm + MODEL.forearm + MODEL.hand - .01));
        if (extra) shift.addScaledVector(reach.normalize(), extra * .5);
      }
      if (shift.length() > .40) shift.setLength(.40);
      shift.multiplyScalar(reachWeight);
    }
    // Lower the pelvis only as far as the actual stance requires. A straight
    // neutral pose has full-length legs; displaced feet must bend the knees
    // instead of silently stretching the lower leg.
    let hipDrop = 0;
    for (let i = 0; i < 2; i++) {
      const hip = v(NEUTRAL_POSE[i ? 'hipL' : 'hipR']).add(shift);
      const horizontal = Math.hypot(ankles[i].x - hip.x, ankles[i].z - hip.z);
      const reach = MODEL.thigh + MODEL.shin - .0002;
      hipDrop = Math.min(hipDrop, ankles[i].y + Math.sqrt(Math.max(.01, reach ** 2 - horizontal ** 2)) - hip.y);
    }
    shift.y += hipDrop;
    const posed = {};
    for (const name of ['pelvis', 'chest', 'neck', 'head']) posed[name] = upperPoint(name).toArray();
    const diagnostics = {};
    for (let i = 0; i < 2; i++) {
      const side = i ? 1 : -1, suffix = i ? 'L' : 'R', f = feet[i];
      const ankle = ankles[i], hip = v(NEUTRAL_POSE['hip' + suffix]).add(shift);
      const knee = solve(hip, ankle, MODEL.thigh, MODEL.shin, new THREE.Vector3(0, 0, 1));
      const footForward = f.direction.clone();
      const right = new THREE.Vector3().crossVectors(f.normal, footForward).normalize();
      const footRotation = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, f.normal, footForward));
      const pitch = f.swing ? -.22 * Math.sin(Math.PI * f.progress) : 0;
      footRotation.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), pitch));
      const toe = local(new THREE.Vector3(0, -.025, .17).applyQuaternion(footRotation).multiplyScalar(scale).add(ankleWorld[i]));
      const shoulder = upperPoint('shoulder' + suffix);
      let wrist, hand, pole;
      if (reachingPose) {
        hand = reachTarget.clone(); hand.x += side * .045;
        wrist = hand.clone().addScaledVector(hand.clone().sub(shoulder).normalize(), -MODEL.hand);
        const reach = wrist.clone().sub(shoulder);
        if (reach.length() > MODEL.upperArm + MODEL.forearm - .0002)
          wrist.copy(shoulder).add(reach.setLength(MODEL.upperArm + MODEL.forearm - .0002));
        hand.copy(wrist).addScaledVector(hand.clone().sub(wrist).normalize(), MODEL.hand);
        if (reachWeight < 1) {
          const neutralWrist = upperPoint('wrist' + suffix);
          wrist.lerp(neutralWrist, 1 - reachWeight);
          hand.lerp(neutralWrist.add(new THREE.Vector3(0, -MODEL.hand, 0)), 1 - reachWeight);
        }
        pole = new THREE.Vector3(side * .7, -.5, .5);
      } else {
        wrist = upperPoint('wrist' + suffix).add(armOffsets[i]);
        const reach = wrist.clone().sub(shoulder);
        if (reach.length() > MODEL.upperArm + MODEL.forearm - .0002)
          wrist.copy(shoulder).add(reach.setLength(MODEL.upperArm + MODEL.forearm - .0002));
        hand = wrist.clone().add(new THREE.Vector3(0, -MODEL.hand, 0));
        pole = new THREE.Vector3(side * .7, 0, .5);
      }
      const armReach = wrist.clone().sub(shoulder);
      const minimumReach = Math.abs(MODEL.upperArm - MODEL.forearm) + .0002;
      const maximumReach = MODEL.upperArm + MODEL.forearm - .0002;
      if (armReach.length() > maximumReach) {
        const correction = shoulder.clone().add(armReach.setLength(maximumReach)).sub(wrist);
        wrist.add(correction); hand.add(correction);
      }
      if (armReach.length() < minimumReach) {
        if (armReach.lengthSq() < 1e-8) armReach.set(0, -1, 0);
        const correction = shoulder.clone().add(armReach.setLength(minimumReach)).sub(wrist);
        wrist.add(correction); hand.add(correction);
      }
      const handDirection = hand.clone().sub(wrist);
      if (handDirection.lengthSq() < 1e-8) handDirection.set(0, -1, 0);
      hand.copy(wrist).add(handDirection.setLength(MODEL.hand));
      const elbow = solve(shoulder, wrist, MODEL.upperArm, MODEL.forearm, pole);
      for (const [name, point] of Object.entries({ hip, knee, ankle, toe, shoulder, elbow, wrist, hand })) posed[name + suffix] = point.toArray();
      diagnostics['foot' + i] = f.anchor.toArray();
      diagnostics['hand' + i] = world(hand).toArray();
    }
    skin.apply(posed);
    root.userData.pose = {
      ...diagnostics, speed, localVelocity: localVelocity.toArray(), turnRate,
      feet: feet.map(f => ({ swing: f.swing, progress: f.progress, yaw: f.yaw, normal: f.normal.toArray(), plantId: f.plantId, supportId: f.attachment?.objectId ?? null, target: f.target?.toArray() ?? null })),
      grounded, time, gripping, reachWeight, skin: skin.diagnostics(), landmarks: skin.getRenderedJoints(),
    };
  }
  function update(agent, time) {
    const elapsed = lastTime === null ? 0 : time - lastTime;
    const endpoint = v(agent.position);
    // Integrate the presentation at a bounded interval, including single-step
    // playback. Otherwise a slow render can skip an entire swing and leave a
    // planted leg behind the physical root. This never advances physics.
    if (lastPosition && wasGrounded === (agent.grounded !== false) && elapsed > 1 / 120 && elapsed <= .5 && endpoint.distanceTo(lastPosition) <= Math.max(1, elapsed * 5)) {
      const start = lastPosition.clone(), startTime = lastTime, yaw = lastHeading;
      const startFrames = lastFrames, startSupport = lastSupport, endFrames = supportFrames(agent);
      const forward = agent.forward ? v(agent.forward) : new THREE.Vector3(Math.cos(agent.heading || 0), 0, -Math.sin(agent.heading || 0));
      const turn = angleDelta(Math.atan2(forward.x, forward.z), yaw);
      const count = Math.ceil(elapsed * 120);
      for (let step = 1; step <= count; step++) {
        const alpha = step / count, heading = yaw + turn * alpha;
        const frames = interpolateFrames(startFrames, endFrames, alpha);
        updateFrame({ ...agent, supportFrames: frames, support: interpolateSupport(startSupport, agent.support, alpha, frames),
          position: start.clone().lerp(endpoint, alpha).toArray(), forward: [Math.sin(heading), 0, Math.cos(heading)] }, startTime + elapsed * alpha);
      }
    } else updateFrame(agent, time);
  }
  return { root, update, ready: skin.ready, dispose: skin.dispose };
}
