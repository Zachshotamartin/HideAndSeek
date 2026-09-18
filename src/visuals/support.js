import * as THREE from 'three';

const UP = new THREE.Vector3(0, 1, 0);
const vector = a => new THREE.Vector3(...a);
const rotation = frame => new THREE.Quaternion().fromArray(frame.quaternion ?? [0, 0, 0, 1]);
export const supportFrames = agent => agent.supportFrames ?? (agent.support?.transform ? [{ objectId: agent.support.objectId, ...agent.support.transform }] : []);
export const frameById = (frames, id) => frames.find(frame => frame.objectId === id);
export const toLocal = (point, frame) => point.clone().sub(vector(frame.position)).applyQuaternion(rotation(frame).invert());
export const toWorld = (point, frame) => point.clone().applyQuaternion(rotation(frame)).add(vector(frame.position));
export const toLocalDirection = (direction, frame) => direction.clone().applyQuaternion(rotation(frame).invert());
export const toWorldDirection = (direction, frame) => direction.clone().applyQuaternion(rotation(frame));

export function interpolateFrame(a, b, alpha) {
  if (!a) return b;
  return { ...b,
    position: a.position.map((value, i) => value + (b.position[i] - value) * alpha),
    quaternion: rotation(a).slerp(rotation(b), alpha).toArray(),
  };
}
export function interpolateFrames(previous, current, alpha) {
  return current.map(frame => interpolateFrame(frameById(previous, frame.objectId), frame, alpha));
}
export function interpolateSupport(a, b, alpha, frames) {
  if (!b) return b;
  const frame = b.transform && frameById(frames, b.objectId);
  if (frame) return { ...b, transform: frame,
    point: toWorld(toLocal(vector(b.point), b.transform), frame).toArray(),
    normal: toWorldDirection(toLocalDirection(vector(b.normal), b.transform), frame).toArray(),
  };
  if (!a || a.geomId !== b.geomId) return b;
  return { ...b, point: a.point.map((value, i) => value + (b.point[i] - value) * alpha),
    normal: vector(a.normal).lerp(vector(b.normal), alpha).normalize().toArray() };
}

function surfaceInfo(frame, heading, scale) {
  const [width, height, depth] = frame.size;
  const slope = frame.kind === 'ramp' ? height / width : 0;
  const localNormal = new THREE.Vector3(-slope, 1, 0).normalize();
  const normal = toWorldDirection(localNormal, frame).normalize();
  if (normal.y < .45) return null;
  const facing = new THREE.Vector3(Math.sin(heading), 0, Math.cos(heading));
  facing.addScaledVector(normal, -facing.dot(normal)).normalize();
  const toe = toLocalDirection(facing.multiplyScalar(.17 * scale), frame);
  // Both the ankle and toe footprint fit on the finite upper face.
  const margin = .09 * scale;
  return { slope, normal,
    minX: -width / 2 + margin - Math.min(0, toe.x), maxX: width / 2 - margin - Math.max(0, toe.x),
    minZ: -depth / 2 + margin - Math.min(0, toe.z), maxZ: depth / 2 - margin - Math.max(0, toe.z),
    planeY: x => frame.kind === 'ramp' ? slope * x : height / 2,
  };
}
function onSurface(point, frame, heading, scale, clampToEdge) {
  const info = surfaceInfo(frame, heading, scale);
  if (!info || info.minX > info.maxX || info.minZ > info.maxZ) return null;
  const localPoint = toLocal(point, frame);
  const down = toLocalDirection(new THREE.Vector3(0, -1, 0), frame);
  const divisor = down.y - info.slope * down.x;
  if (Math.abs(divisor) < 1e-8) return null;
  const distance = (info.planeY(localPoint.x) - localPoint.y) / divisor;
  localPoint.addScaledVector(down, distance);
  const inside = localPoint.x >= info.minX && localPoint.x <= info.maxX && localPoint.z >= info.minZ && localPoint.z <= info.maxZ;
  if (!inside && !clampToEdge) return null;
  localPoint.x = THREE.MathUtils.clamp(localPoint.x, info.minX, info.maxX);
  localPoint.z = THREE.MathUtils.clamp(localPoint.z, info.minZ, info.maxZ);
  localPoint.y = info.planeY(localPoint.x);
  return { point: toWorld(localPoint, frame), normal: info.normal, frame: frame.objectId === undefined ? null : frame };
}
function insideSolid(point, frame) {
  const p = toLocal(point, frame), [width, height, depth] = frame.size;
  return Math.abs(p.x) < width / 2 && Math.abs(p.z) < depth / 2 && p.y > -height / 2 - .002 && p.y < (frame.kind === 'ramp' ? height / width * p.x : height / 2) + .002;
}

/** Finite rendered/physical box and wedge faces, plus the actual floor plane. */
export function findFoothold(desired, { frames = [], surfaces = [], heading = 0, scale = 1, minHeight = -Infinity, maxHeight = Infinity } = {}) {
  const geometry = [...frames, ...surfaces].filter(frame => frame.size && frame.position);
  const candidates = [];
  const test = point => {
    const hits = geometry.map(frame => onSurface(point, frame, heading, scale, false)).filter(Boolean);
    const floor = point.clone(); floor.y = 0;
    hits.push({ point: floor, normal: UP.clone(), frame: null });
    for (const hit of hits) {
      if (hit.point.y < minHeight || hit.point.y > maxHeight) continue;
      const front = new THREE.Vector3(Math.sin(heading), 0, Math.cos(heading));
      front.addScaledVector(hit.normal, -front.dot(hit.normal)).normalize().multiplyScalar(.17 * scale);
      const probes = [hit.point.clone().addScaledVector(hit.normal, .045 * scale), hit.point.clone().add(front).addScaledVector(hit.normal, .045 * scale)];
      if (geometry.some(frame => frame !== hit.frame && probes.some(probe => insideSolid(probe, frame)))) continue;
      candidates.push(hit);
    }
  };
  test(desired);
  if (!candidates.length) {
    // An off-edge step on a raised support must not plant on its infinite plane.
    for (const frame of geometry) {
      const nearest = onSurface(desired, frame, heading, scale, true);
      if (nearest) test(nearest.point);
    }
    // Near an obstacle, try short alternatives rather than putting the foot
    // through the prop onto the floor hidden below it.
    for (const radius of [.08, .16, .24]) for (let angle = 0; angle < 8; angle++)
      test(desired.clone().add(new THREE.Vector3(Math.cos(angle * Math.PI / 4) * radius, 0, Math.sin(angle * Math.PI / 4) * radius)));
  }
  candidates.sort((a, b) => {
    const da = Math.hypot(a.point.x - desired.x, a.point.z - desired.z), db = Math.hypot(b.point.x - desired.x, b.point.z - desired.z);
    return Math.abs(da - db) > 1e-6 ? da - db : b.point.y - a.point.y;
  });
  return candidates.find(hit => Math.hypot(hit.point.x - desired.x, hit.point.z - desired.z) <= .35) ?? null;
}
