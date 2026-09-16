import { TAG_SCHEMA, extras } from './gameRules.js';
export const KNOWN_POSITION_SCHEMA = 'known-opponent-position-210-v1';
export const OBSERVATION_SCHEMAS = [KNOWN_POSITION_SCHEMA, TAG_SCHEMA];
/** Original physical hide-and-seek. Coordinates in this module are metres, Z up.
 * The renderer adapter alone converts to Three.js Y up.
 */
export const DT = 0.08,
  SUBSTEPS = 16,
  PREP = 96,
  PLAY = 144,
  VISION_RANGE = Infinity,
  FOV = (3 * Math.PI) / 4,
  OBJECT_SLOTS = 10,
  LIDAR = 30,
  OBS_DIM = 208;
export class Random {
  constructor(seed) {
    this.state = seed >>> 0 || 1;
  }
  next() {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state / 4294967296;
  }
  uniform(a, b) {
    return a + (b - a) * this.next();
  }
}
const tau = Math.PI * 2,
  clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export function generateArena(seed = 1, scenario = 'shelter', size = 8, nBoxes = 3, nRamps = 1) {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 4294967295)
    throw Error('Use an integer seed from 0 to 4,294,967,295.');
  if (!['shelter','rooms','open','connected-rooms','corridors','multi-exit'].includes(scenario)) throw Error('Unknown arena scenario.');
  if (
    !Number.isFinite(size) ||
    size < 6 ||
    size > 12 ||
    !Number.isInteger(nBoxes) ||
    nBoxes < 0 ||
    nBoxes > 8 ||
    !Number.isInteger(nRamps) ||
    nRamps < 0 ||
    nRamps > 2
  )
    throw Error('Use a 6–12 m arena, 0–8 boxes, and 0–2 ramps.');
  const r = new Random(seed),
    s = size,
    walls = [],
    objects = [];
  let agents;
  const wall = (x, y, sx, sy) =>
    walls.push({ position: [x, y, 1.1], size: [sx, sy, 2.2], yaw: 0 });
  for (const [x, y, sx, sy] of [
    [s / 2, -0.1, s + 0.4, 0.2],
    [s / 2, s + 0.1, s + 0.4, 0.2],
    [-0.1, s / 2, 0.2, s],
    [s + 0.1, s / 2, 0.2, s],
  ])
    wall(x, y, sx, sy);
  if (scenario === 'shelter') {
    const left = 0.12 * s,
      right = 0.5 * s,
      bottom = 0.22 * s,
      top = 0.78 * s,
      gap = 0.52;
    wall(left, (bottom + top) / 2, 0.18, top - bottom);
    wall((left + right) / 2, bottom, right - left, 0.18);
    wall((left + right) / 2, top, right - left, 0.18);
    wall(right, (bottom + s / 2 - gap) / 2, 0.18, s / 2 - gap - bottom);
    wall(right, (s / 2 + gap + top) / 2, 0.18, top - s / 2 - gap);
    agents = [
      { position: [s * 0.36, s * 0.5, 0.25], yaw: 0 },
      { position: [s * 0.78, s * 0.5, 0.25], yaw: Math.PI },
    ];
  } else if (scenario === 'rooms') {
    const split = r.uniform(0.4, 0.6) * s,
      gap = r.uniform(0.35, 0.65) * s;
    wall(split, gap / 2 - 0.35, 0.18, gap - 0.7);
    wall(split, (gap + 0.7 + s) / 2, 0.18, s - gap - 0.7);
    if (r.next() < 0.75) wall(s * 0.23, s * 0.55, s * 0.32, 0.18);
    agents = [
      { position: [s * 0.22, s * 0.22, 0.25], yaw: r.uniform(-Math.PI, Math.PI) },
      { position: [s * 0.78, s * 0.78, 0.25], yaw: r.uniform(-Math.PI, Math.PI) },
    ];
  } else if(['connected-rooms','corridors','multi-exit'].includes(scenario)) {
    const partition=(axis,fixed,lo,hi)=>{
      const span=hi-lo,centers=[lo+span*r.uniform(.23,.31),lo+span*r.uniform(.69,.77)],width=r.uniform(1.10,1.35);let start=lo;
      for(const center of centers){const end=center-width/2;if(end>start){if(axis===0)wall(fixed,(start+end)/2,.18,end-start);else wall((start+end)/2,fixed,end-start,.18);}start=center+width/2;}
      if(start<hi){if(axis===0)wall(fixed,(start+hi)/2,.18,hi-start);else wall((start+hi)/2,fixed,hi-start,.18);}
    };
    if(scenario==='connected-rooms'){partition(0,r.uniform(.43,.57)*s,0,s);partition(1,r.uniform(.43,.57)*s,0,s);}
    else if(scenario==='corridors'){const mid=r.uniform(.45,.55)*s,half=r.uniform(.75,1);partition(1,mid-half,0,s);partition(1,mid+half,0,s);}
    else {const left=.29*s,right=.71*s,bottom=.25*s,top=.75*s;
      for(const x of [left,right]){const gap=(bottom+top)/2+r.uniform(-.2,.2),w=1.2;wall(x,(bottom+gap-w/2)/2,.18,gap-w/2-bottom);wall(x,(gap+w/2+top)/2,.18,top-gap-w/2);}
      for(const y of [bottom,top]){const gap=(left+right)/2+r.uniform(-.2,.2),w=1.2;wall((left+gap-w/2)/2,y,gap-w/2-left,.18);wall((gap+w/2+right)/2,y,right-gap-w/2,.18);}
    }
    agents=[{position:[s*.15,s*.15,.25],yaw:r.uniform(-Math.PI,Math.PI)},{position:[s*.85,s*.85,.25],yaw:r.uniform(-Math.PI,Math.PI)}];
  } else
    agents = [
      { position: [s * 0.25, s * 0.35, 0.25], yaw: 0 },
      { position: [s * 0.75, s * 0.65, 0.25], yaw: Math.PI },
    ];
  const clear = (x, y, sx, sy) => {
    for (const o of [...walls, ...objects]) {
      const [px, py] = o.position,
        [ox, oy] = o.size,
        a = o.yaw || 0,
        ex = (Math.abs(Math.cos(a)) * ox + Math.abs(Math.sin(a)) * oy) / 2,
        ey = (Math.abs(Math.sin(a)) * ox + Math.abs(Math.cos(a)) * oy) / 2;
      if (Math.abs(x - px) < sx / 2 + ex + 0.12 && Math.abs(y - py) < sy / 2 + ey + 0.12)
        return false;
    }
    return agents.every(
      (a) => Math.hypot(x - a.position[0], y - a.position[1]) > 0.45 + Math.max(sx, sy) / 2,
    );
  };
  for (let i = 0; i < nBoxes + nRamps; i++) {
    const kind = i >= nBoxes ? 'ramp' : i % 2 === 0 ? 'plank' : 'box',
      dims =
        kind === 'ramp' ? [0.9, 0.8, 0.7] : kind === 'plank' ? [1.7, 0.25, 0.7] : [0.7, 0.7, 0.7];
    let placed = false,
      x,
      y,
      yaw;
    for (let attempt = 0; attempt < 500; attempt++) {
      if (scenario === 'shelter' && i === 0) {
        x = r.uniform(0.22, 0.4) * s;
        y = r.uniform(0.29, 0.37) * s;
      } else {
        x = r.uniform(0.9, s - 0.9);
        y = r.uniform(0.9, s - 0.9);
      }
      yaw = kind === 'plank' ? 0 : r.uniform(-Math.PI, Math.PI);
      const sx = Math.abs(Math.cos(yaw)) * dims[0] + Math.abs(Math.sin(yaw)) * dims[1],
        sy = Math.abs(Math.sin(yaw)) * dims[0] + Math.abs(Math.cos(yaw)) * dims[1];
      if (clear(x, y, sx, sy)) {
        placed = true;
        break;
      }
    }
    if (placed)
      objects.push({
        id: `prop-${i}`,
        kind,
        position: [x, y, dims[2] / 2 + 0.003],
        size: dims,
        yaw,
        mass: kind === 'plank' ? 1.8 : 1.2,
      });
  }
  for (const agent of agents) {
    const original = [...agent.position];
    for (let attempt = 0; attempt < 40; attempt++) {
      const x = original[0] + r.uniform(-0.45, 0.45),
        y = original[1] + r.uniform(-0.45, 0.45);
      if (!(x > 0.35 && x < s - 0.35 && y > 0.35 && y < s - 0.35)) continue;
      let valid = true;
      for (const o of [...walls, ...objects]) {
        const xy = local([x - o.position[0], y - o.position[1]], o.yaw || 0);
        if (
          Math.hypot(
            Math.max(0, Math.abs(xy[0]) - o.size[0] / 2),
            Math.max(0, Math.abs(xy[1]) - o.size[1] / 2),
          ) < 0.3
        ) {
          valid = false;
          break;
        }
      }
      if (valid) {
        agent.position = [x, y, 0.25];
        break;
      }
    }
    agent.yaw += r.uniform(-0.3, 0.3);
  }
  const rotation = (Math.floor(r.next() * 4) * Math.PI) / 2,
    mirror = r.next() < 0.5;
  for (const item of [...walls, ...objects, ...agents]) {
    let [x, y, z] = item.position,
      angle = item.yaw || 0;
    if (mirror) {
      x = s - x;
      angle = Math.PI - angle;
    }
    const dx = x - s / 2,
      dy = y - s / 2;
    item.position = [
      s / 2 + Math.cos(rotation) * dx - Math.sin(rotation) * dy,
      s / 2 + Math.sin(rotation) * dx + Math.cos(rotation) * dy,
      z,
    ];
    item.yaw = angle + rotation;
  }
  return {
    format: 'hide-seek-physical-arena-v1',
    seed,
    scenario,
    width: s,
    height: s,
    agents,
    objects,
    walls,
  };
}
const nums = (v) => v.map((x) => Number(x.toPrecision(12))).join(' ');
export function arenaXML(arena) {
  // Explicit browser workspace capacity avoids the legacy njmax allocation's
  // 260 MB heap. This changes storage only; contacts and dynamics are unchanged.
  // Resource and native/WASM parity tests cover the maximum supported prop count.
  const p = [
    '<mujoco model="Original hide and seek"><compiler angle="radian"/><option timestep=".005" gravity="0 0 -15" integrator="implicitfast" iterations="30"/><size memory="16M" nconmax="300"/><default><geom friction=".5 .01 .001" condim="3" solref=".015 1"/><joint limited="false"/></default><worldbody><geom name="floor" type="plane" size="20 20 .1" group="0"/>',
  ];
  arena.walls.forEach((w, i) =>
    p.push(
      `<geom name="wall-${i}" type="box" pos="${nums(w.position)}" size="${nums(w.size.map((v) => v / 2))}" euler="0 0 ${w.yaw || 0}" group="1"/>`,
    ),
  );
  arena.agents.forEach((_, i) =>
    p.push(
      `<body name="agent-${i}"><joint name="a${i}x" axis="1 0 0" type="slide" damping="12"/><joint name="a${i}y" axis="0 1 0" type="slide" damping="12"/><joint name="a${i}z" axis="0 0 1" type="slide" damping=".2"/><joint name="a${i}yaw" axis="0 0 1" type="hinge" damping=".8"/><geom name="agent-geom-${i}" type="sphere" size=".25" mass="1" friction=".12 .001 .001" group="2"/></body>`,
    ),
  );
  arena.objects.forEach((o, i) => {
    let g = `type="box" size="${nums(o.size.map((v) => v / 2))}"`;
    if (o.kind === 'ramp') {
      const vertices = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
      ].flatMap((v) => v.map((x, j) => x * o.size[j]));
      p.push(
        `</worldbody><asset><mesh name="ramp-${i}" vertex="${nums(vertices)}"/></asset><worldbody>`,
      );
      g = `type="mesh" mesh="ramp-${i}"`;
    }
    p.push(
      `<body name="object-${i}"><joint name="o${i}x" axis="1 0 0" type="slide" damping=".1"/><joint name="o${i}y" axis="0 1 0" type="slide" damping=".1"/><joint name="o${i}z" axis="0 0 1" type="slide" damping=".1"/><joint name="o${i}yaw" axis="0 0 1" type="hinge" damping=".1"/><geom name="object-geom-${i}" ${g} mass="${o.mass}" group="3"/></body>`,
    );
  });
  p.push('</worldbody><actuator>');
  for (let i = 0; i < 2; i++)
    for (const [axis, gear] of [
      ['x', 24],
      ['y', 24],
      ['yaw', 2.5],
    ])
      p.push(`<motor joint="a${i}${axis}" gear="${gear}"/>`);
  p.push('</actuator><equality>');
  for (let a = 0; a < 2; a++)
    for (let i = 0; i < arena.objects.length; i++)
      p.push(
        `<weld name="grab-${a}-${i}" body1="agent-${a}" body2="object-${i}" active="false" solref=".025 1"/>`,
      );
  for (let i = 0; i < arena.objects.length; i++)
    p.push(`<weld name="lock-${i}" body1="object-${i}" active="false" solref=".015 1"/>`);
  p.push('</equality></mujoco>');
  return p.join('');
}
const local = (v, a) => [
    Math.cos(a) * v[0] + Math.sin(a) * v[1],
    -Math.sin(a) * v[0] + Math.cos(a) * v[1],
  ],
  norm = (v) => Math.hypot(...v),
  sub = (a, b) => a.map((v, i) => v - b[i]);
const slice = (a, i, n = 3) => Array.from(a.subarray(i, i + n));
const yUp = (p) => [p[0], p[2], p[1]];
export function validateArena(input) {
  if (!input || input.format !== 'hide-seek-physical-arena-v1')
    throw Error('This is not a physical hide-and-seek arena.');
  const finite = (v, n) => Array.isArray(v) && v.length === n && v.every(Number.isFinite);
  if (
    !Number.isSafeInteger(input.seed) ||
    input.seed < 0 ||
    input.seed > 4294967295 ||
    !Number.isFinite(input.width) ||
    !Number.isFinite(input.height) ||
    input.width < 6 ||
    input.width > 12 ||
    input.height < 6 ||
    input.height > 12
  )
    throw Error('Arena bounds must be 6–12 metres with a valid integer seed.');
  if (
    !Array.isArray(input.agents) ||
    input.agents.length !== 2 ||
    !Array.isArray(input.objects) ||
    input.objects.length > 10 ||
    !Array.isArray(input.walls) ||
    input.walls.length > 24
  )
    throw Error('Use two agents, at most ten props, and at most 24 walls.');
  for (const a of input.agents)
    if (
      !finite(a.position, 3) ||
      !Number.isFinite(a.yaw) ||
      a.position[0] < 0.25 ||
      a.position[0] > input.width - 0.25 ||
      a.position[1] < 0.25 ||
      a.position[1] > input.height - 0.25 ||
      Math.abs(a.position[2] - 0.25) > 0.05
    )
      throw Error('Agents must start on clear ground inside the arena.');
  for (const o of [...input.objects, ...input.walls])
    if (
      !finite(o.position, 3) ||
      !finite(o.size, 3) ||
      o.size.some((x) => x <= 0 || x > 13) ||
      !Number.isFinite(o.yaw) ||
      o.position.some((x) => Math.abs(x) > 15)
    )
      throw Error('Invalid prop or wall geometry.');
  for (const w of input.walls)
    if (w.size.some((x) => x < 0.05) || w.size[2] > 3)
      throw Error('Walls must have 0.05 m minimum thickness and at most 3 m height.');
  const ids = new Set();
  for (const o of input.objects) {
    if (
      typeof o.id !== 'string' ||
      !/^[\w-]{1,40}$/.test(o.id) ||
      ids.has(o.id) ||
      !['box', 'plank', 'ramp'].includes(o.kind) ||
      !Number.isFinite(o.mass) ||
      o.mass < 0.2 ||
      o.mass > 5 ||
      o.size.some((x) => x < 0.15 || x > 2) ||
      o.position[0] < 0.2 ||
      o.position[0] > input.width - 0.2 ||
      o.position[1] < 0.2 ||
      o.position[1] > input.height - 0.2 ||
      Math.abs(o.position[2] - o.size[2] / 2) > 0.06
    )
      throw Error('Invalid prop. Use unique IDs, 0.15–2 m sides, and 0.2–5 kg mass.');
    ids.add(o.id);
  }
  const arena = structuredClone(input);
  assertClearSpawns(arena);
  assertClearGeometry(arena);
  assertBoundaries(arena);
  return arena;
}
function circleDistance(p, o) {
  const v = local(sub(p, o.position), o.yaw || 0);
  return Math.hypot(
    Math.max(0, Math.abs(v[0]) - o.size[0] / 2),
    Math.max(0, Math.abs(v[1]) - o.size[1] / 2),
  );
}
function assertClearSpawns(arena) {
  for (const a of arena.agents)
    for (const o of [...arena.objects, ...arena.walls])
      if (circleDistance(a.position, o) < 0.28)
        throw Error('A prop or wall overlaps an agent’s starting position.');
  if (norm(sub(arena.agents[0].position, arena.agents[1].position)) < 0.55)
    throw Error('Agent starts overlap.');
}
function assertBoundaries(arena) {
  const w = arena.width,
    h = arena.height,
    edges = [
      [
        [0, 0],
        [w, 0],
      ],
      [
        [w, 0],
        [w, h],
      ],
      [
        [w, h],
        [0, h],
      ],
      [
        [0, h],
        [0, 0],
      ],
    ];
  const contains = (wall, point) => {
    const p = local([point[0] - wall.position[0], point[1] - wall.position[1]], wall.yaw);
    return Math.abs(p[0]) <= wall.size[0] / 2 + 0.01 && Math.abs(p[1]) <= wall.size[1] / 2 + 0.01;
  };
  for (const edge of edges)
    if (
      !arena.walls.some(
        (wall) =>
          wall.position[2] - wall.size[2] / 2 < 0.02 &&
          wall.position[2] + wall.size[2] / 2 >= 0.5 &&
          edge.every((point) => contains(wall, point)),
      )
    )
      throw Error('Keep the four solid outside boundaries so agents remain inside the arena.');
}
function overlapXY(a, b) {
  const basis = (o) => [
    [Math.cos(o.yaw || 0), Math.sin(o.yaw || 0)],
    [-Math.sin(o.yaw || 0), Math.cos(o.yaw || 0)],
  ];
  const aa = basis(a),
    bb = basis(b),
    delta = [b.position[0] - a.position[0], b.position[1] - a.position[1]],
    dot = (x, y) => x[0] * y[0] + x[1] * y[1];
  return [...aa, ...bb].every(
    (axis) =>
      Math.abs(dot(delta, axis)) <
      (a.size[0] / 2) * Math.abs(dot(aa[0], axis)) +
        (a.size[1] / 2) * Math.abs(dot(aa[1], axis)) +
        (b.size[0] / 2) * Math.abs(dot(bb[0], axis)) +
        (b.size[1] / 2) * Math.abs(dot(bb[1], axis)) -
        0.005,
  );
}
function assertClearGeometry(arena) {
  arena.objects.forEach((o, i) => {
    const c = Math.abs(Math.cos(o.yaw)),
      s = Math.abs(Math.sin(o.yaw)),
      rx = (c * o.size[0] + s * o.size[1]) / 2,
      ry = (s * o.size[0] + c * o.size[1]) / 2;
    if (
      o.position[0] - rx < 0 ||
      o.position[0] + rx > arena.width ||
      o.position[1] - ry < 0 ||
      o.position[1] + ry > arena.height
    )
      throw Error('Keep the whole prop inside the arena.');
    if ([...arena.walls, ...arena.objects.slice(0, i)].some((b) => overlapXY(o, b)))
      throw Error('Props must start clear of walls and other props. Move it to open ground.');
  });
}
export function editObject(arena, index, change) {
  const next = structuredClone(arena);
  if (change === null) {
    if (index < 0 || index >= next.objects.length) throw Error('Select a prop first.');
    next.objects.splice(index, 1);
  } else {
    const o = {
      ...change,
      id: index >= 0 ? next.objects[index]?.id : `prop-${Date.now().toString(36)}`,
    };
    if (index < 0) next.objects.push(o);
    else {
      if (!next.objects[index]) throw Error('Select a prop first.');
      next.objects[index] = o;
    }
  }
  return validateArena(next);
}
export class PhysicsSimulation {
  constructor(mj, arena, options = {}) {
    if (options.observationSchema !== undefined && !OBSERVATION_SCHEMAS.includes(options.observationSchema)) throw Error('Unknown opponent observation schema');
    this.observationSchema = options.observationSchema;
    this.mj = mj;
    this.arena = arena;
    this.prep = options.prep ?? PREP;
    this.play = options.play ?? PLAY;
    this.continuous = options.continuous ?? false;
    this.dt = DT;
    this.disableTools = options.disableTools || false;
    this.immovable = options.immovable || false;
    this.model = mj.MjModel.from_xml_string(arenaXML(arena));
    try {
      this.data = new mj.MjData(this.model);
      this.nObj = arena.objects.length;
      this.hitBuffer = new mj.IntBuffer(1);
      this.normalBuffer = new mj.DoubleBuffer(3);
      this.active = Array(this.nObj * 3).fill(0);
      const namedId = (kind, name) => {
        const accessor = this.model[kind](name);
        try {
          return accessor.id;
        } finally {
          accessor.delete();
        }
      };
      this.agentBodies = [0, 1].map((i) => namedId('body', `agent-${i}`));
      this.agentGeoms = [0, 1].map((i) => namedId('geom', `agent-geom-${i}`));
      this.objectBodies = arena.objects.map((_, i) => namedId('body', `object-${i}`));
      arena.agents.forEach((a, i) => this.data.qpos.set([...a.position, a.yaw || 0], i * 4));
      arena.objects.forEach((o, i) => this.data.qpos.set([...o.position, o.yaw || 0], 8 + i * 4));
      this.t = 0;
      this.time = 0;
      this.returns = [0, 0];
      this.hidden = 0;
      this.grips = [-1, -1];
      this.locks = Array(this.nObj).fill(-1);
      this.lockEvents = [0, 0];
      this.grabEvents = [0, 0];
      this.jumpEvents = [0, 0];
      this.path = [0, 0];
      this.collisions = [0, 0];
      this.shielded = 0;
      this.visible = false;
      this.done = false;
      this.lastSeen = [null, null];
      this.actions = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
      ];
      mj.mj_forward(this.model, this.data);
      if (this.immovable)
        this.objectBodies.forEach((b, i) => {
          this.setWeld(2 * this.nObj + i, b, 0);
          this.locks[i] = 2;
        });
      this.initialObjects = arena.objects.map((_, i) => slice(this.data.qpos, 8 + i * 4));
      this.senses();
      this.syncView();
    } catch (error) {
      this.dispose();
      throw error;
    }
  }
  forEachContact(visit) {
    // Both the vector getter and its elements create owned Embind handles.
    // Freeing only each element retains native allocations across arena resets.
    const contacts = this.data.contact;
    try {
      for (let index = 0; index < this.data.ncon; index++) {
        const contact = contacts.get(index);
        try {
          visit(contact);
        } finally {
          contact.delete();
        }
      }
    } finally {
      contacts.delete();
    }
  }
  ray(origin, direction, exclude = -1, groups = [1, 1, 1, 1, 1, 1]) {
    const distance = this.mj.mj_ray(
      this.model,
      this.data,
      origin,
      direction,
      groups,
      true,
      exclude,
      this.hitBuffer,
      this.normalBuffer,
    );
    return [distance, this.hitBuffer.GetView()[0]];
  }
  sees(a, b, ignoreProps = false) {
    const p = slice(this.data.xpos, this.agentBodies[a] * 3),
      q = slice(this.data.xpos, b * 3),
      v = sub(q, p),
      d = norm(v),
      yaw = this.data.qpos[a * 4 + 3];
    if (d > VISION_RANGE || d < 1e-8) return d < 1e-8;
    const angle = Math.atan2(v[1], v[0]) - yaw;
    if (Math.abs(Math.atan2(Math.sin(angle), Math.cos(angle))) > FOV / 2) return false;
    const [, hit] = this.ray(
      p,
      v.map((x) => x / d),
      this.agentBodies[a],
      [1, 1, 1, ignoreProps ? 0 : 1, 1, 1],
    );
    return hit >= 0 && this.model.geom_bodyid[hit] === b;
  }
  senses() {
    this.seen = [
      [false, false],
      [false, false],
    ];
    this.objectSeen = [Array(this.nObj).fill(false), Array(this.nObj).fill(false)];
    for (let a = 0; a < 2; a++) {
      if (a === 1 && this.t < this.prep) continue;
      this.seen[a][1 - a] = this.sees(a, this.agentBodies[1 - a]);
      this.objectSeen[a] = this.objectBodies.map((b) => this.sees(a, b));
      if (this.seen[a][1 - a])
        this.lastSeen[a] = {
          position: slice(this.data.xpos, this.agentBodies[1 - a] * 3),
          t: this.t,
        };
    }
    this.visible = this.seen[1][0];
  }
  setWeld(eq, b1, b2) {
    const p1 = slice(this.data.xpos, b1 * 3),
      p2 = slice(this.data.xpos, b2 * 3),
      yaw1 = b1 === 0 ? 0 : Math.atan2(this.data.xmat[b1 * 9 + 3], this.data.xmat[b1 * 9]),
      yaw2 = b2 === 0 ? 0 : Math.atan2(this.data.xmat[b2 * 9 + 3], this.data.xmat[b2 * 9]);
    const delta = sub(p2, p1),
      xy = local(delta, yaw1),
      a = (yaw2 - yaw1) / 2;
    this.model.eq_data.set(
      [0, 0, 0, xy[0], xy[1], delta[2], Math.cos(a), 0, 0, Math.sin(a), 1],
      eq * 11,
    );
    this.active[eq] = 1;
    for (let a = 0; a < 2; a++) {
      const i = this.grips[a];
      if (i >= 0 && this.locks[i] >= 0) {
        this.active[a * this.nObj + i] = 0;
        this.grips[a] = -1;
      }
    }
    this.mj.mj_setState(this.model, this.data, this.active, 512);
  }
  distance(a, i) {
    const p = sub(
        slice(this.data.xpos, this.agentBodies[a] * 3),
        slice(this.data.xpos, this.objectBodies[i] * 3),
      ),
      v = local(p, this.data.qpos[11 + i * 4]),
      half = this.arena.objects[i].size.map((x) => x / 2);
    return Math.hypot(...[...v, p[2]].map((x, j) => Math.max(0, Math.abs(x) - half[j]))) - 0.25;
  }
  tools(actions) {
    for (let a = 0; a < 2; a++) {
      const candidates = this.arena.objects
        .map((_, i) => i)
        .filter(
          (i) =>
            this.objectSeen[a][i] && this.distance(a, i) < 0.22 && !(a === 1 && this.t < this.prep),
        );
      let chosen = -1;
      const old = this.grips[a];
      if (!this.disableTools && actions[a][3] > 0.5) {
        if (old >= 0 && this.locks[old] < 0) chosen = old;
        else {
          const available = candidates.filter(i => this.locks[i] < 0 && !this.grips.includes(i));
          if (available.length) chosen = available.reduce((i,j) => this.distance(a,i) <= this.distance(a,j) ? i : j);
        }
      }
      if (old !== chosen) {
        if (old >= 0) this.active[a * this.nObj + old] = 0;
        if (chosen >= 0) {
          this.setWeld(a * this.nObj + chosen, this.agentBodies[a], this.objectBodies[chosen]);
          this.grabEvents[a]++;
        }
        this.grips[a] = chosen;
      }
      for (const i of candidates) {
        if (this.disableTools || this.locks[i] === 2) continue;
        if (actions[a][4] > 0.5 && this.locks[i] < 0) {
          this.setWeld(2 * this.nObj + i, this.objectBodies[i], 0);
          this.locks[i] = a;
          this.lockEvents[a]++;
        } else if (actions[a][4] <= 0.5 && this.locks[i] === a) {
          this.active[2 * this.nObj + i] = 0;
          this.locks[i] = -1;
        }
      }
    }
    for (let a = 0; a < 2; a++) {
      const i = this.grips[a];
      if (i >= 0 && this.locks[i] >= 0) {
        this.active[a * this.nObj + i] = 0;
        this.grips[a] = -1;
      }
    }
    this.mj.mj_setState(this.model, this.data, this.active, 512);
  }
  jump(a, request) {
    if (!(request > 0.5)) return;
    const pos = slice(this.data.qpos, 4*a);
    const [dist, geom] = this.ray(pos,[0,0,-1],this.agentBodies[a],[1,1,0,1,0,0]);
    if (dist < 0 || dist > .29 || Math.abs(this.data.qvel[4*a+2]) > .35) return;
    if (this.model.geom_group[geom] === 1) return;
    const ceiling = Math.min(...this.arena.walls.map(w => w.position[2]+w.size[2]/2)) - .15;
    const rise = Math.min(.8, ceiling-(pos[2]-.25));
    if (rise < .1) return;
    const grip = this.grips[a], mass = 1+(grip >= 0 ? this.arena.objects[grip].mass : 0);
    this.data.qvel[4*a+2] = Math.sqrt(2*15*rise)/mass;
    this.jumpEvents[a]++;
  }
  step(input) {
    if (this.done) return;
    const actions = input.map((v) => Array.from(v));
    if (
      actions.length !== 2 ||
      actions.some((v) => ![5, 6].includes(v.length) || v.some((x) => !Number.isFinite(x)))
    )
      throw Error('Actions must contain two finite action vectors.');
    for (const a of actions) for (let j = 0; j < 3; j++) a[j] = clamp(a[j], -1, 1);
    if (this.t < this.prep) actions[1].fill(0);
    for (const action of actions) if (action.length === 5) action.push(0);
    this.actions = actions;
    const before = [slice(this.data.qpos, 0), slice(this.data.qpos, 4)];
    this.tools(actions);
    for (let a=0; a<2; a++) this.jump(a, actions[a][5]);
    for (let a = 0; a < 2; a++) {
      const yaw = this.data.qpos[a * 4 + 3],
        c = Math.cos(yaw),
        s = Math.sin(yaw),
        [x, y] = actions[a];
      this.data.ctrl.set([c * x - s * y, s * x + c * y, actions[a][2]], a * 3);
    }
    const hits = [false, false];
    for (let k = 0; k < SUBSTEPS; k++) this.mj.mj_step(this.model, this.data);
    this.forEachContact((contact) => {
      if (contact.dist <= 0)
        for (let a = 0; a < 2; a++) {
          const g = this.agentGeoms[a];
          if (
            (contact.geom1 === g && contact.geom2 !== 0) ||
            (contact.geom2 === g && contact.geom1 !== 0)
          )
            hits[a] = true;
        }
    });
    this.mj.mj_forward(this.model, this.data);
    for (let a = 0; a < 2; a++) {
      this.path[a] += norm(sub(slice(this.data.qpos, a * 4), before[a]));
      this.collisions[a] += Number(hits[a]);
    }
    const prep = this.t < this.prep;
    this.t++;
    this.time = this.t * DT;
    this.senses();
    if (!prep) {
      const r = this.visible ? -1 : 1;
      this.returns[0] += r;
      this.returns[1] -= r;
      this.hidden += Number(!this.visible);
      if (!this.visible && this.sees(1, this.agentBodies[0], true)) this.shielded++;
    }
    this.done = !this.continuous && this.t >= this.prep + this.play;
    if (
      Array.from(this.data.qpos).some((x) => !Number.isFinite(x)) ||
      Array.from(this.data.qvel).some((x) => Math.abs(x) > 150)
    )
      throw Error('The physical simulation became unstable. Reset the arena.');
    this.syncView();
  }
  observe(a) {
    const pos = slice(this.data.qpos, a * 4),
      yaw = this.data.qpos[a * 4 + 3],
      vel = slice(this.data.qvel, a * 4, 4),
      lv = local(vel, yaw);
    const row = [
      lv[0] / 5,
      lv[1] / 5,
      vel[2] / 5,
      vel[3] / 8,
      pos[2] / 2,
      Math.min(1, this.t / Math.max(1, this.prep)),
      Math.min(1, Math.max(0, this.t - this.prep) / this.play),
      1 - a,
      Number(this.grips[a] >= 0),
      Number(this.grips[a] >= 0 && this.locks[this.grips[a]] >= 0),
    ];
    const b = 1 - a;
    if (this.seen[a][b]) {
      const p = sub(slice(this.data.qpos, b * 4), pos),
        v = local(slice(this.data.qvel, b * 4), yaw),
        xy = local(p, yaw),
        ang = this.data.qpos[b * 4 + 3] - yaw;
      row.push(1, xy[0] / 6, xy[1] / 6, p[2] / 2, v[0] / 5, v[1] / 5, Math.cos(ang), Math.sin(ang));
    } else row.push(...Array(8).fill(0));
    if (this.observationSchema === KNOWN_POSITION_SCHEMA) {
      const delta = sub(slice(this.data.qpos, b * 4), pos);
      const xy = local(delta, yaw);
      row[11] = xy[0] / 6;
      row[12] = xy[1] / 6;
      row[13] = delta[2] / 2;
    }
    const ids = this.arena.objects
      .map((_, i) => i)
      .filter((i) => this.objectSeen[a][i])
      .sort(
        (i, j) =>
          norm(sub(slice(this.data.qpos, 8 + i * 4), pos)) -
          norm(sub(slice(this.data.qpos, 8 + j * 4), pos)),
      );
    for (let k = 0; k < OBJECT_SLOTS; k++) {
      if (k >= ids.length) {
        row.push(...Array(16).fill(0));
        continue;
      }
      const i = ids[k],
        o = this.arena.objects[i],
        p = sub(slice(this.data.qpos, 8 + i * 4), pos),
        v = slice(this.data.qvel, 8 + i * 4, 4),
        xy = local(p, yaw),
        lv = local(v, yaw),
        ang = this.data.qpos[11 + i * 4] - yaw;
      row.push(
        1,
        xy[0] / 6,
        xy[1] / 6,
        p[2] / 2,
        lv[0] / 5,
        lv[1] / 5,
        v[2] / 5,
        Math.cos(ang),
        Math.sin(ang),
        o.size[0] / 2,
        o.size[1] / 2,
        o.size[2] / 2,
        Number(o.kind === 'ramp'),
        Number(this.locks[i] === a),
        Number(this.locks[i] >= 0 && this.locks[i] !== a),
        Number(this.grips[a] === i),
      );
    }
    for (let k = 0; k < LIDAR; k++) {
      if (a === 1 && this.t < this.prep) {
        row.push(1);
        continue;
      }
      const angle = yaw + (k * tau) / LIDAR,
        [d] = this.ray(
          pos,
          [Math.cos(angle), Math.sin(angle), 0],
          this.agentBodies[a],
          [0, 1, 0, 1, 0, 0],
        );
      row.push(d >= 0 ? clamp(d, 0, 6) / 6 : 1);
    }
    if (this.observationSchema === TAG_SCHEMA) row.push(...extras(this, a));
    return new Float32Array(row);
  }
  syncView() {
    this.phase = this.done ? 'finished' : this.t < this.prep ? 'preparation' : 'play';
    this.objects = this.arena.objects.map((o, i) => {
      const p = slice(this.data.qpos, 8 + i * 4),
        a = this.data.qpos[11 + i * 4];
      return {
        ...o,
        position: yUp(p),
        size: yUp(o.size),
        quaternion: [0, -Math.sin(a / 2), 0, Math.cos(a / 2)],
        velocity: yUp(slice(this.data.qvel, 8 + i * 4)),
        angularVelocity: [0, -this.data.qvel[11 + i * 4], 0],
        lockedBy: this.locks[i],
        grabbedBy: this.grips.indexOf(i),
      };
    });
    const agentContacts = [[], []],
      contactSupport = [null, null];
    const supportFrame = (objectIndex, geomId) => {
      const object = this.objects[objectIndex];
      return object ? {
        geomId, objectId: object.id,
        transform: { objectId: object.id, kind: object.kind, size: object.size,
          position: object.position, quaternion: object.quaternion },
      } : { geomId };
    };
    this.forEachContact((c) => {
      if (c.dist <= 0.006)
        for (let a = 0; a < 2; a++) {
          const g = this.agentGeoms[a],
            other = c.geom1 === g ? c.geom2 : c.geom2 === g ? c.geom1 : -1;
          if (other < 0) continue;
          const direction = c.geom1 === g ? -1 : 1;
          const i = this.objectBodies.indexOf(this.model.geom_bodyid[other]);
          if (c.frame[2] * direction > 0.4) contactSupport[a] = {
            ...supportFrame(i, other),
            point: yUp(Array.from(c.pos)),
            normal: yUp(Array.from(c.frame).slice(0, 3).map(x => x * direction)),
            gap: c.dist,
            source: 'contact',
          };
          if (i < 0) continue;
          agentContacts[a].push({
            point: yUp(Array.from(c.pos)),
            normal: yUp(
              Array.from(c.frame)
                .slice(0, 3)
                .map((x) => x * direction),
            ),
            objectId: this.arena.objects[i].id,
          });
        }
    });
    this.agents = [0, 1].map((a) => {
      const p = slice(this.data.qpos, a * 4),
        grip = this.grips[a];
      // Spectator support, not an input to control, rewards or contact forces.
      // MuJoCo's soft sphere contact can leave millimetres of clearance with no
      // active contact. Query the actual geometry instead of freezing the gait
      // through those floor-travel frames. The normal accounts for ramp slope.
      let presentationSupport = contactSupport[a];
      const [downDistance, downGeom] = this.ray(p, [0, 0, -1], this.agentBodies[a], [1, 1, 0, 1, 0, 0]);
      if (downDistance >= 0 && downGeom >= 0) {
        const normal = Array.from(this.normalBuffer.GetView());
        const gap = downDistance * normal[2] - .25;
        const object = this.objectBodies.indexOf(this.model.geom_bodyid[downGeom]);
        const supportVelocity = object < 0 ? [0, 0, 0] : slice(this.data.qvel, 8 + object * 4);
        const relativeNormalSpeed = slice(this.data.qvel, a * 4).reduce((sum, value, axis) => sum + (value - supportVelocity[axis]) * normal[axis], 0);
        if (normal[2] > .45 && gap >= -.04 && gap <= .015 && relativeNormalSpeed < .65) {
          presentationSupport = {
            ...supportFrame(object, downGeom),
            point: yUp([p[0], p[1], p[2] - downDistance]),
            normal: yUp(normal), gap, source: 'proximity',
          };
        }
      }
      let handTarget = null;
      if (grip >= 0) {
        const o = this.arena.objects[grip],
          op = slice(this.data.qpos, 8 + grip * 4),
          angle = this.data.qpos[11 + grip * 4],
          v = local(sub(p, op), angle),
          x = clamp(v[0], -o.size[0] / 2, o.size[0] / 2),
          y = clamp(v[1], -o.size[1] / 2, o.size[1] / 2);
        handTarget = yUp([
          op[0] + Math.cos(angle) * x - Math.sin(angle) * y,
          op[1] + Math.sin(angle) * x + Math.cos(angle) * y,
          clamp(p[2] + 0.1, op[2] - o.size[2] / 2, op[2] + o.size[2] / 2),
        ]);
      }
      if (!handTarget) {
        const forward = [Math.cos(this.data.qpos[a * 4 + 3]), Math.sin(this.data.qpos[a * 4 + 3])];
        const contact = agentContacts[a].find(
          (c) => (c.point[0] - p[0]) * forward[0] + (c.point[2] - p[1]) * forward[1] > 0.02,
        );
        if (contact) handTarget = contact.point;
      }
      return {
        id: a,
        role: a ? 'seeker' : 'hider',
        team: a,
        position: yUp(p),
        velocity: yUp(slice(this.data.qvel, a * 4)),
        heading: -this.data.qpos[a * 4 + 3],
        angularVelocity: -this.data.qvel[a * 4 + 3],
        forward: [Math.cos(this.data.qpos[a * 4 + 3]), 0, Math.sin(this.data.qpos[a * 4 + 3])],
        radius: 0.25,
        colliderRadius: 0.25,
        colliderHeight: 0.5,
        visualHeight: 0.7,
        grounded: presentationSupport !== null,
        support: presentationSupport,
        gripId: grip >= 0 ? this.arena.objects[grip].id : null,
        handTarget,
        contacts: agentContacts[a],
      };
    });
    this.viewArena = {
      ...this.arena,
      objects: this.objects,
      walls: this.arena.walls.map((w) => ({
        ...w,
        position: yUp(w.position),
        size: yUp(w.size),
        quaternion: [0, -Math.sin((w.yaw || 0) / 2), 0, Math.cos((w.yaw || 0) / 2)],
      })),
    };
  }
  vision(role) {
    const a = typeof role === 'number' ? role : role === 'seeker' ? 1 : 0,
      p = slice(this.data.qpos, a * 4),
      heading = this.data.qpos[a * 4 + 3],
      blind = a === 1 && this.t < this.prep,
      polygon = [];
    if (!blind)
      for (let k = 0; k <= 96; k++) {
        const angle = heading - FOV / 2 + (FOV * k) / 96,
          dir = [Math.cos(angle), Math.sin(angle), 0],
          [d] = this.ray(p, dir, this.agentBodies[a]);
        const length = d >= 0 ? Math.min(this.arena.width*2, d) : this.arena.width*2;
        polygon.push(yUp([p[0] + dir[0] * length, p[1] + dir[1] * length, p[2]]));
      }
    return {
      origin: yUp(p),
      heading: -heading,
      fovRadians: FOV,
      range: this.arena.width*2,
      polygon,
      seesOpponent: this.seen[a][1 - a],
      blind,
      lastSeen: this.lastSeen[a]
        ? {
            position: yUp(this.lastSeen[a].position),
            ageSeconds: (this.t - this.lastSeen[a].t) * DT,
          }
        : null,
    };
  }
  info() {
    return {
      t: this.t,
      hidden: this.hidden,
      visible: this.visible,
      shielded: this.shielded,
      returns: this.returns,
      grabs: this.grabEvents,
      locks: this.lockEvents,
      locked: this.locks.filter((v) => v >= 0).length,
      object_displacement: this.initialObjects.map((p, i) =>
        norm(sub(slice(this.data.qpos, 8 + i * 4), p)),
      ),
      path: this.path,
      collisions: this.collisions,
      play_steps: Math.max(0, this.t - this.prep),
    };
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    for (const key of ['hitBuffer', 'normalBuffer', 'data', 'model']) {
      this[key]?.delete();
      this[key] = null;
    }
  }
}
