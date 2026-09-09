/** Physics and restricted actor observations, mirrored in training/sim.py. */
export const RULES = Object.freeze({ minSize: 10, maxSize: 20, maxBlocks: 12, vision: 7, memory: 60, prep: 24, play: 180, radius: .24, tag: .65, speeds: [.30, .34] });
export const ACTIONS = Object.freeze([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]]);
export class Random {
  constructor(seed) { this.state = (Number(seed) >>> 0) || 1; }
  next() { let n = this.state; n ^= n << 13; n ^= n >>> 17; n ^= n << 5; this.state = n >>> 0; return this.state / 4294967296; }
  integer(n) { return Math.min(n - 1, Math.floor(this.next() * n)); }
}
const clamp = (v, low, high) => Math.min(high, Math.max(low, v));
export function gridFor(arena) {
  const grid = new Uint8Array(arena.width * arena.height);
  for (let y = 0; y < arena.height; y++) for (let x = 0; x < arena.width; x++) if (!x || !y || x === arena.width - 1 || y === arena.height - 1) grid[y * arena.width + x] = 1;
  for (const b of arena.blocks) for (let y = b.y; y < b.y + b.height; y++) for (let x = b.x; x < b.x + b.width; x++) grid[y * arena.width + x] = 1;
  return grid;
}
export function occupied(arena, grid, x, y) { return x < 0 || y < 0 || x >= arena.width || y >= arena.height || Boolean(grid[Math.floor(y) * arena.width + Math.floor(x)]); }
export function connected(arena, grid = gridFor(arena)) {
  const first = grid.indexOf(0); if (first < 0) return false;
  const seen = new Set([first]), stack = [first];
  while (stack.length) { const n = stack.pop(); for (const m of [n - 1, n + 1, n - arena.width, n + arena.width]) if (m >= 0 && m < grid.length && !grid[m] && !seen.has(m)) { seen.add(m); stack.push(m); } }
  return seen.size === grid.reduce((a, b) => a + (b === 0), 0);
}
export function lineClear(arena, _grid, a, b) {
  // Exact closed AABB slabs. Even grazing a solid corner blocks sight.
  for (const box of arena.blocks) {
    let near = 0, far = 1;
    for (let axis = 0; axis < 2; axis++) {
      const low = axis ? box.y : box.x, high = low + (axis ? box.height : box.width), delta = b[axis] - a[axis];
      if (Math.abs(delta) < 1e-12) { if (a[axis] < low || a[axis] > high) { near = 2; break; } }
      else { const t1 = (low - a[axis]) / delta, t2 = (high - a[axis]) / delta; near = Math.max(near, Math.min(t1, t2)); far = Math.min(far, Math.max(t1, t2)); }
    }
    if (near <= far) return false;
  }
  return true;
}

export function generateArena(seed = 2709, width = 14, height = 12, count = 6) {
  if (![seed, width, height, count].every(Number.isInteger) || width < RULES.minSize || width > RULES.maxSize || height < RULES.minSize || height > RULES.maxSize || count < 0 || count > RULES.maxBlocks) throw new Error('Use whole-number sizes 10–20 and 0–12 cover blocks.');
  const rng = new Random(seed), arena = { version: 1, seed: seed >>> 0, width, height, blocks: [], spawns: [] };
  let grid = gridFor(arena);
  for (let i = 0; i < count; i++) for (let attempt = 0; attempt < 24; attempt++) {
    const b = { width: 1 + rng.integer(3), height: 1 + rng.integer(3) };
    b.x = 2 + rng.integer(Math.max(1, width - b.width - 3)); b.y = 2 + rng.integer(Math.max(1, height - b.height - 3));
    let overlap = false; for (let y = b.y; y < b.y + b.height; y++) for (let x = b.x; x < b.x + b.width; x++) overlap ||= Boolean(grid[y * width + x]);
    if (overlap) continue;
    arena.blocks.push(b); const next = gridFor(arena);
    if (connected(arena, next)) { grid = next; break; } arena.blocks.pop();
  }
  const free = []; for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (!grid[y * width + x]) free.push([x + .5, y + .5]);
  const seeker = free[rng.integer(free.length)]; let hider;
  for (let i = 0; i < 200; i++) { const p = free[rng.integer(free.length)], d = Math.hypot(p[0] - seeker[0], p[1] - seeker[1]); if (d >= 3 && d <= 6 && lineClear(arena, grid, p, seeker)) { hider = p; break; } }
  if (!hider) { const candidates = free.filter(p => Math.hypot(p[0] - seeker[0], p[1] - seeker[1]) > 2); hider = candidates[rng.integer(candidates.length)]; }
  arena.spawns = [hider.slice(), seeker.slice()]; return arena;
}
export function validateArena(input) {
  if (!input || typeof input !== 'object') throw new Error('Arena must be a JSON object.');
  const { width, height } = input;
  if (![width, height].every(Number.isInteger) || width < 10 || width > 20 || height < 10 || height > 20) throw new Error('Arena width and height must be 10–20.');
  if (!Array.isArray(input.blocks) || input.blocks.length > 12) throw new Error('An arena supports at most 12 cover blocks.');
  const arena = { version: 1, seed: Number(input.seed) >>> 0, width, height, blocks: input.blocks.map(b => {
    if (!b || !['x', 'y', 'width', 'height'].every(k => Number.isInteger(b[k])) || b.width < 1 || b.width > 3 || b.height < 1 || b.height > 3 || b.x < 1 || b.y < 1 || b.x + b.width >= width || b.y + b.height >= height) throw new Error('Cover must fit inside the arena and measure 1–3 cells per side.');
    return { x: b.x, y: b.y, width: b.width, height: b.height };
  }), spawns: input.spawns?.map(p => p?.slice()) };
  for (let i = 0; i < arena.blocks.length; i++) for (let j = i + 1; j < arena.blocks.length; j++) { const a = arena.blocks[i], b = arena.blocks[j]; if (a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y) throw new Error('Cover blocks cannot overlap.'); }
  const grid = gridFor(arena);
  if (!connected(arena, grid)) throw new Error('Keep all open floor connected.');
  if (!Array.isArray(arena.spawns) || arena.spawns.length !== 2 || arena.spawns.some(p => !Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite) || [[-.24, -.24], [-.24, .24], [.24, -.24], [.24, .24]].some(([dx, dy]) => occupied(arena, grid, p[0] + dx, p[1] + dy)))) throw new Error('Both starting positions need clear floor.');
  if (Math.hypot(arena.spawns[0][0] - arena.spawns[1][0], arena.spawns[0][1] - arena.spawns[1][1]) < 2) throw new Error('Starting positions must be at least two cells apart.');
  return arena;
}
export function editBlock(arena, index, block) {
  const next = structuredClone(arena);
  if (block === null) next.blocks.splice(index, 1); else if (index < 0) next.blocks.push(block); else next.blocks[index] = block;
  return validateArena(next);
}
export class Simulation {
  constructor(arena) {
    this.arena = validateArena(arena); this.grid = gridFor(this.arena);
    this.pos = this.arena.spawns.map(p => p.map(Math.fround)); this.vel = [[0, 0], [0, 0]]; this.t = 0; this.prep = arena.prep ?? RULES.prep;
    this.memory = [[0, 0], [0, 0]]; this.age = [60, 60]; this.known = [false, false]; this.ever = false;
    this.visited = new Set(); this.novelty = 0; this.hidden = 0; this.collisions = [0, 0]; this.distance = [0, 0]; this.returns = [0, 0]; this.done = false; this.capture = false;
    this.updateSight(false);
  }
  sight() { const d = Math.hypot(this.pos[0][0] - this.pos[1][0], this.pos[0][1] - this.pos[1][1]); return [d <= RULES.vision && lineClear(this.arena, this.grid, this.pos[0], this.pos[1]), d]; }
  updateSight(age = true) {
    [this.visible] = this.sight();
    for (let r = 0; r < 2; r++) { if (age) this.age[r] = Math.min(60, this.age[r] + 1); if (this.visible && (!r || this.t >= this.prep)) { this.memory[r] = this.pos[1 - r].slice(); this.known[r] = true; this.age[r] = 0; } }
  }
  observe(role) {
    const p = this.pos[role], rays = [];
    for (let a = 0; a < 8; a++) { let first = 12; for (let d = 1; d <= 12; d++) if (occupied(this.arena, this.grid, Math.fround(p[0] + Math.fround(Math.fround(Math.cos(a * Math.PI / 4)) * Math.fround(d * Math.fround(.35)))), Math.fround(p[1] + Math.fround(Math.fround(Math.sin(a * Math.PI / 4)) * Math.fround(d * Math.fround(.35)))))) { first = d; break; } rays.push(first / 12); }
    const seen = this.visible && (!role || this.t >= this.prep), known = this.known[role] && this.age[role] < 60 && (!role || this.t >= this.prep);
    return Float32Array.from([...rays, +seen, ...this.pos[1 - role].map((v, i) => seen ? clamp((v - p[i]) / 7, -1, 1) : 0), +known, ...this.memory[role].map((v, i) => known ? clamp((v - p[i]) / 7, -1, 1) : 0), this.age[role] / 60, ...this.vel[role].map(v => v / RULES.speeds[role]), p[0] / this.arena.width, p[1] / this.arena.height, this.arena.width / 24, this.arena.height / 24, clamp((180 - (this.t - this.prep)) / 180, 0, 1), Math.max(0, this.prep - this.t) / 24]);
  }
  step(actions) {
    if (this.done) return [0, 0];
    if (!Array.isArray(actions) || actions.length !== 2 || actions.some(a => !Number.isInteger(a) || a < 0 || a > 4)) throw new Error('Each action must be 0–4.');
    const [oldVisible, oldDistance] = this.sight(), active = this.t >= this.prep, previous = this.pos.map(p => p.slice());
    for (let r = 0; r < 2; r++) for (let axis = 0; axis < 2; axis++) {
      const delta = r === 1 && !active ? 0 : Math.fround(ACTIONS[actions[r]][axis] * RULES.speeds[r]);
      const target = this.pos[r].slice(); target[axis] = Math.fround(target[axis] + delta);
      const blocked = [[-.24, -.24], [-.24, .24], [.24, -.24], [.24, .24]].some(([dx, dy]) => occupied(this.arena, this.grid, Math.fround(target[0] + dx), Math.fround(target[1] + dy)));
      if (!blocked) this.pos[r][axis] = target[axis]; else if (delta) this.collisions[r]++;
    }
    for (let r = 0; r < 2; r++) { this.vel[r] = this.pos[r].map((v, i) => Math.fround(v - previous[r][i])); this.distance[r] += Math.hypot(...this.vel[r]); }
    const [visible, distance] = this.sight(), both = oldVisible && visible && active, delta = (oldDistance - distance) / 7;
    const rewards = [active ? visible ? -.002 : .002 : 0, active ? -.001 : 0];
    if (both) { rewards[1] += delta * .15; rewards[0] -= delta * .05; }
    if (active && visible && !this.ever) rewards[1] += .03;
    this.ever ||= active && visible;
    const cell = Math.floor(this.pos[1][1]) * this.arena.width + Math.floor(this.pos[1][0]);
    if (active && !this.visited.has(cell) && this.novelty < .15) { rewards[1] += .003; this.novelty = Math.fround(this.novelty + .003); } this.visited.add(cell);
    if (active && !visible) this.hidden++;
    this.capture = active && visible && distance < .65; this.t++; this.done = this.capture || this.t >= this.prep + 180;
    if (this.done) { rewards[0] += this.capture ? -2 : 2; rewards[1] += this.capture ? 2 : -2; }
    rewards.forEach((v, r) => this.returns[r] += v); this.updateSight(); return rewards;
  }
}
