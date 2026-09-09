import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/** Rendering owns no agent logic. The overhead camera is a spectator view. */
export function createView(host, { onPick, onOrbit, transparent = false }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio)); renderer.setClearColor('#142321', transparent ? 0 : 1);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.setAttribute('aria-label', '3D hide and seek arena. Drag to orbit; use the labeled editing controls to place cover.');
  renderer.domElement.setAttribute('role', 'img'); host.append(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = transparent ? null : new THREE.Color('#142321');
  const camera = new THREE.PerspectiveCamera(37, 1, .1, 150);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = false; controls.enablePan = false; controls.minPolarAngle = .18; controls.maxPolarAngle = Math.PI * .43; controls.minDistance = 10; controls.maxDistance = 60;
  controls.addEventListener('change', render);
  const ambient = new THREE.HemisphereLight('#f4ead1', '#496751', 2.4); scene.add(ambient);
  const light = new THREE.DirectionalLight('#fff0cb', 3.4); light.position.set(-8, 22, -10); light.castShadow = true;
  light.shadow.mapSize.set(2048, 2048); light.shadow.camera.left = -25; light.shadow.camera.right = 25; light.shadow.camera.top = 25; light.shadow.camera.bottom = -25; light.shadow.normalBias = .025; light.shadow.bias = -.0002; scene.add(light);
  const world = new THREE.Group(); scene.add(world);
  const material = color => new THREE.MeshStandardMaterial({ color, roughness: .8, metalness: 0 });
  const materials = { floor: material('#294538'), cover: material('#879b72'), rim: material('#49644d'), hider: material('#c8dcaa'), seeker: material('#e5a17e'), eye: material('#23382e'), white: material('#f1f0d4'), crown: material('#d6b26f'), selected: material('#edd298') };
  const persistent = new Set(Object.values(materials));
  function mesh(geometry, mat, x, y, z, group = world) { const m = new THREE.Mesh(geometry, mat); m.position.set(x, y, z); m.castShadow = true; m.receiveShadow = true; group.add(m); return m; }
  function pet(role) {
    const g = new THREE.Group(), body = mesh(new THREE.SphereGeometry(.39, 28, 18), materials[role ? 'seeker' : 'hider'], 0, .41, 0, g); body.scale.set(1, .86, .88);
    for (const x of [-.13, .13]) { mesh(new THREE.SphereGeometry(.085, 12, 10), materials.white, x, .49, -.285, g); mesh(new THREE.SphereGeometry(.045, 12, 10), materials.eye, x, .49, -.355, g); }
    if (!role) { const ring = mesh(new THREE.CylinderGeometry(.2, .19, .1, 6), materials.crown, 0, .77, 0, g); ring.rotation.y = Math.PI / 6; for (let i = 0; i < 3; i++) mesh(new THREE.ConeGeometry(.055, .16, 4), materials.crown, Math.cos(i * Math.PI * 2 / 3) * .13, .88, Math.sin(i * Math.PI * 2 / 3) * .13, g); }
    else { mesh(new THREE.BoxGeometry(.5, .07, .09), materials.eye, 0, .56, -.26, g); }
    world.add(g); return g;
  }
  let agents = [], arena, boxes = [], selected = -1, previous = null, editing = false, activeVision = 'none', disposed = false;
  let ring, rayLines, sightLine, fitFactor = 1;
  const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2(), ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const lineMaterial = new THREE.LineBasicMaterial({ color: '#d6cc90', transparent: true, opacity: .6 });
  function clearWorld() { world.traverse(o => { o.geometry?.dispose(); if (o.material && !persistent.has(o.material) && o.material !== lineMaterial) o.material.dispose(); }); while (world.children.length) world.remove(world.children[0]); }
  function setArena(next, recenter = true) {
    arena = next; previous = null; clearWorld(); boxes = [];
    const { width: w, height: h } = arena;
    mesh(new THREE.BoxGeometry(w, .3, h), materials.floor, w / 2, -.18, h / 2).receiveShadow = true;
    const gridPoints = []; for (let x = 1; x < w; x++) gridPoints.push(x, -.018, 1, x, -.018, h - 1); for (let y = 1; y < h; y++) gridPoints.push(1, -.018, y, w - 1, -.018, y);
    const grid = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(gridPoints, 3)), new THREE.LineBasicMaterial({ color: '#9aaf81', transparent: true, opacity: .11 })); world.add(grid);
    [[w, 1, w / 2, .5], [w, 1, w / 2, h - .5], [1, h - 2, .5, h / 2], [1, h - 2, w - .5, h / 2]].forEach(([bw, bh, x, y]) => mesh(new THREE.BoxGeometry(bw, .45, bh), materials.rim, x, .1, y));
    arena.blocks.forEach((b, i) => {
      const shape = new THREE.Shape(), r = .11, bw = b.width - .08, bh = b.height - .08;
      shape.moveTo(r, 0); shape.lineTo(bw - r, 0); shape.quadraticCurveTo(bw, 0, bw, r); shape.lineTo(bw, bh - r); shape.quadraticCurveTo(bw, bh, bw - r, bh); shape.lineTo(r, bh); shape.quadraticCurveTo(0, bh, 0, bh - r); shape.lineTo(0, r); shape.quadraticCurveTo(0, 0, r, 0);
      const geo = new THREE.ExtrudeGeometry(shape, { depth: 1.35, bevelEnabled: true, bevelSize: .035, bevelThickness: .035, bevelSegments: 2, steps: 1, curveSegments: 4 }); geo.rotateX(-Math.PI / 2);
      const m = mesh(geo, i === selected ? materials.selected : materials.cover, b.x + .04, 0, b.y + b.height - .04); m.userData.block = i; boxes.push(m);
      // Inset top seam makes each piece of cover legible without extra texture assets.
    });
    agents = [pet(0), pet(1)];
    ring = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#b9d2a1', transparent: true, opacity: .65 })); world.add(ring);
    rayLines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial); world.add(rayLines);
    sightLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#e3aa87', transparent: true, opacity: .75 })); world.add(sightLine);
    if (recenter) { controls.target.set(w / 2, -.6, h / 2); camera.position.set(w / 2 + w * 1.06, Math.max(w, h) * 1.24, h / 2 + h * 1.14); fitFactor = Math.max(1, 1.4 / camera.aspect); camera.position.sub(controls.target).multiplyScalar(fitFactor).add(controls.target); controls.update(); }
    render();
  }
  function update(sim, blend = 1) {
    if (!agents.length) return;
    for (let r = 0; r < 2; r++) { const p = sim.pos[r], old = previous?.[r] ?? p; agents[r].position.set(old[0] + (p[0] - old[0]) * blend, 0, old[1] + (p[1] - old[1]) * blend); if (Math.hypot(...sim.vel[r]) > .001) agents[r].rotation.y = Math.atan2(-sim.vel[r][0], -sim.vel[r][1]); }
    const role = activeVision === 'hider' ? 0 : 1, show = activeVision !== 'none' && !(role === 1 && sim.t < sim.prep), p = sim.pos[role];
    ring.visible = show; rayLines.visible = show; sightLine.visible = show && sim.visible;
    if (show) {
      // Circle denotes maximum range; occlusion rays stop at exact blocks.
      const vertices = []; for (let i = 0; i < 128; i++) { const angle = i / 128 * Math.PI * 2; let distance = 7; const dx = Math.cos(angle), dy = Math.sin(angle);
        for (let d = .1; d <= 7; d += .1) { const x = p[0] + dx * d, y = p[1] + dy * d; if (x < 1 || y < 1 || x > arena.width - 1 || y > arena.height - 1 || arena.blocks.some(b => x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height)) { distance = Math.max(0, d - .1); break; } }
        vertices.push(p[0] + dx * distance, .035, p[1] + dy * distance);
      }
      ring.geometry.dispose(); ring.geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      const obs = sim.observe(role), rays = []; for (let a = 0; a < 8; a++) rays.push(p[0], .055, p[1], p[0] + Math.cos(a * Math.PI / 4) * obs[a] * 4.2, .055, p[1] + Math.sin(a * Math.PI / 4) * obs[a] * 4.2);
      rayLines.geometry.dispose(); rayLines.geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(rays, 3));
      sightLine.geometry.dispose(); sightLine.geometry = new THREE.BufferGeometry().setFromPoints(sim.pos.map(q => new THREE.Vector3(q[0], .48, q[1])));
    }
    render();
  }
  function remember(sim) { previous = sim.pos.map(p => p.slice()); }
  function setSelection(n) { selected = n; boxes.forEach((b, i) => b.material = i === n ? materials.selected : materials.cover); render(); }
  function setEditing(value) { editing = value; controls.enableRotate = !value; renderer.domElement.style.cursor = value ? 'crosshair' : 'grab'; }
  function setVision(value) { activeVision = value; }
  function capture() {
    const background = scene.background; scene.background = null; renderer.setClearColor('#142321', 0); renderer.render(scene, camera); const data = renderer.domElement.toDataURL('image/png'); scene.background = background; renderer.setClearColor('#142321', transparent ? 0 : 1); render(); return data;
  }
  function render() { if (!disposed) renderer.render(scene, camera); }
  function resize() { const w = host.clientWidth, h = host.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h; const factor = Math.max(1, 1.4 / camera.aspect); if (arena) camera.position.sub(controls.target).multiplyScalar(factor / fitFactor).add(controls.target); fitFactor = factor; camera.updateProjectionMatrix(); render(); }
  const observer = new ResizeObserver(resize); observer.observe(host);
  const onClick = e => { if (!editing) return; const rect = renderer.domElement.getBoundingClientRect(); pointer.set((e.clientX - rect.left) / rect.width * 2 - 1, -(e.clientY - rect.top) / rect.height * 2 + 1); raycaster.setFromCamera(pointer, camera); const hit = raycaster.intersectObjects(boxes)[0]; const point = new THREE.Vector3(); if (raycaster.ray.intersectPlane(ground, point)) onPick?.({ x: Math.floor(point.x), y: Math.floor(point.z), index: hit?.object.userData.block ?? -1 }); };
  renderer.domElement.addEventListener('click', onClick);
  function orbit(direction) { const offset = camera.position.clone().sub(controls.target); offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), direction * Math.PI / 6); camera.position.copy(controls.target).add(offset); controls.update(); onOrbit?.(); render(); }
  function zoom(factor) { const offset = camera.position.clone().sub(controls.target); const length = Math.max(controls.minDistance, Math.min(controls.maxDistance, offset.length() * factor)); offset.setLength(length); camera.position.copy(controls.target).add(offset); controls.update(); render(); }
  resize();
  return { setArena, update, remember, setSelection, setEditing, setVision, orbit, zoom, render, capture, dispose() { disposed = true; observer.disconnect(); renderer.domElement.removeEventListener('click', onClick); controls.removeEventListener('change', render); controls.dispose(); clearWorld(); Object.values(materials).forEach(m => m.dispose()); lineMaterial.dispose(); renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove(); } };
}
