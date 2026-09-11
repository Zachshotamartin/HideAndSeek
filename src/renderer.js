import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { createCharacter } from "./visuals/character.js";
import { interpolateFrames, interpolateSupport } from "./visuals/support.js";

/** A spectator renderer. All object transforms, contacts and sight come from physics. */
export function createView(
  host,
  { onPick, onOrbit, transparent = false } = {},
) {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(0, 0);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.93;
  const canvas = renderer.domElement;
  canvas.setAttribute(
    "aria-label",
    "Hide and seek arena: blue hider and red seeker with movable objects. Drag to orbit, use arrow keys to orbit and plus or minus to zoom.",
  );
  canvas.setAttribute("role", "img");
  canvas.tabIndex = 0;
  host.append(canvas);
  const scene = new THREE.Scene();
  scene.background = null;
  const environment = new RoomEnvironment(),
    pmrem = new THREE.PMREMGenerator(renderer),
    environmentMap = pmrem.fromScene(environment, 0.04);
  scene.environment = environmentMap.texture;
  scene.environmentIntensity = 0.4;
  environment.dispose();
  pmrem.dispose();
  const camera = new THREE.PerspectiveCamera(39, 1, 0.06, 160),
    controls = new OrbitControls(camera, canvas);
  controls.enableDamping = false;
  controls.enablePan = false;
  controls.minPolarAngle = 0.18;
  controls.maxPolarAngle = Math.PI * 0.46;
  controls.minDistance = 1.8;
  controls.maxDistance = 65;
  scene.add(new THREE.HemisphereLight("#f1f7ff", "#8191a3", 1.15));
  const sun = new THREE.DirectionalLight("#fff1da", 2.45);
  sun.position.set(-10, 24, 10);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -26;
  sun.shadow.camera.right = 26;
  sun.shadow.camera.top = 26;
  sun.shadow.camera.bottom = -26;
  sun.shadow.camera.far = 75;
  sun.shadow.normalBias = 0.012;
  sun.shadow.bias = -0.00015;
  scene.add(sun);
  const fill = new THREE.DirectionalLight("#cfddff", 0.75);
  fill.position.set(16, 12, -10);
  scene.add(fill);
  const world = new THREE.Group(),
    staticWorld = new THREE.Group(),
    propWorld = new THREE.Group(),
    figureWorld = new THREE.Group();
  world.add(staticWorld, propWorld, figureWorld);
  scene.add(world);
  const mat = (color, roughness = 0.7, metalness = 0) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness });
  const mats = {
    floor: mat("#d8d9d5"),
    edge: mat("#b8c0c4"),
    wall: mat("#a6afb6"),
    wallTop: mat("#d2d6d6"),
    base: mat("#536776"),
    box: mat("#f4b83f", 0.46),
    plank: mat("#dc8c39", 0.56),
    ramp: mat("#58afa8", 0.55),
    inset: mat("#bf892f", 0.58),
    steel: mat("#5b6872", 0.4, 0.5),
    pale: mat("#fff0c7", 0.48),
    blue: mat("#438eec", 0.4),
    red: mat("#e15a55", 0.4),
    selected: mat("#ffda70", 0.42),
  };
  const permanent = new Set(Object.values(mats));
  let disposed = false,
    arena = null,
    props = new Map(),
    wallMeshes = [],
    characters = [],
    previous = null,
    lastSim = null,
    selection = -1,
    editing = false,
    visionMode = "both",
    follow = "none",
    cameraMode = "overview",
    fitFactor = 1;
  function mesh(geometry, material, parent = staticWorld, at = [0, 0, 0]) {
    const m = new THREE.Mesh(geometry, material);
    m.position.fromArray(at);
    m.castShadow = true;
    m.receiveShadow = true;
    parent.add(m);
    return m;
  }
  function box(size, at, material, parent = staticWorld, r = 0.035) {
    return mesh(
      new RoundedBoxGeometry(
        ...size,
        2,
        Math.min(r, ...size.map((x) => x * 0.24)),
      ),
      material,
      parent,
      at,
    );
  }
  function disposeGroup(group) {
    const gs = new Set(),
      ms = new Set();
    group.traverse((o) => {
      if (o.geometry) gs.add(o.geometry);
      if (o.material)
        for (const m of Array.isArray(o.material) ? o.material : [o.material])
          if (!permanent.has(m)) ms.add(m);
    });
    gs.forEach((g) => g.dispose());
    ms.forEach((m) => m.dispose());
    group.clear();
  }
  function floorGeometry(w, h) {
    box([w + 0.26, 0.22, h + 0.26], [w / 2, -0.12, h / 2], mats.edge);
    box([w, 0.055, h], [w / 2, -0.0275, h / 2], mats.floor);
    // Wide concrete slab seams, not an agent navigation grid.
    const points = [];
    for (let x = 2; x < w; x += 2) points.push(x, 0.002, 0, x, 0.002, h);
    for (let z = 2; z < h; z += 2) points.push(0, 0.002, z, w, 0.002, z);
    staticWorld.add(
      new THREE.LineSegments(
        new THREE.BufferGeometry().setAttribute(
          "position",
          new THREE.Float32BufferAttribute(points, 3),
        ),
        new THREE.LineBasicMaterial({
          color: "#8a969d",
          transparent: true,
          opacity: 0.17,
        }),
      ),
    );
    for (const [sx, sz, px, pz] of [
      [w + 0.16, 0.12, w / 2, -0.06],
      [w + 0.16, 0.12, w / 2, h + 0.06],
      [0.12, h + 0.16, -0.06, h / 2],
      [0.12, h + 0.16, w + 0.06, h / 2],
    ])
      box([sx, 0.075, sz], [px, 0.055, pz], mats.base, staticWorld, 0.014);
  }
  function wall(spec, index) {
    const size = spec.size,
      group = new THREE.Group();
    staticWorld.add(group);
    group.position.fromArray(spec.position);
    if (spec.quaternion) group.quaternion.fromArray(spec.quaternion);
    const m = box(size, [0, 0, 0], mats.wall, group, 0.055);
    m.userData.block = index;
    wallMeshes.push(m);
    box(
      [Math.max(0.02, size[0] - 0.025), 0.025, Math.max(0.02, size[2] - 0.025)],
      [0, size[1] / 2 + 0.004, 0],
      mats.wallTop,
      group,
      0.008,
    );
    box(
      [size[0] + 0.014, 0.075, size[2] + 0.014],
      [0, -size[1] / 2 + 0.04, 0],
      mats.base,
      group,
      0.012,
    );
    group.traverse((o) => {
      if (o.isMesh) {
        o.material = o.material.clone();
        o.material.transparent = true;
      }
    });
  }
  function rampGeometry([w, h, d]) {
    const p = [
      -w / 2,
      -h / 2,
      -d / 2,
      w / 2,
      -h / 2,
      -d / 2,
      w / 2,
      h / 2,
      -d / 2,
      -w / 2,
      -h / 2,
      d / 2,
      w / 2,
      -h / 2,
      d / 2,
      w / 2,
      h / 2,
      d / 2,
    ];
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    g.setIndex([
      0, 2, 1, 3, 4, 5, 0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4, 0, 3, 5, 0, 5, 2,
    ]);
    g.computeVertexNormals();
    return g;
  }
  function makeProp(o) {
    const group = new THREE.Group();
    propWorld.add(group);
    const [w, h, d] = o.size,
      material = mats[o.kind] || mats.box;
    const shape =
      o.kind === "ramp"
        ? mesh(rampGeometry(o.size), material, group)
        : box(o.size, [0, 0, 0], material, group, 0.055);
    if (o.kind !== "ramp") {
      // Recessed grip plates and corner straps distinguish movable objects.
      for (const side of [-1, 1]) {
        box(
          [Math.min(w * 0.45, 0.55), Math.min(h * 0.17, 0.14), 0.018],
          [0, 0, side * (d / 2 + 0.006)],
          mats.inset,
          group,
          0.018,
        );
        box(
          [Math.min(w * 0.3, 0.32), 0.032, 0.027],
          [0, 0, side * (d / 2 + 0.02)],
          mats.steel,
          group,
          0.012,
        );
      }
      if (h > 0.35)
        for (const x of [-1, 1])
          for (const z of [-1, 1])
            box(
              [0.035, Math.max(0.04, h - 0.08), 0.035],
              [x * (w / 2 - 0.018), 0, z * (d / 2 - 0.018)],
              mats.pale,
              group,
              0.012,
            );
    } else {
      const length = Math.hypot(w, h);
      for (let i = 1; i < 8; i++) {
        const x = -w / 2 + (w * i) / 8,
          y = -h / 2 + (h * i) / 8 + 0.012;
        const strip = box(
          [0.035, 0.012, d * 0.85],
          [x, y, 0],
          mats.pale,
          group,
          0.003,
        );
        strip.rotation.z = Math.atan2(h, w);
      }
    }
    const lock = new THREE.Group();
    group.add(lock);
    lock.position.set(0, h / 2 + 0.025, 0);
    const plate = mesh(
      new THREE.CylinderGeometry(0.105, 0.105, 0.018, 24),
      mats.steel,
      lock,
    );
    plate.castShadow = false;
    const lockBody = box(
        [0.085, 0.016, 0.072],
        [0, 0.018, 0.012],
        mats.pale,
        lock,
        0.014,
      ),
      shackle = mesh(
        new THREE.TorusGeometry(0.028, 0.008, 6, 16, Math.PI),
        mats.pale,
        lock,
        [0, 0.028, -0.025],
      );
    shackle.rotation.x = Math.PI / 2;
    lock.visible = false;
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(shape.geometry, 28),
      new THREE.LineBasicMaterial({
        color: "#fff2c4",
        transparent: true,
        opacity: 0.5,
      }),
    );
    group.add(outline);
    outline.visible = false;
    const index =
      arena?.objects?.findIndex((item) => item.id === o.id) ?? props.size;
    group.userData.objectIndex = index;
    group.userData.objectId = o.id;
    const item = {
      group,
      lock,
      plate,
      outline,
      size: o.size,
      kind: o.kind,
      index,
      shape,
      material,
    };
    props.set(o.id, item);
    return item;
  }
  const visionLayers = ["#418ce0", "#e0655a"].map((color) => {
    const material = new THREE.MeshBasicMaterial({color, transparent: true,
      opacity: 0.12, depthWrite: false, side: THREE.DoubleSide});
    const fill = new THREE.Mesh(new THREE.BufferGeometry(), material);
    const edge = new THREE.LineLoop(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.65}));
    fill.visible = edge.visible = false;
    scene.add(fill, edge);
    return {fill, edge, material};
  });
  function setArena(next, recenter = true) {
    arena = next;
    previous = null;
    lastSim = null;
    characters.forEach((c) => c.dispose());
    characters = [];
    figureWorld.clear();
    disposeGroup(staticWorld);
    disposeGroup(propWorld);
    props.clear();
    wallMeshes = [];
    floorGeometry(arena.width, arena.height);
    const walls =
      arena.walls ||
      arena.blocks?.map((b) => ({
        size: [b.width, 1.25, b.height],
        position: [b.x + b.width / 2, 0.625, b.y + b.height / 2],
      })) ||
      [];
    walls.forEach(wall);
    characters = [createCharacter(0, render), createCharacter(1, render)];
    characters.forEach((c, i) => {
      figureWorld.add(c.root);
      c.root.userData.agent = i;
    });
    for (const o of arena.objects || []) makeProp(o);
    if (recenter) frameOverview();
    render();
  }
  function frameOverview() {
    if (!arena) return;
    cameraMode = "overview";
    const w = arena.width,
      h = arena.height;
    controls.target.set(w / 2, 0.2, h / 2);
    const d = Math.max(w, h) * 1.13;
    fitFactor = Math.max(1, 1.25 / camera.aspect);
    camera.position
      .copy(controls.target)
      .add(
        new THREE.Vector3(d * 0.78, d * 0.88, d * 0.93).multiplyScalar(
          fitFactor,
        ),
      );
    controls.update();
  }
  const mix = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
  function interpolateAgent(a, b, t, supportFrames, supportSurfaces) {
    const p = mix(a.position, b.position, t),
      delta = Math.atan2(
        Math.sin(b.heading - a.heading),
        Math.cos(b.heading - a.heading),
      ),
      heading = a.heading + delta * t;
    return {
      ...b,
      position: p,
      heading,
      forward: [Math.cos(heading), 0, -Math.sin(heading)],
      supportFrames, supportSurfaces,
      support: interpolateSupport(a.support, b.support, t, supportFrames),
    };
  }
  function compatibility(sim) {
    return (
      sim.agents ||
      sim.pos.map((p, r) => ({
        position: [p[0], 0.55, p[1]],
        velocity: [sim.vel[r][0] * 6, 0, sim.vel[r][1] * 6],
        heading: Math.atan2(-sim.vel[r][1], sim.vel[r][0]),
        grounded: true,
        gripId: null,
        contacts: [],
      }))
    );
  }
  function update(sim, blend = 1) {
    if (!arena || disposed) return;
    lastSim = sim;
    const current = compatibility(sim),
      alpha = THREE.MathUtils.clamp(blend, 0, 1),
      time = (sim.time ?? sim.t * 0.08) - (1 - alpha) * (sim.dt || 0.08);
    const supportFrames = interpolateFrames(
      (previous?.objects || []).map(o => ({ ...o, objectId: o.id })),
      (sim.objects || arena.objects || []).map(o => ({ ...o, objectId: o.id })), alpha);
    const supportSurfaces = (arena.walls || []).map(w => ({ ...w, kind: 'box' }));
    current.forEach((a, i) =>
      characters[i]?.update(
        previous?.agents?.[i]
          ? interpolateAgent(previous.agents[i], a, alpha, supportFrames, supportSurfaces)
          : { ...a, supportFrames, supportSurfaces },
        time,
      ),
    );
    const ids = new Set();
    for (const o of sim.objects || arena.objects || []) {
      ids.add(o.id);
      const item = props.get(o.id) || makeProp(o),
        rendered = supportFrames.find(frame => frame.objectId === o.id);
      item.group.position.fromArray(rendered.position);
      item.group.quaternion.fromArray(rendered.quaternion || [0, 0, 0, 1]);
      item.lock.visible = o.lockedBy >= 0;
      item.plate.material =
        o.lockedBy === 0 ? mats.blue : o.lockedBy === 1 ? mats.red : mats.steel;
      item.outline.visible = o.grabbedBy >= 0 || selection === item.index;
      item.shape.material =
        selection === item.index ? mats.selected : item.material;
    }
    for (const [id, item] of props)
      if (!ids.has(id)) {
        disposeGroup(item.group);
        propWorld.remove(item.group);
        props.delete(id);
      }
    updateVision(sim);
    if (follow !== "none" && characters[Number(follow)]) {
      const target = characters[Number(follow)].root.position
          .clone()
          .add(new THREE.Vector3(0, 0.6, 0)),
        shift = target.clone().sub(controls.target);
      camera.position.add(shift);
      controls.target.copy(target);
      controls.update();
    }
    render();
  }
  function updateVision(sim) {
    for (const [role, layer] of visionLayers.entries()) {
      const {fill: visionFill, edge: visionEdge} = layer;
      const show = (visionMode === "both" || visionMode === (role ? "seeker" : "hider")) &&
        typeof sim.vision === "function" && !(role === 1 && sim.t < sim.prep);
      visionFill.visible = visionEdge.visible = show;
      if (!show) continue;
      const sight = sim.vision(role), poly = sight.polygon || [];
      if (poly.length < 2) {visionFill.visible = visionEdge.visible = false; continue;}
      const origin = sight.origin, positions = [], boundary = [origin, ...poly];
      for (let i = 0; i < poly.length - 1; i++)
        for (const point of [origin, poly[i], poly[i + 1]])
          positions.push(point[0], 0.032 + role * 0.002, point[2]);
      visionFill.geometry.dispose();
      visionFill.geometry = new THREE.BufferGeometry().setAttribute("position",
        new THREE.Float32BufferAttribute(positions, 3));
      visionEdge.geometry.dispose();
      visionEdge.geometry = new THREE.BufferGeometry().setFromPoints(
        boundary.map(p => new THREE.Vector3(p[0], 0.037 + role * 0.002, p[2])));
    }
  }

  function remember(sim) {
    previous = {
      agents: structuredClone(compatibility(sim)),
      objects: structuredClone(sim.objects || []),
    };
  }
  function setSelection(index) {
    selection = index;
    for (const item of props.values()) {
      item.shape.material =
        item.index === index ? mats.selected : item.material;
      item.outline.visible =
        item.index === index ||
        lastSim?.objects?.some(
          (o) => o.id === item.group.userData.objectId && o.grabbedBy >= 0,
        );
    }
    render();
  }
  function setEditing(value) {
    editing = value;
    controls.enableRotate = !value;
    canvas.style.cursor = value ? "crosshair" : "grab";
  }
  function setVision(value) {
    visionMode = value;
    if (lastSim) updateVision(lastSim);
    render();
  }
  function setFollow(value) {
    follow = value;
    if (value === "none") {
      frameOverview();
    } else if (characters[Number(value)]) {
      cameraMode = "follow";
      const target = characters[Number(value)].root.position
        .clone()
        .add(new THREE.Vector3(0, 0.55, 0));
      controls.target.copy(target);
      camera.position
        .copy(target)
        .add(
          new THREE.Vector3(1.45, 1.6, -2.2).applyQuaternion(
            characters[Number(value)].root.quaternion,
          ),
        );
      controls.update();
    }
    render();
  }
  function orbit(direction) {
    const offset = camera.position
      .clone()
      .sub(controls.target)
      .applyAxisAngle(new THREE.Vector3(0, 1, 0), (direction * Math.PI) / 6);
    camera.position.copy(controls.target).add(offset);
    controls.update();
    onOrbit?.();
    render();
  }
  function zoom(factor) {
    const offset = camera.position.clone().sub(controls.target);
    offset.setLength(
      THREE.MathUtils.clamp(
        offset.length() * factor,
        controls.minDistance,
        controls.maxDistance,
      ),
    );
    camera.position.copy(controls.target).add(offset);
    controls.update();
    render();
  }
  function render() {
    if (disposed) return;
    const subjects = follow === "none" ? characters : [characters[Number(follow)]];
    const sightlines = subjects.flatMap(character => [0.1, 0.4].map(height => {
      const target = character.root.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0,height,0));
      return {ray: new THREE.Ray(camera.position.clone(), target.clone().sub(camera.position).normalize()), distance: camera.position.distanceTo(target)};
    }));
    for (const wall of wallMeshes) {
      const group = wall.parent, bounds = new THREE.Box3().setFromObject(group);
      const faded = sightlines.some(({ray,distance}) => {
        const point = ray.intersectBox(bounds,new THREE.Vector3());
        return !!point && camera.position.distanceTo(point) < distance-.12;
      });
      group.traverse((o) => {
        if (o.isMesh) {
          o.material.opacity = faded ? 0.14 : 1;
          o.material.depthWrite = !faded;
        }
      });
    }
    renderer.render(scene, camera);
  }
  function capture() {
    render();
    return canvas.toDataURL("image/png");
  }
  function resize() {
    const w = host.clientWidth,
      h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (arena && cameraMode === "overview") {
      const next = Math.max(1, 1.25 / camera.aspect);
      camera.position
        .sub(controls.target)
        .multiplyScalar(next / fitFactor)
        .add(controls.target);
      fitFactor = next;
    }
    render();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(host);
  controls.addEventListener("change", render);
  const raycaster = new THREE.Raycaster(),
    pointer = new THREE.Vector2(),
    ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const click = (e) => {
    if (!editing) return;
    const rect = canvas.getBoundingClientRect();
    pointer.set(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      (-(e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(
        [...props.values()].map((p) => p.group),
        true,
      )[0],
      point = new THREE.Vector3();
    let object = hit?.object;
    while (object && object.userData.objectIndex === undefined)
      object = object.parent;
    if (raycaster.ray.intersectPlane(ground, point))
      onPick?.({
        x: Math.round(point.x * 20) / 20,
        y: Math.round(point.z * 20) / 20,
        index: object?.userData.objectIndex ?? -1,
      });
  };
  const key = (e) => {
    if (
      ![
        "ArrowLeft",
        "ArrowRight",
        "ArrowUp",
        "ArrowDown",
        "+",
        "=",
        "-",
      ].includes(e.key)
    )
      return;
    e.preventDefault();
    if (e.key === "ArrowLeft") orbit(-0.5);
    else if (e.key === "ArrowRight") orbit(0.5);
    else if (["+", "="].includes(e.key)) zoom(0.9);
    else if (e.key === "-") zoom(1.1);
    else {
      const offset = camera.position.clone().sub(controls.target),
        s = new THREE.Spherical().setFromVector3(offset);
      s.phi = THREE.MathUtils.clamp(
        s.phi + (e.key === "ArrowUp" ? -0.12 : 0.12),
        0.18,
        Math.PI * 0.46,
      );
      camera.position.copy(controls.target).add(offset.setFromSpherical(s));
      controls.update();
      render();
    }
  };
  canvas.addEventListener("click", click);
  canvas.addEventListener("keydown", key);
  resize();
  return {
    setArena,
    update,
    remember,
    setSelection,
    setEditing,
    setVision,
    setFollow,
    orbit,
    zoom,
    render,
    capture,
    get ready() {
      return Promise.all(characters.map((c) => c.ready));
    },
    diagnostics() {
      return {
        calls: renderer.info.render.calls,
        triangles: renderer.info.render.triangles,
        agents: characters.map((c) => c.root.userData.pose),
        objects: [...props].map(([id, p]) => ({
          id,
          position: p.group.position.toArray(),
          quaternion: p.group.quaternion.toArray(),
        })),
        visionVisible: visionLayers.some(layer => layer.fill.visible),
        visionRoles: visionLayers.map(layer => layer.fill.visible),
      };
    },
    dispose() {
      disposed = true;
      observer.disconnect();
      canvas.removeEventListener("click", click);
      canvas.removeEventListener("keydown", key);
      controls.removeEventListener("change", render);
      controls.dispose();
      characters.forEach((c) => c.dispose());
      figureWorld.clear();
      disposeGroup(staticWorld);
      disposeGroup(propWorld);
      for (const {fill, edge, material} of visionLayers) {
        fill.geometry.dispose(); edge.geometry.dispose(); material.dispose(); edge.material.dispose();
      }
      Object.values(mats).forEach((m) => m.dispose());
      environmentMap.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    },
  };
}
