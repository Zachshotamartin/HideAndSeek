import {
  generateArena,
  PhysicsSimulation,
  Random,
  editObject,
  validateArena,
  DT,
  PREP,
  OBS_DIM,
} from './core/physics.js';
import { createPolicyControllers, POLICY_FORMATS, BINARY_FORMAT } from './core/policyController.js';
import { PHYSICAL_POLICY_FILE, PHYSICAL_POLICY_DETAILS } from './core/policyAsset.js';
export const metadata = {
  id: 'hide-and-seek',
  title: 'Hide and Seek',
  description: 'Two learned agents move, hide, and manipulate objects in a physical arena.',
  technique: 'Original recurrent PPO self-play · MuJoCo rigid-body physics · Three.js',
  instructions: [
    'Watch the development checkpoint, compare object interactions with tools disabled, or rearrange props and replay the same seed. Both agents’ sight is visible; drag to orbit.',
  ],
  limitations: [
    'This development checkpoint uses an original environment and original policies inspired by the OpenAI study. Reliable useful tool strategies remain unproven; edited arenas can produce unfamiliar or unsuccessful behavior.',
    'Agents use upright collision bodies. Props translate, rise and turn about the vertical axis; they do not tumble freely.',
  ],
};
export function mountExperiment(element, options = {}) {
  const root = document.createElement('section');
  root.className = 'hide-seek';
  root.dataset.ready = 'false';
  root.dataset.embedded = String(Boolean(options.embedded));
  root.innerHTML = `${options.embedded ? '' : `<header class="hs-heading"><p class="hs-eyebrow">Two agents. A physical world.</p><h1>Hide and Seek</h1><p>Give them room, cover, and movable tools. Watch what they learn to do.</p></header>`}
 <div class="hs-layout"><div class="hs-stage-column"><div class="hs-board-shell"><div class="hs-board-top"><span class="hs-round-label">ROUND 01</span><span class="hs-phase">Loading physics</span><span class="hs-seed-label">SEED 2709</span></div><div class="hs-viewport"><p class="hs-loading">Starting the physical arena…</p></div><div class="hs-board-bottom"><span><i class="hs-dot"></i>Blue · Hider</span><span class="hs-visibility">A moment to prepare</span><span><i class="hs-dot hs-seeker"></i>Red · Seeker</span></div></div>
 <div class="hs-playback"><button class="hs-primary" data-action="play" disabled>Play</button><button data-action="step" disabled>Step</button><button data-action="reset" disabled>Reset</button><label class="hs-speed">Playback<select data-field="speed"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label></div><p class="hs-status" role="status" aria-live="polite">Loading the frozen policies and MuJoCo engine…</p>
 <div class="hs-round-stats"><div><span>Play time</span><strong data-stat="time">0.0 s</strong></div><div><span>Time hidden</span><strong data-stat="hidden">0.0 s</strong></div><div><span>Props held / locked</span><strong data-stat="tools">0 / 0</strong></div><div><span>Reward · H / S</span><strong data-stat="reward">0 / 0</strong></div></div>
 </div><aside class="hs-controls"><section class="hs-control-section"><div class="hs-section-heading"><span class="hs-index">01</span><h2>The agents</h2><span class="hs-ready-badge">LOADING</span></div><label>Policy checkpoint<select data-field="model" disabled><option value="trained">Development pair</option></select></label><div class="hs-two-fields"><label>Actions<select data-field="sampling"><option value="sample">Seeded sampling</option><option value="mean">Deterministic</option></select></label><label>Object interactions<select data-field="tools"><option value="on">Push, grab & lock</option><option value="off">Push only</option><option value="fixed">Fixed props</option></select></label></div><p class="hs-help">Two original recurrent neural policies. No training runs in your browser. Changing the tools resets the same scene for a causal comparison.</p><details class="hs-evaluation"><summary class="hs-eval-title">Training & measured behavior</summary><p class="hs-training-record">Loading the training record…</p><p class="hs-eval-note">Evaluation is attached to the supplied checkpoint. No result is inferred from the appearance of a round.</p><p class="hs-mode-note"></p></details></section>
 <section class="hs-control-section"><div class="hs-section-heading"><span class="hs-index">02</span><h2>The arena</h2></div><label>Layout<select data-field="scenario"><option value="shelter">Shelter & doorway</option><option value="rooms">Divided room</option><option value="connected-rooms">Connected rooms</option><option value="corridors">Corridors</option><option value="multi-exit">Multi-exit shelter</option><option value="open">Open arena</option></select></label><div class="hs-three-fields"><label>Size · metres<input data-field="size" type="number" min="6" max="12" step=".5" value="8"></label><label>Boxes / planks<input data-field="count" type="number" min="0" max="8" value="3"></label><label>Ramps<input data-field="ramps" type="number" min="0" max="2" value="1"></label></div><label>Repeatable seed<input data-field="seed" type="number" min="0" max="4294967295" value="2709"></label><div class="hs-actions"><button data-action="generate">Generate arena</button><button data-action="random">New seed</button></div><p class="hs-help">6–12 metres per side. Layouts rotate and reflect, with independent starting positions and up to ten physical props. The policies remain unchanged.</p>
 <details class="hs-editor"><summary>Arrange the props</summary><label>Pointer mode<select data-field="edit"><option value="orbit">Orbit camera</option><option value="select">Select a prop</option><option value="place">Place a prop</option></select></label><p class="hs-help">Click a prop to select it, or click the ground to place one. Coordinates and buttons offer the same controls without a pointer. Editing resets the round.</p><label>Selected prop<select data-field="object"><option value="-1">New prop</option></select></label><label>Object type<select data-field="kind"><option value="box">Box</option><option value="plank">Long plank</option><option value="ramp">Ramp</option></select></label><div class="hs-two-fields"><label>Position X · m<input data-field="x" type="number" step=".05" value="5"></label><label>Position Y · m<input data-field="y" type="number" step=".05" value="2"></label><label>Width · m<input data-field="width" type="number" min=".15" max="2" step=".05" value=".7"></label><label>Depth · m<input data-field="depth" type="number" min=".15" max="2" step=".05" value=".7"></label><label>Height · m<input data-field="height" type="number" min=".15" max="2" step=".05" value=".7"></label><label>Rotation · degrees<input data-field="yaw" type="number" step="15" value="0"></label></div><div class="hs-actions"><button data-action="add">Add prop</button><button data-action="move">Move / resize</button><button data-action="remove">Remove</button></div></details></section>
 <details class="hs-guide" open><summary>How the game works</summary><ul><li><strong>Preparation:</strong> the blue hider gets ${(PREP * DT).toFixed(1)} seconds to move and handle props. The red seeker receives no visual information and cannot act yet.</li><li><strong>Hide and seek:</strong> play continues until you pause or reset. During play, the hider earns +1 each decision while unseen and −1 while seen. The seeker earns the opposite. There is no touch-to-tag rule.</li><li><strong>Movement:</strong> the policies choose forward / sideways force, turning torque, and a grounded jump. The jump reaches low props but is capped below wall height. Acceleration, momentum, contact, friction, and gravity are resolved by MuJoCo.</li><li><strong>Tools:</strong> each policy chooses Keep, Press, or Release for its grab and lock buttons. The requested buttons persist until changed; this is learned control, not a minimum hold timer. Nearby visible objects can be grabbed, carried or pushed, then released. A physical lock anchors a prop; only its owner can unlock it. Ramps are solid wedges that agents can climb.</li><li><strong>Vision:</strong> a forward 135° field reaches across the arena. Walls and props occlude it. Thirty short-range rays sense nearby solid geometry; hidden opponent positions are not provided.</li><li><strong>Learning:</strong> the only game reward is visibility. The agents are not told to build a shelter, grab a box, or follow a prescribed route. A tool may remain unused if the learned policy finds no advantage in it.</li></ul></details>
 <details class="hs-files"><summary>Files & reproducibility</summary><p class="hs-help">Save the arena, a replay of actual decisions, or the frozen neural weights. Imported weights are unverified; published evaluation applies only to the supplied checkpoint.</p><div class="hs-actions"><button data-action="export-scene" disabled>Scene PNG</button><button data-action="export-arena">Arena JSON</button><button data-action="export-replay" disabled>Replay JSON</button><button data-action="export-model" disabled>Model JSON</button><button data-action="import">Import JSON</button></div><input class="hs-file" type="file" accept=".json,application/json" hidden></details></aside></div>`;
  element.append(root);
  const $ = (s) => root.querySelector(s),
    field = (n) => $(`[data-field="${n}"]`),
    button = (n) => $(`[data-action="${n}"]`);
  const abort = new AbortController();
  let disposed = false,
    epoch = 0,
    modelLoading = false,
    visualReady = false,
    view,
    mj,
    sim,
    policies,
    policyStates,
    arena = generateArena(2709),
    rng = new Random(2709),
    playing = false,
    offscreen = false,
    raf = 0,
    lastTime = 0,
    accumulator = 0,
    round = 1,
    selected = -1,
    models = {},
    modelName = 'trained',
    history = [],
    replay = null,
    checkpointEntries = new Map();
  function say(text, error = false) {
    if (disposed) return;
    $('.hs-status').textContent = text;
    $('.hs-status').classList.toggle('hs-error', error);
  }
  function canSee() {
    const r = $('.hs-board-shell').getBoundingClientRect();
    return (
      !document.hidden && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth
    );
  }
  function stop() {
    cancelAnimationFrame(raf);
    raf = 0;
    lastTime = 0;
  }
  function schedule() {
    if (!disposed && !raf && !offscreen && !document.hidden) raf = requestAnimationFrame(frame);
  }
  function refresh() {
    if (!sim) return;
    root.dataset.ready = String(Boolean((policies || replay) && !modelLoading && visualReady));
    root.dataset.replay = String(Boolean(replay));
    button('play').disabled = !visualReady || modelLoading || (!policies && !replay);
    root.dataset.steps = sim.t;
    root.dataset.phase = sim.phase;
    root.dataset.model = modelName;
    root.dataset.checkpoint = models[modelName]?.provenance?.checkpointSHA256 || '';
    root.dataset.trainingSteps = models[modelName]?.training?.totalPolicyInteractions || 0;
    root.dataset.policyFormat = models[modelName]?.format || '';
    root.dataset.jumps = sim.jumpEvents.join(',');
    root.dataset.grabs = sim.grabEvents.reduce((a, b) => a + b, 0);
    root.dataset.locks = sim.lockEvents.reduce((a, b) => a + b, 0);
    $('.hs-phase').textContent = sim.done
      ? 'Round complete'
      : sim.t < sim.prep
        ? 'Preparation'
        : sim.visible
          ? 'Hider in sight'
          : 'Hider unseen';
    $('.hs-round-label').textContent = `ROUND ${String(round).padStart(2, '0')}`;
    $('.hs-seed-label').textContent = `SEED ${arena.seed}`;
    $('.hs-visibility').textContent =
      sim.t < sim.prep
        ? `${((sim.prep - sim.t) * DT).toFixed(1)} s to prepare`
        : sim.done
          ? `${(sim.hidden * DT).toFixed(1)} s hidden`
          : `${(Math.max(0, sim.t - sim.prep) * DT).toFixed(1)} s elapsed`;
    $('[data-stat="time"]').textContent =
      `${(Math.max(0, sim.t - sim.prep) * DT).toFixed(1)} s`;
    $('[data-stat="hidden"]').textContent = `${(sim.hidden * DT).toFixed(1)} s`;
    $('[data-stat="tools"]').textContent =
      `${sim.grips.filter((x) => x >= 0).length} / ${sim.locks.filter((x) => x >= 0).length}`;
    $('[data-stat="reward"]').textContent = sim.returns.join(' / ');
    button('play').textContent = playing
      ? 'Pause'
      : sim.done || (replay && sim.t >= replay.actions.length)
        ? 'Play again'
        : 'Play';
    button('step').disabled = !visualReady || modelLoading || (!policies && !replay) || sim.done;
    button('export-replay').disabled = history.length === 0 || history.length >= 100000;
  }
  function reset(next = arena, recenter = false, keepReplay = false, invalidatePending = true) {
    if (invalidatePending) epoch++;
    modelLoading = false;
    field('model').value = modelName;
    playing = false;
    stop();
    if (!mj) {
      arena = next;
      refreshProps();
      return;
    }
    const nextSim = new PhysicsSimulation(mj, next, {
      disableTools: field('tools').value === 'off',
      immovable: field('tools').value === 'fixed',
      continuous: true,
    });
    sim?.dispose();
    sim = nextSim;
    arena = next;
    policyStates = policies?.map((p) => p.initialState());
    rng = new Random(arena.seed ^ 0x9e3779b9);
    history = [];
    if (!keepReplay) replay = null;
    accumulator = 0;
    selected = -1;
    view?.setArena(sim.viewArena, recenter);
    view?.setSelection(-1);
    view?.update(sim);
    refreshProps();
    refresh();
  }
  function advance() {
    if (!sim || sim.done || (!policies && !replay)) return;
    view?.remember(sim);
    try {
      let actions;
      if (replay) {
        actions = replay.actions[sim.t];
        if (!actions) {
          playing = false;
          say('The recorded replay has reached its last saved decision.');
          refresh();
          return;
        }
      } else
        actions = policies.map((policy, a) => {
          const out = policy.act(sim.observe(a), policyStates[a], {
            deterministic: field('sampling').value === 'mean',
            random: () => rng.next(),
          });
          policyStates[a] = out.state;
          return Array.from(out.action);
        });
      sim.step(actions);
      if (history.length < 100000) history.push(sim.actions.map((a) => Array.from(a)));
      if (sim.done) {
        playing = false;
        say(
          `Round complete. The hider stayed unseen for ${(sim.hidden * DT).toFixed(1)} of ${(sim.play * DT).toFixed(1)} play seconds. ${sim.grabEvents.reduce((a, b) => a + b, 0)} grabs and ${sim.lockEvents.reduce((a, b) => a + b, 0)} locks occurred.`,
        );
      }
      refresh();
    } catch (error) {
      playing = false;
      stop();
      say(error.message, true);
      refresh();
    }
  }
  function frame(time) {
    raf = 0;
    if (disposed || offscreen || document.hidden) return;
    const elapsed = lastTime ? Math.min(150, time - lastTime) : 0;
    lastTime = time;
    const interval = (DT * 1000) / Number(field('speed').value);
    if (playing) {
      accumulator += elapsed;
      let n = 0;
      while (accumulator >= interval && playing && n++ < 8) {
        accumulator -= interval;
        advance();
      }
    }
    view?.update(sim, playing ? Math.min(1, accumulator / interval) : 1);
    if (playing) schedule();
    else lastTime = 0;
  }
  function refreshProps() {
    field('object').replaceChildren(
      ...[
        [-1, 'New prop'],
        ...arena.objects.map((o, i) => [
          i,
          `${i + 1} · ${o.kind} · ${o.position[0].toFixed(1)}, ${o.position[1].toFixed(1)}`,
        ]),
      ].map(([value, text]) => {
        const o = document.createElement('option');
        o.value = value;
        o.textContent = text;
        return o;
      }),
    );
    field('object').value = String(selected);
  }
  function select(index) {
    selected = index;
    field('object').value = String(index);
    view?.setSelection(index);
    const o = arena.objects[index];
    if (!o) return;
    field('kind').value = o.kind;
    for (const [key, value] of [
      ['x', o.position[0]],
      ['y', o.position[1]],
      ['width', o.size[0]],
      ['depth', o.size[1]],
      ['height', o.size[2]],
      ['yaw', (o.yaw * 180) / Math.PI],
    ])
      field(key).value = Number(value.toFixed(3));
  }
  function edit(action) {
    try {
      if (action !== 'add' && selected < 0) throw Error('Select a prop first.');
      const dims = ['width', 'depth', 'height'].map((k) => Number(field(k).value)),
        kind = field('kind').value,
        o = {
          kind,
          size: dims,
          position: [Number(field('x').value), Number(field('y').value), dims[2] / 2 + 0.003],
          yaw: (Number(field('yaw').value) * Math.PI) / 180,
          mass: kind === 'plank' ? 1.8 : 1.2,
        };
      const next = editObject(
        arena,
        action === 'add' ? -1 : selected,
        action === 'remove' ? null : o,
      );
      reset(next);
      say('Prop updated. The physical round and policy memory reset.');
    } catch (error) {
      say(error.message, true);
    }
  }
  function download(data, name) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' })),
      a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  function applyModel(data, name) {
    const next = createPolicyControllers(data);
    if (next.some((p) => p.physicsObservationSize !== OBS_DIM))
      throw Error('The model has an incompatible observation schema.');
    models[name] = data;
    policies = next;
    modelName = name;
    if (!field('model').querySelector(`[value="${name}"]`)) {
      const o = document.createElement('option');
      o.value = name;
      o.textContent = name === 'imported' ? 'Imported pair · unverified' : name;
      field('model').append(o);
    }
    field('model').value = name;
    $('.hs-ready-badge').textContent =
      name === 'imported' ? 'IMPORTED' : name === 'initial' ? 'UNTRAINED' : 'DEVELOPMENT';
    showModelDetails(name);
  }
  function showModelDetails(name) {
    const details = checkpointEntries.get(name)?.details || PHYSICAL_POLICY_DETAILS;
    $('.hs-training-record').textContent =
      name === 'imported'
        ? 'Imported model. Its training history and performance have not been verified.'
        : name === 'initial'
          ? 'Zero game experience: the saved random backbone with uniform Keep / Press / Release outputs. This is a reference initialization, not the pilot’s warm start.'
          : details.training;
    $('.hs-eval-note').textContent =
      name === 'initial'
        ? 'An actual saved initialization checkpoint; compare its decisions with the trained pair on the same arena.'
        : name === 'imported'
          ? 'Published checkpoint evaluation does not apply to imported weights.'
          : details.evaluation;
    $('.hs-mode-note').textContent =
      name === 'trained' ? PHYSICAL_POLICY_DETAILS.modes : '';
  }
  async function selectCheckpoint(id) {
    const token = ++epoch;
    playing = false;
    stop();
    modelLoading = true;
    refresh();
    say('Loading the selected frozen checkpoint…');
    try {
      let data = models[id];
      if (!data) {
        const entry = checkpointEntries.get(id);
        if (!entry) throw Error('Checkpoint is unavailable.');
        const response = await fetch(new URL(`models/${entry.file}`, assetBase), {
          signal: abort.signal,
        });
        if (!response.ok) throw Error('The selected checkpoint could not be loaded.');
        data = await response.json();
      }
      if (disposed || token !== epoch) return;
      applyModel(data, id);
      reset();
      say(
        id === 'initial'
          ? 'The networks are at their saved random initialization.'
          : 'Development checkpoint loaded. The arena and repeatable action seed are unchanged.',
      );
    } catch (error) {
      if (!disposed && token === epoch) {
        modelLoading = false;
        field('model').value = modelName;
        refresh();
        say(`Unable to load checkpoint: ${error.message}`, true);
      }
    }
  }
  async function importFile(file) {
    if (!file) return;
    const token = ++epoch;
    modelLoading = false;
    field('model').value = modelName;
    refresh();
    try {
      if (file.size > 12000000) throw Error('JSON files must be smaller than 12 MB.');
      const text = await file.text();
      if (disposed || token !== epoch) return;
      const data = JSON.parse(text);
      if (POLICY_FORMATS.includes(data.format)) {
        applyModel(data, 'imported');
        reset();
        $('.hs-training-record').textContent =
          'Imported model. Its training history and performance have not been verified.';
        $('.hs-eval-note').textContent =
          'Published checkpoint evaluation does not apply to imported weights.';
        say(data.format === BINARY_FORMAT
          ? 'Imported legacy binary policies loaded. Tools use immediate binary actions.'
          : 'Imported persistent-button policies loaded. Each actor’s memory and requested buttons reset.');
      } else if (data.format === 'hide-seek-replay-v1') {
        const next = validateArena(data.arena);
        if (
          !Array.isArray(data.actions) ||
          !data.actions.length ||
          data.actions.length > 100000 ||
          data.actions.some(
            (a) =>
              !Array.isArray(a) ||
              a.length !== 2 ||
              a.some(
                (v) =>
                  !Array.isArray(v) ||
                  ![5, 6].includes(v.length) ||
                  v.some((x) => !Number.isFinite(x) || Math.abs(x) > 1),
              ),
          )
        )
          throw Error('Invalid recorded actions.');
        if (!['on', 'off', 'fixed'].includes(data.tools))
          throw Error('Invalid replay tool setting.');
        field('tools').value = data.tools;
        reset(next, true);
        replay = data;
        root.dataset.replay = 'true';
        say('Recorded decisions loaded. Play re-simulates the saved physical actions.');
      } else {
        if (data.actors) throw Error('Unsupported policy format; model schemas are not interchangeable.');
        reset(validateArena(data), true);
        say('Arena imported. The frozen policies are ready for this new arrangement.');
      }
      field('seed').value = arena.seed;
      field('size').value = arena.width;
      refresh();
    } catch (error) {
      if (!disposed && token === epoch) say(`Import failed: ${error.message}`, true);
    }
  }
  const click = (e) => {
    const action = e.target.closest('[data-action]')?.dataset.action;
    if (!action) return;
    if (action === 'play' && sim) {
      if (sim.done || (replay && sim.t >= replay.actions.length)) {
        round++;
        const saved = replay;
        reset(arena, false, true);
        replay = saved;
      }
      playing = !playing;
      offscreen = !canSee();
      accumulator = 0;
      lastTime = 0;
      say(
        playing
          ? replay
            ? 'Replaying recorded decisions.'
            : 'Both recurrent policies are choosing physical actions.'
          : 'Playback paused.',
      );
      refresh();
      if (playing) schedule();
      else {
        stop();
        view?.update(sim, 1);
      }
    }
    if (action === 'step') {
      playing = false;
      stop();
      advance();
      view?.update(sim);
    }
    if (action === 'reset') {
      const saved = replay;
      reset(arena, false, true);
      replay = saved;
      say('Reset to the same arena, policy memory, and action seed.');
    }
    if (action === 'generate' || action === 'random')
      try {
        if (action === 'random')
          field('seed').value = crypto.getRandomValues(new Uint32Array(1))[0];
        const next = generateArena(
          Number(field('seed').value),
          field('scenario').value,
          Number(field('size').value),
          Number(field('count').value),
          Number(field('ramps').value),
        );
        round++;
        reset(next, true);
        say(`${arena.objects.length} physical props placed. The policies are unchanged.`);
      } catch (error) {
        say(error.message, true);
      }
    if (['add', 'move', 'remove'].includes(action)) edit(action);
    if (action === 'left' || action === 'right') view?.orbit(action === 'left' ? -1 : 1);
    if (action === 'zoom-in' || action === 'zoom-out')
      view?.zoom(action === 'zoom-in' ? 0.85 : 1 / 0.85);
    if (action === 'export-scene' && view) {
      view.update(sim, 1);
      const a = document.createElement('a');
      a.href = view.capture();
      a.download = `hide-seek-${arena.seed}.png`;
      a.click();
    }
    if (action === 'export-arena') download(arena, `hide-seek-arena-${arena.seed}.json`);
    if (action === 'export-model' && models[modelName])
      download(models[modelName], 'hide-seek-policy.json');
    if (action === 'export-replay' && history.length)
      download(
        {
          format: 'hide-seek-replay-v1',
          arena,
          actions: history,
          tools: field('tools').value,
          model: models[modelName]?.training || modelName,
          policyFormat: models[modelName]?.format,
        },
        `hide-seek-replay-${arena.seed}.json`,
      );
    if (action === 'import') $('.hs-file').click();
  };
  const change = (e) => {
    const name = e.target.dataset.field;
    if (name === 'model') void selectCheckpoint(field('model').value);
    if (name === 'sampling' || name === 'tools') {
      reset();
      say(
        name === 'tools'
          ? 'Object physics updated. Compare the same arena with this intervention.'
          : 'Action sampling changed. The replay seed reset.',
      );
    }
    if (name === 'edit') {
      playing = false;
      stop();
      view?.setEditing(field('edit').value !== 'orbit');
      refresh();
      say(
        field('edit').value === 'place'
          ? 'Click clear ground to place a physical prop.'
          : 'Choose a prop or drag to orbit.',
      );
    }
    if (name === 'object') select(Number(field('object').value));
    if (name === 'kind') {
      const d =
        field('kind').value === 'ramp'
          ? [0.9, 0.8, 0.7]
          : field('kind').value === 'plank'
            ? [1.7, 0.25, 0.7]
            : [0.7, 0.7, 0.7];
      ['width', 'depth', 'height'].forEach((k, i) => (field(k).value = d[i]));
    }
    if (e.target.matches('.hs-file')) {
      const file = e.target.files[0];
      e.target.value = '';
      void importFile(file);
    }
  };
  root.addEventListener('click', click);
  root.addEventListener('change', change);
  const reconcile = () => {
    offscreen = !canSee();
    if (offscreen || document.hidden) {
      stop();
      if (playing) say('Playback is paused while the arena is out of view.');
    } else if (playing) {
      say('Playback resumed.');
      schedule();
    }
  };
  const observer = new IntersectionObserver(
    (entries) => {
      const latest = entries.at(-1);
      if (!latest) return;
      offscreen = !latest.isIntersecting;
      if (offscreen || document.hidden) {
        stop();
        if (playing) say('Playback is paused while the arena is out of view.');
      } else if (playing) {
        say('Playback resumed.');
        schedule();
      }
    },
    { threshold: 0 },
  );
  observer.observe($('.hs-board-shell'));
  document.addEventListener('visibilitychange', reconcile);
  const assetBase = options.assetBase
    ? new URL(
        options.assetBase.endsWith('/') ? options.assetBase : `${options.assetBase}/`,
        document.baseURI,
      )
    : new URL('.', document.baseURI);
  const modelsPromise = fetch(new URL(PHYSICAL_POLICY_FILE, assetBase), {
    signal: abort.signal,
  }).then((r) => {
    if (!r.ok) throw Error('The trained checkpoint could not be loaded.');
    return r.json();
  });
  Promise.all([
    import('./renderer.js'),
    import('./core/loadPhysics.js').then((m) => m.loadPhysics()),
  ])
    .then(async ([renderer, engine]) => {
      if (disposed) return;
      mj = engine;
      $('.hs-loading')?.remove();
      view = renderer.createView($('.hs-viewport'), {
        transparent: true,
        onPick: (p) => {
          if (field('edit').value === 'select') {
            select(p.index);
            say(p.index < 0 ? 'No prop at that position.' : `Prop ${p.index + 1} selected.`);
          } else if (field('edit').value === 'place') {
            field('x').value = p.x;
            field('y').value = p.y;
            edit('add');
          }
        },
      });
      reset(arena, true, true, false);
      await view.ready;
      if (disposed) return;
      visualReady = true;
      refresh();
      button('reset').disabled = false;
      button('export-scene').disabled = false;
      return modelsPromise;
    })
    .then((data) => {
      if (disposed || !data) return;
      models.trained = data;
      if (!policies) applyModel(data, 'trained');
      policyStates ??= policies.map((p) => p.initialState());
      showModelDetails(modelName);
      for (const n of ['play', 'step', 'export-model']) button(n).disabled = false;
      field('model').disabled = false;
      root.dataset.ready = 'true';
      playing = true;
      schedule();
      refresh();
      say(modelName === 'imported'
        ? 'The physical arena is ready. Imported policies remain unverified.'
        : modelName === 'initial'
          ? 'The physical arena and saved zero-experience policies are ready.'
          : 'The physical arena and development policies are ready. Useful tool strategies remain unproven.');
    })
    .catch((error) => {
      if (!disposed && error.name !== 'AbortError') {
        say(`Unable to start: ${error.message}`, true);
        $('.hs-loading')?.remove();
      }
    });
  fetch(new URL(`models/MANIFEST.json?policy=${encodeURIComponent(PHYSICAL_POLICY_FILE)}`, assetBase), { signal: abort.signal, cache: 'no-cache' })
    .then((r) => (r.ok ? r.json() : null))
    .then((manifest) => {
      if (disposed || !Array.isArray(manifest?.checkpoints) || `models/${manifest.file}` !== PHYSICAL_POLICY_FILE) return;
      for (const entry of manifest.checkpoints.slice(0, 12)) {
        if (
          !entry ||
          !/^[-\w]{1,30}$/.test(entry.id) ||
          !/^physical-policy[-\w.]*\.json$/.test(entry.file) ||
          typeof entry.label !== 'string'
        )
          continue;
        checkpointEntries.set(entry.id, entry);
        const existing = field('model').querySelector(`[value="${entry.id}"]`);
        if (existing) existing.textContent = entry.label;
        else {
          const option = document.createElement('option');
          option.value = entry.id;
          option.textContent = entry.label;
          field('model').append(option);
        }
      }
    })
    .catch(() => {});
  // Attach a rejection handler immediately, including when engine compilation fails first.
  modelsPromise.catch(() => {});
  refreshProps();
  return {
    dispose() {
      if (disposed) return;
      disposed = true;
      epoch++;
      playing = false;
      stop();
      abort.abort();
      observer.disconnect();
      document.removeEventListener('visibilitychange', reconcile);
      root.removeEventListener('click', click);
      root.removeEventListener('change', change);
      view?.dispose();
      sim?.dispose();
      root.remove();
    },
  };
}
