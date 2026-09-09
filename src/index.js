import { generateArena, Simulation, Random, editBlock, validateArena } from './core/arena.js';
import { chooseAction, validateModel } from './core/policy.js';

export const metadata = {
  id: 'hide-and-seek', title: 'Hide and Seek',
  description: 'Two pretrained agents play hide and seek in a 3D arena. Rearrange the cover and watch what their learned policies do.',
  technique: 'Two PPO neural policies · restricted vision and memory · Three.js',
  instructions: ['Press Play to watch the pretrained pair. Generate a seeded arena or edit its cover, inspect either agent’s sight, and compare with the same networks before training.'],
  limitations: ['This small experiment learns hiding and pursuit with static cover. It does not reproduce emergent tool use. Policies can collide, lose track of an opponent, and perform worse outside the training distribution.']
};
const fmt = n => Math.round(n).toLocaleString('en-US');
export function mountExperiment(element, options = {}) {
  const root = document.createElement('section'); root.className = 'hide-seek'; root.dataset.ready = 'false'; root.dataset.embedded = String(Boolean(options.embedded));
  root.innerHTML = `${options.embedded ? '' : `<header class="hs-heading"><p class="hs-eyebrow">Two minds. One small world.</p><h1>Hide and Seek</h1><p>Move the cover. Change the arena. Watch two pretrained agents play.</p></header>`}
    <div class="hs-layout"><div class="hs-stage-column">
      <div class="hs-board-shell"><div class="hs-board-top"><span class="hs-round-label">ROUND 01</span><span class="hs-phase">Loading models</span><span class="hs-seed-label">SEED 2709</span></div><div class="hs-viewport"><p class="hs-loading">Preparing the arena…</p></div><div class="hs-board-bottom"><span><i class="hs-dot hs-hider"></i>Crown · Hider</span><span class="hs-visibility">24 steps to hide</span><span><i class="hs-dot hs-seeker"></i>Visor · Seeker</span></div></div>
      <div class="hs-playback"><button class="hs-primary" data-action="play" disabled>Play</button><button data-action="step" disabled>Step</button><button data-action="reset" disabled>Reset</button><label class="hs-speed">Steps / s<select data-field="speed"><option value="6">6</option><option value="12" selected>12</option><option value="24">24</option><option value="48">48</option></select></label></div>
      <p class="hs-status" role="status" aria-live="polite">Loading two frozen neural policies…</p>
      <div class="hs-round-stats"><div><span>Play steps</span><strong data-stat="steps">0 / 180</strong></div><div><span>Hidden steps</span><strong data-stat="hidden">0</strong></div><div><span>Wall collisions</span><strong data-stat="collisions">0 · 0</strong></div><div><span>Reward H · S</span><strong data-stat="reward">0.00 · 0.00</strong></div></div>
      <div class="hs-view-controls"><label>Inspect sight<select data-field="vision"><option value="none">Spectator</option><option value="hider">Hider vision</option><option value="seeker">Seeker vision</option></select></label><div class="hs-camera"><button data-action="left" aria-label="Rotate view left">↶</button><button data-action="right" aria-label="Rotate view right">↷</button><button data-action="zoom-in" aria-label="Zoom in">+</button><button data-action="zoom-out" aria-label="Zoom out">−</button></div><p>Drag to orbit. Scroll or pinch to zoom. The spectator camera sees both agents; the agents do not.</p></div>
    </div><aside class="hs-controls">
      <section class="hs-control-section"><div class="hs-section-heading"><span class="hs-index">01</span><h2>The agents</h2><span class="hs-ready-badge">PRETRAINED</span></div><label>Policy checkpoint<select data-field="model" disabled><option value="trained">Pretrained pair</option><option value="initial">Before training</option></select></label><p class="hs-help">The same two small neural networks run on every new layout. No training happens in your browser.</p><div class="hs-evaluation"><p class="hs-eval-title">Measured on unseen arenas</p><div class="hs-eval-values"><div><strong data-eval="seeker">—</strong><span>Seeker captures</span></div><div><strong data-eval="hider">—</strong><span>Hider survives</span></div></div><p class="hs-eval-note">Loading the evaluation record…</p></div></section>
      <section class="hs-control-section"><div class="hs-section-heading"><span class="hs-index">02</span><h2>The arena</h2></div><div class="hs-three-fields"><label>Width<input data-field="width" type="number" min="10" max="20" value="14"></label><label>Height<input data-field="height" type="number" min="10" max="20" value="12"></label><label>Cover<input data-field="count" type="number" min="0" max="12" value="6"></label></div><label>Repeatable seed<input data-field="seed" type="number" min="1" max="4294967295" value="2709"></label><div class="hs-actions"><button data-action="generate">Generate arena</button><button data-action="random" aria-label="Use a new random seed">New seed</button></div><p class="hs-help">10–20 cells per side, up to 12 blocks. Open floor stays connected. Size 10–18 × 10–16 and 2–9 blocks match the later training range.</p>
      <details class="hs-editor"><summary>Edit cover</summary><label>Pointer mode<select data-field="edit"><option value="orbit">Orbit camera</option><option value="select">Select a block</option><option value="place">Place a block</option></select></label><p class="hs-help">Select a block in the arena, or use these coordinates. Placing, moving, or removing cover resets the round.</p><div class="hs-two-fields"><label>Column X<input data-field="x" type="number" min="1" max="18" value="3"></label><label>Row Y<input data-field="y" type="number" min="1" max="18" value="3"></label><label>Block width<input data-field="bw" type="number" min="1" max="3" value="1"></label><label>Block depth<input data-field="bh" type="number" min="1" max="3" value="2"></label></div><label>Selected block<select data-field="block"><option value="-1">New block</option></select></label><div class="hs-actions"><button data-action="add">Add cover</button><button data-action="move">Move / resize</button><button data-action="remove">Remove</button></div></details></section>
      <details class="hs-guide" open><summary>How the game works</summary><ul><li><strong>Hider:</strong> the crowned sage blob gets 24 preparation steps. It wins by avoiding a tag for 180 play steps.</li><li><strong>Seeker:</strong> the terracotta blob stays blind and still during preparation, then wins by approaching within 0.65 cells with a clear line of sight.</li><li><strong>Cover:</strong> solid blocks stop movement and block sight. The outer rim is also solid. Objects cannot be pushed or grabbed.</li><li><strong>Vision:</strong> both agents see in all directions, at most 7 cells away. Walls occlude opponents. The outline shows visible range; eight short rays show wall sensing.</li><li><strong>Memory:</strong> each agent retains an opponent’s last observed position for 60 steps. It never receives the opponent’s current hidden position.</li></ul><p class="hs-help">Actions are wait, up, right, down, and left. The reward favors a tag or surviving the round; small visibility, progress, and exploration rewards aid offline training.</p></details>
      <details class="hs-files"><summary>Export, import & training details</summary><p class="hs-training-record">Loading training record…</p><p class="hs-help">Exports stay on your device. Imported policies are unverified; the published test results apply only to the supplied pretrained pair.</p><div class="hs-actions"><button data-action="export-scene" disabled>Export scene PNG</button><button data-action="export-arena">Export arena</button><button data-action="export-model" disabled>Export model</button><button data-action="import">Import JSON</button></div><input type="file" class="hs-file" accept=".json,application/json" hidden></details>
    </aside></div>`;
  element.append(root);
  const $ = selector => root.querySelector(selector), field = name => $(`[data-field="${name}"]`), button = name => $(`[data-action="${name}"]`);
  let disposed = false, epoch = 0, view, modelData, models = {}, currentModel = 'trained', arena = generateArena(), sim = new Simulation(arena), rng = new Random(arena.seed ^ 0x9e3779b9), selected = -1;
  let playing = false, offscreen = false, raf = 0, previousTime = 0, accumulator = 0, round = 1, status = '';
  const abort = new AbortController();
  function say(text, error = false) { if (disposed) return; status = text; $('.hs-status').textContent = text; $('.hs-status').classList.toggle('hs-error', error); }
  function visible() { const r = $('.hs-board-shell').getBoundingClientRect(); return !document.hidden && r.bottom > 0 && r.top < window.innerHeight && r.right > 0 && r.left < window.innerWidth; }
  function update() {
    root.dataset.steps = sim.t; root.dataset.phase = sim.done ? sim.capture ? 'tagged' : 'escaped' : sim.t < sim.prep ? 'preparation' : 'seeking'; root.dataset.model = currentModel;
    $('.hs-phase').textContent = sim.done ? sim.capture ? 'Seeker wins' : 'Hider wins' : sim.t < sim.prep ? 'A moment to hide' : sim.visible ? 'In sight' : 'Out of sight';
    $('.hs-round-label').textContent = `ROUND ${String(round).padStart(2, '0')}`; $('.hs-seed-label').textContent = `SEED ${arena.seed}`;
    $('.hs-visibility').textContent = sim.t < sim.prep ? `${sim.prep - sim.t} steps to hide` : sim.done ? sim.capture ? 'Tagged with clear sight' : 'Time is up · not tagged' : `${Math.max(0, 180 - (sim.t - sim.prep))} steps remain`;
    $('[data-stat="steps"]').textContent = `${Math.max(0, sim.t - sim.prep)} / 180`; $('[data-stat="hidden"]').textContent = sim.hidden; $('[data-stat="collisions"]').textContent = sim.collisions.join(' · '); $('[data-stat="reward"]').textContent = sim.returns.map(v => v.toFixed(2)).join(' · ');
    button('play').textContent = playing ? 'Pause' : sim.done ? 'Play again' : 'Play'; button('step').disabled = !view || !models[currentModel] || sim.done;
  }
  function stopFrame() { cancelAnimationFrame(raf); raf = 0; previousTime = 0; }
  function schedule() { if (!disposed && !raf && !document.hidden && !offscreen) raf = requestAnimationFrame(frame); }
  function advance() {
    if (!models[currentModel] || sim.done) return;
    view?.remember(sim); const actions = [0, 1].map(role => chooseAction(models[currentModel].policies[role], sim.observe(role), rng)); sim.step(actions); update();
    if (sim.done) { playing = false; say(sim.capture ? `The seeker tagged the hider after ${sim.t - sim.prep} play steps.` : 'The hider survived all 180 play steps.'); update(); }
  }
  function frame(time) {
    raf = 0; if (disposed || document.hidden || offscreen) return;
    const dt = previousTime ? Math.min(100, time - previousTime) : 0; previousTime = time; const interval = 1000 / Number(field('speed').value);
    if (playing) { accumulator += dt; let n = 0; while (accumulator >= interval && playing && n++ < 6) { accumulator -= interval; advance(); } }
    view?.update(sim, playing ? Math.min(1, accumulator / interval) : 1);
    if (playing) schedule(); else previousTime = 0;
  }
  function refreshBlocks() { field('block').replaceChildren(...[[-1, 'New block'], ...arena.blocks.map((b, i) => [i, `Block ${i + 1} · ${b.x}, ${b.y} · ${b.width}×${b.height}`])].map(([v, text]) => { const o = document.createElement('option'); o.value = v; o.textContent = text; return o; })); field('block').value = String(selected); }
  function reset(next = arena, recenter = false) { epoch++; playing = false; stopFrame(); arena = next; sim = new Simulation(arena); rng = new Random(arena.seed ^ 0x9e3779b9); accumulator = 0; selected = -1; view?.setSelection(-1); view?.setArena(arena, recenter); view?.update(sim); refreshBlocks(); update(); }
  function selectBlock(n) { selected = n; field('block').value = String(n); view?.setSelection(n); const b = arena.blocks[n]; if (b) for (const [f, k] of [['x', 'x'], ['y', 'y'], ['bw', 'width'], ['bh', 'height']]) field(f).value = b[k]; }
  function edit(action) { try { const b = { x: Number(field('x').value), y: Number(field('y').value), width: Number(field('bw').value), height: Number(field('bh').value) }; if (action !== 'add' && selected < 0) throw new Error('Select a cover block first.'); const next = editBlock(arena, action === 'add' ? -1 : selected, action === 'remove' ? null : b); reset(next); say('Cover updated. The round reset; both frozen policies are unchanged.'); } catch (e) { say(e.message, true); } }
  function download(data, filename) { const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })), a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 0); }
  async function importFile(file) {
    if (!file) return; const token = ++epoch;
    try { if (file.size > 500000) throw new Error('JSON files must be smaller than 500 KB.'); const text = await file.text(); if (disposed || token !== epoch) return; const data = JSON.parse(text);
      if (data.format === 'hide-seek-ppo-v1') { models.imported = validateModel(data); if (!field('model').querySelector('[value="imported"]')) { const o = document.createElement('option'); o.value = 'imported'; o.textContent = 'Imported pair · unverified'; field('model').append(o); } currentModel = 'imported'; field('model').value = currentModel; reset(); showEvaluation(); say('Imported policy loaded. Published evaluation does not apply to this model.'); }
      else { const next = validateArena(data); reset(next, true); field('width').value = arena.width; field('height').value = arena.height; field('count').value = arena.blocks.length; field('seed').value = arena.seed; say('Arena imported. The pretrained policies can play immediately.'); }
    } catch (e) { if (!disposed && token === epoch) say(`Import failed: ${e.message}`, true); }
  }
  function showEvaluation() {
    const verified = currentModel === 'trained', ev = modelData?.evaluation;
    $('[data-eval="seeker"]').textContent = verified && ev?.summary ? `${(ev.summary.seekerCapture * 100).toFixed(1)}%` : '—'; $('[data-eval="hider"]').textContent = verified && ev?.summary ? `${(ev.summary.hiderSurvival * 100).toFixed(1)}%` : '—';
    $('.hs-eval-note').textContent = verified && ev?.summary ? `${ev.summary.episodesPerMatchup} rounds across ${ev.summary.arenas} held-out arenas, with 3 starting arrangements each. Seeker vs a reactive fleeing baseline; hider vs a reactive chasing/searching baseline. Separate matchups, not each other.` : verified ? 'Evaluation is being prepared for this model release.' : 'Published scores apply to the supplied pretrained pair. Play this checkpoint to compare its actual behavior.';
    $('.hs-ready-badge').textContent = currentModel === 'trained' ? 'PRETRAINED' : currentModel === 'initial' ? 'UNTRAINED' : 'IMPORTED';
  }
  const click = e => {
    const action = e.target.closest('[data-action]')?.dataset.action; if (!action) return;
    if (action === 'play') { if (sim.done) { round++; reset(); } playing = !playing; offscreen = !visible(); accumulator = 0; previousTime = 0; say(playing ? 'Both agents are choosing actions from their neural policies.' : 'Playback paused.'); update(); if (playing) schedule(); else stopFrame(); }
    if (action === 'step') { playing = false; stopFrame(); advance(); view?.update(sim); say(sim.done ? status : 'Advanced one simultaneous decision.'); }
    if (action === 'reset') { reset(); say('Round reset to the same arena and action seed.'); }
    if (action === 'generate' || action === 'random') { try { if (action === 'random') field('seed').value = crypto.getRandomValues(new Uint32Array(1))[0] || 1; const next = generateArena(Number(field('seed').value), Number(field('width').value), Number(field('height').value), Number(field('count').value)); round++; reset(next, true); say(`${arena.blocks.length} cover blocks placed. The same pretrained models are ready.`); } catch (e) { say(e.message, true); } }
    if (['add', 'move', 'remove'].includes(action)) edit(action);
    if (action === 'left' || action === 'right') view?.orbit(action === 'left' ? -1 : 1);
    if (action === 'zoom-in' || action === 'zoom-out') view?.zoom(action === 'zoom-in' ? .85 : 1 / .85);
    if (action === 'export-scene' && view) { const a = document.createElement('a'); a.href = view.capture(); a.download = `hide-seek-scene-${arena.seed}.png`; a.click(); say('Transparent scene PNG exported.'); }
    if (action === 'export-arena') { download(arena, `hide-seek-arena-${arena.seed}.json`); say('Arena JSON exported.'); }
    if (action === 'export-model' && models[currentModel]) { download(models[currentModel], 'hide-seek-model.json'); say('Frozen neural model exported.'); }
    if (action === 'import') $('.hs-file').click();
  };
  const change = e => { const name = e.target.dataset.field;
    if (name === 'model') { currentModel = field('model').value; reset(); showEvaluation(); say(currentModel === 'initial' ? 'Random initial weights loaded. This is the same architecture before learning.' : 'Checkpoint loaded. The arena and replay seed are unchanged.'); }
    if (name === 'vision') { view?.setVision(field('vision').value); view?.update(sim); }
    if (name === 'edit') { playing = false; stopFrame(); view?.setEditing(field('edit').value !== 'orbit'); update(); say(field('edit').value === 'place' ? 'Click open floor to place a block, or use the coordinates below.' : field('edit').value === 'select' ? 'Click a cover block to select it, or choose it from the list.' : 'Drag the arena to orbit the camera.'); }
    if (name === 'block') selectBlock(Number(field('block').value));
    if (e.target.matches('.hs-file')) { const file = e.target.files[0]; e.target.value = ''; void importFile(file); }
  };
  root.addEventListener('click', click); root.addEventListener('change', change);
  function reconcileVisibility() { offscreen = !visible(); if (document.hidden || offscreen) { stopFrame(); if (playing) say('Playback is paused while this arena is out of view.'); } else { if (playing) say('Playback resumed.'); schedule(); } }
  const intersection = new IntersectionObserver(entries => { const latest = entries.at(-1); if (!latest) return; offscreen = !latest.isIntersecting; if (offscreen || document.hidden) { stopFrame(); if (playing) say('Playback is paused while this arena is out of view.'); } else { if (playing) say('Playback resumed.'); schedule(); } }, { threshold: 0 }); intersection.observe($('.hs-board-shell'));
  document.addEventListener('visibilitychange', reconcileVisibility);
  const assetUrl = new URL('models/hide-seek.json', options.assetBase ? new URL(options.assetBase.endsWith('/') ? options.assetBase : `${options.assetBase}/`, document.baseURI) : new URL('.', document.baseURI));
  Promise.all([import('./renderer.js'), fetch(assetUrl, { signal: abort.signal }).then(r => { if (!r.ok) throw new Error('The pretrained model file could not be loaded.'); return r.json(); })]).then(([renderer, data]) => {
    if (disposed) return; models.trained = validateModel(data.trained); models.initial = validateModel(data.initial); modelData = data;
    $('.hs-loading').remove(); view = renderer.createView($('.hs-viewport'), { transparent: Boolean(options.embedded), onPick: p => { if (field('edit').value === 'select') { selectBlock(p.index); say(p.index < 0 ? 'No cover block at that position.' : `Block ${p.index + 1} selected.`); } else if (field('edit').value === 'place') { field('x').value = p.x; field('y').value = p.y; edit('add'); } } });
    view.setArena(arena); view.update(sim); refreshBlocks(); showEvaluation();
    $('.hs-training-record').textContent = `${fmt(data.training.steps)} simulated arena steps across ${fmt(data.training.episodes)} training episodes. PPO-Clip, two 23→48→48→5 tanh policies; training seed ${data.training.seed}. Offline CPU training took ${Math.round(data.training.seconds)} seconds. Training arenas use seeds below 1,000,000; held-out scores use separate seeds.`;
    for (const n of ['play', 'step', 'reset', 'export-model', 'export-scene']) button(n).disabled = false; field('model').disabled = false; root.dataset.ready = 'true'; update(); say('Two pretrained agents are ready. Press Play to begin.');
  }).catch(e => { if (!disposed && e.name !== 'AbortError') { say(`Unable to start: ${e.message}`, true); $('.hs-loading')?.remove(); } });
  update(); refreshBlocks();
  return { dispose() { if (disposed) return; disposed = true; epoch++; playing = false; stopFrame(); abort.abort(); intersection.disconnect(); document.removeEventListener('visibilitychange', reconcileVisibility); root.removeEventListener('click', click); root.removeEventListener('change', change); view?.dispose(); root.remove(); } };
}
