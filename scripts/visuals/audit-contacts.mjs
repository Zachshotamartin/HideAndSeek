import { chromium } from "@playwright/test";
import load from "@mujoco/mujoco";
import { PhysicsSimulation, generateArena } from "../../src/core/physics.js";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
const repo = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const output = process.argv[2] || path.join(repo, "output/visual-audit");
await mkdir(output, { recursive: true });
const fixture = JSON.parse(
    await readFile(
      path.join(repo, "training/fixtures/native-physics.json"),
      "utf8",
    ),
  ),
  mj = await load();
const serialize = (s) => ({
  time: s.time,
  t: s.t,
  phase: s.t < s.prep ? "preparation" : "play",
  prep: s.prep,
  dt: s.dt,
  agents: structuredClone(s.agents),
  objects: structuredClone(s.objects),
  visions: [s.vision(0), s.vision(1)],
});
const sim = new PhysicsSimulation(mj, fixture.arena, {
    prep: fixture.prep,
    play: fixture.play,
  }),
  arena = structuredClone(sim.viewArena),
  frames = [];
for (const actions of fixture.actions) {
  sim.step(actions);
  frames.push(serialize(sim));
}
sim.dispose();
const rampArena = generateArena(4, "open", 8, 0, 0);
rampArena.agents = [
  { position: [2.6, 4, 0.25], yaw: 0 },
  { position: [6.5, 6.5, 0.25], yaw: 0 },
];
rampArena.objects = [
  {
    id: "ramp-a",
    kind: "ramp",
    position: [4, 4, 0.353],
    size: [1.7, 1.2, 0.7],
    yaw: 0,
    mass: 1.2,
  },
];
const ramp = new PhysicsSimulation(mj, rampArena, {
    prep: 0,
    play: 100,
    immovable: true,
  }),
  rampView = structuredClone(ramp.viewArena),
  rampFrames = [];
for (let i = 0; i < 80; i++) {
  ramp.step([
    [i < 50 ? 1 : 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ]);
  rampFrames.push(serialize(ramp));
}
ramp.dispose();
const server = spawn(
  process.execPath,
  ["node_modules/vite/bin/vite.js", "--host", "127.0.0.1", "--port", "5191"],
  { cwd: repo, stdio: "ignore" },
);
await new Promise((r) => setTimeout(r, 900));
const browser = await chromium.launch({ channel: "chromium" }),
  errors = [];
try {
  const page = await browser.newPage({
    viewport: { width: 1300, height: 900 },
    deviceScaleFactor: 1.5,
  });
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (e) => {
    if (e.type() === "error") errors.push(e.text());
  });
  await page.route("**/visual-audit", (r) =>
    r.fulfill({
      contentType: "text/html",
      body: '<style>body{margin:0;background:#142020}#view{width:100vw;height:100vh}canvas{width:100%;height:100%;display:block}</style><div id="view"></div>',
    }),
  );
  await page.goto("http://127.0.0.1:5191/visual-audit");
  const report = await page.evaluate(
    async ({ arena, frames, rampView, rampFrames }) => {
      const { createView } = await import("/src/renderer.js");
      window.view = createView(document.querySelector("#view"));
      view.setArena(arena);
      await view.ready;
      let maxHandDistance = 0, maxRenderedHandDistance = 0,
        gripFrames = 0,
        gripIndex = 0;
      const snapshots = [];
      for (let i = 0; i < frames.length; i++) {
        const frame = frames[i];
        frame.vision = (role) => frame.visions[role];
        view.update(frame);
        const d = view.diagnostics();
        for (let a = 0; a < 2; a++)
          if (frame.agents[a].gripId !== null) {
            gripFrames++;
            gripIndex = i;
            const target = frame.agents[a].handTarget;
            for(const name of ['handL','handR'])maxRenderedHandDistance=Math.max(maxRenderedHandDistance,Math.hypot(...target.map((v,j)=>v-d.agents[a].landmarks[name][j])));
            for (const hand of ["hand0", "hand1"])
              maxHandDistance = Math.max(
                maxHandDistance,
                Math.hypot(...target.map((v, j) => v - d.agents[a][hand][j])),
              );
          }
        view.remember(frame);
      }
      const f = frames[gripIndex];
      f.vision = (role) => f.visions[role];
      view.update(f);
      view.setFollow("0");
      snapshots.push({
        name: "hide-seek-contact-grip-audit",
        image: view.capture(),
      });
      view.setVision("hider");
      snapshots.push({
        name: "hide-seek-contact-vision-audit",
        image: view.capture(),
      });
      view.setVision("none");
      view.setArena(rampView);
      await view.ready;
      let maxHeight = 0,
        airborne = 0,
        regrounded = false;
      for (const frame of rampFrames) {
        frame.vision = (role) => frame.visions[role];
        view.update(frame);
        view.remember(frame);
        maxHeight = Math.max(maxHeight, frame.agents[0].position[1]);
        if (!frame.agents[0].grounded) airborne++;
        if (airborne && frame.agents[0].grounded) regrounded = true;
      }
      // The second agent stays still throughout this physical fixture. Measure
      // the rendered joint matrices, not the IK inputs, to catch a crouched idle.
      const idle = view.diagnostics().agents[1].landmarks;
      const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
      const diff=(a,b)=>a.map((x,i)=>x-b[i]);
      const kneeFlexDegrees=['L','R'].map(side=>{
        const upper=diff(idle['knee'+side],idle['hip'+side]);
        const lower=diff(idle['ankle'+side],idle['knee'+side]);
        return Math.acos(Math.max(-1,Math.min(1,dot(upper,lower)/Math.hypot(...upper)/Math.hypot(...lower))))*180/Math.PI;
      });
      if(Math.max(...kneeFlexDegrees)>12)throw Error('Idle knees permanently crouched: '+kneeFlexDegrees);
      if (!gripFrames) throw Error("Fixture did not exercise gripping");
      if(maxRenderedHandDistance>.033)throw Error('Rendered stick endpoint too far from physical grip anchor: '+maxRenderedHandDistance);
      if (maxHandDistance > 0.033)
        throw Error(
          "Hand endpoints too far from actual grip anchor: " + maxHandDistance,
        );
      if (!airborne || !regrounded)
        throw Error(
          "Ramp fixture did not exercise airborne-to-supported gait state",
        );
      return {
        snapshots,
        maxHandDistance,
        maxRenderedHandDistance,
        idleKneeFlexDegrees:kneeFlexDegrees,
        gripFrames,
        maxHeight,
        airborne,
        regrounded,
        diagnostics: view.diagnostics(),
      };
    },
    { arena, frames, rampView, rampFrames },
  );
  for (const s of report.snapshots)
    await writeFile(
      path.join(output, s.name + ".png"),
      Buffer.from(s.image.split(",")[1], "base64"),
    );
  delete report.snapshots;
  await page.setViewportSize({ width: 390, height: 580 });
  await page.evaluate(() => view.setFollow("0"));
  await page.waitForTimeout(100);
  await writeFile(
    path.join(output, "hide-seek-connected-mobile.png"),
    Buffer.from(
      (await page.evaluate(() => view.capture())).split(",")[1],
      "base64",
    ),
  );
  await page.evaluate(() => view.dispose());
  if (errors.length) throw Error(errors.join("\n"));
  report.browserErrors = errors;
  report.notes =
    "Scripted actions in the actual MuJoCo engine exercise physical grip/ramp contacts. This is a renderer regression, not evidence of learned policy performance.";
  await writeFile(
    path.join(output, "hide-seek-visual-audit.json"),
    JSON.stringify(report, null, 2),
  );
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
  server.kill();
}
