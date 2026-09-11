import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, extname } from 'node:path';
import { chromium } from '@playwright/test';

// Kept identical to Portfolio's published header. Test built files, with no Vite
// dev-server allowances, external CDN, or JavaScript unsafe-eval permission.
const CSP = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; media-src 'self' blob:; connect-src 'self'; worker-src 'self' blob:; frame-src 'self'; frame-ancestors 'self'; object-src 'none'; base-uri 'self'; form-action 'self'";
const root = fileURLToPath(new URL('../', import.meta.url));
const type = { '.js':'text/javascript', '.mjs':'text/javascript', '.css':'text/css', '.html':'text/html', '.json':'application/json', '.wasm':'application/wasm', '.glb':'model/gltf-binary', '.png':'image/png' };
const worker = `import load from '/__csp/glue.js';
  import { PhysicsSimulation, generateArena } from '/__csp/physics.js';
  try {
    const mj = await load({locateFile:()=>'/__csp/mujoco.wasm'});
    const sim = new PhysicsSimulation(mj,generateArena(77,'open',8,1,1));
    for(let i=0;i<20;i++) sim.step([[.7,.4,.2,1,0],[0,0,0,0,0]]);
    const result={t:sim.t,observations:sim.observe(0).length,version:mj.mj_version()};
    sim.dispose(); postMessage(result);
  } catch(error) { postMessage({error:error.message}); }`;
const fixture = new Map([
  ['/__csp/glue.js','src/vendor/mujoco-csp.js'],
  ['/__csp/physics.js','src/core/physics.js'],
  ['/__csp/mujoco.wasm','node_modules/@mujoco/mujoco/mujoco.wasm'],
]);
const server = createServer(async(req,res)=>{
  const path = decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  res.setHeader('Content-Security-Policy',CSP);
  res.setHeader('Content-Type',type[extname(path)]||'application/octet-stream');
  try {
    if(path==='/__csp/worker.mjs') return res.end(worker);
    const file = fixture.has(path) ? resolve(root,fixture.get(path)) : resolve(root,'dist',path==='/'?'index.html':'.'+path);
    if(!file.startsWith(root)) throw new Error('Invalid path');
    if(path==='/') res.setHeader('Content-Type','text/html');
    res.end(await readFile(file));
  } catch {res.statusCode=404;res.end('Not found');}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const url=`http://127.0.0.1:${server.address().port}`;
const browser=await chromium.launch({headless:true,channel:'chromium'});
const page=await browser.newPage({viewport:{width:1280,height:1000}});
const errors=[],violations=[],requests=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('requestfailed',r=>errors.push(`${r.url()}: ${r.failure()?.errorText}`));
page.on('request',r=>requests.push(r.url()));
await page.exposeFunction('recordCsp',e=>violations.push(e));
await page.addInitScript(()=>document.addEventListener('securitypolicyviolation',e=>window.recordCsp({directive:e.effectiveDirective,blocked:e.blockedURI})));
try {
  await page.goto(url);
  await page.waitForFunction(()=>document.querySelector('.hide-seek')?.dataset.ready==='true',null,{timeout:20000});
  assert.equal(await page.locator('.hide-seek').getAttribute('data-policy-format'),'original-mujoco-persistent-buttons-ppo-v1');
  assert.equal(await page.locator('.hs-ready-badge').textContent(),'DEVELOPMENT');
  await page.locator('[data-field="speed"]').selectOption('4');
  await page.locator('[data-action="play"]').click();
  await page.waitForFunction(()=>document.querySelector('.hide-seek').dataset.phase==='finished',null,{timeout:20000});
  assert.equal(await page.locator('.hide-seek').getAttribute('data-steps'),'240');
  await page.locator('[data-field="model"]').selectOption('initial');
  await page.waitForFunction(()=>document.querySelector('.hide-seek').dataset.model==='initial'&&document.querySelector('.hide-seek').dataset.ready==='true');
  assert.equal(await page.locator('.hide-seek').getAttribute('data-policy-format'),'original-mujoco-persistent-buttons-ppo-v1');
  assert.equal(await page.locator('.hs-ready-badge').textContent(),'UNTRAINED');
  await page.locator('[data-action="step"]').click();
  assert.equal(await page.locator('.hide-seek').getAttribute('data-steps'),'1');
  const result=await page.evaluate(()=>new Promise((resolve,reject)=>{
    const w=new Worker('/__csp/worker.mjs',{type:'module'});
    const timeout=setTimeout(()=>{w.terminate();reject(Error('MuJoCo worker timed out'));},15000);
    w.onmessage=e=>{clearTimeout(timeout);w.terminate();resolve(e.data);};
    w.onerror=e=>{clearTimeout(timeout);w.terminate();reject(Error(e.message));};
  }));
  assert.equal(result.t,20,JSON.stringify(result)); assert.equal(result.observations,138);
  assert.equal(result.version,3013000);
  assert.ok(requests.some(r=>r.endsWith('.wasm')));
  assert.ok(requests.some(r=>r.includes('/models/physical-policy-')));
  assert.equal(requests.filter(r=>/\.glb(?:\?|$)/.test(r)).length,0,'shared code figure needs no GLB');
  assert.ok(requests.every(r=>r.startsWith(url)),'all model, module, character, and WASM requests must remain same origin');
  assert.deepEqual(errors,[]);assert.deepEqual(violations,[]);
  console.log(JSON.stringify({csp:CSP,completedSteps:240,checkpoints:2,worker:result,violations,requests:requests.length}));
} finally {await browser.close();await new Promise(r=>server.close(r));}
