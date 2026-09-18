/** Actual MuJoCo states for shared stick-figure art review, not a policy benchmark. */
import { chromium } from '@playwright/test';
import load from '@mujoco/mujoco';
import { PhysicsSimulation, generateArena } from '../../src/core/physics.js';
import { mkdir, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const output=process.argv[2]||path.join(repo,'output/visual-audit');
await mkdir(output,{recursive:true});
const arena=generateArena(11,'shelter',8,3,1);
arena.agents=[{position:[1.4,1.6,.25],yaw:0},{position:[6.4,6.1,.25],yaw:Math.PI}];
const mj=await load(),sim=new PhysicsSimulation(mj,arena,{prep:0,play:120});
const frames=[];
for(let i=0;i<12;i++){
  sim.step([[.4,0,0,0,0],[-.2,0,0,0,0]]);
  frames.push({time:sim.time,t:sim.t,phase:'play',prep:0,dt:sim.dt,agents:structuredClone(sim.agents),objects:structuredClone(sim.objects)});
}
const idleFrames=[];
for(let i=0;i<40;i++){sim.step([[0,0,0,0,0],[0,0,0,0,0]]);idleFrames.push({time:sim.time,t:sim.t,phase:'play',prep:0,dt:sim.dt,agents:structuredClone(sim.agents),objects:structuredClone(sim.objects)});}
const viewArena=structuredClone(sim.viewArena);sim.dispose();
const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','--host','127.0.0.1','--port','5196'],{cwd:repo,stdio:'ignore'});
await new Promise(r=>setTimeout(r,900));
const browser=await chromium.launch({channel:'chromium'}),errors=[];
try {
 const page=await browser.newPage({viewport:{width:1200,height:1000},deviceScaleFactor:1.5});
 page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/figure-review',r=>r.fulfill({contentType:'text/html',body:'<style>body{margin:0;background:#142020}#view{width:100vw;height:100vh}canvas{display:block;width:100%;height:100%}</style><div id="view"></div>'}));
 await page.goto('http://127.0.0.1:5196/figure-review');

  // An optional live portfolio URL supplies its actual computed CSS background.
  // Capture the canvas element after browser compositing, without raster edits.
  let captureBackground = '#142020';
  if (process.env.CAPTURE_PAGE_URL) {
    const reference = await browser.newPage();
    await reference.goto(process.env.CAPTURE_PAGE_URL);
    captureBackground = await reference.evaluate(() => getComputedStyle(document.body).background);
    await reference.close();
  }
  for(const [i,match] of [...captureBackground.matchAll(/url\(["']?([^"')]+)["']?\)/g)].entries()) {
    const url = match[1], response = await fetch(url);
    if(!response.ok)throw Error('Background texture fetch failed: '+response.status);
    const body=Buffer.from(await response.arrayBuffer());
    const localURL=new URL('/capture-background-'+i,new URL(page.url()).origin).href;
    await page.route(localURL,r=>r.fulfill({contentType:response.headers.get('content-type')||'image/webp',body}));
    captureBackground=captureBackground.replace(url,localURL);
  }
  await page.evaluate(async background => {
    document.body.style.background = background;
    const urls = [...background.matchAll(/url\(["']?([^"')]+)["']?\)/g)].map(m => m[1]);
    await Promise.all(urls.map(url => new Promise(resolve => {
      const image = new Image(); image.onload = image.onerror = resolve; image.src = url;
    })));
  }, captureBackground);
 await page.evaluate(async({arena,frames})=>{
   const {createView}=await import('/src/renderer.js');window.view=createView(document.querySelector('#view'));view.setArena(arena);await view.ready;
   for(const f of frames){view.update(f);view.remember(f);}
 },{arena:viewArena,frames});
 for(const mode of ['overview','character']){
   await page.evaluate(mode=>{
     if(mode==='character'){
       view.setFollow('0');view.orbit(5);view.zoom(.61);
       document.querySelector('canvas').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowDown'}));
       document.querySelector('canvas').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowDown'}));
     }else view.setFollow('none');
     view.capture();
   },mode);
   await page.locator('canvas').screenshot({path:path.join(output,`hide-seek-stick-${mode}.png`)});
 }
 await page.evaluate(frames=>{for(const f of frames){view.update(f);view.remember(f);}},idleFrames);
 await page.locator('canvas').screenshot({path:path.join(output,'hide-seek-stick-idle.png')});
 await page.setViewportSize({width:390,height:580});
 await page.waitForTimeout(100);
 await page.locator('canvas').screenshot({path:path.join(output,'hide-seek-stick-mobile.png')});
 await page.evaluate(()=>view.dispose());
 if(errors.length)throw Error(errors.join('\n'));
 console.log(JSON.stringify({captures:['hide-seek-stick-overview.png','hide-seek-stick-character.png','hide-seek-stick-idle.png','hide-seek-stick-mobile.png'],captureBackground,browserErrors:errors,notes:'Scripted movement in actual MuJoCo physics for art review; no learned-policy performance claim.'}));
}finally{await browser.close();server.kill();}
