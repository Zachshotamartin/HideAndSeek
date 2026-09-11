import fs from 'node:fs/promises';
import { openBrowser } from './browser-helpers.mjs';
const app=await openBrowser(); const p=await app.browser.newPage({viewport:{width:1600,height:1300},deviceScaleFactor:2});
try {
 await p.goto(app.url);await p.locator('[data-ready="true"]').waitFor();
 async function capture(seed,steps,vision,file) {
   await p.locator('[data-field="seed"]').fill(String(seed));await p.locator('[data-action="generate"]').click();
   for(let i=0;i<steps;i++)await p.locator('[data-action="step"]').click();
   await p.locator('[data-field="vision"]').selectOption(vision);
   await p.locator('.hs-files summary').click();
   const [download]=await Promise.all([p.waitForEvent('download'),p.locator('[data-action="export-scene"]').click()]);
   await download.saveAs(`examples/${file}`);
   await p.locator('.hs-files summary').click();
 }
 await fs.mkdir('examples',{recursive:true});await capture(2709,100,'none','hide-and-seek-arena.png');await capture(98231,150,'seeker','hide-and-seek-vision.png');
 console.log('Captured two real transparent WebGL scene exports, without interface edges.');
} finally {await app.close();}
