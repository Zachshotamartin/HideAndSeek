import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {openBrowser} from './browser-helpers.mjs';
const app=await openBrowser();const {page}=app;const errors=[];
page.on('pageerror',e=>errors.push(e.message));
try {
  await page.locator('.hs-files summary').click();
  for(const id of ['trained','initial']){
    await page.locator('[data-field="model"]').selectOption(id);
    await page.waitForFunction(id=>document.querySelector('.hide-seek').dataset.model===id&&!document.querySelector('[data-action="play"]').disabled,id);
    await page.locator('[data-action="reset"]').click();
    await page.locator('[data-action="play"]').click();
    await page.waitForFunction(()=>Number(document.querySelector('.hide-seek').dataset.steps)>5);
    await page.locator('[data-action="play"]').click();
    const download=page.waitForEvent('download');await page.locator('[data-action="export-replay"]').click();
    const replay=JSON.parse(await fs.readFile(await (await download).path(),'utf8'));
    assert.equal(replay.policyFormat,'original-mujoco-relational-jump-policy-pair-v4');
    assert(replay.actions.length>5);
    assert(replay.actions.every(pair=>pair.length===2&&pair.every(a=>a.length===6&&a.every(Number.isFinite))));
  }
  assert.deepEqual(errors,[]);
  console.log('Browser: both v4 checkpoints load, play, pause, reset and export finite six-action replays. No page errors.');
} finally {await app.close();}
