import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
const url = 'http://127.0.0.1:5181';
export async function openBrowser() {
  let server = null;
  if (!await fetch(url).then(r => r.ok).catch(() => false)) {
    server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5181'], { cwd: new URL('..', import.meta.url), stdio: 'ignore' });
    for (let n = 0; n < 100; n++) { if (await fetch(url).then(r => r.ok).catch(() => false)) break; await new Promise(r => setTimeout(r, 100)); }
  }
  const browser = await chromium.launch({ channel: 'chromium', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1200 }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
  await page.goto(url); await page.locator('.hide-seek[data-ready="true"]').waitFor();
  return { page, browser, url, close: async () => { await browser.close(); server?.kill(); } };
}
