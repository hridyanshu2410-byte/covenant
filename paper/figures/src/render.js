// Render figure HTML to vector PDF (no caption, for LaTeX) and PNG previews (with caption).
// Usage: NODE_PATH=/opt/node-tools/node_modules node render.js fig1-teaser [more...]
// Fonts: Libertinus Serif/Sans and Inconsolata must be installed (see README.md).
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
  const out = path.resolve(__dirname, '..');
  for (const name of process.argv.slice(2)) {
    const page = await browser.newPage({ deviceScaleFactor: 4, viewport: { width: 672, height: 1200 } });
    await page.goto('file://' + path.join(__dirname, name + '.html'));
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(300);
    // preview PNG with caption
    const fig = await page.$('#fig');
    await fig.screenshot({ path: path.join(out, name + '-preview.png') });
    // paper version: no caption
    await page.evaluate(() => { const c = document.getElementById('cap'); if (c) c.style.display = 'none'; });
    const box = await fig.boundingBox();
    await fig.screenshot({ path: path.join(out, name + '.png') });
    await page.pdf({ path: path.join(out, name + '.pdf'), width: '672px', height: Math.ceil(box.height + 1) + 'px',
                     printBackground: true, margin: { top: 0, right: 0, bottom: 0, left: 0 }, pageRanges: '1' });
    console.log('rendered', name, Math.round(box.width) + 'x' + Math.round(box.height) + ' px (' + (box.width / 96).toFixed(2) + 'in wide)');
    await page.close();
  }
  await browser.close();
})();
