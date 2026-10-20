// Print the smallest text in each figure as it prints in the paper (pt), with examples.
// Usage: NODE_PATH=/opt/node-tools/node_modules node fontaudit.js <printed width pt> fig1-teaser [more...]
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
  const printed = parseFloat(process.argv[2]);
  for (const name of process.argv.slice(3)) {
    const page = await browser.newPage({ viewport: { width: 672, height: 1200 } });
    await page.goto('file://' + path.join(__dirname, name + '.html'));
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(() => { const c = document.getElementById('cap'); if (c) c.style.display = 'none'; });
    const res = await page.evaluate(() => {
      const fig = document.getElementById('fig'), W = fig.getBoundingClientRect().width, out = [];
      const walker = document.createTreeWalker(fig, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const t = walker.currentNode, el = t.parentElement, s = t.textContent.trim();
        if (!s || getComputedStyle(el).display === 'none') continue;
        let px = parseFloat(getComputedStyle(el).fontSize);
        const svg = el.closest('svg');
        if (svg && el.getScreenCTM) { const m = el.getScreenCTM(); px *= Math.hypot(m.a, m.b); }
        out.push([px, s.slice(0, 50)]);
      }
      return { W, out };
    });
    const k = printed / res.W;
    const sizes = res.out.map(([px, s]) => [px * k, s]).sort((a, b) => a[0] - b[0]);
    const below = sizes.filter(x => x[0] < 6.0);
    console.log(`${name}: min ${sizes[0][0].toFixed(2)}pt, ${below.length} of ${sizes.length} runs below 6pt`);
    const seen = new Set();
    for (const [pt, s] of below) { const key = pt.toFixed(1); if (seen.size < 8 && !seen.has(key)) { seen.add(key); console.log(`  ${pt.toFixed(2)}pt  ${s}`); } }
  }
  await browser.close();
})();
