/*
  html_to_pdf.mjs -- print a local HTML document to PDF.

  Exists because Playwright in this repository is the NODE install (added for the
  browser tests in #21), not a Python one, and adding a second copy for the sake of
  one `page.pdf()` would mean two browser downloads and two versions to keep in step.
  So the Python generators shell out to this.

  It lives in `frontend/tools/` rather than in `scripts/` for a dull but absolute
  reason: Node resolves an ESM import from the SCRIPT's directory, not the working
  directory, so a copy in `scripts/` cannot see `frontend/node_modules` however it is
  invoked. It sits beside `shoot.mjs`, which depends on the same install.

  Waits for `networkidle` because the documents pull IBM Plex from Google Fonts, and a
  PDF printed before the webfont lands is set in a fallback serif at different metrics --
  which reflows every figure caption and is invisible until someone opens the file.

  Run:
      node tools/html-to-pdf.mjs <input.html> <output.pdf>
*/

import { chromium } from '@playwright/test';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error('usage: node tools/html-to-pdf.mjs <input.html> <output.pdf>');
  process.exit(2);
}

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(pathToFileURL(resolve(input)).href, { waitUntil: 'networkidle' });

// Belt and braces on the webfont: `networkidle` covers the request, this covers the
// swap. Without it the first page occasionally prints mid-swap.
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(300);

await page.pdf({
  path: resolve(output),
  format: 'A4',
  printBackground: true,
  margin: { top: '16mm', bottom: '16mm', left: '14mm', right: '14mm' },
});

await browser.close();
console.log(`wrote ${output}`);
