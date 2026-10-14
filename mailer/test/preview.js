/**
 * preview.js -- writes the four versions of the email (beat both, beat the ML model only, beat the
 * Coast Guard only, beat neither) as HTML files to look at in a browser, with an optional
 * picture and a board: node test/preview.js [out-dir] [image.png|jpg]
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { buildResultEmail } from '../email.js';
import { validate } from '../handler.js';

const out = process.argv[2] ?? 'preview';
const imagePath = process.argv[3];
mkdirSync(out, { recursive: true });
const img = imagePath ? `data:image/${imagePath.endsWith('png') ? 'png' : 'jpeg'};base64,${readFileSync(imagePath).toString('base64')}` : null;
const cases = {
  both: { you: 0.58, cg: 0.49, ai: 0.44 },
  ai_only: { you: 0.47, cg: 0.49, ai: 0.44 },
  cg_only: { you: 0.46, cg: 0.42, ai: 0.47 },
  neither: { you: 0.21, cg: 0.49, ai: 0.44 },
};
for (const [name, p] of Object.entries(cases)) {
  const r = validate({
    email: 'you@example.com', name: 'Ada', points: Math.round(1000 * p.you / 0.49), areaKm2: 34.9,
    you: { pos: p.you, found: p.you > 0.4, foundS: 1390, missedKm: 2.3 },
    cg: { pos: p.cg, found: false, missedKm: 1.2 },
    ai: { pos: p.ai, found: true, foundS: 605, name: 'Telemachus' },
    bench: [{ label: 'Parallel Track', pos: 0.493 }, { label: 'Expanding Square', pos: 0.486 },
      { label: 'Telemachus (ML model)', pos: 0.438 }],
    arrival: '30 min',
    // top of the board when they beat both, further down otherwise
    board: name === 'both'
      ? { rank: 1, of: 14, mode: 'map', lines: [[1, 'Ada', 1184], [2, 'Thabo', 1102], [3, 'Lerato', 1040], [4, 'Sam', 998], [5, 'Kim', 951]].map(([rank, n, score]) => ({ rank, name: n, score })) }
      : { rank: 9, of: 14, mode: 'map', lines: [[1, 'Thabo', 1102], [2, 'Lerato', 1040], [3, 'Sam', 998], [4, 'Kim', 951], [5, 'Jo', 930], [9, 'Ada', Math.round(1000 * p.you / 0.49)]].map(([rank, n, score]) => ({ rank, name: n, score })) },
  });
  const { subject, html } = buildResultEmail(r, {
    gameUrl: 'https://aditya-raghunandan.github.io/MaritimeSAROptimisation/game.html',
    voteUrl: 'https://example.com/vote',
  });
  const page = img ? html.replace('cid:flight', img) : html.replace(/<img[^>]*cid:flight[^>]*>/, '');
  writeFileSync(join(out, `${name}.html`), `<!-- ${subject} -->\n${page}`);
  console.log(name, '|', subject);
}
