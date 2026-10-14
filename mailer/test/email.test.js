import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildResultEmail, escapeHtml, standingWords, verdictOf } from '../email.js';
import { corsHeaders, handleSend, Limits, validate } from '../handler.js';

const PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

export const sample = (over = {}) => ({
  email: 'player@example.com',
  name: 'Ada',
  points: 912,
  areaKm2: 34.9,
  you: { pos: 0.52, found: true, foundS: 1390 },
  cg: { pos: 0.49, found: false, missedKm: 1.2 },
  ai: { pos: 0.44, found: false, missedKm: 0.6, name: 'Telemachus' },
  bench: [{ label: 'Parallel Track', pos: 0.493 }, { label: 'Telemachus (ML model)', pos: 0.438 }],
  image: PNG,
  ...over,
});

test('the verdict names who the player beat, all four ways', () => {
  const r = (beatAi, beatCg) => verdictOf({ beatAi, beatCg, ai: { name: 'Telemachus' } });
  assert.match(r(true, true).subject, /beat Telemachus AND the Coast Guard/);
  assert.match(r(true, false).head, /You beat Telemachus\./);
  assert.match(r(false, true).head, /Telemachus beat you/);
  assert.match(r(false, false).subject, /Telemachus won this round/);
});

test('a player who found them is cheered, and told who searched more', () => {
  const r = (beatAi, beatCg) => verdictOf({ beatAi, beatCg, ai: { name: 'the greedy search' }, you: { found: true } });
  assert.match(r(false, false).subject, /You found them!/);
  assert.match(r(false, false).line, /^You found the person in the water\. The Coast Guard’s pattern and the greedy search still searched more/);
  assert.match(r(false, true).line, /The greedy search still searched more/);
  assert.match(r(true, true).subject, /beat the greedy search AND the Coast Guard/);   // a win stays a win
});

test('validate: beat flags come from the numbers, not from the client', () => {
  const r = validate(sample({ beatAi: false }));
  assert.equal(r.beatAi, true);
  assert.equal(r.beatCg, true);
  assert.equal(r.ai.name, 'Telemachus');
  assert.equal(validate(sample({ ai: { pos: 0.6, name: 'Skynet' } })).ai.name, 'the greedy search');
});

test('validate refuses a bad address, a share over 1, and a non-image', () => {
  assert.throws(() => validate(sample({ email: 'not an email' })), /email/);
  assert.throws(() => validate(sample({ you: { pos: 1.4 } })), /you.pos/);
  assert.throws(() => validate(sample({ image: 'data:text/html;base64,PGI+' })), /image/);
});

test('the name is escaped and cut to 16 characters', () => {
  const r = validate(sample({ name: '<script>alert(1)</script>' }));
  assert.equal(r.name.length, 16);
  const { html } = buildResultEmail(r, { gameUrl: 'https://x.example/game.html' });
  assert.ok(!html.includes('<script>'));
  assert.ok(html.includes(escapeHtml(r.name)));
});

test('the email says the numbers, the inline picture and the vote link only when set', () => {
  const r = validate(sample());
  const without = buildResultEmail(r, { gameUrl: 'https://x.example/game.html' });
  assert.ok(without.html.includes('52.0 %') && without.html.includes('44.0 %'));
  assert.ok(without.html.includes('cid:flight'));
  assert.ok(without.html.includes('found them 23:10 into the search'));
  assert.ok(without.html.includes('missed them by 1.2 km'));
  assert.ok(without.html.includes('4,888 football pitches'));
  assert.ok(!without.html.includes('Vote for us'));
  const withVote = buildResultEmail(r, { gameUrl: 'https://x.example/game.html', voteUrl: 'https://vote.example/' });
  assert.ok(withVote.html.includes('Vote for us') && withVote.text.includes('https://vote.example/'));
});

test('handleSend sends one message with the picture attached inline, and stores nothing', async () => {
  const sent = [];
  const out = await handleSend(sample(), {
    env: { SMTP_USER: 'us@example.com', GAME_URL: 'https://g.example' }, send: async (m) => sent.push(m),
  });
  assert.equal(out.status, 200);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].to, 'player@example.com');
  assert.equal(sent[0].attachments[0].cid, 'flight');
  assert.ok(Buffer.isBuffer(sent[0].attachments[0].content));
});

test('handleSend: 400 for bad input, 503 with no SMTP, 502 when the server refuses', async () => {
  assert.equal((await handleSend({ email: 'x' }, { send: async () => {} })).status, 400);
  assert.equal((await handleSend(sample(), {})).status, 503);
  const quiet = console.error;
  console.error = () => {};
  const refused = await handleSend(sample(), { send: async () => { throw new Error('535 auth'); } });
  console.error = quiet;
  assert.equal(refused.status, 502);
});

test('limits: five a ten minutes per sender, and a ceiling for the day', () => {
  const l = new Limits({ perIp: 5, perDay: 7 });
  const t = Date.UTC(2026, 9, 15, 12);
  for (let i = 0; i < 5; i += 1) assert.ok(l.allow('a', t + i));
  assert.equal(l.allow('a', t + 6), false);
  assert.ok(l.allow('a', t + 11 * 60_000));
  assert.ok(l.allow('b', t + 7));
  assert.equal(l.allow('c', t + 8), false);          // 7 sent today
});

test('limits: one laptop at a busy stand, a player every 30 s for an hour, is never turned away', () => {
  const l = new Limits();
  const t = Date.UTC(2026, 9, 15, 9);
  for (let i = 0; i < 120; i += 1) assert.ok(l.allow('stand', t + i * 30_000), `player ${i + 1}`);
});

test('CORS lets the game site in and nobody else by name', () => {
  assert.equal(corsHeaders('https://aditya-raghunandan.github.io', {})['Access-Control-Allow-Origin'],
    'https://aditya-raghunandan.github.io');
  assert.equal(corsHeaders('https://evil.example', {})['Access-Control-Allow-Origin'],
    'https://aditya-raghunandan.github.io');
});

test('the email says where the player stands on the board, and the arrival', () => {
  const board = (rank, of, lines) => ({ rank, of, mode: 'map', lines });
  const line = (rank, name, score, you = false) => ({ rank, name, score, you });
  const top = validate(sample({ arrival: '30 min', board: board(1, 4, [line(1, 'Ada', 1204, true), line(2, 'Bo', 990)]) }));
  const first = buildResultEmail(top, { gameUrl: 'https://x.example/game.html' });
  assert.match(first.subject, /top of the board!$/);
  assert.ok(first.html.includes('You are top of the board right now, ahead of 3 other players.'));
  assert.ok(first.html.includes('Ada (you)') && first.html.includes('1,204'));
  assert.ok(first.html.includes('helicopter arriving 30 min after the call'));
  assert.ok(first.text.includes('1. Ada (you)  1204'));

  const fourth = validate(sample({
    board: board(7, 9, [line(1, '<b>Zed</b>', 1500), line(2, 'Bo', 1400), line(7, 'Ada', 1100, true)]),
  }));
  const behind = buildResultEmail(fourth, {});
  assert.ok(behind.html.includes('You are 7th of 9 on the board right now, 400 points behind &lt;b&gt;Zed&lt;/b&gt;.'));
  assert.ok(!behind.html.includes('<b>Zed</b>'));
  assert.ok(behind.html.includes('…'));
  assert.doesNotMatch(behind.subject, /top of the board/);
  assert.equal(standingWords(validate(sample({ board: board(1, 1, [line(1, 'Ada', 900, true)]) })).board),
    'You are the first name on the board.');
});

test('validate drops a bad board or arrival instead of refusing the email', () => {
  const r = validate(sample({ arrival: '<script>', board: { rank: 5, of: 3, lines: [] } }));
  assert.equal(r.arrival, null);
  assert.equal(r.board, null);
  const lines = validate(sample({ board: { rank: 1, of: 2, mode: 'x', lines: [{ rank: 1, name: 'A', score: 'lots' }, { rank: 2, name: 'B', score: 5 }] } })).board;
  assert.equal(lines.mode, 'map');
  assert.deepEqual(lines.lines, [{ rank: 2, name: 'B', score: 5, you: false }]);
  assert.ok(!buildResultEmail(validate(sample()), {}).html.includes('The leaderboard'));
});
