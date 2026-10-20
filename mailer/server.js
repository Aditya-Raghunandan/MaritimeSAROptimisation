/**
 * server.js -- the mailer as a plain Node server: for the laptop at the stand, or any Node
 * host that allows outbound SMTP. Reads mailer/.env if it is there (never committed).
 *
 *   node server.js            POST /send  {the game's result}  -> the email
 *                             GET  /health                      -> {ok, smtp}
 * PORT defaults to 8787.
 */

import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { corsHeaders, handleSend, Limits, transportFrom } from './handler.js';

try {
  for (const line of readFileSync(new URL('./.env', import.meta.url), 'utf8').split(/\r?\n/)) {
    const m = /^\s*([A-Z_]+)\s*=\s*(.*)\s*$/.exec(line);
    if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^"(.*)"$/, '$1');
  }
} catch { /* no .env: the environment is all there is */ }

const transport = transportFrom(process.env);
const limits = new Limits();
const MAX_BODY = 1_200_000;

const server = createServer((req, res) => {
  const headers = { ...corsHeaders(req.headers.origin), 'Content-Type': 'application/json' };
  const reply = (status, json) => { res.writeHead(status, headers); res.end(JSON.stringify(json)); };
  if (req.method === 'OPTIONS') return reply(204, {});
  if (req.method === 'GET' && (req.url === '/health' || req.url === '/send')) return reply(200, { ok: true, smtp: Boolean(transport) });
  if (req.method !== 'POST' || req.url !== '/send') return reply(404, { ok: false });
  let size = 0;
  const chunks = [];
  req.on('data', (c) => {
    size += c.length;
    if (size > MAX_BODY) { reply(413, { ok: false, error: 'too big' }); req.destroy(); } else chunks.push(c);
  });
  req.on('end', async () => {
    if (size > MAX_BODY) return;
    let body;
    try { body = JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { return reply(400, { ok: false, error: 'not JSON' }); }
    const out = await handleSend(body, {
      send: transport && ((m) => transport.sendMail(m)), limits, ip: req.socket.remoteAddress,
    });
    reply(out.status, out.json);
  });
  return undefined;
});

const port = Number(process.env.PORT) || 8787;
server.listen(port, () => console.log(`mailer on http://localhost:${port} (smtp ${transport ? 'set' : 'NOT set'})`));
