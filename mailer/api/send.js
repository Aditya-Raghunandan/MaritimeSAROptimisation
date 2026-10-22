/**
 * api/send.js -- the mailer as a Vercel function (the public one, for players on their own
 * phones). Vercel parses the JSON body; the SMTP account comes from the project's environment
 * variables, set in Vercel's dashboard, never in this repository.
 */

import { corsHeaders, handleSend, Limits, transportFrom } from '../handler.js';

const transport = transportFrom(process.env);
const limits = new Limits();      // per warm instance: a speed bump, not a wall

export default async function handler(req, res) {
  for (const [k, v] of Object.entries(corsHeaders(req.headers.origin))) res.setHeader(k, v);
  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method === 'GET') return res.status(200).json({ ok: true, smtp: Boolean(transport) });
  if (req.method !== 'POST') return res.status(405).json({ ok: false });
  const ip = String(req.headers['x-forwarded-for'] ?? '').split(',')[0].trim() || '?';
  const out = await handleSend(req.body, { send: transport && ((m) => transport.sendMail(m)), limits, ip });
  return res.status(out.status).json(out.json);
}
