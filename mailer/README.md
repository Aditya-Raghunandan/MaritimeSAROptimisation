# The result mailer

After a game of *Survive the Triangle* a player can ask for their result by email: whether
they beat Telemachus and the Coast Guard, a picture of the path they flew over the cloud
(drawn by the game, `frontend/src/resultCard.js`), the numbers on the same sea, and a link
to vote for us. This folder is the small server that sends it.

It sends through Gmail's SMTP server with the **same variables and the same account as
KnockKnock.prof** (ELEN4010, `src/services/email-service.js`): an App Password, never the
account password. Nothing is stored. The address is used for one message and is never
logged.

## Environment

| Variable | What |
|---|---|
| `SMTP_HOST`, `SMTP_PORT` | `smtp.gmail.com`, `587` |
| `SMTP_USER`, `SMTP_PASS` | the Gmail address and its 16-character App Password |
| `EMAIL_FROM` | e.g. `"Survive the Triangle" <that address>` |
| `GAME_URL` | the game, for the "Play again" button |
| `VOTE_URL` | the voting link, once we have it; the button only appears when it is set |
| `ALLOWED_ORIGINS` | sites allowed to call it, comma separated (default: the GitHub Pages site and localhost) |

Locally these go in `mailer/.env` (git ignores it; `.env.example` lists the names). On a
host they go in its dashboard. **They never go in the repository, the vault, a chat or a
command line.**

## Where it runs

- **The laptop at the stand:** `npm install && npm start` serves `POST http://localhost:8787/send`.
  Build the kiosk's game with `VITE_MAILER_URL=http://localhost:8787/send`. It needs a
  network that lets SMTP out (port 587); if the campus Wi-Fi blocks it, a phone hotspot will not.
- **Public, for players on their phones:** a Vercel project with this folder as its root
  directory. `api/send.js` becomes `https://<project>.vercel.app/api/send`; set the variables
  above in the project's settings, then set the repository variable `MAILER_URL` to that
  address so the deploy builds it into the game (`VITE_MAILER_URL`).
- **Not Render's free tier:** since 26 Sep 2025 it blocks outbound SMTP (ports 25, 465 and
  587) on free web services, so the mail would time out.

## Tests

`npm test` (node's own runner, no network). `npm run preview -- <dir> [picture]` writes the
four versions of the email (beat both, beat Telemachus only, beat the Coast Guard only, beat
neither) as HTML files to look at.
