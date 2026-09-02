# Manifest Browser Agent

A local, headless-browser agent that uses the [Manifest API](https://manifest.omfang.io)
as its perception layer. Given a goal and a start URL it loops:

1. **Perceive** — ask Manifest what actions exist on the current page (with a `requires` graph)
2. **Decide** — ask DeepSeek (`deepseek-v4-flash`, OpenAI-compatible API) which action advances the goal, using that graph
3. **Act** — execute the action via Playwright, observe the new URL, repeat

Output is a CLI run plus a trajectory log (JSON + a case-study Markdown writeup). Not a product.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

cp .env.example .env   # then fill in the two keys
```

`config.py` loads `.env` automatically; real environment variables override it.
`.env` is gitignored.

Auth is via a pre-captured logged-in session, not solved by the agent. Capture one once:

```bash
python -m playwright codegen --save-storage=storage_state.json https://your-site/login
# log in by hand, close the window
```

`storage_state.json` is gitignored.

## Run

```bash
python agent.py --goal "add the blue widget to my cart and check out" \
                --start-url "https://your-site/products/blue-widget" \
                --max-steps 15 \
                --storage-state storage_state.json
```

Trajectories land in `runs/<timestamp>.{json,md}` (gitignored). The final summary
(outcome, step count, file paths) prints to stdout.

## Outcomes

| outcome | meaning |
|---|---|
| `complete` | the model reported the goal satisfied (`done: true`) |
| `stuck` | same action chosen twice with no url/state change, or no actions available twice running |
| `budget_exhausted` | hit `--max-steps` without finishing |
| `error` | Manifest API unavailable, or the decision call failed — aborted, not retried |

## Failure visibility

Every step records the Manifest and DeepSeek call latency separately, the actions offered,
the model's decision + one-line reasoning, and the exact error if perception or execution
failed. Errors are typed (`errors.py`) and fed back into the next decision rather than
swallowed or blindly retried.

## Known limitation

Manifest perceives the page by URL with its own server-side fetch, using its own session —
not our Playwright session. So on a page that requires auth, Manifest sees the logged-out
version even after our agent has logged in: in a `saucedemo.com` login run, once the agent
reaches `/inventory.html` Manifest still reports the three login actions. The agent can cope
(it judged that goal complete from the URL change), but goals that need Manifest to perceive
authenticated pages accurately require a Manifest-side session, which is out of scope here.

`deepseek-v4-flash` is a reasoning model — it spends ~2.5k hidden tokens per decision, so
`MAX_DECISION_TOKENS` is set to 8k in `loop.py`. Decisions take ~1.5–2.5s each.

The Manifest free plan caps `/manifest` calls per month (50 at time of writing); each step
uses one. A 429 aborts the run cleanly as `error`.

## Data & secrets

- **`.env`** holds your `MANIFEST_API_KEY` and `DEEPSEEK_API_KEY`. It is gitignored — keep it
  that way. `.env.example` (committed) has empty placeholders.
- **`storage_state.json`** is a captured logged-in browser session (cookies, tokens). Gitignored.
  Treat it like a password.
- **`runs/`** trajectory files record every step verbatim, including the `value` typed into
  `fill` actions. If a goal contains a credential (`--goal "log in with password ..."`), that
  string is written in plaintext to `runs/*.json` and `runs/*.md`. `runs/` is gitignored, but
  the files sit unencrypted on disk — delete them or scrub values before sharing a trajectory.
  Prefer supplying credentials via `storage_state.json`, not in the goal text.

## Tests

```bash
python test_agent.py   # pure pieces: fence-stripping, requires-graph view, trajectory export
```

The loop itself is exercised by real runs (see the spec's acceptance criteria).

## License

MIT — see [LICENSE](LICENSE).
