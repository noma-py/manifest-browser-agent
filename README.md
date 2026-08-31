# Manifest Browser Agent

A local, headless-browser agent that uses the [Manifest API](https://manifest.omfang.io)
as its perception layer. Given a goal and a start URL it loops:

1. **Perceive** — ask Manifest what actions exist on the current page (with a `requires` graph)
2. **Decide** — ask Sonnet which action advances the goal, using that graph
3. **Act** — execute the action via Playwright, observe the new URL, repeat

Output is a CLI run plus a trajectory log (JSON + a case-study Markdown writeup). Not a product.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

export MANIFEST_API_KEY=...
export ANTHROPIC_API_KEY=...
```

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

Every step records the Manifest and Sonnet call latency separately, the actions offered,
the model's decision + one-line reasoning, and the exact error if perception or execution
failed. Errors are typed (`errors.py`) and fed back into the next decision rather than
swallowed or blindly retried.

## Known limitation

Manifest perceives the page by URL (server-side fetch), so client-side state your Playwright
session built up (a half-filled form, an opened modal) isn't visible to it. Flows that depend
on such state between steps may not perceive correctly. Out of scope for this pass.

## Tests

```bash
python test_agent.py   # pure pieces: fence-stripping, requires-graph view, trajectory export
```

The loop itself is exercised by real runs (see the spec's acceptance criteria).
