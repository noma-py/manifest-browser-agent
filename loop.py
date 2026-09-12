"""Perceive (Manifest) -> decide (DeepSeek) -> act (Playwright), until done or stuck."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import openai
from manifest_api import EXTRACTOR_JS, Action, Manifest, ManifestClient, RateLimitError
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from config import Config
from errors import (
    ActionExecutionError,
    AgentStuckError,
    ManifestUnavailableError,
    NoActionsAvailableError,
)
from trajectory import CallTiming, StepRecord, Trajectory

NETWORK_IDLE_TIMEOUT_MS = 8_000
ACTION_TIMEOUT_MS = 8_000
# authenticated SPA fetches (GA4) measured ~59s server-side; SDK default is 30s.
MANIFEST_TIMEOUT_S = 90.0
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# deepseek-v4-flash is a reasoning model — hidden reasoning scales with how much
# there is to reason about. ~2.5k tokens/turn on a small action list (saucedemo),
# but a real GA4 page with 50+ actions blew the 8k budget (finish_reason=length,
# empty content) by step 6. 24k leaves headroom for that; if a page with even
# more actions still exhausts it, the fix is trimming the actions payload, not
# raising this forever.
MAX_DECISION_TOKENS = 24_000
# a UI can offer a dozen always-blocked actions; the model varying its guess each
# step defeats the same-action stuck check while making zero real progress.
NO_PROGRESS_LIMIT = 5

_CLICK_TYPES = {"click", "submit", "button", "link", "check", "radio", "toggle"}
_FILL_TYPES = {"fill", "text", "input", "textarea", "email", "password",
               "search", "tel", "url", "number"}
_SELECT_TYPES = {"select"}

_SYSTEM = """You drive a headless browser to accomplish a goal. Each turn you get:
- the goal
- the actions Manifest found on the current page, each with its `requires` list
  (action ids that must be completed first) annotated with which are still missing
- a compact summary of your last few steps

`requires` is a best-effort dependency graph inferred from the DOM. Use it: if the
action that advances the goal is blocked by a missing requirement, pick the action
that satisfies that requirement first. Do not repeat an action that already failed
or produced no change - try a different one.

Reply with JSON only, no prose, no code fences:
{"action_id": "<id from the list, or null if done>",
 "value": "<text to type for fill actions, or the option label to pick for select actions, else omit>",
 "reasoning": "<one sentence, referencing the requires graph when relevant>",
 "done": <true if the goal is already satisfied, else false>}"""


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _action_kind(action_type: str, has_value: bool) -> str | None:
    """Collapse Manifest's HTML-ish action types to 'fill' / 'select' / 'click' / None."""
    if action_type in _SELECT_TYPES:
        return "select"
    if action_type in _FILL_TYPES or (action_type not in _CLICK_TYPES and has_value):
        return "fill"
    if action_type in _CLICK_TYPES:
        return "click"
    return None


def _action_view(a: Action, completed: set[str]) -> dict:
    # ponytail: SDK 0.4.0 nests requires as OR-groups (List[List[str]]); we flatten to a
    # flat AND-set rather than modeling OR precisely. Revisit if a real page's requires
    # actually has >1 group and the flattening produces a false "blocked".
    req = {r for group in a.requires for r in (group if isinstance(group, list) else [group])}
    return {
        "id": a.id,
        "label": a.label,
        "type": a.type,
        "description": a.description,
        "required": a.required,
        "requires": a.requires,
        "requires_missing": sorted(req - completed),
        "blocked": bool(req - completed),
    }


class AgentLoop:
    def __init__(self, config: Config, max_steps: int = 15, demo_pace: float = 0.0, headed: bool = False):
        self.cfg = config
        self.max_steps = max_steps
        self.demo_pace = demo_pace
        self.headed = headed
        self.manifest = ManifestClient(api_key=config.manifest_api_key, timeout=MANIFEST_TIMEOUT_S)
        self.llm = openai.OpenAI(api_key=config.deepseek_api_key, base_url=DEEPSEEK_BASE_URL)

    # -- perception -----------------------------------------------------
    # Perceive via the agent's own live page (from-dom), not a cold server-side
    # re-navigation of the URL. That means auth and any client-side-only state
    # (an open picker/overlay/modal that never touched the URL) are both visible,
    # since we're reading the DOM our own browser is actually looking at right now.
    def _fetch_manifest(self, page, url: str) -> tuple[Manifest, CallTiming]:
        started, t0 = _iso(), time.monotonic()
        try:
            page.evaluate(EXTRACTOR_JS)
            dom_context = page.evaluate("window.__semanticAgentLayerExtractDomContext()")
            # no cache_scope: a wizard/funnel's DOM changes every step, so caching
            # would risk serving a stale overlay back on the next perceive.
            m = self.manifest.get_from_dom(url, dom_context)
        except RateLimitError as e:
            # Manifest's 429 covers both a per-minute burst and a hard plan quota
            # ("Monthly manifest limit reached"). Neither is worth retrying here.
            raise ManifestUnavailableError(
                f"Manifest rate limit hit for {url}: {e} "
                "(check your plan's monthly manifest quota)"
            ) from e
        except Exception as e:  # noqa: BLE001 - any other failure is "Manifest unavailable"
            raise ManifestUnavailableError(f"Manifest call failed for {url}: {e}") from e
        return m, CallTiming(started, (time.monotonic() - t0) * 1000)

    # -- decision ------------------------------------------------------
    def _decide(self, goal: str, actions: list[dict], traj: Trajectory) -> tuple[dict, CallTiming]:
        payload = {"goal": goal, "actions": actions, "recent_steps": traj.recent_summary(5)}
        started, t0 = _iso(), time.monotonic()
        resp = self.llm.chat.completions.create(
            model=self.cfg.model,
            max_tokens=MAX_DECISION_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
        )
        timing = CallTiming(started, (time.monotonic() - t0) * 1000)
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()
        if not raw:
            raise ValueError(f"empty decision response (finish_reason={choice.finish_reason})")
        return json.loads(_strip_fences(raw)), timing

    # -- action ------------------------------------------------------
    def _locator(self, page, action: Action):
        loc = action.locator
        if loc is None:
            raise ActionExecutionError(action.id, None, "manifest gave no locator")
        if loc.css:
            return page.locator(loc.css).first
        if loc.role:
            return page.get_by_role(loc.role, name=loc.name)
        raise ActionExecutionError(action.id, loc, "locator has neither css nor role")

    def _select(self, page, action: Action, target, value: str | None) -> None:
        if not value:
            raise ActionExecutionError(action.id, action.locator, "select action needs a value")
        try:
            target.select_option(label=value, timeout=ACTION_TIMEOUT_MS)
            return
        except PlaywrightError:
            pass  # not a native <select> (e.g. an ARIA combobox like mat-select) - open + pick
        target.click(timeout=ACTION_TIMEOUT_MS)
        page.get_by_role("option", name=value).first.click(timeout=ACTION_TIMEOUT_MS)

    def _click_with_escape_retry(self, page, target) -> None:
        try:
            target.click(timeout=ACTION_TIMEOUT_MS)
        except PlaywrightError as e:
            if "intercepts pointer events" not in str(e):
                raise
            # a stuck overlay/backdrop is blocking the click; CDK-style overlays
            # (mat-dialog, cdk-overlay) close on Escape by default - try once.
            page.keyboard.press("Escape")
            target.click(timeout=ACTION_TIMEOUT_MS)

    def _execute(self, page, action: Action, value: str | None) -> str:
        target = self._locator(page, action)
        kind = _action_kind(action.type, value is not None)
        if kind is None:
            raise ActionExecutionError(
                action.id, action.locator, f"unhandled action type {action.type!r}"
            )
        try:
            if kind == "fill":
                target.fill(value or "", timeout=ACTION_TIMEOUT_MS)
                return f"filled {action.label!r} with {value!r}"
            if kind == "select":
                self._select(page, action, target, value)
                return f"selected {value!r} in {action.label!r}"
            self._click_with_escape_retry(page, target)
            return f"clicked {action.label!r}"
        except ActionExecutionError:
            raise
        except Exception as e:  # noqa: BLE001 - wrap the underlying Playwright failure
            raise ActionExecutionError(action.id, action.locator, e) from e

    # -- main loop --------------------------------------------------
    def run(self, goal: str, start_url: str) -> Trajectory:
        traj = Trajectory(goal=goal, start_url=start_url, max_steps=self.max_steps)
        completed: set[str] = set()
        last_action_id: str | None = None
        last_key: tuple[str, str | None] | None = None  # (url_before, manifest fingerprint)
        no_actions_streak = 0
        no_progress_streak = 0

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not self.headed)
            ctx_kw = {"storage_state": self.cfg.storage_state} if self.cfg.storage_state else {}
            context = browser.new_context(viewport={"width": 1280, "height": 800}, **ctx_kw)
            page = context.new_page()
            page.goto(start_url, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
            except PlaywrightError:
                pass  # same best-effort settle used after every action below

            try:
                for step in range(self.max_steps):
                    url_before = page.url
                    rec = StepRecord(step=step, url_before=url_before, actions_available=[])
                    _log(f"[step {step}] perceiving {url_before} ...")

                    try:
                        manifest, rec.manifest_call = self._fetch_manifest(page, url_before)
                    except ManifestUnavailableError as e:
                        _log(f"[step {step}] ✗ {e}")
                        rec.error = str(e)
                        traj.add(rec)
                        traj.finalize("error", str(e))
                        return traj

                    rec.actions_available = [
                        {"id": a.id, "label": a.label, "type": a.type} for a in manifest.actions
                    ]

                    if not manifest.actions:
                        no_actions_streak += 1
                        rec.error = f"{NoActionsAvailableError.__name__}: manifest returned zero actions"
                        _log(f"[step {step}] ✗ {rec.error}")
                        traj.add(rec)
                        if no_actions_streak >= 2:
                            raise AgentStuckError("no actions available for 2 consecutive steps")
                        last_action_id, last_key = None, None
                        continue
                    no_actions_streak = 0

                    # stuck check: did the previous identical action change nothing?
                    key = (url_before, manifest.fingerprint)
                    # (evaluated after we know this step's action_id, below)

                    actions_view = [_action_view(a, completed) for a in manifest.actions]
                    try:
                        decision, rec.decision_call = self._decide(goal, actions_view, traj)
                    except (ValueError, openai.APIError) as e:  # bad/empty JSON, API failure
                        rec.error = f"decision step failed: {e}"
                        _log(f"[step {step}] ✗ {rec.error}")
                        traj.add(rec)
                        traj.finalize("error", rec.error)
                        return traj
                    rec.decision = decision

                    if decision.get("done"):
                        _log(f"[step {step}] ✓ done — {decision.get('reasoning', '')}")
                        traj.add(rec)
                        traj.finalize("complete")
                        return traj

                    action_id = decision.get("action_id")
                    _log(f"[step {step}] → {action_id!r}: {decision.get('reasoning', '')}")

                    if (
                        action_id is not None
                        and action_id == last_action_id
                        and key == last_key
                    ):
                        _log(f"[step {step}] ✗ stuck — {action_id!r} chosen again, no state change")
                        traj.add(rec)
                        raise AgentStuckError(
                            f"action {action_id!r} chosen again with no url/state change"
                        )

                    action = manifest.action(action_id) if action_id else None
                    if action is None:
                        rec.error = f"model chose unknown action_id {action_id!r}"
                        _log(f"[step {step}] ✗ {rec.error}")
                        no_progress_streak += 1
                    else:
                        try:
                            rec.executed = self._execute(page, action, decision.get("value"))
                            completed.add(action.id)
                            _log(f"[step {step}] ✓ {rec.executed}")
                            no_progress_streak = 0
                        except ActionExecutionError as e:
                            rec.error = str(e)
                            _log(f"[step {step}] ✗ {rec.error}")
                            no_progress_streak += 1

                    if no_progress_streak >= NO_PROGRESS_LIMIT:
                        _log(f"[step {step}] ✗ stuck — no action has succeeded in "
                             f"{NO_PROGRESS_LIMIT} steps")
                        traj.add(rec)
                        raise AgentStuckError(
                            f"no action succeeded in the last {NO_PROGRESS_LIMIT} steps"
                        )

                    try:
                        page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
                    except PlaywrightError:
                        pass
                    rec.url_after = page.url
                    traj.add(rec)

                    last_action_id, last_key = action_id, key
                    if self.demo_pace:
                        time.sleep(self.demo_pace)

                traj.finalize("budget_exhausted")
                return traj

            except AgentStuckError as e:
                traj.finalize("stuck", str(e))
                return traj
            finally:
                context.close()
                browser.close()
