"""Perceive (Manifest) -> decide (DeepSeek) -> act (Playwright), until done or stuck."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import openai
from manifest_api import Action, Manifest, ManifestClient
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
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

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
 "value": "<text to type, only for fill actions, else omit>",
 "reasoning": "<one sentence, referencing the requires graph when relevant>",
 "done": <true if the goal is already satisfied, else false>}"""


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _action_view(a: Action, completed: set[str]) -> dict:
    req = set(a.requires)
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
    def __init__(self, config: Config, max_steps: int = 15):
        self.cfg = config
        self.max_steps = max_steps
        self.manifest = ManifestClient(api_key=config.manifest_api_key)
        self.llm = openai.OpenAI(api_key=config.deepseek_api_key, base_url=DEEPSEEK_BASE_URL)

    # -- perception -----------------------------------------------------
    # ponytail: Manifest perceives by URL (server-side fetch), so client-side state
    # our Playwright page built up isn't visible to it. Fine for URL-routed flows;
    # revisit with a DOM-snapshot upload path if state-dependent steps misperceive.
    def _fetch_manifest(self, url: str) -> tuple[Manifest, CallTiming]:
        started, t0 = _iso(), time.monotonic()
        try:
            m = self.manifest.get(url)
        except Exception as e:  # noqa: BLE001 - any failure here is "Manifest unavailable"
            raise ManifestUnavailableError(f"Manifest call failed for {url}: {e}") from e
        return m, CallTiming(started, (time.monotonic() - t0) * 1000)

    # -- decision ------------------------------------------------------
    def _decide(self, goal: str, actions: list[dict], traj: Trajectory) -> tuple[dict, CallTiming]:
        payload = {"goal": goal, "actions": actions, "recent_steps": traj.recent_summary(5)}
        started, t0 = _iso(), time.monotonic()
        resp = self.llm.chat.completions.create(
            model=self.cfg.model,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
        )
        timing = CallTiming(started, (time.monotonic() - t0) * 1000)
        raw = resp.choices[0].message.content or ""
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

    def _execute(self, page, action: Action, value: str | None) -> str:
        target = self._locator(page, action)
        try:
            if action.type == "fill":
                target.fill(value or "", timeout=ACTION_TIMEOUT_MS)
                return f"filled {action.label!r} with {value!r}"
            if action.type in ("click", "submit"):
                target.click(timeout=ACTION_TIMEOUT_MS)
                return f"{action.type} {action.label!r}"
            raise ActionExecutionError(
                action.id, action.locator, f"unknown action type {action.type!r}"
            )
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

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx_kw = {"storage_state": self.cfg.storage_state} if self.cfg.storage_state else {}
            context = browser.new_context(**ctx_kw)
            page = context.new_page()
            page.goto(start_url, wait_until="domcontentloaded")

            try:
                for step in range(self.max_steps):
                    url_before = page.url
                    rec = StepRecord(step=step, url_before=url_before, actions_available=[])

                    try:
                        manifest, rec.manifest_call = self._fetch_manifest(url_before)
                    except ManifestUnavailableError as e:
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
                    except (json.JSONDecodeError, openai.APIError) as e:
                        rec.error = f"decision step failed: {e}"
                        traj.add(rec)
                        traj.finalize("error", rec.error)
                        return traj
                    rec.decision = decision

                    if decision.get("done"):
                        traj.add(rec)
                        traj.finalize("complete")
                        return traj

                    action_id = decision.get("action_id")

                    if (
                        action_id is not None
                        and action_id == last_action_id
                        and key == last_key
                    ):
                        traj.add(rec)
                        raise AgentStuckError(
                            f"action {action_id!r} chosen again with no url/state change"
                        )

                    action = manifest.action(action_id) if action_id else None
                    if action is None:
                        rec.error = f"model chose unknown action_id {action_id!r}"
                    else:
                        try:
                            rec.executed = self._execute(page, action, decision.get("value"))
                            completed.add(action.id)
                        except ActionExecutionError as e:
                            rec.error = str(e)

                    try:
                        page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
                    except PlaywrightError:
                        pass
                    rec.url_after = page.url
                    traj.add(rec)

                    last_action_id, last_key = action_id, key

                traj.finalize("budget_exhausted")
                return traj

            except AgentStuckError as e:
                traj.finalize("stuck", str(e))
                return traj
            finally:
                context.close()
                browser.close()
