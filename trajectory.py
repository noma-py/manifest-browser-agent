"""Trajectory: the run record. JSON = raw data, Markdown = case-study writeup."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class CallTiming:
    started_at: str
    latency_ms: float


@dataclass
class StepRecord:
    step: int
    url_before: str
    actions_available: list[dict]          # trimmed: id / label / type
    decision: dict | None = None           # {action_id, reasoning, done, value?}
    executed: str | None = None            # human-readable description of what ran
    error: str | None = None               # error string if perception/execution failed
    url_after: str | None = None
    manifest_call: CallTiming | None = None
    decision_call: CallTiming | None = None
    timestamp: str = field(default_factory=_now)


@dataclass
class Trajectory:
    goal: str
    start_url: str
    max_steps: int
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    outcome: str = "running"  # complete | stuck | budget_exhausted | error
    error: str | None = None
    steps: list[StepRecord] = field(default_factory=list)

    def add(self, step: StepRecord) -> None:
        self.steps.append(step)

    def finalize(self, outcome: str, error: str | None = None) -> None:
        self.outcome = outcome
        self.error = error
        self.finished_at = _now()

    # -- compact view for the decision prompt (last N steps) -----------------
    def recent_summary(self, n: int = 5) -> list[dict]:
        out = []
        for s in self.steps[-n:]:
            out.append(
                {
                    "step": s.step,
                    "url_before": s.url_before,
                    "chose": (s.decision or {}).get("action_id"),
                    "reasoning": (s.decision or {}).get("reasoning"),
                    "executed": s.executed,
                    "error": s.error,
                    "url_after": s.url_after,
                }
            )
        return out

    # -- exports -----------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        L = [
            f"# Trajectory — {self.goal}",
            "",
            f"- **Start URL:** {self.start_url}",
            f"- **Outcome:** `{self.outcome}`" + (f" — {self.error}" if self.error else ""),
            f"- **Steps taken:** {len(self.steps)} / {self.max_steps}",
            f"- **Started:** {self.started_at}",
            f"- **Finished:** {self.finished_at or '—'}",
            "",
        ]
        for s in self.steps:
            L.append(f"## Step {s.step}")
            L.append("")
            offered = ", ".join(f"`{a['id']}` ({a['type']})" for a in s.actions_available) or "none"
            para = f"At `{s.url_before}`, Manifest offered {len(s.actions_available)} action(s): {offered}. "
            if s.manifest_call:
                para += f"(Manifest call: {s.manifest_call.latency_ms:.0f} ms.) "
            if s.decision:
                d = s.decision
                if d.get("done"):
                    para += f"The model judged the goal complete: \"{d.get('reasoning')}\". "
                else:
                    val = f" with value \"{d['value']}\"" if d.get("value") else ""
                    para += (
                        f"The model chose **`{d.get('action_id')}`**{val} — "
                        f"\"{d.get('reasoning')}\". "
                    )
                if s.decision_call:
                    para += f"(Decision call: {s.decision_call.latency_ms:.0f} ms.) "
            if s.error:
                para += f"**Failure:** {s.error} "
            elif s.executed:
                para += f"Executed: {s.executed}. "
            if s.url_after and s.url_after != s.url_before:
                para += f"The page moved to `{s.url_after}`."
            elif s.executed and not s.error:
                para += "The URL did not change."
            L.append(para.strip())
            L.append("")
        return "\n".join(L)

    def write(self, out_dir: str = "runs") -> tuple[Path, Path]:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.replace(":", "").replace("-", "").replace(".", "_")
        jp = d / f"{stamp}.json"
        mp = d / f"{stamp}.md"
        jp.write_text(self.to_json())
        mp.write_text(self.to_markdown())
        return jp, mp
