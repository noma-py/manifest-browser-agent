"""Explicit error types. Failure modes stay visible, never swallowed."""

from __future__ import annotations


class AgentError(Exception):
    """Base for everything this agent raises."""


class ManifestUnavailableError(AgentError):
    """A Manifest API call failed (network / 5xx / timeout). Aborts the run."""


class NoActionsAvailableError(AgentError):
    """Manifest returned an empty actions list. Feeds the stuck-detector, not an abort."""


class ActionExecutionError(AgentError):
    """A Playwright action failed. Wraps the underlying exception + locator."""

    def __init__(self, action_id: str, locator: object, cause: BaseException | str):
        self.action_id = action_id
        self.locator = locator
        self.cause = cause
        super().__init__(f"action {action_id!r} failed (locator={locator!r}): {cause}")


class AgentStuckError(AgentError):
    """Raised by the stuck-detector, caught by the loop to end the run cleanly."""
