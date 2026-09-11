"""Runnable checks for the pure pieces. `python test_agent.py` — no framework.

The perceive/decide/act loop itself is covered by the acceptance-criteria live runs
(real logged-in flow, impossible goal, killed Manifest API)."""

from manifest_api import Action, Locator

from loop import _action_kind, _action_view, _strip_fences
from trajectory import CallTiming, StepRecord, Trajectory


def test_strip_fences():
    assert _strip_fences('{"a": 1}') == '{"a": 1}'
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_action_view_requires_graph():
    a = Action(id="submit", label="Place order", type="click", description="",
               required=True, requires=["shipping_confirmed", "cart_filled"],
               locator=Locator(css="#go"))
    v = _action_view(a, completed={"cart_filled"})
    assert v["requires_missing"] == ["shipping_confirmed"]
    assert v["blocked"] is True

    v2 = _action_view(a, completed={"cart_filled", "shipping_confirmed"})
    assert v2["blocked"] is False


def test_action_kind():
    assert _action_kind("text", has_value=True) == "fill"
    assert _action_kind("password", has_value=False) == "fill"
    assert _action_kind("submit", has_value=False) == "click"
    assert _action_kind("link", has_value=False) == "click"
    assert _action_kind("weird-unknown", has_value=True) == "fill"   # value => treat as fill
    assert _action_kind("weird-unknown", has_value=False) is None    # give up, surfaces an error
    assert _action_kind("select", has_value=True) == "select"        # mat-select etc., not fill


def test_trajectory_export():
    t = Trajectory(goal="check out", start_url="https://x/cart", max_steps=15)
    t.add(StepRecord(
        step=0, url_before="https://x/cart",
        actions_available=[{"id": "go", "label": "Checkout", "type": "click"}],
        decision={"action_id": "go", "reasoning": "advance to checkout", "done": False},
        executed="click 'Checkout'", url_after="https://x/checkout",
        manifest_call=CallTiming("t", 321.0), decision_call=CallTiming("t", 1200.0),
    ))
    t.finalize("complete")

    assert '"outcome": "complete"' in t.to_json()
    md = t.to_markdown()
    assert "## Step 0" in md
    assert "`go`" in md and "advance to checkout" in md
    assert "moved to `https://x/checkout`" in md

    assert t.recent_summary(5)[0]["chose"] == "go"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
