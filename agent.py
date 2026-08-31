"""CLI entry point for the Manifest browser agent.

    python agent.py --goal "..." --start-url "https://..." [--max-steps 15] [--storage-state path]
"""

from __future__ import annotations

import argparse
import sys

import config
from loop import AgentLoop


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Manifest-driven headless browser agent")
    p.add_argument("--goal", required=True, help="natural-language goal")
    p.add_argument("--start-url", required=True, help="URL to start from")
    p.add_argument("--max-steps", type=int, default=15, help="step budget (default 15)")
    p.add_argument("--storage-state", default=None, help="path to a Playwright storage_state.json")
    args = p.parse_args(argv)

    try:
        cfg = config.load(args.storage_state)
    except config.ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    traj = AgentLoop(cfg, max_steps=args.max_steps).run(args.goal, args.start_url)
    jp, mp = traj.write()

    print("\n" + "=" * 60)
    print(f"goal:     {traj.goal}")
    print(f"outcome:  {traj.outcome}" + (f" ({traj.error})" if traj.error else ""))
    print(f"steps:    {len(traj.steps)} / {traj.max_steps}")
    print(f"json:     {jp}")
    print(f"markdown: {mp}")
    print("=" * 60)
    return 0 if traj.outcome == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
