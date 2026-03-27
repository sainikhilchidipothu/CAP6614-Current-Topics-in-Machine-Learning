#!/usr/bin/env python3
"""Run WikiText-2 perplexity + lm-eval zero-shot for one model (dense baseline)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument(
        "--skip-ppl",
        action="store_true",
        help="Only run zero-shot",
    )
    ap.add_argument(
        "--skip-zeroshot",
        action="store_true",
        help="Only run perplexity",
    )
    ap.add_argument(
        "--zeroshot-limit",
        type=float,
        default=None,
        help="Pass through to eval_zeroshot.py --limit (e.g. 0.02 for smoke test)",
    )
    args = ap.parse_args()

    py = sys.executable
    env = os.environ.copy()
    extra = str(ROOT)
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    if not args.skip_ppl:
        subprocess.run(
            [py, str(ROOT / "scripts" / "eval_ppl_wikitext2.py"), "--model", args.model],
            cwd=str(ROOT),
            env=env,
            check=True,
        )

    if not args.skip_zeroshot:
        cmd = [
            py,
            str(ROOT / "scripts" / "eval_zeroshot.py"),
            "--model",
            args.model,
        ]
        if args.zeroshot_limit is not None:
            cmd += ["--limit", str(args.zeroshot_limit)]
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
