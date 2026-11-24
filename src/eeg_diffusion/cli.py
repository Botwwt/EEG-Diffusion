"""CPU release tools: verify evidence or run a synthetic reference example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .audit import verify_release
from .gaussian_reference import posterior_reference
from .r4_audit import verify_r4_results
from .stack_audit import verify_stack_results
from .stack_inputs import check_o13_inputs


def main(argv: list[str] | None = None) -> int:
    """Run an explicit subcommand; return a nonzero status on invalid artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="Verify frozen hashes and aggregate arithmetic")
    verify.add_argument("--root", type=Path, default=Path.cwd())
    stack = subparsers.add_parser("verify-stack", help="Recompute November stack Top-1 statistics")
    stack.add_argument("--root", type=Path, default=Path.cwd())
    r4 = subparsers.add_parser("verify-r4", help="Recompute the R4 s375 CLIP Intra10 comparison")
    r4.add_argument("--root", type=Path, default=Path.cwd())
    preflight = subparsers.add_parser("check-o13-inputs", help="Inspect prepared O13 replay inputs")
    preflight.add_argument("--rad-root", type=Path, required=True)
    demo = subparsers.add_parser(
        "reference-demo", help="Synthetic Gaussian example; no EEG inference"
    )
    demo.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    status = 0
    try:
        if args.command == "verify":
            result = verify_release(args.root)
        elif args.command == "verify-stack":
            result = verify_stack_results(args.root)
        elif args.command == "verify-r4":
            result = verify_r4_results(args.root)
        elif args.command == "check-o13-inputs":
            result = check_o13_inputs(args.rad_root)
            status = 2 if result["missing"] or result["invalid_arrays"] else 0
        else:
            reference = posterior_reference(
                np.array([[0.8], [0.8]]),
                ["a", "b"],
                [0, 0],
                np.eye(1),
                np.zeros(1),
                np.eye(1),
                np.eye(1),
                0.5 * np.eye(1),
            )
            result = {
                "synthetic": True,
                "unique_trials": reference.unique_trials,
                "posterior_mean": reference.mean.tolist(),
                "posterior_covariance": reference.covariance.tolist(),
                "scope": "Classical Gaussian formula; no image-quality or real-EEG claim.",
            }
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        print(f"eeg-diffusion: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
