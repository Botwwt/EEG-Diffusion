"""CPU release tools: verify evidence or run a synthetic reference example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .audit import verify_release
from .gaussian_reference import posterior_reference


def main(argv: list[str] | None = None) -> int:
    """Run an explicit subcommand; return a nonzero status on invalid artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="Verify frozen hashes and aggregate arithmetic")
    verify.add_argument("--root", type=Path, default=Path.cwd())
    demo = subparsers.add_parser(
        "reference-demo", help="Synthetic Gaussian example; no EEG inference"
    )
    demo.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "verify":
            result = verify_release(args.root)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
