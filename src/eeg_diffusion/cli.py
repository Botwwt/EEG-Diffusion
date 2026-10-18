"""CPU-only release tools. No downloads or GPU jobs are launched."""
import argparse
import json
from pathlib import Path
from .audit import verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify', help='Verify source hashes and released aggregate arithmetic')
    verify.add_argument('--root', type=Path, default=Path.cwd())
    demo = sub.add_parser('reference-demo', help='Synthetic Gaussian reference; not a real EEG experiment')
    demo.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        result = verify_release(args.root)
    else:
        import numpy as np
        from .gaussian_reference import posterior_reference
        r = posterior_reference(np.array([[.8], [.8]]), ['a','b'], [0,0],
                                np.eye(1), np.zeros(1), np.eye(1), np.eye(1), .5*np.eye(1))
        result = {'synthetic':True,'unique_trials':r.unique_trials,
                  'posterior_mean':r.mean.tolist(),'posterior_covariance':r.covariance.tolist(),
                  'scope':'Classical Gaussian formula; no image-quality or real-EEG claim.'}
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

