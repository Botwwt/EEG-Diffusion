"""Check public snapshot integrity without importing Torch or NumPy."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--index', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'provenance/research_snapshot_20261109.json').read_bytes())
    seen = set()
    parsed = 0
    index = {}
    if args.index:
        raw = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=root)
        for item in raw.split(b'\0'):
            if item:
                meta, name = item.split(b'\t', 1)
                index[name.decode('utf8')] = meta.split()[1].decode('ascii')
    patterns = [
        re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})'),
        re.compile(r'hf_[A-Za-z0-9]{30,}'),
        re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{32,}'),
        re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
        re.compile(r'https?://[^/\s:@]+:[^/\s@]+@'),
    ]
    for row in manifest['files']:
        relative = row['path']
        path = (root / relative).resolve()
        if relative in seen or not path.is_relative_to(root):
            raise ValueError('Duplicate or escaping snapshot path')
        seen.add(relative)
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Snapshot mismatch: ' + relative)
        if len(data) > 2 << 20:
            raise ValueError('Oversized snapshot file: ' + relative)
        if args.index:
            blob = b'blob ' + str(len(data)).encode() + b'\0' + data
            if hashlib.sha1(blob).hexdigest() != index.get(relative):
                raise ValueError('Staged blob differs from manifest: ' + relative)
        if path.suffix in ['.py', '.md', '.json', '.sh', '.sbatch', '.txt', '.csv']:
            text = data.decode('utf-8-sig')
            if any(pattern.search(text) for pattern in patterns):
                raise ValueError('Potential secret in: ' + relative)
            if path.suffix == '.py':
                ast.parse(text, filename=relative)
                parsed += 1
    # The new snapshot must not silently alter the earlier frozen release.
    for name in ['source_manifest.json', 'evidence_manifest.json']:
        old = json.loads((root / 'provenance' / name).read_bytes())
        for row in old['files']:
            path = (root / row['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Escaping existing release path')
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Existing release changed: ' + row['path'])
    print(json.dumps({'status': 'PASS', 'snapshot_files': len(seen),
                      'source_bytes': manifest['bytes'], 'python_files_parsed': parsed,
                      'binary_weights_or_EEG_included': False,
                      'scope': 'File integrity, syntax, size and secret-format checks; no scientific replay.'}))


if __name__ == '__main__':
    main()
