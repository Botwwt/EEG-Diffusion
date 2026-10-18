"""Integrity and arithmetic checks of released evidence, not a training replay."""
from pathlib import Path
import hashlib
import json
import math


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_release(root):
    root = Path(root).resolve()
    manifest = read_json(root / "provenance/source_manifest.json")
    failures = []
    for record in manifest["files"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes the repository")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            failures.append(record["path"])
    if failures:
        raise ValueError("Source/asset hash mismatch: " + ", ".join(failures))
    summary = read_json(root / "results/summary.json")
    retrieval = read_json(root / "results/retrieval.json")
    expected = [r for r in retrieval['evaluations'] if r['role'] == 'full_raw']
    if summary['retrieval']['rows'] != expected:
        raise ValueError("Summary disagrees with released source aggregates")
    grouped = {}
    for r in expected:
        if r['split'] != 'official_test200':
            raise ValueError("Mixed evaluation splits")
        grouped.setdefault(r['method'], []).append(r['top1'])
    means = {k: sum(v)/len(v) for k,v in grouped.items()}
    delta_pp = 100 * (means['middle12_joint'] - means['final_joint'])
    if not math.isclose(delta_pp, 47.5, abs_tol=1e-10):
        raise ValueError("Retrieval difference inconsistent")
    image = read_json(root / 'results/image_pairing.json')
    arms = {r['arm']:r for r in image['final']}
    paired = next(c for c in image['paired_comparisons']
                  if c['control'] == 'native_full_target_cyclic_shift')
    delta = arms['native_full_target']['DINO_2AFC'] - arms['native_full_target_cyclic_shift']['DINO_2AFC']
    if not math.isclose(delta, paired['metrics']['DINO_2AFC']['mean_difference'], abs_tol=1e-12):
        raise ValueError("Paired image difference inconsistent")
    return {'source_assets_verified':len(manifest['files']),
            'retrieval_seed_means':means,'retrieval_difference_pp':delta_pp,
            'image_pairing_difference':delta,
            'scope':'File integrity and aggregate arithmetic; no EEG/weight/image-feature replay.'}

