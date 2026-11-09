"""Inspect saved evidence or dispatch existing registered controllers.

No asset download, checkpoint selection, budget extension or historical overwrite.
The default is read-only inspection. GPU stages require complete current funding.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def funding(reservation):
    bounds = load(ROOT/'configs/compute_time_bounds.json')
    charged = sum(float(json.loads(x).get('charged_seconds', 0))
                  for x in (ROOT/'audit/compute_ledger.jsonl').read_text().splitlines())
    remaining = max(0, min(40*3600-charged,
                          bounds['conservative_occupied_time_deadline_epoch']-time.time()))
    return dict(complete_lifecycle_funded=remaining>reservation,
                remaining_authorized_seconds=remaining, reserved_seconds=reservation,
                charged_GPU_job_hours=charged/3600,
                occupied_deadline_utc=bounds['deadline_utc'], resource_limit_changed=False)


def inspect_saved():
    closure = load(ROOT/'audit/idea_Frozen_RealPaired_Conditional_full_score_closure_v1.json')
    assert closure['status'] == 'PASSED_ALL16600_COMPLETED_SCORE_IDENTITY_AND_PAIRED_ARITHMETIC'
    registry = load(ROOT/'audit/current_model_registry_CPU_20261020_v1.json')
    assert sha(registry['inventory']) == registry['inventory_sha256']
    assert sha(ROOT/'audit/current_model_structures_CPU_20261020_v1.json') == registry['structures_sha256']
    for arm in closure['completed_arms']:
        for item in closure['sources'][arm].values():
            assert Path(item['path']).is_file(), item['path']
            assert sha(item['path']) == item['sha256'], item['path']
    reg = ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json'
    cfg = load(reg)
    for path, expected in cfg['source_sha256'].items():
        assert sha(path) == expected, path
    reservation = sum(cfg['phase_caps_seconds'].values())+cfg['reserve_seconds']+45
    return dict(status='SAVED_EVIDENCE_REFERENCES_AND_REGISTERED_SOURCES_VERIFIED',
        completed_image_arms=closure['completed_arms'], queries_per_arm=16600,
        registry_rows=registry['rows'], registry_CPU_strict_loads=registry['strict_loaded_count'],
        published_decoder_images=sum(1 for _ in (Path(cfg['output_root'])/'images').glob('sub-*/*.png')),
        public_decoder_budget=funding(reservation), new_GPU_jobs=0,
        scope='Rechecks saved evidence/source hashes; does not repeat image evaluation or strict checkpoint loads',
        comparable_SOTA_established=False, independent_component_increment=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['inspect', 'resume-existing-arm', 'author-decoder'], default='inspect')
    parser.add_argument('--arm', choices=['Frozen', 'RealPaired', 'Conditional'])
    args = parser.parse_args()
    if args.stage == 'inspect':
        print(json.dumps(inspect_saved(), indent=2))
        return 0
    if args.stage == 'resume-existing-arm':
        if args.arm is None:
            parser.error('--arm is required for resume-existing-arm')
        resolved = ROOT/'configs'/f'idea_interface_{args.arm}_resolved_v1.json'
        state_path = ROOT/'audit'/f'idea_interface_{args.arm}_state_v1.json'
        state = load(state_path)
        assert state['resolved_sha256'] == sha(resolved)
        if state['status'] == 'COMPLETED_FULL_ARM_IMAGES_METRICS_AND_IDENTITY':
            print(json.dumps(dict(status='ALREADY_COMPLETE_NO_REGENERATION', arm=args.arm, new_GPU_jobs=0)))
            return 0
        # The original controller independently recomputes exact remaining phase
        # costs and refuses existing partial outputs requiring a registered repair.
        budget = funding(1)
        if not budget['complete_lifecycle_funded']:
            print(json.dumps(dict(status='RESOURCE_LIMIT_REFUSAL_NO_JOB_STARTED', **budget)))
            return 2
        command = [sys.executable, str(ROOT/'run_idea_interface_arm_v2.py'),
                   '--resolved', str(resolved), '--through', 'report', '--resume-planned-phases']
    else:
        reg = ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json'
        cfg = load(reg)
        budget = funding(sum(cfg['phase_caps_seconds'].values())+cfg['reserve_seconds']+45)
        if not budget['complete_lifecycle_funded']:
            print(json.dumps(dict(status='RESOURCE_LIMIT_REFUSAL_NO_JOB_STARTED', **budget)))
            return 2
        assert not (ROOT/'audit/ATM_published_decoder_state_20261020_v1.json').exists(), 'Preserve earlier state; repair needs a new registration'
        command = [sys.executable, str(ROOT/'run_ATM_published_decoder_v1.py'),
                   '--registration', str(reg), '--registration-sha256', sha(reg)]
    return subprocess.call(command, cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
