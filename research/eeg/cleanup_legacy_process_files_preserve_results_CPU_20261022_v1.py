"""Retire completed-run process files while preserving result evidence.

Final/best models, all images, all score tables, manifests, logs and calibration
models stay in place. Resume buffers require byte-identical saved final tensors.
Early epoch files are limited to reviewed completed three-epoch RIME runs.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from cleanup_BrainHIVE_baseline_preserving_CPU_20261022_v1 import sha, identity, assert_record, live_uses

ROOT = Path(__file__).resolve().parent
SERVER = Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
BASE = ROOT / 'audit/legacy_process_files_cleanup_20261022_v1'
RIME = Path('/root/rime_var_20261015/artifacts')
OLD_PLAN = ROOT / 'audit/BrainHIVE_legacy_retirement_exact_file_plan_20261022_v2.json.gz'
OLD_EPOCH_GROUPS = {
    'balanced_crossfit_v1', 'balanced_direct_bridge_v1', 'balanced_image_oracle_v1',
    'crossfit_scale_v1', 'fixed_random_evidence_v1', 'fixed_semantic_evidence_v1',
    'full_oracle_v1', 'full_train_seed2026101501_v1', 'scale_balanced_conditional_v1',
    'stabilized_direct_bridge_v1', 'trial_fusion_latent_mean_v1', 'trial_fusion_projected_mean_v1'}


def write(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def resume_pairs():
    pairs = []
    for name in ('holdout', 'reconstruction_unit', 'retrieval_raw', 'retrieval_unit'):
        folder = Path('/root/autodl-tmp') / f'atm_encoder_{name}_seed0_20261017_v1'
        pairs.append((folder / 'encoder_resume_last.pt', folder / 'encoder_final_10.pth'))
    for folder in (Path('/root/autodl-tmp/atm_native_prior_witness_seed0_20261017_v1'),
                   Path('/root/autodl-tmp/r4_prior_match_20261017_v1/train')):
        pairs.append((folder / 'prior_resume_last.pt', folder / 'prior_final_150.pth'))
    return pairs


def classification(path):
    p = Path(path)
    if p in {a for a, b in resume_pairs()}:
        return 'REDUNDANT_FINAL_MODEL_PLUS_OPTIMIZER_RESUME_BUFFER'
    if p.parent.parent == RIME and p.parent.name in OLD_EPOCH_GROUPS:
        if re.fullmatch(r'(?:fold[01]_epoch|conditional_epoch|marginal_epoch|epoch)[12]\.pt', p.name):
            return 'EARLY_EPOCH_OF_COMPLETED_FIXED_THREE_EPOCH_LEGACY_RUN'
    if p.parent == RIME / 'repeat_queries_v1' and p.name in {
            'R1.pt', 'balanced_R2.pt', 'balanced_R2_complement.pt', 'session_A_R2.pt', 'session_B_R2.pt'}:
        return 'DERIVED_REPEAT_QUERY_CACHE'
    if p.parent == RIME / 'trial_latents_v1' and p.name in {'latent_mean_R4.pt', 'precision_R4.pt'}:
        return 'DERIVED_REPEAT_DIAGNOSTIC_QUERY_CACHE'
    return None


def tensor_fingerprint(model):
    import torch
    h = hashlib.sha256()
    for name, tensor in model.items():
        assert isinstance(tensor, torch.Tensor), name
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def self_check():
    assert classification(RIME / 'full_oracle_v1/epoch1.pt')
    assert classification(resume_pairs()[0][0])
    for p in (RIME / 'full_oracle_v1/epoch3.pt', RIME / 'full_oracle_v1/latest.pt',
              RIME / 'trial_latents_v1/population_calibration.pt', RIME / 'trial_latents_v1/manifest.json',
              RIME / 'full_generation_v2/image.png', resume_pairs()[0][1],
              Path('/root/brainhive_reproduction_20261020_v1/components/bounded_v1/risk/component_final.pt')):
        assert classification(p) is None, p


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--apply', action='store_true')
    parser.add_argument('--self-check', action='store_true'); args = parser.parse_args()
    self_check()
    if args.self_check:
        print('PROCESS_FILE_SELECTION_PROTECTS_FINAL_MODELS_IMAGES_RESULTS_CALIBRATION'); return
    import torch
    torch.set_num_threads(2)
    assert ROOT == SERVER and not BASE.exists()
    assert sha(OLD_PLAN) == '25af127952982b12880ca3afbec269a0250ea8425772c974a54b6de25dcdb72c'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    old = json.loads(gzip.decompress(OLD_PLAN.read_bytes()))['files']
    targets = [x for x in old if classification(x['path']) and Path(x['path']).is_file()]
    assert len(targets) >= 41 and all(x['links'] == 1 for x in targets)
    protected = {}; proofs = []
    for source, final in resume_pairs():
        a = torch.load(source, map_location='cpu', weights_only=False)
        b = torch.load(final, map_location='cpu', weights_only=True)
        assert a['model'].keys() == b.keys()
        fingerprint = tensor_fingerprint(a['model']); assert fingerprint == tensor_fingerprint(b)
        proofs.append(dict(retired=str(source), retained_final=str(final), final_sha256=sha(final),
            identical_model_tensor_fingerprint=fingerprint, state_entries=len(b),
            retired_nonmodel_fields=[k for k in a if k != 'model'],
            exact_old_optimizer_RNG_resume_retained=False))
        protected[str(final)] = sha(final); del a, b
    for row in targets:
        p = Path(row['path']); assert_record(row)
        if classification(p).startswith('EARLY_EPOCH'):
            final = p.with_name(re.sub(r'([12])\.pt$', '3.pt', p.name))
            assert final.is_file() and final != p
            protected[str(final)] = sha(final)
            for latest in p.parent.glob('latest.pt'):
                protected[str(latest)] = sha(latest)
    recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
    assert not {x['path'] for x in targets}.intersection(recipe['input_sha256'])
    for name in ['results.tsv', 'experiment_ledger.jsonl', 'configs/BrainHIVE_bounded_main_recipe_v1.json',
                 'configs/BrainHIVE_functional_alignment_v2_preregistration_DRAFT.json']:
        protected[str(ROOT / name)] = sha(ROOT / name)
    for name, digest in recipe['source_sha256'].items():
        assert sha(ROOT / name) == digest; protected[str(ROOT / name)] = digest
    # Verify result files declared by the old file plan, and all retained learned states.
    skip = {x['path'] for x in targets}; retained_stats = {}; seen = set()
    retired_before = {x['path'] for x in json.loads((ROOT /
        'audit/baseline_preserving_cleanup_20261022_v1/plan.json').read_text())['targets']}
    for i, row in enumerate(old):
        p = Path(row['path'])
        if row['path'] in skip:
            continue
        if not p.is_file():
            assert row['path'] in retired_before, str(p)
            continue
        current = assert_record(row, content=False)
        if current[:2] not in seen:
            assert sha(p) == row['sha256'], p; seen.add(current[:2])
        retained_stats[str(p)] = current
        if i % 10000 == 0: print('VERIFIED_RETAINED_LEGACY_ENTRIES', i, flush=True)
    assert not live_uses([x['path'] for x in targets])
    BASE.mkdir(mode=0o700)
    before = {p: shutil.disk_usage(p).free for p in ('/root', '/root/autodl-tmp')}
    plan = dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        user_instruction='清理旧文件 结果保留好就行', source_sha256=sha(__file__), targets=targets,
        classifications={x['path']: classification(x['path']) for x in targets},
        resume_model_equivalence=proofs, retained_final_and_source_sha256=protected,
        before_free=before, retained_plan_entries=len(retained_stats),
        irreversible_scope='Old early-epoch states and optimizer/RNG resume buffers are retired; completed final/best model tensors and all result images/scores remain.',
        scientific_results_removed=False, candidate_counts_reset=False)
    write(BASE / 'plan.json', plan)
    if not args.apply: print('PLAN_ONLY'); return
    with (BASE / 'deletions.jsonl').open('x') as journal:
        assert not live_uses([x['path'] for x in targets])
        for row in targets:
            assert_record(row); Path(row['path']).unlink()
            journal.write(json.dumps(dict(path=row['path'], sha256=row['sha256'], removed=True)) + '\n')
            journal.flush(); os.fsync(journal.fileno())
    for p, previous in retained_stats.items(): assert identity(p) == previous, p
    for p, digest in protected.items(): assert sha(p) == digest, p
    for folder in {Path(x['path']).parent for x in targets}:
        write(folder / 'OLD_PROCESS_FILES_RETIRED_20261022_v1.json', dict(
            status='OLD_PROCESS_FILES_RETIRED_FINAL_RESULTS_PRESERVED', receipt=str(BASE / 'receipt.json'),
            plan_sha256=sha(BASE / 'plan.json')))
    after = {p: shutil.disk_usage(p).free for p in before}
    receipt = dict(status='OLD_PROCESS_FILES_REMOVED_FINAL_MODELS_AND_RESULTS_PRESERVED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), plan_sha256=sha(BASE / 'plan.json'),
        removed_files=len(targets), removed_allocated_bytes=sum(x['allocated'] for x in targets),
        before_free=before, after_free=after, retained_verified_old_plan_entries=len(retained_stats),
        old_resume_models_equal_saved_final_models=len(proofs), all_selected_final_models_SHA_unchanged=True,
        generated_images_raw_scores_reports_or_EEG_removed=0, no_new_GPU_work=True,
        formal_training_started=False, retained_sources_and_ledger_unchanged_during_cleanup=True)
    write(BASE / 'receipt.json', receipt); print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
