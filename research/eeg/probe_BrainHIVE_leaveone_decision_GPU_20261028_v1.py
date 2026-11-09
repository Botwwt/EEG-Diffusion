"""D31: finite, read-only complete-T leave-one-repeat decision diagnostic.

Original25/native50/serial/parallel encoders are frozen. Only true held-role
T rows are forwarded. Actual mean3 views and one fixed coordinate-permutation
control are compared at the original25 role's fixed temperature. No fit,
Fisher approximation, image generation, model export or image-admission file.
The finite stability formula is classical, not a novelty claim.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
from datetime import datetime, timezone
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import signal as signal_control
import subprocess
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
N, D, NC = 13230, 4250, 1323
MODELS = ('original25', 'native50', 'serial_reversible', 'parallel_matched')
NEW_ARMS = ('native_continuation', 'serial_reversible', 'parallel_matched')
METHODS = ('native_condition', 'frozen_old_ordinary')
VIEWS = ('mean4', 'mean3_0', 'mean3_1', 'mean3_2', 'mean3_3',
         'permuted_delta_0', 'permuted_delta_1', 'permuted_delta_2', 'permuted_delta_3')
VIEW_METRICS = ('correct_probability', 'correct_minus_nine_wrong_mean_cosine',
                'same_concept_Top1', 'same_concept_2AFC')
TRIALS_SHA = 'b804a105822bfab5eeb853b6942d25109eaf88f3f8a1eb5b503b444a4dc0d393'
INDEX_SHA = 'a12c2758f8555223e08022e5b4a195da37ad3dac37cb6e2806aaf17bac014e67'
TARGETS_SHA = '363e8e74514ca8d1bcbfb29afc2023ed4a5c3a6c4f49bf8b1134e7ae7a090336'
NATIVE_SHA = 'bb6e38350282af7b37ece04689adcec7a5f87dccf68de5968058433faece0597'
OLD_ORDINARY_SHA = 'a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def checked(item):
    path = Path(item['path'])
    assert path.is_file() and sha(path) == item['sha256'], str(path)
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes']
    return path


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    assert not temporary.exists()
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def log_softmax(logits):
    value = np.asarray(logits, np.float64)
    maximum = value.max(axis=-1, keepdims=True)
    shifted = value - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def score_views(cosines, labels, kappa):
    """cosines [B,9,10]; finite exact metrics, with fractional ties."""
    c = np.asarray(cosines, np.float64)
    labels = np.asarray(labels, np.int64)
    assert c.shape == (len(labels), 9, 10) and np.isfinite(c).all()
    assert math.isfinite(float(kappa)) and float(kappa) > 0
    logp = log_softmax(c * kappa)
    probability = np.exp(logp)
    correct = np.take_along_axis(c, labels[:, None, None].repeat(9, axis=1), axis=-1)[..., 0]
    cp = np.take_along_axis(probability, labels[:, None, None].repeat(9, axis=1), axis=-1)[..., 0]
    margin = correct - (c.sum(-1) - correct) / 9.
    maximum = c.max(-1)
    top1 = (correct == maximum) / (c == maximum[..., None]).sum(-1)
    afc = ((correct[..., None] > c).sum(-1) + .5 * (correct[..., None] == c).sum(-1) - .5) / 9.
    per_view = np.stack((cp, margin, top1, afc), axis=-1)
    base_log, base_p = logp[:, :1], probability[:, :1]
    # Sum of the two directed KLs = (p_r-p_0).(log p_r-log p_0).
    sym_per_repeat = .5 * ((probability[:, 1:] - base_p) * (logp[:, 1:] - base_log)).sum(-1)
    sym_per_repeat = sym_per_repeat.reshape(len(c), 2, 4)
    assert sym_per_repeat.min() > -1e-12
    K = sym_per_repeat.mean(-1)  # exactly (1/8) sum_r [KL0r+KLr0]
    return per_view, K, sym_per_repeat


def make_views(raw, permutation):
    """Use the original D30 float32 mean4 convention; all views end in FP32."""
    x = np.asarray(raw)
    assert x.ndim == 4 and x.shape[1:] == (4, 17, 250) and x.dtype == np.float32
    assert np.isfinite(x).all() and np.array_equal(np.sort(permutation), np.arange(D))
    base = x.mean(axis=1, dtype=np.float32).reshape(len(x), D)
    delta = (base[:, None, :].astype(np.float64) - x.reshape(len(x), 4, D).astype(np.float64)) / 3.
    rotated = delta[..., permutation]
    declared_gram = np.einsum('brd,bsd->brs', delta, delta)
    permuted_gram = np.einsum('brd,bsd->brs', rotated, rotated)
    relative = np.linalg.norm(declared_gram - permuted_gram, axis=(1, 2)) / np.maximum(np.linalg.norm(declared_gram, axis=(1, 2)), 1e-20)
    assert relative.max() < 1e-12, 'Common permutation failed repeat Gram preservation'
    declared_spectra = np.stack((np.linalg.eigvalsh(declared_gram), np.linalg.eigvalsh(permuted_gram)), axis=1)
    spectral_error = np.linalg.norm(declared_spectra[:, 0]-declared_spectra[:, 1], axis=-1) / np.maximum(np.linalg.norm(declared_gram, axis=(1, 2)), 1e-20)
    assert spectral_error.max() < 1e-12
    real = (base[:, None, :].astype(np.float64) + delta).astype(np.float32)
    control = (base[:, None, :].astype(np.float64) + rotated).astype(np.float32)
    values = np.concatenate((base[:, None, :], real, control), axis=1)
    actual_delta = values[:, 1:].astype(np.float64).reshape(len(x), 2, 4, D) - base[:, None, None, :]
    actual_gram = np.einsum('bprd,bpsd->bprs', actual_delta, actual_delta)
    gram_difference = np.linalg.norm(actual_gram[:, 0] - actual_gram[:, 1], axis=(1, 2))
    rounding_bound_scale = np.linalg.norm(declared_gram, axis=(1, 2)) + np.square(base.astype(np.float64)).sum(-1) * 1e-10 + 1e-20
    actual_relative = gram_difference / rounding_bound_scale
    assert actual_relative.max() < 3e-5, 'FP32 actual-input Gram discrepancy exceeds declared tolerance'
    actual_spectra = np.linalg.eigvalsh(actual_gram)
    actual_spectral_error = np.linalg.norm(actual_spectra[:, 0]-actual_spectra[:, 1], axis=-1) / rounding_bound_scale
    assert actual_spectral_error.max() < 3e-5
    # Compare formula views with directly averaging the three physical records.
    direct = np.stack([np.delete(x.astype(np.float64), r, axis=1).mean(1).reshape(len(x), D) for r in range(4)], axis=1)
    direct_error = np.linalg.norm(real.astype(np.float64) - direct, axis=-1) / np.maximum(np.linalg.norm(direct, axis=-1), 1e-20)
    assert direct_error.max() < 3e-6
    return values, dict(declared_gram=declared_gram, actual_gram=actual_gram,
        declared_spectra=declared_spectra, actual_spectra=actual_spectra,
        perturbation_norms=np.sqrt(np.maximum(np.diagonal(actual_gram, axis1=-2, axis2=-1), 0.)),
        declared_Gram_relative_error=relative, actual_Gram_scaled_relative_error=actual_relative,
        declared_spectrum_relative_error=spectral_error, actual_spectrum_scaled_relative_error=actual_spectral_error,
        delta_sum_L2=np.linalg.norm(delta.sum(1), axis=-1), physical_mean3_relative_error=direct_error.max(-1))


def validate_split(split_path, manifest_path):
    split = read(split_path)
    rows = split['image_rows']['T']
    assert len(rows) == N and len(set(rows)) == N
    assert not set(rows) & (set(split['image_rows']['D']) | set(split['image_rows']['V']))
    with Path(manifest_path).open(encoding='utf-8', newline='') as stream:
        records = [row for row in csv.DictReader(stream) if row['dataset_split'] == 'train']
    metadata = {int(row['array_row']): row for row in records}
    assert len(records) == len(metadata) == 16540
    assert {metadata[row]['concept_id'] for row in rows} == set(split['train_concepts']['T'])
    groups = {}
    for position, row in enumerate(rows):
        groups.setdefault(metadata[row]['concept_id'], []).append(position)
    assert len(groups) == NC and set(map(len, groups.values())) == {10}
    folds = np.array([split['oof_fold_by_concept'][metadata[row]['concept_id']] for row in rows], np.int8)
    assert set(folds.tolist()) == {0, 1}
    return split, rows, metadata, groups, folds


def self_test(split_path, manifest_path):
    rng = np.random.default_rng(1729)
    raw = rng.normal(size=(3, 4, 17, 250)).astype(np.float32)
    permutation = rng.permutation(D)
    _, proof = make_views(raw, permutation)
    x = raw.astype(np.float64); m = x.mean(1); delta = (m[:, None] - x) / 3.
    assert np.max(np.abs(delta.sum(1))) < 1e-14
    sample = np.einsum('brd,brd->b', (x-m[:, None]).reshape(3, 4, D), (x-m[:, None]).reshape(3, 4, D)) / 3.
    assert np.allclose(np.square(delta).sum((1, 2, 3)) / 4., sample / 12., atol=1e-12)
    small=(x-m[:, None]).reshape(3,4,D)[:,:,:16]
    small_delta=delta.reshape(3,4,D)[:,:,:16]
    assert np.allclose(np.einsum('brd,bre->bde',small_delta,small_delta)/4.,
                       np.einsum('brd,bre->bde',small,small)/36.,atol=1e-12)
    scores = rng.normal(size=(3, 9, 10)); labels = np.array([0, 4, 9])
    metrics, K, per = score_views(scores, labels, 8.)
    logs = log_softmax(scores*8); ps = np.exp(logs)
    for i in range(3):
        for kind in range(2):
            manual = 0.
            for r in range(4):
                v = 1 + kind*4+r
                manual += (ps[i, 0]*(logs[i, 0]-logs[i, v])).sum() + (ps[i, v]*(logs[i, v]-logs[i, 0])).sum()
            assert abs(K[i, kind]-manual/8.) < 1e-12
    tied = np.zeros((1, 9, 10)); tied_metrics, tied_K, _ = score_views(tied, [0], 10.)
    assert np.allclose(tied_metrics[..., 2], .1) and np.allclose(tied_metrics[..., 3], .5) and not tied_K.any()
    perfect = np.zeros((1, 9, 10)); perfect[..., 0] = 1.
    pm, pk, _ = score_views(perfect, [0], 10.)
    assert np.all(pm[..., 2:] == 1) and np.all(pm[..., 1] == 1) and not pk.any()
    split, rows, metadata, groups, folds = validate_split(split_path, manifest_path)
    return dict(status='D31_PURE_CPU_MATH_AND_ACTUAL_SPLIT_SCHEMA_CHECKS_PASSED',
        exact_S_over_12_trace_identity=True, zero_sum_contrasts=True,
        S_over_12_full_matrix_identity_on16_synthetic_coordinates=True,
        fixed_coordinate_permutation_preserves_all_repeat_Gram_and_spectrum=True,
        symmetric_KL_coefficient_one_eighth=True, zero_KL_and_fractional_ties=True,
        perfect_margin_top1_and_2AFC=True, T_rows=len(rows), T_concepts=len(groups),
        T_fold_counts=np.bincount(folds).tolist(), actual_split=ref(split_path), actual_manifest=ref(manifest_path),
        raw_EEG_values_read=False, GPU_initialized=False, new_training=0, new_images=0,
        maximum_synthetic_actual_Gram_relative_error=float(proof['actual_Gram_scaled_relative_error'].max()))


def rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Linux peak RSS required')


class Budget:
    def __init__(self, seconds, output):
        self.started=time.monotonic(); self.seconds=seconds; self.output=output
        self.base=rss(); self.peak=self.base; self.gpu=0; self.stop=False; self.torch=None
        signal_control.signal(signal_control.SIGTERM, lambda *_: setattr(self, 'stop', True))
        signal_control.signal(signal_control.SIGINT, lambda *_: setattr(self, 'stop', True))

    def check(self, stage):
        assert not self.stop and time.monotonic()-self.started < self.seconds, 'Time/stop protection: '+stage
        self.peak=max(self.peak,rss()); assert self.peak-self.base <= 10<<30
        if self.torch is not None and self.torch.cuda.is_initialized():
            self.gpu=max(self.gpu,self.torch.cuda.max_memory_allocated()); assert self.gpu <= 20<<30
        if self.output.exists():
            assert sum(p.stat().st_size for p in self.output.iterdir() if p.is_file()) <= 128<<20
        return dict(seconds=time.monotonic()-self.started,peak_extra_CPU_bytes=self.peak-self.base,peak_GPU_bytes=self.gpu)


def host_proof(authorization):
    command=(Path('/proc')/str(os.getppid())/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py' in command
    uuid=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
    hosts=[item for item in authorization['authorized_hosts'] if item.get('GPU_UUID')==uuid]
    assert len(hosts)==1 and hosts[0]['host']=='CLOUD_LOGIN_HOST'
    assert authorization['existing_ordinary_baseline_bytes_immutable'] is True
    assert authorization['formal_old_D_training_resumed'] is False
    return dict(wrapper_PID=os.getppid(),wrapper_command=command,GPU_UUID=uuid,authorized_host=hosts[0])


def ordinary(c, gamma, beta, state, torch):
    F=torch.nn.functional; u=(c-beta)/gamma
    h=F.gelu(F.linear(u,state['trunk.0.weight'],state['trunk.0.bias']))
    direction=F.linear(h,state['direction.weight'],state['direction.bias'])
    direction=direction-direction.mean(-1,keepdim=True)
    r2=u.square().sum(-1,keepdim=True)
    assert bool((r2>1e-8).all())
    direction=direction-(direction*u).sum(-1,keepdim=True)/r2*u
    delta=.25*r2.sqrt()*direction/(r2+direction.square().sum(-1,keepdim=True)).sqrt()
    scale=r2.sqrt()/(u+delta).norm(dim=-1,keepdim=True)
    result=c+gamma*((scale-1)*u+scale*delta)
    assert torch.isfinite(result).all()
    return result


def digest_state(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(args):
    output=Path(args.output).absolute(); assert not output.exists() and not output.is_symlink()
    budget=Budget(args.max_seconds,output)
    spec_path=Path(args.spec).resolve(); spec=read(spec_path)
    assert spec['status']=='D31_LEAVEONE_DECISION_DIAGNOSTIC_LOCKED_BEFORE_GPU' and spec['seed']==1729
    assert type(spec['include_frozen_ordinary']) is bool
    for key,limit in [('maximum_GPU_bytes',20<<30),('maximum_extra_CPU_bytes',10<<30),('maximum_output_bytes',128<<20)]:
        assert spec.get(key,limit)==limit
    pins={}
    for item in spec['source_pins']:
        path=checked(item); pins[str(path.resolve())]=item['sha256']
    for name in (Path(__file__).name,'brainhive_reversible_eeg_20261028_v1.py'):
        assert pins.get(str((ROOT/name).resolve()))==sha(ROOT/name),name
    for key in ('D30_completion','split','image_manifest'):
        assert pins.get(str(checked(spec[key]).resolve()))==spec[key]['sha256'],key
    done=read(checked(spec['D30_completion']))
    assert done['status']=='D30_ALL9_MATCHED_ADDITIONAL25_EPOCH_T_FITS_AND_TRUE_OOF_COMPLETE'
    assert done['new_training_updates']==2025 and done['new_query_exposures']==1984500
    assert done['original_ordinary_changed'] is False and done['formal_D_resumed'] is False
    for key in ('spec','source','model_source','role_rows','model_bundle','OOF_T_native'):
        checked(done[key])
    d30=read(done['spec']['path'])
    assert d30['status']=='D30_REVERSIBLE_EEG_SPEC_LOCKED'
    for item in d30['source_pins']:
        checked(item)
    physical=read(checked(d30['physical_R4_input_contract']))
    native=read(checked(d30['native_input_contract'])); assert sha(d30['native_input_contract']['path'])==NATIVE_SHA
    assert physical['status']=='PHYSICAL_R4_T_INPUTS_READY' and physical['export_shape']==[N,4,17,250]
    assert physical['mean_replay_exact'] is True and physical['input_contract_sha256']==NATIVE_SHA
    authorization=read(checked(d30['extra_authorization'])); host=host_proof(authorization)
    assert authorization['maximum_GPU_memory_bytes_per_job']==20<<30
    assert authorization['maximum_extra_CPU_memory_bytes_per_job']==10<<30
    refs={Path(item['path']).name:item for item in done['actual_T_inputs']}
    assert refs['trials.npy']['sha256']==TRIALS_SHA and refs['index.jsonl']['sha256']==INDEX_SHA and refs['targets.npy']['sha256']==TARGETS_SHA
    for name,item in refs.items():
        checked(item); assert physical['files'][name]['sha256']==item['sha256']
    split,rows,metadata,groups,folds=validate_split(checked(spec['split']),checked(spec['image_manifest']))
    index=[json.loads(line) for line in Path(refs['index.jsonl']['path']).read_text().splitlines()]
    assert len(index)==N
    for position,row in enumerate(index):
        assert row['row']==position and row['partition']=='T' and row['subject_id']=='sub-01'
        assert row['array_row']==rows[position] and row['concept_id']==metadata[rows[position]]['concept_id']
        assert row['stimulus_id']==metadata[rows[position]]['stimulus_id'] and row['fold']==int(folds[position])
    role_rows=read(done['role_rows']['path']); subject=native['subjects'][0]; assert subject['subject']=='sub-01'
    sources={item['role']:item for item in subject['sources']}
    method_names=METHODS if spec['include_frozen_ordinary'] else METHODS[:1]; nm=len(method_names)
    budget.check('ALL_INPUTS_AND_SOURCE_PINS_VERIFIED')
    output.parent.mkdir(parents=True,exist_ok=True);output.mkdir()
    try:
        write(output/'provenance.json',dict(spec=ref(spec_path),source=ref(__file__),D30_completion=spec['D30_completion'],
            D30_spec=done['spec'],input_refs=refs,split=spec['split'],image_manifest=spec['image_manifest'],host=host,
            methods=list(method_names),models=list(MODELS),views=list(VIEWS),mean4_convention='original D30 numpy float32 mean',
            mean3_formula='m-(x_r-m)/3 computed in float64 then FP32 model input; direct physical mean3 replay checked',
            classic_stability_formula_not_claimed_novel=True,Gaussian_iid_or_independent_R4_augmentation_claimed=False,
            primary='native_condition: real-delta symmetricKL(serial_reversible)-symmetricKL(native50)',
            models_or_images_admission=False,maximum_seconds=args.max_seconds))
        import torch
        import brainhive_reversible_eeg_20261028_v1 as V
        budget.torch=torch;torch.set_num_threads(2);torch.cuda.set_device(0)
        assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
        core_class=V.author_brain_class(checked(d30['author_models_source']),checked(d30['author_layers_source']))
        bundle=torch.load(done['model_bundle']['path'],map_location='cpu',weights_only=True)
        assert tuple(bundle['arms'])==NEW_ARMS and tuple(bundle['roles'])==('full','oof0','oof1') and bundle['spec']==done['spec']
        bank=np.load(done['OOF_T_native']['path'],mmap_mode='r',allow_pickle=False)
        trials=np.load(refs['trials.npy']['path'],mmap_mode='r',allow_pickle=False)
        targets=np.load(refs['targets.npy']['path'],mmap_mode='r',allow_pickle=False)
        assert bank.shape==(N,3,1024) and bank.dtype==np.float32
        assert trials.shape==(N,4,17,250) and trials.dtype==np.float32
        assert targets.shape==(N,1024) and targets.dtype==np.float32
        assert np.isfinite(targets).all()
        candidate_rows=np.empty((N,10),np.int64);labels=np.empty(N,np.int8)
        for selected in groups.values():
            for label,position in enumerate(selected):candidate_rows[position]=selected;labels[position]=label
        permutation=np.random.default_rng(1729).permutation(D)
        cosines=np.full((N,nm,4,9,10),np.nan,np.float64)
        vm=np.full((N,nm,4,9,4),np.nan,np.float64)
        K=np.full((N,nm,4,2),np.nan,np.float64);repeat_K=np.full((N,nm,4,2,4),np.nan,np.float64)
        grams=np.full((N,2,4,4),np.nan,np.float64);declared=np.full((N,4,4),np.nan,np.float64)
        perturb_norm=np.full((N,2,4),np.nan,np.float64);proof_values=np.full((N,6),np.nan,np.float64)
        spectra=np.full((N,2,4),np.nan,np.float64)
        cache_error=np.full((N,4),np.nan,np.float64);kappas=np.full(N,np.nan,np.float64);coverage=np.zeros((N,4),np.uint8)
        ordinary_state=None;freeze=None
        if nm==2:
            freeze=read(checked(spec['ordinary_freeze']));assert freeze['checkpoint']['sha256']==OLD_ORDINARY_SHA
            checked(freeze['checkpoint']);checked(freeze['source']);assert freeze['selected_strength']==1. and freeze['radius_rho_before_strength']==.25
            payload=torch.load(freeze['checkpoint']['path'],map_location='cpu',weights_only=False)
            assert payload['arm']=='ordinary' and payload['calibration']['selected_strength']==1.
            ordinary_state={key:value.cuda() for key,value in payload['model'].items()}
            assert set(ordinary_state)=={'trunk.0.weight','trunk.0.bias','direction.weight','direction.bias'}
        role_proofs=[]
        with torch.inference_mode():
            for role in ('oof0','oof1'):
                held=np.flatnonzero(folds==int(role[-1]));fit=np.flatnonzero(folds!=int(role[-1]))
                assert role_rows[role]['held_compact_rows']==held.tolist() and role_rows[role]['fit_compact_rows']==fit.tolist()
                source=sources[role];original_ref=d30['roles'][role]['checkpoint'];cache_ref=d30['roles'][role]['condition_cache']
                assert original_ref['sha256']==source['checkpoint_sha256'] and cache_ref['sha256']==source['condition_cache_sha256']
                old=torch.load(checked(original_ref),map_location='cpu',weights_only=True); assert old['epochs']==25
                kappa=float(old['logit_scale'].double().exp());assert math.isfinite(kappa) and 0<kappa<1e6
                kappas[held]=kappa
                cache=torch.load(checked(cache_ref),map_location='cpu',weights_only=True,mmap=True)
                assert cache['model_sha256']==source['checkpoint_sha256']
                routes={row['array_row']:row['cache_row'] for row in source['groups']['component_train']}
                assert set(routes)=={index[i]['array_row'] for i in held}
                for mi,name in enumerate(MODELS):
                    model=V.from_core_state(core_class,old['model'],NEW_ARMS[0]) if mi==0 else V.from_architecture_state(core_class,bundle['models'][role][NEW_ARMS[mi-1]],NEW_ARMS[mi-1])
                    model.eval().requires_grad_(False);before=digest_state(model.state_dict());model.cuda()
                    gamma,beta=model.affine();assert gamma.shape==beta.shape==(1024,) and bool((gamma.abs()>1e-6).all())
                    expected_affine=(dict(gamma=old['model']['block.layers.0.norm.weight'],beta=old['model']['block.layers.0.norm.bias']) if mi==0 else bundle['affines'][role][NEW_ARMS[mi-1]])
                    assert torch.equal(gamma.cpu(),expected_affine['gamma']) and torch.equal(beta.cpu(),expected_affine['beta'])
                    for lo in range(0,len(held),256):
                        selected=held[lo:lo+256];views,proof=make_views(np.asarray(trials[selected]),permutation)
                        if mi==0:
                            declared[selected]=proof['declared_gram'];grams[selected]=proof['actual_gram'];perturb_norm[selected]=proof['perturbation_norms']
                            spectra[selected]=proof['actual_spectra']
                            proof_values[selected]=np.stack([proof[key] for key in ('declared_Gram_relative_error','actual_Gram_scaled_relative_error','delta_sum_L2','physical_mean3_relative_error','declared_spectrum_relative_error','actual_spectrum_scaled_relative_error')],axis=-1)
                        prediction=model(torch.from_numpy(views.reshape(-1,D)).cuda()).float()
                        assert torch.isfinite(prediction).all()
                        base=prediction.reshape(len(selected),9,1024)[:,0].cpu().numpy()
                        if mi==0:
                            cr=[routes[index[i]['array_row']] for i in selected]
                            assert [int(cache['array_rows'][r]) for r in cr]==[index[i]['array_row'] for i in selected]
                            expected=cache['condition'][cr].numpy()
                        else:expected=np.asarray(bank[selected,mi-1])
                        errors=np.linalg.norm(base.astype(np.float64)-expected,axis=-1)/np.maximum(np.linalg.norm(expected.astype(np.float64),axis=-1),1e-20)
                        assert errors.max()<3e-5,'Full mean4 saved-bank replay failed'
                        cache_error[selected,mi]=errors
                        target=targets[candidate_rows[selected]].astype(np.float64)
                        target/=np.maximum(np.linalg.norm(target,axis=-1,keepdims=True),1e-20)
                        for method in range(nm):
                            values=prediction if method==0 else ordinary(prediction,gamma,beta,ordinary_state,torch)
                            a=values.cpu().numpy().astype(np.float64).reshape(len(selected),9,1024)
                            a/=np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-20)
                            score=np.einsum('bvd,bjd->bvj',a,target,optimize=True)
                            per,k,repeated=score_views(score,labels[selected],kappa)
                            cosines[selected,method,mi]=score;vm[selected,method,mi]=per;K[selected,method,mi]=k;repeat_K[selected,method,mi]=repeated
                        coverage[selected,mi]+=1
                        event=dict(role=role,model=name,completed=int(lo+len(selected)),expected=len(held),**budget.check('FORWARD_HELD_T_VIEWS'))
                        write(output/'progress.json',event)
                        with (output/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event)+'\n')
                        del prediction
                    assert digest_state(model.state_dict())==before,'Read-only model changed'
                    assert not any(parameter.requires_grad for parameter in model.parameters())
                    del model;gc.collect();torch.cuda.empty_cache()
                role_proofs.append(dict(role=role,FIT=len(fit),held=len(held),fixed_original25_kappa=kappa,
                    old_checkpoint=original_ref,old_cache=cache_ref,only_held_EEG_values_forwarded=True))
                del old,cache
        assert (coverage==1).all() and np.isfinite(cosines).all() and np.isfinite(vm).all() and np.isfinite(K).all()
        # Fixed summary axes: Kreal,Kpermuted, 4 base metrics, 4 real-view means,
        # and 4 control-view means. Every model is reported; none is selected.
        query=np.concatenate((K,vm[:,:,:,0,:],vm[:,:,:,1:5,:].mean(3),vm[:,:,:,5:9,:].mean(3)),axis=-1)
        concept=np.stack([query[selected].mean(0) for selected in groups.values()])
        paired=concept[:,:,2]-concept[:,:,1]  # serial minus matched native50
        interaction=paired[:,:,0]-paired[:,:,1]
        rng=np.random.default_rng(1729);draws=np.empty((1000,nm,4,14));paired_draws=np.empty((1000,nm,14));interaction_draws=np.empty((1000,nm))
        for bi in range(1000):
            selected=rng.integers(0,NC,size=NC);draws[bi]=concept[selected].mean(0)
            paired_draws[bi]=paired[selected].mean(0);interaction_draws[bi]=interaction[selected].mean(0)
            if bi%100==0:budget.check('SHARED_CONCEPT_BOOTSTRAP')
        def interval(mean,bootstrap):return dict(mean=np.asarray(mean).tolist(),descriptive95_lower=np.quantile(bootstrap,.025,axis=0).tolist(),descriptive95_upper=np.quantile(bootstrap,.975,axis=0).tolist())
        result_path=output/'all_query_finite_decisions_and_bootstraps.npz'
        with result_path.open('xb') as stream:
            np.savez_compressed(stream,cosines=cosines,view_metrics=vm,per_query_K=K,per_repeat_symmetric_KL=repeat_K,
                input_declared_repeat_Gram=declared,input_actual_repeat_Gram=grams,perturbation_norms=perturb_norm,
                input_actual_repeat_Gram_spectrum=spectra,
                input_proof_values=proof_values,mean4_cache_relative_errors=cache_error,kappas=kappas,folds=folds,
                labels=labels,candidate_compact_rows=candidate_rows,permutation=permutation,
                per_concept=concept,serial_minus_native50=paired,
                bootstrap_means=draws,bootstrap_serial_minus_native50=paired_draws,
                bootstrap_true_minus_control_interaction=interaction_draws)
        budget.check('ALL_QUERY_ARRAYS_SAVED')
        final_refs=spec['source_pins']+d30['source_pins']+list(refs.values())+[spec['D30_completion']]
        final_refs += [done[key] for key in ('spec','source','model_source','role_rows','model_bundle','OOF_T_native')]
        for role in ('oof0','oof1'):
            final_refs += [d30['roles'][role][key] for key in ('checkpoint','condition_cache')]
        for item in final_refs:checked(item)
        budget.check('ALL_SOURCE_BANK_AND_CHECKPOINT_SHA_REVERIFIED')
        if freeze is not None:checked(freeze['checkpoint']);checked(freeze['source'])
        assert sha(spec_path)==read(output/'provenance.json')['spec']['sha256']
        primary=interval(paired[:,0,0].mean(),paired_draws[:,0,0])
        report=dict(status='D31_ALL13230_TRUE_OOF_FINITE_LEAVEONE_DECISIONS_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),
            source=ref(__file__),spec=ref(spec_path),D30_completion=spec['D30_completion'],role_proofs=role_proofs,
            axes=dict(method=list(method_names),model=list(MODELS),view=list(VIEWS),view_metric=list(VIEW_METRICS),
                summary_metric=['K_real','K_permuted']+['mean4_'+v for v in VIEW_METRICS]+['mean3_mean_'+v for v in VIEW_METRICS]+['control_mean_'+v for v in VIEW_METRICS]),
            T_queries=N,T_concepts=NC,OOF_coverage_min=int(coverage.min()),OOF_coverage_max=int(coverage.max()),
            all_models=interval(concept.mean(0),draws),serial_minus_native50=interval(paired.mean(0),paired_draws),
            serial_minus_native50_true_minus_control_K=interval(interaction.mean(0),interaction_draws),
            primary_native_serial_minus_native50_real_K=primary,
            primary_K_increase_mean_and_lower95_positive=bool(primary['mean']>0 and primary['descriptive95_lower']>0),
            mean4_replay_max_relative_error_by_model=cache_error.max(0).tolist(),
            repeat_Gram_max_declared_relative_error=float(proof_values[:,0].max()),
            repeat_Gram_max_actual_FP32_scaled_relative_error=float(proof_values[:,1].max()),
            repeat_spectrum_max_declared_relative_error=float(proof_values[:,4].max()),
            repeat_spectrum_max_actual_FP32_scaled_relative_error=float(proof_values[:,5].max()),
            actual_Gram_rounding_scale='norm(declared_Gram)+1e-10*norm(mean4)^2+1e-20',
            maximum_delta_sum_L2=float(proof_values[:,2].max()),maximum_direct_physical_mean3_relative_error=float(proof_values[:,3].max()),
            paired_bootstrap=dict(unit='all1323concepts',draws=1000,seed=1729,all_models_and_differences_share_indices=True,descriptive_only=True),
            output_arrays=ref(result_path),encoder_true_concept_OOF=True,
            old_ordinary_previously_fitted_on_all_T=True,whole_ordinary_pipeline_independent_OOF=False,
            original_ordinary_checkpoint_changed=False,model_weights_changed=False,new_training_updates=0,new_images=0,
            model_or_image_admission_written=False,automatic_following_fit_or_images=False,
            Dfit_Dcheck_V_or_official_values_read=False,formal_D_resumed=False,
            classic_finite_stability_formula_not_claimed_novel=True,
            Gaussian_iid_or_independent_R4_augmentation_claimed=False,
            decision='DIAGNOSTIC_ONLY_NO_MODEL_OR_IMAGE_ADMISSION_NO_WINNER_SELECTION',**budget.check('BEFORE_COMPLETION'))
        write(output/'completion.json',report);budget.check('COMPLETE')
        print(json.dumps(dict(status=report['status'],primary_native=primary,**budget.check('FINAL'))),flush=True)
    except BaseException as exc:
        write(output/'failure.json',dict(status='FAILED_D31_DIAGNOSTIC_PRESERVED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),
            seconds=time.monotonic()-budget.started,new_training_updates=0,new_images=0,model_or_image_admission_written=False))
        if (output/'completion.json').exists():os.replace(output/'completion.json',output/'failed_completion_preserved.json')
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec');parser.add_argument('--output',required=True)
    parser.add_argument('--max-seconds',type=float,default=600.);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--split');parser.add_argument('--image-manifest');args=parser.parse_args()
    if args.self_test:
        assert args.split and args.image_manifest and not Path(args.output).exists()
        result=self_test(args.split,args.image_manifest);result['source']=ref(__file__)
        write(args.output,result);print(json.dumps(result),flush=True)
    else:
        assert args.spec and 60<=args.max_seconds<=600
        run(args)


if __name__=='__main__':main()
