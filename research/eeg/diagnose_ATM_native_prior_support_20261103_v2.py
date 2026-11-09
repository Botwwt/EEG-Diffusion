"""Frozen, T-only native-prior support intervention; no fitting or image generation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import datetime
import fcntl
import gc
import importlib.util
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
ap.add_argument('--mode', choices=('sample', 'audit'), required=True)
args = ap.parse_args()
q = json.loads(Path(args.policy).read_bytes())
root = Path(q['output_root'])
started = time.monotonic()
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
sys.path.insert(0, str(Path(q['base_policy']['path']).parent))
import brainhive_relative_fixed_output_contract_20261103_v1 as K
ctx = K.setup(str(K.checked(q['base_policy'])))
assert q['host'] == 'old' and q['seed'] == 1729
assert q['status'] == 'REGISTERED_T_ONLY_NATIVE_PRIOR_CLIPPING_CONFIGURATION_DIAGNOSTIC'
assert q['queries'] == 13230 and q['maximum_output_bytes'] == 128 << 20
assert q['prior_steps'] == 50 and q['CFG'] == 5 and q['batch'] == 128
assert q['new_training_updates'] == q['new_images'] == 0
for pin in q['sources'].values():
    K.checked(pin)
parent = K.read(K.checked(q['parent_policy']))
done = K.read(K.checked(q['parent_completion']))
assert done['policy'] == q['parent_policy']
assert done['status'] == 'COMPLETE_SUB01_NATIVE_ATM_THREE_PRIORS_AND_GENUINE_OOF_BANKS'
previous = K.read(Path(ctx['s']['output_root']) / 'completion.json')
assert previous['status'] == 'COMPLETE_ONE_FROZEN_RELATIVE_RAW1_FULL400_DIAGNOSTIC'
for pin in q['closed_controllers']:
    assert not K.process(pin['PID'], pin['start_ticks'])['live']
inputs = K.read(K.checked(parent['legacy_inputs']))
subject = inputs['subjects']['sub-01']
split = K.read(K.checked(inputs['split']))
T = sorted(split['image_rows']['T'])
assert len(T) == 13230 and not set(T) & (set(split['image_rows']['D']) | set(split['image_rows']['V']))
banks = [v for v in done['banks'] if v['role'] in ('oof0', 'oof1')]
assert len(banks) == 2 and sorted(banks[0]['rows'] + banks[1]['rows']) == T
for bank in banks:
    assert bank['whole_encoder_and_prior_concept_OOF'] and bank['clip_sample'] is True
    for key in ('output', 'prior_final', 'permissions'):
        K.checked(bank[key])
    role = bank['role']
    K.checked(subject['models'][role]['checkpoint'])
    assert sorted(subject['models'][role]['export_array_rows']) == bank['rows']
    permissions = K.read(bank['permissions']['path'])
    assert permissions['export_is_T_OOF'] and permissions['training_concepts_exclude_export']
    assert set(permissions['export_rows']) == set(bank['rows'])
    assert not set(permissions['training_rows']) & set(bank['rows'])
    assert not {row//10 for row in permissions['training_rows']} & {row//10 for row in bank['rows']}
K.checked(parent['native_prior_source'])
K.checked(parent['original_visual_target'])
K.checked(subject['input'])


def guard(starting=False):
    assert time.monotonic() - started < 855
    base = K.resources(ctx, starting=starting)
    used = ctx['C'].allocated(q['output_root']) + ctx['C'].allocated(q['code_root'])
    assert used <= q['maximum_output_bytes']
    required = base['required_remaining_bytes'] + q['maximum_output_bytes'] - used
    assert base['free_bytes'] >= required
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0 < q['maximum_extra_CPU_bytes']
    return dict(inherited=base, own_used_bytes=used,
                required_remaining_bytes=required, free_bytes=base['free_bytes'])


def progress(phase, **values):
    K.atomic(root / 'progress.json', dict(phase=phase, seconds=time.monotonic()-started, **values))


before = guard(starting=True)
import torch
torch.set_num_threads(2)
if args.mode == 'sample':
    repo = Path(ctx['relative']['repository_root'])
    job = K.read(repo / 'audit' / (q['job_name'] + '_job.json'))
    assert job['pid'] == os.getpid() and job['wrapper_pid'] == os.getppid() and job['status'] == 'RUNNING'
    lock = (repo / 'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == q['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(q['maximum_GPU_bytes'] / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0, '/root/atm_official_20261016')
    assert K.ref('/root/atm_official_20261016/models/atms.py')['sha256'] == parent['native_encoder_source_sha256']
    from models.atms import ATMS
    from atm_native_prior_support_sampler_20261103_v2 import sample
    spec = importlib.util.spec_from_file_location('unchanged_native_prior_source', parent['native_prior_source']['path'])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    raw = torch.load(subject['input']['path'], map_location='cpu', weights_only=True, mmap=True)
    assert raw.shape == (16540, 63, 250)
    output = np.load(root/'unclipped_T.npy', mmap_mode='r+', allow_pickle=False)
    prefix = np.load(K.checked(q['preserved_first128']), allow_pickle=False)
    assert output.shape == (13230,1024) and prefix.shape == (128,1024)
    first_rows = banks[0]['rows'][:128]
    original_indices = [T.index(row) for row in first_rows]
    assert np.array_equal(output[original_indices], prefix)
    assert np.isfinite(output).all(1).sum() == 128
    row_index = {row:i for i,row in enumerate(T)}
    parity = []
    for bank in banks:
        role = bank['role']
        encoder = ATMS().eval().requires_grad_(False)
        encoder.load_state_dict(torch.load(subject['models'][role]['checkpoint']['path'], map_location='cpu', weights_only=True, mmap=True), strict=True)
        encoder.cuda()
        prior = native.DiffusionPriorUNet(cond_dim=1024, dropout=.1).eval().requires_grad_(False)
        state = torch.load(bank['prior_final']['path'], map_location='cpu', weights_only=True)
        assert state['epochs'] == 150
        prior.load_state_dict(state['model'], strict=True)
        del state
        prior.cuda()
        old = np.load(bank['output']['path'], mmap_mode='r', allow_pickle=False)
        assert old.shape == (len(bank['rows']),1024)
        for lo in range(0, len(bank['rows']), 128):
            rows = bank['rows'][lo:lo+128]
            with torch.inference_mode():
                c = encoder(raw[rows].cuda(), torch.ones(len(rows), device='cuda', dtype=torch.long)).float()
            seeds = [1729+row for row in rows]
            if lo == 0:
                clipped, clipped_trace = sample(prior, c, seeds, clip_sample=True, trace=True)
                assert np.array_equal(clipped.cpu().numpy(), old[:len(rows)]), 'Original batch128 replay must be byte-exact before intervention'
            value, unclipped_trace = sample(prior, c, seeds, clip_sample=False, trace=(lo == 0))
            result = value.cpu().numpy()
            indices = [row_index[row] for row in rows]
            if role == banks[0]['role'] and lo == 0:
                assert np.array_equal(result, prefix), 'Original first128 unclipped prefix must remain byte-exact'
            else:
                output[indices] = result
            if lo == 0:
                condition_path = root/(role+'_first128_conditions.npy')
                if condition_path.exists():assert np.array_equal(np.load(condition_path), c.cpu().numpy())
                else:np.save(condition_path, c.cpu().numpy(), allow_pickle=False)
                for label, trace in (('clipped',clipped_trace),('unclipped',unclipped_trace)):
                    path = root/(role+'_'+label+'_batch128_trace_v2.npz')
                    assert not path.exists()
                    np.savez(path, **{k:np.asarray(v) for k,v in trace.items()})
                parity.append(dict(role=role, first128_clipped_byte_exact=True,
                    trace_is_first2_inside_same_actual_batch128=True,
                    preserved_first128_unclipped_exact=(role == banks[0]['role'])))
            output.flush()
            if lo % 1024 == 0:
                assert torch.cuda.max_memory_allocated() < q['maximum_GPU_bytes']
                guard()
                progress('T_ONLY_UNCLIPPED_SAMPLING', role=role, completed_queries=lo+len(rows), total_queries=len(bank['rows']))
        del prior,encoder,old
        gc.collect()
        torch.cuda.empty_cache()
    assert np.isfinite(output).all()
    del output
    for bank in banks:
        K.checked(bank['prior_final'])
        K.checked(bank['output'])
        K.checked(subject['models'][bank['role']]['checkpoint'])
    K.atomic(root/'sampling_completion.json', dict(status='COMPLETE_ALL13230_FROZEN_UNCLIPPED_T_WITH_ORIGINAL_BATCH_REPLAY',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), policy=K.ref(args.policy),
        output=K.ref(root/'unclipped_T.npy'), parity=parity, original_weights_and_banks_unchanged=True,
        seconds=time.monotonic()-started, peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
        resources_before=before, resources_after=guard(), new_training_updates=0,new_images=0))
    progress('COMPLETE_T_ONLY_SAMPLING_NO_IMAGES_OR_FITS')
else:
    assert not torch.cuda.is_initialized()
    sampling=K.read(root/'sampling_completion.json')
    assert sampling['policy']==K.ref(args.policy)
    generated=np.load(K.checked(sampling['output']),mmap_mode='r',allow_pickle=False).astype(np.float64)
    teacher=torch.load(parent['original_visual_target']['path'],map_location='cpu',weights_only=True,mmap=True)[T].numpy().astype(np.float64)
    original=np.empty_like(generated);lookup={row:i for i,row in enumerate(T)}
    for bank in banks:
        original[[lookup[row] for row in bank['rows']]]=np.load(bank['output']['path'],allow_pickle=False)
    assert generated.shape==original.shape==teacher.shape==(13230,1024)
    concepts={}
    for i,row in enumerate(T):concepts.setdefault(row//10,[]).append(i)
    assert len(concepts)==1323 and all(len(v)==10 for v in concepts.values())
    groups=list(concepts.values())
    target_unit=teacher/np.linalg.norm(teacher,axis=1,keepdims=True)
    arrays={};means={}
    for name,value in (('clipped',original),('unclipped',generated)):
        norm=np.linalg.norm(value,axis=1);unit=value/norm[:,None]
        cosine=np.sum(unit*target_unit,axis=1)
        top=np.empty(13230);afc=np.empty(13230)
        for ix in groups:
            scores=unit[ix]@target_unit[ix].T
            for j,i in enumerate(ix):
                top[i]=float(np.argmax(scores[j])==j)
                other=np.delete(scores[j],j);positive=scores[j,j]
                afc[i]=np.mean((positive>other)+.5*(positive==other))
        values=dict(target_cosine=cosine,within_concept10_top1=top,within_concept10_two_afc=afc,
                    raw_MSE=np.mean((value-teacher)**2,axis=1),norm=norm,
                    fraction_coordinates_outside_box=np.mean(abs(value)>1,axis=1))
        means[name]={k:float(v.mean()) for k,v in values.items()}
        for k,v in values.items():arrays[name+'__'+k]=v
    draw=np.random.default_rng(1729).integers(0,1323,(1000,1323));comparisons={}
    for key in values:
        delta=arrays['unclipped__'+key]-arrays['clipped__'+key]
        cm=np.array([delta[ix].mean() for ix in groups]);ci=np.quantile(cm[draw].mean(1),[.025,.975])
        arrays['difference__'+key]=delta
        comparisons[key]=dict(difference=float(delta.mean()),descriptive_concept_bootstrap95=ci.tolist(),positive_concepts=int((cm>0).sum()),concepts=1323)
    replay_error=0.
    for role in ('oof0','oof1'):
        for flag,label in ((True,'clipped'),(False,'unclipped')):
            trace=np.load(root/(role+'_'+label+'_batch128_trace_v2.npz'),allow_pickle=False)
            states,eps,noises=trace['states'].astype(np.float64),trace['predictions'].astype(np.float64),trace['variance_noises'].astype(np.float64)
            assert states.shape==(51,2,1024) and eps.shape==noises.shape==(50,2,1024)
            alpha=trace['alphas_cumprod'].astype(np.float64);ts=trace['timesteps'];assert np.array_equal(ts,np.arange(980,-1,-20))
            for j,t in enumerate(ts):
                at=alpha[t];prev=alpha[t-20] if t>=20 else 1.;current=at/prev;beta=1-current
                clean=(states[j]-np.sqrt(1-at)*eps[j])/np.sqrt(at)
                if flag:clean=np.clip(clean,-1,1)
                prediction=np.sqrt(prev)*beta/(1-at)*clean+np.sqrt(current)*(1-prev)/(1-at)*states[j]
                if t>0:prediction+=np.sqrt(max(1e-20,(1-prev)/(1-at)*beta))*noises[j]
                replay_error=max(replay_error,float(np.max(abs(prediction-states[j+1])/(1+abs(prediction)))))
    assert replay_error<1e-5
    np.savez(root/'all13230_per_query_metrics.npz',**arrays)
    result=dict(status='COMPLETE_ALL_T_FROZEN_PRIOR_CLIP_CONFIGURATION_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        policy=K.ref(args.policy),queries=13230,concepts=1323,means=means,fixed_unclipped_minus_clipped=comparisons,
        per_query=K.ref(root/'all13230_per_query_metrics.npz'),all200_DDPM_step_CPU64_scaled_error=replay_error,
        original_13230_cache_reused=True,new_training_updates=0,new_images=0,read_D_or_V_effects=False,
        no_automatic_new_fit_or_images=True,not_a_new_second_innovation=True,
        scope='sub01 physicalR4 genuine encoder+prior concept-OOF T13230; one fixed prior sample each; not image gain or officialR80',
        seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
        resources_before=before,resources_after=guard(),CUDA_initialized=torch.cuda.is_initialized())
    assert not result['CUDA_initialized']
    K.atomic(root/'readout.json',result)
    progress('COMPLETE_ALL_T_CPU_AUDIT')
