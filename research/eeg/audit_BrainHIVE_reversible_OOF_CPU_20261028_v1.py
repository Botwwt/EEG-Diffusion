"""CPU readout of D30: three matched architectures crossed with old ordinary.

The original25epoch encoders and old ordinary are additional fixed references.
No model is fitted, no raw EEG tensor values are read, no CUDA is initialised,
and no image-admission artifact is written. The encoder bank is true concept
OOF; applying ordinary, which previously used allT, is not full-system OOF.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np

N, DIM = 13230, 1024
ARCH = ('native_continuation', 'serial_reversible', 'parallel_matched')
METHOD = ('native_condition', 'frozen_old_ordinary')
METRICS = ('F_MSE', 'F_cosine', 'same_concept_Top1', 'same_concept_2AFC')
PAIRS = (('serial_minus_native_continuation', 1, 0), ('serial_minus_parallel_matched', 1, 2), ('parallel_minus_native_continuation', 2, 0))
ORIGINAL_CONTRACT_SHA = 'bb6e38350282af7b37ece04689adcec7a5f87dccf68de5968058433faece0597'
OLD_ORDINARY_CP_SHA = 'a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'
OLD_ORDINARY_SOURCE_SHA = '2b93c0644bde268ae876f0e2b722106852b25910fedc1cffc97820e96cadf5d7'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def checked(item):
    p = Path(item['path'])
    need(p.is_file() and sha(p) == item['sha256'], 'Source SHA differs: ' + str(p))
    if 'bytes' in item:
        need(p.stat().st_size == item['bytes'], 'Source byte count differs')
    return p


def json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def rss_peak():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Linux VmHWM required for actual CPU audit')


def primitive_ordinary(condition, gamma, beta, state, torch):
    """Independent CPU primitive replay, without importing component classes."""
    F = torch.nn.functional
    c = torch.from_numpy(np.array(condition, dtype=np.float32, copy=True))
    w = gamma.detach().cpu().float()
    b = beta.detach().cpu().float()
    need(w.shape == b.shape == (DIM,) and bool(torch.all(w.abs() > 1e-6)), 'Invalid actual model affine')
    with torch.inference_mode():
        u = (c - b) / w
        h = F.gelu(F.linear(u, state['trunk.0.weight'], state['trunk.0.bias']))
        d = F.linear(h, state['direction.weight'], state['direction.bias'])
        d -= d.mean(-1, keepdim=True)
        norm = torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        r2 = norm.square()
        need(bool(torch.all(r2 > 1e-8)), 'Degenerate original condition')
        d -= (d * u).sum(-1, keepdim=True) / r2 * u
        displacement = .25 * r2.sqrt() * d / (r2 + d.square().sum(-1, keepdim=True)).sqrt()
        scale = norm / torch.linalg.vector_norm(u + displacement, dim=-1, keepdim=True)
        result = c + w * ((scale - 1.) * u + scale * displacement)
        need(bool(torch.isfinite(result).all()), 'Nonfinite ordinary output')
        # Check on the actual new coordinate affine, not an old/full surrogate.
        v = (result - b) / w
        relative_norm_error = float(((v.norm(dim=1) - u.norm(dim=1)).abs() / u.norm(dim=1)).max())
        cosine = (u.double() * v.double()).sum(1) / (u.double().norm(dim=1) * v.double().norm(dim=1))
        max_angle = float(torch.acos(cosine.clamp(-1., 1.)).max())
        need(relative_norm_error < 3e-6 and max_angle < math.atan(.25) + 3e-5, 'Ordinary fixed-radius/norm contract violated')
    return result.numpy(), dict(relative_norm_error=relative_norm_error, maximum_angle=max_angle)


def concept_metrics(values, target):
    """values [10,method,architecture,1024], exact fractional retrieval ties."""
    v = np.asarray(values, np.float64)
    t = np.asarray(target, np.float64)
    vn, tn = np.linalg.norm(v, axis=-1, keepdims=True), np.linalg.norm(t, axis=-1, keepdims=True)
    uv = np.divide(v, vn, out=np.zeros_like(v), where=vn > 0.)
    ut = np.divide(t, tn, out=np.zeros_like(t), where=tn > 0.)
    scores = np.einsum('imad,jd->imaj', uv, ut, optimize=True)
    correct = scores[np.arange(10), :, :, np.arange(10)]
    maximum = scores.max(-1)
    top1 = (correct == maximum) / (scores == maximum[..., None]).sum(-1)
    afc = ((correct[..., None] > scores).sum(-1) + .5 * (correct[..., None] == scores).sum(-1) - .5) / 9.
    mse = np.square(v - t[:, None, None]).mean(-1)
    return np.stack((mse, correct, top1, afc), axis=-1)


def self_check():
    truth = np.eye(10, DIM)
    p = np.broadcast_to(truth[:, None, None], (10, 2, 3, DIM)).copy()
    perfect, opposite, tie = (concept_metrics(x, truth) for x in (p, -p, np.zeros_like(p)))
    need(np.all(perfect[..., 0] == 0.) and np.all(perfect[..., 1:] == 1.), 'Perfect metric identity')
    need(np.all(opposite[..., 2:] == 0.), 'Opposite metric identity')
    need(np.all(tie[..., 2] == .1) and np.all(tie[..., 3] == .5), 'Fractional tie convention')
    return dict(perfect=True, opposite=True, exact_fractional_ties=True)


def interval(values, draws):
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    return dict(mean=np.asarray(values).tolist(), descriptive95_lower=lower.tolist(), descriptive95_upper=upper.tolist())


def primitive_front(value, state, torch, *, inverse=False, serial=True):
    """Author-free trained coupling replay for a small synthetic roundtrip."""
    F = torch.nn.functional
    x = value.reshape(-1, 17, 250)
    left, right = x[..., 0::2].reshape(len(x), 2125), x[..., 1::2].reshape(len(x), 2125)
    def block(v, stage, arm):
        base = 'front.stages.' + str(stage) + '.' + arm + '.'
        return F.linear(F.gelu(F.linear(v, state[base+'fc1.weight'].to(v.dtype), state[base+'fc1.bias'].to(v.dtype))), state[base+'fc2.weight'].to(v.dtype), state[base+'fc2.bias'].to(v.dtype))
    for stage in (range(3, -1, -1) if inverse else range(4)):
        if inverse:
            need(serial, 'Parallel control has no analytic inverse guarantee')
            right = right - block(left, stage, 'g')
            left = left - block(right, stage, 'f')
        else:
            previous_left = left
            left = left + block(right, stage, 'f')
            right = right + block(left if serial else previous_left, stage, 'g')
    return torch.stack((left.reshape(-1,17,125), right.reshape(-1,17,125)), dim=-1).reshape(-1,4250)


def run(completion_path, output, freeze_path, max_seconds):
    start, initial_rss = time.monotonic(), rss_peak()
    need(not output.exists(), 'Preserve prior audit output; choose a new directory')
    output.mkdir(parents=True)
    stage = 'INPUTS'
    try:
        done_ref = ref(completion_path)
        done = read(completion_path)
        need(done['status'] == 'D30_ALL9_MATCHED_ADDITIONAL25_EPOCH_T_FITS_AND_TRUE_OOF_COMPLETE', 'D30 fit incomplete')
        need(tuple(done['arms']) == ARCH and tuple(done['roles']) == ('full','oof0','oof1'), 'Unexpected arm/role axes')
        need(done['OOF_shape'] == [N,3,DIM] and done['OOF_coverage_min'] == done['OOF_coverage_max'] == 1, 'Incomplete OOF denominator')
        need(done['original_ordinary_changed'] is False and done['formal_D_resumed'] is False and done['new_images'] == 0 and done['automatic_images_started'] is False, 'Unexpected old-weight mutation or image claim')
        need(done['fixed_cached_F'] is True and done['original_stochastic_teacher_replayed'] is False and done['no_Dfit_or_held_metric_checkpoint_selection'] is True, 'Training objective or selection permission differs')
        paths = {key: checked(done[key]) for key in ('spec','source','model_source','role_rows','model_bundle','OOF_T_native')}
        spec = read(paths['spec'])
        need(spec['status'] == 'D30_REVERSIBLE_EEG_SPEC_LOCKED' and spec['epochs'] == 25 and spec['seed'] == 1729 and spec['batch_size'] == 1024, 'D30 fixed training spec changed')
        for pin in spec['source_pins']:
            checked(pin)
        pins = {(str(Path(p['path']).resolve()), p['sha256']) for p in spec['source_pins']}
        for key in ('source','model_source'):
            need((str(paths[key].resolve()),done[key]['sha256']) in pins, 'Unregistered model/fit source')
        nc_path = checked(spec['native_input_contract'])
        need(sha(nc_path) == ORIGINAL_CONTRACT_SHA, 'Original native contract changed')
        contract = read(nc_path)
        original_sources = {s['role']:s for s in contract['subjects'][0]['sources']}
        need(contract['subjects'][0]['subject'] == 'sub-01' and set(original_sources) == {'full','oof0','oof1'}, 'Original source role mismatch')
        physical = read(checked(spec['physical_R4_input_contract']))
        index_ref = spec.get('T_index', physical['files']['index.jsonl'])
        target_ref = spec['T_targets']
        index = [json.loads(x) for x in checked(index_ref).read_text(encoding='utf-8').splitlines()]
        need(len(index) == N and all(r['row'] == i and r['target_row'] == i and r['partition'] == 'T' and r['subject_id'] == 'sub-01' for i,r in enumerate(index)), 'Wrong compact T index')
        need(len({r['array_row'] for r in index}) == len({r['stimulus_id'] for r in index}) == N, 'Duplicate T source identities')
        group = {}
        for i,row in enumerate(index):group.setdefault(row['concept_id'],[]).append(i)
        need(len(group) == 1323 and all(len(rows) == 10 and len({index[i]['fold'] for i in rows}) == 1 for rows in group.values()), 'Incomplete same-concept gallery or crossed folds')
        need({r['fold'] for r in index} == {0,1}, 'Missing OOF fold')
        bank = np.load(paths['OOF_T_native'], mmap_mode='r', allow_pickle=False)
        target = np.load(checked(target_ref), mmap_mode='r', allow_pickle=False)
        need(bank.shape == (N,3,DIM) and target.shape == (N,DIM) and bank.dtype == target.dtype == np.float32 and np.isfinite(bank).all() and np.isfinite(target).all(), 'Wrong bank dtype/shape/finite status')
        freeze = read(freeze_path)
        freeze_ref = ref(freeze_path)
        need(freeze['checkpoint']['sha256'] == OLD_ORDINARY_CP_SHA and freeze['source']['sha256'] == OLD_ORDINARY_SOURCE_SHA and freeze['selected_strength'] == 1., 'Wrong original ordinary baseline')
        checked(freeze['checkpoint']);checked(freeze['source'])
        import torch
        torch.set_num_threads(2)
        need(not torch.cuda.is_initialized(), 'Audit must remain CPU-only')
        def tick():
            need(time.monotonic()-start < max_seconds, 'Finite CPU readout deadline')
            need(rss_peak()-initial_rss < 10 << 30, 'Additional CPU memory cap')
            need(not torch.cuda.is_initialized(), 'Unexpected CUDA operation')
        mathematical = self_check()
        ordinary_payload = torch.load(freeze['checkpoint']['path'], weights_only=False, map_location='cpu')
        need(ordinary_payload['arm'] == 'ordinary' and ordinary_payload['calibration']['selected_strength'] == 1., 'Ordinary payload strength differs')
        ordinary_state = ordinary_payload['model']
        need(set(ordinary_state) == {'trunk.0.weight','trunk.0.bias','direction.weight','direction.bias'}, 'Ordinary structure changed')
        bundle = torch.load(paths['model_bundle'], weights_only=True, map_location='cpu')
        need(tuple(bundle['arms']) == ARCH and tuple(bundle['roles']) == ('full','oof0','oof1') and bundle['spec'] == done['spec'], 'Model bundle axes/spec differ')
        role_rows = read(paths['role_rows'])
        checkpoints = {(r['role'],r['arm']):r for r in done['checkpoints']}
        need(len(checkpoints) == 9, 'Missing final model')
        training_log_path = completion_path.parent/'training_log.jsonl'
        training_log_ref = ref(training_log_path)
        events = [json.loads(line) for line in training_log_path.read_text().splitlines()]
        need(len(events) == 225, 'Need all9arms x25 epoch logs')
        event_map = {(e['role'],e['arm'],e['epoch']):e for e in events}
        need(len(event_map) == 225, 'Duplicate epoch event')
        ordinary = np.empty((N,3,DIM),np.float32)
        original_native = np.empty((N,DIM),np.float32)
        original_ordinary = np.empty_like(original_native)
        coverage = np.zeros(N,np.int8)
        source_proofs, parameter_proofs, inverse_proofs = {}, [], []
        max_norm_error = max_angle = 0.
        expected_updates = expected_exposures = 0
        for role in ('full','oof0','oof1'):
            stage='ROLE_'+role;tick()
            selected = list(range(N)) if role=='full' else [r['row'] for r in index if r['fold'] != int(role[-1])]
            held = [] if role=='full' else [r['row'] for r in index if r['fold'] == int(role[-1])]
            need(role_rows[role]['fit_compact_rows'] == selected and role_rows[role]['held_compact_rows'] == held, 'Role FIT/held selection differs')
            need(role_rows[role]['fit_array_rows'] == [index[i]['array_row'] for i in selected] and role_rows[role]['held_array_rows'] == [index[i]['array_row'] for i in held], 'Role source-row route differs')
            source = original_sources[role]
            for kind in ('checkpoint','condition_cache'):
                expected=dict(path=source[kind+'_path'],sha256=source[kind+'_sha256'])
                need(spec['roles'][role][kind]['sha256'] == expected['sha256'], 'Initial source SHA differs')
                checked(expected)
            original = torch.load(source['checkpoint_path'],weights_only=True,map_location='cpu')
            need(original['epochs'] == 25, 'Original encoder epoch budget differs')
            updates=25*math.ceil(len(selected)/1024)
            original_parent = read(checked(dict(path=source['parent_registration_path'],sha256=source['parent_registration_sha256'])))
            need(set(original_parent['training_array_rows']) == {index[i]['array_row'] for i in selected} and set(original_parent['excluded_oof_rows']) == {index[i]['array_row'] for i in held}, 'Original encoder FIT permission differs')
            checkpoint_rng_reference = None
            for ai,arm in enumerate(ARCH):
                tick();record=checkpoints[(role,arm)];p=checked(record['checkpoint'])
                need(record['updates']==updates and record['exposures']==25*len(selected) and record['additional_epochs']==25 and record['initial_epochs']==25 and record['unique_fit_T_rows']==len(selected) and record['paired_order_and_dropout_stream'] is True and record['trainable_logit_parameters']==1, 'Final arm budget proof differs')
                snapshot=torch.load(p,weights_only=True,map_location='cpu')
                need(snapshot['role']==role and snapshot['arm']==arm and snapshot['epoch']==25 and snapshot['updates']==updates and snapshot['initial_checkpoint']['sha256']==source['checkpoint_sha256'] and snapshot['spec']==done['spec'], 'Final snapshot continuation identity differs')
                need(snapshot['scheduler']['last_epoch'] == updates, 'Final cosine scheduler budget differs')
                need(snapshot['optimizer']['state'] and all(int(item['step']) == updates and bool(torch.isfinite(item['exp_avg']).all()) and bool(torch.isfinite(item['exp_avg_sq']).all()) for item in snapshot['optimizer']['state'].values()), 'Final optimizer steps or moments differ')
                checkpoint_rng = (snapshot['CPU_rng'],snapshot['shuffle_rng'],snapshot['CUDA_rng'])
                if checkpoint_rng_reference is None:
                    checkpoint_rng_reference = checkpoint_rng
                else:
                    need(torch.equal(checkpoint_rng[0],checkpoint_rng_reference[0]) and torch.equal(checkpoint_rng[1],checkpoint_rng_reference[1]) and len(checkpoint_rng[2]) == len(checkpoint_rng_reference[2]) and all(torch.equal(a,b) for a,b in zip(checkpoint_rng[2],checkpoint_rng_reference[2])), 'Final paired CPU/shuffle/CUDA RNG states differ')
                model_state=bundle['models'][role][arm]
                need(set(snapshot['model'])==set(model_state) and all(torch.equal(snapshot['model'][key],value) for key,value in model_state.items()), 'Final snapshot/bundle tensor differs')
                count=sum(t.numel() for t in model_state.values())
                need(count==(6452224 if ai==0 else 10822248) and count==record['parameters'], 'Architecture parameter count differs')
                need(all(bool(torch.isfinite(t).all()) for t in model_state.values()), 'Nonfinite final model weights')
                affine=bundle['affines'][role][arm]
                need(torch.equal(affine['gamma'],model_state['core.block.layers.0.norm.weight']) and torch.equal(affine['beta'],model_state['core.block.layers.0.norm.bias']), 'Updated-role affine differs from actual final model')
                for epoch in range(1,26):
                    event=event_map[(role,arm,epoch)];baseline=event_map[(role,ARCH[0],epoch)]
                    need(event['updates']==epoch*math.ceil(len(selected)/1024) and event['seen']==len(selected) and event['coverage_min']==event['coverage_max']==1 and event['Dfit_read'] is False and event['fixed_cached_F'] is True, 'Training row coverage or permission differs')
                    need(event['order_sha256']==baseline['order_sha256'] and event['CUDA_dropout_stream_sha256']==baseline['CUDA_dropout_stream_sha256'], 'Matched order/dropout stream differs')
                    need(math.isfinite(event['mean_InfoNCE']) and math.isfinite(event['grad_norm_last']), 'Nonfinite epoch record')
                expected_updates+=updates;expected_exposures+=25*len(selected)
                parameter_proofs.append(dict(role=role,arm=arm,parameters=count,external_trainable_logit_parameters=1,additional_epochs=25,updates=updates,all25epoch_order_dropout_and_coverage_matched=True,final_bundle_checkpoint_equal=True,actual_affine_from_final_state=True))
                if ai==1:
                    values=torch.from_numpy(np.random.default_rng(1729).normal(scale=.1,size=(4,4250)))
                    with torch.inference_mode():
                        transformed=primitive_front(values,model_state,torch,serial=True)
                        restored=primitive_front(transformed,model_state,torch,inverse=True,serial=True)
                    relative=float((restored-values).norm()/values.norm())
                    need(math.isfinite(relative) and relative<1e-8, 'Trained serial front failed finite float64 synthetic inverse replay')
                    inverse_proofs.append(dict(role=role,synthetic_probe_only=True,float64_trained_front_inverse_relative_error=relative,front_fc2_nonzero_values=sum(int((v!=0).sum()) for k,v in model_state.items() if k.startswith('front.') and '.fc2.' in k)))
                if held:
                    for lo in range(0,len(held),256):
                        rows=held[lo:lo+256];result,notes=primitive_ordinary(bank[rows,ai],affine['gamma'],affine['beta'],ordinary_state,torch);ordinary[rows,ai]=result
                        max_norm_error=max(max_norm_error,notes['relative_norm_error']);max_angle=max(max_angle,notes['maximum_angle']);tick()
                del snapshot
            if held:
                cache=torch.load(source['condition_cache_path'],weights_only=True,map_location='cpu',mmap=True)
                need(cache['model_sha256']==source['checkpoint_sha256'], 'Original cache/source mismatch')
                routes={r['array_row']:r['cache_row'] for r in source['groups']['component_train']}
                need(set(routes)=={index[i]['array_row'] for i in held}, 'Original OOF coverage differs')
                old_gamma=original['model']['block.layers.0.norm.weight'];old_beta=original['model']['block.layers.0.norm.bias']
                for lo in range(0,len(held),256):
                    rows=held[lo:lo+256];cache_rows=[routes[index[i]['array_row']] for i in rows]
                    need(all(int(cache['array_rows'][cr])==index[i]['array_row'] for i,cr in zip(rows,cache_rows)), 'Original cache source-row mapping differs')
                    native=cache['condition'][cache_rows].numpy();original_native[rows]=native
                    old,_=primitive_ordinary(native,old_gamma,old_beta,ordinary_state,torch);original_ordinary[rows]=old
                coverage[held]+=1;del cache
            source_proofs[role]=dict(FIT=len(selected),held=len(held),held_concepts=len(held)//10,old_checkpoint_SHA256=source['checkpoint_sha256'],all50epochs_encoder_FIT_excludes_held_concepts=(role!='full'),final_checkpoint_not_metric_selected=True)
            del original
        need(np.all(coverage==1) and expected_updates==done['new_training_updates'] and expected_exposures==done['new_query_exposures'], 'Final global coverage/training budget differs')
        stage='COMPLETE_METRICS';tick()
        query=np.empty((N,2,3,4),np.float64)
        old_query=np.empty((N,2,4),np.float64)
        for ci,rows in enumerate(group.values()):
            values=np.stack((np.asarray(bank[rows]),ordinary[rows]),axis=1)
            query[rows]=concept_metrics(values,target[rows])
            old_values=np.stack((original_native[rows],original_ordinary[rows]),axis=1)[:,:,None,:]
            old_query[rows]=concept_metrics(old_values,target[rows])[:,:,0,:]
            if ci%32==0:tick()
        concept=np.stack([query[rows].mean(0) for rows in group.values()])
        old_concept=np.stack([old_query[rows].mean(0) for rows in group.values()])
        paired=np.stack([concept[:,:,a]-concept[:,:,b] for _,a,b in PAIRS],axis=2)
        ordinary_effect=concept[:,1]-concept[:,0]
        interaction=np.stack([paired[:,1,i]-paired[:,0,i] for i in range(len(PAIRS))],axis=1)
        versus_original=concept-old_concept[:,:,None]
        rng=np.random.default_rng(1729)
        draws=np.empty((1000,2,3,4));old_draws=np.empty((1000,2,4));paired_draws=np.empty_like(draws);effect_draws=np.empty((1000,3,4));interaction_draws=np.empty_like(effect_draws);reference_draws=np.empty_like(draws)
        for bi in range(1000):
            sample=rng.integers(0,len(group),size=len(group))
            draws[bi]=concept[sample].mean(0);old_draws[bi]=old_concept[sample].mean(0);paired_draws[bi]=paired[sample].mean(0);effect_draws[bi]=ordinary_effect[sample].mean(0);interaction_draws[bi]=interaction[sample].mean(0);reference_draws[bi]=versus_original[sample].mean(0)
        # Mean-component diagnostics are reported without treating them as SNR.
        mean_fraction=[]
        for values in (np.asarray(bank),ordinary):
            total=(values.astype(np.float64)**2).sum(-1).mean(0)
            centre=np.square(values.astype(np.float64).mean(0)).sum(-1)
            mean_fraction.append((centre/total).tolist())
        for pin in spec['source_pins']+[done[k] for k in paths]+[index_ref,target_ref,freeze['checkpoint'],freeze['source']]:checked(pin)
        need(ref(completion_path)==done_ref and ref(training_log_path)==training_log_ref and ref(freeze_path)==freeze_ref, 'Source changed while auditing')
        tick()
        with (output/'all_query_concept_metrics_and_bootstraps.npz').open('xb') as f:
            np.savez_compressed(f,per_query=query,per_concept=concept,old_reference_per_query=old_query,old_reference_per_concept=old_concept,paired=paired,ordinary_effect=ordinary_effect,interaction=interaction,versus_original=versus_original,bootstrap_means=draws,bootstrap_old=old_draws,bootstrap_paired=paired_draws,bootstrap_ordinary_effect=effect_draws,bootstrap_interaction=interaction_draws,bootstrap_vs_original=reference_draws,folds=np.asarray([r['fold'] for r in index],np.int8))
        result=dict(status='D30_COMPLETE_OOF_NATIVE_AND_FROZEN_ORDINARY_CPU_READOUT',UTC=datetime.now(timezone.utc).isoformat(),source=ref(__file__),completion=done_ref,spec=done['spec'],training_log=training_log_ref,ordinary_freeze=freeze_ref,axes=dict(method=list(METHOD),architecture=list(ARCH),metric=list(METRICS),paired_comparison=[r[0] for r in PAIRS]),T_queries=N,T_concepts=1323,role_source_proofs=source_proofs,all9_checkpoint_proofs=parameter_proofs,all225epochs_and_global_updates_verified=expected_updates,global_additional_exposures_verified=expected_exposures,OOF_coverage_min=int(coverage.min()),OOF_coverage_max=int(coverage.max()),metrics=interval(concept.mean(0),draws),original25epoch_frozen_references=interval(old_concept.mean(0),old_draws),paired=interval(paired.mean(0),paired_draws),old_ordinary_effect_by_architecture=interval(ordinary_effect.mean(0),effect_draws),ordinary_by_architecture_interactions=interval(interaction.mean(0),interaction_draws),versus_same_ordinary_state_original25epoch_references=interval(versus_original.mean(0),reference_draws),condition_overall_mean_energy_fraction=mean_fraction,ordinary_constraint_checks=dict(maximum_relative_norm_error=max_norm_error,maximum_global_angle=max_angle,old_strength=1.,old_rho=.25,new_affine_for_each_architecture_and_role=True),trained_serial_front_synthetic_inverse_checks=inverse_proofs,bootstrap=dict(unit='whole1323concepts',draws=1000,seed=1729,all_contrasts_share_indices=True,descriptive_intervals=True,training_seed_uncertainty_not_covered=True),old_ordinary_previously_trained_on_all_T=True,encoder_concept_OOF=True,whole_ordinary_pipeline_independent_OOF=False,new_GPU_jobs=0,GPU_initialized=False,raw_EEG_tensor_values_read=False,independent_encoder_forward_replay=False,Dfit_Dcheck_or_test_values_read=False,new_images=0,image_admission_written=False,automatic_images_authorized=False,decision='T_ONLY_DIRECTIONAL_READOUT_REQUIRES_REAL_IMAGE_AND_HELD_DEVELOPMENT_REVIEW',novelty_established=False,image_gain_established=False,all_source_SHA256_reverified=True,mathematical_self_check=mathematical,compact_results=ref(output/'all_query_concept_metrics_and_bootstraps.npz'),seconds=time.monotonic()-start,peak_RSS_bytes=rss_peak(),peak_extra_CPU_bytes=rss_peak()-initial_rss)
        json_new(output/'summary.json',result)
        print(json.dumps(dict(status=result['status'],seconds=result['seconds'],new_images=0)),flush=True)
        return result
    except BaseException as error:
        json_new(output/'failure.json',dict(status='D30_CPU_OOF_AUDIT_FAILED_PRESERVED',stage=stage,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start,no_automatic_retry=True))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ordinary-freeze',type=Path,default=Path(__file__).resolve().parent/'configs/ordinary_frozen_base_20261028_v1.json')
    p.add_argument('--max-seconds',type=int,default=240);a=p.parse_args();need(60<=a.max_seconds<=600,'Finite CPU budget60..600seconds required')
    run(a.completion,a.output,a.ordinary_freeze,a.max_seconds)


if __name__=='__main__':main()
