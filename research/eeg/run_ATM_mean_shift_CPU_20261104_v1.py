"""Bounded OLD-only full-T frozen means and fixed interventions; no CUDA or fit."""
import os
os.environ['OPENBLAS_NUM_THREADS']=os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,io,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('policy');a=ap.parse_args();p=Path(a.policy);q=json.loads(p.read_bytes())
assert q['status']=='LOCKED_FULL_T_NATIVE_EEG_MEAN_SHIFT_CAUSAL_PREPARATION' and q['host']=='old' and q['seed']==1729
assert q['maximum_output_bytes']==128<<20 and q['maximum_seconds']==900
sys.path.insert(0,str(Path(q['upstream_policy']['path']).parent));import atm_BN_image_contract_20261104_v1 as B
c=B.setup(q['upstream_policy']['path']);K=c['K'];K.checked(q['upstream_policy']);main=c['up'];out=Path(q['output_root']);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==q['sources']['CPU'];from atm_mean_shift_math_20261104_v1 import ARMS,METRICS,moments,interventions,score_within,decompose_delta
assert q['arms']==list(ARMS)
def used():return c['up']['cv']['ctx']['base']['ctx']['C'].allocated(q['project_root'])
cached_resources=None;last_resource_check=-1e30
def guard():
    global cached_resources,last_resource_check
    assert time.monotonic()-started<850
    if cached_resources is None or time.monotonic()-last_resource_check>60:
        cached_resources=B.resources(c);last_resource_check=time.monotonic()
    r=dict(cached_resources);fs=os.statvfs(q['project_root']);r['free_bytes']=fs.f_bavail*fs.f_frsize
    own=used();assert own<=q['maximum_output_bytes'] and r['free_bytes']>=r['required_including_this_project']+q['maximum_output_bytes']-own
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30
    return r
before=guard()
try:
    done=K.read(K.checked(q['upstream_completion']));assert done['status']=='COMPLETE_SIX_ARM_BN_RADIUS_FULL400_IMAGE_DIAGNOSIS'
    launch=K.read(Path(q['upstream_policy']['path']).parent/'actual_launch.json');assert not K.process(launch['PID'],launch['start_ticks'])['live']
    for event in done['events']:
        if event['GPU']:
            r=K.read(K.checked(event['receipt']));assert r['status']=='COMPLETED' and not K.process(r['pid'])['live'] and not K.process(r['wrapper_pid'])['live']
    for pin in q['sources'].values():K.checked(pin)
    import torch
    torch.set_num_threads(2);sys.path.insert(0,'/root/atm_official_20261016');from models.atms import ATMS
    raw=torch.load(K.checked(main['q']['EEG']),map_location='cpu',weights_only=True,mmap=True)
    full_visual=torch.load(K.checked(main['q']['visual_target']),map_location='cpu',weights_only=True,mmap=True)
    T=main['T'];assert len(T)==13230 and np.array_equal(T.reshape(-1,10)//10,np.repeat((T[::10]//10)[:,None],10,axis=1))
    assert raw.shape==(16540,63,250) and full_visual.shape==(16540,1024)
    T_target=full_visual[T].numpy().astype(np.float64);means={'visual':moments(T_target)};states={};forward_refs={};query_errors={};cert={}
    old400=np.load(K.checked(c['BN_GPU']['conditions']),allow_pickle=False)[:,:3]
    for j,arm in enumerate(ARMS[:3]):
        guard();model=ATMS().eval().requires_grad_(False)
        if arm=='direct':pin=main['q']['encoder'];state=torch.load(K.checked(pin),map_location='cpu',weights_only=True,mmap=True)
        else:
            fit=K.read(main['out']/'fits'/arm/'completion.json');assert fit['updates']==830 and fit['exposures']==132300
            pin=fit['checkpoint'];state=torch.load(K.checked(pin),map_location='cpu',weights_only=True,mmap=True)['model']
        model.load_state_dict(state,strict=True);assert sum(v.numel() for v in model.parameters())==3200413
        values=np.empty((len(T),1024),np.float32)
        with torch.inference_mode():
            for lo in range(0,len(T),128):
                guard();ids=T[lo:lo+128];values[lo:lo+len(ids)]=model(raw[ids],torch.ones(len(ids),dtype=torch.long)).numpy()
                K.write(out/'progress.json',dict(arm=arm,T_complete=lo+len(ids),total_T=len(T),CUDA_calls=0,new_optimizer_updates=0))
            errs=[]
            for lo in range(0,400,128):
                ids=[r['array_row'] for r in main['queries'][lo:lo+128]];v=model(raw[ids],torch.ones(len(ids),dtype=torch.long)).numpy();errs.extend(np.max(abs(v-old400[lo:lo+len(ids),j]),1).tolist())
        assert max(errs)<5e-5;query_errors[arm]=errs;states[arm]=values;means[arm]=moments(values);forward_refs[arm]=pin
        cert[arm]=dict(input_array_rows_SHA256=hashlib.sha256(T.tobytes()).hexdigest(),full_T_output_FP32_bytes_SHA256=hashlib.sha256(values.tobytes()).hexdigest(),all400_old_GPU_conditions_CPU_max_error=max(errs),all13230_frozen_CPU_forwards=True)
        del model,state
    mu={name:item['mean'] for name,item in means.items()};candidate=interventions(states['direct'],states['nce_continued'],states['conditional'],mu)
    matrices,arrays=score_within(candidate,T_target);decompositions={a:decompose_delta(states['direct'],states[a]) for a in ARMS[1:3]}
    restored=interventions(old400[:,0],old400[:,1],old400[:,2],mu).astype(np.float32)
    assert np.isfinite(restored).all() and np.linalg.norm(restored,axis=-1).max()<10*np.linalg.norm(T_target,axis=1).max()
    for arm in range(3,6):assert np.allclose(candidate[:,arm]-candidate[:,1],(candidate[:,arm]-candidate[:,1])[0],rtol=0,atol=2e-14)
    payload={'T_array_rows':T,'within_concept_similarity_matrices':matrices}
    for name,item in means.items():
        for key,v in item.items():payload[name+'__'+key]=v
    for key,v in arrays.items():payload['metric__'+key]=v
    for key,v in decompositions.items():
        payload[key+'__mean_shift']=v.pop('mean_shift')
    K.save_array(out/'conditions400.npy',restored);np.savez_compressed(out/'T_statistics.npz',**payload);guard()
    source_no_effect_reads=dict(training_inputs=[main['q']['EEG'],main['q']['visual_target']],models=forward_refs,T_only_moments=True,query400_used_only_for_existing_CPU_replay_and_locked_inputs=True,Dcheck_target_images_features_or_image_effects_NOT_read=True,inference_only_current_EEG_plus_fixed_T_statistics=True)
    result=dict(status='COMPLETE_FULL_T_NATIVE_EEG_MEAN_SHIFT_PREPARATION_NO_IMAGES',UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),policy=K.ref(p),source=K.ref(__file__),N=13230,concepts=1323,parameters_each=3200413,new_optimizer_updates=0,new_images=0,CUDA_calls=0,
        conditions=K.ref(out/'conditions400.npy'),statistics=K.ref(out/'T_statistics.npz'),all_T_encoder_certificates=cert,all400_existing_CPU_errors=query_errors,delta_decompositions=decompositions,
        mean_norms={key:float(np.linalg.norm(v)) for key,v in mu.items()},means={arm:{key:float(v[:,j].mean()) for key,v in arrays.items()} for j,arm in enumerate(ARMS)},source_scope=source_no_effect_reads,
        exact_difference_invariance_before_FP32_rounding=True,centering_classical_NOT_novel_method=True,no_candidate_selection_or_automatic_fit_or_images=True,seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_after=guard())
    K.write(out/'completion.json',result);print(json.dumps({k:result[k] for k in ('status','seconds','delta_decompositions','mean_norms','means')}),flush=True)
except BaseException as exc:
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),partial_evidence_retained=True,no_automatic_retry=True));raise
