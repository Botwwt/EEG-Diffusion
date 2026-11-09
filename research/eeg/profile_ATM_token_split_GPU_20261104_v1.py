"""Finite native P-token split engineering: frozen EEGs, actual four-step images."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,hashlib,inspect,io,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);a=ap.parse_args();path=Path(a.policy);q=json.loads(path.read_bytes());assert q['status']=='LOCKED_NATIVE_IP_TOKEN_SPLIT_ENGINEERING_ONLY' and q['host']=='old'
sys.path.insert(0,str(Path(q['parent_policy']['path']).parent));import atm_mean_image_contract_20261104_v1 as C
c=C.setup(q['parent_policy']['path']);K=c['K'];K.checked(q['parent_policy']);main=c['up'];out=Path(q['output_root']);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==q['sources']['GPU'];before=C.resources(c,starting=True)
done=K.read(K.checked(q['parent_completion']));assert done['status']=='COMPLETE_SIX_ARM_T_MEAN_SHIFT_FULL400_IMAGE_DIAGNOSIS'
owner=K.read(Path(q['parent_policy']['path']).parent/'actual_launch.json');assert not K.process(owner['PID'],owner['start_ticks'])['live']
for p in q['sources'].values():K.checked(p)
job=K.read(c['repo']/'audit'/(q['job_name']+'_job.json'));assert job['status']=='RUNNING' and job['pid']==os.getpid()
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['parent']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(c['parent']['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import load_host,tensor_sha
from brainhive_native_direction_contract_20261103_v1 import tensor_digest
from atm_token_split_runtime_20261104_v1 import TokenSplitAudit
import atm_token_split_runtime_20261104_v1 as split_runtime
import atm_token_mean_split_20261104_v1 as split_math
assert K.ref(split_runtime.__file__)==q['sources']['runtime'] and K.ref(split_math.__file__)==q['sources']['math']
def used():return c['up']['cv']['ctx']['base']['ctx']['C'].allocated(q['project_root'])
def guard():
    assert time.monotonic()-started<850
    own=used();assert own<q['maximum_output_bytes'];fs=os.statvfs(q['project_root']);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+q['maximum_output_bytes']-own
    assert torch.cuda.max_memory_allocated()<c['parent']['maximum_GPU_bytes'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes']
audit=None
try:
    sys.path.insert(0,'/root/atm_official_20261016');from models.atms import ATMS
    T=main['T'].reshape(-1,10);ids=np.array(q['T_rows'],np.int64);donors=np.array(q['donor_T_rows'],np.int64);assert np.array_equal(ids,T[:8,0]) and np.array_equal(donors,T[:8,1])
    raw=torch.load(K.checked(main['q']['EEG']),map_location='cpu',weights_only=True,mmap=True);conditions={};errors={};model_pins={}
    for role in ('old','new'):
        pin=main['q']['encoder'] if role=='old' else K.read(main['out']/'fits/nce_continued/completion.json')['checkpoint'];model_pins[role]=pin
        loaded=torch.load(K.checked(pin),map_location='cpu',weights_only=True,mmap=True);state=loaded if role=='old' else loaded['model'];model=ATMS().cuda().eval().requires_grad_(False);model.load_state_dict(state,strict=True)
        rows=np.concatenate([ids,donors]) if role=='old' else ids
        with torch.inference_mode():gpu=model(raw[rows].cuda(),torch.ones(len(rows),device='cuda',dtype=torch.long)).float().cpu().numpy()
        model=model.cpu()
        with torch.inference_mode():cpu=model(raw[rows],torch.ones(len(rows),dtype=torch.long)).numpy()
        errors[role]=float(abs(cpu-gpu).max());assert errors[role]<5e-5;conditions[role]=gpu;del model,state,loaded;torch.cuda.empty_cache();guard()
    np.savez_compressed(out/'EEG_conditions_T8.npz',old=conditions['old'][:8],new=conditions['new'],old_donor=conditions['old'][8:],T_rows=ids,donor_rows=donors)
    loader=out/'loader';loader.mkdir();pipe,projection,prompt,pooled,timeids,diffusion,loader_proof=load_host(c['cv']['bq'],loader)
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
    assert hashlib.sha256(inspect.getsource(IPAdapterAttnProcessor2_0).encode()).hexdigest()==q['installed_IP_class_source_sha256']
    unet=tensor_digest(pipe.unet,half=True);vae=tensor_digest(pipe.vae,half=True)
    import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
    noise_fn=EA.randn_tensor;prepare_fn=pipe.prepare_latents;capture={}
    def cn(*args,**kwargs):
        value=noise_fn(*args,**kwargs);capture['ancestral'].append(tensor_sha(value));return value
    def ci(*args,**kwargs):
        value=prepare_fn(*args,**kwargs);capture['initial'].append(tensor_sha(value));return value
    EA.randn_tensor=cn;pipe.prepare_latents=ci
    def render(value,seed):
        global capture
        capture={'initial':[],'ancestral':[]}
        with torch.inference_mode():image=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,ip_adapter_image_embeds=[value[:,None].half()],num_inference_steps=4,guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
        pixels=np.asarray(image.convert('RGB')).copy();assert pixels.shape==(512,512,3) and len(capture['initial'])==1 and len(capture['ancestral'])==4
        return image,hashlib.sha256(pixels.tobytes()).hexdigest(),dict(capture)
    replay=[];baseline=np.load(K.checked(c['base']['BN_GPU']['conditions']),allow_pickle=False)
    for j,arm in [(0,'direct'),(1,'nce_continued')]:
        guard();value=torch.tensor(baseline[0,j][None].copy(),device='cuda');image,sha,noise=render(value,c['queries'][0]['image_seed']);ref=c['baseline'][arm][0]
        assert sha==ref['pixels_sha256'] and noise==ref['noise'];image.save(out/(arm+'_existing_q0_replay.webp'),format='WEBP',lossless=True,quality=100);replay.append(dict(arm=arm,pixels_sha256=sha,noise=noise))
    audit=TokenSplitAudit(pipe,projection);records=[];summary=[];reference={}
    variants=[('native_old','native_old',False),('native_new','native_new',False),('self_old','old_mean_new_center',True),('self_new','old_mean_new_center',True),('old_mean_new_center','old_mean_new_center',True),('new_mean_old_center','new_mean_old_center',True),('token_ensemble','token_ensemble',True),('wrong_old_mean','old_mean_new_center',True)]
    for i in range(8):
        with torch.inference_mode():
            x=torch.tensor(conditions['old'][i:i+1],device='cuda');y=torch.tensor(conditions['new'][i:i+1],device='cuda');wrong=torch.tensor(conditions['old'][i+8:i+9],device='cuda');p0=projection(x.half());p1=projection(y.half());pw=projection(wrong.half())
        for tag,mode,inspect_Q in variants:
            guard();oldtoken,newtoken=p0,p1
            if tag=='self_old':oldtoken=newtoken=p0
            elif tag=='self_new':oldtoken=newtoken=p1
            elif tag=='wrong_old_mean':oldtoken=pw
            base=x if tag in ('native_old','self_old','new_mean_old_center') else y
            audit.begin(tag,i,oldtoken,newtoken,mode,inspect_Q);image,sha,noise=render(base,1729+int(ids[i]));summary.append(audit.end())
            if tag in ('native_old','native_new'):reference[i,tag]=(sha,noise)
            if tag in ('self_old','self_new'):
                expected=reference[i,'native_old' if tag=='self_old' else 'native_new'];assert sha==expected[0] and noise==expected[1]
            assert noise==reference[i,'native_old'][1]
            file=out/f'q{i}_{tag}.webp';image.save(file,format='WEBP',lossless=True,quality=100);records.append(dict(query_index=i,array_row=int(ids[i]),donor_array_row=int(donors[i]) if tag=='wrong_old_mean' else None,tag=tag,mode=mode,source_current_query_EEG=True,
                donor_is_diagnostic_control_not_deployable_pair=True if tag=='wrong_old_mean' else False,image=K.ref(file),pixels_sha256=sha,noise=noise,UNet_NFE=4,optimizer_updates=0))
            K.write(out/'progress.json',dict(completed_engineering_images=len(records),target_images=64,tag=tag,query_index=i))
    assert len(records)==64 and len(audit.rows)==13440 and len(audit.cases)==32
    array_data={label+'__'+name:value for label,case in audit.cases.items() for name,value in case.items()};np.savez_compressed(out/'actual_QKV_cases.npz',**array_data);assert (out/'actual_QKV_cases.npz').stat().st_size<16<<20
    K.write(out/'actual_IP_rows.json',audit.rows);K.write(out/'images.json',records);K.write(out/'per_image_execution.json',summary);guard()
    assert tensor_digest(pipe.unet,half=True)==unet and tensor_digest(pipe.vae,half=True)==vae
    result=dict(status='PASS_NATIVE_IP_TOKEN_SPLIT_ENGINEERING_ONLY',policy=K.ref(path),source=K.ref(__file__),T_queries=8,existing_Dcheck_q0_native_pixel_replays=2,T_identity_pixel_replays=16,engineering_images=64,new_research_effect_images=0,new_optimizer_updates=0,
        actual_IP_rows=13440,all70_processors_four_times_each_audited_image=True,actual_QKV_saved_cases=32,CPU_EEG_forward_errors=errors,EEG_models=model_pins,all_frozen_host_weights_unchanged=True,
        maximum_native_GPU_SDPA_F64_scaled_error=max(v['scaled_native_GPU_vs_F64_SDPA_error'] for v in audit.rows),maximum_mean_swap_probability_drift=max(v['actual_mean_swap_probability_drift'] for v in audit.rows if v['probability_preservation_gate_applies']),
        maximum_ideal_common_shift_identity_error=max(v['ideal_common_shift_identity_error'] for v in audit.rows),FP16_probability_gate=.005,FP16_actual_attention_not_claimed_exactly_unchanged=True,
        EEG_forward_cost_deployable_split=2,extra_projection_work_not_free=True,extra_KV_linear_probes_engineering_only=True,no_style_semantic_separation_claim=True,not_novelty_or_image_efficacy=True,
        conditions=K.ref(out/'EEG_conditions_T8.npz'),case_arrays=K.ref(out/'actual_QKV_cases.npz'),case_rows=K.ref(out/'actual_IP_rows.json'),images=K.ref(out/'images.json'),loader=loader_proof,seconds=time.monotonic()-started,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)
    K.write(out/'completion.json',result);print(json.dumps({k:result[k] for k in ['status','seconds','engineering_images','actual_IP_rows','maximum_native_GPU_SDPA_F64_scaled_error','maximum_mean_swap_probability_drift']}),flush=True)
except BaseException as exc:
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),partial_evidence_retained=True,no_automatic_retry=True));raise
finally:
    if audit is not None:audit.close()
    if 'EA' in globals():EA.randn_tensor=noise_fn;pipe.prepare_latents=prepare_fn
