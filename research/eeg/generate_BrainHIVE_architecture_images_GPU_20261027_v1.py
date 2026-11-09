"""One architecture method, actual native images, fixed80/240/400 prefixes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,inspect,json,signal,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import brainhive_architecture_campaign_20261027_v1 as M
import brainhive_gain_calibration_20261027_v1 as U
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N

def deterministic_sample(pipe,prompts,tokens,seed):
    module=inspect.getmodule(type(pipe.scheduler));original=module.randn_tensor;noise=[]
    def observed(*args,**kwargs):
        x=original(*args,**kwargs);noise.append(hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest());return x
    module.randn_tensor=observed
    try:image,proof=N.G.observed_sample(pipe,prompts,tokens,seed,torch)
    finally:module.randn_tensor=original
    assert not noise and proof['UNet_NFE']==4
    return image,proof|dict(ancestral_noise_observations=[],deterministic_Euler_step_noise_draws=0)

def readout(generated,target,baseline,queries,gallery,method,matched):
    new=U.S.metric_arrays(generated,target,queries,gallery);refs={a:U.S.metric_arrays(baseline[:len(queries),j],target,queries,gallery) for j,a in enumerate(U.ARMS)}
    if matched is not None:refs['matched_architecture_control']=U.S.metric_arrays(matched[:len(queries)],target,queries,gallery)
    contrasts={k:U.S.compare(new,v,queries) for k,v in refs.items()};decision=U.S.decision(contrasts['ordinary'],str(len(queries)))
    if len(queries)==400 and decision['action']=='PROMISING_FOR_FULL_EVALUATION':
        if not all(U.S.decision(v,'400')['action']=='PROMISING_FOR_FULL_EVALUATION' for v in contrasts.values()):decision|={'action':'STOP_INSTANCE','reason':'ALL_STRONG_AND_MATCHED_CONTROLS_NOT_PASSED'}
    return dict(status='ARCHITECTURE_ACTUAL_NATIVE_IMAGE_READOUT',method=method,queries=len(queries),gallery=400,means={method:{k:float(v.mean()) for k,v in new.items()}}|{a:{k:float(v.mean()) for k,v in r.items()} for a,r in refs.items()},candidate_minus_references=contrasts,decision=decision,
        channels=M.METHODS[method]['channels'],old_references_use17_channels=True,full63_system_comparison_is_not_equal_input_mechanism_proof=M.METHODS[method]['channels']==63,
        matched_architecture_control=M.METHODS[method]['matched_control'],equal_capacity_or_equal_training_wall_time_claimed=False,independent_confirmation=False,SOTA_established=False)

def main():
    p=argparse.ArgumentParser()
    for key in ['spec','native-registration','output','previous']:p.add_argument('--'+key,type=Path)
    p.add_argument('--method',choices=tuple(M.METHODS),required=True);p.add_argument('--end',type=int,choices=[80,240,400],required=True);p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    started=time.monotonic();assert not a.output.exists() and (a.end==80)==(not a.previous);a.output.mkdir(parents=True);stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));active='INPUTS';newrows=[]
    def log(x):
        x|={'seconds':time.monotonic()-started}
        with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(x),flush=True)
    try:
        spec=U.read(a.spec);assert spec['schema']==M.SCHEMA and spec['method']==a.method
        for x in spec['source_pins']:U.checked(x)
        old=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(old['inputs']));panel=U.read(U.checked(inputs['panel']));queries=panel['Dcheck'];assert len(queries)==400
        conditions=np.load(U.checked(old['check_conditions']),allow_pickle=False);base=np.load(U.checked(old['check_generated_features']),allow_pickle=False);target=np.load(U.checked(old['check_target_features']),allow_pickle=False)
        baseline_rows=[json.loads(x) for x in U.checked(old['check_manifest']).read_text().splitlines()]
        if a.method!='D25_deterministic_Euler4':
            allfit=U.read(U.checked(spec['all10_fit_completion']));assert allfit['all10_subjects_complete'] and allfit['method']==a.method
            sub=U.read(U.checked(allfit['subjects']['sub-01']));values=np.load(U.checked(sub['predictions']),allow_pickle=False,mmap_mode='r');indices=np.load(U.checked(sub['array_rows']),allow_pickle=False);positions={int(r):i for i,r in enumerate(indices)}
            generated_conditions=np.stack([values[positions[q['array_row']]] for q in queries]);model_ref=sub['model']
        else:generated_conditions=conditions[:,1];model_ref=inputs['components']['ordinary']['checkpoint']
        matched=None
        if spec['matched_control_completion'] is not None:
            matched_done=U.read(U.checked(spec['matched_control_completion']));assert matched_done['end']==400
            matched=np.load(U.checked(matched_done['generated_features']),allow_pickle=False)
        prefix=[];prior_features=None;start=0
        if a.previous:
            prev=U.read(a.previous);assert prev['spec']==U.ref(a.spec) and prev['end']==a.end-160
            start=prev['end'];prefix=[json.loads(x) for x in U.checked(prev['manifest']).read_text().splitlines()];prior_features=np.load(U.checked(prev['generated_features']),allow_pickle=False)
            for row in prefix:assert U.sha(row['generated_path'])==row['generated_sha256']
        import brainhive_paired_study_contract_20261025_v1 as C
        cfg=C.read(a.native_registration);C.validate(cfg);C.setup_runtime(cfg)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=a.max_seconds,maximum_output_bytes=512<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stopped[0] and time.monotonic()-started<a.max_seconds-35,'finite architecture images';budget.check()
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);C.resources(cfg,training_start=True)
        from brainhive_component_generator_v1 import load_native
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        folder=a.output/'loader';folder.mkdir();audit=ConvertedWeightAudit(torch,folder);audit.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader=audit.complete();active='OLD_ORDINARY_NATIVE_PIXEL_PARITY'
        with torch.no_grad():ordinary_token=projector(adapter(torch.from_numpy(conditions[:1,1]).cuda().half()))
        img,obs=N.sample_observed(pipe,prompts,ordinary_token,queries[0]['image_seed'],torch);parity=N.save_image(a.output/'old_ordinary_pixel_parity.webp',img)
        for k in ['generated_sha256','generated_pixel_sha256']:assert parity[k]==baseline_rows[1][k]
        U.atomic(a.output/'pixel_parity.json',dict(status='ORIGINAL_ORDINARY_HOST_PIXEL_PARITY_COMPLETE',image=parity,reference=baseline_rows[1],noise=obs))
        sampler_change=None
        if a.method=='D25_deterministic_Euler4':
            from diffusers import EulerDiscreteScheduler
            original_t=pipe.scheduler.timesteps.detach().cpu().tolist();original_sigma=float(pipe.scheduler.init_noise_sigma)
            pipe.scheduler=EulerDiscreteScheduler.from_config(pipe.scheduler.config);pipe.scheduler.set_timesteps(4,device='cuda')
            sampler_change=dict(from_scheduler='EulerAncestralDiscreteScheduler',to_scheduler=type(pipe.scheduler).__name__,old_timesteps=original_t,new_timesteps=pipe.scheduler.timesteps.detach().cpu().tolist(),old_initial_sigma=original_sigma,new_initial_sigma=float(pipe.scheduler.init_noise_sigma),churn=0.,UNet_NFE=4)
            assert original_t==sampler_change['new_timesteps'] and pipe.scheduler.config.prediction_type=='epsilon' and pipe.scheduler.config.timestep_spacing=='trailing'
        active='REGISTERED_ARCHITECTURE_ACTUAL_IMAGES';rows=list(prefix);manifest=a.output/'inference_manifest.jsonl'
        with manifest.open('x') as stream,ThreadPoolExecutor(max_workers=1) as executor:
            for row in prefix:stream.write(json.dumps(row)+'\n')
            for i in range(start,a.end):
                check();q=queries[i];c=np.ascontiguousarray(generated_conditions[i:i+1])
                with torch.no_grad():
                    if M.METHODS[a.method]['output']=='IP4x2048':tokens=torch.from_numpy(c).cuda().half()
                    else:tokens=projector(adapter(torch.from_numpy(c).cuda().half()))
                if a.method=='D25_deterministic_Euler4':image,proof=deterministic_sample(pipe,prompts,tokens,q['image_seed'])
                else:
                    image,proof=N.sample_observed(pipe,prompts,tokens,q['image_seed'],torch)
                    for key in ['initial_UNet_input_sha256','initial_noise_latents_sha256','ancestral_noise_observations']:assert proof[key]==baseline_rows[3*i+1][key]
                if a.method=='D25_deterministic_Euler4':assert hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest()==baseline_rows[3*i+1]['tokens_sha256']
                future=executor.submit(N.save_image,a.output/'pixels'/f'{i:03d}_{q["array_row"]:05d}.webp',image)
                row=q|dict(method=a.method,query_index=i,condition_sha256=hashlib.sha256(c.tobytes()).hexdigest(),condition_shape=list(c.shape),tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),model=model_ref,channels=M.METHODS[a.method]['channels'],native_A_P_bypassed=M.METHODS[a.method]['output']=='IP4x2048',normal_inference_target_access=False,one_image_per_query=True,**proof)|future.result()
                rows.append(row);newrows.append(row);stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                if (i+1)%20==0:log(dict(stage=active,queries=i+1,end=a.end,new_images=len(newrows)))
        active='DINO_AND_FIXED_READOUT';import image_evaluation as E
        assert U.sha(E.DINO_WEIGHT)==E.DINO_SHA256==U.S.DINO_SHA256;dino=E.dino_model();pieces=[]
        for lo in range(0,len(newrows),32):check();pieces.append(E.dino_features([Path(r['generated_path']) for r in newrows[lo:lo+32]],dino,batch_size=32,real_images=False).float().numpy())
        features=np.concatenate(pieces)
        if prior_features is not None:features=np.concatenate([prior_features,features]);assert np.array_equal(features[:start],prior_features)
        U.save_array(a.output/'generated_features.npy',features);effect=readout(features,target,base,queries[:a.end],queries,a.method,matched) if a.end in (80,400) else None
        if effect is not None:U.atomic(a.output/'readout.json',effect)
        U.atomic(a.output/'completion.json',dict(status='ARCHITECTURE_ACTUAL_IMAGE_CHUNK_COMPLETE',method=a.method,spec=U.ref(a.spec),source=U.ref(__file__),start=start,end=a.end,manifest=U.ref(manifest),generated_features=U.ref(a.output/'generated_features.npy'),readout=effect,previous=U.ref(a.previous) if a.previous else None,source_model=model_ref,
            original80_prefix_preserved=True,pixel_parity=U.ref(a.output/'pixel_parity.json'),sampler_change=sampler_change,host=host,loader=loader,wrapper=wrapper,new_images=len(newrows),new_parity_images=1,new_UNet_NFE=4*(len(newrows)+1),per_query_UNet_NFE=4,
            native_A_P_bypassed=M.METHODS[a.method]['output']=='IP4x2048',normal_inference_target_access=False,original_D_resumed=False,full63_is_additional_input_relative_to_old17=M.METHODS[a.method]['channels']==63,seconds=time.monotonic()-started,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=torch.cuda.max_memory_allocated(),SOTA_established=False))
    except BaseException as e:
        U.atomic(a.output/'failure.json',dict(status='FAILED_ARCHITECTURE_IMAGE_CHUNK_PRESERVED',method=a.method,stage=active,error=repr(e),new_images=len(newrows),seconds=time.monotonic()-started,scientific_negative=False,original_D_resumed=False));traceback.print_exc();raise

if __name__=='__main__':main()
