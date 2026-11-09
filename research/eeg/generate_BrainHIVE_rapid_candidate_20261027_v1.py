"""One typed candidate renderer for BrainHIVE rapid stages80 and400.

Stage400 retains the exact first80 candidate rows/images and renders320 new
query images. Each process first renders one zero-module native parity image.
All inference is four native UNet evaluations; target pixels are never inputs.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from PIL import Image
import brainhive_rapid_candidate_pack_20261027_v1 as P
import brainhive_rapid_panel_cache_20261027_v1 as C
import generate_BrainHIVE_trialset_fastscreen_20261027_v1 as G
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N

ROOT=Path(__file__).resolve().parent
sha,ref,read,rows,atomic=C.sha,C.ref,C.read,C.rows,C.atomic


def forward_update(x0,measurement,H,b,K,alpha):
    assert x0.shape==(1,4,64,64) and H.shape[0]==16384 and H.shape[1]==measurement.shape[1]
    assert measurement.shape[0]==1 and b.shape==(H.shape[1],) and K.shape==(H.shape[1],16384)
    assert math.isfinite(float(alpha)) and 0<=alpha<=1
    if alpha==0:return x0
    assert all(torch.isfinite(value).all() for value in (x0,measurement,H,b,K))
    flat=x0.flatten(1)
    return (flat+alpha*(measurement-(flat@H+b))@K).reshape_as(x0)


class LinearStep:
    def __init__(self,scheduler,measurement,H,b,K,alpha):
        assert type(scheduler).__name__=='EulerAncestralDiscreteScheduler'
        assert scheduler.config.prediction_type=='epsilon' and scheduler.config.timestep_spacing=='trailing'
        self.scheduler,self.measurement,self.H,self.b,self.K,self.alpha=scheduler,measurement,H,b,K,alpha
        self.original=scheduler.step;self.events=[]

    def __call__(self,model_output,timestep,sample,generator=None,return_dict=True):
        assert sample.shape==model_output.shape==(1,4,64,64)
        assert self.scheduler.step_index is not None
        index=int(self.scheduler.step_index);sigma_native=self.scheduler.sigmas[index];sigma=float(sigma_native)
        assert math.isfinite(sigma) and sigma>0
        if self.alpha==0:
            result=self.original(model_output,timestep,sample,generator=generator,return_dict=return_dict)
            self.events.append(dict(step_index=index,sigma=sigma,zero_update_exact_original_call=True))
            return result
        original_x0=sample.float()-sigma_native*model_output
        desired=forward_update(original_x0,self.measurement,self.H,self.b,self.K,self.alpha)
        changed=((sample.float()-desired)/sigma).to(model_output.dtype)
        assert torch.isfinite(changed).all()
        actual_x0=sample.float()-sigma_native*changed;delta=actual_x0-original_x0
        before=original_x0.flatten(1)@self.H+self.b;after=actual_x0.flatten(1)@self.H+self.b
        self.events.append(dict(step_index=index,sigma=sigma,alpha=self.alpha,zero_update_exact_original_call=False,
            residual_RMS_before=float((self.measurement-before).square().mean().sqrt()),
            residual_RMS_after=float((self.measurement-after).square().mean().sqrt()),
            actual_latent_update_RMS=float(delta.square().mean().sqrt()),
            actual_latent_update_relative_L2=float(delta.norm()/original_x0.norm().clamp_min(1e-12)),
            x0_epsilon_quantization_max_abs=float((actual_x0-desired).abs().max()),
            residual_reduction_is_not_image_evidence=True))
        return self.original(changed,timestep,sample,generator=generator,return_dict=return_dict)


def sample_linear(pipe,prompts,tokens,seed,measurement,operator,alpha):
    step=LinearStep(pipe.scheduler,measurement,operator['H'],operator['b'],operator['K'],alpha)
    module=inspect.getmodule(type(pipe.scheduler));assert module is not None and hasattr(module,'randn_tensor')
    original_random=module.randn_tensor;noises=[]
    def capture(*args,**kwargs):
        value=original_random(*args,**kwargs)
        noises.append(dict(sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),shape=list(value.shape),dtype=str(value.dtype)))
        return value
    pipe.scheduler.step=step;module.randn_tensor=capture
    try:image,proof=G.observed_sample(pipe,prompts,tokens,seed,torch)
    finally:pipe.scheduler.step=step.original;module.randn_tensor=original_random
    assert len(step.events)==len(noises)==4 and proof['UNet_NFE']==4
    return image,proof|dict(linear_latent_steps=step.events,ancestral_noise_observations=noises,extra_UNet_evaluations=0)


def runtime_sources():
    return {name:ref(path) for name,path in [('generator',__file__),('pack',P.__file__),('panel',C.__file__),
        ('native_sampler',N.__file__),('observation_helper',G.__file__),
        ('native_sample_core',ROOT/'generate_BrainHIVE_paired_effect_20261025_v1.py'),
        ('native_host_loader',ROOT/'brainhive_component_generator_v1.py'),
        ('low_memory_loader',ROOT/'run_BrainHIVE_frozen_function_low_memory_20261025_v3.py')]}


def validate_previous(path,pack_path,panel_path,baseline_path,registration_path,current_sources,queries):
    previous=read(path)
    assert previous['schema']==P.GEN_SCHEMA and previous['status']==P.GEN_STATUS and previous['stage']=='80'
    assert previous['queries']==80 and previous['runtime_sources']==current_sources
    for key,expected in [('pack',pack_path),('panel',panel_path),('baseline_completion',baseline_path),('native_registration',registration_path)]:
        assert previous[key]['sha256']==sha(expected);C.verified(previous[key])
    old=rows(C.verified(previous['manifest']))
    assert len(old)==80 and [row['query_id'] for row in old]==[row['query_id'] for row in queries[:80]]
    for row,query in zip(old,queries[:80]):
        assert all(row[key]==query[key] for key in C.IDENTITY_KEYS)
        assert row['pack_sha256']==sha(pack_path) and row['UNet_NFE']==4 and row['normal_inference_target_access'] is False
        assert sha(row['generated_path'])==row['generated_sha256']
        with Image.open(row['generated_path']) as image:
            assert image.size==(512,512) and hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==row['generated_pixel_sha256']
    return previous,old


def validate_expansion(score_path,arm,panel_path,baseline_path,stage1_completion,pack_path,expected_host_key):
    """Use the locked scorer's action, without embedding a scoring policy."""
    score=read(score_path)
    assert score['schema'].startswith('BrainHIVE_rapid_cached_DINO_CPU_')
    assert score['status']=='COMPLETE_CPU_FIXED400_GALLERY_PAIRED_SCREEN' and score['stage']=='80'
    assert score['panel']['sha256']==sha(panel_path) and score['gallery_targets']==400
    C.verified(score['panel']);C.verified(score['source']);C.verified(score['per_query'])
    candidate=score['candidates'][arm]
    assert candidate['decision']['action']=='EXPAND400','Only an explicit stage1 EXPAND400 action authorizes stage2'
    cache=read(C.verified(candidate['cache']))
    assert cache['generation_completion']['sha256']==sha(stage1_completion)
    assert cache['pack']['sha256']==sha(pack_path) and cache['host_cache_key']==expected_host_key
    native=read(C.verified(score['native_cache']))
    assert native['baseline_completion']['sha256']==sha(baseline_path)
    return ref(score_path)


def validate(args):
    panel=C.validate_panel(args.panel);pack,baseline,index,native=P.validate_pack(args.pack,args.baseline_completion,panel)
    assert baseline['native_NFE_per_image']==4 and baseline['resolution']==[512,512]
    assert baseline['panel']['sha256']==sha(args.panel)
    base_rows=rows(C.verified(baseline['manifest']));by_id={row['query_id']:row for row in base_rows}
    assert len(base_rows)==len(by_id)==400 and set(by_id)=={row['query_id'] for row in panel['queries']}
    for i,query in enumerate(panel['queries']):
        row=by_id[query['query_id']]
        assert all(row[key]==query[key] for key in C.IDENTITY_KEYS)
        assert row['UNet_NFE']==4 and row['normal_inference_target_access'] is False
        assert row['target_image_sha256']==panel['target_gallery'][i]['image_sha256']
        assert sha(row['generated_path'])==row['generated_sha256']
    prior=read(C.verified(baseline['input_provenance']))
    assert prior['native_registration']['sha256']==sha(args.native_registration)
    source_refs=runtime_sources();previous=None;prefix=[]
    if args.stage=='400':
        assert args.stage1_completion and args.stage1_score_summary
        previous,prefix=validate_previous(args.stage1_completion,args.pack,args.panel,args.baseline_completion,
            args.native_registration,source_refs,panel['queries'])
        assert previous['arm']==pack['arm'] and previous['host_cache_key']==pack['host_cache_key']
        assert read(C.verified(previous['zero_module_parity']))['status']=='ACTUAL_ZERO_MODULE_NATIVE_PIXEL_PARITY_PASSED'
        for row in prefix:
            base=by_id[row['query_id']]
            for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256','target_image_sha256'):
                assert row[key]==base[key]
        validate_expansion(args.stage1_score_summary,pack['arm'],args.panel,args.baseline_completion,
            args.stage1_completion,args.pack,pack['host_cache_key'])
    else:assert args.stage1_completion is None and args.stage1_score_summary is None
    return panel,pack,baseline,index,native,by_id,source_refs,previous,prefix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pack','panel','baseline-completion','native-registration','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=('80','400'),required=True);parser.add_argument('--stage1-completion',type=Path)
    parser.add_argument('--stage1-score-summary',type=Path)
    parser.add_argument('--gpu-job-name');parser.add_argument('--max-seconds',type=int,default=1800);parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();assert 120<args.max_seconds<=1800
    invoked=time.monotonic();panel,pack,baseline,index,native,base_rows,sources,previous,prefix=validate(args)
    controls={key:ref(path) for key,path in [('pack',args.pack),('panel',args.panel),('baseline_completion',args.baseline_completion),('native_registration',args.native_registration)]}
    if args.stage1_completion:controls['stage1_completion']=ref(args.stage1_completion)
    if args.stage1_score_summary:controls['stage1_score_summary']=ref(args.stage1_score_summary)
    queries=panel['queries'][:int(args.stage)];new_queries=queries[len(prefix):]
    if args.preflight_only:
        print(json.dumps(dict(status='CPU_RAPID_CANDIDATE_INPUTS_VERIFIED',stage=args.stage,mode=pack['mode'],
            reuse_candidate_images=len(prefix),new_query_images=len(new_queries),parity_images=1,actual_images_generated=0,GPU_started=False)));return 0
    assert args.gpu_job_name and not args.output.exists()
    import brainhive_paired_study_contract_20261025_v1 as F
    config=F.read(args.native_registration);F.validate(config);F.setup_runtime(config)
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from brainhive_paired_supervision_20261026_v3 import safe_bytes_used
    args.output.mkdir(parents=True);D.torch=torch
    document=dict(gpu_job_name=args.gpu_job_name,maximum_seconds=args.max_seconds,maximum_output_bytes=1024<<20,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
    wrapper=D.verify_wrapper(document);budget=D.Budget(document);stopped=[False]
    signal.signal(signal.SIGTERM,lambda number,frame:stopped.__setitem__(0,True))
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    generated=list(prefix);actual_new=0;parity_passed=False
    try:
        positions={row['row']:i for i,row in enumerate(index)}
        native_c=np.load(native['predictions_path'],mmap_mode='r',allow_pickle=False)
        conditions=np.load(pack['payload']['conditions']['path'],mmap_mode='r',allow_pickle=False) if pack['mode']=='native_conditions' else None
        measurements=None;operators={}
        if pack['mode']=='linear_latent_update':
            payload=pack['payload'];measurements=np.load(payload['measurements']['path'],mmap_mode='r',allow_pickle=False)
            for fold,item in payload['fold_operators'].items():
                operators[int(fold)]={name:torch.from_numpy(np.array(np.load(item[name]['path'],allow_pickle=False),copy=True)).float().cuda() for name in ('H','b','K')}
        atomic(args.output/'input_provenance.json',dict(schema=P.GEN_SCHEMA,pack=ref(args.pack),panel=ref(args.panel),
            baseline_completion=ref(args.baseline_completion),native_registration=ref(args.native_registration),runtime_sources=sources,
            stage=args.stage,arm=pack['arm'],mode=pack['mode'],host_family='BrainHIVE',host_cache_key=pack['host_cache_key'],
            new_query_images=len(new_queries),reused_candidate_images=len(prefix),mandatory_parity_images=1,
            normal_inference_target_access=False,no_EEG_fit=True,no_target_pixels_opened_by_generation=True))
        F.resources(config,training_start=True)
        loader=args.output/'loader';loader.mkdir();audit=ConvertedWeightAudit(torch,loader);audit.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader_proof=audit.complete();assert host==baseline['host']==pack['expected_host'] and P.host_key(host)==pack['host_cache_key']
        if pack['mode']=='linear_latent_update':assert abs(float(pipe.vae.config.scaling_factor)-pack['payload']['latent_scaling_factor'])<1e-9
        def render(query,zero=False):
            position=positions[query['row']]
            selected=native_c if zero or conditions is None else conditions
            raw=np.array(selected[position:position+1],copy=True)
            with torch.inference_mode():
                condition=torch.from_numpy(raw).cuda();tokens=projector(adapter(condition.to(torch.float16)))
                if pack['mode']=='linear_latent_update':
                    evidence=np.array(measurements[position:position+1],copy=True);measurement=torch.from_numpy(evidence).float().cuda()
                    image,observation=sample_linear(pipe,prompts,tokens,query['image_seed'],measurement,operators[query['fold']],0 if zero else pack['payload']['alpha'])
                else:image,observation=N.sample_observed(pipe,prompts,tokens,query['image_seed'],torch)
            proof=dict(condition_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                condition_FP16_sha256=hashlib.sha256(condition.to(torch.float16).cpu().contiguous().numpy().tobytes()).hexdigest(),
                tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest())
            if pack['mode']=='linear_latent_update':proof['measurement_sha256']=hashlib.sha256(evidence.tobytes()).hexdigest()
            return image,observation,proof
        first=queries[0];old=base_rows[first['query_id']]
        image,observation,proof=render(first,zero=True)
        for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256'):assert observation[key]==old[key]
        for key in ('condition_sha256','condition_FP16_sha256','tokens_sha256'):assert proof[key]==old[key]
        saved=N.save_image(args.output/'parity'/'zero_module_native.webp',image);actual_new+=1
        with Image.open(old['generated_path']) as original:parity_passed=np.array_equal(np.asarray(image.convert('RGB')),np.asarray(original.convert('RGB')))
        atomic(args.output/'zero_module_pixel_parity.json',dict(status='ACTUAL_ZERO_MODULE_NATIVE_PIXEL_PARITY_PASSED' if parity_passed else 'ACTUAL_ZERO_MODULE_NATIVE_PIXEL_PARITY_FAILED',
            query=first,image=saved,observation=observation,proof=proof,candidate_query_images_started=False))
        assert parity_passed,'STOP before candidate images: zero-module native pixel parity failed'
        inference=args.output/'inference_manifest.jsonl'
        with inference.open('x',encoding='utf-8') as stream:
            for row in prefix:stream.write(json.dumps(row,allow_nan=False)+'\n')
            for query in new_queries:
                assert not stopped[0] and time.monotonic()-invoked<args.max_seconds-120
                budget.check();assert safe_bytes_used(args.output)<(1024<<20)-(16<<20)
                image,observation,proof=render(query);old=base_rows[query['query_id']]
                for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256'):assert observation[key]==old[key]
                if 'ancestral_noise_observations' in old:assert observation['ancestral_noise_observations']==old['ancestral_noise_observations']
                image_ref=N.save_image(args.output/'pixels'/f'{len(generated):03d}_{query["array_row"]:05d}.webp',image);actual_new+=1
                row=query|observation|proof|image_ref|dict(arm=pack['arm'],global_query_index=len(generated),pack_sha256=controls['pack']['sha256'],
                    host_cache_key=pack['host_cache_key'],mode=pack['mode'],normal_inference_target_access=False,oracle_normalization=False,
                    one_image_per_query=True,repeat_protocol='R4',method='BrainHIVE_rapid_typed_candidate')
                generated.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                if len(generated)%20==0:print(json.dumps(dict(stage='RAPID_CANDIDATE_GENERATION',count=len(generated),expected=len(queries),
                    actual_new_images_generated=actual_new,reused_candidate_images=len(prefix),seconds=time.monotonic()-invoked)),flush=True)
        assert len(generated)==len(queries) and actual_new==len(new_queries)+1
        # Target files are only checked/attached after all new query images.
        joined=[]
        for i,row in enumerate(generated):
            old=base_rows[row['query_id']];assert sha(old['target_image_path'])==old['target_image_sha256']
            target={key:old[key] for key in ('target_image_path','score_target_path','target_image_sha256','target_index')}
            joined.append(row if i<len(prefix) else row|target|dict(target_mapping_attached_after_inference=True))
        manifest=args.output/'manifest.jsonl'
        with manifest.open('x',encoding='utf-8') as stream:
            stream.write(''.join(json.dumps(row,allow_nan=False)+'\n' for row in joined));stream.flush();os.fsync(stream.fileno())
        assert joined[:len(prefix)]==prefix
        P.validate_pack(args.pack,args.baseline_completion,panel)
        for source in sources.values():C.verified(source)
        for control in controls.values():C.verified(control)
        budget.check()
        atomic(args.output/'completion.json',dict(schema=P.GEN_SCHEMA,status=P.GEN_STATUS,stage=args.stage,queries=len(queries),arm=pack['arm'],mode=pack['mode'],
            pack=ref(args.pack),panel=ref(args.panel),baseline_completion=ref(args.baseline_completion),native_registration=ref(args.native_registration),
            manifest=ref(manifest),inference_manifest=ref(inference),input_provenance=ref(args.output/'input_provenance.json'),
            stage1_completion=ref(args.stage1_completion) if args.stage1_completion else None,runtime_sources=sources,host_family='BrainHIVE',host=host,
            stage1_score_summary=ref(args.stage1_score_summary) if args.stage1_score_summary else None,
            host_cache_key=pack['host_cache_key'],loader_proof=loader_proof,zero_module_parity=ref(args.output/'zero_module_pixel_parity.json'),
            reused_candidate_images=len(prefix),new_query_images=len(new_queries),parity_images=1,actual_new_images_generated=actual_new,
            new_UNet_evaluations=4*actual_new,native_NFE_per_image=4,resolution=[512,512],CFG=0,IP_scale=1,
            normal_inference_target_access=False,targets_attached_after_generation=True,no_best_of_N=True,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-invoked,development_only=True))
        return 0
    except BaseException as error:
        atomic(args.output/'failure.json',dict(status='INCOMPLETE_RAPID_CANDIDATE_NO_AUTOMATIC_RETRY',error=repr(error),stage=args.stage,
            retained_candidate_rows=len(generated),actual_new_images_generated=actual_new,zero_module_parity_passed=parity_passed,seconds=time.monotonic()-invoked))
        raise


if __name__=='__main__':raise SystemExit(main())
