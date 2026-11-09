"""D05: immutable native baseline plus raw replay and partial-neural replay.

All three arms use the original HIVE A/P/IP/UNet and four native evaluations.
The two exported conditions retain original BrainMLP affine coordinates.  A
new direct-native parity image must match the cached baseline before either
candidate is generated.  Reencoded native is a real separate control, never
silently replaced by cached c.  Other subjects appear only in fitted training
machinery; inference consumes one subject's exported EEG-only condition.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import time
import zipfile

import numpy as np
from PIL import Image

import generate_BrainHIVE_trialset_fastscreen_20261027_v1 as G
import generate_BrainHIVE_prediction_fastscreen_20261027_v1 as P

ROOT=Path(__file__).resolve().parent
SCHEMA='BrainHIVE_neural_clean_native_screen_20261027_v1'
ARMS=('native_parent','native_reencoded','neural_clean')
COORDINATES='raw native BrainMLP OOF output after fixed original-EEG or partial-neural-EEG replay; original affine retained; no target access or scaling'
PREREG_SHA='115f9d92ecd74364c49604084613a5922e9dab94c35b170aaae7b735476d9b22'
MAX_SECONDS=1800
OUTPUT_CAP=768<<20
sha,ref,read,rows,atomic=G.sha,G.ref,G.read,G.rows,G.atomic


def sample_observed(pipe,prompts,tokens,seed,torch):
    # Observe native RNG returns without replacing scheduler.step.  Diffusers'
    # signature-based generator forwarding therefore remains exactly native.
    module=inspect.getmodule(type(pipe.scheduler))
    assert module is not None and hasattr(module,'randn_tensor')
    native=module.randn_tensor;noises=[]
    def capture(*args,**kwargs):
        value=native(*args,**kwargs)
        noises.append(dict(sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                           shape=list(value.shape),dtype=str(value.dtype)))
        return value
    original_step=pipe.scheduler.step
    module.randn_tensor=capture
    try:image,proof=G.observed_sample(pipe,prompts,tokens,seed,torch)
    finally:module.randn_tensor=native
    assert pipe.scheduler.step==original_step and len(noises)==4 and proof['UNet_NFE']==4
    return image,proof|dict(ancestral_noise_observations=noises,scheduler_step_signature_unmodified=True,
                           extra_UNet_evaluations=0)


def replay_gate(native,reencoded):
    assert native.shape==reencoded.shape==(13230,1024)
    max_absolute=max_relative=0.;min_cosine=1.
    for start in range(0,len(native),512):
        original=np.asarray(native[start:start+512],dtype=np.float64)
        fresh=np.asarray(reencoded[start:start+512],dtype=np.float64)
        norm=np.linalg.norm(original,axis=1);other_norm=np.linalg.norm(fresh,axis=1)
        assert (norm>0).all() and (other_norm>0).all()
        delta=fresh-original
        max_absolute=max(max_absolute,float(np.abs(delta).max()))
        max_relative=max(max_relative,float((np.linalg.norm(delta,axis=1)/norm).max()))
        min_cosine=min(min_cosine,float(((original*fresh).sum(1)/(norm*other_norm)).min()))
    assert max_absolute<=1e-3 and max_relative<=1e-4 and min_cosine>=1-1e-6,'Native replay engineering gate failed; no images'
    return dict(status='ALL13230_NATIVE_REPLAY_NUMERICAL_GATE_PASSED',rows=13230,
        max_absolute_error=max_absolute,max_relative_L2=max_relative,minimum_cosine=min_cosine,
        limits=dict(max_absolute_error=1e-3,max_relative_L2=1e-4,minimum_cosine=1-1e-6),
        equality_assumed=False,FP16_and_token_drift='recorded separately per generated query')


def validate_inputs(args):
    assert sha(args.scientific_preregistration)==PREREG_SHA
    panel=read(args.panel)
    assert panel['expected_queries']==80 and len(panel['queries'])==80
    for key in ('index','split','image_manifest'):assert sha(panel[key]['path'])==panel[key]['sha256']
    assert sha(args.image_manifest)==panel['image_manifest']['sha256']
    metadata=G.metadata(args.image_manifest)
    selected=G.choose_panel(rows(panel['index']['path']),read(panel['split']['path']),metadata)
    assert selected['queries']==panel['queries']
    containers={}
    for query in panel['queries']:
        item=metadata[query['array_row']];containers.setdefault(item['image_zip'],[]).append(item)
    for path,items in containers.items():
        with zipfile.ZipFile(path) as archive:
            for item in items:assert archive.getinfo(item['image_member']).file_size==int(item['image_bytes'])
    complete=read(args.fit_completion)
    assert complete['status']=='COMPLETE_RAW_NEURAL_PARTIAL_DENOISER_NATIVE_CONDITIONS'
    assert complete['alpha']==.25 and complete['additional_subject_EEG_at_inference'] is False
    assert complete['no_query_target_access'] is True
    assert complete['engineering_gate_passed'] is True
    index_path=Path(complete['prediction_index']);assert sha(index_path)==complete['prediction_index_sha256']
    index=[G.identity(item) for item in rows(index_path)]
    original=sorted([G.identity(item) for item in rows(panel['index']['path'])],key=lambda item:item['row'])
    assert index==original and len(index)==13230
    positions={item['row']:position for position,item in enumerate(index)}
    native=read(args.native_parent_binding)
    assert native['status']=='COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS'
    assert native['prediction_index_sha256']==complete['prediction_index_sha256']
    assert native['prediction_coordinates']==P.NATIVE_COORDINATES
    receipts={'native_parent':native};predictions={}
    for arm in ARMS:
        item=native if arm=='native_parent' else complete['arms'][arm]
        assert item['prediction_coordinates']==(P.NATIVE_COORDINATES if arm=='native_parent' else COORDINATES)
        assert item['predictions_shape']==[13230,1024] and item['oof_coverage_min']==item['oof_coverage_max']==1
        assert sha(item['predictions_path'])==item['predictions_sha256']
        value=np.load(item['predictions_path'],mmap_mode='r',allow_pickle=False)
        assert value.shape==(13230,1024) and value.dtype==np.float32 and np.isfinite(value).all()
        receipts[arm]=item;predictions[arm]=value
    replay=replay_gate(predictions['native_parent'],predictions['native_reencoded'])
    reference=read(args.native_reference_completion)
    assert reference['status']=='COMPLETED_GENERIC_NATIVE_PREDICTION_SCREEN' and 'native_parent' in reference['arms']
    assert reference['panel']['sha256']==sha(args.panel)
    assert sha(reference['manifest']['path'])==reference['manifest']['sha256']
    assert sha(reference['input_provenance']['path'])==reference['input_provenance']['sha256']
    prior=read(reference['input_provenance']['path'])
    assert prior['native_registration']['sha256']==sha(args.native_registration)
    assert prior['predictions']['native_parent']['sha256']==native['predictions_sha256']
    cached={item['query_id']:item for item in rows(reference['manifest']['path']) if item['arm']=='native_parent'}
    assert len(cached)==80 and set(cached)=={query['query_id'] for query in panel['queries']}
    for query in panel['queries']:
        item=cached[query['query_id']]
        for key in ('row','source_row','array_row','subject_id','concept_id','stimulus_id','fold','image_seed'):
            assert item[key]==query[key]
        raw=np.array(predictions['native_parent'][positions[query['row']]:positions[query['row']]+1],copy=True)
        assert hashlib.sha256(raw.tobytes()).hexdigest()==item['condition_sha256']
        assert sha(item['generated_path'])==item['generated_sha256'] and item['UNet_NFE']==4
        with Image.open(item['generated_path']) as image:
            assert image.size==(512,512)
            assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==item['generated_pixel_sha256']
    return panel,complete,receipts,predictions,positions,reference,cached,replay


def save_image(path,image):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending');assert not path.exists() and not pending.exists()
    image.save(pending,format='WEBP',lossless=True,quality=90,method=4)
    assert pending.stat().st_size<=1<<20
    pixels=np.asarray(image.convert('RGB'))
    with Image.open(pending) as decoded:assert np.array_equal(pixels,np.asarray(decoded.convert('RGB')))
    with pending.open('rb') as stream:os.fsync(stream.fileno())
    pending.rename(path)
    return dict(generated_path=str(path),generated_sha256=sha(path),generated_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','fit-completion','native-parent-binding','native-reference-completion','image-manifest','native-registration','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scientific-preregistration',type=Path,
        default=ROOT/'audit/neural_denoising_fastscreen_20261027_v1/scientific_preregistration.json')
    parser.add_argument('--gpu-job-name');parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=1800)
    args=parser.parse_args(argv);assert 120<args.max_seconds<=MAX_SECONDS
    invoked=time.monotonic();panel,complete,receipts,predictions,positions,reference,cached,replay=validate_inputs(args)
    if args.preflight_only:
        print(json.dumps(dict(status='NEURAL_CLEAN_COMPLETE_INPUTS_CPU_VERIFIED',starts_GPU=False,
            arms=list(ARMS),reused_native_images=80,new_candidate_images=160,parity_images=1)),flush=True)
        return 0
    assert args.gpu_job_name and not args.output_dir.exists()
    import brainhive_paired_study_contract_20261025_v1 as C
    config=C.read(args.native_registration);C.validate(config);C.setup_runtime(config)
    import torch
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from brainhive_paired_supervision_20261026_v3 import safe_bytes_used
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output_dir.mkdir(parents=True)
    document=dict(gpu_job_name=args.gpu_job_name,maximum_seconds=args.max_seconds,maximum_output_bytes=OUTPUT_CAP,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,
        resource_authorization=config['resource_authorization'])
    D.torch=torch;wrapper=D.verify_wrapper(document);budget=D.Budget(document)
    stopped=[False];signal.signal(signal.SIGTERM,lambda number,frame:stopped.__setitem__(0,True))
    generated=[];parity_passed=False
    try:
        binding=dict(schema=SCHEMA,source=ref(__file__),panel=ref(args.panel),fit_completion=ref(args.fit_completion),
            native_parent_binding=ref(args.native_parent_binding),native_reference_completion=ref(args.native_reference_completion),
            native_registration=ref(args.native_registration),scientific_preregistration=ref(args.scientific_preregistration),
            arms=list(ARMS),queries=80,expected_images=240,reused_native_images=80,new_candidate_images=160,parity_images=1,
            new_UNet_NFE_expected=644,no_UNet_training=True,no_F_target_norm_scaling=True,
            additional_subject_EEG_at_inference=False,normal_inference_target_access=False,
            native_reencoded_is_separately_rendered_control=True,cache_replay_equality_assumed=False,
            native_replay_engineering_gate=replay,alpha_not_a_perturbation_norm_bound=True,
            reverse_map_is_not_an_orthogonal_projector=True,
            partial_cleaning_alpha=.25,panel_already_exposed_development=True,independent_confirmation=False)
        atomic(args.output_dir/'input_provenance.json',binding)
        loader=args.output_dir/'loader';loader.mkdir();audit=ConvertedWeightAudit(torch,loader);audit.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader_proof=audit.complete();assert host==reference['host']
        first=panel['queries'][0];position=positions[first['row']]
        raw=np.array(predictions['native_parent'][position:position+1],copy=True)
        with torch.inference_mode():
            tokens=projector(adapter(torch.from_numpy(raw).cuda().to(torch.float16)))
            image,observation=sample_observed(pipe,prompts,tokens,first['image_seed'],torch)
        baseline=cached[first['query_id']]
        assert observation['initial_noise_latents_sha256']==baseline['initial_noise_latents_sha256']
        assert observation['initial_UNet_input_sha256']==baseline['initial_UNet_input_sha256']
        assert hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest()==baseline['tokens_sha256']
        parity=save_image(args.output_dir/'parity'/'actual_direct_native.webp',image)
        with Image.open(baseline['generated_path']) as old:parity_passed=np.array_equal(np.asarray(image.convert('RGB')),np.asarray(old.convert('RGB')))
        atomic(args.output_dir/'native_pixel_parity.json',dict(status='ACTUAL_DIRECT_NATIVE_PIXEL_PARITY_PASSED' if parity_passed else 'ACTUAL_DIRECT_NATIVE_PIXEL_PARITY_FAILED',
            query=first,reference_generated_sha256=baseline['generated_sha256'],reference_pixel_sha256=baseline['generated_pixel_sha256'],
            parity=parity,observation=observation,candidate_images_started=False,UNet_NFE=4))
        assert parity_passed,'Actual native parity failed; preserve image and stop before candidates'
        inference=args.output_dir/'inference_manifest.jsonl'
        with inference.open('x',encoding='utf-8',buffering=1) as stream:
            for query_index,query in enumerate(panel['queries']):
                assert not stopped[0] and time.monotonic()-invoked<args.max_seconds-120
                budget.check();position=positions[query['row']];old=cached[query['query_id']]
                cached_raw=np.array(predictions['native_parent'][position:position+1],copy=True)
                with torch.inference_mode():
                    cached_condition=torch.from_numpy(cached_raw).cuda().to(torch.float16)
                    cached_tokens=projector(adapter(cached_condition))
                assert hashlib.sha256(cached_tokens.cpu().contiguous().numpy().tobytes()).hexdigest()==old['tokens_sha256']
                copied={key:old[key] for key in ('condition_sha256','condition_coordinates','condition_FP16_sha256','tokens_sha256',
                    'generated_path','generated_sha256','generated_pixel_sha256','initial_noise_latents_sha256',
                    'initial_UNet_input_sha256','UNet_NFE','native_IP_K_calls','native_IP_V_calls')}
                base=query|copied|dict(arm='native_parent',global_query_index=query_index,prediction_row=position,
                    baseline_artifact_reused=True,baseline_source_completion_sha256=binding['native_reference_completion']['sha256'],
                    prediction_file_sha256=receipts['native_parent']['predictions_sha256'],
                    one_image_per_query=True,oracle_normalization=False,normal_inference_target_access=False,
                    repeat_protocol='R4',panel_already_exposed_development=True,method='native_parent_immutable_reuse')
                generated.append(base);stream.write(json.dumps(base)+'\n');stream.flush();os.fsync(stream.fileno())
                noise=None
                for arm in ARMS[1:]:
                    raw=np.array(predictions[arm][position:position+1],copy=True)
                    with torch.inference_mode():
                        condition=torch.from_numpy(raw).cuda().to(torch.float16)
                        tokens=projector(adapter(condition))
                        image,proof=sample_observed(pipe,prompts,tokens,query['image_seed'],torch)
                    assert tokens.shape==(1,4,2048) and torch.isfinite(tokens).all()
                    assert proof['initial_noise_latents_sha256']==old['initial_noise_latents_sha256']
                    assert proof['initial_UNet_input_sha256']==old['initial_UNet_input_sha256']
                    if noise is not None:assert noise==proof['ancestral_noise_observations']
                    noise=proof['ancestral_noise_observations']
                    if query_index==0:assert noise==observation['ancestral_noise_observations']
                    fp16_delta=(condition.float()-cached_condition.float())
                    token_delta=(tokens.float()-cached_tokens.float())
                    runtime_drift=dict(FP16_c_equal=bool(torch.equal(condition,cached_condition)),
                        FP16_c_max_absolute_error=float(fp16_delta.abs().max()),
                        FP16_c_relative_L2=float(fp16_delta.norm()/cached_condition.float().norm()),
                        tokens_equal=bool(torch.equal(tokens,cached_tokens)),
                        tokens_max_absolute_error=float(token_delta.abs().max()),
                        tokens_relative_L2=float(token_delta.norm()/cached_tokens.float().norm()),
                        comparison='actual condition/token versus same cached native input; no equality forced')
                    saved=save_image(args.output_dir/arm/'pixels'/f'{query_index:03d}_{query["array_row"]:05d}.webp',image)
                    row=query|proof|saved|dict(arm=arm,global_query_index=query_index,prediction_row=position,
                        condition_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),condition_coordinates=receipts[arm]['prediction_coordinates'],
                        condition_FP16_sha256=hashlib.sha256(condition.cpu().contiguous().numpy().tobytes()).hexdigest(),
                        tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
                        prediction_file_sha256=receipts[arm]['predictions_sha256'],one_image_per_query=True,
                        actual_condition_token_drift_vs_cached_native=runtime_drift,
                        oracle_normalization=False,normal_inference_target_access=False,additional_subject_EEG_at_inference=False,
                        repeat_protocol='R4',panel_already_exposed_development=True,
                        method='partial_neural_native_'+arm,baseline_artifact_reused=False)
                    generated.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                assert safe_bytes_used(args.output_dir)<OUTPUT_CAP-(16<<20)
                if (query_index+1)%10==0:print(json.dumps(dict(stage='NEURAL_CLEAN_NATIVE_SCREEN',queries=query_index+1,
                    new_candidate_images=2*(query_index+1),reused_native_images=query_index+1)),flush=True)
        assert len(generated)==240 and parity_passed
        manifest=G.attach_targets(G.metadata(args.image_manifest),generated,args.output_dir,np,Image)
        for arm in ARMS:assert sha(receipts[arm]['predictions_path'])==receipts[arm]['predictions_sha256']
        for arg,key in ((args.fit_completion,'fit_completion'),(args.native_parent_binding,'native_parent_binding'),
                        (args.native_reference_completion,'native_reference_completion'),(args.scientific_preregistration,'scientific_preregistration')):
            assert sha(arg)==binding[key]['sha256']
        budget.check()
        atomic(args.output_dir/'completion.json',dict(status='COMPLETED_GENERIC_NATIVE_PREDICTION_SCREEN',schema=P.SCHEMA,
            method_schema=SCHEMA,queries=80,arms=list(ARMS),images=240,per_arm_images={arm:80 for arm in ARMS},
            manifest=ref(manifest),inference_manifest=ref(inference),panel=ref(args.panel),input_provenance=ref(args.output_dir/'input_provenance.json'),
            host=host,loader_proof=loader_proof,wrapper=wrapper,native_NFE_per_image=4,resolution=[512,512],CFG=0,IP_scale=1,
            new_candidate_images=160,reused_native_images=80,parity_images=1,new_UNet_NFE=644,
            actual_native_pixel_parity=ref(args.output_dir/'native_pixel_parity.json'),
            no_best_of_N=True,normal_inference_target_access=False,target_pixels_loaded_after_generation=True,
            panel_already_exposed_development=True,independent_confirmation=False,SOTA_established=False,
            additional_subject_EEG_at_inference=False,native_reencoded_separate_control=True,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-invoked,
            prespecified_additional_pair=['neural_clean','native_reencoded'],extra_UNet_evaluations_per_candidate_image=0))
        return 0
    except BaseException as error:
        atomic(args.output_dir/'failure.json',dict(status='FAILED_NEURAL_CLEAN_SCREEN_PRESERVED_NO_AUTOMATIC_RETRY',
            error=repr(error),observed_manifest_rows=len(generated),expected_rows=240,expected_new_candidate_images=160,
            actual_parity_passed=bool(parity_passed),seconds=time.monotonic()-invoked,no_effect_claim=True))
        raise


if __name__=='__main__':
    raise SystemExit(main())
