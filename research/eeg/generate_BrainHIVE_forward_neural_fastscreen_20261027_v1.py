"""D06: finite latent correction through a training-fitted neural surrogate.

Original correct native EEG c and all four UNet evaluations are shared by the
two arms. Only the supplementary EEG-predicted group-A evidence is shuffled.
The operator is fitted to real image latents, not proven valid on generated x0.
This is a fixed surrogate-guidance heuristic, not a physical measurement,
Bayesian posterior, physiological gain, or an error-independence assertion.
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
import zipfile

import numpy as np
import torch
from PIL import Image

import generate_BrainHIVE_trialset_fastscreen_20261027_v1 as G
import generate_BrainHIVE_prediction_fastscreen_20261027_v1 as P
from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import save_image

ROOT=Path(__file__).resolve().parent
SCHEMA='BrainHIVE_forward_neural_native_screen_20261027_v1'
ARMS=('native_parent','neural_correct','neural_sameconcept_shuffled')
ALPHA=.25
ORDER='C-order channel0..3,y0..63,x0..63; native VAE MODE * one scaling_factor'
MAX_SECONDS=1800
OUTPUT_CAP=768<<20
sha,ref,read,rows,atomic=G.sha,G.ref,G.read,G.rows,G.atomic


def forward_update(x0,measurement,H,b,K,alpha=ALPHA):
    assert x0.ndim==4 and x0.shape[1:]==(4,64,64)
    assert measurement.shape==(x0.shape[0],16) and H.shape==(16384,16) and b.shape==(16,) and K.shape==(16,16384)
    assert alpha in (0.,ALPHA)
    if alpha==0.:return x0
    assert all(torch.isfinite(value).all() for value in (x0,measurement,H,b,K))
    flat=x0.flatten(1)
    return (flat+alpha*(measurement-(flat@H+b))@K).reshape_as(x0)


def shuffled_measurement_rows(queries):
    groups={}
    for query in queries:groups.setdefault((query['subject_id'],query['concept_id'],int(query['fold'])),[]).append(query)
    assert len(groups)==8 and all(len(group)==10 for group in groups.values())
    mapping={};receipts=[]
    for (subject,concept,fold),group in sorted(groups.items()):
        group=sorted(group,key=lambda query:int(query['array_row']))
        digest=hashlib.sha256(('D06_sameconcept_measurement_shuffle|1729|'+str(concept)).encode()).hexdigest()
        offset=1+int(digest,16)%9
        for index,query in enumerate(group):
            other=group[(index+offset)%10];assert query['row']!=other['row']
            mapping[query['query_id']]=other['row']
        receipts.append(dict(subject_id=subject,concept_id=concept,fold=fold,offset=offset,sha256=digest,
                             source_rows=[int(item['row']) for item in group]))
    assert len(mapping)==80 and len(set(mapping.values()))==80
    return mapping,receipts


def algebra_checks():
    generator=torch.Generator().manual_seed(1729)
    # Exercise the exact 16384-to16 export dimensions without trained weights.
    H=torch.randn((16384,16),generator=generator,dtype=torch.float64)/128
    b=torch.randn(16,generator=generator,dtype=torch.float64)
    lam=.1*float((H.T@H).trace())/16
    K=torch.linalg.solve(H.T@H+lam*torch.eye(16,dtype=torch.float64),H.T)
    x=torch.randn((1,4,64,64),generator=generator,dtype=torch.float64)
    y=torch.randn((1,16),generator=generator,dtype=torch.float64)
    flat=x.flatten(1);residual=y-(flat@H+b)
    updated=forward_update(x,y,H,b,K)
    expected=flat@H+b+ALPHA*residual@K@H
    affine_error=float(((updated.flatten(1)@H+b)-expected).abs().max())
    assert affine_error<1e-12 and forward_update(x,y,H,b,K,0.) is x
    expected_delta=ALPHA*residual@K
    assert torch.allclose(updated.flatten(1)-flat,expected_delta,atol=1e-14,rtol=1e-12)
    null=torch.randn((1,16384),generator=generator,dtype=torch.float64)
    null=null-(null@H)@torch.linalg.solve(H.T@H,H.T)
    delta=updated.flatten(1)-flat
    assert float((delta@null.T).abs().max())<1e-12
    inverse_error=0.
    epsilon=torch.randn(x.shape,generator=generator,dtype=torch.float64)
    for sigma in (.1,.7,2.,15.):
        x0=x-sigma*epsilon
        inverse_error=max(inverse_error,float((((x-x0)/sigma)-epsilon).abs().max()))
    assert inverse_error<1e-13
    assert {'generator','return_dict'}<=set(inspect.signature(NeuralStep.__call__).parameters)
    return dict(status='CPU_AFFINE_RIDGE_UPDATE_AND_EPSILON_ALGEBRA_PASSED',
        affine_prediction_equation_max_error=affine_error,epsilon_x0_inverse_max_abs=inverse_error,
        zero_update_returns_original_object=True,orthogonal_complement_update_dot_max_abs=float((delta@null.T).abs().max()),
        actual_GPU_zero_update_pixel_parity='still mandatory; not inferred from CPU algebra',
        alpha_is_not_a_norm_bound=True,operator_is_surrogate_not_physical_measurement=True)


class NeuralStep:
    def __init__(self,scheduler,measurement,H,b,K,alpha=ALPHA):
        assert type(scheduler).__name__=='EulerAncestralDiscreteScheduler'
        assert scheduler.config.prediction_type=='epsilon' and scheduler.config.timestep_spacing=='trailing'
        assert alpha in (0.,ALPHA)
        self.scheduler,self.measurement,self.H,self.b,self.K,self.alpha=scheduler,measurement,H,b,K,alpha
        self.original=scheduler.step;self.events=[]

    def __call__(self,model_output,timestep,sample,generator=None,return_dict=True):
        assert sample.shape==model_output.shape==(1,4,64,64)
        assert torch.isfinite(sample).all() and torch.isfinite(model_output).all()
        assert self.scheduler.step_index is not None
        index=int(self.scheduler.step_index);sigma_native=self.scheduler.sigmas[index];sigma=float(sigma_native)
        assert math.isfinite(sigma) and sigma>0
        if self.alpha==0.:
            result=self.original(model_output,timestep,sample,generator=generator,return_dict=return_dict)
            self.events.append(dict(step_index=index,sigma=sigma,alpha_zero_exact_native_call=True))
            return result
        original_x0=sample.float()-sigma_native*model_output
        desired_x0=forward_update(original_x0,self.measurement,self.H,self.b,self.K,self.alpha)
        changed_epsilon=((sample.float()-desired_x0)/sigma).to(model_output.dtype)
        assert changed_epsilon.dtype==model_output.dtype and torch.isfinite(changed_epsilon).all()
        actual_x0=sample.float()-sigma_native*changed_epsilon
        before=original_x0.flatten(1)@self.H+self.b
        after=actual_x0.flatten(1)@self.H+self.b
        actual_delta=actual_x0-original_x0
        self.events.append(dict(step_index=index,sigma=sigma,alpha_zero_exact_native_call=False,alpha=self.alpha,
            surrogate_residual_RMS_before=float((self.measurement-before).square().mean().sqrt()),
            surrogate_residual_RMS_after=float((self.measurement-after).square().mean().sqrt()),
            actual_latent_update_RMS=float(actual_delta.square().mean().sqrt()),
            actual_latent_update_relative_L2=float(actual_delta.norm()/original_x0.norm().clamp_min(1e-12)),
            epsilon_dtype=str(model_output.dtype),x0_epsilon_quantization_max_abs=float((actual_x0-desired_x0).abs().max()),
            interpretation='generated x0 may be outside real-latent operator domain; residual reduction is not image evidence'))
        return self.original(changed_epsilon,timestep,sample,generator=generator,return_dict=return_dict)


def sample_neural(pipe,prompts,tokens,seed,measurement,H,b,K,alpha=ALPHA):
    step=NeuralStep(pipe.scheduler,measurement,H,b,K,alpha)
    module=inspect.getmodule(type(pipe.scheduler));assert module is not None and hasattr(module,'randn_tensor')
    native_random=module.randn_tensor;noises=[]
    def observe(*args,**kwargs):
        value=native_random(*args,**kwargs)
        noises.append(dict(sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                           shape=list(value.shape),dtype=str(value.dtype)))
        return value
    pipe.scheduler.step=step;module.randn_tensor=observe
    try:image,proof=G.observed_sample(pipe,prompts,tokens,seed,torch)
    finally:pipe.scheduler.step=step.original;module.randn_tensor=native_random
    assert len(step.events)==len(noises)==4 and proof['UNet_NFE']==4
    assert all(item['shape']==[1,4,64,64] for item in noises)
    return image,proof|dict(forward_neural_steps=step.events,ancestral_noise_observations=noises,
        extra_UNet_evaluations=0,surrogate_neural_guidance_heuristic=True,EEG_parent_error_independence_assumed=False)


def validate_inputs(args):
    assert sha(args.scientific_preregistration)==args.scientific_preregistration_sha256
    panel=read(args.panel)
    assert panel['expected_queries']==80 and len(panel['queries'])==80
    for key in ('index','split','image_manifest'):assert sha(panel[key]['path'])==panel[key]['sha256']
    assert sha(args.image_manifest)==panel['image_manifest']['sha256']
    metadata=G.metadata(args.image_manifest)
    assert G.choose_panel(rows(panel['index']['path']),read(panel['split']['path']),metadata)['queries']==panel['queries']
    containers={}
    for query in panel['queries']:
        item=metadata[query['array_row']];containers.setdefault(item['image_zip'],[]).append(item)
    for path,items in containers.items():
        with zipfile.ZipFile(path) as archive:
            for item in items:assert archive.getinfo(item['image_member']).file_size==int(item['image_bytes'])
    complete=read(args.fit_completion)
    assert complete['schema']=='BrainHIVE_neural_latent_measurement_20261027_v1'
    assert complete['status']=='COMPLETE_CROSSBRAIN_NEURAL_LATENT_MEASUREMENT_GATE_PASSED'
    assert complete['alpha']==ALPHA and complete['latent_flatten_order']==ORDER
    assert complete['additional_subject_EEG_at_inference'] is False and complete['no_query_target_access'] is True
    assert complete['outer_held_used_for_selection'] is False
    assert complete['latent_scale_already_applied_once'] is True and abs(float(complete['latent_scaling_factor'])-.13025)<1e-9
    assert sha(complete['latent_receipt']['path'])==complete['latent_receipt']['sha256']
    latent_receipt=read(complete['latent_receipt']['path'])
    assert latent_receipt['status']=='COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS'
    assert latent_receipt['shape']==[13230,4,64,64] and latent_receipt['dtype']=='float32'
    assert abs(float(latent_receipt['registration']['VAE_scaling_factor'])-float(complete['latent_scaling_factor']))<1e-9
    index=Path(complete['prediction_index']);assert sha(index)==complete['prediction_index_sha256']
    identities=[G.identity(item) for item in rows(index)]
    assert len(identities)==13230 and identities==sorted([G.identity(item) for item in rows(panel['index']['path'])],key=lambda item:item['row'])
    assert {item['source_row'] for item in identities}==set(latent_receipt['array_rows'])
    positions={item['row']:position for position,item in enumerate(identities)}
    native=read(args.native_parent_binding)
    assert native['status']=='COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS'
    assert native['prediction_index_sha256']==complete['prediction_index_sha256']
    assert native['prediction_coordinates']==P.NATIVE_COORDINATES
    assert native['predictions_shape']==[13230,1024] and native['oof_coverage_min']==native['oof_coverage_max']==1
    assert sha(native['predictions_path'])==native['predictions_sha256']
    native_c=np.load(native['predictions_path'],mmap_mode='r',allow_pickle=False)
    assert native_c.shape==(13230,1024) and native_c.dtype==np.float32 and np.isfinite(native_c).all()
    assert complete['query_measurements_shape']==[13230,16]
    assert sha(complete['query_measurements_path'])==complete['query_measurements_sha256']
    measurements=np.load(complete['query_measurements_path'],mmap_mode='r',allow_pickle=False)
    assert measurements.shape==(13230,16) and measurements.dtype==np.float32 and np.isfinite(measurements).all()
    operators={};operator_checks={}
    assert set(complete['forward_folds'])=={'0','1'}
    for fold,item in complete['forward_folds'].items():
        values={}
        for name,shape in (('H',(16384,16)),('b',(16,)),('K',(16,16384))):
            assert item[name+'_shape']==list(shape) and sha(item[name+'_path'])==item[name+'_sha256']
            value=np.load(item[name+'_path'],allow_pickle=False)
            assert value.shape==shape and value.dtype in (np.float32,np.float64) and np.isfinite(value).all()
            values[name]=value
        H=np.asarray(values['H'],dtype=np.float64);K=np.asarray(values['K'],dtype=np.float64)
        gram=H.T@H;lam=.1*float(np.trace(gram))/16
        assert math.isfinite(lam) and lam>0 and math.isclose(float(item['ridge_lambda']),lam,rel_tol=2e-5,abs_tol=1e-10)
        expected=np.linalg.solve(gram+lam*np.eye(16),H.T)
        relative=float(np.linalg.norm(K-expected)/max(1e-12,np.linalg.norm(expected)))
        assert relative<=1e-4,'Exported K does not solve the registered affine ridge equation'
        operators[int(fold)]=values
        operator_checks[fold]=dict(ridge_lambda_from_H=lam,K_relative_L2_error=relative)
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
        for key in ('row','source_row','array_row','subject_id','concept_id','stimulus_id','fold','image_seed'):assert item[key]==query[key]
        raw=np.array(native_c[positions[query['row']]:positions[query['row']]+1],copy=True)
        assert hashlib.sha256(raw.tobytes()).hexdigest()==item['condition_sha256']
        assert sha(item['generated_path'])==item['generated_sha256'] and item['UNet_NFE']==4
        with Image.open(item['generated_path']) as image:
            assert image.size==(512,512) and hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==item['generated_pixel_sha256']
    shuffle,shuffle_receipt=shuffled_measurement_rows(panel['queries'])
    return panel,complete,native,native_c,positions,measurements,operators,reference,cached,shuffle,shuffle_receipt,operator_checks


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','fit-completion','native-parent-binding','native-reference-completion','image-manifest','native-registration',
                 'scientific-preregistration','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scientific-preregistration-sha256',required=True)
    parser.add_argument('--gpu-job-name');parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=1800)
    args=parser.parse_args(argv);assert 120<args.max_seconds<=MAX_SECONDS
    assert len(args.scientific_preregistration_sha256)==64
    torch.set_num_threads(2)
    invoked=time.monotonic();algebra=algebra_checks()
    panel,complete,native,native_c,positions,measurements,operators,reference,cached,shuffle,shuffle_receipt,operator_checks=validate_inputs(args)
    if args.preflight_only:
        print(json.dumps(dict(status='FORWARD_NEURAL_FULL_INPUTS_AND_CPU_ALGEBRA_VERIFIED',GPU_started=False,
            algebra=algebra,reused_native_images=80,new_candidate_images=160,parity_images=1,operator_checks=operator_checks)),flush=True)
        return 0
    assert args.gpu_job_name and not args.output_dir.exists()
    import brainhive_paired_study_contract_20261025_v1 as C
    config=C.read(args.native_registration);C.validate(config);C.setup_runtime(config)
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from brainhive_paired_supervision_20261026_v3 import safe_bytes_used
    torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
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
            arms=list(ARMS),queries=80,expected_images=240,reused_native_images=80,new_candidate_images=160,parity_images=1,new_UNet_NFE_expected=644,
            alpha=ALPHA,alpha_is_not_a_norm_bound=True,additional_subject_EEG_at_inference=False,no_UNet_training=True,
            original_correct_parent_c_identical_across_arms=True,shuffle_only_supplementary_measurement=True,
            supplementary_measurement_is_EEG_predicted_group_A_not_observed_query_group_A_or_B=True,
            fixed_measurement_shuffle=shuffle_receipt,forward_operator_checks=operator_checks,
            surrogate_guidance_not_physical_measurement_or_Bayesian_posterior=True,
            EEG_parent_error_independence_assumed=False,generated_x0_operator_domain_validity_unestablished=True,
            precedents=['neural encoding models','diffusion posterior/data-consistency guidance','ILVR/ATM structural guidance'],
            normal_inference_target_access=False,panel_already_exposed_development=True,independent_confirmation=False)
        atomic(args.output_dir/'input_provenance.json',binding);atomic(args.output_dir/'CPU_algebra.json',algebra)
        loader=args.output_dir/'loader';loader.mkdir();audit=ConvertedWeightAudit(torch,loader);audit.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader_proof=audit.complete();assert host==reference['host']
        assert abs(float(pipe.vae.config.scaling_factor)-float(complete['latent_scaling_factor']))<1e-9
        gpu_operators={fold:{key:torch.from_numpy(np.array(value,copy=True)).cuda().float() for key,value in item.items()}
                       for fold,item in operators.items()}
        atomic(args.output_dir/'scheduler_runtime.json',dict(class_name=type(pipe.scheduler).__name__,config=dict(pipe.scheduler.config),
            source_file=inspect.getfile(type(pipe.scheduler)),source_sha256=sha(inspect.getfile(type(pipe.scheduler))),
            formula='x0=sample-sigma*epsilon; x0_new=x0+.25*(Ahat-(flat(x0)@H+b))@K; epsilon_new=(sample-x0_new)/sigma cast original dtype',
            extra_UNet_or_VAE_evaluations=0,named_generator_and_return_dict_forwarded=True))
        first=panel['queries'][0];position=positions[first['row']];op=gpu_operators[int(first['fold'])]
        raw=np.array(native_c[position:position+1],copy=True)
        with torch.inference_mode():
            tokens=projector(adapter(torch.from_numpy(raw).cuda().to(torch.float16)))
            image,observation=sample_neural(pipe,prompts,tokens,first['image_seed'],torch.zeros((1,16),device='cuda'),**op,alpha=0.)
        baseline=cached[first['query_id']]
        assert observation['initial_noise_latents_sha256']==baseline['initial_noise_latents_sha256']
        assert observation['initial_UNet_input_sha256']==baseline['initial_UNet_input_sha256']
        assert hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest()==baseline['tokens_sha256']
        parity=save_image(args.output_dir/'parity'/'actual_zero_update.webp',image)
        with Image.open(baseline['generated_path']) as original:parity_passed=np.array_equal(np.asarray(image.convert('RGB')),np.asarray(original.convert('RGB')))
        atomic(args.output_dir/'zero_update_pixel_parity.json',dict(status='ACTUAL_ZERO_UPDATE_NATIVE_PIXEL_PARITY_PASSED' if parity_passed else 'ACTUAL_ZERO_UPDATE_NATIVE_PIXEL_PARITY_FAILED',
            query=first,reference_generated_sha256=baseline['generated_sha256'],reference_pixel_sha256=baseline['generated_pixel_sha256'],
            parity=parity,observation=observation,positive_update_images_started=False,UNet_NFE=4))
        assert parity_passed,'Zero-update actual native pixel parity failed; preserve and stop before candidates'
        inference=args.output_dir/'inference_manifest.jsonl'
        with inference.open('x',encoding='utf-8',buffering=1) as stream:
            for query_index,query in enumerate(panel['queries']):
                assert not stopped[0] and time.monotonic()-invoked<args.max_seconds-120
                budget.check();position=positions[query['row']];old=cached[query['query_id']];op=gpu_operators[int(query['fold'])]
                raw=np.array(native_c[position:position+1],copy=True)
                with torch.inference_mode():tokens=projector(adapter(torch.from_numpy(raw).cuda().to(torch.float16)))
                assert hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest()==old['tokens_sha256']
                copied={key:old[key] for key in ('condition_sha256','condition_coordinates','condition_FP16_sha256','tokens_sha256',
                    'generated_path','generated_sha256','generated_pixel_sha256','initial_noise_latents_sha256','initial_UNet_input_sha256',
                    'UNet_NFE','native_IP_K_calls','native_IP_V_calls')}
                base=query|copied|dict(arm='native_parent',global_query_index=query_index,prediction_row=position,
                    prediction_file_sha256=native['predictions_sha256'],baseline_artifact_reused=True,
                    baseline_source_completion_sha256=binding['native_reference_completion']['sha256'],one_image_per_query=True,
                    oracle_normalization=False,normal_inference_target_access=False,repeat_protocol='R4',
                    panel_already_exposed_development=True,method='native_parent_immutable_reuse')
                generated.append(base);stream.write(json.dumps(base)+'\n');stream.flush();os.fsync(stream.fileno())
                noises=None
                for arm in ARMS[1:]:
                    measurement_row=query['row'] if arm=='neural_correct' else shuffle[query['query_id']]
                    mposition=positions[measurement_row]
                    value=np.array(measurements[mposition:mposition+1],copy=True)
                    with torch.inference_mode():image,proof=sample_neural(pipe,prompts,tokens,query['image_seed'],torch.from_numpy(value).cuda().float(),**op)
                    assert proof['initial_noise_latents_sha256']==old['initial_noise_latents_sha256']
                    assert proof['initial_UNet_input_sha256']==old['initial_UNet_input_sha256']
                    if noises is not None:assert proof['ancestral_noise_observations']==noises
                    noises=proof['ancestral_noise_observations']
                    if query_index==0:assert noises==observation['ancestral_noise_observations']
                    saved=save_image(args.output_dir/arm/'pixels'/f'{query_index:03d}_{query["array_row"]:05d}.webp',image)
                    row=query|proof|saved|dict(arm=arm,global_query_index=query_index,prediction_row=position,
                        condition_sha256=base['condition_sha256'],condition_coordinates=base['condition_coordinates'],
                        condition_FP16_sha256=base['condition_FP16_sha256'],tokens_sha256=base['tokens_sha256'],
                        original_correct_parent_c_identical=True,measurement_source_row=int(measurement_row),measurement_prediction_row=mposition,
                        measurement_sha256=hashlib.sha256(value.tobytes()).hexdigest(),measurement_bank_sha256=complete['query_measurements_sha256'],
                        measurement_coordinates='D04 EEG-predicted outer-fit group-A16 coordinates; no actual query group-A/B input',
                        forward_operator_fold=int(query['fold']),
                        forward_H_sha256=complete['forward_folds'][str(query['fold'])]['H_sha256'],
                        forward_b_sha256=complete['forward_folds'][str(query['fold'])]['b_sha256'],
                        forward_K_sha256=complete['forward_folds'][str(query['fold'])]['K_sha256'],
                        alpha=ALPHA,baseline_artifact_reused=False,one_image_per_query=True,
                        oracle_normalization=False,normal_inference_target_access=False,repeat_protocol='R4',
                        panel_already_exposed_development=True,method='forward_neural_surrogate_'+arm)
                    generated.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                assert safe_bytes_used(args.output_dir)<OUTPUT_CAP-(16<<20)
                if (query_index+1)%10==0:print(json.dumps(dict(stage='FORWARD_NEURAL_SCREEN',queries=query_index+1,
                    new_candidate_images=2*(query_index+1),reused_native_images=query_index+1)),flush=True)
        assert len(generated)==240 and parity_passed
        manifest=G.attach_targets(G.metadata(args.image_manifest),generated,args.output_dir,np,Image)
        assert sha(complete['query_measurements_path'])==complete['query_measurements_sha256']
        for item in complete['forward_folds'].values():
            for key in ('H','b','K'):assert sha(item[key+'_path'])==item[key+'_sha256']
        for arg,key in ((args.fit_completion,'fit_completion'),(args.native_parent_binding,'native_parent_binding'),
                        (args.native_reference_completion,'native_reference_completion'),(args.scientific_preregistration,'scientific_preregistration')):
            assert sha(arg)==binding[key]['sha256']
        budget.check()
        atomic(args.output_dir/'completion.json',dict(status='COMPLETED_GENERIC_NATIVE_PREDICTION_SCREEN',schema=P.SCHEMA,
            method_schema=SCHEMA,queries=80,arms=list(ARMS),images=240,per_arm_images={arm:80 for arm in ARMS},
            manifest=ref(manifest),inference_manifest=ref(inference),panel=ref(args.panel),input_provenance=ref(args.output_dir/'input_provenance.json'),
            host=host,loader_proof=loader_proof,wrapper=wrapper,native_NFE_per_image=4,resolution=[512,512],CFG=0,IP_scale=1,
            new_candidate_images=160,reused_native_images=80,parity_images=1,new_UNet_NFE=644,
            actual_zero_update_pixel_parity=ref(args.output_dir/'zero_update_pixel_parity.json'),
            no_best_of_N=True,normal_inference_target_access=False,target_pixels_loaded_after_generation=True,
            panel_already_exposed_development=True,independent_confirmation=False,SOTA_established=False,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-invoked,
            prespecified_additional_pair=['neural_correct','neural_sameconcept_shuffled'],extra_UNet_evaluations_per_candidate_image=0))
        return 0
    except BaseException as error:
        atomic(args.output_dir/'failure.json',dict(status='FAILED_FORWARD_NEURAL_SCREEN_PRESERVED_NO_AUTOMATIC_RETRY',
            error=repr(error),observed_manifest_rows=len(generated),expected_rows=240,expected_new_candidate_images=160,
            actual_parity_passed=bool(parity_passed),seconds=time.monotonic()-invoked,no_effect_claim=True))
        raise


if __name__=='__main__':raise SystemExit(main())
