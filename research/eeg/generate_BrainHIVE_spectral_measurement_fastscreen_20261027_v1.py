"""Direction03: finite soft spectral latent correction inside native Euler steps.

This is a proximal heuristic, not a physical EEG measurement, exact posterior,
Kalman update, or an independence claim.  ILVR/ATM already motivate structural
guidance.  Native c, A/P/IP/UNet and four NFE remain unchanged.  Before any
positive-beta image a newly rendered zero-beta image must match an immutable
native-parent reference pixel-for-pixel.  The reference's 80 native artifacts
are then reused, with explicit provenance, to avoid repeated baseline work.
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

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_soft_spectral_native_screen_20261027_v1'
ARMS = ('native_parent', 'c_only', 'raw_eeg')
ORDER = 'channel0..3,ky0..3,kx0..3; C-order flatten'
BASIS = 'B[k,n]=alpha(k)*cos(pi*(n+0.5)*k/64); alpha(0)=sqrt(1/64); alpha(k>0)=sqrt(2/64); k=0..3,n=0..63'
sha, ref, read, rows, atomic = G.sha, G.ref, G.read, G.rows, G.atomic
MAX_SECONDS = 1800
OUTPUT_CAP = 768 << 20


def cosine_basis(device='cpu', dtype=torch.float64):
    frequency = torch.arange(4, dtype=dtype, device=device)[:, None]
    position = torch.arange(64, dtype=dtype, device=device)[None, :]
    alpha = torch.full((4, 1), math.sqrt(2 / 64), dtype=dtype, device=device)
    alpha[0] = math.sqrt(1 / 64)
    return alpha * torch.cos(math.pi * (position + .5) * frequency / 64)


def coefficients(latent, basis):
    assert latent.ndim == 4 and latent.shape[1:] == (4, 64, 64)
    return torch.einsum('kn,bcnm,lm->bckl', basis, latent, basis).flatten(1)


def synthesize(value, basis):
    assert value.ndim == 2 and value.shape[1] == 64
    return torch.einsum('kn,bckl,lm->bcnm', basis, value.reshape(-1, 4, 4, 4), basis)


def soft_proximal(x0, measurement, beta, basis):
    assert measurement.shape == beta.shape == (x0.shape[0], 64)
    assert torch.isfinite(measurement).all() and torch.isfinite(beta).all()
    assert bool(((beta >= 0) & (beta <= 1)).all())
    if not bool(torch.any(beta != 0)):
        return x0  # object identity and exact zeros, not an arithmetic replay
    return x0 + synthesize(beta * (measurement - coefficients(x0, basis)), basis)


def algebra_checks():
    basis = cosine_basis()
    assert torch.allclose(basis @ basis.T, torch.eye(4, dtype=torch.float64), atol=2e-14, rtol=0)
    generator = torch.Generator().manual_seed(1729)
    x = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float64)
    epsilon = torch.randn(x.shape, generator=generator, dtype=torch.float64)
    y = torch.randn((1, 64), generator=generator, dtype=torch.float64)
    beta = torch.linspace(0, 1, 64, dtype=torch.float64)[None]
    maximum = 0.
    for sigma in (.1, .7, 2., 15.):
        x0 = x - sigma * epsilon
        recovered = (x - x0) / sigma
        maximum = max(maximum, float((recovered - epsilon).abs().max()))
        assert torch.allclose(recovered, epsilon, atol=1e-13, rtol=1e-13)
    projected = soft_proximal(x, y, beta, basis)
    expected = coefficients(x, basis) + beta * (y - coefficients(x, basis))
    assert torch.allclose(coefficients(projected, basis), expected, atol=1e-13, rtol=1e-13)
    assert soft_proximal(x, y, torch.zeros_like(beta), basis) is x
    # Components outside the selected DCT subspace remain unchanged.
    orthogonal = x - synthesize(coefficients(x, basis), basis)
    orthogonal_after = projected - synthesize(coefficients(projected, basis), basis)
    assert torch.allclose(orthogonal_after, orthogonal, atol=1e-13, rtol=1e-13)
    return dict(status='CPU_FLOAT64_SCHEDULER_AND_ORTHONORMAL_PROX_ALGEBRA_PASSED',
                epsilon_x0_inverse_max_abs=maximum, positive_sigma_values=[.1, .7, 2., 15.],
                zero_beta_returns_original_object=True, unselected_subspace_unchanged=True,
                GPU_pixel_parity='not established by algebra; mandatory actual image gate follows')


class SpectralStep:
    """Intercept only scheduler.step; preserve model-output dtype and RNG draws."""
    def __init__(self, scheduler, measurement, beta, basis):
        assert type(scheduler).__name__ == 'EulerAncestralDiscreteScheduler'
        assert scheduler.config.prediction_type == 'epsilon'
        assert scheduler.config.timestep_spacing == 'trailing'
        self.scheduler, self.measurement, self.beta, self.basis = scheduler, measurement, beta, basis
        self.original = scheduler.step
        self.events = []

    def __call__(self, model_output, timestep, sample, **kwargs):
        assert sample.shape == model_output.shape == (1, 4, 64, 64)
        assert torch.isfinite(model_output).all() and torch.isfinite(sample).all()
        assert self.scheduler.step_index is not None, 'Native scale_model_input must initialize current sigma'
        index = int(self.scheduler.step_index)
        sigma_native = self.scheduler.sigmas[index]
        sigma = float(sigma_native)
        assert math.isfinite(sigma) and sigma > 0
        if not bool(torch.any(self.beta != 0)):
            result = self.original(model_output, timestep, sample, **kwargs)
            self.events.append(dict(step_index=index, sigma=sigma, beta_zero_exact_native_call=True))
            return result
        # Match EulerAncestral's original x0 arithmetic, then use a finite soft
        # correction.  Casting epsilon back preserves its noise dtype/RNG path.
        original_x0 = sample.float() - sigma_native * model_output
        corrected_x0 = soft_proximal(original_x0, self.measurement, self.beta, self.basis)
        changed_epsilon = ((sample.float() - corrected_x0) / sigma).to(model_output.dtype)
        assert changed_epsilon.dtype == model_output.dtype and torch.isfinite(changed_epsilon).all()
        actual_x0 = sample.float() - sigma_native * changed_epsilon
        before = coefficients(original_x0, self.basis)
        after = coefficients(actual_x0, self.basis)
        self.events.append(dict(step_index=index, sigma=sigma, beta_zero_exact_native_call=False,
            measured_residual_RMS_before=float((before-self.measurement).square().mean().sqrt()),
            measured_residual_RMS_after=float((after-self.measurement).square().mean().sqrt()),
            epsilon_dtype=str(model_output.dtype),
            x0_quantization_max_abs=float((actual_x0-corrected_x0).abs().max()),
            interpretation='FP16 epsilon rounding is recorded; no exact-positive-projection claim'))
        return self.original(changed_epsilon, timestep, sample, **kwargs)


def sample_spectral(pipe, prompts, tokens, image_seed, measurement, beta, basis):
    step = SpectralStep(pipe.scheduler, measurement, beta, basis)
    module = inspect.getmodule(type(pipe.scheduler))
    assert module is not None and hasattr(module, 'randn_tensor')
    native_random = module.randn_tensor
    actual_noises = []
    def observed_random(*args, **kwargs):
        value = native_random(*args, **kwargs)
        actual_noises.append(dict(sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                                  shape=list(value.shape), dtype=str(value.dtype)))
        return value
    pipe.scheduler.step = step
    module.randn_tensor = observed_random
    try:
        image, observation = G.observed_sample(pipe, prompts, tokens, image_seed, torch)
    finally:
        pipe.scheduler.step = step.original
        module.randn_tensor = native_random
    assert len(step.events) == len(actual_noises) == 4
    assert all(item['shape'] == [1, 4, 64, 64] for item in actual_noises)
    return image, observation | dict(spectral_steps=step.events, ancestral_noise_observations=actual_noises,
                                    extra_UNet_evaluations=0, soft_proximal_heuristic=True,
                                    EEG_parent_error_independence_assumed=False)


def validate_inputs(args):
    panel = read(args.panel)
    assert panel['expected_queries'] == 80 and len(panel['queries']) == 80
    for key in ('index','split','image_manifest'):
        assert sha(panel[key]['path']) == panel[key]['sha256']
    assert sha(args.image_manifest) == panel['image_manifest']['sha256']
    metadata = G.metadata(args.image_manifest)
    selected = G.choose_panel(rows(panel['index']['path']), read(panel['split']['path']), metadata)
    assert panel['queries'] == selected['queries']
    containers={}
    for query in panel['queries']:
        item=metadata[query['array_row']]
        containers.setdefault(item['image_zip'],[]).append(item)
    for path,items in containers.items():
        with zipfile.ZipFile(path) as archive:
            for item in items:
                assert archive.getinfo(item['image_member']).file_size==int(item['image_bytes'])
    complete = read(args.fit_completion)
    assert complete['status'] == 'COMPLETE_EEG_SPECTRAL_MEASUREMENT_SCREEN'
    assert complete['coefficient_order'] == ORDER and complete['basis_definition'] == BASIS
    assert abs(float(complete['latent_scaling_factor']) - .13025) < 1e-9
    assert complete['latent_scale_already_applied_once'] is True
    assert complete['outer_held_used_for_selection'] is False and complete['no_query_target_scaling'] is True
    assert complete['beta_is_proximal_heuristic'] is True
    assert sha(complete['latent_receipt']['path']) == complete['latent_receipt']['sha256']
    latent_receipt=read(complete['latent_receipt']['path'])
    assert latent_receipt['status']=='COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS'
    assert latent_receipt['shape']==[13230,4,64,64] and latent_receipt['dtype']=='float32'
    assert latent_receipt['sha256']==complete['latent_bank_sha256']
    assert float(latent_receipt['registration']['VAE_scaling_factor'])==float(complete['latent_scaling_factor'])
    index_path = Path(complete['prediction_index'])
    assert sha(index_path) == complete['prediction_index_sha256']
    index = [G.identity(item) for item in rows(index_path)]
    original = sorted([G.identity(item) for item in rows(panel['index']['path'])], key=lambda item:item['row'])
    assert index == original and len(index) == 13230
    assert {item['source_row'] for item in index}==set(latent_receipt['array_rows'])
    positions = {item['row']:position for position,item in enumerate(index)}
    native = read(args.native_parent_binding)
    assert native['status'] == 'COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS'
    assert native['prediction_index_sha256'] == complete['prediction_index_sha256']
    assert native['prediction_coordinates'] == P.NATIVE_COORDINATES
    assert native['predictions_shape'] == [13230,1024] and native['oof_coverage_min'] == native['oof_coverage_max'] == 1
    assert sha(native['predictions_path']) == native['predictions_sha256']
    native_c = np.load(native['predictions_path'], mmap_mode='r', allow_pickle=False)
    assert native_c.shape == (13230,1024) and native_c.dtype == np.float32 and np.isfinite(native_c).all()
    assert sha(complete['basis_path']) == complete['basis_sha256'] and complete['basis_shape'] == [4,64]
    exported_basis = np.load(complete['basis_path'],allow_pickle=False)
    assert np.allclose(exported_basis,cosine_basis().numpy(),atol=2e-7,rtol=2e-7)
    measurements, betas = {}, {}
    for arm in ARMS[1:]:
        receipt = complete['arms'][arm]
        assert receipt['measurements_shape'] == [13230,64] and receipt['beta_shape'] == [2,64]
        assert receipt['oof_coverage_min'] == receipt['oof_coverage_max'] == 1
        assert sha(receipt['measurements_path']) == receipt['measurements_sha256']
        assert sha(receipt['beta_path']) == receipt['beta_sha256']
        value = np.load(receipt['measurements_path'],mmap_mode='r',allow_pickle=False)
        beta = np.load(receipt['beta_path'],allow_pickle=False)
        assert value.shape == (13230,64) and value.dtype == np.float32 and np.isfinite(value).all()
        assert beta.shape == (2,64) and np.isfinite(beta).all() and ((beta>=0)&(beta<=1)).all()
        measurements[arm],betas[arm] = value,beta
    reference = read(args.native_reference_completion)
    assert reference['status'] == 'COMPLETED_GENERIC_NATIVE_PREDICTION_SCREEN' and 'native_parent' in reference['arms']
    assert reference['panel']['sha256'] == sha(args.panel)
    assert sha(reference['manifest']['path']) == reference['manifest']['sha256']
    assert sha(reference['input_provenance']['path']) == reference['input_provenance']['sha256']
    prior = read(reference['input_provenance']['path'])
    assert prior['native_registration']['sha256'] == sha(args.native_registration)
    assert prior['predictions']['native_parent']['sha256'] == native['predictions_sha256']
    reference_rows = {item['query_id']:item for item in rows(reference['manifest']['path']) if item['arm']=='native_parent'}
    assert len(reference_rows) == 80 and set(reference_rows) == {query['query_id'] for query in panel['queries']}
    for query in panel['queries']:
        item=reference_rows[query['query_id']]
        for key in ('row','source_row','array_row','subject_id','concept_id','stimulus_id','fold','image_seed'):
            assert item[key] == query[key]
        raw=np.array(native_c[positions[query['row']]:positions[query['row']]+1],copy=True)
        assert hashlib.sha256(raw.tobytes()).hexdigest() == item['condition_sha256']
        assert sha(item['generated_path']) == item['generated_sha256'] and item['UNet_NFE']==4
        with Image.open(item['generated_path']) as image:
            assert image.size == (512,512)
            assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() == item['generated_pixel_sha256']
    return panel,complete,native,native_c,positions,measurements,betas,reference,reference_rows


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
    parser.add_argument('--gpu-job-name');parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=1800)
    args=parser.parse_args(argv);assert 120<args.max_seconds<=MAX_SECONDS
    invoked=time.monotonic();algebra=algebra_checks()
    panel,complete,native,native_c,positions,measurements,betas,reference,cached=validate_inputs(args)
    if args.preflight_only:
        print(json.dumps(dict(status='SPECTRAL_COMPLETE_INPUTS_AND_CPU_ALGEBRA_VERIFIED',starts_GPU=False,
                              algebra=algebra,new_candidate_images=160,reused_native_images=80,parity_images=1)),flush=True)
        return 0
    assert args.gpu_job_name and not args.output_dir.exists()
    import brainhive_paired_study_contract_20261025_v1 as C
    config=C.read(args.native_registration);C.validate(config);C.setup_runtime(config)
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
    generated=[];parity=None;parity_passed=False
    try:
        binding=dict(schema=SCHEMA,source=ref(__file__),panel=ref(args.panel),fit_completion=ref(args.fit_completion),
            native_parent_binding=ref(args.native_parent_binding),native_reference_completion=ref(args.native_reference_completion),
            native_registration=ref(args.native_registration),arms=list(ARMS),queries=80,expected_images=240,
            new_candidate_images=160,reused_native_images=80,parity_images=1,new_UNet_NFE_expected=644,
            panel_already_exposed_development=True,independent_confirmation=False,
            beta='inner-fit relative-to-train-constant reliability; finite soft proximal heuristic at each of four steps',
            EEG_parent_error_independence_assumed=False,not_exact_posterior_or_physical_measurement=True,
            precedents=['ILVR low-frequency conditioning','ATM structural EEG guidance'],
            scheduler_reference='https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py',
            normal_inference_target_access=False)
        atomic(args.output_dir/'input_provenance.json',binding)
        atomic(args.output_dir/'CPU_algebra.json',algebra)
        loader=args.output_dir/'loader';loader.mkdir();audit=ConvertedWeightAudit(torch,loader);audit.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader_proof=audit.complete();assert host==reference['host'],'Native host changed from baseline reference'
        assert abs(float(pipe.vae.config.scaling_factor)-float(complete['latent_scaling_factor']))<1e-9
        basis=cosine_basis(device='cuda',dtype=torch.float32)
        atomic(args.output_dir/'scheduler_runtime.json',dict(class_name=type(pipe.scheduler).__name__,
            config=dict(pipe.scheduler.config),source_file=inspect.getfile(type(pipe.scheduler)),
            source_sha256=sha(inspect.getfile(type(pipe.scheduler))),
            epsilon_x0_formula='x0=sample-sigma*epsilon; epsilon_new=(sample-x0_prox)/sigma; keep original epsilon dtype',
            native_random_draws_preserved=True))
        first=panel['queries'][0];position=positions[first['row']]
        raw=np.array(native_c[position:position+1],copy=True)
        with torch.inference_mode():
            tokens=projector(adapter(torch.from_numpy(raw).cuda().to(torch.float16)))
            image,observation=sample_spectral(pipe,prompts,tokens,first['image_seed'],
                torch.zeros((1,64),device='cuda'),torch.zeros((1,64),device='cuda'),basis)
        baseline=cached[first['query_id']]
        assert observation['initial_noise_latents_sha256']==baseline['initial_noise_latents_sha256']
        assert observation['initial_UNet_input_sha256']==baseline['initial_UNet_input_sha256']
        assert hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest()==baseline['tokens_sha256']
        parity=save_image(args.output_dir/'parity'/'actual_zero_beta.webp',image)
        with Image.open(baseline['generated_path']) as native_image:
            parity_passed=np.array_equal(np.asarray(image.convert('RGB')),np.asarray(native_image.convert('RGB')))
        atomic(args.output_dir/'zero_beta_pixel_parity.json',dict(status='ACTUAL_ZERO_BETA_NATIVE_PIXEL_PARITY_PASSED' if parity_passed else 'ACTUAL_ZERO_BETA_NATIVE_PIXEL_PARITY_FAILED',
            query=first,reference_generated_sha256=baseline['generated_sha256'],reference_pixel_sha256=baseline['generated_pixel_sha256'],
            parity=parity,observation=observation,positive_beta_images_started=False,UNet_NFE=4))
        assert parity_passed,'Zero-beta native pixel parity failed; artifact retained, no positive-beta images allowed'
        inference=args.output_dir/'inference_manifest.jsonl'
        with inference.open('x',encoding='utf-8',buffering=1) as stream:
            for query_index,query in enumerate(panel['queries']):
                assert not stopped[0] and time.monotonic()-invoked<args.max_seconds-120
                budget.check();position=positions[query['row']];old=cached[query['query_id']]
                raw=np.array(native_c[position:position+1],copy=True)
                with torch.inference_mode():tokens=projector(adapter(torch.from_numpy(raw).cuda().to(torch.float16)))
                assert hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest()==old['tokens_sha256']
                copied={key:old[key] for key in ('condition_sha256','condition_coordinates','condition_FP16_sha256',
                    'tokens_sha256','generated_path','generated_sha256','generated_pixel_sha256',
                    'initial_noise_latents_sha256','initial_UNet_input_sha256','UNet_NFE','native_IP_K_calls','native_IP_V_calls')}
                base=query|copied|dict(arm='native_parent',global_query_index=query_index,prediction_row=position,
                    prediction_file_sha256=native['predictions_sha256'],baseline_artifact_reused=True,
                    baseline_source_completion_sha256=binding['native_reference_completion']['sha256'],
                    one_image_per_query=True,oracle_normalization=False,normal_inference_target_access=False,
                    repeat_protocol='R4',panel_already_exposed_development=True,method='native_parent_immutable_reuse')
                generated.append(base);stream.write(json.dumps(base)+'\n');stream.flush();os.fsync(stream.fileno())
                noise=None
                for arm in ARMS[1:]:
                    y=torch.from_numpy(np.array(measurements[arm][position:position+1],copy=True)).cuda()
                    beta=torch.from_numpy(np.array(betas[arm][int(query['fold']):int(query['fold'])+1],copy=True)).cuda().float()
                    with torch.inference_mode():image,proof=sample_spectral(pipe,prompts,tokens,query['image_seed'],y,beta,basis)
                    assert proof['initial_noise_latents_sha256']==old['initial_noise_latents_sha256']
                    assert proof['initial_UNet_input_sha256']==old['initial_UNet_input_sha256']
                    if noise is not None:assert proof['ancestral_noise_observations']==noise
                    noise=proof['ancestral_noise_observations']
                    if query_index==0:assert noise==observation['ancestral_noise_observations']
                    saved=save_image(args.output_dir/arm/'pixels'/f'{query_index:03d}_{query["array_row"]:05d}.webp',image)
                    if not bool(torch.any(beta!=0)):assert saved['generated_pixel_sha256']==old['generated_pixel_sha256']
                    row=query|proof|saved|dict(arm=arm,global_query_index=query_index,prediction_row=position,
                        condition_sha256=base['condition_sha256'],condition_coordinates=base['condition_coordinates'],
                        condition_FP16_sha256=base['condition_FP16_sha256'],tokens_sha256=base['tokens_sha256'],
                        measurement_file_sha256=complete['arms'][arm]['measurements_sha256'],
                        beta_file_sha256=complete['arms'][arm]['beta_sha256'],measurement_coordinates='64 DCT coefficients of native VAE MODE * one scaling_factor',
                        beta_nonzero_coefficients=int((beta>0).sum()),beta_max=float(beta.max()),
                        native_condition_identical=True,one_image_per_query=True,oracle_normalization=False,
                        normal_inference_target_access=False,repeat_protocol='R4',panel_already_exposed_development=True,
                        method='soft_spectral_native_'+arm,baseline_artifact_reused=False)
                    generated.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                if (query_index+1)%10==0:
                    print(json.dumps(dict(stage='SOFT_SPECTRAL_SCREEN',queries=query_index+1,new_candidate_images=2*(query_index+1),reused_native_images=query_index+1)),flush=True)
                assert safe_bytes_used(args.output_dir)<OUTPUT_CAP-(16<<20)
        assert len(generated)==240 and parity_passed
        manifest=G.attach_targets(G.metadata(args.image_manifest),generated,args.output_dir,np,Image)
        for arm in ARMS[1:]:
            for key in ('measurements','beta'):
                assert sha(complete['arms'][arm][key+'_path'])==complete['arms'][arm][key+'_sha256']
        assert sha(args.fit_completion)==binding['fit_completion']['sha256']
        assert sha(args.native_parent_binding)==binding['native_parent_binding']['sha256']
        budget.check()
        atomic(args.output_dir/'completion.json',dict(status='COMPLETED_GENERIC_NATIVE_PREDICTION_SCREEN',schema=P.SCHEMA,
            method_schema=SCHEMA,queries=80,arms=list(ARMS),images=240,per_arm_images={arm:80 for arm in ARMS},
            manifest=ref(manifest),inference_manifest=ref(inference),panel=ref(args.panel),input_provenance=ref(args.output_dir/'input_provenance.json'),
            host=host,loader_proof=loader_proof,wrapper=wrapper,native_NFE_per_image=4,resolution=[512,512],CFG=0,IP_scale=1,
            new_candidate_images=160,reused_native_images=80,parity_images=1,new_UNet_NFE=644,
            actual_zero_beta_pixel_parity=ref(args.output_dir/'zero_beta_pixel_parity.json'),
            no_best_of_N=True,normal_inference_target_access=False,target_pixels_loaded_after_generation=True,
            panel_already_exposed_development=True,independent_confirmation=False,SOTA_established=False,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-invoked,
            prespecified_additional_pair=['raw_eeg','c_only'],extra_UNet_evaluations_per_candidate_image=0))
        return 0
    except BaseException as error:
        atomic(args.output_dir/'failure.json',dict(status='FAILED_SPECTRAL_SCREEN_PRESERVED_NO_AUTOMATIC_RETRY',
            error=repr(error),observed_manifest_rows=len(generated),expected_rows=240,expected_new_candidate_images=160,
            actual_parity_passed=bool(parity_passed),seconds=time.monotonic()-invoked,no_effect_claim=True))
        raise


if __name__=='__main__':
    raise SystemExit(main())
