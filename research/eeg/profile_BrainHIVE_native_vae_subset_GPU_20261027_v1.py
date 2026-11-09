"""Two exact decode pixel parities and one original image encoder replay, VAE only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_native_vae_subset_20261027_v1 as V

def main():
    p=argparse.ArgumentParser()
    for key in ('spec','native-registration','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args();start=time.monotonic()
    assert a.max_seconds==90 and not a.output.exists();a.output.mkdir(parents=True);spec=U.read(a.spec)
    for ref in spec['source_pins']:U.checked(ref)
    import brainhive_paired_study_contract_20261025_v1 as C
    config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    import fit_BrainHIVE_image_domain_GPU_20261027_v2 as FIT
    import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N
    D.torch=torch;document=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=90,maximum_output_bytes=16<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,
        minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
    wrapper=D.verify_wrapper(document);budget=D.Budget(document);torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;C.resources(config,training_start=True)
    load_start=time.monotonic();pipe,proof=V.load(U.checked(spec['subset_completion']));load_seconds=time.monotonic()-load_start;budget.check()
    fit=U.read(U.checked(spec['fit_completion']));parity=U.read(U.checked(fit['VAE_parity']));stage=U.read(U.checked(spec['image_completion']))
    row=json.loads(U.checked(stage['manifest']).read_text().splitlines()[0]);assert row['arm']=='vae_roundtrip'
    lock=U.read(U.checked(fit['edit_lock']));bank=np.load(U.checked(lock['latents']),mmap_mode='r');observations=[]
    for name,z,expected in [('T_decode',np.load(U.checked(parity['latent'])),parity['image']),('Dcheck_decode',np.array(bank[0,0:1],copy=True),row)]:
        out=N.save_image(a.output/(name+'.webp'),FIT.decode(pipe,torch.from_numpy(z).cuda())[0])
        for key in ('generated_sha256','generated_pixel_sha256'):assert out[key]==expected[key]
        observations.append(dict(name=name,image=out,exact_file_and_pixel_parity=True))
    prepared=U.read(U.checked(fit['prepared_inputs']));mapping=U.read(U.checked(prepared['mapping']));source=mapping['Dcheck400'][0];payload=U.checked(source['generated']).read_bytes()
    with torch.no_grad():actual=pipe.vae.encode(FIT.image_tensor(payload)[None].cuda()).latent_dist.mode()*pipe.vae.config.scaling_factor
    expected=np.load(U.checked(fit['encoded_latents']['Dcheck400']),mmap_mode='r')[0:1]
    error=float(np.linalg.norm(actual.cpu().numpy()-expected)/np.linalg.norm(expected));assert error<3e-5
    budget.check();assert time.monotonic()-start<80
    for ref in spec['source_pins']:U.checked(ref)
    result=dict(status='NATIVE_VAE_ONLY_GPU_TWO_EXACT_PIXEL_PARITIES_AND_ORIGINAL_ENCODER_REPLAY_PASSED',source=U.ref(__file__),spec=U.ref(a.spec),wrapper=wrapper,
        loader=proof,subset_completion=spec['subset_completion'],two_decode_pixel_parities=observations,original_encoder_relative_L2=error,
        no_UNet_text_prior_or_EEG_model_allocated=True,new_UNet_calls=0,new_parameter_updates=0,new_query_images=0,new_engineering_parity_images=2,
        load_seconds=load_seconds,seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
        original_D_resumed=False,no_scientific_recipe_or_result_change=True)
    U.atomic(a.output/'completion.json',result);print(json.dumps({key:result[key] for key in ['status','load_seconds','seconds','peak_extra_CPU_bytes','peak_GPU_bytes','original_encoder_relative_L2']}))

if __name__=='__main__':main()
