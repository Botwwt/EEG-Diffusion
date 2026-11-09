"""Persist the exact native VAE subset and bind all existing image-domain arrays."""
import argparse,json,time
from pathlib import Path
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_native_vae_subset_20261027_v1 as S

def main():
    p=argparse.ArgumentParser()
    for key in ('fit-completion','fit-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();torch.set_num_threads(2);assert not a.output.exists();a.output.mkdir(parents=True)
    fit=U.read(a.fit_completion);audit=U.read(a.fit_audit);assert audit['completion']==U.ref(a.fit_completion)
    assert audit['original_VAE_subset_load_and_widened_FP32_weights_exact'] is True
    root=Path(__file__).resolve().parent
    import brainhive_paired_study_contract_20261025_v1 as C
    config=C.read(root/'configs/paired_effect_20261025_v1/study_review_v3.json');C.setup_runtime(config)
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors import safe_open
    from safetensors.torch import save_file
    reg=U.read(root/'configs/ATM_public_decoder_registration_20261020_v1.json');payload=next(x for x in reg['generator_assets'] if 'sdxl_turbo_fp16' in x['path']);U.checked(payload)
    cfg_path=Path(reg['generator_asset_root'])/'sdxl_config/vae/config.json';cfg=U.read(cfg_path)
    with safe_open(payload['path'],framework='pt',device='cpu') as f:source={key:f.get_tensor(key) for key in f.keys() if key.startswith('first_stage_model.')}
    state=convert_ldm_vae_checkpoint(source,cfg);assert sum(v.numel() for v in state.values())==83653863 and all(v.dtype==torch.float16 for v in state.values())
    expected=U.read(a.fit_completion.parent/'provenance.json')['VAE']['original_FP16_VAE_state_sha256'];assert S.state_sha(state)==expected
    target=a.output/'native_VAE_FP16_exact.safetensors';save_file({key:value.contiguous() for key,value in state.items()},str(target))
    (a.output/'config.json').write_bytes(cfg_path.read_bytes())
    prepared=U.read(U.checked(fit['prepared_inputs']));bindings=dict(encoded_latents=fit['encoded_latents'],conditions=prepared['conditions'],mapping=prepared['mapping'],prepared_inputs=fit['prepared_inputs'])
    for item in list(bindings['encoded_latents'].values())+list(bindings['conditions'].values())+[bindings['mapping'],bindings['prepared_inputs']]:U.checked(item)
    result=dict(status='NATIVE_VAE_SUBSET_EXACT_ORIGINAL_WEIGHTS_AND_EXISTING_IMAGE_DOMAIN_BANK_BOUND',source=U.ref(__file__),loader_source=U.ref(S.__file__),
        original_checkpoint=payload,original_config=U.ref(cfg_path),original_FP16_VAE_state_sha256=expected,VAE_weights=U.ref(target),VAE_config=U.ref(a.output/'config.json'),
        fit=U.ref(a.fit_completion),fit_CPU_audit=U.ref(a.fit_audit),existing_image_domain_inputs=bindings,
        exact_parameters=83653863,states=len(state),VAE_subset_bytes=target.stat().st_size,original_sampler_terminal_latents=False,
        Dcheck_target_pixels_or_F_present=False,reuse_requires_original_host_VAE_and_preprocess_and_image_source_SHAs=True,
        new_GPU_jobs=0,new_images=0,new_fit=0,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.monotonic()-start)
    assert result['CUDA_initialized'] is False;U.atomic(a.output/'completion.json',result);print(json.dumps({k:result[k] for k in ['status','seconds','VAE_subset_bytes','exact_parameters']}))

if __name__=='__main__':main()
