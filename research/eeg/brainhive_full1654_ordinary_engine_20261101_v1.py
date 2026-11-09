"""Original ordinary objective on real fullscope inputs; no automatic fit admission."""
import hashlib
import os
from pathlib import Path
import sys
import brainhive_exposed_native_latent_contract_CPU_20261101_v1 as K

PROFILE_STATUS='REVIEWED_FULL1654_ORIGINAL_ORDINARY_ENGINEERING_ONLY'
METHOD=dict(seed=1729,parameters=525568,queries=165400,images=16540,subjects=10,
            passes=3,batch=16,last_batch=8,updates=31014,exposures=496200,
            lr=1e-4,weight_decay=.01,betas=[.9,.999],eps=1e-8,gradient_clip=1.,
            strength=1.,radius=.25,objective='epsilon_MSE_plus_.1_native_AP_token_cosine',
            fixed_final=True,old_Dfit_postfit_calibration=False)


def stable_seed(namespace):
    return int(hashlib.sha256(('1729|BrainHIVE_bounded_v1|'+namespace).encode()).hexdigest()[:16],16)%(2**63-1)


def validate_scope(s):
    assert s['status']==PROFILE_STATUS and s['host']=='old' and s['GPU_UUID']==K.GPU
    assert s['method']==METHOD and s['temporary_profile_updates']==4
    assert s['formal_training_admitted'] is False and s['benchmark_admitted'] is False
    assert s['official200_previously_exposed'] is True and s['official_results_used_for_previous_direction'] is True
    assert s['independent_H'] is False and s['R80_payloads_allowed'] is False
    assert s['maximum_output_bytes']==64<<20 and s['maximum_GPU_bytes']==20<<30
    assert s['maximum_extra_CPU_bytes']==10<<30 and s['minimum_start_CPU_headroom_bytes']==24<<30
    assert s['minimum_runtime_CPU_headroom_bytes']==10<<30
    assert s['legacy_floor_bytes']==50<<30 and s['disk_buffer_bytes']==8<<30
    assert s['upstream_remaining_budget_reserved_bytes']==14<<30
    assert s['maximum_GPU_seconds']==900 and s['numerical_guard']==dict(atol=1e-5,rtol=5e-6)
    assert s['future_GPU_bit_exactness_claimed'] is False
    return s


def setup_profile(path):
    s=validate_scope(K.read(path))
    for pin in s['source_pins']:K.check(pin)
    K.check(s['old_ordinary'])
    review=K.read(K.check(s['source_data_resource_review']))
    assert review['status']=='PASS_ORIGINAL_ORDINARY_PROFILE_SOURCE_SCOPE_RESOURCE_REVIEW'
    assert review['source_pins']==s['source_pins']
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,str(Path(path).parent))
    return s


def verified_CPU_inputs(s):
    import numpy as np
    import torch
    import brainhive_exposed_fullfit_component_inputs_CPU_20261101_v1 as I
    assert not torch.cuda.is_initialized()
    K.check(s['latent_protocol'])
    latent=K.read(Path(s['latent_output_root'])/'completion.json')
    assert latent['status']=='COMPLETE_EXPOSED_FULL16540_NATIVE_TRAIN_LATENT_INPUT_STAGE'
    assert latent['protocol']==s['latent_protocol'] and latent['parameter_updates']==0
    audit=K.read(K.check(latent['CPU_audit']))
    assert audit['status']=='PASS_ALL16540_LATENT_CHUNKS_AND_ALL13230_OLD_T_VALUE_REPLAY'
    assert audit['old_T_images_checked']==13230 and audit['bank']==latent['bank']
    bank=np.load(K.check(latent['bank']),mmap_mode='r',allow_pickle=False)
    assert bank.shape==(16540,4,64,64) and bank.dtype==np.float32
    inputs=I.load_verified(s['input_bank'],s['asset_protocol'])
    assert inputs['array_rows']==list(range(16540))
    assert not torch.cuda.is_initialized()
    return inputs,bank,dict(input_bank=s['input_bank'],asset_protocol=s['asset_protocol'],
                           latent_protocol=s['latent_protocol'],latents=latent['bank'],
                           latent_stage_completion=K.ref(Path(s['latent_output_root'])/'completion.json'),
                           latent_CPU_audit=latent['CPU_audit'],all165400_raw_OOF_and20_own_affines_verified=True)


def clone_CPU(value,torch):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:clone_CPU(v,torch) for k,v in value.items()}
    if isinstance(value,list):return [clone_CPU(v,torch) for v in value]
    if isinstance(value,tuple):return tuple(clone_CPU(v,torch) for v in value)
    return value


def module_digest(module):
    result=hashlib.sha256()
    for key,value in module.state_dict().items():
        result.update(key.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def load_host(s,out,torch):
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    converted=Path(out)/'converted';converted.mkdir()
    observer=ConvertedWeightAudit(torch,converted);observer.install()
    try:pipe,A,P,prompts,host=load_native(training=True)
    finally:observer.restore()
    proof=observer.complete()
    from diffusers import DDPMScheduler
    ddpm=DDPMScheduler.from_config(pipe.scheduler.config)
    assert ddpm.config.prediction_type=='epsilon' and ddpm.config.num_train_timesteps==1000
    assert all(not p.requires_grad for module in (pipe.unet,A,P) for p in module.parameters())
    assert {p.dtype for p in pipe.unet.parameters()}=={torch.float16}
    return pipe,A,P,prompts,ddpm,host,proof


def make_GPU_inputs(inputs,latents,torch):
    return dict(condition=torch.tensor(inputs['condition'],device='cuda'),
                affine_ids=torch.tensor(inputs['affine_ids'].astype('int64'),device='cuda'),
                weight=torch.tensor(inputs['weight'],device='cuda'),bias=torch.tensor(inputs['bias'],device='cuda'),
                target=inputs['target'].to('cuda'),latent=torch.tensor(latents,device='cuda'))


def orders(torch):
    return [torch.randperm(165400,generator=torch.Generator().manual_seed(stable_seed('pass'+str(i)))) for i in range(3)]


def ordinary_step(model,optimizer,pipe,A,P,prompts,ddpm,data,indices,noise_rng,torch,*,trace=False,baseline=False):
    import torch.nn.functional as F
    from brainhive_component_generator_v1 import denoise
    index=indices.cuda();position=index%16540;affine=data['affine_ids'][index]
    c=data['condition'][index];weight=data['weight'][affine];bias=data['bias'][affine]
    latent=data['latent'][position]
    noise=torch.randn(latent.shape,device='cuda',generator=noise_rng)
    t=torch.randint(0,1000,(len(index),),device='cuda',generator=noise_rng)
    noisy=ddpm.add_noise(latent,noise,t)
    optimizer.zero_grad(set_to_none=True)
    changed,info=model(c,weight,bias)
    if trace:changed.retain_grad()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        with torch.no_grad():teacher=P(A(data['target'][position]))
        tokens=P(A(changed))
        predicted=denoise(pipe,prompts,noisy,t,tokens)
        image_loss=F.mse_loss(predicted.float(),noise)
        token_loss=(1-F.cosine_similarity(tokens.float(),teacher.float(),dim=-1)).mean()
        loss=image_loss+.1*token_loss
        if baseline:
            assert torch.equal(changed,c)
            with torch.no_grad():native_tokens=P(A(c));native=denoise(pipe,prompts,noisy,t,native_tokens)
            assert torch.equal(tokens,native_tokens)
            assert torch.allclose(predicted.float(),native.float(),atol=1e-5,rtol=5e-6)
    assert torch.isfinite(loss)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    assert model.direction.weight.grad is not None and torch.isfinite(model.direction.weight.grad).all()
    assert float(model.direction.weight.grad.norm())>0
    assert all(p.grad is None for module in (pipe.unet,A,P) for p in module.parameters())
    record=dict(epsilon_loss=float(image_loss),token_loss=float(token_loss),loss=float(loss),
                gradient_norm_before_clip=float(norm),batch=len(index),timesteps=t.tolist(),
                mean_gate=float(info['gate'].mean()))
    if trace:
        record.update(indices=indices.clone(),condition=c.detach().cpu(),weight=weight.detach().cpu(),bias=bias.detach().cpu(),
                      changed_condition=changed.detach().cpu(),
                      predicted=predicted.detach().float().cpu(),noise=noise.detach().cpu(),
                      tokens=tokens.detach().float().cpu(),teacher_tokens=teacher.detach().float().cpu(),
                      changed_input_gradient=changed.grad.detach().cpu(),
                      clipped_parameter_gradients={key:value.grad.detach().cpu().clone() for key,value in model.named_parameters()})
    optimizer.step()
    return record


def save_tensor(path,value,torch):
    with Path(path).open('xb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
