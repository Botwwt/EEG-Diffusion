"""T-only CPU diagnosis of sampled native vectors, projection, and epsilon errors.

No GPU contention, no training, no formal image/seed selection. CPU RNG is a
diagnostic stream; formal image inference retains its registered CUDA streams.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path[:0]=['/root/mced_representation_runtime_v1/lib/python3.12/site-packages','/root/atm_official_20261016/Generation']
import torch
import torch.nn.functional as F
from diffusion_prior import DiffusionPriorUNet
from diffusers import DDPMScheduler
from diffusers.models.embeddings import ImageProjection
from diffusers.utils.torch_utils import randn_tensor
from safetensors import safe_open


@torch.inference_mode()
def main():
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(1729)
    cfg=json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    training=Path(cfg['output_root'])/'full';proof=json.loads((training/'sub-01/completion.json').read_text())
    sample=json.loads((ROOT/'audit/budget_sampler_profile.json').read_text());rows=[r['array_row'] for r in sample['files']]
    cache=torch.load(proof['condition_cache'],map_location='cpu',weights_only=True)
    c=cache['condition'][rows]
    y=torch.load(cfg['visual_features_path'],weights_only=True,map_location='cpu',mmap=True)[rows].clone()
    model=DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval().requires_grad_(False)
    model.load_state_dict(torch.load(proof['prior']['checkpoint'],weights_only=True,map_location='cpu'),strict=True)
    projector=ImageProjection(image_embed_dim=1024,cross_attention_dim=2048,num_image_text_embeds=4).eval().requires_grad_(False)
    with safe_open('/dev/shm/eeg_diffusion_next_20261018_v1/adapter.safetensors',framework='pt',device='cpu') as file:
        weights={new:file.get_tensor(old).float() for new,old in [
            ('image_embeds.weight','image_proj.proj.weight'),('image_embeds.bias','image_proj.proj.bias'),
            ('norm.weight','image_proj.norm.weight'),('norm.bias','image_proj.norm.bias')]}
    projector.load_state_dict(weights,strict=True)
    def sample_prior(clip):
        scheduler=DDPMScheduler(clip_sample=clip);scheduler.set_timesteps(50)
        gens=[torch.Generator(device='cpu').manual_seed(1729+i) for i in range(len(rows))]
        h=randn_tensor((len(rows),1024),generator=gens,device='cpu')
        for t in scheduler.timesteps:
            times=torch.full((len(h),),float(t))
            cond=model(h,times,c);uncond=model(h,times,None)
            h=scheduler.step(uncond+5*(cond-uncond),int(t),h,generator=gens).prev_sample
        assert torch.isfinite(h).all();return h
    values=dict(real_native=y,EEG_native_point=c,prior_unclipped_CFG5=sample_prior(False),prior_author_clipped_CFG5=sample_prior(True))
    true_tokens=projector(y).flatten(1);statistics={}
    for name,x in values.items():
        tokens=projector(x).flatten(1)
        statistics[name]=dict(norms=x.norm(dim=-1).tolist(),raw_cosines_to_true_native=F.cosine_similarity(x,y).tolist(),
            native_MSE_per_query=(x-y).square().mean(-1).tolist(),projected_token_cosines_to_true=F.cosine_similarity(tokens,true_tokens).tolist(),
            projected_token_RMS_per_query=tokens.square().mean(-1).sqrt().tolist(),coordinate_min=float(x.min()),coordinate_max=float(x.max()))
    scheduler=DDPMScheduler(clip_sample=False);noise_gen=torch.Generator(device='cpu').manual_seed(1729);errors=[]
    for timestep in [0,250,500,750,980]:
        noise=torch.randn(y.shape,generator=noise_gen);times=torch.full((len(rows),),timestep,dtype=torch.long)
        noisy=scheduler.add_noise(y,noise,times);conditional=model(noisy,times,c);unconditional=model(noisy,times,None)
        alpha=scheduler.alphas_cumprod[timestep]
        for guidance in [1.,5.]:
            eps=unconditional+guidance*(conditional-unconditional)
            x0=(noisy-(1-alpha).sqrt()*eps)/alpha.sqrt()
            errors.append(dict(timestep=timestep,diagnostic_CFG=guidance,alpha_cumprod=float(alpha),
                epsilon_MSE=float((eps-noise).square().mean()),implied_x0_MSE=float((x0-y).square().mean())))
    report=dict(status='CPU_CONDITION_DIAGNOSIS_COMPLETED_ACTUAL_IMAGES_STILL_REQUIRED',rows=rows,subject='sub-01',role='T-only',
        CPU_prior_noise_stream='fixed diagnostic1729+i, same streams for clipping variants; differs from formal CUDA generator algorithm',
        model_checkpoint=proof['prior']['checkpoint'],model_checkpoint_sha256=proof['prior']['checkpoint_sha256'],
        statistics=statistics,epsilon_error_by_timestep=errors,seconds=time.monotonic()-started,
        native_dimension=1024,projected_tokens=[4,2048],trained_parameters_changed=False,formal_method_or_seed_selected=False,
        inference_gate='Do not classify norm shifts as an image failure without the actual same-seed T-image diagnostic; native projector includes LayerNorm')
    (ROOT/'audit/prior_native_cpu_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(dict(rows=rows,conditions=values),ROOT/'audit/prior_native_cpu_diagnostic_vectors.pt')
    print(json.dumps(dict(status=report['status'],seconds=report['seconds'],statistics=statistics)),flush=True)


if __name__=='__main__':main()
