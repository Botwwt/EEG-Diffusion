"""Pinned native four-step renderer for no-grad engineering diagnostics.

Import after runtime setup. No UNet, A/P, VAE, CLIP or ordinary weight is fitted.
"""
import gc,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import EulerAncestralDiscreteScheduler
from brainhive_bounded_component_v1 import BoundedConditionAdapter
from brainhive_component_generator_v1 import load_native,denoise
from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit


def module_digest(module):
    digest=hashlib.sha256()
    for name,value in module.state_dict().items():digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenSemanticRuntime:
    def __init__(self,specification,out):
        s=specification
        source=importlib.util.spec_from_file_location('pinned_CLIP_visual_followon',s['CLIP_source']['path']);module=importlib.util.module_from_spec(source);source.loader.exec_module(module)
        jit=torch.jit.load(s['CLIP_weights']['path'],map_location='cpu').eval()
        weights={name[len('visual.'):]:value for name,value in jit.state_dict().items() if name.startswith('visual.')}
        self.critic=module.VisionTransformer(224,14,1024,24,16,768).eval().requires_grad_(False);self.critic.load_state_dict(weights,strict=True);self.critic.cuda().float();del weights,jit;gc.collect()
        (out/'loader').mkdir();audit=ConvertedWeightAudit(torch,out/'loader');audit.install()
        try:self.pipe,self.A,self.P,self.prompts,self.host=load_native(training=False)
        finally:audit.restore()
        self.loading=audit.complete()
        self.ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
        self.ordinary.load_state_dict(torch.load(s['ordinary']['path'],map_location='cpu',weights_only=True)['model'],strict=True)
        self.modules=dict(A=self.A,P=self.P,UNet=self.pipe.unet,VAE=self.pipe.vae,CLIP=self.critic,ordinary=self.ordinary)
        self.original={name:module_digest(model) for name,model in self.modules.items()}
        self.pipe.vae.float();self.pipe.vae.enable_slicing();self.VAE_float_digest=module_digest(self.pipe.vae)
        self.scheduler_config=json.loads(Path(s['scheduler_config']['path']).read_bytes());self.renders=0;self.total_UNet_calls=0

    @staticmethod
    def preprocess(image):
        image=F.interpolate(image,size=(224,224),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        mean=image.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None];std=image.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
        return (image-mean)/std

    @torch.no_grad()
    def target_feature(self,image):return F.normalize(self.critic(self.preprocess(image)).float(),dim=-1)

    @torch.no_grad()
    def render(self,condition,seed):
        self.pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(self.scheduler_config)
        generator=torch.Generator(device='cuda').manual_seed(int(seed));calls=[]
        hook=self.pipe.unet.register_forward_hook(lambda *unused:calls.append(1))
        try:
            tokens=self.P(self.A(condition.half()))
            z=self.pipe(height=512,width=512,generator=generator,prompt_embeds=self.prompts[0],pooled_prompt_embeds=self.prompts[1],cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='latent').images
        finally:hook.remove()
        assert len(calls)==4
        image=self.pipe.vae.decode(z.float()/self.pipe.vae.config.scaling_factor,return_dict=False)[0]
        feature=F.normalize(self.critic(self.preprocess((image*.5+.5).clamp(0,1))).float(),dim=-1)
        assert torch.isfinite(z).all() and torch.isfinite(feature).all()
        self.renders+=1;self.total_UNet_calls+=4
        return feature,z,generator.get_state().clone()

    @torch.no_grad()
    def manual_latent(self,condition,seed):
        scheduler=EulerAncestralDiscreteScheduler.from_config(self.scheduler_config);scheduler.set_timesteps(4,device='cuda')
        generator=torch.Generator(device='cuda').manual_seed(int(seed))
        z=self.pipe.prepare_latents(1,4,512,512,self.prompts[0].dtype,torch.device('cuda'),generator,None)
        tokens=self.P(self.A(condition.half()))
        for timestep in scheduler.timesteps:
            scaled=scheduler.scale_model_input(z,timestep)
            epsilon=denoise(self.pipe,self.prompts,scaled,timestep,tokens)
            z=scheduler.step(epsilon,timestep,z,generator=generator).prev_sample
        self.total_UNet_calls+=4
        return z,generator.get_state().clone()

    def frozen_audit(self):
        assert self.original['VAE']==module_digest(self.pipe.vae.half());self.pipe.vae.float();assert self.VAE_float_digest==module_digest(self.pipe.vae)
        assert all(self.original[name]==module_digest(model) for name,model in self.modules.items() if name!='VAE')
        assert all(not model.training and all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()) for model in self.modules.values())
        return dict(frozen_weights_unchanged=True,all_frozen_parameters_without_gradients=True,rendered_engineering_conditions=self.renders,total_UNet_calls=self.total_UNet_calls)
