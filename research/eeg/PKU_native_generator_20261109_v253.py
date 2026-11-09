"""Pinned author native A/P/IP, exact frozen Turbo, WWT deployment."""
import ast,json,logging
from pathlib import Path
from typing import Optional
import torch
from torch import nn
from safetensors.torch import load_file
from PKU_HIVE_native_classes_20261109_v239 import ROOT,classes,sha

def load_native():
    from diffusers import StableDiffusionXLPipeline
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.configuration_utils import ConfigMixin,register_to_config
    from diffusers.models.attention_processor import Attention,AttnProcessor2_0
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    class Notice(logging.Filter):
        def filter(self,r):return r.getMessage()!="cross_attention_kwargs ['ip_hidden_states'] are not expected by AttnProcessor2_0 and will be ignored."
    logging.getLogger('diffusers.models.attention_processor').addFilter(Notice())
    proof=json.loads((ROOT/'runs/archive_verify_5096956/completion.json').read_text())
    assert proof['status']=='PASS_ALL_FILE_BYTES_AND_SHA256' and len(proof['files'])==22
    for r in proof['files']:
        assert r['verified_sha256']==r['sha256'] and Path(r['remote']).stat().st_size==r['bytes']
    scope=classes();scope.update(dict(Optional=Optional,ModelMixin=ModelMixin,ConfigMixin=ConfigMixin,register_to_config=register_to_config,
        Attention=Attention,AttnProcessor2_0=AttnProcessor2_0,UNet2DConditionModel=UNet2DConditionModel))
    source=ROOT/'code/HIVE_adapter_source_v253.py';pins=json.loads((ROOT/'code/HIVE_generator_source_pins_v253.json').read_text());assert sha(source)==pins[source.name]
    selected=['IPAttnProcessor2_0','IPProjectionModel','IPAttnAdapterModel']
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name in selected];assert [n.name for n in nodes]==selected
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    assets=ROOT/'assets/native_generator_v243';prior=assets/'prior_H14';turbo=assets/'turbo'
    cfg=json.loads((prior/'fusion_encoder/config.json').read_text());adapter=scope['ResNet1DLayer'](cfg['hidden_size'],cfg['intermediate_size'],norm=False)
    weights=load_file(str(ROOT/'assets/native_teacher_v239/fusion_encoder.safetensors'))
    aw={k[len('adapter.'):]:v for k,v in weights.items() if k.startswith('adapter.')};adapter.load_state_dict(aw,strict=True);del weights,aw
    modules={}
    for name,cls in [('proj','IPProjectionModel'),('attn_adapter','IPAttnAdapterModel')]:
        cfg=json.loads((prior/name/'config.json').read_text());m=scope[cls](**{k:v for k,v in cfg.items() if not k.startswith('_')})
        weights=load_file(str(prior/name/'diffusion_pytorch_model.safetensors'));assert set(weights)==set(m.state_dict());m.load_state_dict(weights,strict=True);del weights
        modules[name]=m.eval().requires_grad_(False).to(dtype=torch.float16)
    counts={k:sum(p.numel() for p in v.parameters()) for k,v in dict(A=adapter,P=modules['proj'],IP=modules['attn_adapter']).items()}
    assert counts==dict(A=8388608,P=8400896,IP=340787200),counts
    pipe=StableDiffusionXLPipeline.from_single_file(str(turbo/'sd_xl_turbo_1.0_fp16.safetensors'),config=str(turbo),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    counts.update(UNet=sum(p.numel() for p in pipe.unet.parameters()),VAE=sum(p.numel() for p in pipe.vae.parameters()))
    assert counts['UNet']==2567463684 and counts['VAE']==83653863,counts
    assert type(pipe.scheduler).__name__=='EulerAncestralDiscreteScheduler' and pipe.scheduler.config.timestep_spacing=='trailing' and pipe.scheduler.config.prediction_type=='epsilon'
    attn=modules['attn_adapter'];assert tuple(pipe.unet.attn_processors)==tuple(attn.attn_processor_keys);attn.bind_unet(pipe.unet)
    for m in pipe.unet.modules():
        if type(m).__name__=='Attention':
            for key in ['norm_q','norm_k']:
                if not hasattr(m,key):setattr(m,key,None)
    pipe.to('cuda');adapter.to(device='cuda',dtype=torch.float16);projector=modules['proj'].cuda()
    for m in list(pipe.components.values())+[adapter,projector]:
        if isinstance(m,nn.Module):m.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():prompt,_,pooled,_=pipe.encode_prompt('',device='cuda',do_classifier_free_guidance=False)
    prompt=prompt.detach().clone();pooled=pooled.detach().clone()
    timeids=pipe._get_add_time_ids((512,512),(0,0),(512,512),dtype=torch.float16,text_encoder_projection_dim=pooled.shape[-1]).cuda()
    return pipe,adapter,projector,(prompt,pooled,timeids),dict(counts=counts,IP_layers=len(attn.attn_processors),author_classes_unchanged=True,
        frozen_weight_dtype='FP16; VAE native force-upcast at decode',generator_revision='71153311d3dbb46851df1931d3ca6e939de83304',prior_revision='135a284109e71f5cde3d3e6d5dee6268a3c7987a',
        dependency_versions='Torch2.3.0cu121/diffusers0.36.0/transformers4.44.2',all_assets_full_CPU_SHA_job='5096956')

def denoise(pipe,prompts,z,t,tokens):
    prompt,pooled,timeids=prompts;n=len(z)
    return pipe.unet(z.to(pipe.unet.dtype),t,encoder_hidden_states=prompt.expand(n,-1,-1),
        added_cond_kwargs=dict(text_embeds=pooled.expand(n,-1),time_ids=timeids.expand(n,-1)),cross_attention_kwargs=dict(ip_hidden_states=tokens),return_dict=False)[0]
