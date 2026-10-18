"""SERVER ONLY: actual source-heldout generation for native public-prior reference.

Frozen conditionalDDPM and matched point/native queries, all through A/P.
Source-only locked new image indices with reused concepts; no oracle selection.
"""
import ast
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import sys
import time
import traceback
from typing import Dict,List,Tuple,Union,Optional

assert sys.platform=='linux'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
R=Path('/root/autodl-tmp/mced_research_bundle');B=Path('/root/autodl-tmp/mced_bootstrap_20261011')
OUT=Path('/root/mced_eeg_native_velocity_prior_samples_v2');TRAIN=Path('/root/mced_eeg_native_velocity_prior_training_v1')
BASE=Path('/root/mced_eeg_native_conditional_prior_samples_v1');PLAN=Path('/root/mced_eeg_native_velocity_prior_image_plan_v1')
NATIVE=Path('/root/mced_eeg_brainhive_feature_contract_v1');AUDIT=Path('/root/mced_eeg_brainhive_reference_audit_v1')
PRIOR=Path('/root/mced_eeg_brainhive_ram_assets_v1');PARENT=Path('/root/mced_eeg_generation_ram_assets_v4')
assert not OUT.exists();OUT.mkdir();ASSET=OUT/'assets';ASSET.mkdir();START=time.monotonic();FDS=[];HOOKS=[];SAVED=None
SEEDS=[2026109501,2026109502]
ARMS=[mode+'_'+str(seed) for mode in ['velocity_CFG5','velocity_CFG5_shift','unchanged','point','point_fit_radius','DDPM_CFG5'] for seed in SEEDS]+['constant_fit_mean','true_image_oracle']
COUNTS=dict(UNet_calls=0,VAE_decoder_calls=0,text_encoder_calls=0,optimizer_updates=0,EEG_calls=0,EEG_states=0)

def identity(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())

def write(n,v,replace=False):
    p=OUT/n
    if replace:
        t=p.with_name(p.name+'.pending');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
    else:
        with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def validate(d):
    c=json.loads((d/'final_commit.json').read_text());assert c['complete']
    for n,v in c['files'].items():assert identity(d/n)==v

def adopt(path,wanted):
    fd=os.open(path,os.O_RDONLY);FDS.append(fd);own=Path('/proc')/str(os.getpid())/'fd'/str(fd)
    assert fcntl.fcntl(fd,1034)==15 and identity(own)==wanted
    assert own.stat().st_ino==Path(path).stat().st_ino
    return own

try:
    write('plan.json',dict(script=identity(Path(__file__)),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),
        arms=ARMS,images_per_arm=32,new_images=128,reused_images=320,training_seeds=SEEDS,prior_training_seeds=[2026110101,2026110102],
        primary='Same locked32images; standardvpriorCFG5 versus pointfitradius,unchanged,rawpoint andoldepsilonCFG5; originalfixedcontrols reused',
        controls='Bothseed velocity cyclicshifts; immutable point/unchanged/epsilon/fitmean/trueimageoracle controls. CFG5fixed,no bestofN.',
        sampler='Strict official SDXL-Turbo, public SynCLR pretrained attention and projection; FP32/TF32off,4steps,CFG0,512px,batch1,noise2026110300+source index',
        queue='Retain read-only sealed weights; load GPU only after original nine-source sampler and independent metrics exit0, to avoid simultaneous large generator loads',
        no_official_test_or_target_payload=True,not_a_full_paper_reproduction=True,novelty_established=False,SOTA=False))
    # Pin shared reference and generator backings now, before waiting.
    validate(PRIOR);validate(AUDIT);validate(NATIVE);validate(PLAN)
    parents=json.loads((PRIOR/'results.json').read_text());assert parents['complete'] and parents['all_whole_SHA256_verified']
    owned={};adopted=[]
    for a in parents['assets']:
        if a['kind']!='weight':continue
        owned[a['path']]=adopt(a['memfd_path'],{k:a[k] for k in ['bytes','sha256']})
        adopted.append(dict(name=a['path'],memfd_path=str(owned[a['path']]),**{k:a[k] for k in ['bytes','sha256']}))
    parent=json.loads((PARENT/'results.json').read_text());assert parent['complete'] and parent['all_whole_SHA256_verified']
    a=next(x for x in parent['assets'] if x['path']=='sdxl_turbo_fp16.safetensors')
    wanted=dict(bytes=6938081905,sha256='e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26')
    assert {k:a[k] for k in wanted}==wanted
    own=adopt(a['memfd_path'],wanted);(ASSET/'sdxl_turbo_fp16.safetensors').symlink_to(own)
    adopted.append(dict(name='sdxl_turbo_fp16.safetensors',memfd_path=str(own),**wanted))
    cm=PARENT/'config_manifest.json';assert identity(cm)==dict(bytes=6240,sha256='d67a0aae3e7eb5bed4a1876f600eb2fc7a62a00c3b9d6f76a8c8a950e1f7b857')
    for row in json.loads(cm.read_text())['files']:
        p=PurePosixPath(row['path']);assert not p.is_absolute() and '..' not in p.parts and '\\' not in row['path']
        assert identity(PARENT/p)=={k:row[k] for k in ['bytes','sha256']}
        q=ASSET/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((PARENT/p).read_bytes())
    native=json.loads((NATIVE/'results.json').read_text());a=next(x for x in native['exports'] if x['stage']=='pre_adapter')
    pre_path=adopt(a['memfd_path'],{k:a[k] for k in ['bytes','sha256']})
    write('asset_adoption.json',dict(complete=True,assets=adopted,derived_pre_feature_memfd=str(pre_path),shared_readonly_backing=True))
    exits=[B/(n+'.exit') for n in ['eeg_multisource_samples_v1','eeg_multisource_metrics_v1','eeg_brainhive_native_samples_v1','eeg_brainhive_native_metrics_v1','eeg_brainhive_teacher_confirmation_samples_v1','eeg_brainhive_teacher_confirmation_metrics_v1','eeg_native_conditional_prior_samples_v1','eeg_native_conditional_prior_metrics_v1']]
    while not (TRAIN/'final_commit.json').exists() or not all(p.exists() for p in exits):
        if (TRAIN/'failure.json').exists():raise RuntimeError('Native EEG source reference failed')
        assert time.monotonic()-START<21600
        write('waiting_status.json',dict(pid=os.getpid(),stage='WAITING_FOR_NATIVE_EEG_AND_ORIGINAL_NINE_SOURCE_GPU_QUEUE',
            native_training_complete=(TRAIN/'final_commit.json').exists(),original_sampler_terminal=exits[0].exists(),
            original_metrics_terminal=exits[1].exists(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=COUNTS),True)
        time.sleep(20)
    assert all(p.read_text().strip()=='0' for p in exits)
    assert (B/'eeg_brainhive_native_reference_v1.exit').read_text().strip()=='0'
    assert (B/'eeg_brainhive_condition_pullback_training_v1.exit').read_text().strip()=='0'
    validate(TRAIN)
    training_result=json.loads((TRAIN/'results.json').read_text());assert training_result['complete'] and training_result['counts']['optimizer_updates']==4500
    os.kill(training_result['pid'],0)
    assert json.loads((TRAIN/'plan.json').read_text())['script']==json.loads((PLAN/'plan.json').read_text())['velocity_training_source']
    prediction_export=next(r for r in training_result['exports'] if r['name']=='private_velocity_prior_predictions.pt')
    prediction_path=adopt(prediction_export['memfd_path'],{k:prediction_export[k] for k in ['bytes','sha256']})
    write('prior_prediction_adoption.json',dict(complete=True,independent_readonly_descriptor=str(prediction_path),**prediction_export))
    validate(BASE)
    base_results=json.loads((BASE/'results.json').read_text())
    assert shutil.disk_usage('/root').free>1024**3
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from diffusers import StableDiffusionXLPipeline
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.configuration_utils import ConfigMixin,register_to_config
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from diffusers.models.attention_processor import Attention,AttnProcessor2_0
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint,convert_ldm_vae_checkpoint,convert_ldm_clip_checkpoint,convert_open_clip_checkpoint
    from safetensors.torch import load_file
    cg=Path('/sys/fs/cgroup');s=dict(x.split() for x in (cg/'memory.stat').read_text().splitlines())
    host=int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text())+int(s['file'])-int(s['shmem'])
    assert torch.cuda.mem_get_info()[0]>20<<30 and host>30<<30
    SAVED=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),torch.get_num_threads(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Reuse only reviewed strict base-generator loader, excluding the old IP adapter.
    source=B/'eeg_generator_conditioning_training_v1.py'
    assert identity(source)==dict(bytes=21503,sha256='8d05598382a46b0f41c0eb97c54ab0570c20fa64b62fc9b6a1694c0385e4a659')
    body=next(n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.Try)).body
    lo=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pipe' for t in n.targets))
    hi=next(i for i,n in enumerate(body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='load_ip_adapter')
    scope=dict(globals());exec(compile(ast.Module(body=body[lo:hi],type_ignores=[]),str(source),'exec'),scope)
    pipe=scope['pipe'];write('strict_base_weight_coverage.json',scope['coverage'])
    assert pipe.unet.encoder_hid_proj is None
    scope=dict(torch=torch,nn=nn,F=F,Dict=Dict,List=List,Tuple=Tuple,Union=Union,Optional=Optional,ModelMixin=ModelMixin,
               ConfigMixin=ConfigMixin,register_to_config=register_to_config,UNet2DConditionModel=UNet2DConditionModel,Attention=Attention,AttnProcessor2_0=AttnProcessor2_0)
    for filename,sha,names in [('main/layers.py','560b038708652fac9baa98a1f8f642812eacb7bbb172a56e2c8199ebbdc5b559',['BaseMLP','ResNet1DLayer','ResNet1DBlock']),
        ('main/models_adapter.py','b6d6322b7970af91a748dafbe779a8e77a1db489446cd6d607e506c36308760f',['IPAttnProcessor2_0','IPProjectionModel','IPAttnAdapterModel']),
        ('main/models_brain.py','fc06ed93c1f1f8659605ec561d87e85f91ca72856fe034f4cf8416c2b60b39b1',['BrainMLP'])]:
        p=AUDIT/'source'/filename;assert identity(p)['sha256']==sha
        nodes=[n for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name in names];assert [n.name for n in nodes]==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
    config=json.loads((PRIOR/'prior/attn_adapter/config.json').read_text());config={k:v for k,v in config.items() if not k.startswith('_')}
    assert tuple(config['attn_processor_keys'])==tuple(pipe.unet.attn_processors)
    attention=scope['IPAttnAdapterModel'](**config)
    weights=load_file(str(owned['attn_adapter/diffusion_pytorch_model.safetensors']),device='cpu')
    assert set(weights)==set(attention.state_dict());attention.load_state_dict(weights,strict=True);del weights
    attention.bind_unet(pipe.unet)
    config=json.loads((PRIOR/'prior/proj/config.json').read_text());config={k:v for k,v in config.items() if not k.startswith('_')}
    proj=scope['IPProjectionModel'](**config);weights=load_file(str(owned['proj/diffusion_pytorch_model.safetensors']),device='cpu')
    assert set(weights)==set(proj.state_dict());proj.load_state_dict(weights,strict=True);del weights
    residual=scope['ResNet1DLayer'](hidden_size=1024,intermediate_size=4096,norm=False)
    weights=load_file(str(owned['fusion_encoder/diffusion_pytorch_model.safetensors']),device='cpu')
    selected={k[len('adapter.'):]:v for k,v in weights.items() if k.startswith('adapter.')}
    assert set(selected)==set(residual.state_dict());residual.load_state_dict(selected,strict=True);del weights,selected
    pipe.to('cuda');proj=proj.cuda();residual=residual.cuda();pipe.set_progress_bar_config(disable=True)
    modules=[pipe.unet,pipe.vae,pipe.text_encoder,pipe.text_encoder_2,attention,proj,residual]
    for m in modules:m.eval().requires_grad_(False)
    fixed=[p for m in modules for p in list(m.parameters())+list(m.buffers())];versions=[p._version for p in fixed]
    protocol=json.loads((PLAN/'plan.json').read_text());ids=protocol['source_image_indices']
    assert ids==base_results['image_indices'] and len(ids)==len(set(ids))==len({i//10 for i in ids})==32
    private=torch.load(prediction_path,weights_only=True);source_ids=private['source_indices'].tolist()
    assert len(source_ids)==len(set(source_ids))==1280
    positions={index:i for i,index in enumerate(source_ids)};selected=[positions[i] for i in ids]
    conditions={}
    with torch.inference_mode():
        for seed in SEEDS:
            query=private['predictions'][str(seed)]['velocity_CFG5'][selected].cuda()
            assert query.shape==(32,1024) and torch.isfinite(query).all()
            conditions['velocity_CFG5_'+str(seed)]=proj(residual(query))
            conditions['velocity_CFG5_shift_'+str(seed)]=proj(residual(query.roll(1,0)))
    del private,query
    assert set(conditions)==set(protocol['new_arms']) and all(v.shape==(32,4,2048) for v in conditions.values())
    trace=[];run=[None]
    def unet_before(m,args,kwargs):
        x=args[0] if args else kwargs['sample'];t=args[1] if len(args)>1 else kwargs['timestep']
        c=kwargs['cross_attention_kwargs']['ip_hidden_states'];assert x.shape==(1,4,64,64) and c.shape==(1,4,2048)
        COUNTS['UNet_calls']+=1
        trace.append(dict(run=run[0],timestep=float(t),latent_sha256=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
                          condition_sha256=hashlib.sha256(c.detach().cpu().numpy().tobytes()).hexdigest()))
    def vae_before(m,args):COUNTS['VAE_decoder_calls']+=1
    def text_before(m,args):COUNTS['text_encoder_calls']+=1
    HOOKS=[pipe.unet.register_forward_pre_hook(unet_before,with_kwargs=True),pipe.vae.decoder.register_forward_pre_hook(vae_before),
           pipe.text_encoder.register_forward_pre_hook(text_before),pipe.text_encoder_2.register_forward_pre_hook(text_before)]
    cohorts=[];initials={int(k):v for k,v in base_results['paired_initial_latent_hashes'].items()}
    with torch.inference_mode():
        for arm in ARMS:
            if arm in protocol['reused_arms']:
                row=next(r for r in base_results['cohorts'] if r['arm']==arm)
                p=BASE/row['file'];assert identity(p)==row['identity'];(OUT/row['file']).symlink_to(p)
                cohorts.append(dict(**row,reused_immutable=True));continue
            images=[]
            for position,index in enumerate(ids):
                run[0]=arm+'/'+str(index);before=len(trace);gen=torch.Generator(device='cuda').manual_seed(2026110300+index)
                image=pipe(prompt='',cross_attention_kwargs={'ip_hidden_states':conditions[arm][position:position+1]},
                    height=512,width=512,generator=gen,num_inference_steps=4,guidance_scale=0.,output_type='np').images
                assert image.shape==(1,512,512,3) and np.isfinite(image).all() and len(trace)-before==4
                first=trace[before]['latent_sha256']
                if index in initials:assert initials[index]==first
                else:initials[index]=first
                images.append(np.rint(np.clip(image[0],0,1)*255).astype(np.uint8))
                if (position+1)%8==0:print(json.dumps(dict(stage='NATIVE_PRIOR_SOURCE_SAMPLE',arm=arm,images=position+1,counts=COUNTS)),flush=True)
            name=arm+'_samples.npz';np.savez_compressed(OUT/name,arr_0=np.stack(images),image_indices=np.asarray(ids,dtype=np.int64))
            row=dict(arm=arm,file=name,images=32,identity=identity(OUT/name),oracle_reference=arm.startswith('true_image_'))
            cohorts.append(row);write(arm+'_ready.json',row)
    assert COUNTS==dict(UNet_calls=512,VAE_decoder_calls=128,text_encoder_calls=256,optimizer_updates=0,EEG_calls=0,EEG_states=0),COUNTS
    assert [p._version for p in fixed]==versions and all(p.grad is None for p in fixed)
    write('trace.json',trace);write('input_bindings.json',dict(training_commit=identity(TRAIN/'final_commit.json'),
        image_plan_commit=identity(PLAN/'final_commit.json'),base_samples_commit=identity(BASE/'final_commit.json'),feature_contract_commit=identity(NATIVE/'final_commit.json'),
        prior_commit=identity(PRIOR/'final_commit.json'),private_prediction_export=prediction_export,
        additional_EEG_or_latent_prior_model_calls=0,only_explicit_oracle_arm_uses_true_source_images=True))
    write('results.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-START,counts=COUNTS,
        cohorts=cohorts,image_indices=ids,paired_initial_latent_hashes=initials,new_images=128,reused_images=320,
        source_heldout_only=True,frozen_generator_and_prior_unchanged=True,no_official_test_or_target_payload=True,
        matched_conditional_distribution_comparison=True,quality_metric=False,novelty_established=False,SOTA=False))
    write('final_commit.json',dict(complete=True,files={p.relative_to(OUT).as_posix():identity(p) for p in OUT.rglob('*') if p.is_file() and not p.is_symlink() and p.name!='waiting_status.json'}))
except BaseException:
    write('failure.json',dict(traceback=traceback.format_exc(),counts=COUNTS,seconds=time.monotonic()-START));raise
finally:
    for h in HOOKS:h.remove()
    if SAVED is not None:
        cpu,cuda,threads,tf32,cudnn=SAVED;torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda);torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=cudnn
    for fd in FDS:os.close(fd)
