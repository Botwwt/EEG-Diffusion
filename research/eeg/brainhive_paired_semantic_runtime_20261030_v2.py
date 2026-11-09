"""Batched native image training utilities, isolated from the old baselines."""
import gc,hashlib,importlib.util,inspect,json
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint

ARMS=('joint_correct','EEG_only_correct','P_only_correct','joint_swapped_targets','joint_absolute_only')

def fingerprint(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def tensor_sha(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def load_context(s,out):
    from brainhive_component_generator_v1 import load_native
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    import brainhive_native_models_v1 as N
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    spec=importlib.util.spec_from_file_location('paired_semantic_CLIP_visual',s['CLIP_source']['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    jit=torch.jit.load(s['CLIP_weights']['path'],map_location='cpu').eval()
    weights={k[7:]:v for k,v in jit.state_dict().items() if k.startswith('visual.')}
    critic=module.VisionTransformer(224,14,1024,24,16,768).eval().requires_grad_(False)
    critic.load_state_dict(weights,strict=True);critic.cuda().float();del weights,jit;gc.collect()
    (out/'loader').mkdir();audit=ConvertedWeightAudit(torch,out/'loader');audit.install()
    try:pipe,A,P,prompts,host=load_native(training=False)
    finally:audit.restore()
    loader=audit.complete();pipe.vae.float();pipe.vae.enable_slicing()
    ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
    ordinary.load_state_dict(torch.load(s['ordinary']['path'],map_location='cpu',weights_only=True)['model'],strict=True)
    initial=torch.load(s['parent']['path'],map_location='cpu',weights_only=True)['model']
    model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval();model.load_state_dict(initial,strict=True)
    ln=[(name,m) for name,m in model.named_modules() if isinstance(m,torch.nn.LayerNorm)];assert len(ln)==1
    name,LN=ln[0];LN.requires_grad_(False)
    parameters_E=[p for p in model.parameters() if p.requires_grad]
    master={n:torch.nn.Parameter(p.detach().float().clone()) for n,p in P.named_parameters()}
    assert sum(p.numel() for p in parameters_E)==6450176 and sum(p.numel() for p in master.values())==8400896
    frozen=dict(A=A,P=P,ordinary=ordinary,UNet=pipe.unet,VAE=pipe.vae,critic=critic)
    return dict(pipe=pipe,A=A,P=P,prompts=prompts,host=host,loader=loader,critic=critic,ordinary=ordinary,
        model=model,initial_E=initial,LN=LN,LN_name=name,E=parameters_E,master=master,frozen=frozen)

def optimizer(context,lr=1e-5):
    return torch.optim.AdamW([dict(params=context['E'],name='EEG'),dict(params=list(context['master'].values()),name='P')],lr=lr,betas=(.9,.999),weight_decay=.01)

def reset_initial(context):
    context['model'].load_state_dict(context['initial_E'],strict=True)
    with torch.no_grad():
        for name,value in context['P'].named_parameters():context['master'][name].copy_(value.float())
    for p in context['E']+list(context['master'].values()):p.grad=None

def tokens(context,eeg):
    from torch.func import functional_call
    c=context['model'](eeg.reshape(len(eeg),4250))
    ln=context['LN'];w=ln.weight[None].expand_as(c);b=ln.bias[None].expand_as(c)
    corrected=context['ordinary'](c,w,b)[0]
    return functional_call(context['P'],{n:p.half() for n,p in context['master'].items()},(context['A'](corrected.half()),)),c,corrected

def sample_pair(context,tk,seed,*,gradient):
    """Actual B=2 native pipeline; independent generators start at same state."""
    assert tk.shape==(2,4,2048)
    pipe=context['pipe'];prompt,pooled,_=context['prompts'];original_forward=pipe.unet.forward
    original_prepare=pipe.prepare_latents;noise_module=inspect.getmodule(type(pipe.scheduler));original_randn=noise_module.randn_tensor
    proof=dict(initial=None,ancestral=[],steps=[],UNet_NFE=0)
    def initial(*args,**kwargs):
        value=original_prepare(*args,**kwargs);assert torch.equal(value[0],value[1]);proof['initial']=tensor_sha(value[0]);return value
    def randn(*args,**kwargs):
        value=original_randn(*args,**kwargs);assert value.shape[0]==2 and torch.equal(value[0],value[1]);proof['ancestral'].append(tensor_sha(value[0]));return value
    def forward(*args,**kwargs):
        proof['UNet_NFE']+=1
        return original_forward(*args,**kwargs)
    def checkpointed(*args,**kwargs):return checkpoint(forward,*args,use_reentrant=False,**kwargs)
    def callback(p,i,t,values):
        proof['steps'].append(dict(step=i,timestep=float(t),latents_sha256=tensor_sha(values['latents'])))
        return values
    pipe.prepare_latents=initial;noise_module.randn_tensor=randn;pipe.unet.forward=checkpointed if gradient else forward
    generators=[torch.Generator(device='cuda').manual_seed(seed) for _ in range(2)]
    kwargs=dict(height=512,width=512,generator=generators,prompt_embeds=prompt.expand(2,-1,-1),
        pooled_prompt_embeds=pooled.expand(2,-1),cross_attention_kwargs={'ip_hidden_states':tk},
        num_inference_steps=4,guidance_scale=0.,output_type='latent',callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents'])
    try:
        if gradient:value=type(pipe).__call__.__wrapped__(pipe,**kwargs).images
        else:
            with torch.no_grad():value=pipe(**kwargs).images
    finally:pipe.unet.forward=original_forward;pipe.prepare_latents=original_prepare;noise_module.randn_tensor=original_randn
    assert value.shape==(2,4,64,64) and proof['UNet_NFE']==4 and len(proof['ancestral'])==4
    assert [v['timestep'] for v in proof['steps']]==[999.,749.,499.,249.]
    # The backward may re-evaluate UNet forward, but no scheduler callback or
    # noise draw is replayed by checkpointing; keep the forward proof detached.
    return value,dict(proof)

def objective(context,features,targets,arm):
    from brainhive_paired_semantic_effect_20261030_v1 import paired_objective
    assert arm in ARMS
    return paired_objective(features,targets.flip(0) if arm=='joint_swapped_targets' else targets,effect_enabled=arm!='joint_absolute_only')

def step(context,opt,arm):
    allowed_E=arm!='P_only_correct';allowed_P=arm!='EEG_only_correct';record={}
    for name,params,allowed in [('EEG',context['E'],allowed_E),('P',list(context['master'].values()),allowed_P)]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        assert norm>0
        record[name]=dict(gradient_norm_before_clip=float(norm),updated=allowed,parameters=sum(p.numel() for p in params))
        if not allowed:
            for p in params:p.grad=None
    opt.step();opt.zero_grad(set_to_none=True)
    return record

def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def snapshot(context,opt,*,arm,updates,order,registration_sha256,log_sha256):
    return dict(EEG=cpu_tree(context['model'].state_dict()),P=cpu_tree(context['master']),optimizer=cpu_tree(opt.state_dict()),
        CPU_RNG=torch.get_rng_state(),CUDA_RNG=[x.cpu() for x in torch.cuda.get_rng_state_all()],arm=arm,updates=updates,
        order=order.copy(),registration_sha256=registration_sha256,log_sha256=log_sha256,
        noise_rule='SHA256(1729|pair_index|fixed_pair_semantic_v1), same two independent generators within pair',
        ordinary_not_updated=True,future_GPU_bit_exactness_not_claimed=True)

def restore(context,opt,value,*,arm,order,registration_sha256):
    import numpy as np
    assert value['arm']==arm and value['registration_sha256']==registration_sha256 and np.array_equal(value['order'],order)
    assert 0<=value['updates']<=len(order)
    context['model'].load_state_dict(value['EEG'],strict=True)
    with torch.no_grad():
        assert set(value['P'])==set(context['master'])
        for n,p in context['master'].items():p.copy_(value['P'][n].to('cuda'))
    opt.load_state_dict(value['optimizer']);torch.set_rng_state(value['CPU_RNG']);torch.cuda.set_rng_state_all(value['CUDA_RNG'])
    assert all(torch.equal(p.detach().cpu(),value['EEG'][n]) for n,p in context['model'].named_parameters())
    assert all(torch.equal(p.detach().cpu(),value['P'][n]) for n,p in context['master'].items())
    return value['updates']
