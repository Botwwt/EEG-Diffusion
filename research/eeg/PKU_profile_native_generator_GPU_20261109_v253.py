"""Actual T EEG generation parity/input-gradient engineering; no efficacy fit."""
import hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ROOT=Path('/dfs/share/maleiLab/wwt/eeg');out=ROOT/('runs/generator_profile_'+os.environ['SLURM_JOB_ID']);out.mkdir();start=time.monotonic()
def write(name,v):
    b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();p=out/name;t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
try:
    import torch
    from torch.nn import functional as F
    from PKU_native_generator_20261109_v253 import load_native,denoise
    from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pipe,A,P,prompts,host=load_native();assert host['IP_layers']==70
    write('loaded_host.json',dict(host=host,load_seconds=time.monotonic()-start))
    base=ROOT/'runs/HIVE_native_sub03_5096542';bank=torch.load(base/'T_true_OOF_conditions.pt',map_location='cpu',weights_only=True,mmap=True)
    positions=[0,10];c=bank['condition'][positions].clone().cuda();affine=[]
    for pos in positions:
        role='oof'+str(int(bank['folds'][pos]));state=torch.load(base/role/'brain_final.pt',map_location='cpu',weights_only=True)['model']
        affine.append((state['block.layers.0.norm.weight'],state['block.layers.0.norm.bias']))
    w=torch.stack([a[0] for a in affine]).cuda();b=torch.stack([a[1] for a in affine]).cuda()
    torch.manual_seed(1729);head=BoundedConditionAdapter(predict_risk=False).cuda();assert sum(p.numel() for p in head.parameters())==525568
    with torch.no_grad():
        zero,_=head(c,w,b,gate_mode='fixed');disabled,_=head(c,w,b,enabled=False)
    assert torch.equal(c,zero) and torch.equal(c,disabled)
    pixels=[];calls=[0]
    def count_calls(*args):calls[0]+=1
    hook=pipe.unet.register_forward_pre_hook(count_calls)
    for i,pos in enumerate(positions):
        hashes=[]
        for kind,condition in [('native',c[i:i+1]),('zero_ordinary',zero[i:i+1])]:
            with torch.no_grad():tokens=P(A(condition.to(torch.float16)))
            seed=1729+pos;g=torch.Generator(device='cuda').manual_seed(seed)
            with torch.no_grad():
                im=pipe(prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],width=512,height=512,num_inference_steps=4,guidance_scale=0.,generator=g,cross_attention_kwargs=dict(ip_hidden_states=tokens)).images[0]
            assert im.size==(512,512);h=hashlib.sha256(im.tobytes()).hexdigest();im.save(out/(kind+'_'+str(pos)+'.png'));hashes.append(h)
        assert hashes[0]==hashes[1];pixels.append(dict(T_position=pos,array_row=int(bank['array_rows'][pos]),seed=1729+pos,native_pixel_SHA=hashes[0],zero_pixel_SHA=hashes[1],pixel_equal=True))
    assert calls[0]==16
    write('native_zero_pixel_parity.json',dict(images=pixels,UNet_calls=16,queries=2,efficacy_evaluation=False))
    frozen=[p for module in [pipe.unet,A,P] for p in module.parameters()];versions=[p._version for p in frozen]
    assert all(not p.requires_grad for p in frozen)
    pipe.unet.train();pipe.unet.enable_gradient_checkpointing();head.zero_grad(set_to_none=True)
    rng=torch.Generator(device='cuda').manual_seed(1729);noisy=torch.randn((2,4,64,64),generator=rng,device='cuda');target=torch.randn((2,4,64,64),generator=rng,device='cuda')
    timesteps=torch.tensor([500,750],device='cuda');fullF=torch.load(base/'deterministic_F_all16540.pt',map_location='cpu',weights_only=True,mmap=True);teacher=fullF[bank['array_rows'][positions]].clone().cuda()
    changed,_=head(c,w,b,gate_mode='fixed');bf_prompts=tuple(x.to(torch.bfloat16) for x in prompts)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        tokens=P(A(changed));pred=denoise(pipe,bf_prompts,noisy,timesteps,tokens)
        with torch.no_grad():real=P(A(teacher))
        loss=F.mse_loss(pred.float(),target)+.1*(1-F.cosine_similarity(tokens.float(),real.float(),dim=-1)).mean()
    assert torch.isfinite(loss);loss.backward();grad=float(head.direction.weight.grad.norm());assert grad>0 and torch.isfinite(head.direction.weight.grad).all()
    assert all(p.grad is None for p in frozen) and [p._version for p in frozen]==versions
    hook.remove();assert calls[0]==17
    peak=torch.cuda.max_memory_allocated();assert peak<20<<30
    assert time.monotonic()-start<1100 and sum(p.stat().st_size for p in out.iterdir() if p.is_file())<32<<20
    r=dict(status='PASS_REAL_NATIVE_GENERATOR_ZERO_HEAD_PIXELS_AND_INPUT_GRADIENT',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),GPU=torch.cuda.get_device_name(0),GPU_count=1,
        seconds=time.monotonic()-start,peak_GPU_allocated_bytes=peak,host_contract=host,T_query_positions=positions,pixel_checks=pixels,UNet_calls=calls[0],direction_gradient_norm=grad,
        gradient_state='Synthetic latent/noise only for graph engineering; not image training input, fitted objective or efficacy',new_optimizer_updates=0,engineering_images=4,efficacy_images=0,
        old_ordinary_changed=False,new_ordinary_fit=False,new_method_or_SOTA=False)
    write('completion.json',r);print(json.dumps(r),flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
