"""New mathematical diagnostic only; never changes a running model or recipe.

For zero-mean local perturbations and identical global-anchor tokens, the
directional derivative of anchored attention at zero is zero. This says
nothing by itself about finite real EEG deviations or heldout image utility.
"""
import datetime,hashlib,json,sys,types
from pathlib import Path
import torch
import torch.nn.functional as F
L=Path(__file__).resolve().parent;S=L/'reproduction_sources/direction_switch_20261028_v1/avde_3be06a90ae69'
inventory=json.loads((S/'source_inventory.json').read_bytes())
for item in inventory['files']:assert hashlib.sha256((S/item['path']).read_bytes()).hexdigest()==item['sha256']
package=types.ModuleType('models');package.__path__=[str(S/'models')];assert 'models' not in sys.modules;sys.modules['models']=package
sys.path.insert(0,str(S));import dist;dist.__dict__['__device']='cpu'
from models.var_eeg import VAREEG
from models.vqvae import VQVAE
from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder
torch.set_num_threads(2);torch.manual_seed(1733);checks={};details={}
for l2 in (False,True):
    vae=VQVAE(vocab_size=32,z_channels=8,ch=32,test_mode=True,v_patch_nums=(1,2,3))
    decoder=VAREEG(vae,depth=2,embed_dim=64,num_heads=4,cond_drop_rate=0.,patch_nums=(1,2,3),attn_l2_norm=l2,flash_if_available=False,fused_if_available=False)
    decoder.init_weights(init_adaln_gamma=.1);model=EEGAnchoredAttentionDecoder(decoder).double().eval()
    c=torch.randn(1,1024,dtype=torch.float64);delta=torch.randn(1,7,1024,dtype=torch.float64);delta-=delta.mean(1,keepdim=True)
    condition=decoder.embed_proj(c);conditioning=decoder.shared_ada_lin(condition)
    block=decoder.blocks[0];attention=block.attn;image_input=torch.randn(1,3,64,dtype=torch.float64)
    qkv=F.linear(image_input,attention.mat_qkv.weight,torch.cat((attention.q_bias,attention.zero_k_bias,attention.v_bias)))
    query=qkv.view(1,3,3,attention.num_heads,attention.head_dim)[:,:,0].transpose(1,2)
    if l2:query=F.normalize(query,dim=-1)*attention.scale_mul_1H11.clamp_max(attention.max_scale_mul).exp()
    def read(lam,perturbation=delta):
        embeddings=model._memory_embeddings(c,c[:,None]+lam*perturbation,condition)
        k,v=model._prepare_layer(block,conditioning,embeddings)
        r=((query@k.transpose(-1,-2))*attention.scale).softmax(-1)@v
        return F.linear(r.transpose(1,2).reshape(1,3,64),attention.proj.weight,None)
    zero=torch.tensor(0.,dtype=torch.float64)
    derivative=torch.autograd.functional.jacobian(read,zero,vectorize=True)
    checks[f'centered_directional_derivative_zero_l2_{l2}']=derivative.abs().max().item()<1e-11
    amplitude=[.1,.05,.025,.0125]
    values=[float(read(torch.tensor(x,dtype=torch.float64)).detach().norm()) for x in amplitude]
    ratios=[values[i]/values[i+1] for i in range(3)]
    checks[f'nonzero_quadratic_response_l2_{l2}']=values[-1]>0 and abs(ratios[-1]-4)<.08
    noncentered=delta+torch.randn(1,1,1024,dtype=torch.float64)
    first=torch.autograd.functional.jacobian(lambda lam:read(lam,noncentered),zero,vectorize=True)
    checks[f'uncentered_control_has_first_order_signal_l2_{l2}']=first.norm().item()>1e-5
    checks[f'memory_axis_permutation_invariant_l2_{l2}']=torch.allclose(read(torch.tensor(1.)),read(torch.tensor(1.),delta[:,[3,2,0,6,4,1,5]]),atol=1e-12,rtol=1e-10)
    # Addressed keys are a synthetic counterexample, not a production method.
    # Values still vanish at lambda=0; fixed heterogeneous keys break the
    # uniform-weight cancellation. No EEG anatomy or benefit is claimed.
    address=torch.randn(1,7,64,dtype=torch.float64)*.2
    _,_,scale,_,shift,_=model._native_condition_parts(block,conditioning)
    anchor=condition[:,None].expand(-1,7,-1)
    base=block.ln_wo_grad(anchor).mul(scale.add(1)).add(shift)
    def addressed(lam):
        de=F.linear(lam*delta,decoder.embed_proj.weight,None)
        current=block.ln_wo_grad(anchor+de).mul(scale.add(1)).add(shift)
        key_input=block.ln_wo_grad(anchor+address+de).mul(scale.add(1)).add(shift)
        k=F.linear(key_input,attention.mat_qkv.weight[64:128],attention.zero_k_bias).view(1,7,4,16).transpose(1,2)
        if l2:k=F.normalize(k,dim=-1)
        v=F.linear(current-base,attention.mat_qkv.weight[128:],None).view(1,7,4,16).transpose(1,2)
        r=((query@k.transpose(-1,-2))*attention.scale).softmax(-1)@v
        return F.linear(r.transpose(1,2).reshape(1,3,64),attention.proj.weight,None)
    addressed_derivative=torch.autograd.functional.jacobian(addressed,zero,vectorize=True)
    checks[f'synthetic_heterogeneous_keys_preserve_zero_anchor_l2_{l2}']=bool((addressed(zero)==0).all())
    checks[f'synthetic_heterogeneous_keys_restore_first_order_l2_{l2}']=addressed_derivative.norm().item()>1e-6
    details[str(l2)]=dict(centered_first_derivative_max_abs=derivative.abs().max().item(),amplitudes=amplitude,response_norms=values,halving_ratios=ratios,
        uncentered_derivative_norm=first.norm().item(),synthetic_addressed_derivative_norm=addressed_derivative.norm().item())
assert all(checks.values()),checks
value=dict(status='PASS_MATHEMATICAL_CENTERED_EEG_ATTENTION_FIRST_ORDER_CANCELLATION_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,details=details,
    author_reduced_modules=True,random_weights=True,float64=True,real_EEG_used=False,pretrained_VAR_used=False,GPU_jobs=0,trained_models=0,research_images=0,running_scientific_recipe_changed=False,
    scope='Exact local derivative at all-memory-tokens-equal-global anchor; does not prove finite EEG is weak or predict image failure. Synthetic key addresses demonstrate one way to break symmetry, not a new admitted architecture.',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),production_anchor_sha256=hashlib.sha256((L/'EEG_anchored_attention_20261029_v1.py').read_bytes()).hexdigest())
out=L/'audit/AR_EEG_centering_order_20261029_v1';out.mkdir(exist_ok=True);p=out/'CPU_actual_v1.json';assert not p.exists();p.write_bytes((json.dumps(value,indent=2)+'\n').encode());print(json.dumps(value))
