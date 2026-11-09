"""Actual-author tiny-model tests of observation hooks and probability math."""
import datetime,hashlib,json,sys,time,types
from pathlib import Path
import torch
import torch.nn.functional as F
L=Path(__file__).resolve().parent;S=L/'reproduction_sources/direction_switch_20261028_v1/avde_3be06a90ae69';start=time.monotonic()
inventory=json.loads((S/'source_inventory.json').read_bytes())
for pin in inventory['files']:assert hashlib.sha256((S/pin['path']).read_bytes()).hexdigest()==pin['sha256']
package=types.ModuleType('models');package.__path__=[str(S/'models')];assert 'models' not in sys.modules;sys.modules['models']=package;sys.path.insert(0,str(S))
import dist
dist.__dict__['__device']='cpu'
from models.var_eeg import VAREEG
from models.vqvae import VQVAE
from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder
torch.set_num_threads(2);torch.manual_seed(1729)
vae=VQVAE(vocab_size=32,z_channels=8,ch=32,test_mode=True,v_patch_nums=(1,2,3));d=VAREEG(vae,depth=2,embed_dim=64,num_heads=4,cond_drop_rate=0.,patch_nums=(1,2,3),flash_if_available=False,fused_if_available=False)
d.init_weights(init_adaln_gamma=.1);m=EEGAnchoredAttentionDecoder(d).eval();c=torch.randn(1,1024);codes=[];logits=[]
def observe_codes(module,args):codes.append(args[0].detach().clone())
def observe_logits(module,args,value):logits.append(value.detach().clone())
h=vae.quantize.embedding.register_forward_pre_hook(observe_codes);lh=d.head.register_forward_hook(observe_logits)
with torch.inference_mode():hooked=m.generate(c,seed=1729,guidance=1.5,top_k=5)
h.remove();lh.remove()
with torch.inference_mode():plain=m.generate(c,seed=1729,guidance=1.5,top_k=5)
checks={'hooked_and_unhooked_pixels_exact':torch.equal(hooked,plain),'one_code_and_raw2branch_logit_per_scale':len(codes)==len(logits)==3,
 'all_code_shapes_exact':[tuple(x.shape) for x in codes]==[(1,1),(1,4),(1,9)],'all2branch_logit_shapes_exact':[tuple(x.shape) for x in logits]==[(2,1,32),(2,4,32),(2,9,32)]}
with torch.inference_mode():
 prefix=vae.quantize.idxBl_to_var_input(codes);full=torch.cat([m(prefix,c),m(prefix,torch.zeros_like(c))],0);cached=torch.cat(logits,1)
error=float((full-cached).abs().max());checks['raw_sampler_logits_match_teacher_forward_on_same_sampled_prefix']=error<2e-6
checks['all_KV_caches_released']=all(not b.attn.caching and b.attn.cached_k is None for b in d.blocks)
a=torch.tensor([[2.,-1.,.3]]);b=torch.tensor([[.2,.4,-.5]]);target=torch.tensor([2]);guided=2.5*a-1.5*b
true_loss=F.cross_entropy(guided,target);wrong_loss=2.5*F.cross_entropy(a,target)-1.5*F.cross_entropy(b,target)
checks['NLL_not_linear_interpolation_of_NLLs']=abs(float(true_loss-wrong_loss))>.01
cutoff=guided.topk(1,-1).values[...,-1:];supported=bool(guided.gather(-1,target[:,None])>=cutoff);checks['target_outside_topk_explicitly_reported']=supported is False
assert all(checks.values()),checks
out=L/'audit/AR_Dfit200_actual_prefix_20261029_v1';out.mkdir(exist_ok=True);p=out/'CPU_observation_actual_v1.json';assert not p.exists()
result=dict(status='PASS_ACTUAL_AUTHOR_REDUCED_RANDOM_MODEL_HOOKS_CACHE_PREFIX_AND_GUIDANCE_MATH_CPU',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,cached_full_logits_max_error=error,seconds=time.monotonic()-start,
 real_pretrained_weights_or_EEG_or_research_images_used=False,new_GPU_jobs=0,producer_sha256=hashlib.sha256((L/'diagnose_AR_native_Dfit_actual_prefix_GPU_20261029_v1.py').read_bytes()).hexdigest())
p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
