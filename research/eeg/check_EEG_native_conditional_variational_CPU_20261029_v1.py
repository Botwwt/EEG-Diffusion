"""Actual author tiny-prior tests: native coordinates, ELBO and both gradients."""
import datetime,hashlib,json,sys,time,types
from pathlib import Path
import torch
from torch import nn
L=Path(__file__).resolve().parent;S=L/'reproduction_sources/direction_switch_20261028_v1/avde_3be06a90ae69';sys.path.insert(0,str(S))
inventory=json.loads((S/'source_inventory.json').read_bytes())
for x in inventory['files']:assert hashlib.sha256((S/x['path']).read_bytes()).hexdigest()==x['sha256']
package=types.ModuleType('models');package.__path__=[str(S/'models')];assert 'models' not in sys.modules;sys.modules['models']=package
import dist
dist.__dict__['__device']='cpu'
from models.vqvae import VQVAE
from models.var_eeg import VAREEG
from EEG_native_conditional_variational_core_20261029_v1 import NativeConditionDecoder,EEGGaussianPrior,ImageVariationalBank,native_moments,diagonal_KL,token_ELBO,reparameterize
torch.set_num_threads(2);torch.manual_seed(1729);started=time.monotonic();checks={}
vae=VQVAE(vocab_size=32,z_channels=8,ch=32,test_mode=True,v_patch_nums=(1,2,3));d=VAREEG(vae,depth=2,embed_dim=64,num_heads=4,cond_drop_rate=0.,patch_nums=(1,2,3),flash_if_available=False,fused_if_available=False);d.init_weights(init_adaln_gamma=.1)
prior=d.state_dict();frozen={k:v.clone() for k,v in prior.items() if not k.startswith('embed_proj.')};classes=torch.randn(6,64)*.1;native=NativeConditionDecoder(d,classes)
checks['all_existing_native_tensors_exactly_preserved']=all(torch.equal(d.state_dict()[k],v) for k,v in frozen.items())
checks['random_EEG_adapter_removed_no_new_generator_parameters']=not any(k.startswith('embed_proj.') for k in d.state_dict()) and not any(p.requires_grad for p in native.parameters())
prefix=torch.randn(4,13,8);z=torch.randn(4,64,requires_grad=True)
with torch.no_grad():expected=d(prefix,z.detach())
actual=native(prefix,z);checks['native_condition_forward_exact_original_identity_interface']=torch.equal(actual.detach(),expected)
actual.square().mean().backward();checks['latent_gradient_through_frozen_native_prior']=z.grad is not None and bool(z.grad.norm()>0) and all(p.grad is None for p in native.parameters())
center,scale=native_moments(classes)
class Encoder(nn.Module):
 def __init__(self):super().__init__();self.body=nn.Linear(5,7);self.head=nn.Linear(7,64)
 def forward(self,x):return self.head(self.body(x).tanh()),None,None
p=EEGGaussianPrior(Encoder(),center,scale).eval();q=ImageVariationalBank(4,center,scale);x=torch.randn(4,5);ids=torch.arange(4);pm,ps=p(x);qm,qs=q(ids)
checks['initial_EEG_and_image_priors_same_native_moments']=torch.equal(pm,qm) and torch.equal(ps,qs)
checks['same_Gaussian_KL_zero']=torch.equal(diagonal_KL(qm,qs,pm,ps),torch.zeros(4))
qa=torch.randn(4,64);qs0=torch.randn(4,64)*.2;pa=torch.randn(4,64);ps0=torch.randn(4,64)*.2
reference=torch.distributions.kl_divergence(torch.distributions.Normal(qa,qs0.exp()),torch.distributions.Normal(pa,ps0.exp())).sum(-1)
checks['KL_matches_independent_Torch_distribution_formula']=torch.allclose(diagonal_KL(qa,qs0,pa,ps0),reference,atol=1e-5,rtol=1e-5)
rng=torch.Generator().manual_seed(1729);targets=torch.randint(0,32,(4,14));optq=torch.optim.Adam(q.parameters(),lr=.03);optp=torch.optim.Adam(p.parameters(),lr=.01)
for step in range(3):
 optq.zero_grad(set_to_none=True);p.zero_grad(set_to_none=True);qm,qs=q(ids);pm,ps=p(x);z=reparameterize(qm,qs,generator=rng);logits=native(prefix,z);loss,info=token_ELBO(logits,targets,qm,qs,pm.detach(),ps.detach());loss.backward()
 checks['E_step_teacher_grad_no_EEG_grad']=bool(q.mean_delta.grad.norm()>0) and all(v.grad is None for v in p.parameters());optq.step()
 optq.zero_grad(set_to_none=True);optp.zero_grad(set_to_none=True);qm,qs=q(ids);pm,ps=p(x);kl=diagonal_KL(qm.detach(),qs.detach(),pm,ps).mean()/14;kl.backward();checks['M_step_EEG_grad_no_teacher_grad']=p.encoder.head.weight.grad is not None and bool(p.encoder.head.weight.grad.norm()>0) and all(v.grad is None for v in q.parameters());optp.step()
checks['EEG_backbone_receives_gradient_after_head_learning']=p.encoder.body.weight.grad is not None and bool(p.encoder.body.weight.grad.norm()>0)
checks['single_ELBO_consistent_nats_normalization']=torch.allclose(loss.detach(),(info['token_NLL_nats_per_query']+info['KL_nats_per_query']).mean()/14)
checks['all_generator_values_unchanged_after_E_and_M_updates']=all(torch.equal(d.state_dict()[k],v) for k,v in frozen.items()) and all(v.grad is None for v in native.parameters())
with torch.no_grad():im1=native.generate(seed=1729,top_k=5);im2=native.generate(classes[-1:],seed=1729,top_k=5)
checks['disabled_condition_exact_native_null_class_pixels']=torch.equal(im1,im2)
assert all(checks.values()),checks
out=L/'audit/EEG_native_conditional_variational_20261029_v1';out.mkdir(exist_ok=True);f=out/'CPU_contract_actual_v1.json';assert not f.exists()
result=dict(status='PASS_REAL_AUTHOR_REDUCED_RANDOM_PRIOR_NATIVE_COORDINATES_AND_CONDITIONAL_ELBO_CPU',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,seconds=time.monotonic()-started,source_sha256=hashlib.sha256((L/'EEG_native_conditional_variational_core_20261029_v1.py').read_bytes()).hexdigest(),pretrained_weights_or_real_EEG_or_research_images_used=False,new_GPU_jobs=0)
f.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
