"""Necessary causal-prefix/gradient and row-optimizer mathematical checks."""
import datetime,hashlib,json,sys,time,types
from pathlib import Path
import torch
L=Path(__file__).resolve().parent;S=L/'reproduction_sources/direction_switch_20261028_v1/avde_3be06a90ae69'
sys.path.insert(0,str(S));package=types.ModuleType('models');package.__path__=[str(S/'models')];sys.modules['models']=package
import dist
dist.__dict__['__device']='cpu'
from models.vqvae import VQVAE
from models.var_eeg import VAREEG
from EEG_native_marginal_core_20261029_v2 import NativeMarginalDecoder,RowGaussianAdam
torch.set_num_threads(2);torch.manual_seed(1729);started=time.monotonic();checks={}
v=VQVAE(vocab_size=32,z_channels=8,ch=32,test_mode=True,v_patch_nums=(1,2,3))
d=VAREEG(v,depth=2,embed_dim=64,num_heads=4,cond_drop_rate=0.,patch_nums=(1,2,3),flash_if_available=False,fused_if_available=False)
d.init_weights(init_adaln_gamma=.1);n=NativeMarginalDecoder(d,torch.randn(6,64)*.1)
x=torch.randn(4,13,8);z=torch.randn(4,64,requires_grad=True)
full=n(x,z);short=n.marginal_logits(x[:,:4],z,5)
checks['coarse_logits_equal_complete_native_forward']=torch.allclose(short,full[:,:5],atol=2e-6,rtol=2e-5)
g1=torch.autograd.grad(full[:,:5].square().sum(),z)[0];g2=torch.autograd.grad(short.square().sum(),z)[0]
checks['coarse_latent_gradient_equals_full_forward']=torch.allclose(g1,g2,atol=2e-6,rtol=2e-5)
x2=x.clone();x2[:,4:]=torch.randn_like(x2[:,4:])*100
checks['later_true_tokens_cannot_affect_coarse_target']=torch.allclose(n(x2,z)[:,:5],short,atol=2e-6,rtol=2e-5)
try:n.marginal_logits(x[:,:2],z,3);checks['incomplete_scale_rejected']=False
except ValueError:checks['incomplete_scale_rejected']=True
bank=RowGaussianAdam(3,7,'cpu');ref=torch.nn.Parameter(torch.zeros(2,7));opt=torch.optim.Adam([ref],lr=.03)
for _ in range(3):
    leaf=bank.leaf(torch.tensor([1]));loss=(leaf-.2).square().sum();loss.backward();bank.step(torch.tensor([1]),leaf)
    opt.zero_grad();(ref-.2).square().sum().backward();opt.step()
checks['row_Adam_matches_independent_Torch_Adam']=torch.allclose(bank.value[1],ref,atol=1e-7,rtol=1e-6)
checks['unvisited_image_values_and_moments_unchanged']=all(bool(torch.equal(t[[0,2]],torch.zeros_like(t[[0,2]]))) for t in bank.state_dict().values())
state={k:v.clone() for k,v in bank.state_dict().items()};copy=RowGaussianAdam(3,7,'cpu');copy.load_state_dict(state)
checks['complete_row_Adam_resume_exact']=all(torch.equal(copy.state_dict()[k],v) for k,v in state.items())
checks['frozen_generator_has_no_parameter_gradient']=all(p.grad is None and not p.requires_grad for p in n.parameters())
assert all(checks.values()),checks
result=dict(status='PASS_EXACT_COARSE_MARGINAL_AND_ROW_OPTIMIZER_CPU',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,seconds=time.monotonic()-started,source_sha256=hashlib.sha256((L/'EEG_native_marginal_core_20261029_v2.py').read_bytes()).hexdigest(),real_EEG_or_public_weights=False)
out=L/'audit/EEG_native_conditional_variational_20261029_v1/marginal_CPU_actual_v2.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
