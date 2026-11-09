"""SlurmCPU source contracts, real P, explicit objective gradients and actual RNG schema."""
import ast,json,math,os,shutil,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
import torch
from safetensors.torch import load_file
from PKU_downstream_core_20261109_v303 import R,ROOT,CASES,flags,sha,write
start=time.monotonic();torch.set_num_threads(4);out=R/'runs/EEG_downstream_engineering_v303';out.mkdir(exist_ok=True)
pins=json.loads((R/'code/EEG_downstream_pins_v303.json').read_text())
for name,h in pins.items():
    p=R/'code'/name;assert sha(p)==h
    if name.endswith('.py'):ast.parse(p.read_text())
assert shutil.disk_usage(R).free>100<<30
s=load_file(str(R/'assets/native_generator_v243/prior_H14/proj/diffusion_pytorch_model.safetensors'));assert set(s)=={'proj.weight','proj.bias','norm.weight','norm.bias'} and s['proj.weight'].shape==(8192,1024);assert sum(v.numel() for v in s.values())==8400896
x=torch.linspace(-1,1,1024).reshape(1,-1);h=torch.nn.functional.linear(x,s['proj.weight'].float(),s['proj.bias'].float()).reshape(1,4,2048);p=torch.nn.functional.layer_norm(h,(2048,),s['norm.weight'].float(),s['norm.bias'].float(),1e-5);assert p.shape==(1,4,2048) and torch.isfinite(p).all()
objective_checks=[]
for case in CASES:
    f=flags(case);ep=torch.tensor([.2,.3],dtype=torch.float64,requires_grad=True);en=torch.tensor([.4,.1],dtype=torch.float64,requires_grad=True);rv=torch.tensor(.005,dtype=torch.float64,requires_grad=True);tau=.01
    loss=ep.mean()+f['pair']*tau*torch.nn.functional.softplus(f['sign']*(ep-en)/tau).mean()+f['replay']*rv;g=torch.autograd.grad(loss,(ep,en,rv));sig=torch.sigmoid(f['sign']*(ep-en)/tau)
    assert torch.allclose(g[0],.5+.5*f['pair']*f['sign']*sig,atol=1e-14,rtol=0);assert torch.allclose(g[1],-.5*f['pair']*f['sign']*sig,atol=1e-14,rtol=0);assert float(g[2])==f['replay'];objective_checks.append(case)
gen=torch.Generator().manual_seed(1729);s0=gen.get_state();assert torch.equal(torch.Generator().set_state(s0).get_state(),s0)
policy=json.loads((R/'code/EEG_downstream_policy_v303.json').read_text());assert policy['cases']==CASES and policy['updates_per_case']==827 and policy['microbatch']==4 and policy['batch']==16
r=dict(status='PASS_SOURCE_CONTRACT_REAL_NATIVE_P_OBJECTIVE_GRADIENTS_RNG_AND_STORAGE',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,cases=objective_checks,real_P_parameters=8400896,CPU_RNG_bytes=s0.numel(),real_GPU_prefix_still_required=True,new_training_updates=0,new_images=0,source_sha256=sha(__file__));write(out/'completion.json',r);print(json.dumps(r),flush=True)
