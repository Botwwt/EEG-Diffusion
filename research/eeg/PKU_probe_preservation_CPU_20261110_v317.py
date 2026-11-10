"""Finite real native-P CPU contract, projection algebra, differentiability checks."""
import json,os,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
import torch
from PKU_conditional_preservation_core_20261110_v317 import R,ROOT,CASES,check_fixture,write,sha
from PKU_HIVE_native_classes_20261109_v239 import classes
ROOT.mkdir(exist_ok=False);out=ROOT/'engineering';out.mkdir();start=time.monotonic();torch.set_num_threads(4)
assert json.loads((R/'runs/EEG_downstream_scores_v303/analysis_completion.json').read_text())['queries']==830
spec=json.loads((R/'code/EEG_preservation_policy_v317.json').read_text());assert spec['cases']==CASES and spec['total_updates']==4135
fixture=check_fixture()
# Real native P shape and parameterization; no UNet or efficacy claim.
from safetensors.torch import load_file
weights=load_file(str(R/'assets/native_generator_v243/prior_H14/proj/diffusion_pytorch_model.safetensors'))
assert set(weights)=={'proj.weight','proj.bias','norm.weight','norm.bias'}
assert weights['proj.weight'].shape==(8192,1024) and sum(v.numel() for v in weights.values())==8400896
g=torch.Generator().manual_seed(1729);x=torch.randn(2,1024,generator=g,dtype=torch.float32)
w=weights['proj.weight'].float().clone().requires_grad_(True)
y=torch.nn.functional.layer_norm(torch.nn.functional.linear(x,w,weights['proj.bias'].float()).reshape(2,4,2048),(2048,),weights['norm.weight'].float(),weights['norm.bias'].float(),1e-5)
loss=y.square().mean();loss.backward();assert w.grad is not None and torch.isfinite(w.grad).all() and w.grad.norm()>0
checks=dict(status='PASS_NATIVE_P_WEIGHT_SHAPE_GRADIENT_AND_PROJECTION_ALGEBRA',CPU_job=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,fixture=fixture,cases=CASES,trainable_parameters=8388608,heldout_inputs_read=False,source_sha256=sha(__file__),real_GPU_prefix_pending=True)
write(out/'completion.json',checks);print(json.dumps(checks),flush=True)
