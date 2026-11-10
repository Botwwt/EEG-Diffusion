"""Real Torch load-state counter regression; numerical train steps unchanged."""
import json,os,socket,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
import torch
R=Path('/dfs/share/maleiLab/wwt/eeg');torch.set_num_threads(4);start=time.monotonic()
old=(R/'code/PKU_train_preservation_GPU_20261110_v317.py').read_text();new=(R/'code/PKU_train_preservation_GPU_20261110_v318.py').read_text()
insert='    # load_state_dict increments version counters even for unchanged frozen values.\n    # CPU checkpoint audit verifies frozen values; track new mutations after restore.\n    versions=[p._version for p in frozen]\n'
assert new.count(insert)==1 and new.replace(insert,'')==old
torch.manual_seed(1729)
m=torch.nn.Linear(1024,8192);m.bias.requires_grad_(False)
saved={k:v.detach().clone() for k,v in m.state_dict().items()}
before=m.bias._version;m.load_state_dict(saved,strict=True);after=m.bias._version
assert after>before and torch.equal(m.bias,saved['bias']) and torch.equal(m.weight,saved['weight'])
versions=[m.bias._version];x=torch.randn(2,1024);loss=m(x).square().mean();loss.backward()
assert m.bias.grad is None and m.weight.grad.norm()>0
opt=torch.optim.AdamW([m.weight],lr=1e-5,weight_decay=.01);opt.step()
assert [m.bias._version]==versions and torch.equal(m.bias,saved['bias'])
with torch.no_grad():m.bias.add_(0.)
assert [m.bias._version]!=versions
v=dict(status='PASS_REAL_NATIVE_SIZE_LINEAR_FROZEN_VALUE_RESTORE_COUNTER_AND_MUTATION_DETECTION',CPU_job=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,bias_counter_before_restore=before,bias_counter_after_restore=after,restored_values_bitexact=True,training_step_freeze_guard_pass=True,postbaseline_mutation_detected=True,only_counter_baseline_line_added=True,new_GPU_jobs=0,new_scientific_updates=0)
p=R/'runs/EEG_preservation_v317/engineering/restore_counter_regression_v318.json';assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
