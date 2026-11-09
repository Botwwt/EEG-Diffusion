import json,tempfile
from pathlib import Path
import numpy as np
import torch
from eeg_alignment_training_core_20261028_v1 import *
from profile_EEG_labram_real_T_GPU_20261028_v1 import objective as profiled_objective
torch.set_num_threads(2)
class SmallEEG(torch.nn.Module):
    def __init__(self):
        super().__init__();self.head=torch.nn.Linear(6,4);self.dropout=torch.nn.Dropout(.2);self.logit_scale=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,x,input_chans):return self.head(self.dropout(x)),None,None
torch.manual_seed(1729);x=np.random.default_rng(1729).normal(size=(11,6)).astype('float32');y=np.random.default_rng(2).normal(size=(11,4)).astype('float32');selected=np.arange(11)
m=SmallEEG();opt=optimizer_for(m);rng=torch.Generator().manual_seed(1729)
c=torch.from_numpy(y).requires_grad_();target=torch.from_numpy(y[::-1].copy());a=objective(c,target,m.logit_scale)[0];b=profiled_objective(c,target,m.logit_scale)[0];assert torch.equal(a,b)
row=run_epoch(m,opt,rng,x,y,selected,[],epoch=1,updates=0,total_updates=12,warmup_updates=3,batch=4,device='cpu');assert row['updates']==3 and row['queries']==11
saved=snapshot(m,opt,rng,epoch=1,updates=3,spec_sha256='fixture',role='full',device='cpu')
continuous=run_epoch(m,opt,rng,x,y,selected,[],epoch=2,updates=3,total_updates=12,warmup_updates=3,batch=4,device='cpu');expected={k:v.clone() for k,v in m.state_dict().items()};expected_optimizer=cpu_tree(opt.state_dict())
torch.manual_seed(999);restored=SmallEEG();other=optimizer_for(restored);other_rng=torch.Generator().manual_seed(5)
restore(saved,restored,other,other_rng,spec_sha256='fixture',role='full',updates_per_epoch=3,total_epochs=4,device='cpu')
resumed=run_epoch(restored,other,other_rng,x,y,selected,[],epoch=2,updates=3,total_updates=12,warmup_updates=3,batch=4,device='cpu')
assert continuous['order_sha256']==resumed['order_sha256'] and continuous['loss']==resumed['loss']
assert all(torch.equal(expected[k],v) for k,v in restored.state_dict().items())
for key,state in other.state_dict()['state'].items():
    assert all(torch.equal(v,expected_optimizer['state'][key][name]) for name,v in state.items())
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    def fail(point):raise RuntimeError('injected prior to commit')
    try:publish_epoch(root,saved,row,fault=fail)
    except RuntimeError:pass
    assert not (root/'epoch_001').exists() and len(list(root.glob('*.pending.*')))==1
    commit=publish_epoch(root,saved,row);assert sha(root/'epoch_001/resume.pt')==commit['resume_sha256']
assert learning_rate(0,12,3)==1e-6 and learning_rate(2,12,3)==5e-4 and abs(learning_rate(11,12,3)-1e-6)<1e-15
print(json.dumps(dict(status='PASS_COMPLETE_ROWS_STOCHASTIC_EXACT_RESUME_ADAM_RNG_AND_ATOMIC_COMMIT',formal_models=0,GPU_jobs=0,profile_objective_identical=True)))
