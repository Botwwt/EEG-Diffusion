"""Verify the actual D37 operator cannot distinguish joint time-list order."""
import datetime
import json
from pathlib import Path
import torch
from brainhive_temporal_conditional_attention_20261028_v1 import attention_delta
torch.set_num_threads(2);torch.manual_seed(1729)
q=torch.randn(1,2,7,8,dtype=torch.float64);k=torch.randn(1,2,4,8,dtype=torch.float64)
dk=torch.randn(1,2,4,5,8,dtype=torch.float64);dv=torch.randn_like(dk);dk-=dk.mean(3,keepdim=True);dv-=dv.mean(3,keepdim=True)
permutation=torch.tensor([4,2,0,3,1]);errors={}
for mode in ['conditional','flat_residual']:
    result=attention_delta(q,k,dk,dv,mode=mode)
    changed=attention_delta(q,k,dk[:,:,:,permutation],dv[:,:,:,permutation],mode=mode)
    errors[mode]=float((result-changed).abs().max());assert errors[mode]<1e-12
root=Path(__file__).resolve().parent;out=root/'audit/crossdisciplinary_20261028_v1/D37_time_permutation_actual_CPU_v1.json';assert not out.exists()
value=dict(status='D37_PRIMARY_AND_FLAT_OPERATOR_JOINT_TIME_LIST_PERMUTATION_INVARIANCE_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    maximum_absolute_errors=errors,interpretation='The added operator treats the five already-derived EEG time contributions as a set. The native W_t projections still encode original time coordinates; this is not invariance to reversing the raw EEG itself.',
    actual_EEG_or_image_values_read=False,new_GPU_jobs=0,new_image_candidates=0,method_or_gate_changed=False)
out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');print(json.dumps(value))
