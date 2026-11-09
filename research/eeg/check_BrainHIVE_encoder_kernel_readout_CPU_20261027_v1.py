"""Check fixed positive, null, and negative interpretations before data readout."""
import argparse,json
from pathlib import Path
import numpy as np
import brainhive_encoder_kernel_innovation_20261027_v1 as K
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
concepts=[str(i//10) for i in range(820)];subjects=[f'sub-{j:02}' for j in range(1,11)];tests=[]
for delta,expect in [(-.02,True),(0.,False),(.02,False)]:
    s=np.ones((10,820,4,5,3));s[:,:,:,1,:2]+=delta
    r=K.readout(s,concepts,subjects);assert r['kernel_specific_conditional_increment_supported']==expect;tests.append(dict(kernel_minus_control=delta,expected=expect,action=r['action']))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='FIXED_KERNEL_READOUT_POSITIVE_NULL_NEGATIVE_CHECKED',tests=tests,new_GPU_jobs=0),indent=2)+'\n')
print('3 FIXED READOUT CHECKS PASSED')
