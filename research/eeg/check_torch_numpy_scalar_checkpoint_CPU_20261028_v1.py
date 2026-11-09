import io,json,pickle
import numpy as np
import torch
from torch_numpy_scalar_checkpoint_20261028_v1 import load_tensor_checkpoint
buffer=io.BytesIO();torch.save(dict(model={'a':torch.arange(9).reshape(3,3)},metric=np.float64(.125)),buffer);buffer.seek(0)
value,proof=load_tensor_checkpoint(buffer)
assert torch.equal(value['model']['a'],torch.arange(9).reshape(3,3)) and value['metric']==.125
class UnknownGlobalFixture:
    def __reduce__(self):return (len,('abc',))
bad=io.BytesIO();torch.save(dict(metric=np.float64(.5),value=UnknownGlobalFixture()),bad);bad.seek(0)
try:load_tensor_checkpoint(bad)
except pickle.UnpicklingError:rejected=True
else:rejected=False
assert rejected
print(json.dumps(dict(status='PASS_TENSOR_NUMPY_SCALAR_COMPATIBILITY_AND_UNKNOWN_GLOBAL_REJECTION',proof=proof,new_GPU_jobs=0,new_models=0)))
