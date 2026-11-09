import io,json,pickle
import numpy as np
import torch
from torch_numpy_scalar_checkpoint_20261028_v2 import load_tensor_checkpoint
proofs=[]
for protocol in (2,4):
    stream=io.BytesIO();torch.save(dict(model={'a':torch.arange(9).reshape(3,3)},metric=np.float64(.125)),stream,pickle_protocol=protocol);stream.seek(0)
    value,proof=load_tensor_checkpoint(stream)
    assert torch.equal(value['model']['a'],torch.arange(9).reshape(3,3)) and value['metric']==.125
    proofs.append(dict(protocol=protocol,**proof))
class UnknownGlobalFixture:
    def __reduce__(self):return (len,('abc',))
bad=io.BytesIO();torch.save(dict(metric=np.float64(.5),value=UnknownGlobalFixture()),bad);bad.seek(0)
try:load_tensor_checkpoint(bad)
except pickle.UnpicklingError:rejected=True
else:rejected=False
assert rejected
print(json.dumps(dict(status='PASS_PROTOCOL2_4_TENSOR_SCALAR_BYTES_AND_UNKNOWN_GLOBAL_REJECTION',proofs=proofs,GPU_jobs=0)))
