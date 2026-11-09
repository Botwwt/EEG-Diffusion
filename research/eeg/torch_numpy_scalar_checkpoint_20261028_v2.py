"""Read author tensor checkpoints containing legacy NumPy scalar metadata.

Uses Torch's existing tensor-global allowlist, adding only NumPy dtype/scalar
constructors and the legacy bytes encoding function. Unknown globals are rejected.
"""
import codecs,io,pickle,types
import numpy as np
import torch


def load_tensor_checkpoint(path):
    original_position=path.tell() if hasattr(path,'tell') else None
    try:
        return torch.load(path,map_location='cpu',weights_only=True),dict(mode='torch_weights_only',extra_globals=[])
    except pickle.UnpicklingError:
        # The fallback still resolves every global through the explicit table;
        # it also handles newer protocol framing absent in Torch 2.3's reader.
        pass
    allowed=dict(torch._weights_only_unpickler._get_allowed_globals())
    allowed.update({'numpy.core.multiarray.scalar':np.core.multiarray.scalar,
                    'numpy._core.multiarray.scalar':np.core.multiarray.scalar,
                    'numpy.dtype':np.dtype,'_codecs.encode':codecs.encode})
    used=[]
    class TensorAndScalarUnpickler(pickle.Unpickler):
        def find_class(self,module,name):
            key=module+'.'+name
            if key not in allowed:
                raise pickle.UnpicklingError('Unapproved checkpoint global: '+key)
            used.append(key)
            return allowed[key]
    if original_position is not None:path.seek(original_position)
    def explicit_load(stream,**kwargs):return TensorAndScalarUnpickler(stream,**kwargs).load()
    def explicit_loads(data,**kwargs):return explicit_load(io.BytesIO(data),**kwargs)
    reader=types.SimpleNamespace(__name__='explicit_tensor_numpy_scalar_reader',Unpickler=TensorAndScalarUnpickler,load=explicit_load,loads=explicit_loads)
    value=torch.load(path,map_location='cpu',pickle_module=reader,weights_only=False)
    return value,dict(mode='explicit_tensor_numpy_scalar_and_legacy_bytes_globals',extra_globals=sorted(set(used)&{'numpy.core.multiarray.scalar','numpy._core.multiarray.scalar','numpy.dtype','_codecs.encode'}),all_resolved_globals=sorted(set(used)))
