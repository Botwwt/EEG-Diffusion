"""Audited AVDE encoder constructor and explicit clinical checkpoint transfer.

Does not call the author's package factory that globally changes initialization.
The pretraining token decoder and final norm are explicitly excluded, following
the author's mean-pooling downstream architecture. No arbitrary missing keys.
"""
import hashlib,sys,types
from functools import partial
from pathlib import Path
import torch

CLINICAL_SHA256='7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c'
LABRAM_SOURCE_SHA256='8e82ad1e186d1c4724a601a24b2d9ecd4c0ca582b89fe4b1ec2cfa5d242fa6cf'
NEW_DOWNSTREAM_KEYS={'fc_norm.weight','fc_norm.bias','head.weight','head.bias'}
EXCLUDED_PRETRAIN_KEYS={'student.mask_token','student.norm.weight','student.norm.bias',
    'student.lm_head.weight','student.lm_head.bias','lm_head.weight','lm_head.bias',
    'projection_head.0.weight','projection_head.0.bias'}

def build_encoder(author_source,vendor,*,seed=1729):
    source=Path(author_source);vendor=Path(vendor)
    assert hashlib.sha256((source/'models/labram.py').read_bytes()).hexdigest()==LABRAM_SOURCE_SHA256
    sys.path.insert(0,str(vendor));sys.path.insert(0,str(source))
    if 'models' in sys.modules:
        assert list(sys.modules['models'].__path__)==[str(source/'models')]
    else:
        package=types.ModuleType('models');package.__path__=[str(source/'models')];sys.modules['models']=package
    import dist
    dist.__dict__['__device']='cpu'
    from models.labram import NeuralTransformer
    # fork_rng prevents this private constructor from changing a caller's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=NeuralTransformer(patch_size=200,embed_dim=200,depth=12,num_heads=10,
            mlp_ratio=4,qk_norm=partial(torch.nn.LayerNorm,eps=1e-6),
            norm_layer=partial(torch.nn.LayerNorm,eps=1e-6),num_classes=1024,
            use_mean_pooling=True,use_abs_pos_emb=True,use_rel_pos_bias=False,
            qkv_bias=False,init_values=.1,drop_path_rate=.1)
    return model

def transfer_clinical(model,checkpoint_path):
    path=Path(checkpoint_path)
    with path.open('rb') as f:actual_sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual_sha==CLINICAL_SHA256
    from torch_numpy_scalar_checkpoint_20261028_v3 import load_tensor_checkpoint
    checkpoint,reader_proof=load_tensor_checkpoint(path)
    assert isinstance(checkpoint,dict) and isinstance(checkpoint.get('model'),dict)
    original=checkpoint['model'];assert len(original)==229
    assert EXCLUDED_PRETRAIN_KEYS<=set(original)
    selected={}
    for key,tensor in original.items():
        assert isinstance(tensor,torch.Tensor)
        if key in EXCLUDED_PRETRAIN_KEYS:continue
        mapped=key[8:] if key.startswith('student.') else key
        assert mapped not in selected
        selected[mapped]=tensor
    target=model.state_dict();assert set(target)-set(selected)==NEW_DOWNSTREAM_KEYS
    assert not set(selected)-set(target)
    for key,value in selected.items():
        assert value.shape==target[key].shape and value.dtype==target[key].dtype,key
        assert torch.isfinite(value).all(),key
    initial_new={k:target[k].clone() for k in NEW_DOWNSTREAM_KEYS}
    result=model.load_state_dict(selected,strict=False)
    assert set(result.missing_keys)==NEW_DOWNSTREAM_KEYS and not result.unexpected_keys
    loaded=model.state_dict()
    for key,value in selected.items():assert torch.equal(loaded[key].cpu(),value.cpu()),key
    for key,value in initial_new.items():assert torch.equal(loaded[key],value),key
    return dict(status='ALL_EXPECTED_CLINICAL_BACKBONE_TENSORS_EXACTLY_TRANSFERRED',
        source_sha256=actual_sha,source_state_tensors=len(original),
        loaded_tensors=len(selected),loaded_parameter_elements=sum(v.numel() for v in selected.values()),
        new_downstream_keys=sorted(NEW_DOWNSTREAM_KEYS),excluded_pretraining_keys=sorted(EXCLUDED_PRETRAIN_KEYS),
        downstream_final_norm='New fc_norm; pretraining student.norm excluded as in author mean-pooling loader',
        reader_proof=reader_proof,parameter_count=sum(p.numel() for p in model.parameters()))
