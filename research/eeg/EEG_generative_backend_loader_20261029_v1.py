"""Strict, explicit author VQ/VAR transfer without importing the model factory."""
import hashlib,json,sys,types
from pathlib import Path
import torch

SCALES=(1,2,3,4,5,6,8,10,13,16)
VQ_SHA='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c'
VAR_SHA='4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def exact_transfer(model,state,*,new_keys,unused):
    assert isinstance(state,dict) and all(isinstance(v,torch.Tensor) for v in state.values())
    expected=model.state_dict();assert set(expected)-set(state)==set(new_keys)
    assert set(state)-set(expected)==set(unused)
    for k,shape in unused.items():assert list(state[k].shape)==shape,k
    initial={k:expected[k].clone() for k in new_keys};selected={k:v for k,v in state.items() if k in expected}
    for k,v in selected.items():
        assert v.shape==expected[k].shape and v.dtype==expected[k].dtype,k
        assert torch.isfinite(v).all() if v.is_floating_point() and 'attn_bias_for_masking' not in k else True,k
    result=model.load_state_dict(selected,strict=False)
    assert set(result.missing_keys)==set(new_keys) and not result.unexpected_keys
    current=model.state_dict()
    for k,v in selected.items():assert torch.equal(current[k],v),k
    for k,v in initial.items():assert torch.equal(current[k],v),k
    return dict(source_tensors=len(state),loaded_tensors=len(selected),loaded_elements=sum(v.numel() for v in selected.values()),new_seeded_keys=sorted(new_keys),unused_keys_with_expected_shapes=unused,all_transferred_tensors_bitwise_equal=True)

def build_vq_only(author_source,vendor,vq_path,*,seed=1729):
    source=Path(author_source);sys.path.insert(0,str(vendor));sys.path.insert(0,str(source))
    inventory=json.loads((source/'source_inventory.json').read_bytes())
    for item in inventory['files']:assert sha(source/item['path'])==item['sha256']
    if 'models' not in sys.modules:
        package=types.ModuleType('models');package.__path__=[str(source/'models')];sys.modules['models']=package
    else:assert list(sys.modules['models'].__path__)==[str(source/'models')]
    from models.vqvae import VQVAE
    assert sha(vq_path)==VQ_SHA;torch.random.default_generator.manual_seed(seed)
    vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,test_mode=True,share_quant_resi=4,v_patch_nums=SCALES)
    assert sum(p.numel() for p in vae.parameters())==108948355
    state=torch.load(vq_path,map_location='cpu',weights_only=True,mmap=True);proof=exact_transfer(vae,state,new_keys=set(),unused={});del state
    vae.eval().requires_grad_(False)
    return vae,dict(status='EXACT_FROZEN_AUTHOR_VQ_TRANSFER_ALL_TENSORS',sha256=VQ_SHA,**proof)

def build_and_load(author_source,vendor,vq_path,var_path,*,seed=1729):
    source=Path(author_source);sys.path.insert(0,str(vendor));sys.path.insert(0,str(source))
    inventory=json.loads((source/'source_inventory.json').read_bytes())
    for item in inventory['files']:assert sha(source/item['path'])==item['sha256']
    if 'models' not in sys.modules:
        package=types.ModuleType('models');package.__path__=[str(source/'models')];sys.modules['models']=package
    else:assert list(sys.modules['models'].__path__)==[str(source/'models')]
    import dist
    dist.__dict__['__device']='cpu'
    from models.vqvae import VQVAE
    from models.var_eeg import VAREEG
    assert sha(vq_path)==VQ_SHA and sha(var_path)==VAR_SHA
    torch.random.default_generator.manual_seed(seed)
    vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,test_mode=True,share_quant_resi=4,v_patch_nums=SCALES)
    decoder=VAREEG(vae,depth=16,embed_dim=1024,num_heads=16,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=.1*16/24,norm_eps=1e-6,shared_aln=False,cond_drop_rate=0.,attn_l2_norm=True,patch_nums=SCALES,flash_if_available=False,fused_if_available=False)
    decoder.init_weights(init_adaln=.5,init_adaln_gamma=1e-5,init_head=.02,init_std=-1)
    assert sum(p.numel() for p in vae.parameters())==108948355
    assert sum(p.numel() for p in decoder.parameters())==310308096
    vq_state=torch.load(vq_path,map_location='cpu',weights_only=True,mmap=True)
    vq_proof=exact_transfer(vae,vq_state,new_keys=set(),unused={})
    del vq_state
    var_state=torch.load(var_path,map_location='cpu',weights_only=True,mmap=True)
    var_proof=exact_transfer(decoder,var_state,new_keys={'embed_proj.weight','embed_proj.bias'},unused={'class_emb.weight':[1001,1024]})
    del var_state
    vae.eval().requires_grad_(False);decoder.eval()
    return vae,decoder,dict(status='EXACT_AUTHOR_VQ_AND_EXPLICIT_CLASS_TO_EEG_VAR_TRANSFER',VQ=vq_proof,VAR=var_proof,VQ_SHA256=VQ_SHA,VAR_SHA256=VAR_SHA,seed=seed,new_EEG_projection_pretrained=False,clinical_EEG_encoder_part_of_this_loader=False,factory_global_initialization_monkeypatch_invoked=False,attention_l2_normalization=True,conditional_dropout_disabled_for_engineering=True)
