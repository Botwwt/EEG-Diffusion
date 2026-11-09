"""Actual author full EEG encoder, random weights, single-forward contract checks."""
import datetime,hashlib,importlib,json,sys,time,types
from functools import partial
from pathlib import Path
import numpy as np
import torch

L=Path(__file__).resolve().parent
S=L/'reproduction_sources/direction_switch_20261028_v1/avde_3be06a90ae69'
V=L/'reproduction_sources/multiscale_runtime_20261028_v1/vendor'
sys.path.insert(0,str(V));sys.path.insert(0,str(S))
package=types.ModuleType('models');package.__path__=[str(S/'models')]
assert 'models' not in sys.modules
sys.modules['models']=package
import dist
dist.__dict__['__device']='cpu'
from models.labram import NeuralTransformer,get_input_chans
from eeg_data.preprocess_utils import chan_order
from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform,global_and_patch_conditions

torch.set_num_threads(2);torch.manual_seed(1729);began=time.monotonic()
encoder=NeuralTransformer(patch_size=200,embed_dim=200,depth=12,num_heads=10,mlp_ratio=4,qk_norm=partial(torch.nn.LayerNorm,eps=1e-6),norm_layer=partial(torch.nn.LayerNorm,eps=1e-6),num_classes=1024,use_mean_pooling=True,use_abs_pos_emb=True,use_rel_pos_bias=False,qkv_bias=False,init_values=.1,drop_path_rate=.1)
positions=get_input_chans([x.upper() for x in chan_order]);assert len(positions)==64
original_parameter_ids={id(p) for p in encoder.parameters()}
transform=CacheNativeEEGTransform(tuple(chan_order),np.zeros(63),np.ones(63))
rng=np.random.default_rng(1729);source=rng.normal(size=(2,63,250)).astype(np.float32)
converted=transform(source,channel_names=chan_order)
checks={}
checks['real63_named_channels_and_200_point_shape']=converted.shape==(2,63,1,200) and len(set(positions))==64
checks['no_test_batch_statistics']=np.array_equal(converted[:1],transform(source[:1],channel_names=chan_order))
try:transform(source,channel_names=chan_order[::-1])
except ValueError:checks['wrong_channel_order_rejected']=True
else:checks['wrong_channel_order_rejected']=False
eeg=torch.from_numpy(converted).requires_grad_();encoder.eval()
global_eeg,memory=global_and_patch_conditions(encoder,eeg,input_chans=positions)
reference=encoder(eeg,input_chans=positions)[0]
checks['global_condition_exact_author_forward']=torch.equal(global_eeg,reference)
checks['all_63_patches_retained']=memory.shape==(2,63,1024)
checks['memory_mean_matches_same_global']=torch.allclose(memory.mean(1),global_eeg,atol=1e-7,rtol=1e-5)
checks['patch_deviations_are_nonzero']=bool((memory-global_eeg[:,None]).norm()>0)
calls=[];h=encoder.register_forward_pre_hook(lambda m,args:calls.append(1))
encoder.train()
try:train_global,train_memory=global_and_patch_conditions(encoder,eeg,input_chans=positions)
finally:h.remove()
checks['one_stochastic_encoder_forward_per_pair']=len(calls)==1
checks['temporary_norm_hook_removed']=not encoder.norm._forward_hooks
(train_global.square().mean()+train_memory.square().mean()).backward()
checks['gradient_reaches_raw_EEG']=bool(torch.isfinite(eeg.grad).all() and eeg.grad.norm()>0)
checks['gradient_reaches_original_temporal_conv']=bool(next(encoder.patch_embed.parameters()).grad.norm()>0)
checks['original_encoder_parameter_objects_preserved']={id(p) for p in encoder.parameters()}==original_parameter_ids
record=dict(status='PASS' if all(checks.values()) else 'FAIL',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,seconds=time.monotonic()-began,author_full_encoder_parameters=sum(p.numel() for p in encoder.parameters()),encoder_forward_features_source_sha256=hashlib.sha256((S/'models/labram.py').read_bytes()).hexdigest(),random_weights=True,real_EEG_used=False,pretrained_checkpoint_loaded=False,GPU_jobs=0,trained_models=0)
out=L/'audit/multiscale_EEG_20261028_v1';out.mkdir(exist_ok=True)
target=out/'EEG_interface_CPU_actual_v1.json'
if target.exists() and json.loads(target.read_bytes())['status']=='FAIL':
    b=target.read_bytes();(out/('EEG_interface_CPU_failed_'+hashlib.sha256(b).hexdigest()[:12]+'.json')).write_bytes(b)
target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps(record));assert all(checks.values()),checks
