"""One native-EEG engineering task; previous scientific studies remain immutable."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
STATUS='REGISTERED_NATIVE_EEG_DIRECTIONAL_TEACHER_ENGINEERING_ONLY'
def setup(protocol):
 spec=json.loads(Path(protocol).read_bytes());old_path=Path(spec['previous_gradient_protocol']['path'])
 sys.path.insert(0,str(old_path.parent));import brainhive_overlap_gradient_contract_CPU_20261103_v1 as H
 old,base,G=H.setup(spec['previous_gradient_protocol']['path']);G.checked(spec['previous_gradient_protocol']);G.checked(spec['previous_gradient_CPU_closure'])
 assert G.read(G.checked(spec['previous_gradient_CPU_closure']))['status']=='PASS_INDEPENDENT_ALL80_OVERLAP_STATE_LOSS_AND_SAVED_GRADIENT_CPU_AUDIT'
 assert spec['status']==STATUS and spec['host']=='old' and spec['seed']==1729
 assert spec['profile_indices']==list(range(0,80,8)) and spec['profile_domains']==[1,2] and spec['batch_engineering_queries']==list(range(8))
 assert spec['maximum_output_bytes']==384<<20 and spec['maximum_GPU_bytes']==20<<30 and spec['maximum_extra_CPU_bytes']==10<<30
 assert spec['temporary_arms']==['positive_only','directional','sameconcept_wrong_EEG'] and spec['temporary_steps_per_arm']==4
 assert spec['temporary_optimizer']==dict(lr=1e-5,weight_decay=.01,betas=[.9,.999],eps=1e-8,clip=1.)
 assert spec['new_formal_training_admitted'] is False and spec['native_output_affine_frozen'] and spec['ordinary_immutable']
 for pin in spec['source_pins']:G.checked(pin)
 return spec,old,base,G,H
def resources(spec,old,base,G,H,start=False):
 before=H.resources(old,base,G,start=start);used=0
 for root in (Path(spec['output_root']),Path(spec['code_root'])):
  if root.exists():
   for p in root.rglob('*'):
    assert not p.is_symlink()
    if p.is_file():used+=p.stat().st_blocks*512
 assert used<=spec['maximum_output_bytes'];remaining=spec['maximum_output_bytes']-used
 required=before['existing']['required_with_all_prior_commitments']+before['additional_remaining_commitments']+remaining
 assert before['existing']['external']['free_bytes']>=required
 return dict(inherited=before,own_allocated_bytes=used,own_remaining_bytes=remaining,required_with_all_previous_commitments=required,no_old_budget_released=True)
def freeze_affine(model):
 names=['block.layers.0.norm.weight','block.layers.0.norm.bias']
 for name,parameter in model.named_parameters():parameter.requires_grad_(name not in names)
 assert sum(p.numel() for p in model.parameters())==6452224
 assert sum(p.numel() for p in model.parameters() if p.requires_grad)==6450176
 return tuple(p for p in model.parameters() if p.requires_grad)
def tensor_digest(module,half=False):
 import torch
 h=hashlib.sha256()
 for name,value in module.state_dict().items():
  tensor=value.detach()
  if half and tensor.is_floating_point():tensor=tensor.half()
  tensor=tensor.cpu().contiguous();h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(tensor.numpy().tobytes())
 return h.hexdigest()
def projected_losses(prediction,teacher,direction):
 # Actual autograd carries the mean-to-sum Jacobian normalization automatically.
 d=prediction[0].double()-prediction[1].double();h=teacher[0].double()-teacher[1].double();delta=direction.double();energy=delta.square().mean()
 assert float(energy)>0
 ae=(d*delta).mean()/energy;af=(h*delta).mean()/energy;gap=d-h
 full=gap.square().mean()/energy;parallel=(ae-af).square();orth=(gap-(ae-af)*delta).square().mean()/energy
 assert abs(float(full-parallel-orth))<1e-10
 return full,parallel,orth,ae,af,energy
