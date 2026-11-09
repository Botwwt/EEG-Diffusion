import json
import torch
from EEG_generative_backend_loader_20261029_v1 import exact_transfer
class Fixture(torch.nn.Module):
    def __init__(self):super().__init__();self.backbone=torch.nn.Linear(4,4);self.embed_proj=torch.nn.Linear(4,4)
torch.manual_seed(1729);model=Fixture();state={k:torch.ones_like(v)*.25 for k,v in model.state_dict().items() if not k.startswith('embed_proj.')};state['class_emb.weight']=torch.zeros(5,4)
proof=exact_transfer(model,state,new_keys={'embed_proj.weight','embed_proj.bias'},unused={'class_emb.weight':[5,4]})
checks={}
for label,change in [('extra',lambda s:s.update(unknown=torch.ones(1))),('missing',lambda s:s.pop('backbone.bias')),('shape',lambda s:s.update({'backbone.weight':torch.ones(3,4)}))]:
    copy=dict(state);change(copy)
    try:exact_transfer(Fixture(),copy,new_keys={'embed_proj.weight','embed_proj.bias'},unused={'class_emb.weight':[5,4]})
    except AssertionError:checks[label]=True
    else:checks[label]=False
assert all(checks.values());print(json.dumps(dict(status='PASS_EXACT_TRANSFER_AND_UNKNOWN_MISSING_SHAPE_REJECTION',proof=proof,checks=checks,GPU_jobs=0,pretrained_model_loaded=False)))
