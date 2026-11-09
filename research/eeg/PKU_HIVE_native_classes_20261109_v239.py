"""Unchanged pinned author class ASTs, without unrelated optional imports."""
import ast,hashlib,json
from pathlib import Path
from typing import Dict,List,Tuple,Union
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path('/dfs/share/maleiLab/wwt/eeg')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def classes():
    pins=json.loads((ROOT/'code/HIVE_author_source_pins_v239.json').read_text())
    scope=dict(torch=torch,nn=nn,F=F,Dict=Dict,List=List,Tuple=Tuple,Union=Union)
    for name,names in [('HIVE_layers_source_v239.py',['BaseMLP','ResNet1DLayer','ResNet1DBlock','FusionEncoder']),
                       ('HIVE_brain_source_v239.py',['BrainMLP'])]:
        p=ROOT/'code'/name;assert sha(p)==pins[name]
        nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name in names]
        assert [n.name for n in nodes]==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
    return scope
def teacher(scope):
    from safetensors.torch import load_file
    cfg=json.loads((ROOT/'code/HIVE_fusion_config_v239.json').read_text())
    model=scope['FusionEncoder'](**{k:v for k,v in cfg.items() if not k.startswith('_')})
    a=scope['ResNet1DLayer'](cfg['hidden_size'],cfg['intermediate_size'],norm=False)
    weights=load_file(str(ROOT/'assets/native_teacher_v239/fusion_encoder.safetensors'))
    w={k[len('vision_model.'):]:v for k,v in weights.items() if k.startswith('vision_model.')}
    aw={k[len('adapter.'):]:v for k,v in weights.items() if k.startswith('adapter.')}
    assert len(w)+len(aw)==len(weights)
    model.load_state_dict(w,strict=True);a.load_state_dict(aw,strict=True)
    assert sum(p.numel() for p in model.parameters())==11015168
    assert sum(p.numel() for p in a.parameters())==8388608
    model.eval().requires_grad_(False)
    return model
