"""Reviewed upstream Brain-HIVE math under the existing pinned Torch runtime.

Only the named public class bodies are executed. This avoids unrelated optional
ATM/NICE dependencies; it does not replace the author's MLP or Fusion Prior.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'reproduction_sources/BrainHIVE_f06db69f'
WORK = Path('/root/brainhive_reproduction_20261020_v1')
ASSETS = WORK / 'assets'
PYTHON_PATHS = [
    '/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages',
    '/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages',
    '/root/mced_representation_runtime_v1/lib/python3.12/site-packages',
    '/root/mced_eeg_brainhive_parquet_packages_v1',
]
sys.path[:0] = PYTHON_PATHS
import torch
from torch import nn
import torch.nn.functional as F


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    torch.save(value, temporary)
    temporary.replace(path)


def classes(include_adapter=False):
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    scope = dict(torch=torch, nn=nn, F=F, Dict=Dict, List=List, Optional=Optional,
                 Tuple=Tuple, Union=Union, ModelMixin=ModelMixin, ConfigMixin=ConfigMixin,
                 register_to_config=register_to_config, Attention=Attention,
                 AttnProcessor2_0=AttnProcessor2_0, UNet2DConditionModel=UNet2DConditionModel)
    receipt = json.loads((SOURCE / '_fetch_receipt.json').read_text())
    expected = {a['path']: a['sha256'] for a in receipt['artifacts']}
    selections = [('main/layers.py', ['BaseMLP', 'ResNet1DLayer', 'ResNet1DBlock', 'FusionEncoder']),
                  ('main/models_brain.py', ['BrainMLP'])]
    if include_adapter:
        selections.append(('main/models_adapter.py', ['IPAttnProcessor2_0', 'IPProjectionModel',
                                                      'IPAttnAdapterModel', 'FusionEncoderModel']))
    for name, selected in selections:
        path = SOURCE / name
        assert sha(path) == expected[name]
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name in selected]
        assert [n.name for n in nodes] == selected
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def prior_models(device='cpu', include_attention=False):
    from safetensors.torch import load_file
    scope = classes(include_adapter=True)
    models = {}
    for name, cls in [('fusion_encoder', 'FusionEncoderModel'), ('proj', 'IPProjectionModel')]:
        directory = ASSETS / 'prior_H14' / name
        cfg = json.loads((directory / 'config.json').read_text())
        model = scope[cls](**{k: v for k, v in cfg.items() if not k.startswith('_')})
        payload = directory / 'diffusion_pytorch_model.safetensors'
        verified = json.loads((WORK / 'audit/native_asset_verification.json').read_text())
        assert sha(payload) == verified['files'][str(payload)]['sha256']
        weights = load_file(str(payload))
        assert set(weights) == set(model.state_dict())
        model.load_state_dict(weights, strict=True)
        models[name] = model.to(device).eval().requires_grad_(False)
    if include_attention:
        directory = ASSETS / 'prior_H14/attn_adapter'
        cfg = json.loads((directory / 'config.json').read_text())
        model = scope['IPAttnAdapterModel'](**{k: v for k, v in cfg.items() if not k.startswith('_')})
        payload = directory / 'diffusion_pytorch_model.safetensors'
        verified = json.loads((WORK / 'audit/native_asset_verification.json').read_text())
        assert sha(payload) == verified['files'][str(payload)]['sha256']
        weights = load_file(str(payload))
        assert set(weights) == set(model.state_dict())
        model.load_state_dict(weights, strict=True)
        models['attn_adapter'] = model.to(device).eval().requires_grad_(False)
    return models
