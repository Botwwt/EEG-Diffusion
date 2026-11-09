"""Execute reviewed upstream class bodies on synthetic CPU inputs, without GPU."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'reproduction_sources/BrainHIVE_f06db69f'
PATCHED = BASE / 'reproduction_sources/BrainHIVE_f06db69f_parent_compat_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path, names, scope):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert [n.name for n in nodes] == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)


def main():
    import torch
    from torch import nn
    import torch.nn.functional as F
    from typing import Dict, List, Tuple, Union
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    scope = dict(torch=torch, nn=nn, F=F, Dict=Dict, List=List, Tuple=Tuple, Union=Union,
                 ATM=type('ATM', (nn.Module,), {}))
    extract(SOURCE / 'main/layers.py',
            ['BaseMLP', 'ResNet1DLayer', 'ResNet1DBlock', 'FusionEncoder'], scope)
    extract(SOURCE / 'main/models_brain.py', ['BrainMLP', 'BrainEncoder'], scope)
    tree = ast.parse((SOURCE / 'main/models_clip.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BrainCLIPModel')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_brain_features')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE / 'main/models_clip.py'), 'exec'), scope)
    torch.manual_seed(1729)
    model = scope['BrainEncoder'](backbone='brain_mlp', dropout=.1).eval()
    owner = SimpleNamespace(config=SimpleNamespace(brain_backbone='brain_mlp'), brain_model=model)
    signals = torch.zeros((2, 17, 250))
    kwargs = {'subject_ids': torch.ones(2, dtype=torch.long)}
    try:
        scope['get_brain_features'](owner, signals, **kwargs)
    except TypeError as error:
        original_error = str(error)
        assert 'subject_ids' in original_error
    else:
        raise AssertionError('Expected upstream default MLP signature failure')
    parameters = sum(p.numel() for p in model.parameters())
    del model, owner
    extract(PATCHED / 'main/models_brain.py', ['BrainMLP', 'BrainEncoder'], scope)
    torch.manual_seed(1729)
    model = scope['BrainEncoder'](backbone='brain_mlp', dropout=.1).eval()
    owner = SimpleNamespace(config=SimpleNamespace(brain_backbone='brain_mlp'), brain_model=model)
    with torch.inference_mode():
        output = scope['get_brain_features'](owner, signals, **kwargs)
    assert output.shape == (2, 1024) and torch.isfinite(output).all()
    state_sha = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    torch.manual_seed(1729)
    other = scope['BrainEncoder'](backbone='brain_mlp', dropout=.1).eval()
    state_sha2 = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in other.parameters())).hexdigest()
    assert state_sha == state_sha2
    cfg = json.loads((SOURCE / '_metadata/H14/fusion_encoder/config.json').read_text())
    with torch.device('meta'):
        fuser = scope['FusionEncoder'](**{k: cfg[k] for k in
            ('hidden_size', 'intermediate_size', 'num_layers', 'proj_meta')})
        adapter = scope['ResNet1DLayer'](1024, 4096, norm=False)
        projector = nn.Sequential(nn.Linear(1024, 4 * 2048), nn.LayerNorm(2048))
    channels = ast.literal_eval(next(l.split('=', 1)[1].strip("' ") for l in
        (SOURCE / 'scripts/build_reconstruction.sh').read_text().splitlines()
        if l.startswith('SELECTED_CHANNELS=')))
    fixed_channels = ast.literal_eval(next(l.split('=', 1)[1].strip("' ") for l in
        (PATCHED / 'scripts/build_reconstruction.sh').read_text().splitlines()
        if l.startswith('SELECTED_CHANNELS=')))
    assert channels.count('Oz') == 2 and 'POz' not in channels
    assert len(fixed_channels) == len(set(fixed_channels)) == 17 and 'POz' in fixed_channels
    assert 'ip_hidden_states = proj(fusion_adapter(brain_embeds))' in (PATCHED / 'build_reconstruction.py').read_text()
    assert 'ip_hidden_states = proj(brain_embeds)' in (SOURCE / 'build_reconstruction.py').read_text()
    report = {'utc': datetime.now(timezone.utc).isoformat(),
              'status': 'SOURCE_DEFECT_REPRODUCED_AND_PATCHED_CPU_SIGNATURE_CHECK_PASSED',
              'script_sha256': digest(Path(__file__)), 'upstream_error': original_error,
              'synthetic_input_shape': list(signals.shape), 'output_shape': list(output.shape),
              'output_finite': True, 'brain_parameters': parameters,
              'trainable_logit_temperature_parameters': 1,
              'H14_fuser_parameters': sum(p.numel() for p in fuser.parameters()),
              'prior_residual_adapter_parameters': sum(p.numel() for p in adapter.parameters()),
              'projection_parameters': sum(p.numel() for p in projector.parameters()),
              'seed_before_initialization_repeatable_on_CPU': state_sha == state_sha2,
              'original_reconstruction_channels': channels, 'corrected_channels': fixed_channels,
              'required_channel_jsonl_present_in_source': (SOURCE / 'configs/THINGS_EEG_CHANNELS.jsonl').exists(),
              'fusion_residual_restored_by_source_check_only': True,
              'pretrained_weights_loaded': False, 'actual_EEG_model_forward': False,
              'optimizer_updates': 0, 'generated_images': 0, 'GPU_used': False,
              'paper_reproduction_complete': False, 'component_gain_established': False}
    if os.name == 'posix':
        asset = Path('/root/mced_eeg_brainhive_ram_assets_v1')
        old = json.loads((asset / 'results.json').read_text())
        report['legacy_volatile_assets_currently_available'] = [
            {'relative_path': a['relative_path'], 'exists': (asset / a['relative_path']).exists()}
            for a in old['assets']]
        report['server_free_bytes'] = {p: shutil.disk_usage(p).free for p in
                                      ('/root', '/root/autodl-tmp', '/dev/shm')}
        layout = Path('/root/mced_eeg_full_montage_cache_v3/sub-01_train_layout.json')
        meta = json.loads(layout.read_text())
        report['known_source_layout'] = {k: meta.get(k) for k in
            ('source_asset', 'raw_shape', 'raw_dtype', 'channels', 'mean_shape', 'output_shape')}
        report['known_source_layout_sha256'] = digest(layout)
        report['selected_indices_from_actual_channel_metadata'] = [meta['channels'].index(n) for n in fixed_channels]
    out = BASE / 'audit/BrainHIVE_parent_CPU_check_20261020_v1.json'
    assert not out.exists(), 'Preserve executed audit'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'brain_parameters',
        'H14_fuser_parameters', 'prior_residual_adapter_parameters', 'projection_parameters', 'GPU_used')}))


if __name__ == '__main__':
    main()
