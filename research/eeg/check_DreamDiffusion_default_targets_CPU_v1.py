"""Execute the unmodified released dataset method on labelled synthetic input.

This checks a default-entrypoint problem, not the published experiments. No real
EEG, model weights, GPU, image downloads, training, or generated artifacts are used.
"""
from pathlib import Path
import ast
import hashlib
import json
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'reproduction_sources/DreamDiffusion_tree_8e147972'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    output = ROOT/'audit/DreamDiffusion_default_target_CPU_check_20261020_v1.json'
    assert not output.exists()
    dataset_path = SOURCE/'code/dataset.py'
    entrypoint_path = SOURCE/'code/eeg_ldm.py'
    tree = ast.parse(dataset_path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EEGDataset')
    identity = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'identity')
    create = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_EEG_dataset')
    defaults = dict(zip([a.arg for a in create.args.args][-len(create.args.defaults):], create.args.defaults))
    assert isinstance(defaults['imagenet_path'], ast.Constant) and defaults['imagenet_path'].value is None
    entry = ast.parse(entrypoint_path.read_text())
    main = next(n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'create_EEG_dataset']
    assert len(calls) == 1 and all(k.arg != 'imagenet_path' for k in calls[0].keywords)
    fixture = dict(dataset=[dict(eeg=torch.zeros(128, 500), label=0, image=0)],
                   labels=['synthetic_input_only'], images=['synthetic_input_only'])
    processor = lambda **kwargs: {'pixel_values': torch.zeros(1, 3, 224, 224)}
    torch_fixture = SimpleNamespace(load=lambda *args, **kwargs: fixture,
                                    from_numpy=torch.from_numpy, tensor=torch.tensor)
    namespace = dict(Dataset=Dataset, torch=torch_fixture, np=np, interp1d=interp1d, Image=Image,
                     AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor))
    exec(compile(ast.Module(body=[identity, node], type_ignores=[]), str(dataset_path), 'exec'), namespace)
    instance = namespace['EEGDataset']('synthetic_fixture_no_file_read', None,
                                      image_transform=lambda x: x, subject=0)
    np.random.seed(1729)
    first, second = instance[0], instance[0]
    a = hashlib.sha256(first['image'].tobytes()).hexdigest()
    b = hashlib.sha256(second['image'].tobytes()).hexdigest()
    assert a != b and torch.equal(first['eeg'], second['eeg'])
    assert not torch.cuda.is_initialized()
    proof = dict(status='DEFAULT_ENTRYPOINT_OMITS_IMAGE_ROOT_AND_DATASET_RETURNS_RANDOM_TARGETS',
                 source_sha256=sha(__file__), dataset_sha256=sha(dataset_path),
                 entrypoint_sha256=sha(entrypoint_path), dataset_method_line=node.lineno,
                 entrypoint_call_line=calls[0].lineno,
                 observed_entrypoint_keywords=[k.arg for k in calls[0].keywords],
                 imagenet_path_default=None, synthetic_EEG_shape=[128, 500],
                 returned_EEG_shape=list(first['eeg'].shape), returned_target_shape=list(first['image'].shape),
                 first_target_sha256=a, second_target_sha256=b, unchanged_EEG=True,
                 fixture_replaces_only=['torch.load dataset input', 'CLIP AutoProcessor output'],
                 GPU_used=False, actual_models_or_real_subjects_evaluated=False,
                 seconds=time.monotonic()-started,
                 limits=['This proves the released default-code path; it does not establish which configuration produced the paper results.',
                         'Source inspection is not a completed DreamDiffusion reproduction.',
                         'An explicit image-root interface and actual stimulus mapping are prerequisites before using this branch as a baseline.'])
    output.write_text(json.dumps(proof, indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as f:
        f.write(json.dumps(dict(stage='DreamDiffusion_default_target_source_diagnosis', GPU_used=False,
                               seconds=proof['seconds'], receipt=str(output)))+'\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
