"""Finite download/CPU inspection of pinned author weights; no GPU or fitting."""
import argparse, datetime, hashlib, json, os, shutil, time, urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
a = p.parse_args()
spec = json.loads(a.manifest.read_bytes())
root = Path(spec['output_root'])
assert root.is_absolute() and root.is_relative_to('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
root.mkdir(parents=True, exist_ok=True)
start = time.monotonic()
deadline = start + spec['download_deadline_seconds']

def save(name, value):
    value = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), **value)
    target = root / name
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    os.replace(temporary, target)

def available_memory():
    line = next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    return int(line.split()[1])*1024

def digests(path):
    sha = hashlib.sha256()
    blob = hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0')
    with path.open('rb') as source:
        for data in iter(lambda: source.read(4 << 20), b''):
            sha.update(data)
            blob.update(data)
    return dict(sha256=sha.hexdigest(), git_blob_sha1=blob.hexdigest())

def verify(entry, path):
    assert path.stat().st_size == entry['bytes'], str(path)
    actual = digests(path)
    for key in ('sha256', 'git_blob_sha1'):
        if key in entry:
            assert actual[key] == entry[key], key
    return dict(path=str(path), bytes=path.stat().st_size, **actual)

try:
    assert shutil.disk_usage(root).free >= spec['minimum_free_disk_bytes']
    assert available_memory() >= spec['minimum_free_memory_bytes']
    receipts = []
    # The small clinical pretrained encoder is inspected first. No THINGS-
    # trained final checkpoint is being borrowed across our protected split.
    for entry in sorted(spec['weights'], key=lambda x: x['bytes']):
        assert time.monotonic() < deadline
        name = entry['name']
        assert Path(name).name == name
        target = root / name
        if target.exists():
            receipts.append(verify(entry, target))
            continue
        temporary = root / (name + '.partial')
        assert not temporary.exists(), 'Retain and inspect partial attempts before retrying'
        request = urllib.request.Request(entry['url'], headers={'User-Agent': 'EEG-reproducible-public-assets'})
        done, last = 0, 0.
        with urllib.request.urlopen(request, timeout=35) as response, temporary.open('xb') as dest:
            assert response.status == 200
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('Finite public download deadline')
                data = response.read(1 << 20)
                if not data:
                    break
                done += len(data)
                assert done <= entry['bytes']
                dest.write(data)
                if time.monotonic()-last >= 5:
                    save('progress.json', dict(status='DOWNLOADING_PINNED_PUBLIC_WEIGHTS', current=name, current_bytes=done, expected_bytes=entry['bytes'], completed=receipts, seconds=time.monotonic()-start, GPU_jobs=0))
                    last=time.monotonic()
        receipt=verify(entry, temporary)
        os.replace(temporary, target)
        receipt['path']=str(target)
        receipts.append(receipt)
        save('progress.json', dict(status='PINNED_PUBLIC_ASSET_VERIFIED', current=name, completed=receipts, seconds=time.monotonic()-start, GPU_jobs=0))
    assert available_memory() >= spec['minimum_free_memory_bytes']
    # CPU-only, restricted unpickling. Strict model loading and real CUDA
    # backward/inference profiles remain separate, unperformed admission steps.
    import torch
    torch.set_num_threads(1)
    checkpoint_inventory=[]
    for entry in spec['weights']:
        checkpoint=torch.load(root/entry['name'], map_location='cpu', weights_only=True)
        state=checkpoint.get('model',checkpoint) if isinstance(checkpoint,dict) else checkpoint
        assert isinstance(state,dict)
        tensors={k:v for k,v in state.items() if isinstance(v,torch.Tensor)}
        assert tensors
        shapes={k:list(v.shape) for k,v in tensors.items()}
        checkpoint_inventory.append(dict(name=entry['name'], state_tensors=len(tensors), elements=sum(v.numel() for v in tensors.values()), tensor_shapes=shapes))
        del checkpoint,state,tensors
    save('completion.json', dict(status='PINNED_PUBLIC_WEIGHTS_AND_RESTRICTED_CPU_CHECKPOINT_READ_COMPLETE', assets=receipts, checkpoint_inventory=checkpoint_inventory, seconds=time.monotonic()-start, GPU_jobs=0, trained_models=0, strict_architecture_load_or_scientific_benefit_proved=False))
except BaseException as error:
    import traceback
    save('failure_'+str(time.time_ns())+'.json', dict(status='PUBLIC_ASSET_PREPARATION_FAILED', error=repr(error), traceback=traceback.format_exc(), seconds=time.monotonic()-start, partial_files_preserved=True, GPU_jobs=0, trained_models=0))
    raise
