"""Fetch bounded public source/metadata only; never model weights or GPU work."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parent
COMMIT = 'f06db69fcc4a029439e369c5c3dba6099e57560e'
REPO = 'ssshamiii/Brain-HIVE'
DEST = BASE / 'reproduction_sources' / ('BrainHIVE_' + COMMIT[:8])
MAX_ONE = 3_000_000
MAX_TOTAL = 12_000_000


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'EEG-Reproduction-Audit'})
    with urllib.request.urlopen(request, timeout=25) as response:
        data = response.read(MAX_ONE + 1)
    if len(data) > MAX_ONE:
        raise RuntimeError('Source/metadata exceeds bounded download: ' + url)
    return data


def save(relative, data):
    path = DEST / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError('Refusing to overwrite changed evidence: ' + str(path))
    path.write_bytes(data)
    return {'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    started = datetime.now(timezone.utc).isoformat()
    tree_url = f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'
    raw = fetch(tree_url)
    tree = json.loads(raw)
    if tree.get('truncated'):
        raise RuntimeError('Truncated source tree')
    items = [x for x in tree['tree'] if x['type'] == 'blob' and
             (x['path'].endswith(('.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.md'))
              or x['path'] in ('LICENSE', 'requirements.txt'))]
    if sum(x['size'] for x in items) > MAX_TOTAL:
        raise RuntimeError('Source exceeds total size bound')
    receipts = [save('_source_tree.json', raw)]

    def download(item):
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{item['path']}"
        data = fetch(url)
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != item['sha'] or len(data) != item['size']:
            raise RuntimeError('Source blob verification failed: ' + item['path'])
        receipt = save(item['path'], data)
        receipt.update(url=url, git_blob_sha=blob, expected_git_blob_sha=item['sha'])
        return receipt

    with ThreadPoolExecutor(6) as pool:
        receipts.extend(pool.map(download, items))
    metadata_urls = {
        'fusion_prior_H14': 'https://huggingface.co/api/models/fakekungfu/Fusion-Prior-H14_B32_VAE?blobs=true',
        'fusion_prior_SynCLR': 'https://huggingface.co/api/models/fakekungfu/Fusion-Prior-SynCLR_B32_VAE?blobs=true',
        'visual_embeddings': 'https://huggingface.co/api/datasets/fakekungfu/Brain-HIVE_Visual_Embeddings?blobs=true',
    }
    summaries = {}
    for name, url in metadata_urls.items():
        data = fetch(url)
        receipts.append(save('_metadata/' + name + '.json', data))
        meta = json.loads(data)
        files = meta.get('siblings', [])
        root_weights = [x for x in files if x['rfilename'].count('/') == 1 and
                        x['rfilename'].endswith('.safetensors')]
        summaries[name] = {'revision': meta.get('sha'), 'files': len(files),
                           'root_weight_bytes': sum(x.get('size', 0) for x in root_weights),
                           'root_weights': root_weights}
        if name == 'fusion_prior_H14':
            revision = meta['sha']
            for item in files:
                if item['rfilename'] in ('fusion_encoder/config.json', 'proj/config.json', 'attn_adapter/config.json'):
                    config_url = ('https://huggingface.co/fakekungfu/Fusion-Prior-H14_B32_VAE/resolve/'
                                  + revision + '/' + item['rfilename'])
                    receipts.append(save('_metadata/H14/' + item['rfilename'], fetch(config_url)))
    report = {'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'PINNED_SOURCE_AND_METADATA_FETCHED_NO_MODELS_OR_GPU',
              'repository': REPO, 'commit': COMMIT, 'tree': tree['sha'],
              'source_files': len(items), 'all_source_blobs_verified': True,
              'downloaded_total_bytes': sum(x['bytes'] for x in receipts),
              'metadata_summary': summaries, 'artifacts': receipts,
              'weights_downloaded': False, 'gpu_work_started': False}
    save('_fetch_receipt.json', (json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({k: report[k] for k in ('status', 'commit', 'source_files', 'downloaded_total_bytes')}))
    for name, item in summaries.items():
        print(name, item['revision'], 'root_weight_bytes', item['root_weight_bytes'])


if __name__ == '__main__':
    main()
