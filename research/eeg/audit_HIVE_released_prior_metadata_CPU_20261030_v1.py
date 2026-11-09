"""Read small public metadata/config files only; never download model weights."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audit/HIVE_released_prior_metadata_20261030_v1'
MAXIMUM_BYTES = 2 * 1024**2
PRIOR_REVISIONS = {
    'H14_B32_VAE': '135a284109e71f5cde3d3e6d5dee6268a3c7987a',
    'SynCLR_B32_VAE': 'e949706d5c344f38f80bc3492b438226205ce474',
}


def get_small_json(name, url):
    parsed = urlparse(url)
    assert parsed.scheme == 'https' and parsed.netloc in ('api.github.com', 'huggingface.co')
    assert not parsed.path.endswith(('.safetensors', '.pt', '.pth', '.bin', '.parquet'))
    started = time.monotonic()
    result = dict(name=name, requested_url=url, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        request = Request(url, headers={'User-Agent': 'EEG-public-metadata-research/1.0', 'Accept': 'application/json'})
        with urlopen(request, timeout=25) as response:
            raw = response.read(MAXIMUM_BYTES + 1)
            assert len(raw) <= MAXIMUM_BYTES, 'Metadata size bound exceeded'
            data = json.loads(raw)
            result.update(status='FETCHED_JSON', HTTP_status=response.status, final_url=response.url,
                          bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        path = OUT / (name + '.json')
        path.write_bytes(raw)
        result['local_path'] = path.relative_to(ROOT).as_posix()
        return result, data
    except Exception as error:
        result.update(status='UNRESOLVED_FETCH_FAILURE', error_type=type(error).__name__, error=str(error),
                      HTTP_status=error.code if isinstance(error, HTTPError) else None)
        return result, None
    finally:
        result['seconds'] = time.monotonic() - started


def main():
    OUT.mkdir(exist_ok=False)
    requests = [
        ('author_GitHub_main_commit', 'https://api.github.com/repos/ssshamiii/Brain-HIVE/commits/main'),
        ('author_public_models', 'https://huggingface.co/api/models?author=fakekungfu&limit=100'),
        ('visual_cache_model_main', 'https://huggingface.co/api/models/fakekungfu/Brain-HIVE_Visual_Embeddings?blobs=true'),
        ('visual_cache_dataset_main', 'https://huggingface.co/api/datasets/fakekungfu/Brain-HIVE_Visual_Embeddings?blobs=true'),
    ]
    for label, revision in PRIOR_REVISIONS.items():
        repository = 'fakekungfu/Fusion-Prior-' + label
        requests.append((label + '_pinned_metadata', f'https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true'))
        for component in ('fusion_encoder', 'proj', 'attn_adapter'):
            requests.append((label + '_' + component + '_config', f'https://huggingface.co/{repository}/resolve/{revision}/{component}/config.json'))
    receipts, fetched = [], {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {executor.submit(get_small_json, name, url): name for name, url in requests}
        for task in as_completed(pending):
            receipt, data = task.result()
            receipts.append(receipt)
            if data is not None:
                fetched[receipt['name']] = data
    summary = dict(status='COMPLETE_BOUNDED_PUBLIC_METADATA_ATTEMPTS_NOT_MODEL_REPRODUCTION',
                   UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   receipts=sorted(receipts, key=lambda x: x['name']),
                   weight_or_feature_payload_bytes_downloaded=0, GPU_jobs=0, neural_updates=0,
                   existing_study_sources_or_protocols_changed=False)
    if 'author_GitHub_main_commit' in fetched:
        commit = fetched['author_GitHub_main_commit']
        summary['observed_author_main_commit'] = dict(sha=commit['sha'], tree=commit['commit']['tree']['sha'],
                                                     message=commit['commit']['message'])
    if 'author_public_models' in fetched:
        summary['observed_author_public_model_ids'] = [x.get('id', x.get('modelId')) for x in fetched['author_public_models']]
    priors = {}
    for label, revision in PRIOR_REVISIONS.items():
        metadata = fetched.get(label + '_pinned_metadata')
        item = dict(requested_revision=revision)
        if metadata:
            assert metadata['sha'] == revision
            item.update(actual_revision=metadata['sha'], gated=metadata.get('gated'), private=metadata.get('private'))
            item['root_weights'] = [x for x in metadata.get('siblings', [])
                                    if x['rfilename'] in (component + '/diffusion_pytorch_model.safetensors' for component in ('fusion_encoder', 'proj', 'attn_adapter'))]
        item['configs'] = {component: fetched.get(label + '_' + component + '_config')
                           for component in ('fusion_encoder', 'proj', 'attn_adapter')}
        priors[label] = item
    summary['priors'] = priors
    for label in ('visual_cache_model_main', 'visual_cache_dataset_main'):
        data = fetched.get(label)
        if data:
            summary[label] = dict(sha=data.get('sha'), gated=data.get('gated'),
                                  SynCLR_feature_files=[x for x in data.get('siblings', []) if 'synclr' in x['rfilename'].lower()])
    (OUT / 'actual_metadata_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print(json.dumps(dict(directory=str(OUT),successful=len(fetched),attempts=len(requests),
                          all_small_public_JSON=True,models_downloaded=0,GPU_jobs=0)))


if __name__ == '__main__':
    main()
