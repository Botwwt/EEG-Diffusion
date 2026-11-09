"""Public HF -> bounded RAM chunks -> SSH server. No large local files."""
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
import urllib.request

sys.path.insert(0, r'C:\Users\Asus\AppData\Roaming\Python\Python311\site-packages')
import paramiko

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'reproduction_sources/BrainHIVE_f06db69f'
REMOTE = '/root/brainhive_reproduction_20261020_v1/assets'
CHUNK = 8 << 20


def main():
    prior = json.loads((SOURCE / '_metadata/fusion_prior_H14.json').read_text())
    features = json.loads((SOURCE / '_metadata/visual_embeddings.json').read_text())
    choices = []
    # Small train caches first permit source/data checks while attention downloads.
    for item in features['siblings']:
        name = item['rfilename']
        if name.startswith('things-eeg/') and name.endswith('.parquet') and any(
                x in name for x in ('CLIP-ViT-H-14-laion2B-s32B-b79K',
                                   'CLIP-ViT-B-32-laion2B-s34B-b79K', '_vae-')):
            choices.append(dict(path='features/' + name, repo='datasets/fakekungfu/Brain-HIVE_Visual_Embeddings',
                                revision=features['sha'], source_path=name, bytes=item['size'],
                                sha256=item['lfs']['sha256']))
    choices.sort(key=lambda a: ('things_train_' not in a['source_path'], a['source_path']))
    assert len(choices) == 12
    for item in prior['siblings']:
        name = item['rfilename']
        if name.count('/') == 1 and name.endswith('.safetensors'):
            choices.append(dict(path='prior_H14/' + name, repo='fakekungfu/Fusion-Prior-H14_B32_VAE',
                                revision=prior['sha'], source_path=name, bytes=item['size'],
                                sha256=item['lfs']['sha256']))
    assert len(choices) == 15
    # Prior F and P first, larger attention K/V last.
    choices[12:] = sorted(choices[12:], key=lambda a: a['bytes'])
    password = getpass.getpass('SSH password: ')
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('CLOUD_LOGIN_HOST', port=54067, username='root',
                   password=password, timeout=25, banner_timeout=25)
    password = None
    started = time.monotonic()
    try:
        sftp = client.open_sftp()

        def mkdir(path):
            parts = PurePosixPath(path).parts
            current = ''
            for part in parts:
                current = '/' if part == '/' else current.rstrip('/') + '/' + part
                try:
                    sftp.stat(current)
                except FileNotFoundError:
                    sftp.mkdir(current)

        mkdir(REMOTE)
        # This is persistent root storage, never the local C/D filesystem.
        _, stdout, stderr = client.exec_command(
            "/root/miniconda3/bin/python -c \"import shutil; print(shutil.disk_usage('/root').free)\"",
            timeout=20)
        free = int(stdout.read().decode().strip())
        assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
        total = sum(a['bytes'] for a in choices)
        assert free > total + (2 << 30), (free, total)
        plan = dict(utc=datetime.now(timezone.utc).isoformat(), assets=choices,
                    total_bytes=total, root_free_before=free, local_large_files_written=False,
                    maximum_HTTP_chunk_bytes=CHUNK, source_commit='f06db69fcc4a029439e369c5c3dba6099e57560e')
        with sftp.open(REMOTE + '/transfer_plan.json', 'w') as file:
            file.write(json.dumps(plan, indent=2) + '\n')
        for a in choices:
            remote = REMOTE + '/' + a['path']
            mkdir(str(PurePosixPath(remote).parent))
            try:
                existing = sftp.stat(remote)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                raise RuntimeError('Preserve existing payload; verify before a separate retry: ' + remote)
            pending = remote + '.partial'
            try:
                sftp.stat(pending)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError('Preserve partial transfer; resume needs chunk verification: ' + pending)
            h = hashlib.sha256()
            with sftp.open(pending, 'wb', bufsize=CHUNK) as destination:
                destination.set_pipelined(True)
                for lo in range(0, a['bytes'], CHUNK):
                    hi = min(lo + CHUNK, a['bytes']) - 1
                    url = ('https://huggingface.co/' + a['repo'] + '/resolve/' + a['revision']
                           + '/' + a['source_path'] + '?download=true&brainhive_range=' + str(lo))
                    errors = []
                    for attempt in range(3):
                        try:
                            request = urllib.request.Request(url, headers={
                                'Range': f'bytes={lo}-{hi}', 'Accept-Encoding': 'identity',
                                'User-Agent': 'EEG-Reproduction-Audit'})
                            with urllib.request.urlopen(request, timeout=35) as response:
                                assert response.status == 206, response.status
                                assert response.headers['Content-Range'] == f"bytes {lo}-{hi}/{a['bytes']}"
                                data = response.read(CHUNK + 1)
                            assert len(data) == hi - lo + 1
                            break
                        except Exception as exc:
                            errors.append(type(exc).__name__ + ':' + str(exc)[:120])
                    else:
                        raise RuntimeError('Range retries exhausted: ' + repr(errors))
                    destination.write(data)
                    h.update(data)
                    if lo == 0 or (hi + 1) // CHUNK % 8 == 0 or hi + 1 == a['bytes']:
                        print(json.dumps(dict(stage='STREAM_PUBLIC_ASSET', path=a['path'],
                            completed_bytes=hi + 1, total_bytes=a['bytes'],
                            elapsed_seconds=time.monotonic() - started)), flush=True)
            assert h.hexdigest() == a['sha256'], a['path']
            assert sftp.stat(pending).st_size == a['bytes']
            sftp.rename(pending, remote)
            with sftp.open(remote + '.ready.json', 'w') as file:
                file.write(json.dumps(a | dict(complete_stream_SHA256_verified=True,
                    utc=datetime.now(timezone.utc).isoformat())) + '\n')
        for name in ('fusion_encoder/config.json', 'proj/config.json', 'attn_adapter/config.json'):
            data = (SOURCE / '_metadata/H14' / name).read_bytes()
            remote = REMOTE + '/prior_H14/' + name
            mkdir(str(PurePosixPath(remote).parent))
            with sftp.open(remote, 'wb') as file:
                file.write(data)
        with sftp.open(REMOTE + '/transfer_complete.json', 'w') as file:
            file.write(json.dumps(dict(status='ALL_PUBLIC_PAYLOADS_STREAM_SHA256_VERIFIED',
                assets=len(choices), bytes=total, seconds=time.monotonic() - started,
                local_large_files_written=False, server_native_rehash_still_required=True)) + '\n')
        print('PUBLIC_ASSET_TRANSFER_COMPLETE', total, flush=True)
        sftp.close()
    finally:
        client.close()


if __name__ == '__main__':
    main()
