"""Copy pinned public author assets through bounded RAM directly to the server."""
import argparse
import datetime
import getpass
import hashlib
import json
from pathlib import Path
import posixpath
import shlex
import sys
import urllib.request

sys.path.insert(0, r'C:\Users\Asus\AppData\Roaming\Python\Python311\site-packages')
import paramiko

ROOT = '/root/atm_author_pretrained_20261020_v2'
REVISION = 'dbe34bb2407164f70883c661c76664f0d596e522'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    raw = args.metadata.read_bytes()
    metadata = json.loads(raw)['data']
    assert metadata['sha'] == REVISION
    allowed = {f'fintune_ckpts/sub-{i:02d}/diffusion_prior.pt' for i in range(1, 11)}
    allowed |= {f'emb_eeg/ATM_S_eeg_features_sub-{i:02d}_test.pt' for i in range(1, 11)}
    rows = [x for x in metadata['siblings'] if x['rfilename'] in allowed]
    assert len(rows) == 20 and {x['rfilename'] for x in rows} == allowed
    assert sum(x['size'] for x in rows) < 400_000_000
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('CLOUD_LOGIN_HOST', port=54067, username='root',
                   password=getpass.getpass('SSH password: '), timeout=25)
    _, so, se = client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote('import shutil; print(shutil.disk_usage("/root").free)'), timeout=30)
    free_bytes = int(so.read().decode().strip())
    assert so.channel.recv_exit_status() == 0, se.read().decode()
    assert free_bytes - sum(x['size'] for x in rows) > 2*1024**3
    with client.open_sftp() as sftp:
        sftp.mkdir(ROOT, mode=0o700)  # A fresh immutable destination is required.
        assert sftp.normalize(ROOT) == ROOT
        registration = dict(status='REGISTERED_BEFORE_TRANSFER_NO_GPU_OR_GENERATION',
                            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            revision=REVISION, metadata_sha256=hashlib.sha256(raw).hexdigest(),
                            metadata_source=json.loads(raw)['url'],
                            files=rows, expected_total_bytes=sum(x['size'] for x in rows),
                            local_disk_asset_bytes=0,
                            scope='Author prior and published EEG test features; not encoder weights or a full reproduction')
        with sftp.file(ROOT+'/registration.json', 'wx') as f:
            f.write(json.dumps(registration, indent=2)+'\n')
        completed = []
        for item in rows:
            relative = item['rfilename']
            assert relative in allowed and '..' not in relative.split('/')
            parent = ROOT
            for part in relative.split('/')[:-1]:
                parent = posixpath.join(parent, part)
                try:
                    sftp.mkdir(parent, mode=0o700)
                except OSError:
                    assert sftp.normalize(parent) == parent
            target = ROOT+'/'+relative
            partial = target+'.partial'
            url = 'https://huggingface.co/datasets/LidongYang/EEG_Image_decode/resolve/'+REVISION+'/'+relative
            digest, count = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=45) as response, sftp.file(partial, 'wx') as output:
                output.set_pipelined(True)
                while True:
                    block = response.read(512*1024)
                    if not block:
                        break
                    count += len(block)
                    assert count <= item['size']
                    digest.update(block)
                    output.write(block)
            assert count == item['size'] and digest.hexdigest() == item['lfs']['sha256']
            _, so, se = client.exec_command('sha256sum -- '+shlex.quote(partial), timeout=60)
            server_hash = so.read().decode().split()[0]
            assert so.channel.recv_exit_status() == 0, se.read().decode()
            assert server_hash == digest.hexdigest()
            sftp.rename(partial, target)
            completed.append(dict(path=target, bytes=count, sha256=server_hash))
            print(json.dumps(dict(status='VERIFIED_SERVER_ASSET', file=relative, bytes=count,
                                  completed=len(completed), expected=len(rows))), flush=True)
        receipt = dict(status='COMPLETED_VERIFIED_PUBLIC_AUTHOR_ASSET_TRANSFER',
                       UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       files=completed, local_disk_asset_bytes=0, images_generated=0,
                       registration_sha256=hashlib.sha256((json.dumps(registration, indent=2)+'\n').encode()).hexdigest())
        with sftp.file(ROOT+'/completion.json', 'wx') as f:
            f.write(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(dict(status=receipt['status'], files=len(completed))), flush=True)
    client.close()


if __name__ == '__main__':
    main()
