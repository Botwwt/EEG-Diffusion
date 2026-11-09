"""Resume only missing verified ranges: current local proxy -> RAM -> SFTP.

No large local files. This avoids the unreliable reverse CONNECT tunnel while
retaining the server's exact range map and requiring full native SHA at the end.
"""
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, r'C:\Users\Asus\AppData\Roaming\Python\Python311\site-packages')
import paramiko

ASSETS='/root/brainhive_reproduction_20261020_v1/assets'
CHUNK=8 << 20


def main():
    started=time.monotonic()
    password=getpass.getpass('SSH password: ')
    client=paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('CLOUD_LOGIN_HOST', port=54067, username='root',
                   password=password, timeout=25, banner_timeout=25)
    password=None
    client.get_transport().set_keepalive(15)
    sftp=client.open_sftp()
    try:
        with sftp.open(ASSETS+'/transfer_plan.json') as f:plan=json.load(f)
        asset=next(a for a in plan['assets'] if a['path'].startswith('prior_H14/attn_adapter/'))
        path=ASSETS+'/'+asset['path']; state_path=path+'.range_state.json'
        with sftp.open(state_path) as f:state=json.load(f)
        assert state['expected_sha256']==asset['sha256']
        assert sftp.stat(path+'.partial').st_size==asset['bytes']
        todo=[lo for lo in range(state['initial_sequential_prefix_bytes'],asset['bytes'],CHUNK)
              if str(lo) not in state['completed_ranges']]

        def fetch(lo):
            hi=min(lo+CHUNK,asset['bytes'])-1
            url=('https://huggingface.co/'+asset['repo']+'/resolve/'+asset['revision']
                 +'/'+asset['source_path']+'?download=true&ram_resume_range='+str(lo))
            errors=[]
            for attempt in range(3):
                try:
                    assert time.monotonic()-started < 7200
                    req=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}','Accept-Encoding':'identity'})
                    with urllib.request.urlopen(req,timeout=45) as response:
                        assert response.status==206, 'Expected 206, got '+str(response.status)
                        assert response.headers['Content-Range']==f"bytes {lo}-{hi}/{asset['bytes']}",response.headers.get('Content-Range')
                        data=response.read(CHUNK+1)
                    assert len(data)==hi-lo+1
                    return lo,data
                except Exception as e:errors.append(type(e).__name__+':'+str(e)[:120])
            raise RuntimeError('Public range retries exhausted '+str(lo)+' '+repr(errors))

        # At most four fetched chunks (32 MiB), plus one being written.
        with ThreadPoolExecutor(4) as pool:
            pending={}
            iterator=iter(todo)
            for _ in range(4):
                lo=next(iterator,None)
                if lo is not None:pending[lo]=pool.submit(fetch,lo)
            for lo in todo:
                lo,data=pending.pop(lo).result()
                with sftp.open(path+'.partial','r+b',bufsize=CHUNK) as destination:
                    destination.set_pipelined(True)
                    destination.seek(lo)
                    destination.write(data)
                state['completed_ranges'][str(lo)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                with sftp.open(state_path+'.pending','w') as f:f.write(json.dumps(state,indent=2)+'\n')
                sftp.posix_rename(state_path+'.pending',state_path)
                complete=state['initial_sequential_prefix_bytes']+sum(x['bytes'] for x in state['completed_ranges'].values())
                progress=dict(stage='PUBLIC_PROXY_RAM_SFTP_RESUME',completed_bytes=complete,total_bytes=asset['bytes'],seconds=time.monotonic()-started,local_large_files_written=False)
                with sftp.open(ASSETS+'/server_transfer_progress.json.pending','w') as f:f.write(json.dumps(progress,indent=2)+'\n')
                sftp.posix_rename(ASSETS+'/server_transfer_progress.json.pending',ASSETS+'/server_transfer_progress.json')
                print(json.dumps(progress),flush=True)
                nxt=next(iterator,None)
                if nxt is not None:pending[nxt]=pool.submit(fetch,nxt)
        # Reuse native verifier; no relay is needed when all ranges are present.
        command='/root/miniconda3/bin/python -u /root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1/finish_BrainHIVE_public_assets_server_v2.py'
        _,stdout,stderr=client.exec_command(command,timeout=60)
        text=stdout.read().decode();errors=stderr.read().decode()
        assert stdout.channel.recv_exit_status()==0,(text,errors)
        print(text,flush=True)
    finally:
        sftp.close()
        client.close()


if __name__=='__main__':main()
