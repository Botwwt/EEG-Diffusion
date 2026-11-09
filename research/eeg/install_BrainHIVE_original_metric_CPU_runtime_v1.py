"""Install only two hashed CPU wheels in a dedicated server directory."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parent
DEST=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runtime/BrainHIVE_original_metrics_0p24_v1')


def main():
    assert not DEST.exists(),'Preserve an existing package installation'
    rows=[]
    for name,version,match in [('scikit-image','0.24.0','cp312-cp312-manylinux_2_17_x86_64'),
                               ('lazy-loader','0.4','py3-none-any.whl')]:
        with urllib.request.urlopen('https://pypi.org/pypi/'+name+'/'+version+'/json',timeout=30) as response:
            assert response.status==200;metadata=json.load(response)
        selected=[x for x in metadata['urls'] if match in x['filename']]
        assert len(selected)==1
        item=selected[0];digest=item['digests']['sha256']
        assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
        rows.append(dict(name=name,version=version,filename=item['filename'],bytes=item['size'],
                         url=item['url'],sha256=digest,identity_source='Authenticated official PyPI JSON; verified again by pip --require-hashes'))
    lock=ROOT/'audit/BrainHIVE_original_metric_CPU_requirements_v1.txt';assert not lock.exists()
    lock.write_text(''.join(x['url']+' --hash=sha256:'+x['sha256']+'\n' for x in rows))
    manifest=ROOT/'audit/BrainHIVE_original_metric_CPU_download_manifest_v1.json';assert not manifest.exists()
    manifest.write_text(json.dumps(rows,indent=2)+'\n')
    report=ROOT/'audit/BrainHIVE_original_metric_CPU_pip_report_v1.json'
    args=[sys.executable,'-m','pip','install','--no-deps','--no-cache-dir','--disable-pip-version-check',
          '--require-hashes','--target',str(DEST),'--report',str(report),'-r',str(lock)]
    result=subprocess.run(args,capture_output=True,text=True,timeout=180)
    (ROOT/'audit/BrainHIVE_original_metric_CPU_install_v1.log').write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode==0,result.stderr[-1200:]
    actual=json.loads(report.read_text())
    assert len(actual['install'])==2
    assert {x['download_info']['archive_info']['hashes']['sha256'] for x in actual['install']}=={x['sha256'] for x in rows}
    print('HASHED_CPU_METRIC_WHEELS_INSTALLED',[(x['name'],x['version'],x['bytes']) for x in rows],flush=True)


if __name__=='__main__':main()
