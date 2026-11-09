"""Finite OLD pretrained engineering sequence; neither formal fit nor retries."""
import datetime, hashlib, json, os, subprocess, time, traceback
from pathlib import Path
B=Path(__file__).resolve().parent.parent
R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(name,value):
    with (B/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
began=time.monotonic();events=[];phase='PRETRAINED_RESOURCE_V4'
try:
    for tag,worker,spec in [('resource_v4','EEG_memory_AR_resource_GPU_20261029_v4.py','resource_spec_v4.json'),('interface_v2','EEG_generative_backend_profile_GPU_20261029_v2.py','interface_spec_v2.json')]:
        phase=tag
        assert time.monotonic()-began<1900
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        command=['/root/miniconda3/bin/python','-B',str(R/'run_gpu_job_v2.py'),'--name','EEG_generative_pretrained_OLD_'+tag+'_20261029','--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(B/'code'/worker),'--spec',str(B/spec)]
        start=time.monotonic()
        with (B/(tag+'_stdout.log')).open('xb') as out,(B/(tag+'_stderr.log')).open('xb') as err:
            result=subprocess.run(command,cwd=B/'code',stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=930)
        record={'phase':tag,'command':command,'exit_code':result.returncode,'seconds':time.monotonic()-start};events.append(record);write(tag+'_job_result.json',record)
        assert result.returncode==0,result.returncode
        done=B/tag/'completion.json';value=json.loads(done.read_bytes())
        assert value['status'].startswith('PASS_REAL_')
    write('completion.json',{'status':'COMPLETE_ACTUAL_PRETRAINED_RESOURCE_V4_AND_SCHEMA_CORRECTED_INTERFACE_V2','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'events':events,'seconds':time.monotonic()-began,'formal_fit':False,'research_images':0})
except BaseException as error:
    write('failure.json',{'status':'ACTUAL_PRETRAINED_SEQUENCE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY','phase':phase,'error':repr(error),'traceback':traceback.format_exc(),'events':events,'seconds':time.monotonic()-began});raise
