"""Finite NEW-card sequence: close pinned resource v4, then true interface v2."""
import argparse,datetime,hashlib,json,os,signal,subprocess,time
from pathlib import Path
def ref(path):
    with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h)
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def run(path):
    spec=json.loads(path.read_bytes());out=Path(spec['queue_output']);assert not out.exists();out.mkdir();begin=time.monotonic();resource=Path(spec['resource_output']);deadline=begin+1800
    try:
        for item in spec['source_pins']:assert ref(Path(item['path']))==item
        worker=Path(spec['worker']['path']);assert ref(worker)==spec['worker'];profile=Path(spec['profile_spec']['path']);assert ref(profile)==spec['profile_spec']
        while True:
            assert time.monotonic()<deadline,'Finite resource predecessor wait expired'
            assert not (resource/'failure.json').exists(),'Predecessor resource failure preserved'
            if (resource/'completion.json').exists():break
            time.sleep(2)
        actual=json.loads((resource/'completion.json').read_bytes());assert actual['status']=='PASS_REAL_46_T_LOGICAL16_MICRO2_ALL3_ARMS_PRETRAINED_VAR_RESOURCE' and actual['resource_probe_version']==4
        assert actual['pretrained_VAR_loaded'] and actual['native_resume_serialization_probe']['retired_after_verification']
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            assert time.monotonic()<deadline;time.sleep(1)
        R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1');command=['/root/miniconda3/bin/python','-B',str(R/'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'),'--name','EEG_AR_NEW_full_pretrained_native_anchored_interface_20261029_v2','--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(worker),'--spec',str(profile)]
        with (out/'interface_stdout.log').open('xb') as stdout,(out/'interface_stderr.log').open('xb') as stderr:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,cwd=out)
            fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split();write(out/'interface_launch.json',dict(pid=child.pid,start_ticks=int(fields[19]),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,profile_spec=ref(profile)))
            try:code=child.wait(timeout=960)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM);raise
        assert code==0,code;completion=Path(json.loads(profile.read_bytes())['output'])/'completion.json';result=json.loads(completion.read_bytes());assert result['status']=='PASS_REAL_T_GENUINE_OOF_FULL_PRETRAINED_VQ_VAR_EEG_ENGINEERING'
        write(out/'completion.json',dict(status='NEW_PRETRAINED_RESOURCE_V4_AND_INTERFACE_V2_ACTUALLY_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),resource_completion=ref(resource/'completion.json'),interface_completion=ref(completion),seconds=time.monotonic()-begin,formal_fit=False))
    except BaseException as error:
        write(out/'failure.json',dict(status='FINITE_NEW_PRETRAINED_SEQUENCE_FAILURE_PRESERVED',error=repr(error),seconds=time.monotonic()-begin));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
