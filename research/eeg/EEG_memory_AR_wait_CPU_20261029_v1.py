"""Finite NEW-card generative engineering gate; no automatic formal fit."""
import argparse,datetime,hashlib,json,subprocess,time,traceback
from pathlib import Path
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def checked(ref):
    path=Path(ref['path']);assert sha(path)==ref['sha256'],str(path)
    if 'bytes' in ref:assert path.stat().st_size==ref['bytes']
    return path
def run(root):
    started=time.monotonic();R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1');phase='WAIT_VERIFIED_PUBLIC_ASSETS_OLD_REAL_T_BANK_AND_ENGINEERING';marker=root/'engineering_ready_marker_v1.json'
    try:
        policy=json.loads((root/'waiting_policy_v1.json').read_bytes())
        for ref in policy['registered_sources']:checked(ref)
        while not marker.exists():
            assert time.monotonic()-started<10800
            (root/'queue_progress.json').write_text(json.dumps(dict(status='NEW_AR_ENGINEERING_ACTUALLY_QUEUED_AWAITING_VERIFIED_BACKEND_INPUTS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,new_GPU_jobs=0,new_formal_fits=0),indent=2)+'\n',encoding='utf8');time.sleep(20)
        ready=json.loads(marker.read_bytes());assert ready['status']=='NEW_AR_ENGINEERING_INPUTS_OLD_BANK_AND_PROFILE_ACTUALLY_VERIFIED'
        old=json.loads(checked(ready['old_profile_completion']).read_bytes());assert old['status']=='PASS_REAL_T_GENUINE_OOF_FULL_PRETRAINED_VQ_VAR_EEG_ENGINEERING' and old['temporary_updates']==3
        bank=json.loads(checked(ready['T_token_bank_completion']).read_bytes());assert ready['T_token_bank_verified_complete'] is True
        spec=checked(ready['profile_spec']);body=json.loads(spec.read_bytes())
        for ref in body['source_pins']:checked(ref)
        worker=checked(ready['profile_worker']);assert worker.is_relative_to(root) and spec.is_relative_to(root)
        assert sha(worker)==policy['profile_worker_sha256'] and body['GPU_UUID']=='GPU-62a1aee8-4740-dc59-93f4-d50644db650d'
        phase='ACTUAL_NEW_REAL_T_FULL_PRETRAINED_BACKEND_ENGINEERING'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        command=['/root/miniconda3/bin/python','-B',str(R/'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'),'--name','EEG_AR_NEW_native_anchored_profile_20261029_v1','--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(worker),'--spec',str(spec)]
        with (root/'profile_stdout_v1.log').open('xb') as out,(root/'profile_stderr_v1.log').open('xb') as err:
            result=subprocess.run(command,stdout=out,stderr=err,stdin=subprocess.DEVNULL,timeout=950)
        assert result.returncode==0,result.returncode
        done=Path(body['output'])/'completion.json';proof=json.loads(done.read_bytes());assert proof['status']=='PASS_REAL_T_GENUINE_OOF_FULL_PRETRAINED_VQ_VAR_EEG_ENGINEERING'
        value=dict(status='NEW_REAL_EEG_PRETRAINED_GENERATIVE_ENGINEERING_ACTUALLY_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),profile_completion=dict(path=str(done),bytes=done.stat().st_size,sha256=sha(done)),ready_marker=dict(path=str(marker),sha256=sha(marker)),seconds=time.monotonic()-started,new_formal_fits=0,new_research_images=0)
        (root/'queue_completion.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    except BaseException as error:
        (root/'queue_failure.json').write_text(json.dumps(dict(status='NEW_GENERATIVE_ENGINEERING_QUEUE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started),indent=2)+'\n',encoding='utf8');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
