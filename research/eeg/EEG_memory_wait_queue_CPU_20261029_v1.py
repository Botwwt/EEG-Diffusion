"""Finite actual follow-on queue; never preempts or changes the D40 lifecycle."""
import argparse,datetime,json,subprocess,time,traceback
from pathlib import Path

def main(root):
    started=time.monotonic();R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1');P=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1');D=P/'rapid_iteration_20261027_v1/candidates/D40_spatial_innovation'
    try:
        while True:
            assert time.monotonic()-started<3600
            assert not (D/'lifecycle_v1/failure.json').exists() and not (D/'factorial_images_v1/queue/failure.json').exists(),'D40 failed: preserve; no new GPU task'
            ready=(root/'transfer_completion.json').is_file();done=(D/'lifecycle_v1/completion.json').is_file();idle=not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if ready and done and idle:
                closure=json.loads((D/'lifecycle_v1/completion.json').read_bytes());assert closure['status']=='D40_THREE_FITS_EIGHT_IMAGE_ARMS_SCORES_AND_CPU_AUDITS_CLOSED'
                queue=json.loads((D/'factorial_images_v1/queue/completion.json').read_bytes());assert queue['status']=='D40_THREE_FITS_AND_ALL_EIGHT_FIXED_IMAGE_ARMS_ACTUALLY_CLOSED'
                break
            (root/'queue_progress.json').write_text(json.dumps(dict(status='FINITE_EEG_MEMORY_FOLLOWON_WAITING_D40_AND_FIXED_FINALS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),transfer_complete=ready,D40_complete=done,GPU_idle=idle,new_GPU_jobs=0),indent=2)+'\n',encoding='utf8');time.sleep(10)
        command=['/root/miniconda3/bin/python','-B',str(R/'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'),'--name','EEG_memory_real_OOF_export_20261029_v1','--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(root/'code/EEG_memory_export_GPU_20261029_v1.py'),'--spec',str(root/'export_spec_v1.json')]
        with (root/'GPU_stdout_v1.log').open('xb') as out,(root/'GPU_stderr_v1.log').open('xb') as err:
            result=subprocess.run(command,stdout=out,stderr=err,stdin=subprocess.DEVNULL,timeout=950)
        assert result.returncode==0,result.returncode
        proof=json.loads((root/'export_v1/completion.json').read_bytes());assert proof['status']=='COMPLETE_REAL_EEG_T13230_GENUINE_OOF_AND_DCHECK400_LOCAL_MEMORY'
        (root/'queue_completion.json').write_text(json.dumps(dict(status='EEG_MEMORY_FOLLOWON_ACTUALLY_COMPLETED_AFTER_D40',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,seconds=time.monotonic()-started,export_completion=str(root/'export_v1/completion.json'),new_images=0,new_fits=0),indent=2)+'\n',encoding='utf8')
    except BaseException as error:
        (root/'queue_failure.json').write_text(json.dumps(dict(status='EEG_MEMORY_FOLLOWON_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started),indent=2)+'\n',encoding='utf8');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
