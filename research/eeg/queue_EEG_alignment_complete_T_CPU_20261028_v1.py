"""Finite 27-segment queue; stops on failure, no automatic parameter changes."""
import argparse,datetime,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
from eeg_alignment_training_core_20261028_v1 import sha

def main(spec_path):
    began=time.monotonic();spec=json.loads(spec_path.read_bytes());root=Path(spec['output']);assert not root.exists();root.mkdir()
    events=[];R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    try:
        assert shutil.disk_usage(root).free>20<<30
        for pin in spec['source_pins']:assert sha(pin['path'])==pin['sha256'],pin['path']
        for role in ('full','oof0','oof1'):
            for segment in range(1,10):
                assert time.monotonic()-began<21000
                label=f'EEG_multiscale_alignment_{role}_s{segment:02d}_20261028_v1'
                command=['/root/miniconda3/bin/python','-B',str(R/'run_gpu_job_v2.py'),'--name',label,'--max-seconds','900','--','/root/miniconda3/bin/python','-B',spec['trainer'],'--spec',str(spec_path),'--role',role,'--segment',str(segment)]
                value=dict(status='FIXED_EEG_ENCODER_QUEUE_EXECUTING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),role=role,segment=segment,events=events,command=command)
                (root/'queue_progress.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
                with (root/(label+'.stdout.log')).open('xb') as out,(root/(label+'.stderr.log')).open('xb') as err:
                    phase=time.monotonic();result=subprocess.run(command,stdout=out,stderr=err,stdin=subprocess.DEVNULL,timeout=950)
                events.append(dict(role=role,segment=segment,exit_code=result.returncode,seconds=time.monotonic()-phase));assert result.returncode==0,(role,segment)
            completion=json.loads((root/role/'completion.json').read_bytes());assert completion['epochs']==50
        (root/'queue_completion.json').write_text(json.dumps(dict(status='ALL_THREE_SUB01_FULL_T_ROLES_FIXED_FINAL_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),events=events,seconds=time.monotonic()-began,formal_all10_complete=False,new_images=0),indent=2)+'\n',encoding='utf8')
    except BaseException as error:
        (root/'queue_failure.json').write_text(json.dumps(dict(status='EEG_ENCODER_QUEUE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-began),indent=2)+'\n',encoding='utf8');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);main(p.parse_args().spec)
