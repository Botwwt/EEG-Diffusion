"""Run the source-aligned vision-dropout correction after verified CPU inputs."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')


def main():
    assert json.loads((WORK/'audit/input_preparation_complete.json').read_text())['status']=='ALL10_EEG_AND_PUBLIC_H14_FEATURES_VERIFIED'
    code=0
    for role in ['published','full','oof0','oof1']:
        command=[sys.executable,str(ROOT/'run_gpu_job_v2.py'),'--name','BrainHIVE_source_dropout_'+role+'_alignment_v2',
                 '--max-seconds','10800','--',sys.executable,str(ROOT/'train_BrainHIVE_parent_v2.py'),'--role',role]
        code=subprocess.call(command,cwd=ROOT)
        if code!=0:break
    (WORK/'audit/source_dropout_alignment_queue_completion.json').write_text(json.dumps(dict(
        exit_code=code,status='SOURCE_ALIGNED_PARENT_TRAINED_IMAGES_NEXT' if code==0 else 'FAILED_PRESERVE_CHECKPOINTS',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        previous_fixed_target_engineering_run_preserved=True,images_generated=0))+'\n')
    return code


if __name__=='__main__':raise SystemExit(main())
