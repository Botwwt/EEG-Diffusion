"""Finish the registered original-paper metrics after the current fixed queue.

Only existing saved images are evaluated. No generation, training, second
architecture, or adaptive experiment selection is performed in this queue.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')
PYTHON='/root/miniconda3/bin/python'
STATE=WORK/'audit/original_metric_queue_v1.json'


def read(path):return json.loads(path.read_text())


def write(value):
    temp=STATE.with_name(STATE.name+'.pending')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(STATE)


def gpu(name,entry,arguments):
    command=[PYTHON,str(ROOT/entry)]+arguments
    argv=[PYTHON,str(ROOT/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','10800','--']+command
    receipt=ROOT/'audit'/(name+'_job.json')
    assert not receipt.exists(),'Preserve any prior attempt; do not retry implicitly'
    write(dict(status='RUNNING_ORIGINAL_AUTHOR_METRICS',job=name,argv=argv))
    with (WORK/'logs'/(name+'.log')).open('x') as log:
        code=subprocess.call(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert code==0,'Preserve failed job for explicit diagnosis: '+name
    record=read(receipt);assert record['status']=='COMPLETED' and record['exit_code']==0


def main():
    assert not STATE.exists(),'Never overwrite an earlier queue'
    assert read(WORK/'audit/original_metric_assets_CPU_v3.json')['status']=='ALL5_AUTHOR_METRIC_ASSETS_STRICT_LOAD_VERIFIED_CPU'
    assert read(WORK/'audit/original_metric_math_CPU_v1.json')['status']=='PINNED_AUTHOR_METRIC_ANALYTIC_CHECKS_PASSED_CPU'
    entries=['brainhive_original_metrics_runtime_v1.py','check_BrainHIVE_original_real_images_v1.py',
        'score_BrainHIVE_original_images_v1.py','queue_BrainHIVE_original_metrics_v1.py']
    specification=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in entries},
        existing_common_metrics_unchanged=True,existing_images_only=True,
        precondition='ALL3_NATIVE_R80_IMAGE_ARMS_AND_LOCKED_PER_QUERY_METRICS_COMPLETE',
        prerequisite_wait_seconds=7200,maximum_seconds_per_GPU_job=10800,
        arms=['native_R80_correct_v1','native_R80_permuted_v1','native_R80_missing_v1'],
        images_per_arm=2000,actual_real_gallery_checks_first=True,
        no_generated_scores_used_for_selection=True,no_component_training_in_this_queue=True)
    registration=WORK/'audit/original_metric_queue_registration_v1.json'
    assert not registration.exists();registration.write_text(json.dumps(specification,indent=2)+'\n')
    write(dict(status='WAITING_FOR_EXISTING_COMMON_METRICS_QUEUE',registration=str(registration)))
    started=time.monotonic()
    while True:
        previous=read(WORK/'audit/native_benchmark_queue_v2.json')
        if previous['status']==specification['precondition']:break
        assert previous['status']=='RUNNING_FIXED_NATIVE_BENCHMARK',previous
        assert time.monotonic()-started<7200,'Finite prerequisite wait expired; no implicit restart'
        time.sleep(15)
    for name,expected in specification['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    gpu('BrainHIVE_original_metric_real_checks_v1','check_BrainHIVE_original_real_images_v1.py',[])
    assert read(WORK/'audit/original_real_image_checks_v1/completion.json')['status']=='ALL200_REAL_IMAGE_IDENTITY_AND_FIXED_DERANGED_CHECKS_PASSED'
    results={}
    for run in specification['arms']:
        gpu('BrainHIVE_'+run+'_original_metrics_v1','score_BrainHIVE_original_images_v1.py',['--run-name',run])
        completion=WORK/'evaluation'/run/'original_paper_metrics_v1/completion.json'
        result=read(completion)
        assert result['status']=='ALL2000_AUTHOR_CODE_QUERY_METRICS_COMPLETE',result['status']
        results[run]=str(completion)
    write(dict(status='ALL3_NATIVE_R80_AUTHOR_EIGHT_METRICS_AND_QUERY_RECORDS_COMPLETE',
        queries_per_arm=2000,results=results,component_training_complete=False,
        component_gain_established=False,independent_H=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(dict(status='FAILED_PRESERVED_NEEDS_EXPLICIT_DIAGNOSIS',
            error_type=type(error).__name__,error=str(error)[:1200]))
        raise
