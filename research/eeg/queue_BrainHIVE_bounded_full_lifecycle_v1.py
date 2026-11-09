"""Finite complete-data main comparison; failed independent arms stay recorded."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')
PYTHON='/root/miniconda3/bin/python'
STATE=WORK/'audit/bounded_full_lifecycle_queue_v1.json'


def read(p):return json.loads(Path(p).read_text())


def write(value):
    t=STATE.with_suffix('.json.pending');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(STATE)


def gpu(name,entry,arguments,seconds):
    assert not (ROOT/'audit'/(name+'_job.json')).exists(),'Preserve prior attempts; never silently rerun'
    argv=[PYTHON,str(ROOT/'run_gpu_job_v2.py'),'--name',name,'--max-seconds',str(seconds),'--',PYTHON,'-u',str(ROOT/entry)]+arguments
    write(dict(status='RUNNING_FIXED_COMPLETE_BRAINHIVE_COMPONENT_LIFECYCLE',job=name,argv=argv))
    with (WORK/'logs'/(name+'.log')).open('x') as log:
        code=subprocess.call(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    return code


def cpu(entry,arguments,seconds=3600):
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    subprocess.run([PYTHON,str(ROOT/entry)]+arguments,cwd=ROOT,timeout=seconds,check=True,env=env)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',required=True);a=parser.parse_args()
    recipe=Path(a.recipe);config=read(recipe);assert not STATE.exists();failures={};trained=[]
    write(dict(status='WAITING_FOR_COMPLETE_ALLOWED_T_LATENT_BANK',maximum_wait_seconds=3600))
    started=time.monotonic()
    while True:
        job=read(ROOT/'audit/BrainHIVE_component_all_T_latents_v2_job.json')
        if job['status']=='COMPLETED' and Path(config['latent_receipt']).exists():break
        assert job['status']=='RUNNING',job
        assert time.monotonic()-started<3600
        time.sleep(15)
    # Main idea first after the completed native R80 reproduction. Complete and
    # save each version before continuing, with all recipes frozen in advance.
    completed=[];gallery=WORK/'gallery/V/manifest.jsonl';gallery_features=WORK/'evaluation/bounded_V_gallery_features_v1'
    assert config['fixed_arm_order']==['risk','ordinary','parent']
    for arm in config['fixed_arm_order']:
        if arm!='parent':
            name='BrainHIVE_bounded_full_T_'+arm+'_v1'
            code=gpu(name,'train_BrainHIVE_bounded_component_v1.py',['--arm',arm,'--recipe',str(recipe)],43200)
            progress=WORK/'components/bounded_v1'/arm/'progress.json'
            if code!=0 and progress.exists() and read(progress).get('status')=='RESOURCE_STOPPED_RESUMABLE':
                used=read(ROOT/'audit'/(name+'_job.json'))['charged_seconds'];remaining=min(7200,50400-used)
                assert remaining>120
                code=gpu(name+'_exact_resume1','train_BrainHIVE_bounded_component_v1.py',
                    ['--arm',arm,'--recipe',str(recipe),'--resume'],remaining)
            if code==0:trained.append(arm)
            else:
                failures[arm]='Training failure; original receipt/log/checkpoint preserved, no fabricated negative result'
                continue
        run=config['run_names'][arm]
        code=gpu('BrainHIVE_'+run+'_images_v1','generate_BrainHIVE_bounded_V_v1.py',['--arm',arm,'--recipe',str(recipe)],14400)
        if code!=0:failures[arm]='Full image job failed or reached its finite bound; saved images retained';continue
        try:cpu('audit_BrainHIVE_image_arm_CPU_v1.py',['--run-name',run])
        except Exception as error:
            failures[arm]='Image identity audit failed: '+str(error)[:500]
            continue
        if not (gallery_features/'completion.json').exists():
            code=gpu('BrainHIVE_bounded_V_gallery_features_v1','extract_image_features.py',
                ['--manifest',str(gallery),'--output',str(gallery_features),'--real-gallery'],14400)
            if code!=0:failures['gallery_evaluator']='Preserved evaluator failure';break
        features=WORK/'evaluation'/run/'features';manifest=WORK/'images'/run/'manifest.jsonl'
        code=gpu('BrainHIVE_'+run+'_features_v1','extract_image_features.py',['--manifest',str(manifest),'--output',str(features)],14400)
        if code!=0:failures[arm]='Preserved full image-feature evaluation failure';continue
        try:
            cpu('score_full_manifest.py',['--manifest',str(manifest),'--features',str(features),'--gallery-manifest',str(gallery),
                '--gallery-features',str(gallery_features),'--output',str(WORK/'evaluation'/run/'scores')])
            cpu('score_low_level.py',['--manifest',str(manifest),'--gallery-manifest',str(gallery),'--output',str(WORK/'evaluation'/run/'low_level')])
            completed.append(arm)
        except Exception as error:failures[arm]=type(error).__name__+': '+str(error)[:500]
    if set(completed)=={'parent','ordinary','risk'}:
        cpu('compare_BrainHIVE_bounded_V_CPU_v1.py',['--recipe',str(recipe)],seconds=7200)
        write(dict(status='ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED',completed=completed,
            failures=failures,main_candidate_configurations=1,automatic_hyperparameter_expansion=False,
            mechanism_confirmed=False,independent_SOTA_established=False))
    else:write(dict(status='FINITE_MAIN_COMPARISON_PARTIAL_WITH_EXPLICIT_FAILURES',completed=completed,failures=failures,goal_complete=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(dict(status='FAILED_PRESERVED_NEEDS_EXPLICIT_DIAGNOSIS',error_type=type(error).__name__,error=str(error)[:1600]))
        raise
